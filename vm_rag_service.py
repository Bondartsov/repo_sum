"""
FastAPI сервис для RAG-as-a-Service на VM с Jina v3.

Основные эндпоинты:
- POST /embeddings - получение эмбеддингов от Jina v3
- POST /search - гибридный поиск по векторам
- POST /index - индексация документов
- GET /health - проверка состояния
"""

import os
import sys
import asyncio
import logging
import gc
import psutil
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import uvicorn

# Добавляем текущую директорию в Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем оригинальные (локальные) версии для VM
try:
    from rag.embedder import CPUEmbedder
    from rag.vector_store import QdrantVectorStore
    from rag.search_service import SearchService
    from rag.indexer_service import IndexerService
    from config import get_config
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все зависимости установлены на VM")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic модели для API
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="Список текстов для эмбеддинга")
    task: Optional[str] = Field("retrieval.passage", description="Задача для dual task архитектуры")
    truncate_dim: Optional[int] = Field(None, description="Желаемая размерность эмбеддингов (по умолчанию 1024)")
    normalize: bool = Field(True, description="Применять L2 нормализацию")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    top_k: int = Field(10, description="Количество результатов")
    use_hybrid: bool = Field(True, description="Использовать гибридный поиск")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Фильтры по метаданным")
    task: str = Field("retrieval.query", description="Задача для query эмбеддинга")

class IndexRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="Документы для индексации")
    batch_size: int = Field(512, description="Размер батча для обработки")
    recreate: bool = Field(False, description="Пересоздать коллекцию")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    embedding_dim: int
    processing_time: float

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query_time: float
    total_found: int
    hybrid_used: bool

class IndexResponse(BaseModel):
    indexed_count: int
    status: str
    processing_time: float
    collection_info: Dict[str, Any]

# Глобальные сервисы
services = {}

def check_memory_usage() -> Dict[str, Any]:
    """
    Проверка использования памяти на VM.

    Адаптивные пороги для 60GB RAM:
    - Critical: >92% (осталось <5GB) - реальная опасность OOM
    - Warning: >85% (осталось <9GB) - начинаем мониторить

    Returns:
        Словарь с информацией о памяти
    """
    try:
        memory = psutil.virtual_memory()
        available_gb = round(memory.available / (1024**3), 2)

        # Адаптивная логика: критично когда осталось <5GB
        # Для 60GB: 92%, для 32GB: 84%
        total_gb = round(memory.total / (1024**3), 2)
        critical_threshold = 100 - (5.0 / total_gb * 100)  # Динамический порог
        warning_threshold = 100 - (9.0 / total_gb * 100)

        return {
            "total_gb": total_gb,
            "available_gb": available_gb,
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent,
            "is_critical": memory.percent > critical_threshold,  # ~92% для 60GB
            "is_warning": memory.percent > warning_threshold     # ~85% для 60GB
        }
    except Exception as e:
        logger.warning(f"Не удалось получить информацию о памяти: {e}")
        return {"error": str(e)}

def force_garbage_collection() -> Dict[str, Any]:
    """
    Принудительная очистка памяти.
    
    Returns:
        Информация об очистке
    """
    try:
        # Получаем состояние памяти до очистки
        before = psutil.virtual_memory().percent
        
        # Выполняем сборку мусора
        collected = gc.collect()
        
        # Получаем состояние после очистки
        after = psutil.virtual_memory().percent
        
        return {
            "objects_collected": collected,
            "memory_before_percent": before,
            "memory_after_percent": after,
            "memory_freed_percent": round(before - after, 2)
        }
    except Exception as e:
        logger.error(f"Ошибка при сборке мусора: {e}")
        return {"error": str(e)}

def memory_check_middleware():
    """
    Middleware для проверки памяти перед тяжелыми операциями.
    """
    memory_info = check_memory_usage()
    
    if memory_info.get("is_critical", False):
        logger.warning(f"⚠️ Критический уровень памяти: {memory_info['percent_used']:.1f}%")
        gc_result = force_garbage_collection()
        logger.info(f"Сборка мусора: освобождено {gc_result.get('memory_freed_percent', 0):.1f}%")
        
        # Если после сборки мусора все еще критично - возвращаем ошибку
        updated_memory = check_memory_usage()
        if updated_memory.get("is_critical", False):
            raise HTTPException(
                status_code=507,
                detail=f"Недостаточно памяти на VM: {updated_memory['percent_used']:.1f}% использовано"
            )
    
    elif memory_info.get("is_warning", False):
        logger.warning(f"⚠️ Высокий уровень памяти: {memory_info['percent_used']:.1f}%")
    
    return memory_info

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и очистка сервисов"""
    logger.info("🚀 Инициализация RAG-as-a-Service на VM...")
    
    try:
        # Получаем конфигурацию
        config = get_config()
        
        # Инициализируем эмбеддер с Jina v3
        logger.info("Инициализация Jina v3 эмбеддера...")
        services['embedder'] = CPUEmbedder(
            embedding_config=config.rag.embeddings,
            parallelism_config=config.rag.parallelism
        )
        
        # Прогрев модели
        if config.rag.embeddings.warmup_enabled:
            services['embedder'].warmup()
        
        # Инициализируем векторное хранилище
        logger.info("Инициализация Qdrant...")
        services['vector_store'] = QdrantVectorStore(config.rag.vector_store)
        await services['vector_store'].initialize_collection()
        
        # Инициализируем сервисы поиска и индексации
        services['search_service'] = SearchService(
            config=config,
            silent_mode=True  # Отключаем консольный вывод для VM сервиса
        )
        
        services['indexer_service'] = IndexerService(
            config=config
        )
        
        logger.info("✅ Все сервисы успешно инициализированы")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации сервисов: {e}")
        raise
    
    finally:
        # Очистка ресурсов
        logger.info("Очистка ресурсов...")
        if 'vector_store' in services:
            await services['vector_store'].close()
        if 'indexer_service' in services:
            await services['indexer_service'].close()

# Создаем FastAPI приложение
app = FastAPI(
    title="RAG-as-a-Service VM",
    description="Сервис эмбеддингов и поиска на базе Jina v3 и Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "RAG-as-a-Service VM",
        "version": "0.5",
        "model": "jinaai/jina-embeddings-v3",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    try:
        # Единый стандарт: "connected" для всех успешных подключений
        health_info = {
            "status": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {}
        }
        
        # Проверка эмбеддера
        if 'embedder' in services:
            embedder_stats = services['embedder'].get_stats()
            # Единый стандарт: "connected" если готов
            health_info['services']['embedder'] = {
                "status": "connected" if embedder_stats.get('is_warmed_up') else "warming_up",
                "model": embedder_stats.get('model_name'),
                "provider": embedder_stats.get('provider')
            }
        
        # Проверка векторного хранилища
        if 'vector_store' in services:
            vs_health = await services['vector_store'].health_check()
            health_info['services']['vector_store'] = vs_health
            health_info['collection_status'] = 'exists' if vs_health.get('status') == 'connected' else 'unknown'
            health_info['qdrant_status'] = vs_health.get('status', 'unknown')
            health_info['vector_count'] = vs_health.get('collection_info', {}).get('vectors_count', 0)
        
        return health_info
        
    except Exception as e:
        logger.error(f"Ошибка health check: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """Получение эмбеддингов от Jina v3"""
    if 'embedder' not in services:
        raise HTTPException(status_code=503, detail="Embedder не инициализирован")
    
    try:
        # Проверяем память перед тяжелой операцией
        memory_check_middleware()
        
        start_time = asyncio.get_event_loop().time()
        
        # Получаем эмбеддинги с поддержкой dual task
        embeddings = services['embedder'].embed_texts(
            texts=request.texts,
            task=request.task,
            deadline_ms=30000
        )
        
        
        expected_dim = embeddings.shape[1]
        if request.truncate_dim and request.truncate_dim != expected_dim:
            logger.warning(f"Запрошено усечение до {request.truncate_dim}d, но сервис возвращает {expected_dim}d. Сжатие отключено.")
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model_name="jinaai/jina-embeddings-v3",
            embedding_dim=embeddings.shape[1],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Ошибка получения эмбеддингов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка эмбеддинга: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Гибридный поиск по документам"""
    if 'search_service' not in services:
        raise HTTPException(status_code=503, detail="Search service не инициализирован")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Выполняем поиск через SearchService
        results = await services['search_service'].search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            use_hybrid=request.use_hybrid,
            task=request.task
        )
        
        query_time = asyncio.get_event_loop().time() - start_time
        
        # Конвертируем SearchResult объекты в словари для Pydantic
        results_dicts = [asdict(r) for r in results]
        
        return SearchResponse(
            results=results_dicts,
            query_time=query_time,
            total_found=len(results),
            hybrid_used=request.use_hybrid
        )
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """Индексация документов с защитой от OOM"""
    if 'indexer_service' not in services:
        raise HTTPException(status_code=503, detail="Indexer service не инициализирован")
    
    try:
        # КРИТИЧНО: Проверяем память перед индексацией
        memory_info = memory_check_middleware()
        logger.info(f"🧠 Память перед индексацией: {memory_info.get('percent_used', 0):.1f}%")
        
        # Автоматически уменьшаем batch_size при высоком потреблении памяти
        original_batch_size = request.batch_size
        if memory_info.get('is_warning', False):
            request.batch_size = min(1, original_batch_size // 4)
            logger.warning(f"⚠️ Уменьшен batch_size: {original_batch_size} -> {request.batch_size}")
        
        start_time = asyncio.get_event_loop().time()
        
        # 🔍 ДИАГНОСТИКА 1: Что получил VM endpoint
        logger.info(f"📥 VM: Получено {len(request.documents)} документов")
        if request.documents:
            first_doc_raw = request.documents[0]
            logger.info(f"📥 VM: Первый document RAW = {first_doc_raw}")
            logger.info(f"📥 VM: Тип документа = {type(first_doc_raw)}")
            if isinstance(first_doc_raw, dict):
                logger.info(f"📥 VM: Ключи документа = {list(first_doc_raw.keys())}")
                logger.info(f"📥 VM: doc.get('text') = '{first_doc_raw.get('text', 'KEY_NOT_FOUND')[:100]}'")
            
        # Подготавливаем документы для индексации
        points = []
        for doc in request.documents:
            # ✅ ИСПРАВЛЕНИЕ: Извлекаем текст из правильного места
            # Сначала пробуем doc['text'], если нет - берём doc['payload']['content']
            text = doc.get('text', '') or doc.get('payload', {}).get('content', '')
            
            point = {
                'id': doc.get('id'),
                'text': text,
                'metadata': doc.get('metadata', {}),
                'timestamp': doc.get('timestamp', datetime.now(timezone.utc).isoformat())
            }
            points.append(point)
        
        # 🔍 ДИАГНОСТИКА 2: Что передаём в IndexerService
        if points:
            first_point = points[0]
            logger.info(f"📤 VM: Первый point после обработки = {first_point}")
            logger.info(f"📤 VM: point['text'] = '{first_point.get('text', 'EMPTY')[:100]}'")
        
        # Выполняем индексацию
        indexed_count = await services['indexer_service'].index_documents(
            documents=points,
            batch_size=request.batch_size,
            recreate_collection=request.recreate
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Получаем информацию о коллекции
        collection_info = {}
        if 'vector_store' in services:
            vs_health = await services['vector_store'].health_check()
            collection_info = vs_health.get('collection_info', {})
        
        return IndexResponse(
            indexed_count=indexed_count,
            status="success",
            processing_time=processing_time,
            collection_info=collection_info
        )
        
    except Exception as e:
        logger.error(f"Ошибка индексации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка индексации: {str(e)}")

@app.post("/collection/recreate")
async def recreate_collection():
    """Пересоздание коллекции"""
    if 'vector_store' not in services:
        raise HTTPException(status_code=503, detail="Vector store не инициализирован")
    
    try:
        await services['vector_store'].initialize_collection(recreate=True)
        return {"status": "success", "message": "Коллекция пересоздана"}
    except Exception as e:
        logger.error(f"Ошибка пересоздания коллекции: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка пересоздания: {str(e)}")

@app.get("/collection/info")
async def get_collection_info():
    """Информация о коллекции"""
    if 'vector_store' not in services:
        raise HTTPException(status_code=503, detail="Vector store не инициализирован")
    
    try:
        health_info = await services['vector_store'].health_check()
        return {
            "collection_info": health_info.get('collection_info', {}),
            "status": health_info.get('status'),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Ошибка получения информации о коллекции: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Статистика всех сервисов"""
    try:
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {}
        }
        
        if 'embedder' in services:
            stats['services']['embedder'] = services['embedder'].get_stats()
        
        if 'vector_store' in services:
            stats['services']['vector_store'] = services['vector_store'].get_stats()
        
        if 'search_service' in services:
            stats['services']['search_service'] = services['search_service'].get_search_stats()
        
        return stats
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

def check_service_status():
    """Проверка статуса сервиса через HTTP"""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("\n✅ Сервис работает")
            print(f"📊 Статус: {health_data.get('status', 'unknown')}")
            print(f"🕐 Время: {health_data.get('timestamp', 'N/A')}")
            
            # Показываем статус компонентов
            services_status = health_data.get('services', {})
            if services_status:
                print("\n📦 Компоненты:")
                for service_name, service_info in services_status.items():
                    status = service_info.get('status', 'unknown')
                    emoji = "✅" if status == "connected" else "⚠️"
                    print(f"  {emoji} {service_name}: {status}")
            
            # Статистика коллекции
            if 'vector_count' in health_data:
                print(f"\n📚 Векторов в коллекции: {health_data['vector_count']}")
            
            return True
        else:
            print(f"⚠️ Сервис вернул код {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Сервис не запущен (не удалось подключиться к localhost:8000)")
        return False
    except Exception as e:
        print(f"❌ Ошибка проверки статуса: {e}")
        return False

def stop_service():
    """Остановка сервиса"""
    import signal
    import subprocess
    
    try:
        # Ищем процесс uvicorn на порту 8000
        result = subprocess.run(
            ["lsof", "-ti:8000"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"🔍 Найдено процессов: {len(pids)}")
            
            for pid in pids:
                try:
                    pid_int = int(pid)
                    os.kill(pid_int, signal.SIGTERM)
                    print(f"✅ Процесс {pid} остановлен")
                except ProcessLookupError:
                    print(f"⚠️ Процесс {pid} уже остановлен")
                except Exception as e:
                    print(f"❌ Ошибка остановки процесса {pid}: {e}")
            
            print("✅ Сервис остановлен")
            return True
        else:
            print("ℹ️ Сервис не запущен (порт 8000 свободен)")
            return False
            
    except FileNotFoundError:
        print("⚠️ Команда lsof не найдена, используем альтернативный метод...")
        # Альтернативный метод через ps + grep
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if 'vm_rag_service.py' in line and 'python' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = int(parts[1])
                        os.kill(pid, signal.SIGTERM)
                        print(f"✅ Процесс {pid} остановлен")
                        return True
            
            print("ℹ️ Сервис не запущен")
            return False
        except Exception as e:
            print(f"❌ Ошибка остановки: {e}")
            return False
    except Exception as e:
        print(f"❌ Ошибка остановки сервиса: {e}")
        return False

def start_service():
    """Запуск сервиса"""
    logger.info("🚀 Запуск RAG-as-a-Service на VM...")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",  # Слушаем на всех интерфейсах
            port=8000,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"❌ Ошибка запуска сервиса: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG-as-a-Service управление на VM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python vm_rag_service.py          # Запустить сервис
  python vm_rag_service.py start    # Запустить сервис
  python vm_rag_service.py status   # Проверить статус
  python vm_rag_service.py stop     # Остановить сервис
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='start',
        choices=['start', 'stop', 'status'],
        help='Команда для выполнения (по умолчанию: start)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'status':
        success = check_service_status()
        sys.exit(0 if success else 1)
    
    elif args.command == 'stop':
        success = stop_service()
        sys.exit(0 if success else 1)
    
    elif args.command == 'start':
        start_service()
    
    else:
        parser.print_help()
        sys.exit(1)
