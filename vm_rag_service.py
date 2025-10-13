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
from pathlib import Path
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import numpy as np
import uvicorn

# Добавляем текущую директорию в Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Настройка логирования (ПЕРЕД импортами RAG модулей!)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройка диагностического логгера (ПОСЛЕ logger!)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Создаём отдельный handler для диагностики
diag_handler = logging.FileHandler(log_dir / "diagnostics.log", encoding='utf-8')
diag_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
diag_logger = logging.getLogger("diagnostics")
diag_logger.addHandler(diag_handler)
diag_logger.setLevel(logging.INFO)

# ✅ ИСПРАВЛЕНИЕ РЕКУРСИИ: Используем Factory Pattern
try:
    from rag.factory import RAGFactory
    from rag.context import ExecutionContext
    from rag.embedder import CPUEmbedder
    from rag.vector_store import QdrantVectorStore
    from rag.search_service import SearchService
    from rag.indexer_service import IndexerService
    from config import get_config
    
except ImportError as e:
    logger.error(f"Ошибка импорта: {e}")
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все зависимости установлены на VM")
    sys.exit(1)

# ✅ КРИТИЧНО: Устанавливаем VM контекст на уровне модуля (перед созданием app)!
# Это гарантирует, что все последующие вызовы RAGFactory используют VM контекст
# и создают локальные компоненты (CPUEmbedder, QdrantVectorStore), а НЕ remote-клиенты
RAGFactory.set_context(ExecutionContext.VM)
logger.info("🔧 VM контекст установлен на уровне модуля (перед созданием app)")

# Pydantic модели для API
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="Список текстов для эмбеддинга")
    task: Optional[str] = Field("retrieval.passage", description="Задача для dual task архитектуры")
    truncate_dim: Optional[int] = Field(None, description="Желаемая размерность эмбеддингов (по умолчанию 1024)")
    normalize: bool = Field(True, description="Применять L2 нормализацию")

class SearchRequest(BaseModel):
    # Векторный протокол (приоритет)
    dense_vector: Optional[List[float]] = Field(None, description="Dense вектор запроса (1024d)")
    sparse_vector: Optional[Dict[int, float]] = Field(None, description="Sparse вектор (BM25/SPLADE)")
    
    # Текстовый протокол (legacy, для совместимости)
    query: Optional[str] = Field(None, description="Поисковый запрос (legacy)")
    
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

# === Observability: JSON logging + Prometheus metrics + ASGI middleware ===
import time  # для высокоточного замера длительности
from fastapi import Request, Response  # локальные импорты допустимы
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Переключаем формат логгера на JSON (безопасный, без контента документов)
try:
    _root = logging.getLogger()
    # очищаем старые хендлеры
    for _h in list(_root.handlers):
        _root.removeHandler(_h)
    _json_handler = logging.StreamHandler()
    _json_formatter = jsonlogger.JsonFormatter(
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"}
    )
    _json_handler.setFormatter(_json_formatter)
    _root.addHandler(_json_handler)
    _root.setLevel(logging.INFO)
    # обновляем локальный логгер модуля
    logger = logging.getLogger(__name__)
except Exception as _e:
    # fallback на basicConfig, чтобы не падать при отсутствии зависимости
    logging.basicConfig(level=logging.INFO)
    logger.warning(f"JSON logging setup failed, fallback to basic config: {_e}")

# Prometheus метрики уровня модуля
request_duration_seconds = Histogram(
    'rag_request_duration_seconds',
    'Endpoint duration seconds',
    ['endpoint', 'status'],
    buckets=[0.05,0.1,0.2,0.5,1,2,3,5,10,20,60,120]
)
requests_total = Counter(
    'rag_requests_total',
    'Requests total',
    ['endpoint', 'status']
)
inprogress_requests = Gauge(
    'rag_inprogress_requests',
    'In-progress requests',
    ['endpoint']
)
dropped_documents_total = Counter(
    'rag_dropped_documents_total',
    'Dropped documents total',
    ['reason']
)
timeouts_total = Counter(
    'rag_timeouts_total',
    'Timeouts total',
    ['endpoint']
)
# Опционально: метрика по потреблению памяти процесса
memory_usage_bytes = Gauge(
    'rag_memory_usage_bytes',
    'Process memory usage bytes'
)

def _normalize_endpoint(path: str) -> str:
    """Нормализация пути к каноническим эндпоинтам для метрик."""
    if path in ('/health', '/v1/health'):
        return '/v1/health'
    if path in ('/embeddings', '/v1/embeddings'):
        return '/v1/embeddings'
    if path in ('/search', '/v1/search'):
        return '/v1/search'
    if path == '/v1/search_v2':
        return '/v1/search_v2'
    if path in ('/index', '/v1/index'):
        return '/v1/index'
    return 'other'

@app.middleware("http")
async def _observability_middleware(request: Request, call_next):
    """ASGI middleware: метрики Prometheus + структурные JSON-логи с корреляцией."""
    endpoint = _normalize_endpoint(request.url.path)
    inprogress_requests.labels(endpoint).inc()
    start = time.perf_counter()

    # Корреляция
    trace_id = request.headers.get('X-Trace-Id') or uuid.uuid4().hex
    batch_id = request.headers.get('X-Batch-Id') if endpoint == '/v1/index' else None

    status_code = 500
    try:
        response = await call_next(request)
        status_code = getattr(response, "status_code", 200)
        return response
    except asyncio.TimeoutError:
        # Учитываем таймауты
        try:
            timeouts_total.labels(endpoint).inc()
        except Exception:
            pass
        status_code = 504
        raise
    except Exception:
        status_code = 500
        raise
    finally:
        elapsed = time.perf_counter() - start
        # Метрики
        try:
            inprogress_requests.labels(endpoint).dec()
        except Exception:
            pass
        try:
            requests_total.labels(endpoint, str(status_code)).inc()
            request_duration_seconds.labels(endpoint, str(status_code)).observe(elapsed)
        except Exception:
            pass
        # Обновление gauge по памяти (опционально)
        try:
            memory_usage_bytes.set(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass

        # Безопасное структурное логирование без контента
        extra = {
            "endpoint": endpoint,
            "trace_id": trace_id,
            "elapsed_ms": int(elapsed * 1000),
        }
        if batch_id:
            extra["batch_id"] = batch_id

        # Если эндпоинт-обработчик положил агрегаты в request.state — добавим их в лог
        counts = {}
        st = getattr(request, "state", None)
        if st:
            if hasattr(st, "documents_count"):
                counts["documents_count"] = getattr(st, "documents_count")
            if hasattr(st, "results_count"):
                counts["results_count"] = getattr(st, "results_count")
        if counts:
            extra["counts"] = counts

        try:
            logger.info("request_completed", extra=extra)
        except Exception:
            # Логирование не должно ломать обработку запроса
            pass

# Эндпоинт метрик Prometheus
@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
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
@app.get("/v1/health")
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
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest, http_request: Request):
    """Получение эмбеддингов от Jina v3"""
    if 'embedder' not in services:
        raise HTTPException(status_code=503, detail="Embedder не инициализирован")
    
    try:
        # Проверяем память перед тяжелой операцией
        memory_check_middleware()
        try:
            http_request.state.documents_count = len(request.texts or [])
        except Exception:
            pass
        
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
@app.post("/v1/search", response_model=SearchResponse)
@app.post("/v1/search_v2", response_model=SearchResponse)
async def search_documents(request: SearchRequest, http_request: Request):
    """Гибридный поиск по документам (поддержка векторного и текстового протоколов)"""
    if 'search_service' not in services:
        raise HTTPException(status_code=503, detail="Search service не инициализирован")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Векторный протокол (приоритет) - используем готовые векторы
        if request.dense_vector:
            logger.info("🔵 Векторный протокол: используем готовые векторы")
            
            # Конвертируем list в numpy array
            dense_vector = np.array(request.dense_vector, dtype=np.float32)
            
            # Прямой поиск в vector_store с готовыми векторами
            raw_results = await asyncio.to_thread(
                services['vector_store'].search,
                query_vector=dense_vector,
                top_k=request.top_k,
                filters=request.filters,
                use_hybrid=request.use_hybrid,
                sparse_vector=request.sparse_vector
            )
            
            # Конвертируем в SearchResult объекты
            from rag.search_service import SearchResult
            results = []
            for result in raw_results:
                payload = result.get('payload', {})
                search_result = SearchResult(
                    chunk_id=result['id'],
                    file_path=payload.get('file_path', ''),
                    file_name=payload.get('file_name', ''),
                    chunk_name=payload.get('chunk_name', ''),
                    chunk_type=payload.get('chunk_type', ''),
                    language=payload.get('language', ''),
                    start_line=payload.get('start_line', 0),
                    end_line=payload.get('end_line', 0),
                    score=result['score'],
                    content=payload.get('content', ''),
                    metadata=payload
                )
                results.append(search_result)
        
        # Текстовый протокол (legacy fallback)
        elif request.query:
            logger.info("🟡 Текстовый протокол (legacy): генерируем эмбеддинги из текста")
            if not request.query.strip():
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": {
                            "type": "validation_error",
                            "message": "text query must be non-empty",
                            "details": [
                                {"field": "query", "issue": "empty"}
                            ],
                            "request_id": str(uuid.uuid4()),
                            "api_contract": "v1.0.0"
                        }
                    }
                )
            # Выполняем поиск через SearchService (с генерацией эмбеддингов)
            results = await services['search_service'].search(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
                use_hybrid=request.use_hybrid,
                task=request.task
            )
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "type": "validation_error",
                        "message": "either dense_vector or query is required",
                        "details": [
                            {"field": "dense_vector", "issue": "missing"},
                            {"field": "query", "issue": "missing"}
                        ],
                        "request_id": str(uuid.uuid4()),
                        "api_contract": "v1.0.0"
                    }
                }
            )
        
        query_time = asyncio.get_event_loop().time() - start_time
        
        # Конвертируем SearchResult объекты в словари для Pydantic
        try:
            http_request.state.results_count = len(results)
        except Exception:
            pass
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
@app.post("/v1/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks, http_request: Request):
    """Индексация документов с защитой от OOM"""
    logger.info(f"🔵 НАЧАЛО endpoint /index: получено {len(request.documents) if hasattr(request, 'documents') else 0} документов")
    try:
        http_request.state.documents_count = len(request.documents or [])
    except Exception:
        pass

    if 'indexer_service' not in services:
        logger.error("❌ IndexerService не инициализирован!")
        raise HTTPException(status_code=503, detail="Indexer service не инициализирован")
    
    try:
        logger.info("🔵 Шаг 1: Проверка памяти...")
        # КРИТИЧНО: Проверяем память перед индексацией
        memory_info = memory_check_middleware()
        logger.info(f"🧠 Память перед индексацией: {memory_info.get('percent_used', 0):.1f}%")
        
        # Автоматически уменьшаем batch_size при высоком потреблении памяти
        original_batch_size = request.batch_size
        if memory_info.get('is_warning', False):
            request.batch_size = max(1, original_batch_size // 4)  # ✅ ИСПРАВЛЕНО: max вместо min
            logger.warning(f"⚠️ Уменьшен batch_size: {original_batch_size} -> {request.batch_size}")
        
        start_time = asyncio.get_event_loop().time()
        
        logger.info("🔵 Шаг 2: Диагностика входных данных...")
        # 🔍 ДИАГНОСТИКА 1: Что получил VM endpoint
        diag_logger.info(f"📥 VM: Получено документов: {len(request.documents)}")
        if request.documents:
            first_doc_raw = request.documents[0]
            diag_logger.info(f"📥 VM: Тип документа = {type(first_doc_raw)}")
            if isinstance(first_doc_raw, dict):
                keys = list(first_doc_raw.keys())
                has_text = 'text' in first_doc_raw
                text_len = len((first_doc_raw.get('text') or ''))
                diag_logger.info(f"📥 VM: Ключи документа = {keys}")
                diag_logger.info(f"📥 VM: Поля: has_text={has_text}, text_len={text_len}")
        
        logger.info("🔵 Шаг 3: Подготовка points...")
        # Подготавливаем документы для индексации
        points = []
        invalid_docs = []
        for doc in request.documents:
            # ✅ ИСПРАВЛЕНИЕ: Извлекаем текст из правильного места
            # Сначала пробуем doc['text'], если нет - берём doc['payload']['content']
            text = (doc.get('text', '') or doc.get('payload', {}).get('content', '')).strip()
            if not text:
                invalid_docs.append({
                    "id": doc.get('id') or doc.get('metadata', {}).get('file_path', 'unknown'),
                    "reason": "empty_text"
                })
            point = {
                'id': doc.get('id'),
                'text': text,
                'metadata': doc.get('metadata', {}),
                'timestamp': doc.get('timestamp', datetime.now(timezone.utc).isoformat())
            }
            points.append(point)
        
        logger.info(f"🔵 Шаг 4: Points подготовлены: {len(points)} точек")
        # Валидация: отклоняем пустые тексты до индексации
        if invalid_docs:
            try:
                dropped_documents_total.labels('empty_text').inc(len(invalid_docs))
            except Exception:
                pass
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "type": "validation_error",
                        "message": "text must be non-empty",
                        "details": invalid_docs,
                        "request_id": str(uuid.uuid4()),
                        "api_contract": "v1.0.0"
                    }
                }
            )
        # 🔍 ДИАГНОСТИКА 2: Что передаём в IndexerService
        if points:
            first_point = points[0]
            safe_meta_keys = list((first_point.get('metadata') or {}).keys())
            text_len = len(first_point.get('text') or '')
            diag_logger.info(f"📤 VM: Первый point после обработки - keys={list(first_point.keys())}, meta_keys={safe_meta_keys}, text_len={text_len}")
        
        logger.info("🔵 Шаг 5: Вызов IndexerService.index_documents()...")
        # Выполняем индексацию
        indexed_count = await services['indexer_service'].index_documents(
            documents=points,
            batch_size=request.batch_size,
            recreate_collection=request.recreate
        )
        
        logger.info(f"🔵 Шаг 6: IndexerService завершён, indexed_count={indexed_count}")
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"🔵 Шаг 7: processing_time={processing_time:.3f}s")
        
        logger.info("🔵 Шаг 8: Получение collection_info...")
        # Получаем информацию о коллекции
        collection_info = {}
        if 'vector_store' in services:
            vs_health = await services['vector_store'].health_check()
            collection_info = vs_health.get('collection_info', {})
        
        logger.info(f"🔵 Шаг 9: collection_info получен: {collection_info}")
        
        logger.info("🔵 Шаг 10: Создание IndexResponse...")
        response = IndexResponse(
            indexed_count=indexed_count,
            status="success",
            processing_time=processing_time,
            collection_info=collection_info
        )
        
        logger.info(f"🔵 Шаг 11: ВОЗВРАЩАЕМ response: indexed_count={indexed_count}, status=success")
        return response
        
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
