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
import hashlib
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, conlist, conint, confloat
import pydantic as _p
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
logger.info(f"Pydantic version (runtime): {_p.__version__}")

# Pydantic v1/v2 совместимость для строковых ограничений
try:
    from typing_extensions import Annotated
except ImportError:
    from typing import Annotated

try:
    from pydantic import StringConstraints as _StrC
except ImportError:
    _StrC = None

# Настройка диагностического логгера (ПОСЛЕ logger!)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Создаём отдельный handler для диагностики
diag_handler = logging.FileHandler(log_dir / "diagnostics.log", encoding='utf-8')
diag_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
diag_logger = logging.getLogger("diagnostics")
diag_logger.addHandler(diag_handler)
diag_logger.setLevel(logging.INFO)

# Универсальный шов ConStr: v2 -> Annotated[str, StringConstraints], v1 -> pydantic.constr
def ConStr(min_length: int = None, regex: str = None):
    if _StrC is not None:
        kwargs = {}
        if min_length is not None:
            kwargs['min_length'] = min_length
        if regex is not None:
            kwargs['pattern'] = regex
        return Annotated[str, _StrC(**kwargs)]
    else:
        # v1: вернуть функциональный тип через pydantic.constr
        from pydantic import constr as _v1_constr
        return _v1_constr(min_length=min_length, regex=regex)

# ✅ ИСПРАВЛЕНИЕ РЕКУРСИИ: Используем Factory Pattern
try:
    from rag.factory import RAGFactory
    from rag.context import ExecutionContext
    from rag.embedder import CPUEmbedder
    from rag.vector_store import QdrantVectorStore
    from rag.search_service import SearchService
    from rag.indexer_service import IndexerService, IndexingRejectedAll, IndexBatchResult
    from rag.exceptions import VectorStoreConnectionError
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

class RejectedReason(BaseModel):
    id: ConStr(min_length=1)
    reason: ConStr(min_length=1)
    details: Optional[Dict[str, Any]] = None

class IndexedMetadata(BaseModel):
    file_path: ConStr(min_length=1) = Field(..., description="Путь к файлу")
    line_start: conint(ge=0) = Field(..., description="Начальная строка (0 или больше)")
    line_end: conint(ge=0) = Field(..., description="Конечная строка (0 или больше)")
    language: ConStr(min_length=1) = Field(..., description="Язык файла")
    repo: ConStr(min_length=1) = Field(..., description="Репозиторий")
    chunk_type: ConStr(min_length=1) = Field(..., description="Тип чанка")

class IndexedDocument(BaseModel):
    id: ConStr(min_length=1) = Field(..., description="Уникальный идентификатор документа")
    text: ConStr(min_length=1) = Field(..., description="Текст документа (не пустой)")
    metadata: IndexedMetadata
    embedding_version: ConStr(min_length=1) = Field(..., description="Версия эмбеддинга/схемы")
    content_sha256: ConStr(regex=r'^[A-Fa-f0-9]{64}$') = Field(..., description="SHA256 контента (64 hex)")
    # Опционально поддерживаем ключ идемпотентности от клиента
    document_idempotency_key: Optional[str] = Field(None, description="Опциональный ключ идемпотентности от клиента")

class IndexRequest(BaseModel):
    api_contract: ConStr(min_length=1) = Field(..., description="Версия контракта API, ожидается 'v1.0.0'")
    batch_id: Optional[ConStr(min_length=1)] = Field(None, description="Опциональный UUID-подобный идентификатор батча")
    documents: conlist(IndexedDocument, min_items=1, max_items=128) = Field(..., description="Список документов (1..128)")

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
    accepted: int
    rejected: int
    rejected_reasons: List[RejectedReason] = Field(default_factory=list)
    elapsed_ms: int

class ErrorObject(BaseModel):
    type: ConStr(min_length=1)
    message: ConStr(min_length=1)
    details: List[RejectedReason] = Field(default_factory=list)
    request_id: ConStr(min_length=1)
    api_contract: ConStr(min_length=1)

class IndexErrorResponse(BaseModel):
    error: ErrorObject

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
    # Явно указываем поля в формате JSON-логов и переименовываем стандартные
    _json_formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(message)s %(endpoint)s %(trace_id)s %(batch_id)s %(elapsed_ms)s %(counts)s",
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
    path = request.url.path
    if path == "/metrics":
        # Исключаем /metrics из инструментирования
        return await call_next(request)
    endpoint = _normalize_endpoint(path)
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
        # Учитываем таймауты и логируем предупреждение (безопасные поля)
        try:
            timeouts_total.labels(endpoint).inc()
        except Exception:
            pass
        elapsed_to_now = time.perf_counter() - start
        warn_extra = {
            "endpoint": endpoint,
            "trace_id": trace_id,
            "elapsed_ms": int(elapsed_to_now * 1000),
        }
        if batch_id:
            warn_extra["batch_id"] = batch_id
        try:
            logger.warning("request_timeout", extra=warn_extra)
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
        # Валидация числовых параметров
        if request.top_k is None or request.top_k <= 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "type": "validation_error",
                        "message": "invalid top_k",
                        "details": [
                            {"field": "top_k", "issue": "must_be_positive"}
                        ],
                        "request_id": str(uuid.uuid4()),
                        "api_contract": "v1.0.0"
                    }
                }
            )
        
        # Векторный протокол (приоритет) - используем готовые векторы
        if request.dense_vector:
            logger.info("🔵 Векторный протокол: используем готовые векторы")
            
            # Серверная валидация dense-вектора (Фаза 4.3)
            # 1) Предварительная проверка типа/наличия (без логирования содержимого)
            if not isinstance(request.dense_vector, (list, tuple, np.ndarray)):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": {
                            "type": "validation_error",
                            "message": "invalid dense_vector",
                            "details": [
                                {"field": "dense_vector", "issue": "invalid_type"}
                            ],
                            "request_id": str(uuid.uuid4()),
                            "api_contract": "v1.0.0"
                        }
                    }
                )
            
            # 2) Конвертация в np.array
            try:
                dense_vector = np.array(request.dense_vector, dtype=np.float32)
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": {
                            "type": "validation_error",
                            "message": "invalid dense_vector",
                            "details": [
                                {"field": "dense_vector", "issue": "invalid_type"}
                            ],
                            "request_id": str(uuid.uuid4()),
                            "api_contract": "v1.0.0"
                        }
                    }
                )
            
            # 3) Определение ожидаемой размерности
            expected_dim = None
            try:
                cfg = get_config()
                # Предпочтительно из конфига (embedding_dim/truncate_dim), затем vector_size
                expected_dim = (
                    getattr(cfg.rag.embeddings, "embedding_dim", None)
                    or getattr(cfg.rag.embeddings, "truncate_dim", None)
                    or getattr(cfg.rag.vector_store, "vector_size", None)
                )
            except Exception:
                expected_dim = None
            
            if not expected_dim:
                try:
                    emb = services.get("embedder")
                    if emb is not None:
                        stats = {}
                        try:
                            stats = emb.get_stats()
                        except Exception:
                            stats = {}
                        expected_dim = (
                            stats.get("dimension")
                            or stats.get("embedding_dim")
                            or getattr(emb, "embedding_dim", None)
                        )
                except Exception:
                    expected_dim = None
            
            if not expected_dim:
                expected_dim = 1024  # Fallback
            
            # 4) Пост-валидация размерности и числовой валидности
            len_vec = dense_vector.shape[0] if dense_vector.ndim == 1 else dense_vector.size
            has_nan = bool(np.isnan(dense_vector).any())
            has_inf = bool(np.isinf(dense_vector).any())
            
            # Безопасные диагностические логи (только агрегаты)
            try:
                diag_logger.info(f"dense_vector_diag: dim={len_vec}, has_nan={has_nan}, has_inf={has_inf}")
            except Exception:
                pass
            
            issues = []
            if int(len_vec) != int(expected_dim):
                issues.append({
                    "field": "dense_vector",
                    "issue": "invalid_dim",
                    "expected": int(expected_dim),
                    "actual": int(len_vec)
                })
            if has_nan or has_inf:
                issues.append({"field": "dense_vector", "issue": "nan_inf_detected"})
            
            if issues:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": {
                            "type": "validation_error",
                            "message": "invalid dense_vector",
                            "details": issues,
                            "request_id": str(uuid.uuid4()),
                            "api_contract": "v1.0.0"
                        }
                    }
                )
            
            # 5) Прямой поиск в vector_store с валидным вектором
            raw_results = await services['vector_store'].search(
                query_vector=dense_vector,
                top_k=request.top_k,
                filters=request.filters,
                use_hybrid=request.use_hybrid,
                sparse_vector=request.sparse_vector
            )
            
            # Конвертируем в SearchResult объекты
            from rag.search_service import SearchResult
            results = []
            logger.debug(f"type(raw_results)={type(raw_results)}")
            if hasattr(raw_results, "__aiter__"):
                raw_results = [x async for x in raw_results]
            elif asyncio.iscoroutine(raw_results):
                raw_results = await raw_results
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
        
        # Текстовый протокол (legacy fallback) и sparse-only ветка
        elif request.sparse_vector is not None:
            logger.info("🔵 Векторный протокол: sparse-only поиск")
            # Минимальная валидация sparse_vector
            if not isinstance(request.sparse_vector, dict) or not request.sparse_vector:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": {
                            "type": "validation_error",
                            "message": "invalid sparse_vector",
                            "details": [
                                {"field": "sparse_vector", "issue": "invalid_type_or_empty"}
                            ],
                            "request_id": str(uuid.uuid4()),
                            "api_contract": "v1.0.0"
                        }
                    }
                )
            # Выполняем чисто sparse-поиск напрямую через vector_store
            vs = services['vector_store']
            search_filter = vs._build_search_filter(request.filters)
            raw_results = await vs._search_sparse(
                sparse_vector=request.sparse_vector,
                top_k=request.top_k,
                search_filter=search_filter
            )
            from rag.search_service import SearchResult
            results = []
            logger.debug(f"type(raw_results)={type(raw_results)}")
            if hasattr(raw_results, "__aiter__"):
                raw_results = [x async for x in raw_results]
            elif asyncio.iscoroutine(raw_results):
                raw_results = await raw_results
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
        elif request.query is not None:
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
            # Ветка без явного протокола
            path = http_request.url.path
            if path == "/v1/search_v2":
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": {
                            "type": "validation_error",
                            "message": "either dense_vector or sparse_vector is required",
                            "details": [
                                {"field": "dense_vector", "issue": "missing"},
                                {"field": "sparse_vector", "issue": "missing"}
                            ],
                            "request_id": str(uuid.uuid4()),
                            "api_contract": "v1.0.0"
                        }
                    }
                )
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
async def index_documents(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    embedding_version_header: Optional[str] = Header(None, alias="X-Embedding-Version"),
    api_contract_header: Optional[str] = Header(None, alias="X-API-Contract"),
    batch_id_header: Optional[str] = Header(None, alias="X-Batch-Id"),
):
    """
    Индексация документов в соответствии с контрактом:
    - Успех: 200 и {accepted, rejected, rejected_reasons[], elapsed_ms}
    - Полная отбраковка: 422 и IndexErrorResponse
    - Ошибки бэкенда: 503
    """
    # Метаданные запроса
    try:
        http_request.state.documents_count = len(request.documents or [])
    except Exception:
        pass

    if 'indexer_service' not in services:
        logger.error("❌ IndexerService не инициализирован!")
        return JSONResponse(status_code=503, content={
            "error": {
                "type": "backend_unavailable",
                "message": "Indexer service не инициализирован",
                "details": [],
                "request_id": str(uuid.uuid4()),
                "api_contract": "v1.0.0"
            }
        })

    # Контракт API: заголовок имеет приоритет как источник истины
    api_contract = (api_contract_header or request.api_contract or "").strip()
    if api_contract != "v1.0.0":
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "invalid_request",
                    "message": "Unsupported or mismatched api_contract",
                    "details": [{"field": "api_contract", "issue": "unsupported_or_mismatch", "expected": "v1.0.0", "actual": api_contract}],
                    "request_id": str(uuid.uuid4()),
                    "api_contract": api_contract or "unknown"
                }
            }
        )

    # Ограничение размера батча (количества документов в запросе)
    docs_count = len(request.documents)
    if docs_count < 1 or docs_count > 128:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "invalid_request",
                    "message": "batch size out of allowed range",
                    "details": [{"issue": "batch_size_out_of_range", "min": 1, "max": 128, "actual": docs_count}],
                    "request_id": str(uuid.uuid4()),
                    "api_contract": api_contract
                }
            }
        )

    # Корреляция
    batch_id = batch_id_header or request.batch_id
    start_perf = asyncio.get_event_loop().time()

    # Проверка памяти
    memory_info = memory_check_middleware()

    # Preflight-валидации (агрегация отказов)
    preflight_rejected: Dict[str, Dict[str, Any]] = {}
    normalized_docs: List[Dict[str, Any]] = []

    enforce_sha = (os.getenv("RAG_ENFORCE_SHA256", "false").lower() in ("1", "true", "yes"))
    eff_embedding_version = (embedding_version_header or "").strip() or None

    for doc in request.documents:
        doc_id = doc.id
        text = (doc.text or "").strip()
        if not text:
            preflight_rejected[doc_id] = {"reason": "empty_text", "details": None}
            continue

        # Проверка SHA256 при включённой фиче
        if enforce_sha:
            computed = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if (doc.content_sha256 or "").lower() != computed.lower():
                preflight_rejected[doc_id] = {
                    "reason": "sha256_mismatch",
                    "details": {"expected": computed, "actual": doc.content_sha256}
                }
                continue

        meta = doc.metadata.dict()
        ev = eff_embedding_version or doc.embedding_version
        meta['embedding_version'] = ev

        normalized_docs.append({
            "id": doc_id,
            "text": text,
            "metadata": meta,
            "embedding_version": ev,
            "document_idempotency_key": doc.document_idempotency_key,
            "content_sha256": doc.content_sha256,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    # Учёт preflight-дропов в метриках
    if preflight_rejected:
        reasons_count: Dict[str, int] = {}
        for _, r in preflight_rejected.items():
            reasons_count[r["reason"]] = reasons_count.get(r["reason"], 0) + 1
        for reason, cnt in reasons_count.items():
            try:
                dropped_documents_total.labels(reason).inc(cnt)
            except Exception:
                pass

    # Если все документы отбраковались на preflight
    if not normalized_docs and preflight_rejected:
        elapsed_ms = int((asyncio.get_event_loop().time() - start_perf) * 1000)
        # Лог агрегатов
        logger.info(f"/index rejected all at preflight: accepted=0 rejected={len(preflight_rejected)} batch_id={batch_id} elapsed_ms={elapsed_ms}")
        try:
            http_request.state.results_count = 0
        except Exception:
            pass

        details = [
            {"id": did, "reason": info["reason"], "details": info.get("details")}
            for did, info in preflight_rejected.items()
        ]
        return JSONResponse(
            status_code=422,
            content=IndexErrorResponse(
                error=ErrorObject(
                    type="validation_error",
                    message="all documents rejected",
                    details=[RejectedReason(**d) for d in details],
                    request_id=str(uuid.uuid4()),
                    api_contract=api_contract
                )
            ).dict()
        )

    # Вызов IndexerService
    try:
        batch_result: IndexBatchResult = await services['indexer_service'].index_documents(
            documents=normalized_docs,
            batch_size=min(128, 128),  # жёсткий лимит согласно контракту
            recreate_collection=False
        )
        accepted_ids = set(batch_result.accepted_ids or [])
        rejected_map = dict(preflight_rejected)
        # Дополняем причинами отказов, пришедшими из сервиса
        for did, info in (batch_result.rejected or {}).items():
            reason = info.get("reason") or "unknown"
            details = info.get("details")
            rejected_map[did] = {"reason": reason, "details": details}

        accepted = len(accepted_ids)
        rejected = len(rejected_map)
        elapsed_ms = int((asyncio.get_event_loop().time() - start_perf) * 1000)

        # Логи и state для middleware
        try:
            http_request.state.results_count = accepted + rejected
        except Exception:
            pass
        logger.info(f"/index completed: accepted={accepted} rejected={rejected} batch_id={batch_id} elapsed_ms={elapsed_ms}")

        # Если все документы отвергнуты -> 422
        if accepted == 0 and rejected > 0:
            details = [
                {"id": did, "reason": info["reason"], "details": info.get("details")}
                for did, info in rejected_map.items()
            ]
            return JSONResponse(
                status_code=422,
                content=IndexErrorResponse(
                    error=ErrorObject(
                        type="validation_error",
                        message="all documents rejected",
                        details=[RejectedReason(**d) for d in details],
                        request_id=str(uuid.uuid4()),
                        api_contract=api_contract
                    )
                ).dict()
            )

        # Частичный или полный успех -> 200
        rejected_reasons_list = [
            RejectedReason(id=did, reason=info["reason"], details=info.get("details"))
            for did, info in rejected_map.items()
        ]
        return IndexResponse(
            accepted=accepted,
            rejected=rejected,
            rejected_reasons=rejected_reasons_list,
            elapsed_ms=elapsed_ms
        )

    except IndexingRejectedAll as e:
        # Полная отбраковка на уровне сервиса
        rejected_map = dict(preflight_rejected)
        rejected_map.update(getattr(e, "rejected", {}) or {})
        details = [
            {"id": did, "reason": info.get("reason", "rejected"), "details": info.get("details")}
            for did, info in rejected_map.items()
        ]
        elapsed_ms = int((asyncio.get_event_loop().time() - start_perf) * 1000)
        logger.info(f"/index rejected all by service: accepted=0 rejected={len(rejected_map)} batch_id={batch_id} elapsed_ms={elapsed_ms}")
        return JSONResponse(
            status_code=422,
            content=IndexErrorResponse(
                error=ErrorObject(
                    type="validation_error",
                    message="all documents rejected",
                    details=[RejectedReason(**d) for d in details],
                    request_id=str(uuid.uuid4()),
                    api_contract=api_contract
                )
            ).dict()
        )
    except (VectorStoreConnectionError, asyncio.TimeoutError) as e:
        # Бэкенд недоступен/таймаут
        elapsed_ms = int((asyncio.get_event_loop().time() - start_perf) * 1000)
        logger.error(f"/index backend unavailable: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "backend_unavailable",
                    "message": "vector store or dependencies unavailable",
                    "details": [{"issue": "exception", "message": str(e)}],
                    "request_id": str(uuid.uuid4()),
                    "api_contract": api_contract
                }
            }
        )
    except Exception as e:
        # Ошибки upsert/прочие
        elapsed_ms = int((asyncio.get_event_loop().time() - start_perf) * 1000)
        logger.error(f"/index upsert_failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "upsert_failed",
                    "message": "failed to index documents",
                    "details": [{"issue": "exception", "message": str(e)}],
                    "request_id": str(uuid.uuid4()),
                    "api_contract": api_contract
                }
            }
        )

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
