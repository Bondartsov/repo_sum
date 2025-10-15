"""
HTTP клиент для удалённого векторного хранилища через RAG-as-a-Service на VM.

Заменяет прямое подключение к Qdrant на HTTP запросы к FastAPI сервису на VM,
где работает Qdrant с 1024d векторами и гибридным поиском.
"""

import os
import logging
import time
import asyncio
import hashlib
import uuid
import json
from uuid import uuid5, NAMESPACE_URL
from typing import List, Dict, Optional, Any, Callable
from config import RemoteServiceConfig, get_config
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import aiohttp
from aiohttp import ClientTimeout
from .event_loop_manager import run_async_safe
from .transport_client import AiohttpTransportClient
from .vm_diagnostics import diagnose_vm_connection
from .retry_policy import RetryPolicy, RetryConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)

# Настройка диагностического логгера
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Создаём отдельный handler для диагностики
diag_handler = logging.FileHandler(log_dir / "diagnostics.log", encoding='utf-8')
diag_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
diag_logger = logging.getLogger("diagnostics")
diag_logger.addHandler(diag_handler)
# Уровень по умолчанию INFO; включить DEBUG при RAG_DIAGNOSTICS_DEBUG=true
diag_logger.setLevel(logging.INFO)
try:
    if os.getenv("RAG_DIAGNOSTICS_DEBUG", "false").strip().lower() == "true":
        diag_logger.setLevel(logging.DEBUG)
except Exception:
    pass


def _safe_message(message: str) -> str:
    if not isinstance(message, str):
        return str(message)
    try:
        message.encode('ascii')
        return message
    except UnicodeEncodeError:
        return message.encode('ascii', 'ignore').decode('ascii')

def _log(logger_method, message: str, *args, **kwargs):
    logger_method(_safe_message(message), *args, **kwargs)


def _compute_content_sha256(text: str) -> str:
    """Возвращает hex-строку SHA-256 для UTF-8 текста."""
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_stable_point_id(meta: dict, embedding_version: str, content_sha256: str, fallback_str: str) -> str:
    """
    Генерирует детерминированный UUIDv5 для point.id на основе метаданных чанка.
    Основано на (file_path, line_start, line_end, embedding_version, content_sha256).
    """
    key = f"{meta.get('file_path','')}|{meta.get('line_start','')}|{meta.get('line_end','')}|{embedding_version}|{content_sha256}"
    return str(uuid5(NAMESPACE_URL, key))


def _normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Нормализует метаданные для соответствия серверному контракту.
    
    Преобразует синонимичные поля (start_line/end_line → line_start/line_end)
    и гарантирует наличие всех обязательных полей по контракту IndexedMetadata.
    
    Args:
        meta: Исходные метаданные
        
    Returns:
        Нормализованные метаданные с правильными именами полей
    """
    m = dict(meta or {})
    
    # Переименовываем синонимы в требуемые поля
    if "line_start" not in m and "start_line" in m:
        m["line_start"] = int(m.get("start_line") or 0)
    if "line_end" not in m and "end_line" in m:
        m["line_end"] = int(m.get("end_line") or 0)
    
    # Обязательные поля по контракту сервера (IndexedMetadata)
    m["file_path"] = m.get("file_path") or "unknown"
    m["language"] = m.get("language") or "unknown"
    m["repo"] = m.get("repo") or "default"
    m["chunk_type"] = m.get("chunk_type") or "code"
    
    # Чистим легаси-ключи
    m.pop("start_line", None)
    m.pop("end_line", None)
    
    return m


class RemoteVMVectorStore:
    """
    HTTP клиент для удалённого векторного хранилища на VM.
    
    Возможности:
    - HTTP запросы к Qdrant через FastAPI сервис на VM
    - Индексация документов через HTTP
    - Гибридный поиск (dense + sparse) через HTTP
    - Health check удалённого сервиса
    """
    
    def __init__(self, vector_store_config=None, remote_service_config: Optional[RemoteServiceConfig] = None, transport_client: Optional[AiohttpTransportClient] = None):
        """
        Инициализация удалённого векторного хранилища.
        
        Args:
            vector_store_config: Конфигурация векторного хранилища (игнорируется, для совместимости)
        """
        # Читаем конфигурацию из переменных окружения
        self.remote_config = remote_service_config or RemoteServiceConfig()
        env_host = os.getenv("RAG_SERVICE_HOST")
        env_port = os.getenv("RAG_SERVICE_PORT")
        host = env_host or self.remote_config.host
        port = int(env_port) if env_port is not None else self.remote_config.port
        base_url = f"http://{host}:{port}"
        self.search_endpoint = os.getenv("RAG_SEARCH_ENDPOINT", base_url + self.remote_config.search_endpoint)
        self.index_endpoint = os.getenv("RAG_INDEX_ENDPOINT", base_url + self.remote_config.index_endpoint)
        self.text_search_endpoint = os.getenv("RAG_TEXT_SEARCH_ENDPOINT", base_url + "/v1/search")
        self.health_endpoint = os.getenv("RAG_HEALTH_ENDPOINT", f"http://{host}:{port}{self.remote_config.health_endpoint}")
        self.service_host = host
        self.service_port = port
        # Заголовки контракта API и версия эмбеддингов
        self.api_contract = os.getenv("RAG_API_CONTRACT", "v1.0.0")
        self.embedding_version = os.getenv("RAG_EMBEDDING_VERSION", "2025-10-A")

        self.timeout_seconds = int(os.getenv("RAG_TIMEOUT_SECONDS", str(self.remote_config.timeout_seconds)))
        self.max_retries = int(os.getenv("RAG_MAX_RETRIES", str(self.remote_config.max_retries)))
        self.retry_delay = float(os.getenv("RAG_RETRY_DELAY", str(self.remote_config.retry_delay)))

        # Таймауты для разных операций (конфигурируемые)
        self.search_timeout = int(os.getenv("RAG_SEARCH_TIMEOUT", "300"))  # 5 минут для поиска
        self.index_timeout = int(os.getenv("RAG_INDEX_TIMEOUT", "1800"))  # 30 минут для индексации
        self.health_timeout = int(os.getenv("RAG_HEALTH_TIMEOUT", "60"))  # 1 минута для health check

        # Инициализация транспортного клиента (если не передан)
        default_headers = {
            "X-API-Contract": self.api_contract,
            "X-Embedding-Version": self.embedding_version,
        }
        self.transport_client = transport_client or AiohttpTransportClient(default_headers=default_headers)

        # === TimeoutProfiles + RetryPolicy + CircuitBreaker (пер-эндпойнтово) ===
        try:
            cfg = get_config(require_api_key=False)
            self.timeout_profiles = getattr(cfg.rag, "timeout_profiles", None)
        except Exception:
            self.timeout_profiles = None

        # Настройка профилей ретраев для /search и /index
        rp_search = (self.timeout_profiles.retry_search if self.timeout_profiles else None)
        rp_index = (self.timeout_profiles.retry_index if self.timeout_profiles else None)

        # Безопасные дефолты
        search_max_attempts = int(rp_search.max_attempts) if rp_search else 3
        search_base_delay = float(rp_search.base_delay) if rp_search else 1.0
        search_max_delay  = float(rp_search.max_delay)  if rp_search else 8.0

        index_max_attempts = int(rp_index.max_attempts) if rp_index else 5
        index_base_delay = float(rp_index.base_delay) if rp_index else 2.0
        index_max_delay  = float(rp_index.max_delay)  if rp_index else 60.0

        # Политики повторов (совместимы с embedder)
        self.retry_policy_search = RetryPolicy(RetryConfig(
            max_attempts=search_max_attempts,
            base_delay=search_base_delay,
            max_delay=search_max_delay,
            exponential_base=2.0,
            timeout_seconds=float(self.timeout_seconds),
            retryable_exceptions=(
                asyncio.TimeoutError,
                aiohttp.ClientError,
                ConnectionError,
                RuntimeError,
                ValueError,
            )
        ))
        self.retry_policy_index = RetryPolicy(RetryConfig(
            max_attempts=index_max_attempts,
            base_delay=index_base_delay,
            max_delay=index_max_delay,
            exponential_base=2.0,
            timeout_seconds=float(self.timeout_seconds),
            retryable_exceptions=(
                asyncio.TimeoutError,
                aiohttp.ClientError,
                ConnectionError,
                RuntimeError,
                ValueError,
            )
        ))
        
        # Diagnostic retries override via env (RAG_INDEX_RETRIES)
        try:
            _rir_str = os.getenv("RAG_INDEX_RETRIES", "").strip()
            rir_val = int(_rir_str) if _rir_str != "" else None
        except Exception:
            rir_val = None
        # diagnostic_no_retry flag preserved for back-compat logs
        self.diagnostic_no_retry = (rir_val == 0)
        if rir_val is not None:
            try:
                # Semantics: max_attempts = RAG_INDEX_RETRIES + 1 (first attempt + N retries)
                self.retry_policy_index.config.max_attempts = max(1, int(rir_val) + 1)
                if rir_val == 0:
                    diag_logger.info("diagnostic no-retry mode active (RAG_INDEX_RETRIES=0)")
                else:
                    diag_logger.info(f"diagnostic retries override: RAG_INDEX_RETRIES={rir_val} -> max_attempts={self.retry_policy_index.config.max_attempts}")
            except Exception:
                pass

        # Circuit Breaker (консервативные дефолты)
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=2,
            timeout_seconds=300.0,
            half_open_max_calls=1
        ))

        # Статистика
        self.stats = {
            'total_searches': 0,
            'total_indexed': 0,
            'total_search_time': 0.0,
            'total_index_time': 0.0,
            'error_count': 0,
            'retry_count': 0,
            'empty_text_detected_total': 0,
            'client_dropped_too_large_total': 0,
        }
        
        self._connected = False
        self._collection_exists = False
        
        _log(logger.info, f"RemoteVMVectorStore инициализирован: поиск={self.search_endpoint}, индексация={self.index_endpoint}")
    def initialize_collection(self, recreate: bool = False) -> None:
        """Синхронная инициализация коллекции на VM с правильным event loop management."""
        return run_async_safe(
            self._async_initialize_collection(recreate=recreate),
            timeout=300  # HOTFIX: 5 минут (было 60s)
        )

    def index_documents(self, points: List[Dict], progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
        """Синхронная индексация документов с правильным event loop management."""
        return run_async_safe(
            self._async_index_documents(points, progress_cb=progress_cb),
            timeout=1800  # HOTFIX: 30 минут (было 300s) - индексация может быть очень долгой при swap
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
        sparse_vector: Optional[Dict[int, float]] = None
    ) -> List[Dict]:
        """Синхронный поиск в удалённом хранилище с правильным event loop management."""
        return run_async_safe(
            self._async_search(query_vector, top_k, filters, use_hybrid, sparse_vector),
            timeout=300  # HOTFIX: 5 минут (было 60s)
        )

    def search_by_text(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """Синхронный поиск по тексту с правильным event loop management."""
        return run_async_safe(
            self._async_search_by_text(query_text, top_k, filters, use_hybrid),
            timeout=300  # HOTFIX: 5 минут (было 60s)
        )

    def health_check(self) -> Dict[str, Any]:
        """Синхронный health-check удалённого сервиса (унифицированный формат)."""
        return run_async_safe(
            self._async_health_check(),
            timeout=60  # HOTFIX: 1 минута (было 30s)
        )

    check_health = health_check

    def heartbeat(self) -> bool:
        """
        Быстрый синхронный ping сервиса (таймаут ≤ 2s). Возвращает True/False.
        Fail-fast: любые исключения -> False.
        """
        return run_async_safe(
            self._async_heartbeat(),
            timeout=3  # немного больше локального бюджета, чтобы корректно собрать результат
        )

    async def _async_heartbeat(self) -> bool:
        """
        Асинхронный быстрый ping /health с коротким таймаутом (≤ 2s).
        """
        try:
            # Рассчитываем бюджет таймаута: жесткий лимит 2.0s с учетом профиля
            total = 2.0
            tp = getattr(self, "timeout_profiles", None)
            if tp is not None:
                try:
                    total = min(2.0, float(tp.health_total_sec))
                except Exception:
                    total = 2.0
            timeout_ctx = ClientTimeout(total=total, sock_read=total)
            result = await self.transport_client.get_json(self.health_endpoint, timeout=timeout_ctx)
            status = str(result.get("status", "")).lower()
            return status in ("connected", "healthy", "ok")
        except Exception:
            return False
    def get_collection_info(self) -> Dict[str, Any]:
        """Синхронное получение сведений о коллекции с правильным event loop management."""
        return run_async_safe(
            self._async_get_collection_info(),
            timeout=60  # HOTFIX: 1 минута (было 30s)
        )

    def close_sync(self) -> None:
        """Синхронно закрывает соединение с правильным event loop management."""
        return run_async_safe(
            self._async_close(),
            timeout=30  # HOTFIX: 30 секунд (было 10s)
        )

    
    async def _async_initialize_collection(self, recreate: bool = False) -> None:
        """
        Инициализирует коллекцию через удалённый сервис.
        
        Args:
            recreate: Пересоздать коллекцию если она уже существует
        """
        try:
            health_info = await self._async_health_check()
            
            # Проверяем успешный статус (поддерживаем разные варианты для обратной совместимости)
            if health_info['status'] in ('connected', 'ok', 'healthy'):
                self._connected = True
                self._collection_exists = health_info.get('collection_status') == 'exists'
                
                if recreate and self._collection_exists:
                    _log(logger.info, "Пересоздание коллекции через удалённый сервис...")
                    # Пересоздание будет обработано на стороне VM сервиса
                    await self._recreate_collection()
                
                _log(logger.info, f"Коллекция {'существует' if self._collection_exists else 'будет создана'} на VM")
            else:
                raise ConnectionError(f"Не удалось подключиться к VM сервису: {health_info.get('error')}")
                
        except Exception as e:
            _log(logger.error, f"Ошибка инициализации коллекции через VM: {e}")
            raise
    
    async def _recreate_collection(self) -> None:
        """Пересоздаёт коллекцию через удалённый сервис"""
        try:
            recreate_endpoint = f"http://{self.service_host}:{self.service_port}/collection/recreate"
            # Таймаут на основе health_total_sec с учетом fallback
            if getattr(self, "timeout_profiles", None):
                health_total = min(float(self.health_timeout), float(self.timeout_profiles.health_total_sec))
            else:
                health_total = float(self.health_timeout)
            timeout_ctx = ClientTimeout(total=health_total, sock_read=health_total)
            try:
                result = await self.transport_client.post_json(
                    recreate_endpoint,
                    payload={},
                    timeout=timeout_ctx,
                    headers=None,
                )
                _log(logger.info, f"Коллекция пересоздана: {result}")
                self._collection_exists = True
            except aiohttp.ClientResponseError as cre:
                _log(logger.error, f"Ошибка пересоздания коллекции: HTTP {cre.status}: {cre.message}")
                return
        except Exception as e:
            _log(logger.error, f"Ошибка пересоздания коллекции через VM: {e}")
            raise
    
    async def _async_index_documents(self, points: List[Dict], progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> int:
        """
        Индексирует документы через удалённый сервис.
        
        Args:
            points: Список точек для индексации (формат: [{"id": ..., "text": ..., "metadata": ...}, ...])
            
        Returns:
            Количество успешно проиндексированных документов
        """
        if not points:
            return 0
        
        start_time = time.time()
        
        try:
            # 🔍 ДИАГНОСТИКА 1: Входные данные (safe-логирование без контента)
            diag_logger.info(f"📥 КЛИЕНТ: Получено {len(points)} points для индексации")
            if points:
                first_point = points[0]
                diag_logger.info(f"📥 КЛИЕНТ: Ключи первого point = {list(first_point.keys())}")
                
            # Подготовка данных для удалённого сервиса (Client Preflight: фильтрация и обогащение)
            doc_max_bytes_env = os.getenv("RAG_INDEX_DOC_MAX_BYTES", "262144")
            try:
                doc_max_bytes = int(doc_max_bytes_env)
            except Exception:
                doc_max_bytes = 262144  # 256 KiB fallback

            empty_count = 0
            too_large_count = 0
            filtered_docs: List[Dict[str, Any]] = []
            remap_logged = False

            for i, point in enumerate(points):
                # Извлекаем текст так же, как при старой сборке payload
                text = point.get("text", "") or point.get("payload", {}).get("content", "")
                if not isinstance(text, str):
                    text = str(text)

                if text.strip() == "":
                    empty_count += 1
                    self.stats['empty_text_detected_total'] = self.stats.get('empty_text_detected_total', 0) + 1
                    continue

                text_bytes_len = len(text.encode("utf-8"))
                if text_bytes_len > doc_max_bytes:
                    too_large_count += 1
                    self.stats['client_dropped_too_large_total'] = self.stats.get('client_dropped_too_large_total', 0) + 1
                    continue

                content_sha256 = _compute_content_sha256(text)
                embedding_version = self.embedding_version
                document_idempotency_key = f"{content_sha256}:{embedding_version}"

                # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: извлекаем метаданные из правильного источника
                # Сначала пытаемся извлечь из payload.metadata (primary), затем из point.metadata (fallback)
                raw_meta = {}
                try:
                    payload_meta = point.get("payload", {}).get("metadata", {})
                    if payload_meta:
                        raw_meta.update(payload_meta)
                except Exception:
                    pass
                
                try:
                    point_meta = point.get("metadata")
                    if point_meta:
                        raw_meta.update(point_meta)
                except Exception:
                    pass
                
                # Нормализуем метаданные используя существующую helper функцию
                meta = _normalize_metadata(raw_meta)
                
                orig_id = str(point.get("id") or point.get("payload", {}).get("id") or f"doc_{i}")
                new_id = _make_stable_point_id(meta, embedding_version, content_sha256, orig_id)
                meta["original_id"] = orig_id

                doc = {
                    "id": new_id,
                    "text": text,
                    "metadata": meta,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content_sha256": content_sha256,
                    "embedding_version": embedding_version,
                    "document_idempotency_key": document_idempotency_key,
                }
                filtered_docs.append(doc)

                # Диагностика: логируем ремап id один раз на батч
                if not remap_logged:
                    try:
                        diag_logger.info(f"id remap: first original_id={orig_id}, new_id={new_id}")
                    except Exception:
                        pass
                    remap_logged = True

            preflight_accepted_count = len(filtered_docs)
            diag_logger.info(
                f"🧪 КЛИЕНТ preflight: received={len(points)}, accepted={preflight_accepted_count}, "
                f"empty_dropped={empty_count}, too_large_dropped={too_large_count}, max_bytes={doc_max_bytes}"
            )

            if preflight_accepted_count == 0:
                # Нечего индексировать после фильтрации
                return 0

            payload = {
                "documents": filtered_docs,
                "batch_size": min(512, preflight_accepted_count),  # Батчевая обработка на сервере
                "recreate": False
            }
            
            # 🔍 ДИАГНОСТИКА 2: Подготовленный payload (safe-логирование без контента)
            if payload["documents"]:
                first_doc = payload["documents"][0]
                diag_logger.info(f"📤 КЛИЕНТ: Ключи первого document после подготовки = {list(first_doc.keys())}")
            
            # HTTP запрос на индексацию с прогресс‑колбэком
            total_docs = len(payload["documents"])
            result_info = await self._make_index_request_with_retry(payload)
            accepted_count = int(result_info.get("accepted", 0))
            rejected_count = int(result_info.get("rejected", 0))

            # Single-shot progress update (flat flow; no recursion/redelivery)
            if progress_cb is not None:
                try:
                    processed = min(total_docs, accepted_count + rejected_count)
                    percent = 100.0 if total_docs == 0 else min(100.0, 100.0 * processed / max(1, total_docs))
                    progress_cb({
                        "phase": "index",
                        "accepted": accepted_count,
                        "rejected": rejected_count,
                        "dropped": 0,
                        "total": total_docs,
                        "percent": percent,
                        "eta_sec": 0.0,
                        "depth": 0
                    })
                except Exception:
                    pass
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_indexed'] += accepted_count
            self.stats['total_index_time'] += elapsed_time
            
            _log(logger.info,
                f"Индексация через VM завершена: {accepted_count}/{len(points)} документов "
                f"за {elapsed_time:.3f}s ({accepted_count/elapsed_time:.1f} док/с)"
            )
            
            return accepted_count
            
        except Exception as e:
            self.stats['error_count'] += 1
            _log(logger.error, f"Ошибка индексации документов через VM: {e}")
            raise
    
    async def _make_index_request_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отправляет один батч документов на /index с плоской логикой:
        - 200: парсит {accepted, rejected, rejected_reasons[], elapsed_ms}; завершает батч
        - 422: парсит IndexErrorResponse; НЕ ретраит; завершает батч с accepted=0
        - 503/сетевые: применяет RetryPolicy для /index; при исчерпании попыток — поднимает исключение
        """
        # Размер батча для бюджетирования таймаута и логов
        documents = payload.get("documents") or []
        try:
            batch_size = len(documents)
        except Exception:
            batch_size = 0

        # Таймаут-профили
        tp = getattr(self, "timeout_profiles", None)
        index_base = float(tp.index_base_sec) if tp else 60.0
        index_step = float(tp.index_per_batch_step_sec) if tp else 0.2

        # Fallback: старое значение index_timeout
        index_timeout_fallback = float(getattr(self, "index_timeout", index_base))

        total_timeout_sec = max(index_timeout_fallback, index_base + batch_size * index_step)
        timeout_ctx = ClientTimeout(total=total_timeout_sec, sock_read=total_timeout_sec)

        endpoint = self.index_endpoint
        batch_id = uuid.uuid4().hex

        # Диагностика запроса (sanitized)
        try:
            doc0 = documents[0] if documents else {}
            doc0_id = str(doc0.get("id"))
            text_len = len(str(doc0.get("text", "")))
            emb_ver = doc0.get("embedding_version") or (doc0.get("metadata") or {}).get("embedding_version") or self.embedding_version
            doc0_keys = list(doc0.keys())
        except Exception:
            doc0_id, text_len, emb_ver, doc0_keys = ("?", 0, self.embedding_version, [])
        try:
            diag_logger.debug(f"POST /index url={endpoint} batch_id={batch_id} size={batch_size}")
            sanitized_headers = {
                "Content-Type": "application/json",
                "X-API-Contract": self.api_contract,
                "X-Embedding-Version": self.embedding_version,
                "X-Batch-Id": batch_id,
            }
            diag_logger.debug(f"request headers (sanitized) {json.dumps(sanitized_headers, ensure_ascii=False)}")
            diag_logger.debug(f"doc0: keys={doc0_keys}, id={doc0_id}, text_len={text_len}, embedding_version={emb_ver}")
        except Exception:
            pass

        attempt_counter = {"n": 0}
        max_attempts = int(getattr(self.retry_policy_index, "config", RetryConfig()).max_attempts)

        non_retryable_reasons = {"empty_text", "sha256_mismatch", "invalid_vector", "duplicate"}

        def _extract_reasons(items: Any) -> List[str]:
            out: List[str] = []
            if not isinstance(items, (list, tuple)):
                return out
            for it in items:
                r = None
                if isinstance(it, str):
                    r = it
                elif isinstance(it, dict):
                    r = it.get("reason") or it.get("type") or it.get("error")
                if r is None:
                    r = "unknown"
                out.append(str(r))
            return out

        async def _single_attempt():
            attempt_counter["n"] += 1
            # Безопасное логирование без контента
            _log(logger.info, f"📤 Индексация: batch_size={batch_size}, timeout={total_timeout_sec:.1f}s, endpoint={endpoint}, batch_id={batch_id}, attempt={attempt_counter['n']}/{max_attempts}")
            try:
                result = await self.transport_client.post_json(
                    endpoint,
                    payload=payload,
                    timeout=timeout_ctx,
                    headers={"X-Batch-Id": batch_id},
                )
                # HTTP 200 JSON
                accepted = int(result.get("accepted", 0) or result.get("indexed_count", 0) or 0)
                rejected = int(result.get("rejected", 0) or 0)
                elapsed_ms = int(result.get("elapsed_ms", 0) or 0)
                reasons_raw = result.get("rejected_reasons") or result.get("rejected_details") or []
                reasons = _extract_reasons(reasons_raw)

                # Логи по контракту
                _log(logger.info, f"VM : {accepted}/{batch_size} {elapsed_ms/1000.0:.3f}s")
                try:
                    diag_logger.debug(f"parsed: accepted={accepted}, rejected={rejected}, elapsed_ms={elapsed_ms}, rejected_reasons_count={len(reasons)}")
                    # WARN про неизвестные причины
                    for r in reasons:
                        if r not in non_retryable_reasons:
                            _log(logger.warning, f"index: unknown rejected reason '{r}' (no retry)")
                            break
                except Exception:
                    pass

                return {
                    "status_code": 200,
                    "accepted": accepted,
                    "rejected": rejected,
                    "reasons": reasons,
                    "elapsed_ms": elapsed_ms,
                    "batch_size": batch_size,
                }

            except aiohttp.ClientResponseError as cre:
                status = int(getattr(cre, "status", 0) or 0)
                msg = cre.message or ""
                data: Dict[str, Any] = {}
                try:
                    start = msg.find("{")
                    json_text = msg[start:].strip() if start != -1 else msg
                    data = json.loads(json_text) if json_text else {}
                except Exception:
                    data = {}

                if status == 422:
                    error_obj = data.get("error", data)
                    details = error_obj.get("details") or []
                    details_count = len(details) if isinstance(details, list) else 0
                    error_type = str(error_obj.get("type") or "validation_error")
                    reasons = _extract_reasons(details)

                    _log(logger.info, f"422 validation_error: rejected={details_count}")
                    try:
                        first_n = 5
                        diag_logger.debug(f"rejected_reasons (first {first_n}): {reasons[:first_n]}")
                        diag_logger.debug(f"parsed: validation_error rejected={details_count}, details_count={details_count}")
                        # WARN про неизвестные причины
                        for r in reasons:
                            if r not in non_retryable_reasons:
                                _log(logger.warning, f"index: unknown rejected reason '{r}' (no retry)")
                                break
                    except Exception:
                        pass

                    # Не ретраим 422 — возвращаем как финальный результат
                    return {
                        "status_code": 422,
                        "accepted": 0,
                        "rejected": details_count,
                        "reasons": reasons,
                        "error_type": error_type,
                        "details_count": details_count,
                        "batch_size": batch_size,
                        "elapsed_ms": data.get("elapsed_ms"),
                    }

                if status == 503:
                    _log(logger.info, f"transient error: HTTP 503, retrying attempt {attempt_counter['n']}/{max_attempts}")
                    raise

                # Прочие HTTP ошибки — эскалируем (пусть RetryPolicy/внешний код решает)
                _log(logger.debug, f"HTTP error {status}: {msg[:256]}")
                raise

            except (asyncio.TimeoutError, aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, ConnectionError, OSError) as e:
                _log(logger.info, f"transient error: {type(e).__name__}: {e}, retrying attempt {attempt_counter['n']}/{max_attempts}")
                raise

        async def _attempt_with_cb():
            return await self.circuit_breaker.call(_single_attempt)

        result_info = await self.retry_policy_index.execute_with_retry(_attempt_with_cb)

        # В этом месте result_info либо 200, либо 422 (на 503 — исключение)
        return result_info
    
    async def _index_with_redelivery(
        self,
        docs: List[Dict],
        recreate: bool,
        depth: int = 0,
        max_depth: int = 8,
        total_docs: Optional[int] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        _acc: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        DEPRECATED: плоский режим без рекурсии/redelivery.
        Оставлено только для обратной совместимости вызовов.
        Выполняется одна отправка батча с управляемыми ретраями согласно RetryPolicy.
        """
        docs = list(docs or [])
        total_in_attempt = len(docs)
        if total_in_attempt == 0:
            return 0

        payload = {
            "documents": docs,
            "batch_size": min(512, total_in_attempt),
            "recreate": False,
        }

        result = await self._make_index_request_with_retry(payload)
        accepted = int(result.get("accepted", 0))
        rejected = int(result.get("rejected", 0))

        if progress_cb is not None:
            try:
                processed = min(total_in_attempt, accepted + rejected)
                total = total_docs or total_in_attempt
                percent = 100.0 if total == 0 else min(100.0, 100.0 * processed / max(1, total))
                progress_cb({
                    "phase": "index",
                    "accepted": accepted,
                    "rejected": rejected,
                    "dropped": 0,
                    "total": total,
                    "percent": percent,
                    "eta_sec": 0.0,
                    "depth": 0
                })
            except Exception:
                pass

        return accepted

    async def _async_search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
        sparse_vector: Optional[Dict[int, float]] = None
    ) -> List[Dict]:
        """
        Выполняет поиск через удалённый сервис с готовыми векторами.
        
        Args:
            query_vector: Dense вектор запроса (1024d для Jina v3)
            top_k: Количество результатов
            filters: Фильтры по метаданным
            use_hybrid: Использовать гибридный поиск
            sparse_vector: Sparse вектор (BM25/SPLADE)
            
        Returns:
            Список результатов поиска
        """
        start_time = time.time()
        
        try:
            # ✅ ИСПРАВЛЕНИЕ: Передаём готовые векторы (векторный протокол)
            # Конвертируем numpy array в list для JSON сериализации
            dense_vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
            
            payload = {
                "dense_vector": dense_vector_list,  # ✅ Передаём dense вектор
                "sparse_vector": sparse_vector,     # ✅ Передаём sparse вектор (опционально)
                "top_k": top_k,
                "use_hybrid": use_hybrid and sparse_vector is not None,
                "filters": filters or {},
            }
            
            _log(logger.debug, f"Отправка векторного поиска: dense_vector={len(dense_vector_list)}d, sparse={sparse_vector is not None}")
            
            # HTTP запрос на поиск
            results = await self._make_search_request_with_retry(payload)
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += elapsed_time
            
            _log(logger.debug, f"Поиск через VM завершён: {len(results)} результатов за {elapsed_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.stats['error_count'] += 1
            _log(logger.error, f"Ошибка поиска через VM: {e}")
            return []  # Возвращаем пустой результат при ошибке

    async def _async_search_by_text(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Выполняет поиск по тексту через удалённый сервис.
        
        Args:
            query_text: Текст запроса
            top_k: Количество результатов
            filters: Фильтры по метаданным
            use_hybrid: Использовать гибридный поиск
            
        Returns:
            Список результатов поиска
        """
        start_time = time.time()
        
        try:
            payload = {
                "query": query_text,
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "filters": filters or {},
                "task": "retrieval.query"
            }
            
            results = await self._make_search_request_with_retry(payload)
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += elapsed_time
            
            _log(logger.debug, f"Текстовый поиск через VM: '{query_text[:50]}...' -> {len(results)} результатов за {elapsed_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.stats['error_count'] += 1
            _log(logger.error, f"Ошибка текстового поиска через VM: {e}")
            return []
    
    async def _make_search_request_with_retry(self, payload: Dict[str, Any]) -> List[Dict]:
        """
        Выполняет запрос на поиск через RetryPolicy + CircuitBreaker с пер-эндпойнтовым таймаутом.
        total_timeout_sec = min(self.search_timeout, timeout_profiles.search_total_p95_sec)
        """
        tp = getattr(self, "timeout_profiles", None)
        if tp is not None and getattr(self, "search_timeout", None) is not None:
            total_timeout_sec = min(float(self.search_timeout), float(tp.search_total_p95_sec))
        elif tp is not None:
            total_timeout_sec = float(tp.search_total_p95_sec)
        else:
            total_timeout_sec = float(self.search_timeout)

        # Раздельные таймауты клиента: короткий connect и полный sock_read
        connect_timeout_sec = min(1.0, max(0.05, total_timeout_sec * 0.2))
        sock_read_timeout_sec = total_timeout_sec
        timeout_ctx = ClientTimeout(total=total_timeout_sec, connect=connect_timeout_sec, sock_read=sock_read_timeout_sec)
        endpoint = self.text_search_endpoint if 'query' in payload else self.search_endpoint

        async def _single_attempt():
            _log(logger.debug, f"🔎 Поиск: timeout={total_timeout_sec:.1f}s, endpoint={endpoint}")
            return await self.transport_client.post_json(
                endpoint,
                payload=payload,
                timeout=timeout_ctx,
                headers=None,
            )

        async def _attempt_with_cb():
            return await self.circuit_breaker.call(_single_attempt)

        result = await self.retry_policy_search.execute_with_retry(_attempt_with_cb)

        if "results" in result:
            return result["results"]

        raise ValueError(f"Неожиданный формат ответа поиска: {result.keys()}")
    
    async def _async_health_check(self) -> Dict[str, Any]:
        """
        Асинхронная проверка состояния удалённого векторного хранилища с диагностикой.
        """
        import asyncio
        import aiohttp
        
        health_info = {
            "status": "unknown",
            "components": {
                "vector_store": {
                    "service_host": self.service_host,
                    "service_port": self.service_port,
                    "search_endpoint": self.search_endpoint,
                    "index_endpoint": self.index_endpoint,
                    "collection_status": "unknown",
                }
            },
            "error": None,
            "diagnostic": None,
        }

        start_time = time.time()
        
        try:
            # Таймаут на основе health_total_sec с учетом fallback
            if getattr(self, "timeout_profiles", None):
                health_total = min(float(self.health_timeout), float(self.timeout_profiles.health_total_sec))
            else:
                health_total = float(self.health_timeout)
            timeout_ctx = ClientTimeout(total=health_total, sock_read=health_total)
            result = await self.transport_client.get_json(self.health_endpoint, timeout=timeout_ctx)
            response_time_ms = (time.time() - start_time) * 1000

            health_info["status"] = "connected"
            health_info["components"]["vector_store"]["collection_status"] = result.get("collection_status", "unknown")
            health_info["components"]["vector_store"]["qdrant_status"] = result.get("qdrant_status", "unknown")
            health_info["components"]["vector_store"]["vector_count"] = result.get("vector_count", 0)
            health_info["components"]["vector_store"]["response_time_ms"] = response_time_ms
            health_info["components"]["vector_store"]["http_status"] = 200
            self._connected = True
            self._collection_exists = health_info["components"]["vector_store"]["collection_status"] == "exists"
        
        except aiohttp.ClientResponseError as e:
            response_time_ms = (time.time() - start_time) * 1000
            health_info["status"] = "error"
            health_info["error"] = f"HTTP {e.status}: {e.message}"
            health_info["components"]["vector_store"]["http_status"] = e.status
            health_info["components"]["vector_store"]["response_time_ms"] = response_time_ms

            health_info["diagnostic"] = {
                "error_type": "http_error",
                "http_status": e.status,
                "recommendation": self._get_http_error_recommendation(e.status),
                "response_time_ms": response_time_ms
            }
            self._connected = False

        except aiohttp.ClientConnectorError as e:
            response_time_ms = (time.time() - start_time) * 1000
            health_info["status"] = "error"
            health_info["error"] = f"ClientConnectorError: {e}"
            
            try:
                diagnostics = await diagnose_vm_connection(self.service_host, self.service_port)
                health_info["diagnostic"] = {
                    "error_type": "connection_refused",
                    "vm_host": self.service_host,
                    "vm_port": self.service_port,
                    "response_time_ms": response_time_ms,
                    "host_reachable": diagnostics['host_reachable'],
                    "port_open": diagnostics['port_open'],
                    "http_responding": diagnostics['http_responding'],
                    "latency_ms": diagnostics.get('latency_ms'),
                    "recommendations": diagnostics['recommendations']
                }
            except Exception as diag_error:
                _log(logger.warning, f"Ошибка запуска vm_diagnostics: {diag_error}")
                health_info["diagnostic"] = {
                    "error_type": "connection_refused",
                    "vm_host": self.service_host,
                    "vm_port": self.service_port,
                    "recommendation": (
                        f"VM сервис недоступен на {self.service_host}:{self.service_port}. "
                        f"Проверьте: 1) VM запущена, 2) Firewall не блокирует порт {self.service_port}, "
                        f"3) Сетевое подключение"
                    ),
                    "troubleshooting_commands": [
                        f"curl http://{self.service_host}:{self.service_port}/health",
                        f"ping {self.service_host}",
                        "python vm_start.py start"
                    ],
                    "response_time_ms": response_time_ms
                }
            self._connected = False

        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            health_info["status"] = "error"
            health_info["error"] = f"TimeoutError: Request timeout after {response_time_ms:.0f}ms"
            health_info["diagnostic"] = {
                "error_type": "timeout",
                "timeout_ms": 30000,
                "elapsed_ms": response_time_ms,
                "recommendation": (
                    f"VM сервис не отвечает в срок (timeout: {response_time_ms:.0f}ms). "
                    f"Проверьте: 1) Загрузку VM сервера, 2) Сетевую задержку, 3) VM процессы"
                ),
                "troubleshooting_commands": [
                    f"curl -w '@curl-format.txt' http://{self.service_host}:{self.service_port}/health",
                    f"ssh user@{self.service_host} 'top -b -n 1'",
                    "Увеличьте timeout в конфигурации"
                ],
                "response_time_ms": response_time_ms
            }
            self._connected = False

        except ValueError as e:
            response_time_ms = (time.time() - start_time) * 1000
            health_info["status"] = "error"
            health_info["error"] = f"ValueError: {e}"
            health_info["diagnostic"] = {
                "error_type": "invalid_response",
                "recommendation": (
                    "VM сервис вернул некорректный JSON. "
                    "Проверьте: 1) Версию VM сервиса, 2) Логи VM, 3) Формат API"
                ),
                "troubleshooting_commands": [
                    f"curl -v http://{self.service_host}:{self.service_port}/health",
                    "Проверьте логи VM: journalctl -u rag-vm-service -n 50"
                ],
                "response_time_ms": response_time_ms
            }
            self._connected = False

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            health_info["status"] = "error"
            health_info["error"] = f"{type(e).__name__}: {e}"
            health_info["diagnostic"] = {
                "error_type": "unknown",
                "exception_type": type(e).__name__,
                "recommendation": f"Неизвестная ошибка: {type(e).__name__}. Проверьте логи системы.",
                "response_time_ms": response_time_ms
            }
            self._connected = False

        return health_info
    
    def _get_http_error_recommendation(self, status_code: int) -> str:
        """
        Возвращает рекомендацию по устранению HTTP ошибки.
        
        Args:
            status_code: HTTP статус код
            
        Returns:
            Строка с рекомендацией
        """
        recommendations = {
            500: "Internal Server Error на VM. Проверьте логи VM сервиса и Qdrant.",
            503: "Service Unavailable. Qdrant может быть недоступна или перегружена.",
            502: "Bad Gateway. Проблема с прокси или upstream сервисом.",
            504: "Gateway Timeout. VM сервис не отвечает вовремя.",
            404: "Not Found. Проверьте правильность health endpoint URL.",
            401: "Unauthorized. Проверьте настройки аутентификации.",
            403: "Forbidden. Недостаточно прав для доступа к ресурсу."
        }
        
        return recommendations.get(
            status_code,
            f"HTTP {status_code}. Проверьте документацию VM API и логи сервиса."
        )
    
    async def _async_get_collection_info(self) -> Dict[str, Any]:
        """
        Получает информацию о коллекции через удалённый сервис используя shared HTTP session.
        
        Returns:
            Информация о коллекции
        """
        try:
            info_endpoint = f"http://{self.service_host}:{self.service_port}/collection/info"
            # Таймаут на основе health_total_sec с учетом fallback
            if getattr(self, "timeout_profiles", None):
                health_total = min(float(self.health_timeout), float(self.timeout_profiles.health_total_sec))
            else:
                health_total = float(self.health_timeout)
            timeout_ctx = ClientTimeout(total=health_total, sock_read=health_total)
            result = await self.transport_client.get_json(info_endpoint, timeout=timeout_ctx)
            return result
        
        except aiohttp.ClientResponseError as e:
            return {
                'error': f"HTTP {e.status}: {e.message}",
                'status': 'error'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        stats = self.stats.copy()
        
        # Вычисляем средние показатели
        if stats['total_searches'] > 0:
            stats['avg_search_time'] = stats['total_search_time'] / stats['total_searches']
        else:
            stats['avg_search_time'] = 0.0
            
        if stats['total_indexed'] > 0 and stats['total_index_time'] > 0:
            stats['avg_index_rate'] = stats['total_indexed'] / stats['total_index_time']
        else:
            stats['avg_index_rate'] = 0.0
        
        stats.update({
            'connected': self._connected,
            'collection_exists': self._collection_exists,
            'service_url': f"http://{self.service_host}:{self.service_port}",
            'search_endpoint': self.search_endpoint,
            'index_endpoint': self.index_endpoint
        })
        
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        self.stats = {
            'total_searches': 0,
            'total_indexed': 0,
            'total_search_time': 0.0,
            'total_index_time': 0.0,
            'error_count': 0,
            'retry_count': 0,
            'empty_text_detected_total': 0,
            'client_dropped_too_large_total': 0,
        }
        _log(logger.info, "Статистика RemoteVMVectorStore сброшена")
    async def close(self) -> None:
        """Асинхронная совместимость для существующих вызовов."""
        await self._async_close()

    async def _async_close(self) -> None:
        """Закрывает соединения (для удалённого клиента не требуется)"""
        self._connected = False
        _log(logger.info, "RemoteVMVectorStore закрыт")


# Обратная совместимость: алиас для старого класса
QdrantVectorStore = RemoteVMVectorStore
