"""
HTTP клиент для удалённых эмбеддингов через RAG-as-a-Service на VM.

Заменяет локальную загрузку моделей на HTTP запросы к FastAPI сервису на VM,
где работает Jina v3 с полными 1024d векторами.
"""

import os
import logging
import time
from typing import List, Optional, Dict, Any
from config import EmbeddingConfig, ParallelismConfig, RemoteServiceConfig
import numpy as np
import json
from .exceptions import EmbeddingException
from .event_loop_manager import run_async_safe, get_shared_http_session

logger = logging.getLogger(__name__)


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


class RemoteVMEmbedder:
    """
    HTTP клиент для получения эмбеддингов от Jina v3 сервиса на VM.
    
    Возможности:
    - HTTP запросы к FastAPI сервису на VM (10.61.11.54:8000)
    - Jina v3 dual task support (retrieval.query/passage)
    - Контроль целостности размерности 1024d
    - Батчевая обработка через HTTP
    - Retry логика с понятными сообщениями об ошибках
    """
    
    def __init__(self, embedding_config: Optional[EmbeddingConfig] = None, 
                 parallelism_config: Optional[ParallelismConfig] = None, 
                 remote_service_config: Optional[RemoteServiceConfig] = None):
        """
        Инициализация удалённого эмбеддера.
        
        Args:
            embedding_config: Конфигурация эмбеддингов (игнорируется, для совместимости)
            parallelism_config: Конфигурация параллелизма (игнорируется, для совместимости)
        """
        # Читаем конфигурацию из переменных окружения
        # Настройки удалённого сервиса
        self.remote_config = remote_service_config or RemoteServiceConfig()
        env_host = os.getenv("RAG_SERVICE_HOST")
        env_port = os.getenv("RAG_SERVICE_PORT")
        host = env_host or self.remote_config.host
        port = int(env_port) if env_port is not None else self.remote_config.port
        base_url = f"http://{host}:{port}"
        self.embeddings_endpoint = os.getenv("RAG_EMBEDDINGS_ENDPOINT", base_url + self.remote_config.embeddings_endpoint)
        self.service_host = host
        self.service_port = port

        # Параметры эмбеддера
        self.model_name = embedding_config.model_name if embedding_config else os.getenv("EMB_MODEL_ID", "jinaai/jina-embeddings-v3")
        self.provider_name = (
            embedding_config.provider if embedding_config and getattr(embedding_config, "provider", None) else os.getenv("EMBEDDING_PROVIDER", "remote-vm")
        )
        self.embedding_dim = getattr(embedding_config, "embedding_dim", int(os.getenv("EMB_DIM", "1024")))
        self.truncate_dim = self.embedding_dim

        # HTTP параметры 
        self.timeout_seconds = int(os.getenv("RAG_TIMEOUT_SECONDS", str(self.remote_config.timeout_seconds)))
        self.max_retries = int(os.getenv("RAG_MAX_RETRIES", str(self.remote_config.max_retries)))
        self.retry_delay = float(os.getenv("RAG_RETRY_DELAY", str(self.remote_config.retry_delay)))
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'total_texts': 0,
            'total_time': 0.0,
            'error_count': 0,
            'retry_count': 0,
            'avg_response_time': 0.0
        }
        
        self._is_warmed_up = False
        _log(logger.info, f"RemoteVMEmbedder инициализирован: {self.embeddings_endpoint}")
    

    def embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        deadline_ms: int = None
    ) -> np.ndarray:
        """Синхронная обёртка над запросом embeddings к VM с правильным event loop management."""
        if not texts:
            return np.array([])

        # 1.1.3: Используем config вместо hardcode
        if deadline_ms is None:
            deadline_ms = self.timeout_seconds * 1000  # Из config (по умолчанию 60s)
        
        base_timeout = deadline_ms / 1000.0
        
        # 1.1.1: Гармонизация timeout - рассчитываем total_timeout
        # Формула: base × retries + sum(delay × 2^i для exponential backoff)
        # Пример: 60s × 3 + (2s + 4s + 8s) = 180s + 14s = 194s
        backoff_total = sum(self.retry_delay * (2 ** i) for i in range(self.max_retries))
        total_timeout = (base_timeout * self.max_retries) + backoff_total
        
        _log(logger.debug, 
            f"Timeout конфигурация: base={base_timeout}s, retries={self.max_retries}, "
            f"backoff_total={backoff_total}s, total={total_timeout}s"
        )

        try:
            # Используем EventLoopManager для правильной работы с async кодом
            return run_async_safe(
                self._async_embed_texts(texts, task=task, deadline_ms=deadline_ms),
                timeout=total_timeout  # Гармонизированный timeout
            )
        except Exception as exc:
            self.stats['error_count'] += 1
            message = f"Удалённый сервис эмбеддингов недоступен: {exc}"
            _log(logger.error, message, exc_info=True)
            raise EmbeddingException(message, provider=self.provider_name, model_name=self.model_name)

    async def _async_embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        deadline_ms: int = 30000
    ) -> np.ndarray:
        """Выполняет фактический HTTP запрос к VM и обновляет статистику."""
        if not texts:
            return np.array([])

        start_time = time.time()

        try:
            payload = {
                "texts": texts,
                "task": task or "retrieval.passage",
                "truncate_dim": self.truncate_dim,
                "normalize": True
            }

            embeddings = await self._make_request_with_retry(payload, deadline_ms)

            embeddings_array = np.array(embeddings, dtype=np.float32)
            # Гарантируем корректную форму [N, D] и соответствие количеству текстов
            if embeddings_array.ndim == 0:
                raise ValueError("VM вернул скаляр вместо массива эмбеддингов")
            if embeddings_array.ndim == 1:
                if len(texts) == 1:
                    embeddings_array = embeddings_array.reshape(1, -1)
                else:
                    raise ValueError(
                        f"VM вернул 1D вектор при батче из {len(texts)} текстов: shape={embeddings_array.shape}"
                    )
            if embeddings_array.ndim != 2:
                raise ValueError(f"Некорректная форма эмбеддингов от VM: ndim={embeddings_array.ndim}")
            if embeddings_array.shape[0] != len(texts):
                raise ValueError(
                    f"Несовпадение размеров: embeddings={embeddings_array.shape[0]} vs texts={len(texts)}"
                )

            elapsed_time = time.time() - start_time
            self.stats['total_requests'] += 1
            self.stats['total_texts'] += len(texts)
            self.stats['total_time'] += elapsed_time
            self.stats['avg_response_time'] = self.stats['total_time'] / self.stats['total_requests']

            _log(logger.debug, 
                f"Получены embeddings с VM: {len(texts)} элементов, "
                f"shape={embeddings_array.shape}, time={elapsed_time:.3f}s"
            )

            return embeddings_array

        except Exception as e:
            self.stats['error_count'] += 1
            _log(logger.error, f"Ошибка получения embeddings с VM: {e}")
            raise EmbeddingException("Удалённый сервис эмбеддингов вернул ошибку", provider=self.provider_name, model_name=self.model_name, details=str(e))

    async def _make_request_with_retry(
        self, 
        payload: Dict[str, Any], 
        deadline_ms: int
    ) -> List[List[float]]:
        """
        Выполняет HTTP запрос с retry логикой используя shared HTTP session.
        
        Args:
            payload: Данные для отправки
            deadline_ms: Дедлайн в миллисекундах
            
        Returns:
            Список эмбеддингов
        """
        import asyncio
        import aiohttp
        from .exceptions import VMConnectionError, VMTimeoutError
        
        # 1.1.2: Tracking остатка времени
        start_time = time.time()
        total_timeout = deadline_ms / 1000.0
        base_timeout = deadline_ms / 1000.0
        
        for attempt in range(self.max_retries):
            try:
                # Проверяем оставшееся время перед каждой попыткой
                elapsed = time.time() - start_time
                remaining = total_timeout - elapsed
                
                if remaining <= 0:
                    raise VMTimeoutError(
                        message="Исчерпано время для retry попыток",
                        timeout_seconds=total_timeout,
                        elapsed_seconds=elapsed,
                        operation="embedding",
                        retry_attempt=attempt + 1
                    )
                
                # Адаптивный request timeout - используем оставшееся время
                request_timeout_seconds = min(base_timeout, remaining)
                
                _log(logger.debug, 
                    f"Попытка {attempt + 1}/{self.max_retries}: "
                    f"request_timeout={request_timeout_seconds:.1f}s, remaining={remaining:.1f}s"
                )
                
                # Используем shared HTTP session с connection pooling
                session = await get_shared_http_session()
                
                # Создаем новый timeout для данного запроса
                request_timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
                
                async with session.post(
                    self.embeddings_endpoint,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=request_timeout
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Ожидаем формат: {"embeddings": [[...], [...], ...]}
                        if "embeddings" in result:
                            return result["embeddings"]
                        else:
                            raise ValueError(f"Неожиданный формат ответа: {result.keys()}")
                    
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"HTTP {response.status}: {error_text}")
            
            # 1.2.2: Улучшенная обработка ошибок с использованием специфичных исключений
            except aiohttp.ClientConnectorError as e:
                _log(logger.warning, 
                    f"Ошибка подключения к VM (попытка {attempt + 1}/{self.max_retries}): {e}"
                )
                self.stats['retry_count'] += 1
                
                if attempt < self.max_retries - 1:
                    # Проверяем оставшееся время перед backoff
                    elapsed = time.time() - start_time
                    remaining = total_timeout - elapsed
                    delay = min(self.retry_delay * (2 ** attempt), remaining / 2)
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
                    else:
                        raise VMConnectionError(
                            message="VM сервис недоступен, время исчерпано",
                            vm_host=self.service_host,
                            vm_port=self.service_port,
                            error_details=str(e)
                        )
                else:
                    # Последняя попытка - выбрасываем специфичное исключение
                    raise VMConnectionError(
                        message="VM сервис недоступен после всех retry попыток",
                        vm_host=self.service_host,
                        vm_port=self.service_port,
                        error_details=str(e)
                    )
            
            except asyncio.TimeoutError as e:
                elapsed = time.time() - start_time
                _log(logger.warning, 
                    f"Timeout при запросе к VM (попытка {attempt + 1}/{self.max_retries}, "
                    f"elapsed={elapsed:.1f}s)"
                )
                self.stats['retry_count'] += 1
                
                if attempt < self.max_retries - 1:
                    # Проверяем оставшееся время перед backoff
                    remaining = total_timeout - elapsed
                    delay = min(self.retry_delay * (2 ** attempt), remaining / 2)
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
                    else:
                        raise VMTimeoutError(
                            message="VM сервис не отвечает, время исчерпано",
                            timeout_seconds=total_timeout,
                            elapsed_seconds=elapsed,
                            operation="embedding",
                            retry_attempt=attempt + 1
                        )
                else:
                    # Последняя попытка - выбрасываем специфичное исключение
                    raise VMTimeoutError(
                        message="VM сервис не отвечает после всех retry попыток",
                        timeout_seconds=total_timeout,
                        elapsed_seconds=elapsed,
                        operation="embedding",
                        retry_attempt=attempt + 1
                    )
                
            except Exception as e:
                _log(logger.warning, f"Ошибка HTTP запроса (попытка {attempt + 1}): {e}")
                self.stats['retry_count'] += 1
                
                if attempt < self.max_retries - 1:
                    elapsed = time.time() - start_time
                    remaining = total_timeout - elapsed
                    delay = min(self.retry_delay * (2 ** attempt), remaining / 2)
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    raise  # Последняя попытка - пробрасываем ошибку
    
    async def _async_health_check(self) -> Dict[str, Any]:
        """
        Асинхронная проверка состояния удалённого сервиса.
        """
        health_info = {
            "status": "unknown",
            "components": {
                "embedder": {
                    "service_url": self.embeddings_endpoint,
                    "provider": self.provider_name,
                    "model_name": self.model_name,
                    "embedding_dim": self.embedding_dim,
                    "truncate_dim": self.truncate_dim,
                }
            },
            "error": None,
        }

        try:
            test_payload = {
                "texts": ["test"],
                "task": "retrieval.query",
                "truncate_dim": self.truncate_dim,
                "normalize": True,
            }

            session = await get_shared_http_session()
            async with session.post(self.embeddings_endpoint, json=test_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "embeddings" in result and len(result["embeddings"]) > 0:
                        actual_dim = len(result["embeddings"][0])
                        health_info["status"] = "connected"  # Единый стандарт: "connected"
                        health_info["components"]["embedder"]["actual_embedding_dim"] = actual_dim
                        if actual_dim != self.truncate_dim:
                            health_info["components"]["embedder"]["warning"] = (
                                f"Размерность не соответствует ожидаемой: {actual_dim} vs {self.truncate_dim}"
                            )
                    else:
                        health_info["status"] = "error"
                        health_info["error"] = "Некорректный формат ответа"
                else:
                    error_text = await response.text()
                    health_info["status"] = "error"
                    health_info["error"] = f"HTTP {response.status}: {error_text}"
        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)

        return health_info

    def check_health(self) -> Dict[str, Any]:
        """
        Синхронная обёртка для health check.
        """
        return run_async_safe(self._async_health_check(), timeout=30)
    
    def warmup(self) -> None:
        """
        Прогрев удалённого сервиса с правильным event loop management.
        """
        if self._is_warmed_up:
            return
            
        _log(logger.info, "Прогрев удалённого VM сервиса...")
        
        try:
            # Используем EventLoopManager для правильной работы с async кодом
            run_async_safe(self._async_warmup(), timeout=30)
        except Exception as e:
            _log(logger.warning, f"Ошибка прогрева VM сервиса: {e}")
            # Не критично, продолжаем работу
            
    async def _async_warmup(self) -> None:
        """Асинхронный прогрев сервиса"""
        try:
            health_info = await self._async_health_check()
            
            if health_info['status'] in ('connected', 'ok', 'healthy'):
                self._is_warmed_up = True
                _log(logger.info, f"VM сервис готов: {health_info['components']['embedder'].get('actual_embedding_dim', 'N/A')}d векторы")
            else:
                _log(logger.warning, f"VM сервис не готов: {health_info.get('error', 'Unknown error')}")
                
        except Exception as e:
            _log(logger.error, f"Ошибка асинхронного прогрева: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        stats = self.stats.copy()
        stats.update({
            'service_url': self.embeddings_endpoint,
            'provider': self.provider_name,
            'model_name': self.model_name,
            'is_warmed_up': self._is_warmed_up,
            'embedding_dim': self.embedding_dim,
            'truncate_dim': self.truncate_dim
        })
        
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        self.stats = {
            'total_requests': 0,
            'total_texts': 0,
            'total_time': 0.0,
            'error_count': 0,
            'retry_count': 0,
            'avg_response_time': 0.0
        }
        _log(logger.info, "Статистика RemoteVMEmbedder сброшена")


# Обратная совместимость: алиас для старого класса
CPUEmbedder = RemoteVMEmbedder
