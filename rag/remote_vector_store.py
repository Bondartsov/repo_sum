"""
HTTP клиент для удалённого векторного хранилища через RAG-as-a-Service на VM.

Заменяет прямое подключение к Qdrant на HTTP запросы к FastAPI сервису на VM,
где работает Qdrant с 1024d векторами и гибридным поиском.
"""

import os
import logging
import time
from typing import List, Dict, Optional, Any
from config import RemoteServiceConfig
import numpy as np
from datetime import datetime, timezone
from .event_loop_manager import run_async_safe, get_shared_http_session
from .vm_diagnostics import diagnose_vm_connection

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


class RemoteVMVectorStore:
    """
    HTTP клиент для удалённого векторного хранилища на VM.
    
    Возможности:
    - HTTP запросы к Qdrant через FastAPI сервис на VM
    - Индексация документов через HTTP
    - Гибридный поиск (dense + sparse) через HTTP
    - Health check удалённого сервиса
    """
    
    def __init__(self, vector_store_config=None, remote_service_config: Optional[RemoteServiceConfig] = None):
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
        self.service_host = host
        self.service_port = port

        self.timeout_seconds = int(os.getenv("RAG_TIMEOUT_SECONDS", str(self.remote_config.timeout_seconds)))
        self.max_retries = int(os.getenv("RAG_MAX_RETRIES", str(self.remote_config.max_retries)))
        self.retry_delay = float(os.getenv("RAG_RETRY_DELAY", str(self.remote_config.retry_delay)))

        # Статистика
        self.stats = {
            'total_searches': 0,
            'total_indexed': 0,
            'total_search_time': 0.0,
            'total_index_time': 0.0,
            'error_count': 0,
            'retry_count': 0
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

    def index_documents(self, points: List[Dict]) -> int:
        """Синхронная индексация документов с правильным event loop management."""
        return run_async_safe(
            self._async_index_documents(points),
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
            
            # Используем shared HTTP session
            session = await get_shared_http_session()
            
            async with session.post(recreate_endpoint) as response:
                if response.status == 200:
                    result = await response.json()
                    _log(logger.info, f"Коллекция пересоздана: {result}")
                    self._collection_exists = True
                else:
                    error_text = await response.text()
                    _log(logger.error, f"Ошибка пересоздания коллекции: HTTP {response.status}: {error_text}")
                        
        except Exception as e:
            _log(logger.error, f"Ошибка пересоздания коллекции через VM: {e}")
            raise
    
    async def _async_index_documents(self, points: List[Dict]) -> int:
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
            # 🔍 ДИАГНОСТИКА 1: Входные данные
            _log(logger.info, f"📥 КЛИЕНТ: Получено {len(points)} points для индексации")
            if points:
                first_point = points[0]
                _log(logger.info, f"📥 КЛИЕНТ: Первый point = {first_point}")
                _log(logger.info, f"📥 КЛИЕНТ: Ключи первого point = {list(first_point.keys())}")
                _log(logger.info, f"📥 КЛИЕНТ: point['text'] = '{first_point.get('text', 'KEY_NOT_FOUND')[:100]}'")
                
            # Подготовка данных для удалённого сервиса
            payload = {
                "documents": [
                    {
                        "id": str(point.get("id", f"doc_{i}")),
                        # ✅ ИСПРАВЛЕНИЕ: Извлекаем текст из правильного места
                        # Сначала пробуем point['text'], если нет - берём point['payload']['content']
                        "text": point.get("text", "") or point.get("payload", {}).get("content", ""),
                        "metadata": point.get("metadata", {}),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    for i, point in enumerate(points)
                ],
                "batch_size": min(512, len(points)),  # Батчевая обработка на сервере
                "recreate": False
            }
            
            # 🔍 ДИАГНОСТИКА 2: Подготовленный payload
            if payload["documents"]:
                first_doc = payload["documents"][0]
                _log(logger.info, f"📤 КЛИЕНТ: Первый document после подготовки = {first_doc}")
                _log(logger.info, f"📤 КЛИЕНТ: document['text'] = '{first_doc.get('text', 'EMPTY')[:100]}'")
            
            # HTTP запрос на индексацию
            indexed_count = await self._make_index_request_with_retry(payload)
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_indexed'] += indexed_count
            self.stats['total_index_time'] += elapsed_time
            
            _log(logger.info, 
                f"Индексация через VM завершена: {indexed_count}/{len(points)} документов "
                f"за {elapsed_time:.3f}s ({indexed_count/elapsed_time:.1f} док/с)"
            )
            
            return indexed_count
            
        except Exception as e:
            self.stats['error_count'] += 1
            _log(logger.error, f"Ошибка индексации документов через VM: {e}")
            raise
    
    async def _make_index_request_with_retry(self, payload: Dict[str, Any]) -> int:
        """
        Выполняет запрос на индексацию с retry логикой используя shared HTTP session.
        
        Args:
            payload: Данные для индексации
            
        Returns:
            Количество проиндексированных документов
        """
        import asyncio
        
        for attempt in range(self.max_retries):
            try:
                # Используем shared HTTP session с connection pooling
                session = await get_shared_http_session()
                
                # 🔍 ЛОГ 1: Перед отправкой
                _log(logger.info, f"📤 Отправка на VM: {len(payload.get('documents', []))} документов, endpoint={self.index_endpoint}")
                
                async with session.post(
                    self.index_endpoint,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    
                    # 🔍 ЛОГ 2: HTTP статус
                    _log(logger.info, f"📥 Ответ VM: HTTP {response.status}, headers={dict(response.headers)}")
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # 🔍 ЛОГ 3: Полный ответ
                        _log(logger.info, f"📊 JSON ответ VM: {result}")
                        
                        # Ожидаем формат: {"indexed_count": 123, "status": "success"}
                        if "indexed_count" in result:
                            indexed_count = result["indexed_count"]
                            # 🔍 ЛОГ 4: Извлеченное значение
                            _log(logger.info, f"✅ Extracted indexed_count = {indexed_count}, type = {type(indexed_count).__name__}")
                            return indexed_count
                        else:
                            # 🔍 ЛОГ 5: Неожиданный формат
                            _log(logger.error, f"❌ Ключ 'indexed_count' отсутствует! Доступные ключи: {list(result.keys())}")
                            raise ValueError(f"Неожиданный формат ответа индексации: {result.keys()}")
                    
                    else:
                        error_text = await response.text()
                        _log(logger.error, f"❌ HTTP {response.status}: {error_text}")
                        raise RuntimeError(f"HTTP {response.status}: {error_text}")
            
            except Exception as e:
                _log(logger.warning, f"Ошибка индексации (попытка {attempt + 1}): {e}")
                self.stats['retry_count'] += 1
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    raise  # Последняя попытка - пробрасываем ошибку
    
    async def _async_search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
        sparse_vector: Optional[Dict[int, float]] = None
    ) -> List[Dict]:
        """
        Выполняет поиск через удалённый сервис.
        
        Args:
            query_vector: Вектор запроса (не используется напрямую - текст будет векторизован на VM)
            top_k: Количество результатов
            filters: Фильтры по метаданным
            use_hybrid: Использовать гибридный поиск
            sparse_vector: Разреженный вектор (не используется - обрабатывается на VM)
            
        Returns:
            Список результатов поиска
        """
        start_time = time.time()
        
        try:
            # Подготовка запроса для удалённого поиска
            # Поскольку у нас нет текста запроса здесь, используем заглушку
            # В реальном использовании query должен содержать текст
            payload = {
                "query": "search_query_placeholder",  # Будет заменен в search_service
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "filters": filters or {},
                "task": "retrieval.query"
            }
            
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
        Выполняет запрос на поиск с retry логикой используя shared HTTP session.
        
        Args:
            payload: Данные запроса
            
        Returns:
            Список результатов поиска
        """
        import asyncio
        
        for attempt in range(self.max_retries):
            try:
                # Используем shared HTTP session с connection pooling
                session = await get_shared_http_session()
                
                async with session.post(
                    self.search_endpoint,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Ожидаем формат: {"results": [...], "query_time": 0.123}
                        if "results" in result:
                            return result["results"]
                        else:
                            raise ValueError(f"Неожиданный формат ответа поиска: {result.keys()}")
                    
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"HTTP {response.status}: {error_text}")
            
            except Exception as e:
                _log(logger.warning, f"Ошибка поиска (попытка {attempt + 1}): {e}")
                self.stats['retry_count'] += 1
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise  # Последняя попытка - пробрасываем ошибку
    
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
            "diagnostic": None,  # 1.3.1: Добавляем диагностическую информацию
        }

        start_time = time.time()
        
        try:
            health_endpoint = f"http://{self.service_host}:{self.service_port}{self.remote_config.health_endpoint}"
            session = await get_shared_http_session()
            
            async with session.get(health_endpoint) as response:
                response_time_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    health_info["status"] = "connected"  # Единый стандарт: "connected" для успешного подключения
                    health_info["components"]["vector_store"]["collection_status"] = result.get("collection_status", "unknown")
                    health_info["components"]["vector_store"]["qdrant_status"] = result.get("qdrant_status", "unknown")
                    health_info["components"]["vector_store"]["vector_count"] = result.get("vector_count", 0)
                    health_info["components"]["vector_store"]["response_time_ms"] = response_time_ms
                    health_info["components"]["vector_store"]["http_status"] = response.status
                    self._connected = True
                    self._collection_exists = health_info["components"]["vector_store"]["collection_status"] == "exists"
                else:
                    error_text = await response.text()
                    health_info["status"] = "error"
                    health_info["error"] = f"HTTP {response.status}: {error_text}"
                    health_info["components"]["vector_store"]["http_status"] = response.status
                    health_info["components"]["vector_store"]["response_time_ms"] = response_time_ms
                    
                    # Диагностическая информация для HTTP ошибок
                    health_info["diagnostic"] = {
                        "error_type": "http_error",
                        "http_status": response.status,
                        "recommendation": self._get_http_error_recommendation(response.status),
                        "response_time_ms": response_time_ms
                    }
                    self._connected = False
                    
        except aiohttp.ClientConnectorError as e:
            response_time_ms = (time.time() - start_time) * 1000
            health_info["status"] = "error"
            health_info["error"] = f"ClientConnectorError: {e}"
            
            # 2.3.2: Используем vm_diagnostics для комплексной диагностики
            try:
                diagnostics = await diagnose_vm_connection(self.service_host, self.service_port)
                
                health_info["diagnostic"] = {
                    "error_type": "connection_refused",
                    "vm_host": self.service_host,
                    "vm_port": self.service_port,
                    "response_time_ms": response_time_ms,
                    # Добавляем детальные результаты диагностики
                    "host_reachable": diagnostics['host_reachable'],
                    "port_open": diagnostics['port_open'],
                    "http_responding": diagnostics['http_responding'],
                    "latency_ms": diagnostics.get('latency_ms'),
                    "recommendations": diagnostics['recommendations']
                }
            except Exception as diag_error:
                # Fallback на базовую диагностику если vm_diagnostics не сработала
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
                        f"python vm_start.py start"
                    ],
                    "response_time_ms": response_time_ms
                }
            
            self._connected = False
            
        except asyncio.TimeoutError as e:
            response_time_ms = (time.time() - start_time) * 1000
            health_info["status"] = "error"
            health_info["error"] = f"TimeoutError: Request timeout after {response_time_ms:.0f}ms"
            
            # Диагностическая информация для timeout
            health_info["diagnostic"] = {
                "error_type": "timeout",
                "timeout_ms": 30000,  # Default timeout
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
            
            # Диагностическая информация для invalid JSON
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
            
            # Общая диагностическая информация
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
            
            # Используем shared HTTP session
            session = await get_shared_http_session()
            
            async with session.get(info_endpoint) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        'error': f"HTTP {response.status}: {error_text}",
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
            'retry_count': 0
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
