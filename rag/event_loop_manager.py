"""
Единый Event Loop Manager для правильной работы async/sync кода.

Решает проблемы:
- Множественные event loops через asyncio.run()
- TCP TIME_WAIT состояния соединений
- ConnectionRefusedError при массовых вызовах
"""

import asyncio
import logging
import threading
import atexit
from typing import Coroutine, TypeVar, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar('T')


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


class HTTPSessionManager:
    """
    Централизованное управление HTTP сессиями с connection pooling.
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()
        
    async def get_session(self) -> aiohttp.ClientSession:
        """
        Получает переиспользуемую HTTP сессию с connection pooling.
        
        Returns:
            Настроенная aiohttp.ClientSession
        """
        async with self._lock:
            if self._session is None or self._session.closed:
                # Создаем connector с оптимизированными настройками
                self._connector = aiohttp.TCPConnector(
                    limit=100,              # Общий лимит соединений
                    limit_per_host=20,      # Лимит на хост
                    keepalive_timeout=30,   # Keep-alive timeout
                    enable_cleanup_closed=True,  # Автоочистка закрытых соединений
                    ttl_dns_cache=300,      # DNS cache TTL
                    use_dns_cache=True,     # Включить DNS cache
                )
                
                # Настройки таймаутов
                timeout = aiohttp.ClientTimeout(
                    total=60,       # Общий таймаут запроса
                    connect=10,     # Таймаут подключения 
                    sock_read=30,   # Таймаут чтения
                    sock_connect=5  # Таймаут socket соединения
                )
                
                # Создаем сессию
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    timeout=timeout,
                    headers={
                        'User-Agent': 'repo-sum-rag-client/1.0',
                        'Connection': 'keep-alive'
                    }
                )
                
                _log(logger.debug, "HTTP session создана с connection pooling")
        
        return self._session
    
    async def close(self) -> None:
        """Закрывает HTTP сессию и освобождает ресурсы."""
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                _log(logger.debug, "HTTP session закрыта")
            
            if self._connector and not self._connector.closed:
                await self._connector.close()
                _log(logger.debug, "HTTP connector закрыт")
                
            self._session = None
            self._connector = None
    
    def __del__(self):
        """Cleanup при удалении объекта."""
        if self._session and not self._session.closed:
            try:
                # Попытка graceful закрытия
                asyncio.create_task(self.close())
            except Exception:
                pass  # Игнорируем ошибки в деструкторе


class EventLoopManager:
    """
    Singleton для управления единым event loop в приложении.
    
    Решает проблемы:
    - asyncio.run() создает новый event loop каждый раз
    - Множественные event loops вызывают TCP проблемы
    - Неправильное управление ресурсами
    """
    
    _instance: Optional['EventLoopManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        if EventLoopManager._instance is not None:
            raise RuntimeError("EventLoopManager is singleton. Use get_instance().")
            
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._session_manager: Optional[HTTPSessionManager] = None
        self._running = False
        
        # Регистрируем cleanup при выходе
        atexit.register(self._cleanup)
        
    @classmethod
    def get_instance(cls) -> 'EventLoopManager':
        """Получает singleton instance EventLoopManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _ensure_loop(self) -> None:
        """Гарантирует что event loop запущен и готов к работе."""
        if self._loop is None or self._loop.is_closed():
            self._start_background_loop()
    
    def _start_background_loop(self) -> None:
        """Запускает event loop в background thread."""
        if self._running:
            return
            
        # Создаем новый event loop
        self._loop = asyncio.new_event_loop()
        
        # Настраиваем thread pool executor
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="rag-async"
        )
        
        # Запускаем loop в отдельном потоке
        def run_loop():
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_forever()
            except Exception as e:
                _log(logger.error, f"Event loop error: {e}")
            finally:
                self._loop.close()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        self._running = True
        
        # Инициализируем HTTP session manager
        future = asyncio.run_coroutine_threadsafe(
            self._init_session_manager(), 
            self._loop
        )
        future.result(timeout=10)  # Ждем инициализации
        
        _log(logger.info, "EventLoopManager: background loop started")
    
    async def _init_session_manager(self) -> None:
        """Инициализирует HTTP session manager."""
        self._session_manager = HTTPSessionManager()
        _log(logger.debug, "HTTPSessionManager initialized")
    
    def run_async(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = 60) -> T:
        """
        Правильно выполняет coroutine из синхронного контекста.
        
        Args:
            coro: Корутина для выполнения
            timeout: Таймаут выполнения в секундах
            
        Returns:
            Результат выполнения корутины
            
        Raises:
            asyncio.TimeoutError: При превышении таймаута
            Exception: Любые ошибки из корутины
        """
        self._ensure_loop()
        
        if not self._loop or self._loop.is_closed():
            raise RuntimeError("Event loop is not running")
        
        try:
            # Выполняем корутину в background event loop
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(timeout=timeout)
            
        except Exception as e:
            _log(logger.error, f"Error running async operation: {e}")
            raise
    
    async def get_http_session(self) -> aiohttp.ClientSession:
        """
        Получает переиспользуемую HTTP сессию.
        
        Returns:
            Настроенная aiohttp.ClientSession с connection pooling
        """
        if not self._session_manager:
            raise RuntimeError("HTTPSessionManager not initialized")
        
        return await self._session_manager.get_session()
    
    def _cleanup(self) -> None:
        """Очистка ресурсов при завершении."""
        if not self._running:
            return
            
        try:
            if self._session_manager and self._loop and not self._loop.is_closed():
                # Закрываем HTTP session
                future = asyncio.run_coroutine_threadsafe(
                    self._session_manager.close(),
                    self._loop
                )
                future.result(timeout=5)
            
            if self._executor:
                try:
                    self._executor.shutdown(wait=True)
                except Exception as e:
                    _log(logger.error, f"Ошибка завершения thread pool: {e}")
            
            if self._loop and not self._loop.is_closed():
                # Отменяем все pending tasks перед остановкой loop
                try:
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    # Даём время на завершение отмены
                    if pending:
                        import time
                        time.sleep(0.1)
                except Exception as e:
                    _log(logger.error, f"Ошибка отмены tasks: {e}")
                
                self._loop.call_soon_threadsafe(self._loop.stop)
                
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
                
            self._running = False
            _log(logger.info, "EventLoopManager cleanup completed")
            
        except Exception as e:
            _log(logger.error, f"Error during cleanup: {e}")
    
    def get_stats(self) -> dict:
        """Возвращает статистику работы event loop manager."""
        return {
            'running': self._running,
            'loop_closed': self._loop.is_closed() if self._loop else True,
            'thread_alive': self._thread.is_alive() if self._thread else False,
            'executor_shutdown': self._executor._shutdown if self._executor else True,
            'session_manager_active': self._session_manager is not None
        }


# Глобальная функция для удобного доступа
def run_async_safe(coro: Coroutine[Any, Any, T], timeout: Optional[float] = 60) -> T:
    """
    Безопасно выполняет async код из sync контекста.
    
    Args:
        coro: Корутина для выполнения
        timeout: Таймаут в секундах
        
    Returns:
        Результат выполнения корутины
        
    Example:
        result = run_async_safe(some_async_function())
    """
    manager = EventLoopManager.get_instance()
    return manager.run_async(coro, timeout)


async def get_shared_http_session() -> aiohttp.ClientSession:
    """
    Получает переиспользуемую HTTP сессию для всего приложения.
    
    Returns:
        Настроенная aiohttp.ClientSession с connection pooling
    """
    manager = EventLoopManager.get_instance()
    return await manager.get_http_session()
