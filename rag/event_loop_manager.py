# rag/event_loop_manager.py
from __future__ import annotations

import asyncio
import logging
import threading
import atexit
import time
from typing import Any, Awaitable, Optional, TypeVar

import aiohttp

__all__ = [
    "EventLoopManager",
    "run_async_safe",
    "get_shared_http_session",
]

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _safe_message(message: str) -> str:
    if not isinstance(message, str):
        return str(message)
    try:
        message.encode("ascii")
        return message
    except UnicodeEncodeError:
        return message.encode("ascii", "ignore").decode("ascii")


def _log(logger_method, message: str, *args, **kwargs):
    logger_method(_safe_message(message), *args, **kwargs)


# -------------------- small helpers --------------------

def _pulse_loop(loop: asyncio.AbstractEventLoop, timeout: float = 0.5) -> None:
    """
    Дать event loop один «тик» без создания корутин/тасок.
    Реализация: постим колбэк в цикл и ждём threading.Event.
    """
    ev = threading.Event()

    def _set():
        ev.set()

    try:
        loop.call_soon_threadsafe(_set)
        ev.wait(timeout=timeout)
    except Exception:
        # Не валим процесс на shutdown
        pass


# -------------------- HTTPSessionManager --------------------

class HTTPSessionManager:
    """
    Централизованная aiohttp-сессия с connection pooling.
    Управляется EventLoopManager'ом; без магии в __del__.
    """
    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=20,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )
                timeout = aiohttp.ClientTimeout(
                    total=60, connect=10, sock_read=30, sock_connect=5
                )
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    timeout=timeout,
                    headers={
                        "User-Agent": "repo-sum-rag-client/1.0",
                        "Connection": "keep-alive",
                    },
                )
                _log(logger.debug, "HTTP session создана с connection pooling")
            return self._session

    async def close(self) -> None:
        async with self._lock:
            try:
                if self._session and not self._session.closed:
                    await self._session.close()
                    _log(logger.debug, "HTTP session закрыта")
            finally:
                # TCPConnector.close() синхронный
                if self._connector and not self._connector.closed:
                    try:
                        self._connector.close()
                        _log(logger.debug, "HTTP connector закрыт")
                    except Exception as e:
                        _log(logger.debug, f"Ошибка закрытия connector: {e}")
                self._session = None
                self._connector = None


# -------------------- EventLoopManager --------------------

class EventLoopManager:
    """
    Потокобезопасный singleton, поднимающий отдельный поток с asyncio event loop.
    Гарантирует:
      - безопасный запуск корутин из sync-кода с таймаутом
      - корректную отмену задач при таймауте + дренаж (без создания sleep-корутин)
      - аккуратный shutdown без зависаний/RecursionError
      - общую aiohttp-сессию
    Публичный API:
      - get_instance()
      - get_stats()
      - run_async(coro, timeout)
    Внешняя обёртка: run_async_safe(coro, timeout)
    """

    _instance: Optional["EventLoopManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if EventLoopManager._instance is not None:
            raise RuntimeError("EventLoopManager is singleton. Use get_instance().")

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()

        self._session_manager: Optional[HTTPSessionManager] = None

        self._stats = {
            "started_at": time.time(),
            "submitted": 0,
            "completed": 0,
            "cancelled": 0,
            "timeouts": 0,
            "shutdowns": 0,
        }

        self._start_loop_thread()
        atexit.register(self._atexit_shutdown)

    # ---------- Singleton ----------

    @classmethod
    def get_instance(cls) -> "EventLoopManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = EventLoopManager()
        return cls._instance

    # ---------- Public API ----------

    def get_stats(self) -> dict:
        return dict(self._stats)

    def run_async(self, coro: Awaitable[Any], timeout: Optional[float] = None) -> Any:
        """
        Выполняет корутину внутри фонового event loop.
        Корректно обрабатывает таймаут: cancel + дренаж (без sleep-корутин).
        """
        loop = self._ensure_loop()
        self._stats["submitted"] += 1

        cfut = asyncio.run_coroutine_threadsafe(_wrap_coro(coro), loop)

        try:
            result = cfut.result(timeout=timeout)
            self._stats["completed"] += 1
            return result

        except asyncio.TimeoutError:
            self._stats["timeouts"] += 1
            self._cancel_and_drain(loop, cfut)
            raise

        except TimeoutError:
            self._stats["timeouts"] += 1
            self._cancel_and_drain(loop, cfut)
            raise

        except Exception:
            if not cfut.done():
                self._cancel_and_drain(loop, cfut)
            else:
                self._stats["completed"] += 1
            raise

    async def get_http_session(self) -> aiohttp.ClientSession:
        if self._session_manager is None:
            raise RuntimeError("HTTPSessionManager not initialized")
        return await self._session_manager.get_session()

    def shutdown(self, timeout: float = 2.0) -> None:
        """
        Быстрый и безопасный останов:
          - закрыть HTTP-сессию
          - отменить pending задачи (кроме текущей shutdown-задачи) и дождаться их завершения
          - остановить цикл и отпустить поток
        Повторные вызовы безопасны. Жёстко ограничиваем время ожидания.
        """
        if self._stop_event.is_set():
            return

        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return

        self._stop_event.set()
        self._stats["shutdowns"] += 1

        # 1) Закрываем HTTP-сессию в цикле
        try:
            if self._session_manager is not None:
                fut = asyncio.run_coroutine_threadsafe(self._session_manager.close(), loop)
                fut.result(timeout=timeout)
        except Exception:
            pass

        # 2) Отменяем все задачи (кроме текущей) и даём им завершиться
        try:
            fut = asyncio.run_coroutine_threadsafe(_graceful_shutdown(loop), loop)
            fut.result(timeout=timeout)
        except Exception:
            pass

        # 3) Останавливаем цикл и даём «тик» без корутин
        try:
            loop.call_soon_threadsafe(loop.stop)
            _pulse_loop(loop, timeout=0.5)
        except Exception:
            pass

        # 4) Ждём поток недолго и отпускаем (поток — daemon)
        if thread.is_alive():
            thread.join(timeout=timeout)

        # ссылки освободим, остальное — в worker.finally
        self._loop = None
        self._thread = None

    # ---------- Internals ----------

    def _start_loop_thread(self) -> None:
        def _loop_worker():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready_event.set()

            try:
                self._loop.run_forever()
            finally:
                # Финальная очистка всех оставшихся задач
                try:
                    current = asyncio.current_task()
                    pending = [t for t in asyncio.all_tasks()
                               if t is not current and not t.done()]
                    if pending:
                        for t in pending:
                            t.cancel()
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    # финальный тик (безопасно — мы внутри потока цикла)
                    self._loop.run_until_complete(asyncio.sleep(0))
                except Exception:
                    pass
                finally:
                    try:
                        self._loop.close()
                    except Exception:
                        pass

        # daemon=True, чтобы никогда не удерживать процесс при редких зависаниях
        self._thread = threading.Thread(
            target=_loop_worker, name="EventLoopManagerThread", daemon=True
        )
        self._thread.start()

        # Дожидаемся готовности цикла
        self._ready_event.wait(timeout=3.0)

        # Инициализируем SessionManager в самом цикле
        init_fut = asyncio.run_coroutine_threadsafe(self._init_session_manager(), self._loop)  # type: ignore[arg-type]
        init_fut.result(timeout=3.0)
        _log(logger.info, "EventLoopManager: background loop started")

    async def _init_session_manager(self) -> None:
        self._session_manager = HTTPSessionManager()
        _log(logger.debug, "HTTPSessionManager initialized")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or (self._thread and not self._thread.is_alive()):
            self._ready_event.clear()
            self._stop_event.clear()
            self._start_loop_thread()
        return self._loop  # type: ignore[return-value]

    def _cancel_and_drain(self, loop: asyncio.AbstractEventLoop, cfut) -> None:
        """
        Отмена задачи при таймауте + короткий дренаж БЕЗ создания корутин.
        """
        try:
            cfut.cancel()
        except Exception:
            pass
        # Два коротких «пульса» — достаточно для Windows/3.13
        _pulse_loop(loop, timeout=0.3)
        _pulse_loop(loop, timeout=0.3)
        self._stats["cancelled"] += 1

    def _atexit_shutdown(self) -> None:
        # Быстрый, не блокирующий выход на случай непредвидённых состояний
        try:
            self.shutdown(timeout=1.5)
        except Exception:
            pass


# -------------------- helpers --------------------

async def _wrap_coro(coro: Awaitable[Any]) -> Any:
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except Exception:
        raise


async def _graceful_shutdown(loop: asyncio.AbstractEventLoop) -> None:
    """
    Корректная отмена «чужих» задач внутри event loop.
    Главное: НЕ отменяем текущую shutdown-задачу.
    """
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    if pending:
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    # Никаких sleep здесь снаружи — «тик» делает вызывающая сторона (через _pulse_loop).


# -------------------- public convenience API --------------------

def run_async_safe(coro: Awaitable[Any], timeout: Optional[float] = None) -> Any:
    manager = EventLoopManager.get_instance()
    return manager.run_async(coro, timeout)


async def get_shared_http_session() -> aiohttp.ClientSession:
    manager = EventLoopManager.get_instance()
    return await manager.get_http_session()
