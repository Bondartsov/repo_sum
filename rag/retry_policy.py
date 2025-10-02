"""
Адаптивная retry стратегия для HTTP запросов к VM.

Этот модуль предоставляет переиспользуемую retry политику с учётом:
- Оставшегося времени (adaptive timeouts)
- Exponential backoff с ограничением max_delay
- Конфигурируемых retryable exceptions
- Полной типизации для type safety

Автор: AI Assistant
Дата: 1 октября 2025
"""

import asyncio
import time
import logging
from typing import Optional, Callable, TypeVar, Tuple
from dataclasses import dataclass, field
import aiohttp

logger = logging.getLogger(__name__)

# Импортируем CircuitBreakerOpenException для исключения из retry
try:
    from .circuit_breaker import CircuitBreakerOpenException
except ImportError:
    # Если circuit_breaker не доступен, создаём dummy класс
    class CircuitBreakerOpenException(Exception):
        pass

T = TypeVar('T')


@dataclass
class RetryConfig:
    """
    Конфигурация retry политики.
    
    Attributes:
        max_attempts: Максимальное количество попыток (включая первую)
        base_delay: Базовая задержка перед первым retry (секунды)
        max_delay: Максимальная задержка между попытками (секунды)
        exponential_base: База для exponential backoff (обычно 2.0)
        timeout_seconds: Общий таймаут для всех попыток (секунды)
        retryable_exceptions: Tuple исключений, при которых делать retry
    """
    max_attempts: int = 5  # HOTFIX: больше попыток (было 3)
    base_delay: float = 10.0  # HOTFIX: больше задержка (было 2.0s)
    max_delay: float = 120.0  # HOTFIX: до 2 минут между попытками (было 30s)
    exponential_base: float = 2.0
    timeout_seconds: float = 600.0  # HOTFIX: 10 минут общий таймаут (было 60s)
    retryable_exceptions: Tuple[type, ...] = (
        asyncio.TimeoutError,
        aiohttp.ClientError,
        ConnectionError,
        ValueError,
        KeyError,
        OSError,
    )
    # Исключения которые НЕ должны trigger retry (даже если в retryable_exceptions)
    non_retryable_exceptions: Tuple[type, ...] = field(default_factory=lambda: (
        CircuitBreakerOpenException,
    ))
    
    def __post_init__(self):
        """Валидация конфигурации"""
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts должно быть >= 1, получено: {self.max_attempts}")
        if self.base_delay < 0:
            raise ValueError(f"base_delay должна быть >= 0, получено: {self.base_delay}")
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) должна быть >= base_delay ({self.base_delay})"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds должна быть > 0, получено: {self.timeout_seconds}")
        if self.exponential_base < 1:
            raise ValueError(
                f"exponential_base должна быть >= 1, получено: {self.exponential_base}"
            )


class RetryPolicy:
    """
    Адаптивная retry политика с учётом оставшегося времени.
    
    Возможности:
    - Tracking оставшегося времени между попытками
    - Exponential backoff с ограничением
    - Adaptive timeout для каждой попытки
    - Детальное логирование retry событий
    - Type-safe через generics
    
    Пример использования:
    ```python
    config = RetryConfig(max_attempts=5, base_delay=10.0, timeout_seconds=600.0)  # HOTFIX значения
    policy = RetryPolicy(config)
    
    async def risky_operation():
        # ... potentially failing operation
        return result
    
    result = await policy.execute_with_retry(risky_operation)
    ```
    """
    
    def __init__(self, config: RetryConfig):
        """
        Инициализация retry политики.
        
        Args:
            config: Конфигурация retry поведения
        """
        self.config = config
        self._stats = {
            'total_executions': 0,
            'total_retries': 0,
            'successful_executions': 0,
            'failed_executions': 0
        }
    
    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Выполняет асинхронную функцию с retry логикой.
        
        Args:
            func: Асинхронная функция для выполнения
            *args: Позиционные аргументы для функции
            **kwargs: Именованные аргументы для функции
            
        Returns:
            Результат выполнения функции
            
        Raises:
            asyncio.TimeoutError: Если исчерпано общее время
            Exception: Последнее исключение, если все попытки неудачны
        """
        self._stats['total_executions'] += 1
        start_time = time.time()
        last_exception: Optional[Exception] = None
        
        for attempt in range(self.config.max_attempts):
            # Проверяем оставшееся время
            elapsed = time.time() - start_time
            remaining = self.config.timeout_seconds - elapsed
            
            if remaining <= 0:
                self._stats['failed_executions'] += 1
                logger.warning(
                    f"Retry timeout: исчерпано {elapsed:.1f}s из "
                    f"{self.config.timeout_seconds:.1f}s после {attempt} попыток"
                )
                raise asyncio.TimeoutError(
                    f"Retry timeout: {elapsed:.1f}s / {self.config.timeout_seconds:.1f}s "
                    f"после {attempt} попыток"
                )
            
            try:
                # Выполняем функцию с ограничением по оставшемуся времени
                logger.debug(
                    f"Попытка {attempt + 1}/{self.config.max_attempts}, "
                    f"remaining={remaining:.1f}s"
                )
                
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=remaining
                )
                
                # Успех!
                if attempt > 0:
                    logger.info(
                        f"Успех после {attempt + 1} попыток "
                        f"(затрачено {time.time() - start_time:.2f}s)"
                    )
                
                self._stats['successful_executions'] += 1
                if attempt > 0:
                    self._stats['total_retries'] += attempt
                
                return result
            
            except self.config.retryable_exceptions as e:
                # Проверяем, не является ли это non-retryable exception
                if isinstance(e, self.config.non_retryable_exceptions):
                    logger.debug(
                        f"Non-retryable exception: {type(e).__name__}: {e}"
                    )
                    self._stats['failed_executions'] += 1
                    raise
                
                last_exception = e
                elapsed = time.time() - start_time
                remaining = self.config.timeout_seconds - elapsed
                
                logger.debug(
                    f"Retryable exception на попытке {attempt + 1}: "
                    f"{type(e).__name__}: {e}"
                )
                
                if attempt < self.config.max_attempts - 1:
                    # Не последняя попытка - делаем backoff
                    delay = self._calculate_delay(attempt, remaining)
                    
                    # delay=0.0 валиден (мгновенный retry), sleep только если > 0
                    if delay > 0:
                        logger.debug(
                            f"Backoff: {delay:.2f}s перед попыткой {attempt + 2}"
                        )
                        await asyncio.sleep(delay)
                    # Продолжаем к следующей итерации независимо от delay
                else:
                    # Последняя попытка - пробрасываем ошибку
                    logger.warning(
                        f"Все {self.config.max_attempts} попытки исчерпаны: "
                        f"{type(e).__name__}: {e}"
                    )
                    self._stats['failed_executions'] += 1
                    self._stats['total_retries'] += attempt
                    raise
        
        # Не должно произойти, но для type safety
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry failed без exception (не должно произойти)")
    
    def _calculate_delay(self, attempt: int, remaining_time: float) -> float:
        """
        Вычисляет задержку для следующей попытки.
        
        Args:
            attempt: Номер текущей попытки (0-based)
            remaining_time: Оставшееся время в секундах
            
        Returns:
            Задержка в секундах, ограниченная max_delay и remaining_time/2
        """
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Ограничиваем max_delay
        delay = min(delay, self.config.max_delay)
        
        # Ограничиваем половиной оставшегося времени
        # (чтобы оставить время на саму операцию)
        delay = min(delay, remaining_time / 2)
        
        # Не меньше нуля
        delay = max(0, delay)
        
        return delay
    
    def get_stats(self) -> dict:
        """
        Возвращает статистику работы retry policy.
        
        Returns:
            Словарь со статистикой:
            - total_executions: Общее количество вызовов execute_with_retry
            - total_retries: Общее количество retry попыток
            - successful_executions: Количество успешных выполнений
            - failed_executions: Количество неудачных выполнений
            - success_rate: Процент успешных выполнений
            - avg_retries_per_execution: Среднее количество retry на выполнение
        """
        stats = self._stats.copy()
        
        if stats['total_executions'] > 0:
            stats['success_rate'] = (
                stats['successful_executions'] / stats['total_executions'] * 100
            )
            stats['avg_retries_per_execution'] = (
                stats['total_retries'] / stats['total_executions']
            )
        else:
            stats['success_rate'] = 0.0
            stats['avg_retries_per_execution'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self._stats = {
            'total_executions': 0,
            'total_retries': 0,
            'successful_executions': 0,
            'failed_executions': 0
        }
