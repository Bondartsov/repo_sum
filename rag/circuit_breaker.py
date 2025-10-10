"""
Circuit Breaker pattern для защиты от постоянных падений VM.

Этот модуль реализует классический Circuit Breaker pattern для предотвращения
каскадных падений при недоступности удалённого сервиса.

Состояния Circuit Breaker:
- CLOSED: Нормальная работа, все запросы проходят
- OPEN: Сервис недоступен, запросы блокируются
- HALF_OPEN: Пробное восстановление, ограниченные запросы

Автор: AI Assistant
Дата: 1 октября 2025
"""

import time
import logging
from enum import Enum
from typing import Optional, Callable, TypeVar, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """
    Состояния Circuit Breaker.
    
    CLOSED: Нормальная работа - все запросы проходят через
    OPEN: Сервис недоступен - запросы блокируются, fail-fast
    HALF_OPEN: Пробное восстановление - ограниченное количество запросов
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """
    Конфигурация Circuit Breaker.
    
    Attributes:
        failure_threshold: Количество неудач для перехода в OPEN
        success_threshold: Количество успехов для перехода из HALF_OPEN в CLOSED
        timeout_seconds: Время до перехода из OPEN в HALF_OPEN (секунды)
        half_open_max_calls: Максимальное количество вызовов в HALF_OPEN состоянии
        excluded_exceptions: Tuple исключений, которые НЕ считаются failure
    """
    failure_threshold: int = 10  # HOTFIX: было 5
    success_threshold: int = 2
    timeout_seconds: float = 300.0  # HOTFIX: было 60.0 (5 минут)
    half_open_max_calls: int = 1
    excluded_exceptions: tuple = ()
    
    def __post_init__(self):
        """Валидация конфигурации"""
        if self.failure_threshold < 1:
            raise ValueError(
                f"failure_threshold должен быть >= 1, получено: {self.failure_threshold}"
            )
        if self.success_threshold < 1:
            raise ValueError(
                f"success_threshold должен быть >= 1, получено: {self.success_threshold}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds должен быть > 0, получено: {self.timeout_seconds}"
            )
        if self.half_open_max_calls < 1:
            raise ValueError(
                f"half_open_max_calls должен быть >= 1, получено: {self.half_open_max_calls}"
            )


class CircuitBreakerOpenException(Exception):
    """
    Исключение, выбрасываемое когда Circuit Breaker в состоянии OPEN.
    
    Это исключение сигнализирует клиенту, что сервис недоступен и
    запросы блокируются для предотвращения каскадных падений.
    """
    def __init__(self, message: str, time_until_retry: float):
        self.time_until_retry = time_until_retry
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit Breaker для защиты от каскадных падений удалённого сервиса.
    
    Возможности:
    - Автоматическое определение недоступности сервиса
    - Fail-fast при OPEN состоянии (не тратим время на запросы)
    - Автоматическое восстановление через HALF_OPEN
    - Детальная статистика и метрики
    - Thread-safe операции
    
    Пример использования:
    ```python
    config = CircuitBreakerConfig(
        failure_threshold=10,         # HOTFIX: было 5
        success_threshold=2,
        timeout_seconds=300.0         # HOTFIX: было 60.0 (5 минут)
    )
    breaker = CircuitBreaker(config)
    
    try:
        result = await breaker.call(risky_operation, arg1, arg2)
    except CircuitBreakerOpenException as e:
        logger.warning(f"Circuit breaker open: {e}")
        # Fallback logic
    ```
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """
        Инициализация Circuit Breaker.
        
        Args:
            config: Конфигурация поведения
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change_time: float = time.time()
        self.half_open_calls = 0
        
        # Статистика
        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': {
                'closed_to_open': 0,
                'open_to_half_open': 0,
                'half_open_to_closed': 0,
                'half_open_to_open': 0
            }
        }
        
        logger.info(
            f"Circuit Breaker инициализирован: "
            f"failure_threshold={config.failure_threshold}, "
            f"timeout={config.timeout_seconds}s"
        )
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Выполняет функцию через Circuit Breaker.
        
        Args:
            func: Асинхронная функция для выполнения
            *args: Позиционные аргументы для функции
            **kwargs: Именованные аргументы для функции
            
        Returns:
            Результат выполнения функции
            
        Raises:
            CircuitBreakerOpenException: Если Circuit Breaker в состоянии OPEN
            Exception: Любое исключение из функции при других состояниях
        """
        self._stats['total_calls'] += 1
        
        # Проверяем текущее состояние
        if self.state == CircuitState.OPEN:
            # Проверяем timeout для перехода в HALF_OPEN
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                # Всё ещё OPEN - блокируем запрос
                self._stats['rejected_calls'] += 1
                time_until_retry = self._time_until_retry()
                
                logger.debug(
                    f"Circuit breaker OPEN: запрос отклонён, "
                    f"следующая попытка через {time_until_retry:.0f}s"
                )
                
                raise CircuitBreakerOpenException(
                    f"Circuit breaker OPEN: VM сервис недоступен. "
                    f"Следующая попытка через {time_until_retry:.0f}s",
                    time_until_retry=time_until_retry
                )
        
        if self.state == CircuitState.HALF_OPEN:
            # Ограничиваем количество вызовов в HALF_OPEN
            if self.half_open_calls >= self.config.half_open_max_calls:
                self._stats['rejected_calls'] += 1
                raise CircuitBreakerOpenException(
                    "Circuit breaker HALF_OPEN: ожидание результата пробного запроса",
                    time_until_retry=5.0  # Примерное время
                )
            self.half_open_calls += 1
        
        # Пытаемся выполнить функцию
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            # Проверяем, нужно ли считать это failure
            if not isinstance(e, self.config.excluded_exceptions):
                self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """
        Проверяет, достаточно ли времени прошло для попытки восстановления.
        
        Returns:
            True если можно переходить в HALF_OPEN
        """
        if not self.last_failure_time:
            return False
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout_seconds
    
    def _transition_to_half_open(self):
        """Переходит из OPEN в HALF_OPEN состояние"""
        logger.info(
            f"Circuit breaker: переход OPEN -> HALF_OPEN "
            f"(прошло {time.time() - self.last_failure_time:.0f}s)"
        )
        
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.last_state_change_time = time.time()
        self._stats['state_changes']['open_to_half_open'] += 1
    
    def _on_success(self):
        """
        Обработка успешного вызова.
        
        Обновляет счётчики и может переводить в CLOSED состояние.
        """
        self._stats['successful_calls'] += 1
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            logger.debug(
                f"Circuit breaker HALF_OPEN: успех {self.success_count}/"
                f"{self.config.success_threshold}"
            )
            
            if self.success_count >= self.config.success_threshold:
                # Достаточно успехов - переходим в CLOSED
                logger.info(
                    f"Circuit breaker: переход HALF_OPEN -> CLOSED "
                    f"(успешных попыток: {self.success_count})"
                )
                
                self.state = CircuitState.CLOSED
                self.success_count = 0
                self.last_state_change_time = time.time()
                self._stats['state_changes']['half_open_to_closed'] += 1
    
    def _on_failure(self, exception: Exception):
        """
        Обработка неудачного вызова.
        
        Args:
            exception: Исключение, вызвавшее failure
        """
        self._stats['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        logger.debug(
            f"Circuit breaker failure: {type(exception).__name__} "
            f"(count: {self.failure_count}/{self.config.failure_threshold})"
        )
        
        if self.state == CircuitState.HALF_OPEN:
            # Failure в HALF_OPEN - сразу обратно в OPEN
            logger.warning(
                "Circuit breaker: переход HALF_OPEN -> OPEN "
                "(failure при восстановлении)"
            )
            
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
            self.last_state_change_time = time.time()
            self._stats['state_changes']['half_open_to_open'] += 1
            
        elif self.failure_count >= self.config.failure_threshold:
            # Превышен порог неудач - переходим в OPEN
            logger.warning(
                f"Circuit breaker: переход CLOSED -> OPEN "
                f"(failures: {self.failure_count})"
            )
            
            self.state = CircuitState.OPEN
            self.last_state_change_time = time.time()
            self._stats['state_changes']['closed_to_open'] += 1
    
    def _time_until_retry(self) -> float:
        """
        Возвращает время до следующей попытки в секундах.
        
        Returns:
            Секунды до следующей попытки, или 0 если можно пробовать сейчас
        """
        if not self.last_failure_time:
            return 0.0
        
        elapsed = time.time() - self.last_failure_time
        remaining = max(0.0, self.config.timeout_seconds - elapsed)
        
        return remaining
    
    def get_state(self) -> Dict[str, Any]:
        """
        Возвращает текущее состояние Circuit Breaker.
        
        Returns:
            Словарь с детальной информацией о состоянии:
            - state: Текущее состояние (closed/open/half_open)
            - failure_count: Текущее количество неудач
            - success_count: Текущее количество успехов (в HALF_OPEN)
            - time_until_retry: Время до следующей попытки (для OPEN)
            - time_in_current_state: Время в текущем состоянии (секунды)
        """
        current_time = time.time()
        
        state_info = {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'time_in_current_state': current_time - self.last_state_change_time
        }
        
        if self.state == CircuitState.OPEN:
            state_info['time_until_retry'] = self._time_until_retry()
        
        return state_info
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику работы Circuit Breaker.
        
        Returns:
            Словарь со статистикой:
            - total_calls: Общее количество вызовов
            - successful_calls: Количество успешных вызовов
            - failed_calls: Количество неудачных вызовов
            - rejected_calls: Количество отклонённых вызовов (OPEN)
            - success_rate: Процент успешных вызовов
            - rejection_rate: Процент отклонённых вызовов
            - state_changes: История переходов между состояниями
            - current_state: Текущее состояние
        """
        stats = self._stats.copy()
        stats['current_state'] = self.get_state()
        
        if stats['total_calls'] > 0:
            successful = stats['successful_calls']
            rejected = stats['rejected_calls']
            stats['success_rate'] = (successful / stats['total_calls']) * 100
            stats['rejection_rate'] = (rejected / stats['total_calls']) * 100
        else:
            stats['success_rate'] = 0.0
            stats['rejection_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Сбрасывает статистику (но НЕ состояние!)"""
        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': {
                'closed_to_open': 0,
                'open_to_half_open': 0,
                'half_open_to_closed': 0,
                'half_open_to_open': 0
            }
        }
    
    def reset(self):
        """Полный сброс Circuit Breaker в начальное состояние"""
        logger.info("Circuit breaker: полный сброс в CLOSED состояние")
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change_time = time.time()
        self.half_open_calls = 0
        self.reset_stats()
