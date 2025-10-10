"""
Unit тесты для Circuit Breaker Pattern.

Автор: AI Assistant
Дата: 1 октября 2025
"""

import pytest
import pytest_asyncio
import asyncio
import time

from rag.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpenException
)


# ============================================================================
# Фикстуры
# ============================================================================

@pytest.fixture
def default_config():
    """Стандартная конфигурация для тестов"""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=5.0,
        half_open_max_calls=1
    )


@pytest.fixture
def circuit_breaker(default_config):
    """Circuit breaker с дефолтной конфигурацией"""
    return CircuitBreaker(default_config)


@pytest_asyncio.fixture
async def success_func():
    """Mock функция, которая всегда успешна"""
    async def _func():
        return "success"
    return _func


@pytest_asyncio.fixture
async def failure_func():
    """Mock функция, которая всегда падает"""
    async def _func():
        raise ValueError("Test error")
    return _func


# ============================================================================
# TestCircuitBreakerConfig
# ============================================================================

class TestCircuitBreakerConfig:
    """Тесты валидации конфигурации Circuit Breaker"""
    
    def test_default_config_valid(self):
        """Дефолтная конфигурация должна быть валидной"""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 60.0
        assert config.half_open_max_calls == 1
    
    def test_custom_config_valid(self):
        """Кастомная конфигурация должна сохранять значения"""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout_seconds=120.0,
            half_open_max_calls=2
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout_seconds == 120.0
        assert config.half_open_max_calls == 2
    
    def test_failure_threshold_validation(self):
        """failure_threshold должен быть >= 1"""
        with pytest.raises(ValueError, match="failure_threshold должен быть >= 1"):
            CircuitBreakerConfig(failure_threshold=0)
        
        with pytest.raises(ValueError, match="failure_threshold должен быть >= 1"):
            CircuitBreakerConfig(failure_threshold=-1)
    
    def test_success_threshold_validation(self):
        """success_threshold должен быть >= 1"""
        with pytest.raises(ValueError, match="success_threshold должен быть >= 1"):
            CircuitBreakerConfig(success_threshold=0)
        
        with pytest.raises(ValueError, match="success_threshold должен быть >= 1"):
            CircuitBreakerConfig(success_threshold=-1)
    
    def test_timeout_validation(self):
        """timeout_seconds должен быть > 0"""
        with pytest.raises(ValueError, match="timeout_seconds должен быть > 0"):
            CircuitBreakerConfig(timeout_seconds=0)
        
        with pytest.raises(ValueError, match="timeout_seconds должен быть > 0"):
            CircuitBreakerConfig(timeout_seconds=-5.0)
    
    def test_half_open_max_calls_validation(self):
        """half_open_max_calls должен быть >= 1"""
        with pytest.raises(ValueError, match="half_open_max_calls должен быть >= 1"):
            CircuitBreakerConfig(half_open_max_calls=0)
        
        with pytest.raises(ValueError, match="half_open_max_calls должен быть >= 1"):
            CircuitBreakerConfig(half_open_max_calls=-1)


# ============================================================================
# TestCircuitBreakerStates
# ============================================================================

class TestCircuitBreakerStates:
    """Тесты state machine логики Circuit Breaker"""
    
    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, circuit_breaker):
        """Начальное состояние должно быть CLOSED"""
        state = circuit_breaker.get_state()
        assert state['state'] == CircuitState.CLOSED.value
        assert state['failure_count'] == 0
        assert state['success_count'] == 0
    
    @pytest.mark.asyncio
    async def test_successful_call_in_closed_state(self, circuit_breaker, success_func):
        """Успешный вызов в CLOSED должен оставить состояние CLOSED"""
        result = await circuit_breaker.call(success_func)
        
        assert result == "success"
        state = circuit_breaker.get_state()
        assert state['state'] == CircuitState.CLOSED.value
        assert state['failure_count'] == 0
    
    @pytest.mark.asyncio
    async def test_transition_closed_to_open(self, circuit_breaker, failure_func):
        """После failure_threshold падений должен перейти в OPEN"""
        # Делаем 3 неудачных вызова (failure_threshold=3)
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failure_func)
        
        state = circuit_breaker.get_state()
        assert state['state'] == CircuitState.OPEN.value
        assert state['failure_count'] == 3
    
    @pytest.mark.asyncio
    async def test_open_state_rejects_calls(self, circuit_breaker, success_func):
        """В состоянии OPEN должны отклоняться все вызовы"""
        # Переводим в OPEN
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 3
        circuit_breaker.last_failure_time = time.time()  # Используем time.time(), не event loop time
        
        with pytest.raises(CircuitBreakerOpenException, match="Circuit breaker OPEN"):
            await circuit_breaker.call(success_func)
    
    @pytest.mark.asyncio
    async def test_transition_open_to_half_open_after_timeout(self, circuit_breaker, success_func):
        """После timeout должен перейти из OPEN в HALF_OPEN"""
        # Переводим в OPEN с временем в прошлом
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 3
        circuit_breaker.last_failure_time = time.time() - 10.0  # 10 секунд назад (> timeout_seconds=5)
        
        # Пробуем вызвать - должен перейти в HALF_OPEN
        result = await circuit_breaker.call(success_func)
        
        assert result == "success"
        state = circuit_breaker.get_state()
        # Один успех не закрывает circuit (нужно success_threshold=2)
        assert state['state'] == CircuitState.HALF_OPEN.value
        assert state['success_count'] == 1
    
    @pytest.mark.asyncio
    async def test_half_open_success_transitions_to_closed(self, circuit_breaker, success_func):
        """Success в HALF_OPEN после success_threshold попыток переводит в CLOSED"""
        # Переводим в HALF_OPEN и увеличиваем half_open_max_calls для множественных вызовов
        circuit_breaker.config.half_open_max_calls = 2
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.half_open_calls = 0
        circuit_breaker.success_count = 0
        
        # Делаем success_threshold=2 успешных вызова
        await circuit_breaker.call(success_func)
        await circuit_breaker.call(success_func)
        
        state = circuit_breaker.get_state()
        assert state['state'] == CircuitState.CLOSED.value
        assert state['success_count'] == 0  # Сбрасывается при переходе в CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_failure_transitions_to_open(self, circuit_breaker, failure_func):
        """Failure в HALF_OPEN переводит обратно в OPEN"""
        # Переводим в HALF_OPEN
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.half_open_calls = 0
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(failure_func)
        
        state = circuit_breaker.get_state()
        assert state['state'] == CircuitState.OPEN.value
    
    @pytest.mark.asyncio
    async def test_half_open_max_calls_limit(self, circuit_breaker, success_func):
        """В HALF_OPEN должно быть ограничение на concurrent calls"""
        # Переводим в HALF_OPEN
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.half_open_calls = 1  # Уже 1 вызов в процессе (max=1)
        
        with pytest.raises(CircuitBreakerOpenException, match="ожидание результата пробного запроса"):
            await circuit_breaker.call(success_func)


# ============================================================================
# TestCircuitBreakerCounters
# ============================================================================

class TestCircuitBreakerCounters:
    """Тесты счётчиков Circuit Breaker"""
    
    @pytest.mark.asyncio
    async def test_failure_count_increments(self, circuit_breaker, failure_func):
        """failure_count должен инкрементироваться"""
        with pytest.raises(ValueError):
            await circuit_breaker.call(failure_func)
        
        state = circuit_breaker.get_state()
        assert state['failure_count'] == 1
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(failure_func)
        
        state = circuit_breaker.get_state()
        assert state['failure_count'] == 2
    
    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, circuit_breaker, success_func, failure_func):
        """Успешный вызов должен сбросить failure_count"""
        # Делаем 2 неудачных вызова
        for _ in range(2):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failure_func)
        
        state = circuit_breaker.get_state()
        assert state['failure_count'] == 2
        
        # Делаем успешный вызов
        await circuit_breaker.call(success_func)
        
        state = circuit_breaker.get_state()
        assert state['failure_count'] == 0
    
    @pytest.mark.asyncio
    async def test_success_count_in_half_open(self, circuit_breaker, success_func):
        """success_count должен инкрементироваться в HALF_OPEN"""
        # Переводим в HALF_OPEN
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.success_count = 0
        
        await circuit_breaker.call(success_func)
        
        state = circuit_breaker.get_state()
        assert state['success_count'] == 1


# ============================================================================
# TestCircuitBreakerTimings
# ============================================================================

class TestCircuitBreakerTimings:
    """Тесты timeout логики Circuit Breaker"""
    
    @pytest.mark.asyncio
    async def test_time_until_retry_when_open(self, circuit_breaker):
        """time_until_retry должно корректно считаться в OPEN состоянии"""
        # Переводим в OPEN
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 3
        circuit_breaker.last_failure_time = asyncio.get_event_loop().time()
        
        state = circuit_breaker.get_state()
        assert state['time_until_retry'] is not None
        assert 0 <= state['time_until_retry'] <= circuit_breaker.config.timeout_seconds
    
    @pytest.mark.asyncio
    async def test_time_until_retry_none_when_closed(self, circuit_breaker):
        """time_until_retry НЕ присутствует в CLOSED состоянии"""
        state = circuit_breaker.get_state()
        assert 'time_until_retry' not in state
    
    @pytest.mark.asyncio
    async def test_last_failure_time_updated(self, circuit_breaker, failure_func):
        """last_failure_time должно обновляться при падениях"""
        state_before = circuit_breaker.get_state()
        
        await asyncio.sleep(0.1)  # Небольшая задержка
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(failure_func)
        
        # last_failure_time должно обновиться
        assert circuit_breaker.last_failure_time is not None


# ============================================================================
# TestCircuitBreakerEdgeCases
# ============================================================================

class TestCircuitBreakerEdgeCases:
    """Тесты граничных случаев Circuit Breaker"""
    
    @pytest.mark.asyncio
    async def test_exactly_failure_threshold_opens_circuit(self):
        """Ровно failure_threshold падений должно открыть circuit"""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config)
        
        async def fail():
            raise ValueError("error")
        
        # Делаем ровно 5 падений
        for _ in range(5):
            with pytest.raises(ValueError):
                await cb.call(fail)
        
        state = cb.get_state()
        assert state['state'] == CircuitState.OPEN.value
    
    @pytest.mark.asyncio
    async def test_one_less_than_threshold_keeps_closed(self):
        """failure_threshold - 1 падений НЕ должно открыть circuit"""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config)
        
        async def fail():
            raise ValueError("error")
        
        # Делаем 4 падения (< 5)
        for _ in range(4):
            with pytest.raises(ValueError):
                await cb.call(fail)
        
        state = cb.get_state()
        assert state['state'] == CircuitState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_exactly_success_threshold_closes_circuit(self):
        """Ровно success_threshold успехов должно закрыть circuit"""
        config = CircuitBreakerConfig(success_threshold=3)
        cb = CircuitBreaker(config)
        
        async def succeed():
            return "ok"
        
        # Переводим в HALF_OPEN и разрешаем 3 concurrent вызова
        cb.config.half_open_max_calls = 3
        cb.state = CircuitState.HALF_OPEN
        cb.success_count = 0
        
        # Делаем ровно 3 успеха
        for _ in range(3):
            await cb.call(succeed)
        
        state = cb.get_state()
        assert state['state'] == CircuitState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure_in_closed(self):
        """Смешанные успехи/падения в CLOSED должны корректно обрабатываться"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)
        
        async def succeed():
            return "ok"
        
        async def fail():
            raise ValueError("error")
        
        # Чередуем успехи и падения
        await cb.call(succeed)  # success -> failure_count=0
        
        with pytest.raises(ValueError):
            await cb.call(fail)  # failure -> failure_count=1
        
        await cb.call(succeed)  # success -> failure_count=0 (сбрасывается!)
        
        with pytest.raises(ValueError):
            await cb.call(fail)  # failure -> failure_count=1
        
        state = cb.get_state()
        assert state['state'] == CircuitState.CLOSED.value
        assert state['failure_count'] == 1
    
    @pytest.mark.asyncio
    async def test_exception_propagation(self, circuit_breaker):
        """Исключения должны корректно пробрасываться"""
        async def custom_error():
            raise KeyError("custom message")
        
        with pytest.raises(KeyError, match="custom message"):
            await circuit_breaker.call(custom_error)
    
    @pytest.mark.asyncio
    async def test_return_value_propagation(self, circuit_breaker):
        """Возвращаемые значения должны корректно пробрасываться"""
        async def return_dict():
            return {"key": "value", "number": 42}
        
        result = await circuit_breaker.call(return_dict)
        assert result == {"key": "value", "number": 42}


# ============================================================================
# TestCircuitBreakerStats
# ============================================================================

class TestCircuitBreakerStats:
    """Тесты get_state() метода"""
    
    @pytest.mark.asyncio
    async def test_get_state_includes_all_fields(self, circuit_breaker):
        """get_state() должен возвращать все необходимые поля"""
        state = circuit_breaker.get_state()
        
        assert 'state' in state
        assert 'failure_count' in state
        assert 'success_count' in state
        assert 'time_in_current_state' in state
        # time_until_retry присутствует только в OPEN состоянии
    
    @pytest.mark.asyncio
    async def test_get_state_types(self, circuit_breaker):
        """get_state() должен возвращать корректные типы"""
        state = circuit_breaker.get_state()
        
        assert isinstance(state['state'], str)
        assert isinstance(state['failure_count'], int)
        assert isinstance(state['success_count'], int)
        assert isinstance(state['time_in_current_state'], (int, float))


# ============================================================================
# TestCircuitBreakerIntegration
# ============================================================================

class TestCircuitBreakerIntegration:
    """Integration тесты для реальных сценариев"""
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_closed_open_halfopen_closed(self):
        """Полный жизненный цикл: CLOSED -> OPEN -> HALF_OPEN -> CLOSED"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout_seconds=0.2,  # Короткий timeout для быстрого теста
            half_open_max_calls=1
        )
        cb = CircuitBreaker(config)
        
        async def flaky_func(should_fail: bool):
            if should_fail:
                raise ValueError("fail")
            return "success"
        
        # 1. CLOSED: успешные вызовы
        result = await cb.call(flaky_func, False)
        assert result == "success"
        assert cb.get_state()['state'] == CircuitState.CLOSED.value
        
        # 2. CLOSED -> OPEN: 2 падения
        with pytest.raises(ValueError):
            await cb.call(flaky_func, True)
        with pytest.raises(ValueError):
            await cb.call(flaky_func, True)
        assert cb.get_state()['state'] == CircuitState.OPEN.value
        
        # 3. OPEN: вызовы отклоняются
        with pytest.raises(CircuitBreakerOpenException):
            await cb.call(flaky_func, False)
        
        # 4. Ждём timeout
        await asyncio.sleep(0.3)
        
        # 5. OPEN -> HALF_OPEN -> CLOSED: успешный вызов
        result = await cb.call(flaky_func, False)
        assert result == "success"
        assert cb.get_state()['state'] == CircuitState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_concurrent_calls_in_closed(self):
        """Concurrent вызовы в CLOSED должны работать корректно"""
        cb = CircuitBreaker(CircuitBreakerConfig())
        
        async def slow_success():
            await asyncio.sleep(0.1)
            return "ok"
        
        # Запускаем 5 concurrent вызовов
        tasks = [cb.call(slow_success) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(r == "ok" for r in results)
        assert cb.get_state()['state'] == CircuitState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_recovery_after_long_downtime(self):
        """Recovery после длительного downtime"""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            timeout_seconds=0.5
        )
        cb = CircuitBreaker(config)
        
        async def fail():
            raise ValueError("error")
        
        async def succeed():
            return "ok"
        
        # Открываем circuit
        with pytest.raises(ValueError):
            await cb.call(fail)
        
        assert cb.get_state()['state'] == CircuitState.OPEN.value
        
        # Ждём долго (симуляция длительного downtime)
        await asyncio.sleep(1.0)  # > timeout_seconds
        
        # Должны восстановиться
        result = await cb.call(succeed)
        assert result == "ok"
        assert cb.get_state()['state'] == CircuitState.CLOSED.value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
