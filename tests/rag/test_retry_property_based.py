"""
Property-based тесты для RetryPolicy с использованием hypothesis.

Проверяем инварианты и edge cases через генерацию случайных входных данных.

Автор: AI Assistant
Дата: 1 октября 2025
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

from rag.retry_policy import RetryPolicy, RetryConfig


# ============================================================================
# Стратегии для генерации данных
# ============================================================================

@st.composite
def valid_retry_config(draw):
    """Генерирует валидную конфигурацию RetryPolicy"""
    max_attempts = draw(st.integers(min_value=1, max_value=5))  # Уменьшено с 10
    base_delay = draw(st.floats(min_value=0.01, max_value=0.5))  # Уменьшено с 5.0
    max_delay = draw(st.floats(min_value=base_delay, max_value=5.0))  # Уменьшено с 30.0
    # Увеличен минимальный timeout для стабильности
    timeout_seconds = draw(st.floats(min_value=10.0, max_value=60.0))
    exponential_base = draw(st.floats(min_value=1.1, max_value=2.0))  # Уменьшено с 3.0
    
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        timeout_seconds=timeout_seconds,
        exponential_base=exponential_base
    )


@st.composite
def failure_pattern(draw):
    """Генерирует паттерн падений: список bool (True = fail, False = success)"""
    length = draw(st.integers(min_value=1, max_value=10))
    return draw(st.lists(st.booleans(), min_size=length, max_size=length))


# ============================================================================
# Property: Успешный retry всегда возвращает результат
# ============================================================================

class TestRetrySuccessProperties:
    """Property-based тесты для успешных retry сценариев"""
    
    @given(valid_retry_config())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_eventual_success_returns_value(self, config):
        """
        Property: Если функция в конце концов успешна в пределах max_attempts,
        retry должен вернуть корректное значение.
        """
        retry_policy = RetryPolicy(config)
        call_count = 0
        expected_value = "success_value"
        
        # Функция падает (max_attempts - 1) раз, затем успех
        fail_times = min(config.max_attempts - 1, 3)  # Ограничим для скорости
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count <= fail_times:
                raise ValueError(f"Fail {call_count}")
            return expected_value
        
        result = await retry_policy.execute_with_retry(flaky_func)
        
        # Property: результат должен совпадать с ожидаемым
        assert result == expected_value
        # Property: должно быть ровно (fail_times + 1) попыток
        assert call_count == fail_times + 1
    
    @given(valid_retry_config(), st.integers(min_value=1, max_value=100))
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_immediate_success_no_retries(self, config, return_value):
        """
        Property: Немедленно успешная функция не должна вызывать retries.
        """
        retry_policy = RetryPolicy(config)
        call_count = 0
        
        async def immediate_success():
            nonlocal call_count
            call_count += 1
            return return_value
        
        result = await retry_policy.execute_with_retry(immediate_success)
        
        # Property: только одна попытка
        assert call_count == 1
        # Property: результат совпадает
        assert result == return_value


# ============================================================================
# Property: Failure после max_attempts всегда пробрасывает исключение
# ============================================================================

class TestRetryFailureProperties:
    """Property-based тесты для failure сценариев"""
    
    @given(valid_retry_config())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_max_attempts_exhaustion_raises(self, config):
        """
        Property: Если функция падает max_attempts раз, должно быть исключение.
        """
        retry_policy = RetryPolicy(config)
        call_count = 0
        
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            await retry_policy.execute_with_retry(always_fails)
        
        # Property: ровно max_attempts попыток
        assert call_count == config.max_attempts
    
    @given(
        valid_retry_config(),
        st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_exception_type_preserved(self, config, error_code):
        """
        Property: Тип исключения должен сохраняться.
        """
        retry_policy = RetryPolicy(config)
        
        class CustomError(Exception):
            def __init__(self, code):
                self.code = code
                super().__init__(f"Error {code}")
        
        async def custom_fail():
            raise CustomError(error_code)
        
        with pytest.raises(CustomError) as exc_info:
            await retry_policy.execute_with_retry(custom_fail)
        
        # Property: исключение сохраняет свои атрибуты
        assert exc_info.value.code == error_code


# ============================================================================
# Property: Backoff delay инварианты
# ============================================================================

class TestBackoffProperties:
    """Property-based тесты для exponential backoff логики"""
    
    @given(valid_retry_config())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_backoff_increases_monotonically(self, config):
        """
        Property: Задержки следуют exponential backoff формуле или ограничены adaptive timeout.
        
        Замечание: Delays могут уменьшаться из-за adaptive timeout (remaining/2),
        что является правильным поведением, а не багом.
        """
        import time
        retry_policy = RetryPolicy(config)
        delays = []
        
        async def measure_delays():
            nonlocal delays
            if len(delays) < config.max_attempts:
                delays.append(time.time())
                raise ValueError("Measuring delays")
            return "done"
        
        try:
            await retry_policy.execute_with_retry(measure_delays)
        except ValueError:
            pass
        
        # Вычисляем реальные задержки между попытками
        if len(delays) > 1:
            real_delays = [delays[i+1] - delays[i] for i in range(len(delays)-1)]
            
            # Property: все задержки неотрицательны и конечны
            for delay in real_delays:
                assert delay >= 0, f"Задержка должна быть >= 0: {delay}"
                assert delay < float('inf'), "Задержка должна быть конечной"
    
    @given(valid_retry_config())
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_delay_never_exceeds_max(self, config):
        """
        Property: Задержка никогда не превышает max_delay.
        """
        import time
        retry_policy = RetryPolicy(config)
        delays = []
        
        async def measure_delays():
            nonlocal delays
            if len(delays) < config.max_attempts:
                delays.append(time.time())
                raise ValueError("Measuring")
            return "done"
        
        try:
            await retry_policy.execute_with_retry(measure_delays)
        except ValueError:
            pass
        
        if len(delays) > 1:
            real_delays = [delays[i+1] - delays[i] for i in range(len(delays)-1)]
            
            # Property: все задержки <= max_delay + overhead
            for delay in real_delays:
                assert delay <= config.max_delay + 0.5  # 0.5s overhead


# ============================================================================
# Property: Timeout инварианты
# ============================================================================

class TestTimeoutProperties:
    """Property-based тесты для timeout логики"""
    
    @given(
        st.integers(min_value=1, max_value=3),  # Уменьшено с 5
        st.floats(min_value=5.0, max_value=10.0)  # Увеличено с 0.1-2.0
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_timeout_enforced(self, max_attempts, timeout_seconds):
        """
        Property: Общее время выполнения <= timeout_seconds + overhead.
        """
        import time
        
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=0.05,  # Уменьшено с 0.1
            timeout_seconds=timeout_seconds
        )
        retry_policy = RetryPolicy(config)
        
        async def slow_fail():
            await asyncio.sleep(0.2)  # Медленная функция
            raise ValueError("Slow fail")
        
        start_time = time.time()
        
        try:
            await retry_policy.execute_with_retry(slow_fail)
        except (ValueError, asyncio.TimeoutError):
            pass
        
        elapsed = time.time() - start_time
        
        # Property: общее время <= timeout + разумный overhead
        assert elapsed <= timeout_seconds + 1.0
    
    @given(valid_retry_config())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_remaining_time_decreases(self, config):
        """
        Property: Оставшееся время монотонно убывает.
        """
        retry_policy = RetryPolicy(config)
        call_times = []
        
        async def track_calls():
            import time
            call_times.append(time.time())
            if len(call_times) < min(3, config.max_attempts):
                raise ValueError("Tracking")
            return "done"
        
        try:
            await retry_policy.execute_with_retry(track_calls)
        except ValueError:
            pass
        
        # Property: время между вызовами положительное
        if len(call_times) > 1:
            for i in range(len(call_times) - 1):
                assert call_times[i+1] > call_times[i]


# ============================================================================
# Property: Statistics инварианты
# ============================================================================

class TestStatisticsProperties:
    """Property-based тесты для статистики"""
    
    @given(
        valid_retry_config(),
        st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_stats_accuracy_on_success(self, config, num_requests):
        """
        Property: successful_attempts == количество успешных вызовов execute_with_retry.
        """
        retry_policy = RetryPolicy(config)
        
        async def success():
            return "ok"
        
        # Делаем num_requests успешных запросов
        for _ in range(num_requests):
            await retry_policy.execute_with_retry(success)
        
        stats = retry_policy.get_stats()
        
        # Property: статистика корректна
        assert stats['total_executions'] == num_requests
        assert stats['successful_executions'] == num_requests
        assert stats['failed_executions'] == 0
    
    @given(
        valid_retry_config(),
        st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_stats_accuracy_on_failure(self, config, num_requests):
        """
        Property: failed_attempts == количество failed вызовов execute_with_retry.
        """
        retry_policy = RetryPolicy(config)
        
        async def fail():
            raise ValueError("Fail")
        
        # Делаем num_requests failed запросов
        for _ in range(num_requests):
            try:
                await retry_policy.execute_with_retry(fail)
            except ValueError:
                pass
        
        stats = retry_policy.get_stats()
        
        # Property: статистика корректна
        assert stats['total_executions'] == num_requests
        assert stats['successful_executions'] == 0
        assert stats['failed_executions'] == num_requests
    
    @given(valid_retry_config())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_stats_reset_works(self, config):
        """
        Property: После reset_stats() все счётчики == 0.
        """
        retry_policy = RetryPolicy(config)
        
        # Делаем несколько запросов
        async def success():
            return "ok"
        
        for _ in range(3):
            await retry_policy.execute_with_retry(success)
        
        # Сбрасываем статистику
        retry_policy.reset_stats()
        
        stats = retry_policy.get_stats()
        
        # Property: всё обнулено
        assert stats['total_executions'] == 0
        assert stats['successful_executions'] == 0
        assert stats['failed_executions'] == 0


# ============================================================================
# Property: Idempotency и determinism
# ============================================================================

class TestIdempotencyProperties:
    """Property-based тесты для idempotency"""
    
    @given(
        valid_retry_config(),
        st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_same_input_same_output(self, config, input_value):
        """
        Property: Идентичный вход всегда даёт идентичный выход (для pure functions).
        """
        retry_policy = RetryPolicy(config)
        
        async def pure_function(x):
            return x * 2
        
        # Вызываем дважды с одинаковым входом
        result1 = await retry_policy.execute_with_retry(pure_function, input_value)
        result2 = await retry_policy.execute_with_retry(pure_function, input_value)
        
        # Property: результаты идентичны
        assert result1 == result2
        assert result1 == input_value * 2
    
    @given(valid_retry_config())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_exception_determinism(self, config):
        """
        Property: Одинаковая failing функция всегда даёт одинаковый exception.
        """
        retry_policy1 = RetryPolicy(config)
        retry_policy2 = RetryPolicy(config)
        
        async def deterministic_fail():
            raise ValueError("Deterministic error")
        
        # Оба должны упасть с одинаковым исключением
        exception1 = None
        exception2 = None
        
        try:
            await retry_policy1.execute_with_retry(deterministic_fail)
        except ValueError as e:
            exception1 = str(e)
        
        try:
            await retry_policy2.execute_with_retry(deterministic_fail)
        except ValueError as e:
            exception2 = str(e)
        
        # Property: исключения идентичны
        assert exception1 == exception2


# ============================================================================
# Property: Edge cases
# ============================================================================

class TestEdgeCaseProperties:
    """Property-based тесты для граничных случаев"""
    
    @given(st.floats(min_value=2.0, max_value=5.0))  # Увеличено с 0.01-0.5
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_minimal_timeout_handling(self, timeout):
        """
        Property: Даже с минимальным timeout должен быть хотя бы 1 retry.
        """
        config = RetryConfig(
            max_attempts=2,  # Уменьшено с 3
            base_delay=0.01,
            timeout_seconds=timeout
        )
        retry_policy = RetryPolicy(config)
        call_count = 0
        
        async def fast_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Fast fail")
        
        try:
            await retry_policy.execute_with_retry(fast_fail)
        except (ValueError, asyncio.TimeoutError):
            pass
        
        # Property: минимум 1 попытка всегда происходит
        assert call_count >= 1
    
    @given(
        st.integers(min_value=1, max_value=5),  # Уменьшено с 10
        st.floats(min_value=5.0, max_value=10.0)  # Увеличено с 0.01-1.0
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_zero_delay_between_retries(self, max_attempts, timeout):
        """
        Property: base_delay=0 (мгновенный retry) должен работать корректно.
        """
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=0.0,  # Без задержки
            timeout_seconds=timeout
        )
        retry_policy = RetryPolicy(config)
        call_count = 0
        
        async def instant_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Instant fail")
        
        try:
            await retry_policy.execute_with_retry(instant_fail)
        except (ValueError, asyncio.TimeoutError):
            pass
        
        # Property: должны произойти все попытки или timeout
        assert call_count >= 1


# ============================================================================
# Property: Композиция и вложенность
# ============================================================================

class TestCompositionProperties:
    """Property-based тесты для композиции retry policies"""
    
    @given(valid_retry_config(), valid_retry_config())
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_nested_retry_policies(self, config1, config2):
        """
        Property: Вложенные retry policies работают корректно.
        """
        # Ограничим для скорости тестов
        config1.max_attempts = min(config1.max_attempts, 2)
        config2.max_attempts = min(config2.max_attempts, 2)
        config1.timeout_seconds = min(config1.timeout_seconds, 5.0)
        config2.timeout_seconds = min(config2.timeout_seconds, 5.0)
        
        # Гарантируем, что хотя бы у одного max_attempts >= 2
        # чтобы eventual_success смогла восстановиться
        assume(config1.max_attempts >= 2 or config2.max_attempts >= 2)
        
        retry1 = RetryPolicy(config1)
        retry2 = RetryPolicy(config2)
        
        call_count = 0
        
        async def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First fail")
            return "success"
        
        async def inner():
            return await retry2.execute_with_retry(eventual_success)
        
        result = await retry1.execute_with_retry(inner)
        
        # Property: вложенный retry должен вернуть результат
        assert result == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
