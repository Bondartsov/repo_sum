"""
Unit тесты для rag/retry_policy.py

Проверяет:
- RetryConfig валидацию
- RetryPolicy adaptive timeout логику
- Exponential backoff
- Time tracking
- Статистику

Обновлено: 1 октября 2025 - совместимость с новым API
"""

import pytest
import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock
import aiohttp

from rag.retry_policy import RetryPolicy, RetryConfig


class TestRetryConfig:
    """Тесты для RetryConfig валидации"""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию"""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.timeout_seconds == 60.0
        # Обновлённый список retryable exceptions
        assert asyncio.TimeoutError in config.retryable_exceptions
        assert aiohttp.ClientError in config.retryable_exceptions
        assert ConnectionError in config.retryable_exceptions
    
    def test_custom_config(self):
        """Тест кастомной конфигурации"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=3.0,
            timeout_seconds=120.0,
            retryable_exceptions=(ValueError,)
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 3.0
        assert config.timeout_seconds == 120.0
        assert config.retryable_exceptions == (ValueError,)
    
    def test_validation_max_attempts(self):
        """Тест валидации max_attempts"""
        # Обновлённый message pattern
        with pytest.raises(ValueError, match="max_attempts должно быть >= 1"):
            RetryConfig(max_attempts=0)
        
        with pytest.raises(ValueError, match="max_attempts должно быть >= 1"):
            RetryConfig(max_attempts=-1)
    
    def test_validation_base_delay(self):
        """Тест валидации base_delay"""
        # base_delay теперь может быть >= 0 (не > 0)
        config = RetryConfig(base_delay=0.0)
        assert config.base_delay == 0.0
        
        with pytest.raises(ValueError, match="base_delay должна быть >= 0"):
            RetryConfig(base_delay=-1.0)
    
    def test_validation_max_delay(self):
        """Тест валидации max_delay"""
        # Обновлённый message pattern
        with pytest.raises(ValueError, match="max_delay.*должна быть >= base_delay"):
            RetryConfig(base_delay=10.0, max_delay=5.0)
    
    def test_validation_exponential_base(self):
        """Тест валидации exponential_base"""
        # Обновлённый message pattern
        with pytest.raises(ValueError, match="exponential_base должна быть >= 1"):
            RetryConfig(exponential_base=0.5)
    
    def test_validation_timeout_seconds(self):
        """Тест валидации timeout_seconds"""
        # Обновлённый message pattern
        with pytest.raises(ValueError, match="timeout_seconds должна быть > 0"):
            RetryConfig(timeout_seconds=0.0)


class TestRetryPolicy:
    """Тесты для RetryPolicy логики"""
    
    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Тест успешного выполнения с первой попытки"""
        config = RetryConfig(max_attempts=3)
        policy = RetryPolicy(config)
        
        async def success_func():
            return "success"
        
        result = await policy.execute_with_retry(success_func)
        
        assert result == "success"
        
        # Проверяем статистику (обновлённый API)
        stats = policy.get_stats()
        assert stats['total_executions'] == 1
        assert stats['successful_executions'] == 1
        assert stats['failed_executions'] == 0
    
    @pytest.mark.asyncio
    async def test_retry_on_retryable_exception(self):
        """Тест retry при retryable исключении"""
        config = RetryConfig(max_attempts=3, base_delay=0.01)  # Малая задержка для теста
        policy = RetryPolicy(config)
        
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError("Timeout")
            return "success"
        
        result = await policy.execute_with_retry(failing_func)
        
        assert result == "success"
        assert call_count == 3
        
        # Проверяем статистику (обновлённый API)
        stats = policy.get_stats()
        assert stats['total_executions'] == 1  # Одно успешное выполнение
        assert stats['successful_executions'] == 1
        assert stats['failed_executions'] == 0
    
    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Тест превышения максимального количества попыток"""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)
        
        async def always_failing_func():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            await policy.execute_with_retry(always_failing_func)
        
        # Проверяем статистику (обновлённый API)
        stats = policy.get_stats()
        assert stats['total_executions'] == 1  # Одно неудачное выполнение
        assert stats['successful_executions'] == 0
        assert stats['failed_executions'] == 1
    
    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Тест немедленного fail при non-retryable исключении"""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=(asyncio.TimeoutError,)  # Только TimeoutError
        )
        policy = RetryPolicy(config)
        
        call_count = 0
        
        async def failing_with_valueerror():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError, match="Non-retryable error"):
            await policy.execute_with_retry(failing_with_valueerror)
        
        # Должна быть только одна попытка, т.к. ValueError не retryable
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Тест exponential backoff задержек"""
        config = RetryConfig(
            max_attempts=4,
            base_delay=0.1,
            exponential_base=2.0,
            timeout_seconds=10.0
        )
        policy = RetryPolicy(config)
        
        delays: List[float] = []
        call_count = 0
        last_time = time.time()
        
        async def failing_func():
            nonlocal call_count, last_time
            current_time = time.time()
            if call_count > 0:
                delays.append(current_time - last_time)
            last_time = current_time
            call_count += 1
            
            if call_count < 4:
                raise asyncio.TimeoutError("Timeout")
            return "success"
        
        await policy.execute_with_retry(failing_func)
        
        # Проверяем что задержки увеличиваются экспоненциально
        # delay1 ≈ 0.1s, delay2 ≈ 0.2s, delay3 ≈ 0.4s
        assert len(delays) == 3
        assert 0.08 < delays[0] < 0.15  # ~0.1s ± погрешность
        assert 0.18 < delays[1] < 0.25  # ~0.2s ± погрешность
        assert 0.38 < delays[2] < 0.45  # ~0.4s ± погрешность
    
    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        """Тест соблюдения общего timeout"""
        config = RetryConfig(
            max_attempts=10,
            base_delay=0.5,
            timeout_seconds=1.0  # Короткий timeout
        )
        policy = RetryPolicy(config)
        
        async def slow_failing_func():
            await asyncio.sleep(0.2)  # Каждый вызов занимает 0.2s
            raise asyncio.TimeoutError("Timeout")
        
        start_time = time.time()
        
        with pytest.raises(asyncio.TimeoutError):
            await policy.execute_with_retry(slow_failing_func)
        
        elapsed = time.time() - start_time
        
        # Должно прерваться около 1 секунды, не дожидаясь всех 10 попыток
        assert elapsed < 2.0  # С запасом на погрешность
    
    @pytest.mark.asyncio
    async def test_remaining_time_tracking(self):
        """Тест tracking оставшегося времени"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.1,
            timeout_seconds=0.5  # Очень короткий timeout
        )
        policy = RetryPolicy(config)
        
        async def failing_func():
            raise asyncio.TimeoutError("Timeout")
        
        with pytest.raises(asyncio.TimeoutError):
            await policy.execute_with_retry(failing_func)
    
    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Тест ограничения максимальной задержки"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=2.0,  # Ограничение на 2 секунды
            exponential_base=10.0,  # Агрессивный рост
            timeout_seconds=20.0
        )
        policy = RetryPolicy(config)
        
        delays: List[float] = []
        call_count = 0
        last_time = time.time()
        
        async def failing_func():
            nonlocal call_count, last_time
            current_time = time.time()
            if call_count > 0:
                delays.append(current_time - last_time)
            last_time = current_time
            call_count += 1
            
            if call_count < 5:
                raise asyncio.TimeoutError("Timeout")
            return "success"
        
        await policy.execute_with_retry(failing_func)
        
        # Все задержки должны быть <= max_delay
        for delay in delays:
            assert delay <= 2.1  # max_delay + погрешность
    
    @pytest.mark.asyncio
    async def test_stats_reset(self):
        """Тест сброса статистики"""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        policy = RetryPolicy(config)
        
        # Первый вызов с retry
        call_count = 0
        
        async def func_with_one_retry():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return "success"
        
        await policy.execute_with_retry(func_with_one_retry)
        
        stats_before = policy.get_stats()
        assert stats_before['total_executions'] >= 1
        assert stats_before['successful_executions'] >= 1
        
        # Сброс статистики
        policy.reset_stats()
        
        stats_after = policy.get_stats()
        assert stats_after['total_executions'] == 0
        assert stats_after['successful_executions'] == 0
        assert stats_after['failed_executions'] == 0
    
    @pytest.mark.asyncio
    async def test_function_with_args_and_kwargs(self):
        """Тест передачи аргументов в функцию"""
        config = RetryConfig(max_attempts=1)
        policy = RetryPolicy(config)
        
        async def func_with_params(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result = await policy.execute_with_retry(
            func_with_params,
            10, 20, c=30
        )
        
        assert result == "10-20-30"
    
    @pytest.mark.asyncio
    async def test_concurrent_retries(self):
        """Тест параллельного использования retry policy"""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)
        
        call_counts = {0: 0, 1: 0, 2: 0}
        
        async def task(task_id: int):
            call_counts[task_id] += 1
            # Первые 2 попытки fail, третья успешна
            if call_counts[task_id] < 3:
                raise asyncio.TimeoutError()
            return task_id
        
        # Запускаем 3 задачи параллельно
        tasks = [policy.execute_with_retry(task, i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # Проверяем что все задачи завершились успешно
        assert results == [0, 1, 2]


class TestRetryPolicyEdgeCases:
    """Тесты граничных случаев"""
    
    @pytest.mark.asyncio
    async def test_zero_base_delay_with_retries(self):
        """Тест с нулевой базовой задержкой (теперь это валидно)"""
        config = RetryConfig(base_delay=0.0, max_attempts=2)
        policy = RetryPolicy(config)
        
        call_count = 0
        
        async def failing_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return "success"
        
        result = await policy.execute_with_retry(failing_once)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_single_attempt_no_retry(self):
        """Тест с max_attempts=1 (без retry)"""
        config = RetryConfig(max_attempts=1)
        policy = RetryPolicy(config)
        
        async def failing_func():
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            await policy.execute_with_retry(failing_func)
        
        stats = policy.get_stats()
        assert stats['total_executions'] == 1
        assert stats['failed_executions'] == 1
    
    @pytest.mark.asyncio
    async def test_very_long_timeout(self):
        """Тест с очень большим timeout"""
        config = RetryConfig(
            max_attempts=2,
            base_delay=0.01,
            timeout_seconds=1000.0  # Очень большой timeout
        )
        policy = RetryPolicy(config)
        
        call_count = 0
        
        async def failing_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return "success"
        
        result = await policy.execute_with_retry(failing_once)
        assert result == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
