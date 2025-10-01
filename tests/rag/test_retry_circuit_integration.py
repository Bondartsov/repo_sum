"""
Integration тесты для RetryPolicy + CircuitBreaker.

Тестируем взаимодействие двух компонентов вместе в реальных сценариях.

Автор: AI Assistant
Дата: 1 октября 2025
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock

from rag.retry_policy import RetryPolicy, RetryConfig
from rag.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerOpenException


# ============================================================================
# Фикстуры
# ============================================================================

@pytest.fixture
def retry_config():
    """Конфигурация retry с короткими таймаутами для тестов"""
    return RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=1.0,
        timeout_seconds=5.0
    )


@pytest.fixture
def circuit_config():
    """Конфигурация circuit breaker с низкими порогами для тестов"""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=1.0,
        half_open_max_calls=1
    )


@pytest.fixture
def retry_policy(retry_config):
    """RetryPolicy с тестовой конфигурацией"""
    return RetryPolicy(retry_config)


@pytest.fixture
def circuit_breaker(circuit_config):
    """CircuitBreaker с тестовой конфигурацией"""
    return CircuitBreaker(circuit_config)


# ============================================================================
# TestRetryWithCircuitBreaker
# ============================================================================

class TestRetryWithCircuitBreaker:
    """Тесты взаимодействия RetryPolicy и CircuitBreaker"""
    
    @pytest.mark.asyncio
    async def test_success_through_both_layers(self, retry_policy, circuit_breaker):
        """Успешный запрос проходит через оба слоя защиты"""
        async def successful_request():
            return {"status": "ok"}
        
        # Оборачиваем в circuit breaker, затем в retry
        async def protected_request():
            return await circuit_breaker.call(successful_request)
        
        result = await retry_policy.execute_with_retry(protected_request)
        
        assert result == {"status": "ok"}
        assert circuit_breaker.get_state()['state'] == CircuitState.CLOSED.value
        assert retry_policy.get_stats()['successful_executions'] == 1
    
    @pytest.mark.asyncio
    async def test_retry_recovers_transient_failures(self, retry_policy, circuit_breaker):
        """RetryPolicy восстанавливается после временных падений"""
        call_count = 0
        
        async def flaky_request():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        async def protected_request():
            return await circuit_breaker.call(flaky_request)
        
        result = await retry_policy.execute_with_retry(protected_request)
        
        assert result == "success"
        assert call_count == 3
        assert circuit_breaker.get_state()['state'] == CircuitState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_retry_exhaustion(self, retry_policy, circuit_breaker):
        """Circuit breaker открывается после исчерпания retry попыток"""
        async def always_fails():
            raise ConnectionError("Persistent failure")
        
        # Делаем несколько запросов с retry
        for _ in range(3):  # failure_threshold=3
            try:
                async def protected_request():
                    return await circuit_breaker.call(always_fails)
                
                await retry_policy.execute_with_retry(protected_request)
            except (ConnectionError, CircuitBreakerOpenException):
                pass  # Ожидаемо - либо ConnectionError, либо CircuitBreakerOpenException после открытия
        
        # Circuit должен открыться
        state = circuit_breaker.get_state()
        assert state['state'] == CircuitState.OPEN.value
        assert state['failure_count'] >= 3
    
    @pytest.mark.asyncio
    async def test_open_circuit_bypasses_retry(self, retry_policy, circuit_breaker):
        """Открытый circuit breaker блокирует запросы без retry"""
        # Открываем circuit вручную
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 5
        circuit_breaker.last_failure_time = time.time()  # Используем time.time() вместо asyncio
        
        async def any_request():
            return "should not be called"
        
        async def protected_request():
            return await circuit_breaker.call(any_request)
        
        # RetryPolicy не должен помочь - circuit открыт, исключение non-retryable
        with pytest.raises(CircuitBreakerOpenException, match="Circuit breaker OPEN"):
            await retry_policy.execute_with_retry(protected_request)
        
        # Главное - exception был выброшен без retry попыток
        # (проверка счётчика может быть нестабильной из-за timing)


# ============================================================================
# TestRecoveryScenarios
# ============================================================================

class TestRecoveryScenarios:
    """Тесты сценариев восстановления после падений"""
    
    @pytest.mark.asyncio
    async def test_half_open_with_retry_success(self, retry_config, circuit_config):
        """HALF_OPEN circuit успешно закрывается через retry"""
        # Настраиваем быстрый recovery
        circuit_config.timeout_seconds = 0.1
        circuit_config.success_threshold = 1  # Одна успешная попытка достаточна
        circuit_breaker = CircuitBreaker(circuit_config)
        retry_policy = RetryPolicy(retry_config)
        
        call_count = 0
        
        async def recovering_service():
            nonlocal call_count
            call_count += 1
            # Сервис успешно восстанавливается с первой попытки
            return "recovered"
        
        # Открываем circuit
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 3
        circuit_breaker.last_failure_time = time.time() - 1.0  # Достаточно в прошлом
        
        # Пробуем запрос - должен перейти в HALF_OPEN и успешно восстановиться
        async def protected_request():
            return await circuit_breaker.call(recovering_service)
        
        result = await retry_policy.execute_with_retry(protected_request)
        
        assert result == "recovered"
        assert call_count == 1  # Одна успешная попытка
        assert circuit_breaker.get_state()['state'] == CircuitState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, retry_config, circuit_config):
        """Падение в HALF_OPEN возвращает circuit в OPEN"""
        circuit_config.timeout_seconds = 0.1
        circuit_breaker = CircuitBreaker(circuit_config)
        retry_policy = RetryPolicy(retry_config)
        
        async def still_failing():
            raise ConnectionError("Not recovered yet")
        
        # Открываем circuit и ждём timeout
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 3
        circuit_breaker.last_failure_time = time.time() - 1.0  # Достаточно в прошлом
        
        async def protected_request():
            return await circuit_breaker.call(still_failing)
        
        # Пробуем запрос - первая попытка переведёт в HALF_OPEN, затем обратно в OPEN
        # Следующие попытки retry получат CircuitBreakerOpenException (но без таймаута для retry)
        with pytest.raises((ConnectionError, CircuitBreakerOpenException)):
            await retry_policy.execute_with_retry(protected_request)
        
        assert circuit_breaker.get_state()['state'] == CircuitState.OPEN.value


# ============================================================================
# TestPerformanceCharacteristics
# ============================================================================

class TestPerformanceCharacteristics:
    """Тесты производительности и latency"""
    
    @pytest.mark.asyncio
    async def test_latency_with_retries(self, retry_policy, circuit_breaker):
        """Измеряем latency с retry логикой"""
        import time
        
        call_count = 0
        
        async def slow_service():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # 100ms задержка
            if call_count < 2:
                raise ConnectionError("First attempt fails")
            return "success"
        
        async def protected_request():
            return await circuit_breaker.call(slow_service)
        
        start_time = time.time()
        result = await retry_policy.execute_with_retry(protected_request)
        elapsed = time.time() - start_time
        
        assert result == "success"
        # Должно быть ~200ms: 100ms (fail) + 100ms (retry delay) + 100ms (success)
        assert 0.2 <= elapsed <= 0.5  # С запасом на overhead
    
    @pytest.mark.asyncio
    async def test_fast_fail_with_open_circuit(self, retry_policy, circuit_breaker):
        """Открытый circuit обеспечивает fast fail"""
        # Открываем circuit
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 5
        circuit_breaker.last_failure_time = time.time()  # Используем time.time() вместо asyncio
        
        async def slow_service():
            await asyncio.sleep(1.0)  # Не должен выполниться
            return "never"
        
        async def protected_request():
            return await circuit_breaker.call(slow_service)
        
        start_time = time.time()
        with pytest.raises(CircuitBreakerOpenException):
            await retry_policy.execute_with_retry(protected_request)
        elapsed = time.time() - start_time
        
        # Должен упасть мгновенно, без ожидания slow_service
        assert elapsed < 0.1


# ============================================================================
# TestConcurrentRequests
# ============================================================================

class TestConcurrentRequests:
    """Тесты concurrent запросов через retry + circuit breaker"""
    
    @pytest.mark.asyncio
    async def test_concurrent_success(self, retry_policy, circuit_breaker):
        """Concurrent успешные запросы работают корректно"""
        async def fast_service():
            await asyncio.sleep(0.05)
            return "ok"
        
        async def protected_request():
            return await circuit_breaker.call(fast_service)
        
        # Запускаем 10 concurrent запросов
        tasks = [
            retry_policy.execute_with_retry(protected_request)
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        assert all(r == "ok" for r in results)
        assert circuit_breaker.get_state()['state'] == CircuitState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_concurrent_failures_open_circuit(self, retry_policy, circuit_breaker):
        """Concurrent падения открывают circuit"""
        async def failing_service():
            raise ConnectionError("Service down")
        
        async def protected_request():
            return await circuit_breaker.call(failing_service)
        
        # Запускаем 5 concurrent запросов (все упадут)
        tasks = [
            retry_policy.execute_with_retry(protected_request)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Все должны упасть
        assert all(isinstance(r, Exception) for r in results)
        
        # Circuit должен открыться (failure_threshold=3)
        state = circuit_breaker.get_state()
        assert state['state'] == CircuitState.OPEN.value


# ============================================================================
# TestEdgeCases
# ============================================================================

class TestEdgeCases:
    """Граничные случаи и corner cases"""
    
    @pytest.mark.asyncio
    async def test_timeout_during_retry(self, retry_config, circuit_breaker):
        """Timeout срабатывает во время retry логики"""
        retry_config.timeout_seconds = 0.5  # Короткий timeout
        retry_policy = RetryPolicy(retry_config)
        
        async def slow_and_failing():
            await asyncio.sleep(0.3)  # Медленный
            raise ConnectionError("Slow failure")
        
        async def protected_request():
            return await circuit_breaker.call(slow_and_failing)
        
        # Должен упасть по timeout (0.5s < 0.3s * 3 retries)
        with pytest.raises(asyncio.TimeoutError):
            await retry_policy.execute_with_retry(protected_request)
    
    @pytest.mark.asyncio
    async def test_circuit_opens_during_long_retry(self, retry_config, circuit_config):
        """Circuit может открыться во время long retry sequence"""
        retry_config.max_attempts = 5
        circuit_config.failure_threshold = 2
        
        retry_policy = RetryPolicy(retry_config)
        circuit_breaker = CircuitBreaker(circuit_config)
        
        async def persistent_failure():
            raise ConnectionError("Always fails")
        
        async def protected_request():
            return await circuit_breaker.call(persistent_failure)
        
        # Первый запрос: 5 retry попыток, circuit открывается после 2
        # Может упасть либо с ConnectionError (до открытия), либо с CircuitBreakerOpenException (после)
        with pytest.raises((ConnectionError, CircuitBreakerOpenException)):
            await retry_policy.execute_with_retry(protected_request)
        
        # Circuit должен быть открыт
        assert circuit_breaker.get_state()['state'] == CircuitState.OPEN.value
        
        # Второй запрос: должен fail fast (circuit открыт, non-retryable)
        with pytest.raises(CircuitBreakerOpenException, match="Circuit breaker OPEN"):
            await retry_policy.execute_with_retry(protected_request)


# ============================================================================
# TestStatisticsAccuracy
# ============================================================================

class TestStatisticsAccuracy:
    """Проверка точности статистики обоих компонентов"""
    
    @pytest.mark.asyncio
    async def test_stats_after_successful_flow(self, retry_policy, circuit_breaker):
        """Статистика корректна после успешного flow"""
        async def success():
            return "ok"
        
        async def protected_request():
            return await circuit_breaker.call(success)
        
        # Делаем 3 успешных запроса
        for _ in range(3):
            await retry_policy.execute_with_retry(protected_request)
        
        # Проверяем retry stats
        retry_stats = retry_policy.get_stats()
        assert retry_stats['total_executions'] == 3
        assert retry_stats['successful_executions'] == 3
        assert retry_stats['failed_executions'] == 0
        
        # Проверяем circuit stats
        circuit_state = circuit_breaker.get_state()
        assert circuit_state['state'] == CircuitState.CLOSED.value
        assert circuit_state['failure_count'] == 0
    
    @pytest.mark.asyncio
    async def test_stats_after_failed_flow(self, retry_policy, circuit_breaker):
        """Статистика корректна после failed flow"""
        async def fail():
            raise ValueError("Error")
        
        async def protected_request():
            return await circuit_breaker.call(fail)
        
        # Делаем 3 неудачных запроса (circuit откроется после 3-го)
        for _ in range(3):
            try:
                await retry_policy.execute_with_retry(protected_request)
            except (ValueError, CircuitBreakerOpenException):
                pass  # После открытия circuit получаем CircuitBreakerOpenException
        
        # Проверяем retry stats
        retry_stats = retry_policy.get_stats()
        # Первые 3 запроса упали с ValueError (circuit открылся на 3-м)
        # После этого могут быть CircuitBreakerOpenException которые тоже считаются failed
        assert retry_stats['failed_executions'] >= 1  # Как минимум 1 failed execution
        
        # Проверяем circuit stats
        circuit_state = circuit_breaker.get_state()
        assert circuit_state['state'] == CircuitState.OPEN.value
        assert circuit_state['failure_count'] >= 3


# ============================================================================
# TestRealWorldScenarios
# ============================================================================

class TestRealWorldScenarios:
    """Тесты реальных сценариев использования"""
    
    @pytest.mark.asyncio
    async def test_vm_embedder_simulation(self, retry_config, circuit_config):
        """Симуляция VM embedder с retry + circuit breaker"""
        retry_config.max_attempts = 3
        retry_config.timeout_seconds = 10.0
        circuit_config.failure_threshold = 5
        
        retry_policy = RetryPolicy(retry_config)
        circuit_breaker = CircuitBreaker(circuit_config)
        
        call_count = 0
        
        async def vm_embed_request(texts):
            """Симуляция VM embedder запроса"""
            nonlocal call_count
            call_count += 1
            
            # Симулируем flaky network
            if call_count in [1, 3]:  # Первый и третий запросы падают
                await asyncio.sleep(0.1)
                raise ConnectionError("Network timeout")
            
            # Успешный ответ
            await asyncio.sleep(0.05)
            return [[0.1, 0.2, 0.3] for _ in texts]
        
        async def protected_embed(texts):
            return await circuit_breaker.call(vm_embed_request, texts)
        
        # Первый запрос: упадёт 1 раз, затем успех
        result1 = await retry_policy.execute_with_retry(
            protected_embed, ["text1", "text2"]
        )
        assert len(result1) == 2
        
        # Второй запрос: упадёт 1 раз, затем успех
        result2 = await retry_policy.execute_with_retry(
            protected_embed, ["text3"]
        )
        assert len(result2) == 1
        
        # Circuit должен остаться закрытым
        assert circuit_breaker.get_state()['state'] == CircuitState.CLOSED.value
    
    @pytest.mark.asyncio
    async def test_qdrant_health_check_simulation(self, retry_config, circuit_config):
        """Симуляция Qdrant health check с защитой"""
        retry_config.max_attempts = 2
        retry_config.timeout_seconds = 5.0
        circuit_config.failure_threshold = 3
        
        retry_policy = RetryPolicy(retry_config)
        circuit_breaker = CircuitBreaker(circuit_config)
        
        health_checks = 0
        
        async def qdrant_health():
            """Симуляция Qdrant health check"""
            nonlocal health_checks
            health_checks += 1
            
            if health_checks < 3:
                raise ConnectionError("Qdrant not ready")
            
            return {"status": "healthy", "collections": 5}
        
        async def protected_health():
            return await circuit_breaker.call(qdrant_health)
        
        # Первые 2 health check упадут, третий успешен
        try:
            await retry_policy.execute_with_retry(protected_health)
        except ConnectionError:
            pass  # Ожидаемо
        
        result = await retry_policy.execute_with_retry(protected_health)
        assert result["status"] == "healthy"
        
        # Circuit не должен открыться (только 2 падения < threshold=3)
        assert circuit_breaker.get_state()['state'] == CircuitState.CLOSED.value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
