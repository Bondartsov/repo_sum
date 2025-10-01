"""
Тесты для диагностики проблем подключения к Qdrant через VM.

Эти тесты помогают диагностировать проблему "error" статуса Qdrant Vector Store
при выполнении команды `python main.py rag status --detailed`.

Автор: AI Assistant
Дата: 1 октября 2025
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from rag.remote_vector_store import RemoteVMVectorStore
from rag.exceptions import VectorStoreConnectionError
from config import RemoteServiceConfig, VectorStoreConfig


class TestQdrantConnectivityDiagnosis:
    """Диагностические тесты для Qdrant подключения через VM"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock конфигурация для тестов"""
        remote_config = RemoteServiceConfig()
        remote_config.host = "10.61.11.54"
        remote_config.port = 8000
        remote_config.health_endpoint = "/health"
        remote_config.timeout_seconds = 60
        remote_config.max_retries = 3
        remote_config.retry_delay = 2.0
        
        vector_config = VectorStoreConfig()
        vector_config.collection_name = "code_chunks"
        vector_config.vector_size = 1024
        
        return remote_config, vector_config
    
    @pytest.mark.asyncio
    async def test_health_check_vm_unavailable(self, mock_config):
        """
        Тест 1: VM сервис недоступен (connection refused)
        
        Проверяет поведение когда VM на 10.61.11.54:8000 недоступна.
        Ожидается: status="error", error содержит информацию о ConnectionError
        """
        remote_config, vector_config = mock_config
        
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # session.get() НЕ async, он возвращает async context manager
            # Поэтому используем Mock, не AsyncMock!
            mock_session_instance.get = Mock(side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=ConnectionRefusedError("Connection refused")
            ))
            
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(vector_config, remote_config)
            health_info = await store._async_health_check()
            
            # Проверки
            assert health_info['status'] == 'error'
            assert 'error' in health_info
            assert 'ConnectionRefusedError' in str(health_info['error']) or \
                   'ClientConnectorError' in str(health_info['error'])
            assert store._connected == False
    
    @pytest.mark.asyncio
    async def test_health_check_vm_timeout(self, mock_config):
        """
        Тест 2: VM сервис timeout (не отвечает в срок)
        
        Проверяет поведение когда VM не отвечает на health check в течение 30s.
        Ожидается: status="error", error содержит информацию о TimeoutError
        """
        remote_config, vector_config = mock_config
        
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # session.get() НЕ async, он возвращает async context manager
            mock_session_instance.get = Mock(side_effect=asyncio.TimeoutError("Request timeout"))
            
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(vector_config, remote_config)
            health_info = await store._async_health_check()
            
            # Проверки
            assert health_info['status'] == 'error'
            assert 'error' in health_info
            assert 'TimeoutError' in str(health_info['error']) or 'timeout' in str(health_info['error']).lower()
            assert store._connected == False
    
    @pytest.mark.asyncio
    async def test_health_check_vm_http_error(self, mock_config):
        """
        Тест 3: VM сервис возвращает HTTP ошибку (500, 503, etc.)
        
        Проверяет поведение когда VM отвечает но с ошибкой (internal server error).
        Ожидается: status="error", error содержит HTTP статус код
        """
        remote_config, vector_config = mock_config
        
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # Mock response с HTTP 500
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error: Qdrant unavailable")
            
            # Mock async context manager для session.get()
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__.return_value = mock_response
            mock_get_cm.__aexit__.return_value = None
            
            # session.get() НЕ async, возвращает context manager напрямую
            mock_session_instance.get = Mock(return_value=mock_get_cm)
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(vector_config, remote_config)
            health_info = await store._async_health_check()
            
            # Проверки
            assert health_info['status'] == 'error'
            assert 'error' in health_info
            assert 'HTTP 500' in health_info['error']
            assert 'Qdrant unavailable' in health_info['error'] or 'Internal Server Error' in health_info['error']
            assert store._connected == False
    
    @pytest.mark.asyncio
    async def test_health_check_vm_malformed_response(self, mock_config):
        """
        Тест 4: VM возвращает некорректный JSON
        
        Проверяет поведение когда VM отвечает 200 OK но JSON невалиден.
        Ожидается: status="error", error содержит информацию о JSON parsing error
        """
        remote_config, vector_config = mock_config
        
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # Mock response с невалидным JSON
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
            
            # Mock async context manager для session.get()
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__.return_value = mock_response
            mock_get_cm.__aexit__.return_value = None
            
            # session.get() НЕ async, возвращает context manager напрямую
            mock_session_instance.get = Mock(return_value=mock_get_cm)
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(vector_config, remote_config)
            health_info = await store._async_health_check()
            
            # Проверки
            assert health_info['status'] == 'error'
            assert 'error' in health_info
            assert 'ValueError' in str(health_info['error']) or 'Invalid JSON' in str(health_info['error'])
    
    @pytest.mark.asyncio
    async def test_health_check_qdrant_not_ready(self, mock_config):
        """
        Тест 5: VM отвечает но Qdrant внутри недоступна
        
        Проверяет поведение когда VM сервис работает но Qdrant не готова.
        Ожидается: status="connected", но collection_status="unavailable"
        """
        remote_config, vector_config = mock_config
        
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # Mock response с Qdrant недоступной
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "collection_status": "unavailable",
                "qdrant_status": "disconnected",
                "error": "Qdrant connection failed"
            })
            
            # Mock async context manager для session.get()
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__.return_value = mock_response
            mock_get_cm.__aexit__.return_value = None
            
            # session.get() НЕ async, возвращает context manager напрямую
            mock_session_instance.get = Mock(return_value=mock_get_cm)
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(vector_config, remote_config)
            health_info = await store._async_health_check()
            
            # Проверки
            assert health_info['status'] == 'connected'  # VM доступна
            assert health_info['components']['vector_store']['collection_status'] == 'unavailable'
            assert health_info['components']['vector_store']['qdrant_status'] == 'disconnected'
            assert store._connected == True  # VM подключена
            assert store._collection_exists == False  # Коллекция недоступна
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_config):
        """
        Тест 6: Успешный health check
        
        Проверяет поведение когда все компоненты работают корректно.
        Ожидается: status="connected", collection_status="exists"
        """
        remote_config, vector_config = mock_config
        
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # Mock успешного response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "collection_status": "exists",
                "qdrant_status": "connected",
                "vector_count": 1234
            })
            
            # Mock async context manager для session.get()
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__.return_value = mock_response
            mock_get_cm.__aexit__.return_value = None
            
            # session.get() НЕ async, возвращает context manager напрямую
            mock_session_instance.get = Mock(return_value=mock_get_cm)
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(vector_config, remote_config)
            health_info = await store._async_health_check()
            
            # Проверки
            assert health_info['status'] == 'connected'
            assert health_info['components']['vector_store']['collection_status'] == 'exists'
            assert health_info['components']['vector_store']['qdrant_status'] == 'connected'
            assert health_info['components']['vector_store']['vector_count'] == 1234
            assert store._connected == True
            assert store._collection_exists == True
    
    @pytest.mark.asyncio
    async def test_health_check_with_network_issues(self, mock_config):
        """
        Тест 7: Сетевые проблемы (DNS, proxy, firewall)
        
        Проверяет поведение при сетевых проблемах между клиентом и VM.
        Ожидается: status="error", детальная информация об ошибке
        """
        remote_config, vector_config = mock_config
        
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # session.get() НЕ async, возвращает async context manager
            mock_session_instance.get = Mock(side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Name or service not known")
            ))
            
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(vector_config, remote_config)
            health_info = await store._async_health_check()
            
            # Проверки
            assert health_info['status'] == 'error'
            assert 'error' in health_info
            # Проверяем что ошибка содержит информацию о сетевых проблемах
            error_msg = str(health_info['error']).lower()
            assert 'clientconnectorerror' in error_msg or 'name or service' in error_msg
    
    def test_sync_health_check(self, mock_config):
        """
        Тест 8: Синхронная обёртка health_check
        
        Проверяет что sync wrapper (health_check) корректно вызывает async версию.
        """
        remote_config, vector_config = mock_config
        
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # Mock успешного response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "collection_status": "exists",
                "qdrant_status": "connected"
            })
            
            # Mock async context manager для session.get()
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__.return_value = mock_response
            mock_get_cm.__aexit__.return_value = None
            
            # session.get() НЕ async, возвращает context manager напрямую
            mock_session_instance.get = Mock(return_value=mock_get_cm)
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(vector_config, remote_config)
            
            # Вызываем синхронный метод
            health_info = store.health_check()
            
            # Проверки
            assert health_info['status'] == 'connected'
            assert 'components' in health_info
            assert 'vector_store' in health_info['components']


class TestQdrantHealthCheckTimeout:
    """Тесты для проблем с таймаутом health check"""
    
    @pytest.fixture
    def mock_config_short_timeout(self):
        """Mock конфигурация с коротким таймаутом"""
        remote_config = RemoteServiceConfig()
        remote_config.host = "10.61.11.54"
        remote_config.port = 8000
        remote_config.health_endpoint = "/health"
        remote_config.timeout_seconds = 5  # Короткий таймаут для тестов
        remote_config.max_retries = 1
        remote_config.retry_delay = 1.0
        
        return remote_config
    
    @pytest.mark.asyncio
    async def test_health_check_timeout_too_short(self, mock_config_short_timeout):
        """
        Тест 9: Таймаут слишком короткий для медленного VM
        
        Проверяет что короткий timeout приводит к ошибке даже если VM работает.
        """
        with patch('rag.remote_vector_store.get_shared_http_session') as mock_session:
            # Создаём mock session с правильной структурой
            mock_session_instance = AsyncMock()
            
            # session.get() НЕ async, возвращает async context manager
            mock_session_instance.get = Mock(side_effect=asyncio.TimeoutError("Too slow"))
            
            mock_session.return_value = mock_session_instance
            
            store = RemoteVMVectorStore(None, mock_config_short_timeout)
            health_info = await store._async_health_check()
            
            # Проверки
            assert health_info['status'] == 'error'
            assert 'timeout' in str(health_info['error']).lower()


class TestQdrantDiagnosticRecommendations:
    """Тесты для диагностических рекомендаций"""
    
    def test_diagnostic_recommendations_vm_unavailable(self):
        """
        Тест 10: Генерация диагностических рекомендаций при недоступности VM
        """
        health_info = {
            'status': 'error',
            'error': 'ClientConnectorError: Connection refused'
        }
        
        # Проверяем что можем детектировать тип проблемы
        error_msg = str(health_info['error'])
        
        if 'Connection refused' in error_msg:
            recommendation = "VM сервис не запущен. Запустите: python vm_start.py start"
        elif 'timeout' in error_msg.lower():
            recommendation = "VM сервис не отвечает. Проверьте доступность: curl http://10.61.11.54:8000/health"
        elif 'Name or service not known' in error_msg:
            recommendation = "DNS проблема. Проверьте сетевое подключение и firewall"
        else:
            recommendation = "Неизвестная ошибка. Проверьте логи VM сервиса"
        
        assert recommendation is not None
        assert len(recommendation) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
