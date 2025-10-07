"""
Unit тесты для rag/context.py - детекция контекста выполнения.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from rag.context import ExecutionContext, detect_execution_context, get_context_info


class TestExecutionContext:
    """Тесты для детекции контекста выполнения"""
    
    def test_explicit_vm_context_via_env(self):
        """Тест: Явное указание VM контекста через переменную окружения"""
        with patch.dict(os.environ, {'RAG_EXECUTION_CONTEXT': 'vm'}):
            context = detect_execution_context()
            assert context == ExecutionContext.VM
    
    def test_explicit_client_context_via_env(self):
        """Тест: Явное указание CLIENT контекста через переменную окружения"""
        with patch.dict(os.environ, {'RAG_EXECUTION_CONTEXT': 'client'}):
            context = detect_execution_context()
            assert context == ExecutionContext.CLIENT
    
    def test_vm_context_via_hostname(self):
        """Тест: Определение VM контекста через hostname"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('socket.gethostname', return_value='vm-server-01'):
                context = detect_execution_context()
                assert context == ExecutionContext.VM
    
    def test_vm_context_via_qdrant_port(self):
        """Тест: Определение VM контекста через доступность Qdrant порта"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('socket.gethostname', return_value='client-pc'):
                # Мокируем успешное подключение к порту 6333
                mock_socket = MagicMock()
                mock_socket.connect_ex.return_value = 0  # Успешное подключение
                
                with patch('socket.socket', return_value=mock_socket):
                    context = detect_execution_context()
                    assert context == ExecutionContext.VM
    
    def test_client_context_default(self):
        """Тест: CLIENT контекст по умолчанию"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('socket.gethostname', return_value='client-pc'):
                # Мокируем неуспешное подключение к порту 6333
                mock_socket = MagicMock()
                mock_socket.connect_ex.return_value = 1  # Подключение не удалось
                
                with patch('socket.socket', return_value=mock_socket):
                    context = detect_execution_context()
                    assert context == ExecutionContext.CLIENT
    
    def test_vm_context_via_directories(self):
        """Тест: Определение VM контекста через VM-специфичные директории"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('socket.gethostname', return_value='client-pc'):
                mock_socket = MagicMock()
                mock_socket.connect_ex.return_value = 1
                
                with patch('socket.socket', return_value=mock_socket):
                    with patch('os.path.exists', return_value=True):
                        context = detect_execution_context()
                        assert context == ExecutionContext.VM
    
    def test_get_context_info(self):
        """Тест: Получение информации о контексте"""
        info = get_context_info()
        
        assert 'context' in info
        assert 'env_variable' in info
        assert 'hostname' in info
        assert 'qdrant_local_available' in info
        assert 'vm_directories_exist' in info
        assert 'detection_method' in info
        
        # Контекст должен быть одним из допустимых значений
        assert info['context'] in ('vm', 'client', 'unknown')
    
    def test_env_variable_priority(self):
        """Тест: Переменная окружения имеет наивысший приоритет"""
        with patch.dict(os.environ, {'RAG_EXECUTION_CONTEXT': 'client'}):
            # Даже если hostname указывает на VM
            with patch('socket.gethostname', return_value='vm-server'):
                context = detect_execution_context()
                # Env переменная имеет приоритет
                assert context == ExecutionContext.CLIENT
    
    def test_case_insensitive_env(self):
        """Тест: Переменная окружения case-insensitive"""
        test_values = ['VM', 'Vm', 'vm', 'CLIENT', 'Client', 'client']
        
        for value in test_values:
            with patch.dict(os.environ, {'RAG_EXECUTION_CONTEXT': value}):
                context = detect_execution_context()
                expected = ExecutionContext.VM if value.lower() == 'vm' else ExecutionContext.CLIENT
                assert context == expected


class TestContextInfo:
    """Тесты для get_context_info()"""
    
    def test_context_info_structure(self):
        """Тест: Структура возвращаемой информации"""
        info = get_context_info()
        
        required_keys = [
            'context',
            'env_variable',
            'hostname',
            'qdrant_local_available',
            'vm_directories_exist',
            'detection_method'
        ]
        
        for key in required_keys:
            assert key in info, f"Отсутствует ключ {key} в context_info"
    
    def test_detection_method_values(self):
        """Тест: Метод детекции возвращает корректные значения"""
        info = get_context_info()
        
        valid_methods = [
            'environment_variable',
            'hostname',
            'qdrant_port_check',
            'vm_directories',
            'default_fallback'
        ]
        
        assert info['detection_method'] in valid_methods


if __name__ == '__main__':
    pytest.main([__file__, '-v'])