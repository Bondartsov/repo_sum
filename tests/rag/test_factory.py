"""
Unit тесты для rag/factory.py - Factory Pattern для RAG компонентов.
"""

import os
import pytest
from unittest.mock import patch
from rag.factory import RAGFactory
from rag.context import ExecutionContext
from config import get_config


class TestRAGFactory:
    """Тесты для RAGFactory"""
    
    def setup_method(self):
        """Сброс состояния Factory перед каждым тестом"""
        RAGFactory.reset_context()
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        RAGFactory.reset_context()
    
    def test_set_and_get_context(self):
        """Тест: Явная установка и получение контекста"""
        # Устанавливаем VM контекст
        RAGFactory.set_context(ExecutionContext.VM)
        context = RAGFactory.get_context()
        assert context == ExecutionContext.VM
        
        # Сбрасываем и устанавливаем CLIENT контекст
        RAGFactory.reset_context()
        RAGFactory.set_context(ExecutionContext.CLIENT)
        context = RAGFactory.get_context()
        assert context == ExecutionContext.CLIENT
    
    def test_context_caching(self):
        """Тест: Контекст кэшируется после первого определения"""
        # Первый вызов - определяет контекст
        context1 = RAGFactory.get_context()
        
        # Второй вызов - должен вернуть кэшированное значение
        context2 = RAGFactory.get_context()
        
        assert context1 == context2
        assert RAGFactory._context is not None
    
    def test_reset_context(self):
        """Тест: Сброс контекста"""
        RAGFactory.set_context(ExecutionContext.VM)
        assert RAGFactory._context == ExecutionContext.VM
        
        RAGFactory.reset_context()
        assert RAGFactory._context is None
    
    @pytest.mark.parametrize("mock_env", [
        pytest.param({}, id="no_env"),
    ])
    def test_create_vector_store_vm_context(self, mock_env):
        """Тест: Создание локального QdrantVectorStore в VM контексте"""
        RAGFactory.set_context(ExecutionContext.VM)
        
        config = get_config()
        
        with patch.dict(os.environ, mock_env, clear=False):
            vector_store = RAGFactory.create_vector_store(config)
            
            # Должен быть создан локальный QdrantVectorStore
            assert type(vector_store).__name__ == 'QdrantVectorStore'
            assert type(vector_store).__module__ == 'rag.vector_store'
            
            # У локального не должно быть HTTP endpoints
            assert not hasattr(vector_store, 'search_endpoint')
            assert not hasattr(vector_store, 'index_endpoint')
    
    def test_create_vector_store_client_context(self):
        """Тест: Создание RemoteVMVectorStore в CLIENT контексте"""
        RAGFactory.set_context(ExecutionContext.CLIENT)
        
        config = get_config()
        vector_store = RAGFactory.create_vector_store(config)
        
        # Должен быть создан удалённый RemoteVMVectorStore
        assert type(vector_store).__name__ == 'RemoteVMVectorStore'
        assert type(vector_store).__module__ == 'rag.remote_vector_store'
        
        # У удалённого должны быть HTTP endpoints
        assert hasattr(vector_store, 'search_endpoint')
        assert hasattr(vector_store, 'index_endpoint')
    
    def test_create_embedder_vm_context(self):
        """Тест: Создание локального CPUEmbedder в VM контексте"""
        RAGFactory.set_context(ExecutionContext.VM)
        
        config = get_config()
        embedder = RAGFactory.create_embedder(config)
        
        # Должен быть создан локальный CPUEmbedder
        assert type(embedder).__name__ == 'CPUEmbedder'
        assert type(embedder).__module__ == 'rag.embedder'
    
    def test_create_embedder_client_context(self):
        """Тест: Создание RemoteVMEmbedder в CLIENT контексте"""
        RAGFactory.set_context(ExecutionContext.CLIENT)
        
        config = get_config()
        embedder = RAGFactory.create_embedder(config)
        
        # Должен быть создан удалённый RemoteVMEmbedder
        assert type(embedder).__name__ == 'RemoteVMEmbedder'
        assert type(embedder).__module__ == 'rag.remote_embedder'
    
    def test_get_factory_info_vm(self):
        """Тест: Информация о Factory в VM контексте"""
        RAGFactory.set_context(ExecutionContext.VM)
        
        info = RAGFactory.get_factory_info()
        
        assert info['current_context'] == 'vm'
        assert info['context_cached'] is True
        assert info['context_override'] is True
        assert info['expected_embedder'] == 'CPUEmbedder'
        assert info['expected_vector_store'] == 'QdrantVectorStore'
    
    def test_get_factory_info_client(self):
        """Тест: Информация о Factory в CLIENT контексте"""
        RAGFactory.set_context(ExecutionContext.CLIENT)
        
        info = RAGFactory.get_factory_info()
        
        assert info['current_context'] == 'client'
        assert info['context_cached'] is True
        assert info['expected_embedder'] == 'RemoteVMEmbedder'
        assert info['expected_vector_store'] == 'RemoteVMVectorStore'
    
    def test_no_recursion_in_vm_context(self):
        """
        КРИТИЧЕСКИЙ ТЕСТ: Проверка что в VM контексте невозможна рекурсия.
        
        Проверяет что:
        1. В VM контексте создаётся локальный QdrantVectorStore
        2. У него нет HTTP endpoints (search_endpoint, index_endpoint)
        3. Следовательно, он не может отправить HTTP запрос обратно на VM
        """
        RAGFactory.set_context(ExecutionContext.VM)
        
        config = get_config()
        vector_store = RAGFactory.create_vector_store(config)
        
        # Проверяем что это локальная версия
        assert type(vector_store).__name__ == 'QdrantVectorStore'
        assert type(vector_store).__module__ == 'rag.vector_store'
        
        # КЛЮЧЕВАЯ ПРОВЕРКА: Нет HTTP клиента -> нет рекурсии
        assert not hasattr(vector_store, 'search_endpoint'), \
            "Локальный QdrantVectorStore НЕ должен иметь search_endpoint"
        assert not hasattr(vector_store, 'index_endpoint'), \
            "Локальный QdrantVectorStore НЕ должен иметь index_endpoint"
        assert not hasattr(vector_store, 'service_host'), \
            "Локальный QdrantVectorStore НЕ должен иметь service_host"
        
        # Проверяем что есть прямой Qdrant клиент
        assert hasattr(vector_store, 'active_client'), \
            "Локальный QdrantVectorStore должен иметь active_client (прямое подключение)"


class TestFactoryIntegration:
    """Интеграционные тесты для Factory Pattern"""
    
    def setup_method(self):
        """Сброс состояния перед каждым тестом"""
        RAGFactory.reset_context()
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        RAGFactory.reset_context()
    
    def test_factory_creates_compatible_components(self):
        """Тест: Factory создаёт совместимые компоненты"""
        config = get_config()
        
        # В любом контексте компоненты должны быть совместимы
        for context in [ExecutionContext.VM, ExecutionContext.CLIENT]:
            RAGFactory.set_context(context)
            
            embedder = RAGFactory.create_embedder(config)
            vector_store = RAGFactory.create_vector_store(config)
            
            # Оба должны иметь необходимые методы
            assert hasattr(embedder, 'embed_texts')
            assert hasattr(vector_store, 'search')
            assert hasattr(vector_store, 'index_documents')
            
            RAGFactory.reset_context()
    
    def test_multiple_component_creation(self):
        """Тест: Множественное создание компонентов использует один контекст"""
        RAGFactory.set_context(ExecutionContext.VM)
        
        config = get_config()
        
        # Создаём несколько компонентов
        vs1 = RAGFactory.create_vector_store(config)
        emb1 = RAGFactory.create_embedder(config)
        vs2 = RAGFactory.create_vector_store(config)
        
        # Все должны быть локальными версиями
        assert type(vs1).__name__ == 'QdrantVectorStore'
        assert type(emb1).__name__ == 'CPUEmbedder'
        assert type(vs2).__name__ == 'QdrantVectorStore'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])