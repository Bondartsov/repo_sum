"""
Integration тесты для Factory Pattern - проверка устранения рекурсии.

Проверяет полный flow создания компонентов через Factory
и гарантирует отсутствие рекурсии на VM сервере.
"""

import os
import pytest
from unittest.mock import patch
from rag.factory import RAGFactory
from rag.context import ExecutionContext
from rag.indexer_service import IndexerService
from rag.search_service import SearchService
from config import get_config


class TestFactoryIntegrationVMContext:
    """Integration тесты для VM контекста"""
    
    def setup_method(self):
        """Установка VM контекста перед каждым тестом"""
        RAGFactory.set_context(ExecutionContext.VM)
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        RAGFactory.reset_context()
    
    @pytest.mark.parametrize("mock_env", [
        pytest.param({}, id="no_mock"),
    ])
    def test_indexer_service_uses_local_components_on_vm(self, mock_env):
        """
        КРИТИЧЕСКИЙ ТЕСТ: IndexerService на VM использует локальные компоненты.
        
        Проверяет что:
        1. IndexerService создаёт локальный QdrantVectorStore
        2. IndexerService создаёт локальный CPUEmbedder
        3. У компонентов нет HTTP endpoints
        """
        config = get_config()
        
        with patch.dict(os.environ, mock_env, clear=False):
            # Создаём IndexerService (внутри использует Factory)
            indexer = IndexerService(config, silent_mode=True)
            
            # Проверяем что использует локальный vector_store
            assert type(indexer.vector_store).__name__ == 'QdrantVectorStore'
            assert type(indexer.vector_store).__module__ == 'rag.vector_store'
            
            # Проверяем что использует локальный embedder
            assert type(indexer.embedder).__name__ == 'CPUEmbedder'
            assert type(indexer.embedder).__module__ == 'rag.embedder'
            
            # КЛЮЧЕВАЯ ПРОВЕРКА: Нет HTTP endpoints -> нет рекурсии
            assert not hasattr(indexer.vector_store, 'search_endpoint')
            assert not hasattr(indexer.vector_store, 'index_endpoint')
    
    @pytest.mark.parametrize("mock_env", [
        pytest.param({}, id="no_mock"),
    ])
    def test_search_service_uses_local_components_on_vm(self, mock_env):
        """
        Тест: SearchService на VM использует локальные компоненты.
        """
        config = get_config()
        
        with patch.dict(os.environ, mock_env, clear=False):
            # Создаём SearchService (внутри использует Factory)
            search_service = SearchService(config, silent_mode=True)
            
            # Проверяем что использует локальный vector_store
            assert type(search_service.vector_store).__name__ == 'QdrantVectorStore'
            
            # Проверяем что использует локальный embedder
            assert type(search_service.embedder).__name__ == 'CPUEmbedder'


class TestFactoryIntegrationClientContext:
    """Integration тесты для CLIENT контекста"""
    
    def setup_method(self):
        """Установка CLIENT контекста перед каждым тестом"""
        RAGFactory.set_context(ExecutionContext.CLIENT)
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        RAGFactory.reset_context()
    
    @pytest.mark.parametrize("mock_env", [
        pytest.param({}, id="no_mock"),
    ])
    def test_indexer_service_uses_remote_components_on_client(self, mock_env):
        """
        Тест: IndexerService на клиенте использует удалённые компоненты.
        """
        config = get_config()
        
        with patch.dict(os.environ, mock_env, clear=False):
            # Создаём IndexerService
            indexer = IndexerService(config, silent_mode=True)
            
            # Проверяем что использует remote vector_store
            assert type(indexer.vector_store).__name__ == 'RemoteVMVectorStore'
            assert type(indexer.vector_store).__module__ == 'rag.remote_vector_store'
            
            # Проверяем что использует remote embedder
            assert type(indexer.embedder).__name__ == 'RemoteVMEmbedder'
            assert type(indexer.embedder).__module__ == 'rag.remote_embedder'
            
            # Проверяем наличие HTTP endpoints
            assert hasattr(indexer.vector_store, 'search_endpoint')
            assert hasattr(indexer.vector_store, 'index_endpoint')
    
    @pytest.mark.parametrize("mock_env", [
        pytest.param({}, id="no_mock"),
    ])
    def test_search_service_uses_remote_components_on_client(self, mock_env):
        """
        Тест: SearchService на клиенте использует удалённые компоненты.
        """
        config = get_config()
        
        with patch.dict(os.environ, mock_env, clear=False):
            # Создаём SearchService
            search_service = SearchService(config, silent_mode=True)
            
            # Проверяем что использует remote vector_store
            assert type(search_service.vector_store).__name__ == 'RemoteVMVectorStore'
            
            # Проверяем что использует remote embedder
            assert type(search_service.embedder).__name__ == 'RemoteVMEmbedder'


class TestRecursionPrevention:
    """Специальные тесты для проверки устранения рекурсии"""
    
    def setup_method(self):
        """Сброс Factory перед каждым тестом"""
        RAGFactory.reset_context()
    
    def teardown_method(self):
        """Очистка после каждого теста"""
        RAGFactory.reset_context()
    
    def test_vm_indexer_cannot_create_recursion(self):
        """
        КРИТИЧЕСКИЙ ТЕСТ: IndexerService на VM не может создать рекурсию.
        
        Сценарий который вызывал рекурсию ДО исправления:
        1. VM endpoint /index вызывает IndexerService.index_documents()
        2. IndexerService импортирует vector_store из rag.__init__
        3. ДО: Получал RemoteVMVectorStore из-за алиаса -> рекурсия
        4. ПОСЛЕ: Factory выбирает локальный QdrantVectorStore -> рекурсии нет
        """
        # Симулируем VM контекст
        RAGFactory.set_context(ExecutionContext.VM)
        
        config = get_config()
        
        # Создаём IndexerService как это делает vm_rag_service.py
        indexer = IndexerService(config, silent_mode=True)
        
        # Получаем vector_store который будет использоваться для индексации
        vector_store = indexer.vector_store
        
        # КРИТИЧЕСКИЕ ПРОВЕРКИ:
        
        # 1. Это должен быть локальный QdrantVectorStore
        assert type(vector_store).__name__ == 'QdrantVectorStore', \
            f"На VM должен использоваться локальный QdrantVectorStore, получен: {type(vector_store).__name__}"
        
        assert type(vector_store).__module__ == 'rag.vector_store', \
            f"vector_store должен быть из модуля rag.vector_store, получен: {type(vector_store).__module__}"
        
        # 2. У локального QdrantVectorStore НЕТ HTTP endpoints
        assert not hasattr(vector_store, 'search_endpoint'), \
            "Локальный QdrantVectorStore НЕ должен иметь search_endpoint (признак Remote версии)"
        
        assert not hasattr(vector_store, 'index_endpoint'), \
            "Локальный QdrantVectorStore НЕ должен иметь index_endpoint (признак Remote версии)"
        
        assert not hasattr(vector_store, 'service_host'), \
            "Локальный QdrantVectorStore НЕ должен иметь service_host (признак Remote версии)"
        
        # 3. У локального QdrantVectorStore есть прямой клиент Qdrant
        assert hasattr(vector_store, 'active_client'), \
            "Локальный QdrantVectorStore должен иметь active_client (прямое подключение к Qdrant)"
        
        assert hasattr(vector_store, 'http_client'), \
            "Локальный QdrantVectorStore должен иметь http_client"
        
        # 4. Метод index_documents - это прямая индексация в Qdrant, а НЕ HTTP запрос
        import inspect
        source_file = inspect.getfile(vector_store.index_documents)
        assert 'vector_store.py' in source_file, \
            f"index_documents должен быть из rag/vector_store.py, получен: {source_file}"
        
        print("✅ УСПЕХ: Рекурсия НЕВОЗМОЖНА - IndexerService использует локальные компоненты")
    
    def test_client_indexer_uses_remote_correctly(self):
        """
        Тест: IndexerService на клиенте корректно использует remote компоненты.
        
        Проверяет что на клиенте создаются Remote версии с HTTP endpoints.
        """
        # Симулируем CLIENT контекст
        RAGFactory.set_context(ExecutionContext.CLIENT)
        
        config = get_config()
        indexer = IndexerService(config, silent_mode=True)
        
        # Проверяем что использует remote версии
        assert type(indexer.vector_store).__name__ == 'RemoteVMVectorStore'
        assert type(indexer.embedder).__name__ == 'RemoteVMEmbedder'
        
        # Проверяем наличие HTTP endpoints (нормально для клиента)
        assert hasattr(indexer.vector_store, 'search_endpoint')
        assert hasattr(indexer.vector_store, 'index_endpoint')
        
        print("✅ УСПЕХ: Клиент корректно использует remote компоненты")
    
    def test_context_switching_changes_components(self):
        """
        Тест: Переключение контекста меняет создаваемые компоненты.
        
        Проверяет что Factory правильно реагирует на смену контекста.
        """
        config = get_config()
        
        # VM контекст
        RAGFactory.set_context(ExecutionContext.VM)
        vs_vm = RAGFactory.create_vector_store(config)
        assert type(vs_vm).__name__ == 'QdrantVectorStore'
        
        # Переключаемся на CLIENT
        RAGFactory.set_context(ExecutionContext.CLIENT)
        vs_client = RAGFactory.create_vector_store(config)
        assert type(vs_client).__name__ == 'RemoteVMVectorStore'
        
        # Убеждаемся что это разные типы
        assert type(vs_vm) != type(vs_client)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])