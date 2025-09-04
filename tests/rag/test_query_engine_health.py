"""
Unit тесты для исправленной health_check функциональности CPUQueryEngine.

Проверяет корректность исправления критических багов:
1. _check_vector_store() - использование health_check() вместо is_connected()
2. _ensure_embeddings() - использование config.vector_store.vector_size вместо embedding_dim
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from rag.query_engine import CPUQueryEngine, SearchResult
from config import QueryEngineConfig, VectorStoreConfig, EmbeddingConfig
from rag.embedder import CPUEmbedder
from rag.vector_store import QdrantVectorStore


@pytest.mark.unit
class TestQueryEngineHealthFixes:
    """Unit тесты для исправленной функциональности CPUQueryEngine"""
    
    def create_test_config(self) -> QueryEngineConfig:
        """Создает тестовую конфигурацию с правильной размерностью вектора"""
        config = QueryEngineConfig()
        config.vector_store = VectorStoreConfig()
        config.vector_store.vector_size = 384  # Размерность для тестов
        return config
    
    @pytest.mark.asyncio
    async def test_check_vector_store_success(self):
        """Тестирует успешную проверку vector_store через health_check"""
        # Настраиваем mock компоненты
        mock_embedder = Mock(spec=CPUEmbedder)
        mock_vector_store = AsyncMock(spec=QdrantVectorStore)
        mock_vector_store.health_check.return_value = {
            "status": "connected", 
            "timestamp": "2025-09-04T19:17:00Z",
            "collection_status": "exists"
        }
        
        mock_config = self.create_test_config()
        
        # Создаем QueryEngine с mock SearchService
        with patch('rag.query_engine.SearchService'):
            query_engine = CPUQueryEngine(mock_embedder, mock_vector_store, mock_config)
            
            # Тестируем метод _check_vector_store
            result = await query_engine._check_vector_store()
            
            # Проверяем результат
            assert result is True
            mock_vector_store.health_check.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_check_vector_store_disconnected(self):
        """Тестирует случай когда vector_store отключен"""
        mock_embedder = Mock(spec=CPUEmbedder)
        mock_vector_store = AsyncMock(spec=QdrantVectorStore)
        mock_vector_store.health_check.return_value = {
            "status": "error", 
            "error": "Connection failed",
            "collection_status": "not_found"
        }
        
        mock_config = self.create_test_config()
        
        with patch('rag.query_engine.SearchService'):
            query_engine = CPUQueryEngine(mock_embedder, mock_vector_store, mock_config)
            
            result = await query_engine._check_vector_store()
            
            assert result is False
            mock_vector_store.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_vector_store_exception(self):
        """Тестирует обработку исключения при health_check"""
        mock_embedder = Mock(spec=CPUEmbedder)
        mock_vector_store = AsyncMock(spec=QdrantVectorStore)
        mock_vector_store.health_check.side_effect = Exception("Network error")
        
        mock_config = self.create_test_config()
        
        with patch('rag.query_engine.SearchService'):
            query_engine = CPUQueryEngine(mock_embedder, mock_vector_store, mock_config)
            
            result = await query_engine._check_vector_store()
            
            assert result is False
            mock_vector_store.health_check.assert_called_once()
    
    def test_ensure_embeddings_uses_config_dimension(self):
        """Тестирует что _ensure_embeddings использует правильную размерность из конфигурации"""
        # Настраиваем mock эмбеддер для провокации fallback'а
        mock_embedder = Mock(spec=CPUEmbedder)
        mock_embedder.embed_texts.side_effect = Exception("Embedding failed")  # Принудительный fallback
        
        mock_vector_store = Mock(spec=QdrantVectorStore)
        
        # Создаем конфигурацию с явно указанной размерностью
        mock_config = self.create_test_config()
        mock_config.vector_store.vector_size = 384  # Тестовая размерность
        
        with patch('rag.query_engine.SearchService'):
            query_engine = CPUQueryEngine(mock_embedder, mock_vector_store, mock_config)
            
            # Создаем тестовые результаты без эмбеддингов
            test_results = [
                SearchResult(
                    chunk_id="test_1",
                    file_path="test.py", 
                    file_name="test.py",
                    chunk_name="test_func",
                    chunk_type="function",
                    language="python",
                    start_line=1,
                    end_line=5,
                    score=0.9,
                    content="def test_func(): pass",
                    metadata={},
                    embedding=None  # Нет эмбеддинга - будет fallback
                ),
                SearchResult(
                    chunk_id="test_2",
                    file_path="test2.py",
                    file_name="test2.py", 
                    chunk_name="another_func",
                    chunk_type="function",
                    language="python",
                    start_line=10,
                    end_line=15,
                    score=0.8,
                    content="def another_func(): return True",
                    metadata={},
                    embedding=None  # Нет эмбеддинга - будет fallback
                )
            ]
            
            # Вызываем метод
            query_engine._ensure_embeddings(test_results)
            
            # Проверяем что fallback эмбеддинги имеют правильную размерность
            assert test_results[0].embedding is not None
            assert len(test_results[0].embedding) == 384  # Из конфигурации
            
            assert test_results[1].embedding is not None
            assert len(test_results[1].embedding) == 384  # Из конфигурации
            
            # Проверяем что метод embed_texts был вызван (и упал)
            mock_embedder.embed_texts.assert_called_once_with([
                "def test_func(): pass", 
                "def another_func(): return True"
            ])
    
    def test_ensure_embeddings_fallback_default_dimension(self):
        """Тестирует fallback на размерность по умолчанию при отсутствии конфигурации"""
        mock_embedder = Mock(spec=CPUEmbedder)
        mock_embedder.embed_texts.side_effect = Exception("Embedding failed")
        
        mock_vector_store = Mock(spec=QdrantVectorStore)
        
        # Создаем конфигурацию БЕЗ vector_store (для тестирования fallback)
        mock_config = QueryEngineConfig()
        
        with patch('rag.query_engine.SearchService'):
            query_engine = CPUQueryEngine(mock_embedder, mock_vector_store, mock_config)
            
            test_results = [
                SearchResult(
                    chunk_id="test_fallback",
                    file_path="test_fallback.py",
                    file_name="test_fallback.py",
                    chunk_name="fallback_func", 
                    chunk_type="function",
                    language="python",
                    start_line=1,
                    end_line=3,
                    score=0.7,
                    content="def fallback_func(): pass",
                    metadata={},
                    embedding=None
                )
            ]
            
            # Вызываем метод
            query_engine._ensure_embeddings(test_results)
            
            # Проверяем что используется размерность по умолчанию (384)
            assert test_results[0].embedding is not None
            assert len(test_results[0].embedding) == 384  # Default размерность
    
    def test_ensure_embeddings_successful_case(self):
        """Тестирует успешный случай получения эмбеддингов без fallback"""
        # Настраиваем mock эмбеддер для успешной работы
        mock_embedder = Mock(spec=CPUEmbedder)
        test_embeddings = np.array([
            np.random.random(384),  # Первый эмбеддинг
            np.random.random(384)   # Второй эмбеддинг
        ])
        mock_embedder.embed_texts.return_value = test_embeddings
        
        mock_vector_store = Mock(spec=QdrantVectorStore)
        mock_config = self.create_test_config()
        
        with patch('rag.query_engine.SearchService'):
            query_engine = CPUQueryEngine(mock_embedder, mock_vector_store, mock_config)
            
            test_results = [
                SearchResult(
                    chunk_id="success_1",
                    file_path="success1.py",
                    file_name="success1.py",
                    chunk_name="success_func1",
                    chunk_type="function",
                    language="python",
                    start_line=1,
                    end_line=5,
                    score=0.95,
                    content="def success_func1(): return 'success'",
                    metadata={},
                    embedding=None  # Будет заполнен реальным эмбеддингом
                ),
                SearchResult(
                    chunk_id="success_2",
                    file_path="success2.py",
                    file_name="success2.py",
                    chunk_name="success_func2",
                    chunk_type="function",
                    language="python",
                    start_line=10,
                    end_line=15,
                    score=0.85,
                    content="def success_func2(): return True",
                    metadata={},
                    embedding=None  # Будет заполнен реальным эмбеддингом
                )
            ]
            
            # Вызываем метод
            query_engine._ensure_embeddings(test_results)
            
            # Проверяем что эмбеддинги назначены корректно
            assert test_results[0].embedding is not None
            assert len(test_results[0].embedding) == 384
            np.testing.assert_array_equal(test_results[0].embedding, test_embeddings[0])
            
            assert test_results[1].embedding is not None
            assert len(test_results[1].embedding) == 384
            np.testing.assert_array_equal(test_results[1].embedding, test_embeddings[1])
            
            # Проверяем что embed_texts был вызван с правильными аргументами
            mock_embedder.embed_texts.assert_called_once_with([
                "def success_func1(): return 'success'",
                "def success_func2(): return True"
            ])
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self):
        """Интеграционный тест health_check метода с исправленным _check_vector_store"""
        mock_embedder = Mock(spec=CPUEmbedder)
        mock_vector_store = AsyncMock(spec=QdrantVectorStore)
        mock_vector_store.health_check.return_value = {
            "status": "connected",
            "timestamp": "2025-09-04T19:17:00Z"
        }
        
        mock_config = self.create_test_config()
        
        with patch('rag.query_engine.SearchService'):
            query_engine = CPUQueryEngine(mock_embedder, mock_vector_store, mock_config)
            
            # Вызываем health_check
            health_result = await query_engine.health_check()
            
            # Проверяем структуру ответа
            assert 'status' in health_result
            assert 'embedder_status' in health_result
            assert 'vector_store_status' in health_result
            assert 'stats' in health_result
            
            # Проверяем что статус здоровый
            assert health_result['status'] == 'healthy'
            assert health_result['embedder_status'] == 'ok'
            assert health_result['vector_store_status'] == 'ok'
            
            # Проверяем что _check_vector_store был вызван через health_check()
            mock_vector_store.health_check.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
