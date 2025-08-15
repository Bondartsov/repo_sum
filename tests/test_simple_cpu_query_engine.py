#!/usr/bin/env python3
"""
Упрощенный тест для CPUQueryEngine с прямыми импортами.
"""

import asyncio
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_imports():
    """Тестирует импорты CPUQueryEngine"""
    from rag.embedder import CPUEmbedder
    from rag.vector_store import QdrantVectorStore
    from rag.query_engine import CPUQueryEngine
    from config import get_config
    
    assert CPUEmbedder is not None
    assert QdrantVectorStore is not None
    assert CPUQueryEngine is not None
    assert get_config is not None

def test_config_loading():
    """Тестирует загрузку конфигурации"""
    from config import get_config
    
    config = get_config(require_api_key=False)
    assert config is not None
    assert hasattr(config, 'rag')
    assert hasattr(config.rag, 'embeddings')
    assert hasattr(config.rag, 'query_engine')

def test_embedder_creation():
    """Тестирует создание CPUEmbedder"""
    from rag.embedder import CPUEmbedder
    from config import get_config
    
    config = get_config(require_api_key=False)
    embedder = CPUEmbedder(config.rag.embeddings, config.rag.parallelism)
    assert embedder is not None
    
    stats = embedder.get_stats()
    assert isinstance(stats, dict)
    assert 'provider' in stats

def test_query_engine_creation():
    """Тестирует создание CPUQueryEngine"""
    from rag.embedder import CPUEmbedder
    from rag.query_engine import CPUQueryEngine
    from config import get_config
    
    config = get_config(require_api_key=False)
    embedder = CPUEmbedder(config.rag.embeddings, config.rag.parallelism)
    
    # Создаём заглушку векторного хранилища
    class MockVectorStore:
        def __init__(self):
            self.host = config.rag.vector_store.host
            self.port = config.rag.vector_store.port
            self.collection_name = config.rag.vector_store.collection_name
            
        def is_connected(self):
            return False  # Заглушка
    
    vector_store = MockVectorStore()
    
    # Создаём CPUQueryEngine
    query_engine = CPUQueryEngine(
        embedder=embedder,
        store=vector_store,
        qcfg=config.rag.query_engine
    )
    assert query_engine is not None
    
    # Тестируем базовые методы
    stats = query_engine.get_stats()
    assert isinstance(stats, dict)
    assert 'total_queries' in stats
    
    cache_cleared = query_engine.clear_cache()
    assert isinstance(cache_cleared, int)

async def test_query_engine_async_methods():
    """Тестирует асинхронные методы CPUQueryEngine"""
    from rag.embedder import CPUEmbedder
    from rag.query_engine import CPUQueryEngine
    from config import get_config
    
    config = get_config(require_api_key=False)
    embedder = CPUEmbedder(config.rag.embeddings, config.rag.parallelism)
    
    # Создаём заглушку векторного хранилища
    class MockVectorStore:
        def __init__(self):
            self.host = config.rag.vector_store.host
            self.port = config.rag.vector_store.port
            self.collection_name = config.rag.vector_store.collection_name
            
        def is_connected(self):
            return False  # Заглушка
    
    vector_store = MockVectorStore()
    
    # Создаём CPUQueryEngine
    query_engine = CPUQueryEngine(
        embedder=embedder,
        store=vector_store,
        qcfg=config.rag.query_engine
    )
    
    # Тестируем асинхронный health check
    health = await query_engine.health_check()
    assert isinstance(health, dict)
    assert 'status' in health
    
    await query_engine.close()

if __name__ == "__main__":
    # Обратная совместимость для прямого запуска
    import pytest
    pytest.main([__file__, "-v"])