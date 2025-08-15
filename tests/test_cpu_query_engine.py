#!/usr/bin/env python3
"""
Тест для проверки интеграции CPUQueryEngine с RAG системой.

Проверяет:
- Импорт CPUQueryEngine
- Базовую инициализацию
- Совместимость с существующими компонентами
"""

import asyncio
import logging
import sys
from pathlib import Path
import pytest

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture
def query_engine():
    """Фикстура для создания CPUQueryEngine"""
    from rag import CPUEmbedder, QdrantVectorStore
    from rag.query_engine import CPUQueryEngine
    from config import get_config
    
    # Получаем конфигурацию
    config = get_config(require_api_key=False)
    
    # Создаём компоненты (без подключения к реальному Qdrant)
    embedder = CPUEmbedder(
        config.rag.embeddings,
        config.rag.parallelism
    )
    
    # Создаём заглушку для QdrantVectorStore
    if QdrantVectorStore is None:
        from rag.vector_store import QdrantVectorStore as DirectQdrantVectorStore
        vector_store = DirectQdrantVectorStore.__new__(DirectQdrantVectorStore)
    else:
        vector_store = QdrantVectorStore.__new__(QdrantVectorStore)
        
    vector_store.host = config.rag.vector_store.host
    vector_store.port = config.rag.vector_store.port
    vector_store.collection_name = config.rag.vector_store.collection_name
    
    # Создаём CPUQueryEngine
    query_engine = CPUQueryEngine(
        embedder=embedder,
        store=vector_store,
        qcfg=config.rag.query_engine
    )
    
    yield query_engine
    
    # Очистка после теста
    asyncio.run(query_engine.close())

def test_imports():
    """Тестирует импорты RAG компонентов"""
    # Тестируем импорт из пакета rag
    from rag import CPUEmbedder, QdrantVectorStore, SearchService
    
    # Импортируем CPUQueryEngine напрямую из модуля
    from rag.query_engine import CPUQueryEngine
    
    # Тестируем импорт конфигурации
    from config import get_config, QueryEngineConfig
    
    # Тестируем импорт исключений
    from rag.exceptions import QueryEngineException, VectorStoreException
    
    # Проверяем, что все импорты прошли успешно
    assert CPUEmbedder is not None
    assert SearchService is not None
    assert CPUQueryEngine is not None
    assert get_config is not None
    assert QueryEngineConfig is not None

def test_basic_initialization():
    """Тестирует базовую инициализацию CPUQueryEngine"""
    from rag import CPUEmbedder, QdrantVectorStore
    
    # Импортируем CPUQueryEngine напрямую из модуля
    from rag.query_engine import CPUQueryEngine
        
    from config import get_config
    
    # Получаем конфигурацию
    config = get_config(require_api_key=False)
    
    # Создаём компоненты (без подключения к реальному Qdrant)
    embedder = CPUEmbedder(
        config.rag.embeddings,
        config.rag.parallelism
    )
    
    # Создаём заглушку для QdrantVectorStore
    if QdrantVectorStore is None:
        from rag.vector_store import QdrantVectorStore as DirectQdrantVectorStore
        vector_store = DirectQdrantVectorStore.__new__(DirectQdrantVectorStore)
    else:
        vector_store = QdrantVectorStore.__new__(QdrantVectorStore)
        
    vector_store.host = config.rag.vector_store.host
    vector_store.port = config.rag.vector_store.port
    vector_store.collection_name = config.rag.vector_store.collection_name
    
    # Создаём CPUQueryEngine
    query_engine = CPUQueryEngine(
        embedder=embedder,
        store=vector_store,
        qcfg=config.rag.query_engine
    )
    
    # Проверяем основные атрибуты
    assert query_engine.embedder is not None
    assert query_engine.vector_store is not None
    assert query_engine.config is not None
    assert query_engine.cache is not None
    assert query_engine.stats is not None

async def test_basic_functionality(query_engine):
    """Тестирует базовую функциональность CPUQueryEngine"""
    # Тестируем методы статистики
    stats = query_engine.get_stats()
    assert isinstance(stats, dict)
    assert 'total_queries' in stats
    
    # Тестируем методы кэша
    cache_cleared = query_engine.clear_cache()
    assert isinstance(cache_cleared, int)
    
    # Тестируем сброс статистики
    query_engine.reset_stats()
    new_stats = query_engine.get_stats()
    assert new_stats['total_queries'] == 0
    
    # Тестируем health check
    health = await query_engine.health_check()
    assert isinstance(health, dict)
    assert 'status' in health

def test_configuration_compatibility():
    """Тестирует совместимость с существующей конфигурацией"""
    from config import get_config
    
    config = get_config(require_api_key=False)
    qcfg = config.rag.query_engine
    
    # Проверяем наличие всех необходимых настроек
    required_attrs = [
        'max_results', 'rrf_enabled', 'use_hybrid',
        'mmr_enabled', 'mmr_lambda', 'cache_ttl_seconds',
        'cache_max_entries', 'score_threshold',
        'concurrent_users_target', 'search_workers', 'embed_workers'
    ]
    
    for attr in required_attrs:
        assert hasattr(qcfg, attr), f"Отсутствует атрибут {attr}"

if __name__ == "__main__":
    # Обратная совместимость для прямого запуска
    import pytest
    pytest.main([__file__, "-v"])