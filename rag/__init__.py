"""
RAG (Retrieval-Augmented Generation) система для анализатора репозиториев.

Основные компоненты:
- CPUEmbedder: CPU-оптимизированный эмбеддер с поддержкой FastEmbed и Sentence Transformers
- VectorStore: Интерфейс для работы с Qdrant векторной базой данных
- QueryEngine: Движок поиска с поддержкой гибридного поиска и MMR
"""

# ✅ ИСПРАВЛЕНО: Импортируем REMOTE версии с алиасами для обратной совместимости
from .remote_embedder import RemoteVMEmbedder as CPUEmbedder
from .remote_vector_store import RemoteVMVectorStore as QdrantVectorStore
from .exceptions import (
    RagException,
    EmbeddingException,
    VectorStoreException,
    VectorStoreConnectionError,
    QueryEngineException,
    ModelLoadException,
    OutOfMemoryException
)

# Базовые классы
VectorStore = None  # Базовый класс пока не реализован

try:
    from .query_engine import CPUQueryEngine
    QueryEngine = None  # Базовый класс пока не реализован
except ImportError:
    # Модуль еще не реализован
    QueryEngine = None
    CPUQueryEngine = None

try:
    from .search_service import SearchService
except ImportError:
    # Модуль еще не реализован
    SearchService = None

try:
    from .indexer_service import IndexerService
except ImportError:
    # Модуль еще не реализован
    IndexerService = None

__all__ = [
    # Основные классы
    'CPUEmbedder',
    'VectorStore',
    'QdrantVectorStore',
    'QueryEngine',
    'CPUQueryEngine',
    'SearchService',
    'IndexerService',
    
    # Исключения
    'RagException',
    'EmbeddingException',
    'VectorStoreException',
    'VectorStoreConnectionError',
    'QueryEngineException',
    'ModelLoadException',
    'OutOfMemoryException',
]

__version__ = "0.1.0"
__author__ = "RAG Team"
__description__ = "CPU-оптимизированная RAG система для анализа кода"
