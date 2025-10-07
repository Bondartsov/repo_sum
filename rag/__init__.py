"""
RAG (Retrieval-Augmented Generation) система для анализатора репозиториев.

Основные компоненты:
- CPUEmbedder: CPU-оптимизированный эмбеддер с поддержкой FastEmbed и Sentence Transformers
- VectorStore: Интерфейс для работы с Qdrant векторной базой данных
- QueryEngine: Движок поиска с поддержкой гибридного поиска и MMR

Архитектура:
- Factory Pattern для автоматического выбора Local/Remote реализаций
- Контекст выполнения (VM/CLIENT) определяется автоматически
- Устранение проблемы рекурсии через правильный выбор компонентов
"""

# ✅ ИСПРАВЛЕНО: Factory Pattern для автоматического выбора реализаций
from .factory import RAGFactory
from .context import ExecutionContext, detect_execution_context, get_context_info

# Convenience функции для создания компонентов
def create_embedder(config):
    """Создаёт embedder с автоматическим выбором Local/Remote реализации"""
    return RAGFactory.create_embedder(config)

def create_vector_store(config):
    """Создаёт vector_store с автоматическим выбором Local/Remote реализации"""
    return RAGFactory.create_vector_store(config)

def create_search_service(config, silent_mode=False):
    """Создаёт search_service с автоматическим выбором компонентов"""
    return RAGFactory.create_search_service(config, silent_mode)

def create_indexer_service(config, silent_mode=False):
    """Создаёт indexer_service с автоматическим выбором компонентов"""
    return RAGFactory.create_indexer_service(config, silent_mode)

# Прямые импорты для обратной совместимости и явного использования
from .embedder import CPUEmbedder as LocalCPUEmbedder
from .remote_embedder import RemoteVMEmbedder
from .vector_store import QdrantVectorStore as LocalQdrantVectorStore
from .remote_vector_store import RemoteVMVectorStore

# Алиасы для обратной совместимости (устаревшие, используйте Factory)
# DEPRECATED: Используйте RAGFactory.create_*() или явные Local/Remote импорты
CPUEmbedder = RemoteVMEmbedder  # По умолчанию Remote для клиентов
QdrantVectorStore = RemoteVMVectorStore  # По умолчанию Remote для клиентов
from .exceptions import (
    RagException,
    EmbeddingException,
    VectorStoreException,
    VectorStoreConnectionError,
    QueryEngineException,
    ModelLoadException,
    OutOfMemoryException
)

# Dynamic provider selection via environment variables
import os as _os
_emb_provider = (_os.getenv('EMBEDDING_PROVIDER') or '').lower().strip()
_vs_provider = (_os.getenv('VECTOR_STORE_PROVIDER') or '').lower().strip()

try:
    if _emb_provider != 'remote-vm':
        from .embedder import CPUEmbedder as _LocalCPU  # type: ignore
        CPUEmbedder = _LocalCPU  # type: ignore
except Exception:
    # keep remote embedder as default
    pass

try:
    if _vs_provider != 'remote-vm':
        from .vector_store import QdrantVectorStore as _LocalVS  # type: ignore
        QdrantVectorStore = _LocalVS  # type: ignore
except Exception:
    # keep remote vector store as default
    pass

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
    # Factory Pattern (рекомендуется для нового кода)
    'RAGFactory',
    'ExecutionContext',
    'detect_execution_context',
    'get_context_info',
    
    # Convenience функции
    'create_embedder',
    'create_vector_store',
    'create_search_service',
    'create_indexer_service',
    
    # Прямые импорты (для явного использования)
    'LocalCPUEmbedder',
    'RemoteVMEmbedder',
    'LocalQdrantVectorStore',
    'RemoteVMVectorStore',
    
    # Устаревшие алиасы (обратная совместимость)
    'CPUEmbedder',  # DEPRECATED: Используйте RAGFactory или LocalCPUEmbedder
    'QdrantVectorStore',  # DEPRECATED: Используйте RAGFactory или LocalQdrantVectorStore
    
    # Сервисы
    'VectorStore',
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

__version__ = "0.5"
__author__ = "RAG Team"
__description__ = "CPU-оптимизированная RAG система для анализа кода"
