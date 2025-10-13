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

# Экспорт реализуется через фабрику (RAGFactory) и convenience-функции.
# Прямые классы и устаревшие алиасы не экспортируются здесь, чтобы исключить обход фабрики
# и предотвратить рекурсию на VM.
from .exceptions import (
    RagException,
    EmbeddingException,
    VectorStoreException,
    VectorStoreConnectionError,
    QueryEngineException,
    ModelLoadException,
    OutOfMemoryException
)

# Экспорт CPUEmbedder для совместимости с импортами вида 'from rag import CPUEmbedder'
from .embedder import CPUEmbedder

# Динамический выбор провайдера по переменным окружения отключён.
# Используйте RAGFactory для создания корректных реализаций в текущем контексте.

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
    # Factory API (рекомендуется для нового кода)
    'RAGFactory',
    'ExecutionContext',
    'detect_execution_context',
    'get_context_info',

    # Convenience функции (вызовы фабрики)
    'create_embedder',
    'create_vector_store',
    'create_search_service',
    'create_indexer_service',

    # Сервисы (интерфейсы/доступные сущности пакета)
    'VectorStore',
    'QueryEngine',
    'CPUQueryEngine',
    'SearchService',
    'IndexerService',
    'CPUEmbedder',

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
