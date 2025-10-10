"""
Factory для создания RAG компонентов с автоматическим выбором реализации.

Паттерн Factory автоматически выбирает между Local и Remote реализациями
на основе контекста выполнения (VM vs CLIENT), устраняя проблему рекурсии
при индексации на VM сервере.
"""

import logging
from typing import Optional
from .context import ExecutionContext, detect_execution_context
from config import Config

logger = logging.getLogger(__name__)


class RAGFactory:
    """
    Factory для создания RAG компонентов с учётом контекста выполнения.
    
    Автоматически выбирает правильную реализацию (Local/Remote)
    на основе контекста, устраняя проблему рекурсии на VM.
    
    Примеры:
        >>> # Автоматическая детекция контекста
        >>> config = get_config()
        >>> embedder = RAGFactory.create_embedder(config)
        >>> vector_store = RAGFactory.create_vector_store(config)
        
        >>> # Явное указание контекста (для тестов)
        >>> RAGFactory.set_context(ExecutionContext.VM)
        >>> embedder = RAGFactory.create_embedder(config)  # Будет Local
    """
    
    _context: Optional[ExecutionContext] = None
    _context_override: bool = False
    
    @classmethod
    def set_context(cls, context: ExecutionContext) -> None:
        """
        Явно устанавливает контекст выполнения.
        
        Используется для:
        - Принудительной установки контекста в тестах
        - Явного указания контекста при старте VM сервиса
        - Переопределения автоматической детекции
        
        Args:
            context: Контекст для установки (VM или CLIENT)
            
        Example:
            >>> RAGFactory.set_context(ExecutionContext.VM)
            >>> # Все последующие создания будут использовать VM контекст
        """
        cls._context = context
        cls._context_override = True
        logger.info(f"✅ RAG контекст установлен явно: {context.value}")
    
    @classmethod
    def get_context(cls) -> ExecutionContext:
        """
        Возвращает текущий контекст (с автодетекцией если не установлен).
        
        Если контекст не был явно установлен через set_context,
        выполняется автоматическая детекция через detect_execution_context().
        
        Returns:
            ExecutionContext: Текущий контекст выполнения
            
        Example:
            >>> context = RAGFactory.get_context()
            >>> if context == ExecutionContext.VM:
            >>>     print("Запущено на VM сервере")
        """
        if cls._context is None:
            cls._context = detect_execution_context()
            if not cls._context_override:
                logger.info(f"🔍 Автодетекция контекста: {cls._context.value}")
        return cls._context
    
    @classmethod
    def reset_context(cls) -> None:
        """
        Сбрасывает кэшированный контекст для повторной детекции.
        
        Используется в тестах для сброса состояния между тестами.
        
        Example:
            >>> RAGFactory.set_context(ExecutionContext.VM)
            >>> RAGFactory.reset_context()
            >>> # Следующий get_context() выполнит автодетекцию заново
        """
        cls._context = None
        cls._context_override = False
        logger.debug("🔄 Контекст сброшен, будет выполнена повторная детекция")
    
    @classmethod
    def create_embedder(cls, config: Config):
        """
        Создаёт embedder на основе контекста выполнения.
        
        Выбор реализации:
        - VM контекст: CPUEmbedder (локальная модель Jina v3)
        - CLIENT контекст: RemoteVMEmbedder (HTTP клиент к VM)
        
        Args:
            config: Конфигурация системы
            
        Returns:
            CPUEmbedder или RemoteVMEmbedder в зависимости от контекста
            
        Raises:
            ImportError: Если требуемая реализация недоступна
            
        Example:
            >>> config = get_config()
            >>> embedder = RAGFactory.create_embedder(config)
            >>> # На VM: CPUEmbedder, на клиенте: RemoteVMEmbedder
        """
        context = cls.get_context()
        
        try:
            if context == ExecutionContext.VM:
                # На VM используем локальный embedder
                from .embedder import CPUEmbedder
                logger.info("✅ Factory: Создан локальный CPUEmbedder (VM контекст)")
                return CPUEmbedder(
                    embedding_config=config.rag.embeddings,
                    parallelism_config=config.rag.parallelism,
                    remote_service_config=None  # Не нужен на VM
                )
            else:
                # На клиенте используем remote embedder
                from .remote_embedder import RemoteVMEmbedder
                logger.info("✅ Factory: Создан RemoteVMEmbedder (CLIENT контекст)")
                return RemoteVMEmbedder(
                    embedding_config=config.rag.embeddings,
                    parallelism_config=config.rag.parallelism,
                    remote_service_config=config.rag.remote_service
                )
        except ImportError as e:
            logger.error(f"❌ Ошибка импорта embedder для контекста {context.value}: {e}")
            raise
    
    @classmethod
    def create_vector_store(cls, config: Config):
        """
        Создаёт vector store на основе контекста выполнения.
        
        Выбор реализации:
        - VM контекст: QdrantVectorStore (прямое подключение к Qdrant)
        - CLIENT контекст: RemoteVMVectorStore (HTTP клиент к VM)
        
        Args:
            config: Конфигурация системы
            
        Returns:
            QdrantVectorStore или RemoteVMVectorStore в зависимости от контекста
            
        Raises:
            ImportError: Если требуемая реализация недоступна
            
        Example:
            >>> config = get_config()
            >>> vector_store = RAGFactory.create_vector_store(config)
            >>> # На VM: QdrantVectorStore, на клиенте: RemoteVMVectorStore
        """
        context = cls.get_context()
        
        try:
            if context == ExecutionContext.VM:
                # На VM используем локальный Qdrant
                from .vector_store import QdrantVectorStore
                logger.info("✅ Factory: Создан локальный QdrantVectorStore (VM контекст)")
                return QdrantVectorStore(config.rag.vector_store)
            else:
                # На клиенте используем remote vector store
                from .remote_vector_store import RemoteVMVectorStore
                logger.info("✅ Factory: Создан RemoteVMVectorStore (CLIENT контекст)")
                return RemoteVMVectorStore(
                    vector_store_config=config.rag.vector_store,
                    remote_service_config=config.rag.remote_service
                )
        except ImportError as e:
            logger.error(f"❌ Ошибка импорта vector_store для контекста {context.value}: {e}")
            raise
    
    @classmethod
    def create_search_service(cls, config: Config, silent_mode: bool = False):
        """
        Создаёт SearchService с правильными компонентами.
        
        SearchService использует embedder и vector_store,
        которые автоматически выбираются через Factory.
        
        Args:
            config: Конфигурация системы
            silent_mode: Отключить консольный вывод
            
        Returns:
            SearchService с правильными компонентами
            
        Example:
            >>> search_service = RAGFactory.create_search_service(config)
        """
        from .search_service import SearchService
        
        logger.info(f"✅ Factory: Создан SearchService ({cls.get_context().value} контекст)")
        return SearchService(config=config, silent_mode=silent_mode)
    
    @classmethod
    def create_indexer_service(cls, config: Config, silent_mode: bool = False):
        """
        Создаёт IndexerService с правильными компонентами.
        
        IndexerService использует embedder и vector_store,
        которые автоматически выбираются через Factory.
        
        Args:
            config: Конфигурация системы
            silent_mode: Отключить консольный вывод
            
        Returns:
            IndexerService с правильными компонентами
            
        Example:
            >>> indexer_service = RAGFactory.create_indexer_service(config)
        """
        from .indexer_service import IndexerService
        
        logger.info(f"✅ Factory: Создан IndexerService ({cls.get_context().value} контекст)")
        return IndexerService(config=config, silent_mode=silent_mode)
    
    @classmethod
    def get_factory_info(cls) -> dict:
        """
        Возвращает информацию о текущем состоянии Factory для диагностики.
        
        Returns:
            dict: Информация о Factory состоянии
            
        Example:
            >>> info = RAGFactory.get_factory_info()
            >>> print(f"Контекст: {info['current_context']}")
            >>> print(f"Ожидаемый embedder: {info['expected_embedder']}")
        """
        context = cls.get_context()
        
        return {
            'current_context': context.value,
            'context_cached': cls._context is not None,
            'context_override': cls._context_override,
            'expected_embedder': 'CPUEmbedder' if context == ExecutionContext.VM else 'RemoteVMEmbedder',
            'expected_vector_store': 'QdrantVectorStore' if context == ExecutionContext.VM else 'RemoteVMVectorStore',
            'expected_search_service': 'SearchService (local components)' if context == ExecutionContext.VM else 'SearchService (remote components)',
            'expected_indexer_service': 'IndexerService (local components)' if context == ExecutionContext.VM else 'IndexerService (remote components)'
        }