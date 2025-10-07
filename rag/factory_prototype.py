"""
Прототип: Factory для создания RAG компонентов с учётом контекста.

Минимальная реализация для проверки концепции устранения рекурсии.
"""

import logging
from typing import Optional
from .context_prototype import ExecutionContext, detect_execution_context

logger = logging.getLogger(__name__)


class RAGFactoryPrototype:
    """
    Factory прототип для создания RAG компонентов.
    
    Автоматически выбирает правильную реализацию (Local/Remote)
    на основе контекста выполнения.
    """
    
    _context: Optional[ExecutionContext] = None
    
    @classmethod
    def set_context(cls, context: ExecutionContext) -> None:
        """
        Явно устанавливает контекст выполнения.
        
        Args:
            context: Контекст для принудительной установки
        """
        cls._context = context
        logger.info(f"✅ RAG контекст установлен явно: {context.value}")
    
    @classmethod
    def get_context(cls) -> ExecutionContext:
        """
        Возвращает текущий контекст (с автодетекцией если не установлен).
        
        Returns:
            ExecutionContext: Текущий контекст выполнения
        """
        if cls._context is None:
            cls._context = detect_execution_context()
            logger.info(f"🔍 Автодетекция контекста: {cls._context.value}")
        return cls._context
    
    @classmethod
    def reset_context(cls) -> None:
        """Сбрасывает кэшированный контекст для повторной детекции"""
        cls._context = None
        logger.debug("🔄 Контекст сброшен")
    
    @classmethod
    def create_embedder(cls, config):
        """
        Создаёт embedder на основе контекста.
        
        Args:
            config: Конфигурация системы
            
        Returns:
            CPUEmbedder (local) или RemoteVMEmbedder в зависимости от контекста
        """
        context = cls.get_context()
        
        if context == ExecutionContext.VM:
            # На VM используем локальный embedder
            from .embedder import CPUEmbedder
            logger.info("✅ Factory: Создан локальный CPUEmbedder (VM контекст)")
            return CPUEmbedder(
                config.rag.embeddings,
                config.rag.parallelism
            )
        else:
            # На клиенте используем remote embedder
            from .remote_embedder import RemoteVMEmbedder
            logger.info("✅ Factory: Создан RemoteVMEmbedder (CLIENT контекст)")
            return RemoteVMEmbedder(
                config.rag.embeddings,
                config.rag.parallelism,
                config.rag.remote_service
            )
    
    @classmethod
    def create_vector_store(cls, config):
        """
        Создаёт vector store на основе контекста.
        
        Args:
            config: Конфигурация системы
            
        Returns:
            QdrantVectorStore (local) или RemoteVMVectorStore в зависимости от контекста
        """
        context = cls.get_context()
        
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
                config.rag.vector_store,
                config.rag.remote_service
            )
    
    @classmethod
    def get_factory_info(cls) -> dict:
        """
        Возвращает информацию о текущем состоянии Factory для диагностики.
        
        Returns:
            dict: Информация о Factory
        """
        context = cls.get_context()
        return {
            'current_context': context.value,
            'context_cached': cls._context is not None,
            'expected_embedder': 'CPUEmbedder' if context == ExecutionContext.VM else 'RemoteVMEmbedder',
            'expected_vector_store': 'QdrantVectorStore' if context == ExecutionContext.VM else 'RemoteVMVectorStore'
        }