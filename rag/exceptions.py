"""
Кастомные исключения для RAG системы.

Иерархия исключений:
- RagException (базовый класс)
  - EmbeddingException (ошибки эмбеддингов)
    - ModelLoadException (ошибки загрузки модели)
    - OutOfMemoryException (нехватка памяти)
  - VectorStoreException (ошибки векторной БД)
  - QueryEngineException (ошибки поискового движка)
"""


class RagException(Exception):
    """Базовое исключение для всех ошибок RAG системы"""
    
    def __init__(self, message: str, details: str = None):
        """
        Инициализация исключения RAG.
        
        Args:
            message: Основное сообщение об ошибке
            details: Дополнительные детали ошибки
        """
        self.message = message
        self.details = details
        
        full_message = message
        if details:
            full_message += f" Детали: {details}"
            
        super().__init__(full_message)


class EmbeddingException(RagException):
    """Исключение для ошибок связанных с эмбеддингами"""
    
    def __init__(self, message: str, provider: str = None, model_name: str = None, details: str = None):
        """
        Инициализация исключения эмбеддингов.
        
        Args:
            message: Основное сообщение об ошибке
            provider: Провайдер эмбеддингов (fastembed, sentence-transformers)
            model_name: Имя модели
            details: Дополнительные детали ошибки
        """
        self.provider = provider
        self.model_name = model_name
        
        # Формируем расширенное сообщение
        extended_message = message
        if provider:
            extended_message += f" (провайдер: {provider})"
        if model_name:
            extended_message += f" (модель: {model_name})"
            
        super().__init__(extended_message, details)


class ModelLoadException(EmbeddingException):
    """Исключение для ошибок загрузки модели эмбеддингов"""
    
    def __init__(self, provider: str, model_name: str, error_message: str):
        """
        Инициализация исключения загрузки модели.
        
        Args:
            provider: Провайдер эмбеддингов
            model_name: Имя модели
            error_message: Сообщение об ошибке
        """
        message = "Не удалось загрузить модель эмбеддингов"
        super().__init__(
            message=message,
            provider=provider,
            model_name=model_name,
            details=error_message
        )


class OutOfMemoryException(EmbeddingException):
    """Исключение для ошибок нехватки памяти при обработке эмбеддингов"""
    
    def __init__(self, batch_size: int = None, available_memory: float = None, details: str = None):
        """
        Инициализация исключения нехватки памяти.
        
        Args:
            batch_size: Размер батча, который привёл к ошибке
            available_memory: Доступная память в ГБ
            details: Дополнительные детали ошибки
        """
        self.batch_size = batch_size
        self.available_memory = available_memory
        
        message = "Нехватка памяти при обработке эмбеддингов"
        
        if batch_size is not None:
            message += f" (размер батча: {batch_size})"
        if available_memory is not None:
            message += f" (доступно памяти: {available_memory:.1f}GB)"
            
        super().__init__(message=message, details=details)


class VectorStoreException(RagException):
    """Исключение для ошибок векторной базы данных"""
    
    def __init__(self, message: str, collection_name: str = None, operation: str = None, details: str = None):
        """
        Инициализация исключения векторной БД.
        
        Args:
            message: Основное сообщение об ошибке
            collection_name: Имя коллекции
            operation: Операция, которая привела к ошибке
            details: Дополнительные детали ошибки
        """
        self.collection_name = collection_name
        self.operation = operation
        
        extended_message = message
        if collection_name:
            extended_message += f" (коллекция: {collection_name})"
        if operation:
            extended_message += f" (операция: {operation})"
            
        super().__init__(extended_message, details)


class QueryEngineException(RagException):
    """Исключение для ошибок поискового движка"""
    
    def __init__(self, message: str, query: str = None, search_type: str = None, details: str = None):
        """
        Инициализация исключения поискового движка.
        
        Args:
            message: Основное сообщение об ошибке
            query: Поисковый запрос
            search_type: Тип поиска (dense, sparse, hybrid)
            details: Дополнительные детали ошибки
        """
        self.query = query
        self.search_type = search_type
        
        extended_message = message
        if search_type:
            extended_message += f" (тип поиска: {search_type})"
        if query and len(query) <= 50:
            extended_message += f" (запрос: '{query}')"
        elif query:
            extended_message += f" (запрос: '{query[:47]}...')"
            
        super().__init__(extended_message, details)


# Дополнительные специализированные исключения

class ConnectionException(VectorStoreException):
    """Исключение для ошибок подключения к векторной БД"""
    
    def __init__(self, host: str, port: int, error_message: str):
        """
        Инициализация исключения подключения.
        
        Args:
            host: Хост БД
            port: Порт БД
            error_message: Сообщение об ошибке
        """
        message = f"Не удалось подключиться к векторной БД {host}:{port}"
        super().__init__(
            message=message,
            operation="connection",
            details=error_message
        )


class CollectionNotFoundException(VectorStoreException):
    """Исключение для случаев, когда коллекция не найдена"""
    
    def __init__(self, collection_name: str):
        """
        Инициализация исключения отсутствующей коллекции.
        
        Args:
            collection_name: Имя коллекции
        """
        message = "Коллекция не найдена"
        super().__init__(
            message=message,
            collection_name=collection_name,
            operation="access"
        )


class TimeoutException(RagException):
    """Исключение для превышения таймаута операции"""
    
    def __init__(self, operation: str, timeout_seconds: float, elapsed_seconds: float = None):
        """
        Инициализация исключения таймаута.
        
        Args:
            operation: Операция, которая превысила таймаут
            timeout_seconds: Установленный таймаут в секундах
            elapsed_seconds: Затраченное время в секундах
        """
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        
        message = f"Превышен таймаут операции '{operation}' ({timeout_seconds}s)"
        if elapsed_seconds is not None:
            message += f", затрачено: {elapsed_seconds:.2f}s"
            
        super().__init__(message)


class InvalidConfigurationException(RagException):
    """Исключение для ошибок конфигурации RAG системы"""
    
    def __init__(self, config_field: str, value: str, reason: str):
        """
        Инициализация исключения конфигурации.
        
        Args:
            config_field: Поле конфигурации с ошибкой
            value: Неверное значение
            reason: Причина ошибки
        """
        self.config_field = config_field
        self.value = value
        self.reason = reason
        
        message = f"Неверная конфигурация поля '{config_field}': {value}"
        super().__init__(message, reason)


# Утилитные функции для работы с исключениями

def is_recoverable_error(exception: Exception) -> bool:
    """
    Определяет, является ли ошибка восстановимой.
    
    Args:
        exception: Исключение для проверки
        
    Returns:
        True если ошибка может быть обработана с помощью retry или fallback
    """
    recoverable_types = (
        OutOfMemoryException,
        ConnectionException, 
        TimeoutException
    )
    
    return isinstance(exception, recoverable_types)


def should_fallback(exception: Exception) -> bool:
    """
    Определяет, следует ли использовать fallback при данной ошибке.
    
    Args:
        exception: Исключение для проверки
        
    Returns:
        True если следует попробовать fallback провайдер/метод
    """
    fallback_types = (
        ModelLoadException,
        OutOfMemoryException,
        EmbeddingException
    )
    
    return isinstance(exception, fallback_types)


def get_error_category(exception: Exception) -> str:
    """
    Возвращает категорию ошибки для логирования и мониторинга.
    
    Args:
        exception: Исключение для категоризации
        
    Returns:
        Строка с категорией ошибки
    """
    if isinstance(exception, ModelLoadException):
        return "model_load"
    elif isinstance(exception, OutOfMemoryException):
        return "memory"
    elif isinstance(exception, ConnectionException):
        return "connection"
    elif isinstance(exception, TimeoutException):
        return "timeout"
    elif isinstance(exception, EmbeddingException):
        return "embedding"
    elif isinstance(exception, VectorStoreException):
        return "vector_store"
    elif isinstance(exception, QueryEngineException):
        return "query_engine"
    elif isinstance(exception, InvalidConfigurationException):
        return "configuration"
    elif isinstance(exception, RagException):
        return "rag"
    else:
        return "unknown"


class VectorStoreConnectionError(VectorStoreException):
    """Ошибка подключения к векторному хранилищу"""
    
    def __init__(self, message: str, host: str = None, port: int = None):
        self.host = host
        self.port = port
        
        extended_message = message
        if host and port:
            extended_message += f" (адрес: {host}:{port})"
            
        super().__init__(
            message=extended_message,
            operation="connection"
        )


class VectorStoreConfigurationError(VectorStoreException):
    """Ошибка конфигурации векторного хранилища"""
    
    def __init__(self, message: str, config_field: str = None):
        self.config_field = config_field
        
        extended_message = message
        if config_field:
            extended_message += f" (поле: {config_field})"
            
        super().__init__(
            message=extended_message,
            operation="configuration"
        )


class VMConnectionError(EmbeddingException):
    """
    Исключение для ошибок подключения к удалённому VM сервису эмбеддингов.
    
    Это исключение возникает когда:
    - VM сервис недоступен (connection refused)
    - Firewall блокирует соединение
    - DNS не может разрешить адрес VM
    - Сетевые проблемы между клиентом и VM
    """
    
    def __init__(self, message: str, vm_host: str, vm_port: int, error_details: str = None):
        """
        Инициализация исключения подключения к VM.
        
        Args:
            message: Основное сообщение об ошибке
            vm_host: Хост VM сервиса
            vm_port: Порт VM сервиса
            error_details: Детали ошибки подключения
        """
        self.vm_host = vm_host
        self.vm_port = vm_port
        self.error_details = error_details
        
        extended_message = f"{message} (VM: {vm_host}:{vm_port})"
        
        super().__init__(
            message=extended_message,
            provider="remote-vm",
            details=error_details
        )
    
    def get_diagnostic_info(self) -> dict:
        """
        Возвращает диагностическую информацию для troubleshooting.
        
        Returns:
            Словарь с информацией для диагностики проблемы
        """
        return {
            "error_type": "vm_connection_error",
            "vm_host": self.vm_host,
            "vm_port": self.vm_port,
            "recommendation": (
                f"Проверьте: 1) VM сервис запущен на {self.vm_host}:{self.vm_port}, "
                f"2) Firewall не блокирует порт {self.vm_port}, "
                f"3) Сетевое подключение к {self.vm_host}"
            ),
            "troubleshooting_commands": [
                f"curl http://{self.vm_host}:{self.vm_port}/health",
                f"ping {self.vm_host}",
                f"telnet {self.vm_host} {self.vm_port}"
            ]
        }


class VMTimeoutError(EmbeddingException):
    """
    Исключение для ошибок таймаута при обращении к удалённому VM сервису.
    
    Это исключение возникает когда:
    - VM сервис не отвечает в установленное время
    - Сетевая задержка превышает допустимую
    - VM сервис перегружен и медленно обрабатывает запросы
    - Конфликт между outer и inner timeout
    """
    
    def __init__(
        self, 
        message: str, 
        timeout_seconds: float, 
        elapsed_seconds: float = None,
        operation: str = "embedding",
        retry_attempt: int = None
    ):
        """
        Инициализация исключения таймаута VM.
        
        Args:
            message: Основное сообщение об ошибке
            timeout_seconds: Установленный таймаут в секундах
            elapsed_seconds: Фактически затраченное время в секундах
            operation: Операция, которая превысила таймаут
            retry_attempt: Номер попытки retry (если применимо)
        """
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.operation = operation
        self.retry_attempt = retry_attempt
        
        extended_message = f"{message} (таймаут: {timeout_seconds}s"
        if elapsed_seconds is not None:
            extended_message += f", затрачено: {elapsed_seconds:.2f}s"
        if retry_attempt is not None:
            extended_message += f", попытка: {retry_attempt}"
        extended_message += ")"
        
        super().__init__(
            message=extended_message,
            provider="remote-vm",
            details=f"Операция '{operation}' превысила таймаут"
        )
    
    def get_diagnostic_info(self) -> dict:
        """
        Возвращает диагностическую информацию для troubleshooting.
        
        Returns:
            Словарь с информацией для диагностики проблемы
        """
        recommendation = (
            f"VM сервис не отвечает в срок ({self.timeout_seconds}s). "
        )
        
        if self.elapsed_seconds and self.elapsed_seconds > self.timeout_seconds * 2:
            recommendation += "Рассмотрите увеличение таймаута в конфигурации. "
        
        recommendation += "Проверьте: 1) Загрузку VM, 2) Сетевую задержку, 3) Размер батча"
        
        return {
            "error_type": "vm_timeout_error",
            "timeout_seconds": self.timeout_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "operation": self.operation,
            "retry_attempt": self.retry_attempt,
            "recommendation": recommendation,
            "suggested_actions": [
                f"Увеличить timeout_seconds в config (текущий: {self.timeout_seconds}s)",
                "Уменьшить batch_size для снижения нагрузки",
                "Проверить загрузку VM: top / htop на VM сервере",
                "Измерить сетевую задержку: ping -c 10 <vm_host>"
            ]
        }
