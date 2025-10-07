"""
Прототип: Детекция контекста выполнения (VM vs CLIENT).

Минимальная реализация для проверки концепции Factory Pattern.
"""

import os
import socket
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ExecutionContext(Enum):
    """Контекст выполнения приложения"""
    VM = "vm"
    CLIENT = "client"
    UNKNOWN = "unknown"


def detect_execution_context() -> ExecutionContext:
    """
    Автоматически определяет контекст выполнения.
    
    Стратегия детекции для прототипа:
    1. Проверка переменной окружения RAG_EXECUTION_CONTEXT (явное указание)
    2. Проверка доступности локального Qdrant на порту 6333
    3. По умолчанию - CLIENT контекст
    
    Returns:
        ExecutionContext: Определённый контекст (VM или CLIENT)
    """
    # 1. Явное указание через переменную окружения
    env_context = os.getenv('RAG_EXECUTION_CONTEXT', '').lower()
    if env_context == 'vm':
        logger.info("🔍 Контекст: VM (установлен через RAG_EXECUTION_CONTEXT)")
        return ExecutionContext.VM
    elif env_context == 'client':
        logger.info("🔍 Контекст: CLIENT (установлен через RAG_EXECUTION_CONTEXT)")
        return ExecutionContext.CLIENT
    
    # 2. Проверка наличия локального Qdrant (порт 6333)
    # Если Qdrant доступен локально -> это VM
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 6333))
        sock.close()
        
        if result == 0:
            logger.info("🔍 Контекст: VM (обнаружен локальный Qdrant на порту 6333)")
            return ExecutionContext.VM
    except Exception as e:
        logger.debug(f"Ошибка проверки Qdrant порта: {e}")
    
    # 3. По умолчанию - клиентский контекст
    logger.info("🔍 Контекст: CLIENT (по умолчанию)")
    return ExecutionContext.CLIENT


def get_context_info() -> dict:
    """
    Возвращает информацию о текущем контексте для диагностики.
    
    Returns:
        dict: Информация о контексте выполнения
    """
    context = detect_execution_context()
    
    # Проверяем доступность Qdrant
    qdrant_available = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 6333))
        sock.close()
        qdrant_available = (result == 0)
    except:
        pass
    
    return {
        'context': context.value,
        'env_variable': os.getenv('RAG_EXECUTION_CONTEXT', 'not_set'),
        'qdrant_local_available': qdrant_available,
        'hostname': socket.gethostname()
    }