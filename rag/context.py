"""
Определение контекста выполнения для автоматического выбора Local/Remote компонентов.

Контекст определяет какие реализации RAG компонентов использовать:
- VM: Локальные компоненты (QdrantVectorStore, CPUEmbedder)
- CLIENT: Удалённые компоненты (RemoteVMVectorStore, RemoteVMEmbedder)
"""

import os
import socket
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionContext(Enum):
    """
    Контекст выполнения приложения.
    
    Определяет какие реализации RAG компонентов использовать:
    - VM: Запущено на VM сервере - используем локальные компоненты
    - CLIENT: Запущено на клиенте - используем удалённые компоненты
    - UNKNOWN: Не удалось определить - используем CLIENT по умолчанию
    """
    VM = "vm"
    CLIENT = "client"
    UNKNOWN = "unknown"


def detect_execution_context() -> ExecutionContext:
    """
    Автоматически определяет контекст выполнения.
    
    Стратегия детекции (в порядке приоритета):
    1. Переменная окружения RAG_EXECUTION_CONTEXT (явное указание)
    2. Hostname содержит 'vm', 'rag-server', 'ubuntu' (VM индикаторы)
    3. Доступность локального Qdrant на порту 6333 (VM индикатор)
    4. Наличие VM-специфичных директорий (/etc/qdrant, /var/lib/qdrant)
    5. По умолчанию - CLIENT контекст
    
    Returns:
        ExecutionContext: Определённый контекст выполнения
        
    Examples:
        >>> # На VM сервере
        >>> context = detect_execution_context()
        >>> assert context == ExecutionContext.VM
        
        >>> # На клиенте
        >>> context = detect_execution_context()
        >>> assert context == ExecutionContext.CLIENT
    """
    # 1. Явное указание через переменную окружения (наивысший приоритет)
    env_context = os.getenv('RAG_EXECUTION_CONTEXT', '').lower().strip()
    if env_context == 'vm':
        logger.info("🔍 Контекст: VM (установлен через RAG_EXECUTION_CONTEXT)")
        return ExecutionContext.VM
    elif env_context == 'client':
        logger.info("🔍 Контекст: CLIENT (установлен через RAG_EXECUTION_CONTEXT)")
        return ExecutionContext.CLIENT
    
    # 2. Проверка hostname на VM-специфичные маркеры
    try:
        hostname = socket.gethostname().lower()
        vm_markers = ['vm', 'rag-server', 'ubuntu', 'qdrant-server']
        
        if any(marker in hostname for marker in vm_markers):
            logger.info(f"🔍 Контекст: VM (hostname содержит VM маркер: {hostname})")
            return ExecutionContext.VM
    except Exception as e:
        logger.debug(f"Не удалось проверить hostname: {e}")
    
    # 3. Проверка наличия локального Qdrant (порт 6333)
    # Если Qdrant доступен локально -> скорее всего это VM
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 6333))
        sock.close()
        
        if result == 0:
            logger.info("🔍 Контекст: VM (обнаружен локальный Qdrant на порту 6333)")
            return ExecutionContext.VM
    except Exception as e:
        logger.debug(f"Не удалось проверить Qdrant порт: {e}")
    
    # 4. Проверка VM-специфичных директорий (только для Linux/Unix систем)
    try:
        vm_directories = [
            '/etc/qdrant',
            '/var/lib/qdrant',
            '/opt/rag-service'
        ]
        
        if any(os.path.exists(path) for path in vm_directories):
            logger.info("🔍 Контекст: VM (обнаружены VM-специфичные директории)")
            return ExecutionContext.VM
    except Exception as e:
        logger.debug(f"Не удалось проверить VM директории: {e}")
    
    # 5. По умолчанию - клиентский контекст (безопасный fallback)
    logger.info("🔍 Контекст: CLIENT (по умолчанию - не обнаружено VM маркеров)")
    return ExecutionContext.CLIENT


def get_context_info() -> dict:
    """
    Возвращает подробную информацию о текущем контексте для диагностики.
    
    Returns:
        dict: Информация о контексте выполнения со всеми проверками
        
    Example:
        >>> info = get_context_info()
        >>> print(f"Контекст: {info['context']}")
        >>> print(f"Qdrant локально: {info['qdrant_local_available']}")
    """
    context = detect_execution_context()
    
    # Проверяем доступность локального Qdrant
    qdrant_available = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 6333))
        sock.close()
        qdrant_available = (result == 0)
    except:
        pass
    
    # Проверяем VM директории
    vm_dirs_exist = False
    try:
        vm_directories = ['/etc/qdrant', '/var/lib/qdrant', '/opt/rag-service']
        vm_dirs_exist = any(os.path.exists(path) for path in vm_directories)
    except:
        pass
    
    return {
        'context': context.value,
        'env_variable': os.getenv('RAG_EXECUTION_CONTEXT', 'not_set'),
        'hostname': socket.gethostname(),
        'qdrant_local_available': qdrant_available,
        'vm_directories_exist': vm_dirs_exist,
        'detection_method': _get_detection_method()
    }


def _get_detection_method() -> str:
    """Определяет каким методом был определён контекст"""
    env_context = os.getenv('RAG_EXECUTION_CONTEXT', '').lower().strip()
    if env_context in ('vm', 'client'):
        return 'environment_variable'
    
    try:
        hostname = socket.gethostname().lower()
        vm_markers = ['vm', 'rag-server', 'ubuntu', 'qdrant-server']
        if any(marker in hostname for marker in vm_markers):
            return 'hostname'
    except:
        pass
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 6333))
        sock.close()
        if result == 0:
            return 'qdrant_port_check'
    except:
        pass
    
    try:
        vm_dirs = ['/etc/qdrant', '/var/lib/qdrant', '/opt/rag-service']
        if any(os.path.exists(path) for path in vm_dirs):
            return 'vm_directories'
    except:
        pass
    
    return 'default_fallback'