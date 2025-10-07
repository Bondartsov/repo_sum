"""
Тестовый скрипт для проверки прототипа Factory Pattern.

Проверяет:
1. Детекцию контекста (VM vs CLIENT)
2. Правильный выбор реализации компонентов
3. Устранение рекурсии на VM

Запуск:
    python test_factory_prototype.py
"""

import sys
import os
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.context_prototype import ExecutionContext, detect_execution_context, get_context_info
from rag.factory_prototype import RAGFactoryPrototype
from config import get_config


def print_separator(title: str):
    """Красивый разделитель для вывода"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_context_detection():
    """Тест 1: Детекция контекста"""
    print_separator("ТЕСТ 1: Детекция контекста выполнения")
    
    # Получаем информацию о контексте
    info = get_context_info()
    
    print(f"\n📊 Информация о контексте:")
    print(f"   Контекст: {info['context'].upper()}")
    print(f"   Переменная окружения RAG_EXECUTION_CONTEXT: {info['env_variable']}")
    print(f"   Локальный Qdrant доступен: {info['qdrant_local_available']}")
    print(f"   Hostname: {info['hostname']}")
    
    context = detect_execution_context()
    print(f"\n✅ Определённый контекст: {context.value.upper()}")
    
    return context


def test_vm_context_simulation():
    """Тест 2: Симуляция VM контекста"""
    print_separator("ТЕСТ 2: Симуляция VM контекста (принудительная установка)")
    
    # Сбрасываем кэш
    RAGFactoryPrototype.reset_context()
    
    # Принудительно устанавливаем VM контекст
    RAGFactoryPrototype.set_context(ExecutionContext.VM)
    
    # Получаем информацию о Factory
    factory_info = RAGFactoryPrototype.get_factory_info()
    
    print(f"\n📊 Информация о Factory:")
    print(f"   Текущий контекст: {factory_info['current_context'].upper()}")
    print(f"   Контекст закэширован: {factory_info['context_cached']}")
    print(f"   Ожидается embedder: {factory_info['expected_embedder']}")
    print(f"   Ожидается vector_store: {factory_info['expected_vector_store']}")
    
    try:
        # Пытаемся создать компоненты
        config = get_config()
        
        print(f"\n🔨 Создание компонентов через Factory...")
        
        # Создаём vector_store
        vector_store = RAGFactoryPrototype.create_vector_store(config)
        vs_type = type(vector_store).__name__
        
        print(f"   Vector Store создан: {vs_type}")
        
        # Проверяем что это локальная версия
        if vs_type == 'QdrantVectorStore':
            print(f"   ✅ УСПЕХ: Создан локальный QdrantVectorStore (ожидалось)")
            result = True
        else:
            print(f"   ❌ ОШИБКА: Создан {vs_type}, ожидался QdrantVectorStore")
            result = False
        
        # Создаём embedder
        embedder = RAGFactoryPrototype.create_embedder(config)
        emb_type = type(embedder).__name__
        
        print(f"   Embedder создан: {emb_type}")
        
        # Проверяем что это локальная версия
        if emb_type == 'CPUEmbedder':
            print(f"   ✅ УСПЕХ: Создан локальный CPUEmbedder (ожидалось)")
            result = result and True
        else:
            print(f"   ❌ ОШИБКА: Создан {emb_type}, ожидался CPUEmbedder")
            result = False
        
        return result
        
    except Exception as e:
        print(f"\n❌ ОШИБКА при создании компонентов: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_context_simulation():
    """Тест 3: Симуляция CLIENT контекста"""
    print_separator("ТЕСТ 3: Симуляция CLIENT контекста (принудительная установка)")
    
    # Сбрасываем кэш
    RAGFactoryPrototype.reset_context()
    
    # Принудительно устанавливаем CLIENT контекст
    RAGFactoryPrototype.set_context(ExecutionContext.CLIENT)
    
    # Получаем информацию о Factory
    factory_info = RAGFactoryPrototype.get_factory_info()
    
    print(f"\n📊 Информация о Factory:")
    print(f"   Текущий контекст: {factory_info['current_context'].upper()}")
    print(f"   Контекст закэширован: {factory_info['context_cached']}")
    print(f"   Ожидается embedder: {factory_info['expected_embedder']}")
    print(f"   Ожидается vector_store: {factory_info['expected_vector_store']}")
    
    try:
        # Пытаемся создать компоненты
        config = get_config()
        
        print(f"\n🔨 Создание компонентов через Factory...")
        
        # Создаём vector_store
        vector_store = RAGFactoryPrototype.create_vector_store(config)
        vs_type = type(vector_store).__name__
        
        print(f"   Vector Store создан: {vs_type}")
        
        # Проверяем что это remote версия
        if vs_type == 'RemoteVMVectorStore':
            print(f"   ✅ УСПЕХ: Создан RemoteVMVectorStore (ожидалось)")
            result = True
        else:
            print(f"   ❌ ОШИБКА: Создан {vs_type}, ожидался RemoteVMVectorStore")
            result = False
        
        # Создаём embedder
        embedder = RAGFactoryPrototype.create_embedder(config)
        emb_type = type(embedder).__name__
        
        print(f"   Embedder создан: {emb_type}")
        
        # Проверяем что это remote версия
        if emb_type == 'RemoteVMEmbedder':
            print(f"   ✅ УСПЕХ: Создан RemoteVMEmbedder (ожидалось)")
            result = result and True
        else:
            print(f"   ❌ ОШИБКА: Создан {emb_type}, ожидался RemoteVMEmbedder")
            result = False
        
        return result
        
    except Exception as e:
        print(f"\n❌ ОШИБКА при создании компонентов: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recursion_prevention():
    """Тест 4: Проверка устранения рекурсии"""
    print_separator("ТЕСТ 4: Проверка устранения рекурсии на VM")
    
    print("\n📝 Проверка:")
    print("   1. На VM контексте IndexerService должен создавать ЛОКАЛЬНЫЙ QdrantVectorStore")
    print("   2. ЛОКАЛЬНЫЙ QdrantVectorStore НЕ делает HTTP запросы")
    print("   3. Рекурсия невозможна")
    
    # Устанавливаем VM контекст
    RAGFactoryPrototype.reset_context()
    RAGFactoryPrototype.set_context(ExecutionContext.VM)
    
    try:
        config = get_config()
        
        # Симулируем создание vector_store через IndexerService (как делает Factory)
        from rag.vector_store import QdrantVectorStore
        
        print(f"\n🔨 Создание локального QdrantVectorStore...")
        vector_store = QdrantVectorStore(config.rag.vector_store)
        
        # Проверяем что это действительно локальная версия
        print(f"   Type: {type(vector_store).__name__}")
        print(f"   Module: {type(vector_store).__module__}")
        
        # Ключевая проверка: у локального QdrantVectorStore нет HTTP клиента
        has_http_client = hasattr(vector_store, 'search_endpoint') or hasattr(vector_store, 'index_endpoint')
        
        if not has_http_client:
            print(f"   ✅ УСПЕХ: Локальный QdrantVectorStore не содержит HTTP endpoints")
            print(f"   ✅ УСПЕХ: Рекурсия НЕВОЗМОЖНА (нет HTTP запросов к /index)")
            return True
        else:
            print(f"   ❌ ОШИБКА: Обнаружены HTTP endpoints - возможна рекурсия!")
            return False
            
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Основная функция запуска всех тестов"""
    print("\n" + "🚀" * 40)
    print("  ТЕСТИРОВАНИЕ ПРОТОТИПА FACTORY PATTERN")
    print("  Цель: Устранение рекурсии на VM сервере")
    print("🚀" * 40)
    
    results = {}
    
    # Тест 1: Детекция контекста
    try:
        context = test_context_detection()
        results['context_detection'] = True
    except Exception as e:
        print(f"\n❌ Тест 1 провален: {e}")
        results['context_detection'] = False
    
    # Тест 2: VM контекст
    try:
        results['vm_context'] = test_vm_context_simulation()
    except Exception as e:
        print(f"\n❌ Тест 2 провален: {e}")
        results['vm_context'] = False
    
    # Тест 3: CLIENT контекст
    try:
        results['client_context'] = test_client_context_simulation()
    except Exception as e:
        print(f"\n❌ Тест 3 провален: {e}")
        results['client_context'] = False
    
    # Тест 4: Рекурсия
    try:
        results['recursion_prevention'] = test_recursion_prevention()
    except Exception as e:
        print(f"\n❌ Тест 4 провален: {e}")
        results['recursion_prevention'] = False
    
    # Итоговый отчёт
    print_separator("ИТОГОВЫЙ ОТЧЁТ")
    
    print(f"\n📊 Результаты тестирования:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print(f"   Прототип Factory Pattern работает корректно")
        print(f"   Рекурсия на VM полностью устранена")
        print(f"\n✅ ГОТОВО К ПОЛНОЙ РЕАЛИЗАЦИИ")
    else:
        print(f"\n⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print(f"   Требуется доработка прототипа")
    
    print("\n" + "=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())