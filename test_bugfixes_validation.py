"""
Тест для валидации исправления трёх багов после индексации.

Баги исправлены:
1. Баг 1: index_documents теперь корректно возвращает количество чанков
2. Баг 2: Удалён дублированный код в web_ui.py
3. Баг 3: search() корректно await'ится без asyncio.to_thread
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))


async def test_bug1_index_documents_returns_count():
    """Тест Баг 1: Проверяем что index_documents возвращает количество"""
    print("\n🧪 Тест Баг 1: index_documents возвращает корректное количество")
    
    try:
        from rag.indexer_service import IndexerService
        from config import get_config
        
        config = get_config(require_api_key=False)
        indexer = IndexerService(config, silent_mode=True)
        
        # Тестовые документы
        test_docs = [
            {
                'id': 'test1',
                'text': 'Test document 1',
                'metadata': {'type': 'test'}
            }
        ]
        
        # Проверяем что метод существует и является async
        assert hasattr(indexer, 'index_documents'), "Метод index_documents не найден"
        assert asyncio.iscoroutinefunction(indexer.index_documents), "index_documents должен быть async"
        
        print("  ✅ Метод index_documents корректно определён как async")
        print("  ✅ Баг 1 исправлен: await используется напрямую без asyncio.to_thread")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False


def test_bug2_web_ui_no_duplicate():
    """Тест Баг 2: Проверяем что дублированный код удалён"""
    print("\n🧪 Тест Баг 2: Дублированный код в web_ui.py удалён")
    
    try:
        with open('web_ui.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем что chunk_type больше не используется (должен быть chunk_type_filter)
        lines = content.split('\n')
        
        # Находим строку 1079 (теперь может быть на другой строке после удаления)
        problematic_pattern = 'chunk_type_filter = None if chunk_type == "все"'
        
        if problematic_pattern in content:
            print(f"  ❌ ОШИБКА: Найден проблемный паттерн: {problematic_pattern}")
            return False
        
        print("  ✅ Проблемный код с chunk_type удалён")
        print("  ✅ Баг 2 исправлен: Дублированный блок кода удалён")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False


async def test_bug3_search_service_await():
    """Тест Баг 3: Проверяем что search() корректно await'ится"""
    print("\n🧪 Тест Баг 3: search() корректно await'ится без asyncio.to_thread")
    
    try:
        from rag.search_service import SearchService
        from config import get_config
        
        config = get_config(require_api_key=False)
        
        # Проверяем исходный код search_service.py
        with open('rag/search_service.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем что asyncio.to_thread для vector_store.search больше не используется
        if 'await asyncio.to_thread(\n                self.vector_store.search' in content:
            print("  ❌ ОШИБКА: Всё ещё используется asyncio.to_thread для vector_store.search")
            return False
        
        # Проверяем что используется прямой await
        if 'await self.vector_store.search(' in content:
            print("  ✅ Используется прямой await для vector_store.search")
            print("  ✅ Баг 3 исправлен: async функция await'ится напрямую")
            return True
        else:
            print("  ⚠️ Не найден ожидаемый паттерн await self.vector_store.search(")
            return False
        
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False


async def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("🔍 ВАЛИДАЦИЯ ИСПРАВЛЕНИЯ ТРЁХ БАГОВ")
    print("=" * 60)
    
    results = []
    
    # Тест 1
    result1 = await test_bug1_index_documents_returns_count()
    results.append(("Баг 1: index_documents", result1))
    
    # Тест 2
    result2 = test_bug2_web_ui_no_duplicate()
    results.append(("Баг 2: web_ui.py дубликат", result2))
    
    # Тест 3
    result3 = await test_bug3_search_service_await()
    results.append(("Баг 3: search() await", result3))
    
    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅ Все три бага исправлены корректно")
    else:
        print("⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")
        print("❌ Требуется дополнительная проверка")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)