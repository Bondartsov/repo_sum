# AGENTS.md

This file provides guidance to agents when working with code in this repository.

---

## 📚 Консолидированная документация

**НАЧНИТЕ ЗДЕСЬ**: Вся актуальная информация о проекте консолидирована согласно стандартам Memory Bank:

### Центральная документация:
- 🗺️ **[ROADMAP.md](ROADMAP.md)** - **ГЛАВНЫЙ ДОКУМЕНТ** с полной roadmap, milestone, техническими деталями и планами развития
- 📋 **[.clinerules/](.clinerules/)** - Memory Bank система с 6 ключевыми файлами

### Memory Bank структура (.clinerules/):
- 📊 **[projectbrief.md](.clinerules/projectbrief.md)** - суть проекта, цели, требования
- 🎯 **[productContext.md](.clinerules/productContext.md)** - продуктовый контекст, пользователи, сценарии

### Вспомогательные файлы:
- 📋 **[custom_instructions.md](.clinerules/custom_instructions.md)** - правила работы с Memory Bank
- 🧪 **[MOCKS_RULES.md](.clinerules/MOCKS_RULES.md)** - правила для mock объектов
- 🔍 **[audit_results.md](.clinerules/audit_results.md)** - результаты технического аудита

> ⚠️ **ВАЖНО**: При работе с проектом сначала изучите ROADMAP.md и .clinerules/ для полного контекста


## Нестандартные команды
- **Запуск ядра**: `python main.py`  
- **Запуск веб-интерфейса**: `python run_web.py`  
- **Проверка зависимостей**: `python scripts/verify_requirements.py` (кастомная логика, не стандартный `pip check`)  
- **Запуск RAG-тестов**:  
  - Все тесты: `pytest tests/rag/`  
  - Специальный раннер: `python tests/rag/run_rag_tests.py`  

---

## Уникальные паттерны
- **Гибридный поиск**: объединение dense (FastEmbed) и sparse (BM25 + кастомный sparse-энкодер).  
- **Собственный гибридный эмбеддер**: реализован в [`rag/embedder.py`](rag/embedder.py).  
- **Fail-fast валидация**: ошибки должны выявляться максимально рано.  
- **CPU-first оптимизация**: алгоритмы оптимизированы под CPU, GPU не требуется.  
- **Генерация документации через LLM**: пайплайн [`file_scanner.py`](file_scanner.py) → [`code_chunker.py`](code_chunker.py) → [`doc_generator.py`](doc_generator.py).  
- **Правила разработки**: все изменения и исключения фиксируются в `.clinerules/`.  

---

## Особенности тестирования
- **Кастомный раннер**: `tests/rag/run_rag_tests.py` — обязательный для RAG-модулей.  
- **Строгая категоризация тестов**: юнит, интеграционные, e2e, property-based.  
- **Property-based тесты**: реализованы через `hypothesis`.
- **Контракты и стратегии**: см. [`tests/rag/TESTING_STRATEGY.md`](tests/rag/TESTING_STRATEGY.md).
- **Offline/mock режим обязателен**:
  - Все новые тесты должны работать без сети.
  - Сетевые вызовы замоканы.
  - Для проверки использовать [`tests/test_offline_no_network.py`](tests/test_offline_no_network.py) как эталон.

---

## Важные напоминания
- Следовать PEP8 + SOLID + DRY.  
- Строгая типизация обязательна.  
- Все изменения должны сопровождаться обновлением правил в `.clinerules/`.
