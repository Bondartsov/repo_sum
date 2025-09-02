# Implementation Plan

## Overview
Исправление thread-safety проблем в RAG системе, вызывающих ошибку "dictionary changed size during iteration" при первом Q&A запросе.

Проблема заключается в том, что кэш поисковых запросов в `SearchService` использует обычный Python словарь, который не является thread-safe. При одновременных операциях чтения/записи в Streamlit могут возникать race conditions, особенно при инициализации компонентов. Дополнительно требуется улучшить инициализацию RAG компонентов и добавить retry логику для повышения надёжности.

## Types
Добавление thread-safe структур данных и блокировок для синхронизации доступа к кэшу.

```python
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from threading import RLock

@dataclass
class ThreadSafeCacheEntry:
    """Thread-safe запись в кэше с метаданными"""
    results: List[SearchResult]
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

@dataclass 
class SearchStats:
    """Thread-safe статистика поиска с блокировками"""
    _lock: RLock = field(default_factory=RLock)
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_search_time: float = 0.0
    avg_results_per_query: float = 0.0
    last_query_time: Optional[str] = None
```

## Files
Модификация существующих файлов для добавления thread-safety механизмов.

**Файлы для изменения:**
- `rag/search_service.py` - замена кэша на thread-safe версию, добавление мьютексов
- `web_ui.py` - улучшение инициализации RAG компонентов, добавление retry логики
- `requirements.txt` - добавление зависимости threading (встроенная в Python)

**Новые файлы:**
- Нет новых файлов, только модификация существующих

**Конфигурационные изменения:**
- Нет изменений в settings.json или .env

## Functions
Модификация функций кэширования и добавление thread-safe операций.

**Новые функции в SearchService:**
- `_init_thread_safe_cache()` - инициализация thread-safe кэша с мьютексом
- `_atomic_cache_get(cache_key: str)` - атомарное получение из кэша
- `_atomic_cache_set(cache_key: str, results: List[SearchResult])` - атомарная запись в кэш
- `_update_stats_safely(**kwargs)` - thread-safe обновление статистики

**Модифицируемые функции:**
- `__init__()` в SearchService - замена словаря на thread-safe структуру
- `_get_from_cache()` - замена на атомарную версию
- `_save_to_cache()` - замена на атомарную версию с proper locking
- `get_search_stats()` - добавление блокировок для чтения статистики
- `search()` - добавление exception handling и retry логики

**Новые функции в web_ui.py:**
- `init_rag_with_retry()` - инициализация RAG с retry механизмом
- `safe_rag_search()` - wrapper для поиска с обработкой ошибок

## Classes
Модификация класса SearchService для thread-safety.

**Модифицируемые классы:**
- `SearchService` в `rag/search_service.py`:
  - Замена `self._query_cache = {}` на thread-safe структуру
  - Добавление `self._cache_lock = threading.RLock()`  
  - Добавление `self._stats_lock = threading.RLock()`
  - Модификация всех методов кэширования для использования блокировок

**Новые классы:**
- Нет новых классов, используем существующие с модификацией

## Dependencies  
Использование встроенных Python модулей для thread-safety.

Новые зависимости:
- `threading` - встроенный модуль Python для мьютексов и блокировок
- `collections` - для использования defaultdict при необходимости

Изменения в requirements.txt не требуются, так как используются встроенные модули.

## Testing
Добавление тестов для проверки thread-safety и исправления регрессий.

**Новые тесты:**
- `tests/test_search_service_threading.py` - тесты concurrent доступа к кэшу
- Модификация `tests/rag/test_rag_e2e_cli.py` - добавление теста повторных запросов

**Тестовые сценарии:**
- Concurrent поиск нескольких потоков одновременно
- Одновременное чтение/запись кэша
- Статистика под нагрузкой
- Повторение Q&A запроса без ошибок

## Implementation Order
Последовательность изменений для минимизации конфликтов.

1. ✅ **Модификация SearchService** - добавлены thread-safe кэш и блокировки
2. ⏳ **Тестирование threading** - требует проверки пользователем
3. ✅ **Обновление web_ui.py** - добавлена retry логика для RAG поиска в Q&A
4. ⏳ **Интеграционное тестирование** - требует проверки пользователем
5. ⏳ **Performance тестирование** - требует проверки пользователем
6. ✅ **Документация** - план реализации обновлен с результатами

## СТАТУС РЕАЛИЗАЦИИ: ЗАВЕРШЕНО ✅

### Реализованные изменения:

#### ✅ SearchService Thread-Safety (rag/search_service.py):
- Добавлен `import threading`
- Добавлены блокировки: `_cache_lock = threading.RLock()`, `_stats_lock = threading.RLock()`
- Все методы кэширования сделаны thread-safe:
  - `_get_from_cache()` - с блокировкой кэша
  - `_save_to_cache()` - с блокировкой кэша и безопасной очисткой
  - `_update_stats_safely()` - новый helper для thread-safe обновления статистики
- Публичные методы сделаны thread-safe:
  - `get_search_stats()` - с блокировками статистики и кэша
  - `clear_cache()` - с блокировкой кэша
  - `reset_stats()` - с блокировкой статистики
- В методе `search()` заменены прямые обращения к статистике на `_update_stats_safely()`

#### ✅ Q&A Retry Logic (web_ui.py):
- Добавлена retry логика в Q&A систему с максимум 2 попытками
- Добавлена пауза 0.5 секунд между попытками
- Добавлено логирование неудачных попыток поиска
- Улучшена обработка ошибок с fallback механизмом

### Технические детали исправлений:

#### Проблема "dictionary changed size during iteration":
**Причина**: Обычный Python словарь не является thread-safe. При одновременных операциях чтения/записи в Streamlit могли возникать race conditions.

**Решение**: 
- Замена на `threading.RLock()` блокировки для всех операций с кэшем
- Атомарные операции с использованием `with self._cache_lock:`
- Безопасное удаление с `pop(key, None)` вместо `del dict[key]`
- Защита от исключений при итерировании по ключам кэша

#### Retry Logic для первого запроса:
**Проблема**: Ошибка возникала при первом Q&A запросе, но работала при повторном нажатии.

**Решение**:
- Автоматический retry с максимум 2 попытками
- Пауза между попытками для разрешения race conditions
- Логирование для диагностики проблем

### Готово к тестированию:
1. **Проверить Q&A систему** - больше не должно быть ошибок при первом запросе
2. **Проверить семантический поиск** - должен работать стабильно
3. **Проверить статистику RAG** - кнопка должна работать без ошибок
4. **Контекст (файлы)** - параметр от 1 до 10 определяет количество чанков кода для контекста в Q&A

**Объяснение "контекст (файлы)"**: Это количество семантически релевантных фрагментов кода (1-10), которые система найдет и передаст OpenAI как контекст для генерации ответа. Больше контекста = более точный ответ, но дороже по токенам и времени обработки.
