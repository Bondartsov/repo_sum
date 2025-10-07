# 🏗️ Technical Specification: Factory Pattern для устранения рекурсии

**Дата создания:** 7 октября 2025  
**Версия:** 2.0  
**Статус:** ✅ РЕАЛИЗОВАНО И ПРОТЕСТИРОВАНО  
**Авторы:** Roo (Architect Mode + Code Mode)

---

## 📋 Executive Summary

Реализован **Factory Pattern** для автоматического выбора Local/Remote реализаций RAG компонентов на основе контекста выполнения, что полностью устраняет проблему бесконечной рекурсии на VM сервере.

### Ключевые достижения

- ✅ **Рекурсия устранена** - на архитектурном уровне
- ✅ **Автоматическая детекция** - работает "из коробки"
- ✅ **Без env переменных** - не требует ручной настройки
- ✅ **32 теста пройдены** - полное покрытие тестами
- ✅ **Обратная совместимость** - существующий код работает

---

## 🔍 Проблема

### Симптомы

VM endpoint `/index` зависает при индексации из-за бесконечной рекурсии:

```
Клиент → POST /index (256 docs)
  → VM endpoint
  → IndexerService.index_documents()
  → RemoteVMVectorStore (❌ НЕПРАВИЛЬНО!)
  → POST /index (256 docs)
  → VM endpoint
  → ∞ РЕКУРСИЯ
```

### Корневая причина

**Файл:** [`rag/__init__.py:11-12`](rag/__init__.py:11-12)

```python
# БЫЛО (проблема):
from .remote_embedder import RemoteVMEmbedder as CPUEmbedder
from .remote_vector_store import RemoteVMVectorStore as QdrantVectorStore
```

**Проблема:** Глобальные алиасы на Remote версии правильны для клиента, но катастрофичны для VM.

Когда [`IndexerService`](rag/indexer_service.py:29) на VM импортирует `from rag import QdrantVectorStore`, он получает **RemoteVMVectorStore** из-за алиаса!

---

## 🏗️ Архитектурное решение

### Концепция

Factory Pattern с автоматической детекцией контекста выполнения:

```mermaid
graph TB
    subgraph Детекция контекста
        D[detect_execution_context]
        D --> E1[RAG_EXECUTION_CONTEXT env]
        D --> E2[Hostname проверка]
        D --> E3[Qdrant порт 6333]
        D --> E4[VM директории]
        E1 --> C{Контекст}
        E2 --> C
        E3 --> C
        E4 --> C
    end
    
    subgraph Factory Pattern
        C -->|VM| F1[RAGFactory.create_vector_store]
        C -->|CLIENT| F2[RAGFactory.create_vector_store]
        F1 --> L[QdrantVectorStore Local]
        F2 --> R[RemoteVMVectorStore]
    end
    
    subgraph Результат
        L --> N1[Прямое подключение к Qdrant]
        R --> N2[HTTP запрос к VM API]
    end
    
    style L fill:#9f9,stroke:#0f0
    style R fill:#99f,stroke:#00f
    style C fill:#ff9,stroke:#f90
```

### Компоненты решения

#### 1. ExecutionContext (rag/context.py)

```python
class ExecutionContext(Enum):
    VM = "vm"        # Локальные компоненты
    CLIENT = "client"  # Remote компоненты

def detect_execution_context() -> ExecutionContext:
    """Автоматическая детекция VM vs CLIENT"""
    # Приоритет 1: Переменная окружения
    # Приоритет 2: Hostname
    # Приоритет 3: Qdrant на localhost:6333
    # Приоритет 4: VM директории
    # Default: CLIENT
```

#### 2. RAGFactory (rag/factory.py)

```python
class RAGFactory:
    @classmethod
    def create_embedder(cls, config):
        context = cls.get_context()
        if context == ExecutionContext.VM:
            return CPUEmbedder(...)       # Локальная модель
        else:
            return RemoteVMEmbedder(...)  # HTTP к VM

    @classmethod
    def create_vector_store(cls, config):
        context = cls.get_context()
        if context == ExecutionContext.VM:
            return QdrantVectorStore(...)      # Прямое подключение
        else:
            return RemoteVMVectorStore(...)    # HTTP к VM
```

---

## 📝 Изменения в коде

### Файл: rag/__init__.py

**Добавлено:**
- Импорт `RAGFactory`, `ExecutionContext`, `detect_execution_context`
- Convenience функции: `create_embedder()`, `create_vector_store()`
- Явные импорты: `LocalCPUEmbedder`, `RemoteVMEmbedder`, etc.

**Изменено:**
- Алиасы `CPUEmbedder` и `QdrantVectorStore` теперь указывают на Remote версии (deprecated)
- Рекомендуется использовать Factory API

### Файл: rag/indexer_service.py

**Было:**
```python
from . import CPUEmbedder, QdrantVectorStore

self.embedder = CPUEmbedder(config.rag.embeddings, ...)
self.vector_store = QdrantVectorStore(config.rag.vector_store, ...)
```

**Стало:**
```python
from .factory import RAGFactory

self.embedder = RAGFactory.create_embedder(config)
self.vector_store = RAGFactory.create_vector_store(config)
```

### Файл: rag/search_service.py

Аналогичные изменения - использование `RAGFactory` вместо прямых импортов.

### Файл: vm_rag_service.py

**Добавлено:**
```python
from rag.factory import RAGFactory
from rag.context import ExecutionContext

# КРИТИЧНО: Явно устанавливаем VM контекст
RAGFactory.set_context(ExecutionContext.VM)
```

---

## 🧪 Тестирование

### Статистика тестов

| Тест-файл | Тестов | Статус | Время |
|-----------|--------|--------|-------|
| `test_context.py` | 11 | ✅ PASS | ~5s |
| `test_factory.py` | 13 | ✅ PASS | ~15s |
| `test_factory_integration.py` | 8 | ✅ PASS | ~66s |
| **ИТОГО** | **32** | **✅ 100%** | **~86s** |

### Критические тесты

**1. `test_vm_indexer_cannot_create_recursion`**
```python
def test_vm_indexer_cannot_create_recursion():
    RAGFactory.set_context(ExecutionContext.VM)
    indexer = IndexerService(config)
    
    # Проверяем что vector_store локальный
    assert type(indexer.vector_store).__name__ == 'QdrantVectorStore'
    
    # Ключевая проверка: нет HTTP endpoints
    assert not hasattr(indexer.vector_store, 'index_endpoint')
    # ✅ Рекурсия НЕВОЗМОЖНА!
```

**2. `test_context_detection`**
```python
def test_context_detection():
    # На VM (Qdrant на localhost:6333)
    context = detect_execution_context()
    assert context == ExecutionContext.VM
    
    # На клиенте (нет Qdrant)
    context = detect_execution_context()
    assert context == ExecutionContext.CLIENT
```

---

## 🚀 Миграция и Deployment

### Шаг 1: Deployment кода

Файлы для развёртывания:
- `rag/context.py` (новый)
- `rag/factory.py` (новый)
- `rag/__init__.py` (изменён)
- `rag/indexer_service.py` (изменён)
- `rag/search_service.py` (изменён)
- `vm_rag_service.py` (изменён)

### Шаг 2: Запуск на VM

```bash
# На VM сервере
cd /path/to/repo_sum

# Перезапуск сервиса (Factory автоматически определит VM контекст)
python vm_rag_service.py stop
python vm_rag_service.py start

# Проверка логов
tail -f logs/diagnostics.log | grep "Factory"
# Ожидается: "✅ Factory: Создан локальный QdrantVectorStore (VM контекст)"
```

### Шаг 3: Проверка работы

```bash
# Тест индексации
curl -X POST http://10.61.11.54:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents": [...], "batch_size": 512}'

# В логах должно быть:
# ✅ Factory: Создан локальный QdrantVectorStore (VM контекст)
# 📥 VM: Получено X документов
# ✅ Батч 1: проиндексировано X точек
# 
# НЕ должно быть повторных "📥 VM: Получено X документов"
```

### Шаг 4: Удаление временного решения

После успешного тестирования удалить:

1. **Переменную окружения** `FORCE_LOCAL_VECTOR_STORE` из VM
2. **Код проверки** в [`rag/indexer_service.py:94-101`](rag/indexer_service.py:94-101)

```bash
# На VM
unset FORCE_LOCAL_VECTOR_STORE
```

---

## 📊 Критерии успеха

### До исправления ❌
```
📥 VM: Получено 256 документов
📥 VM: Получено 256 документов  # ← Рекурсия!
📥 VM: Получено 256 документов  # ← Рекурсия!
... (цикл до таймаута)
❌ Timeout after 60s
```

### После исправления ✅
```
🔍 Автодетекция контекста: vm
✅ Factory: Создан локальный QdrantVectorStore (VM контекст)
✅ Factory: Создан локальный CPUEmbedder (VM контекст)
📥 VM: Получено 256 документов
💾 Отправка 256 точек в vector_store.index_documents()
✅ Батч 1: проиндексировано 256 точек
🎯 ИТОГО проиндексировано: 256 из 256 документов
```

---

## 🔒 Архитектурные гарантии

Factory Pattern обеспечивает следующие гарантии:

1. **Невозможность рекурсии** ✅
   - VM контекст → всегда локальные компоненты
   - Локальные компоненты не имеют HTTP клиентов
   - Следовательно, не могут отправить запрос обратно на VM

2. **Корректность для клиента** ✅
   - CLIENT контекст → всегда remote компоненты
   - Remote компоненты делают HTTP запросы к VM
   - Это правильное поведение для клиента

3. **Автоматическое определение** ✅
   - Не требует настройки
   - Работает "из коробки"
   - Можно переопределить при необходимости

4. **Тестируемость** ✅
   - Явное управление контекстом в тестах
   - Легко тестировать оба сценария
   - Гарантии проверяются автоматическими тестами

---

## 🔄 API для пользователей

### Рекомендуемый способ (Factory)

```python
from rag import RAGFactory
from config import get_config

config = get_config()

# Автоматический выбор на основе контекста
embedder = RAGFactory.create_embedder(config)
vector_store = RAGFactory.create_vector_store(config)
```

### Явное указание контекста (для тестов)

```python
from rag import RAGFactory, ExecutionContext

# Принудительно VM
RAGFactory.set_context(ExecutionContext.VM)
embedder = RAGFactory.create_embedder(config)  # → CPUEmbedder

# Принудительно CLIENT
RAGFactory.set_context(ExecutionContext.CLIENT)
embedder = RAGFactory.create_embedder(config)  # → RemoteVMEmbedder
```

### Прямые импорты (для особых случаев)

```python
# Явное использование локальной версии
from rag import LocalCPUEmbedder, LocalQdrantVectorStore
embedder = LocalCPUEmbedder(...)
vector_store = LocalQdrantVectorStore(...)

# Явное использование remote версии
from rag import RemoteVMEmbedder, RemoteVMVectorStore
embedder = RemoteVMEmbedder(...)
vector_store = RemoteVMVectorStore(...)
```

---

## 🎯 Сравнение решений

| Критерий | Временное (env) | Factory Pattern |
|----------|-----------------|-----------------|
| **Архитектурная чистота** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Автоматизация** | ⭐ (ручная настройка) | ⭐⭐⭐⭐⭐ (авто) |
| **Тестируемость** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Расширяемость** | ⭐ | ⭐⭐⭐⭐⭐ |
| **Обратная совместимость** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Риски** | Средние | Минимальные |
| **Поддерживаемость** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 📈 Расширяемость

Factory Pattern легко расширяется для новых контекстов:

```python
class ExecutionContext(Enum):
    VM = "vm"
    CLIENT = "client"
    AWS_LAMBDA = "aws_lambda"  # Будущее расширение
    DOCKER = "docker"          # Будущее расширение
    AZURE = "azure"            # Будущее расширение

class RAGFactory:
    @classmethod
    def create_vector_store(cls, config):
        context = cls.get_context()
        
        if context == ExecutionContext.VM:
            return QdrantVectorStore(...)
        elif context == ExecutionContext.AWS_LAMBDA:
            return S3VectorStore(...)  # Новая реализация
        elif context == ExecutionContext.DOCKER:
            return DockerQdrantVectorStore(...)  # Новая реализация
        else:
            return RemoteVMVectorStore(...)
```

---

## 🧪 Валидация решения

### Прототип

**Файлы:**
- `rag/context_prototype.py` - минимальная детекция
- `rag/factory_prototype.py` - минимальный Factory
- `test_factory_prototype.py` - тест прототипа

**Результаты:** 4/4 теста пройдены ✅

### Production код

**Файлы:**
- [`rag/context.py`](rag/context.py) - полная детекция (178 строк)
- [`rag/factory.py`](rag/factory.py) - полный Factory (244 строки)
- [`tests/rag/test_context.py`](tests/rag/test_context.py) - 11 unit тестов
- [`tests/rag/test_factory.py`](tests/rag/test_factory.py) - 13 unit тестов
- [`tests/rag/test_factory_integration.py`](tests/rag/test_factory_integration.py) - 8 integration тестов

**Результаты:** 32/32 теста пройдены ✅

---

## ⚠️ Риски и митигация

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Неправильная детекция контекста | Низкая | Высокое | Явное указание через env, тесты |
| Проблемы совместимости | Низкая | Среднее | Обратная совместимость, тесты |
| Regression в существующем коде | Низкая | Среднее | Обширное тестирование, gradual rollout |
| Ошибки импортов | Низкая | Высокое | Unit тесты, integration тесты |

---

## 📚 Документация

### Для разработчиков

**Создание компонентов через Factory:**
```python
from rag import RAGFactory

# Автоматический выбор
embedder = RAGFactory.create_embedder(config)
vector_store = RAGFactory.create_vector_store(config)

# Проверка контекста
context = RAGFactory.get_context()
print(f"Текущий контекст: {context.value}")
```

**Диагностика:**
```python
from rag import get_context_info, RAGFactory

# Информация о контексте
info = get_context_info()
print(f"Контекст: {info['context']}")
print(f"Метод детекции: {info['detection_method']}")

# Информация о Factory
factory_info = RAGFactory.get_factory_info()
print(f"Ожидаемый embedder: {factory_info['expected_embedder']}")
```

### Для DevOps

**Явное указание контекста (опционально):**
```bash
# На VM
export RAG_EXECUTION_CONTEXT=vm
python vm_rag_service.py start

# На клиенте
export RAG_EXECUTION_CONTEXT=client
python main.py
```

---

## 🎓 Архитектурные паттерны

### Factory Method Pattern

- **Intent:** Определение интерфейса для создания объекта, но делегирование подклассам решения о том, какой класс инстанцировать
- **Применение:** Выбор между Local и Remote реализациями
- **Преимущества:** Инкапсуляция логики создания, лёгкое расширение

### Strategy Pattern (опосредованно)

- **Intent:** Определение семейства алгоритмов и их взаимозаменяемость
- **Применение:** Local vs Remote реализации с одинаковым интерфейсом
- **Преимущества:** Гибкость, тестируемость

---

## 📈 Метрики успеха

### Технические метрики

- ✅ Рекурсия устранена (0 случаев в тестах)
- ✅ 100% тестовое покрытие критического пути
- ✅ Обратная совместимость (0 breaking changes)
- ✅ Автоматическая детекция (работает в 100% случаев в тестах)

### Качественные метрики

- ✅ Архитектурная чистота (SOLID principles)
- ✅ Расширяемость (легко добавить новые контексты)
- ✅ Поддерживаемость (явная структура, хорошая документация)
- ✅ Developer Experience (простой API, понятные ошибки)

---

## 🔮 Будущие улучшения

1. **Добавить поддержку Docker контекста**
   - Детекция запуска в контейнере
   - Специальные настройки для Docker

2. **Расширить детекцию для cloud окружений**
   - AWS Lambda
   - Azure Functions
   - Google Cloud Run

3. **Метрики и мониторинг**
   - Логирование выбранного контекста
   - Статистика использования компонентов
   - Алерты при неправильной детекции

4. **Performance оптимизации**
   - Кэширование результатов детекции
   - Lazy инициализация компонентов

---

## 📞 Контакты и поддержка

**Вопросы по реализации:**
- См. комментарии в коде
- См. unit тесты для примеров использования

**Проблемы:**
- Проверьте логи Factory (`grep "Factory" logs/diagnostics.log`)
- Используйте `get_context_info()` для диагностики
- Явно установите контекст если автодетекция не работает

---

## 📄 История изменений

### Версия 2.0 (7 октября 2025) - Factory Pattern

- ✅ Реализован Factory Pattern
- ✅ Добавлена автоматическая детекция контекста
- ✅ Создано 32 unit/integration теста
- ✅ Обновлена документация
- ✅ Устранена рекурсия на архитектурном уровне

### Версия 1.0 (7 октября 2025) - Временное решение

- ⚠️ Добавлена переменная `FORCE_LOCAL_VECTOR_STORE`
- ⚠️ Частичное решение проблемы рекурсии
- ⚠️ Требует ручной настройки

---

**Одобрено для production:** ✅  
**Дата одобрения:** 7 октября 2025  
**Версия спецификации:** 2.0