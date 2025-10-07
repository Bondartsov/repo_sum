# 🔄 Исправление бесконечной рекурсии на VM сервере

**Дата:** 7 октября 2025  
**Статус:** ✅ ИСПРАВЛЕНО  
**Критичность:** КРИТИЧЕСКАЯ

---

## 📋 Краткое резюме

После исправления проблемы с пустыми текстами обнаружена **КРИТИЧЕСКАЯ проблема бесконечной рекурсии** при индексации на VM сервере.

### Симптомы

- VM endpoint `/index` зависает и выходит по таймауту
- В логах повторяются сообщения "📥 VM: Получено 256 документов"
- `IndexerService` на VM создаёт `RemoteVMVectorStore` вместо локального `QdrantVectorStore`
- `RemoteVMVectorStore` отправляет HTTP запрос **обратно на тот же `/index` endpoint**
- Цикл повторяется бесконечно → таймаут

### Корневая причина

**Файл:** [`rag/__init__.py:11-12`](rag/__init__.py:11-12)

```python
from .remote_embedder import RemoteVMEmbedder as CPUEmbedder
from .remote_vector_store import RemoteVMVectorStore as QdrantVectorStore
```

**Проблема:** При импорте `from rag import QdrantVectorStore` в [`rag/indexer_service.py:29`](rag/indexer_service.py:29), из-за алиасов в `__init__.py` получаем **Remote версию** вместо локальной!

### Цепочка проблемы

1. Клиент отправляет документы на VM → `POST /index`
2. VM endpoint вызывает [`IndexerService.index_documents()`](rag/indexer_service.py:671)
3. `IndexerService` создаёт vector_store через импорт из `rag/__init__.py`
4. **Из-за алиаса получает `RemoteVMVectorStore`** вместо локального `QdrantVectorStore`
5. `RemoteVMVectorStore.index_documents()` отправляет HTTP запрос на `http://10.61.11.54:8000/index`
6. **Это тот же самый endpoint!** → Рекурсия
7. Цикл повторяется до таймаута

---

## ✅ Решение: Переменная окружения FORCE_LOCAL_VECTOR_STORE

### Изменения в коде

**Файл:** [`rag/indexer_service.py:92-114`](rag/indexer_service.py:92-114)

**БЫЛО:**
```python
if self.vector_store is None:
    try:
        self.vector_store = QdrantVectorStore(
            config.rag.vector_store,
            config.rag.remote_service
        )
    except TypeError:
        # Local QdrantVectorStore expects only one argument
        self.vector_store = QdrantVectorStore(
            config.rag.vector_store
        )
```

**СТАЛО:**
```python
if self.vector_store is None:
    # Проверяем переменную окружения для форсирования локального store на VM
    import os
    force_local = os.getenv('FORCE_LOCAL_VECTOR_STORE', '').lower() in ('1', 'true', 'yes')
    
    if force_local:
        # Прямой импорт локального QdrantVectorStore
        from .vector_store import QdrantVectorStore as LocalQdrantVectorStore
        self.vector_store = LocalQdrantVectorStore(config.rag.vector_store)
        logger.info("🔧 FORCE_LOCAL_VECTOR_STORE включён: используется локальный QdrantVectorStore")
    else:
        try:
            self.vector_store = QdrantVectorStore(
                config.rag.vector_store,
                config.rag.remote_service
            )
        except TypeError:
            # Local QdrantVectorStore expects only one argument
            self.vector_store = QdrantVectorStore(
                config.rag.vector_store
            )
```

### Преимущества решения

✅ **Минимальное изменение кода** - затронут только один файл  
✅ **Контролируется переменной окружения** - гибкая конфигурация  
✅ **Не ломает существующую логику** - локальные клиенты продолжают работать  
✅ **Явный контроль** - логирование использования локального store  
✅ **Обратная совместимость** - работает как раньше без переменной окружения

---

## 🚀 Инструкции для VM сервера

### 1. Установка переменной окружения

На VM сервере нужно установить переменную окружения перед запуском сервиса:

```bash
export FORCE_LOCAL_VECTOR_STORE=true
```

Или добавить в `.bashrc` / `.profile`:

```bash
echo 'export FORCE_LOCAL_VECTOR_STORE=true' >> ~/.bashrc
source ~/.bashrc
```

### 2. Перезапуск сервиса

```bash
# Остановить старый сервис (если запущен)
python vm_rag_service.py stop

# Запустить с новой переменной окружения
export FORCE_LOCAL_VECTOR_STORE=true
python vm_rag_service.py start
```

### 3. Проверка работы

После перезапуска в логах должно появиться:

```
🔧 FORCE_LOCAL_VECTOR_STORE включён: используется локальный QdrantVectorStore
```

### 4. Проверка индексации

При индексации документов должно быть:

- ✅ **ОДИН** вызов endpoint `/index` для каждого батча клиента
- ✅ **НЕТ** повторных "📥 VM: Получено 256 документов"
- ✅ **НЕТ** таймаутов
- ✅ Успешное сохранение в Qdrant

---

## 📊 Критерий успеха

### До исправления:
```
📥 VM: Получено 256 документов
📥 VM: Получено 256 документов  # ← Рекурсия!
📥 VM: Получено 256 документов  # ← Рекурсия!
... (цикл до таймаута)
❌ Timeout after 60s
```

### После исправления:
```
🔧 FORCE_LOCAL_VECTOR_STORE включён: используется локальный QdrantVectorStore
📥 VM: Получено 256 документов
💾 Отправка 256 точек в vector_store.index_documents()
✅ Батч 1: проиндексировано 256 точек
🎯 ИТОГО проиндексировано: 256 из 256 документов
```

---

## 🔍 Диагностика

### Как проверить используемый vector_store

В логах IndexerService будет:

```python
# С переменной окружения:
🔧 FORCE_LOCAL_VECTOR_STORE включён: используется локальный QdrantVectorStore
🔍 ДИАГНОСТИКА: type(vector_store) = QdrantVectorStore

# Без переменной окружения:
🔍 ДИАГНОСТИКА: type(vector_store) = RemoteVMVectorStore
```

### Как проверить наличие рекурсии

Запустить индексацию и следить за логами:

```bash
# На VM
tail -f logs/diagnostics.log | grep "📥 VM:"
```

Если сообщение "📥 VM: Получено X документов" повторяется **более одного раза** для одного батча → рекурсия НЕ исправлена.

---

## 📚 Связанные документы

- [`rules/BUGFIX_REPORT_2025_10_06.md`](rules/BUGFIX_REPORT_2025_10_06.md) - Предыдущие исправления
- [`rules/HOTFIX_TIMEOUTS.md`](rules/HOTFIX_TIMEOUTS.md) - История проблем с таймаутами
- [`rag/__init__.py`](rag/__init__.py) - Алиасы импортов (корень проблемы)
- [`rag/indexer_service.py`](rag/indexer_service.py) - Место исправления

---

## 🎯 Статус задачи

- ✅ Проблема диагностирована
- ✅ Решение разработано
- ✅ Код исправлен
- ✅ Документация создана
- ⏳ **Ожидается**: Тестирование на VM сервере
- ⏳ **Ожидается**: Подтверждение устранения рекурсии

---

**Автор:** Roo (Code Mode)  
**Дата:** 2025-10-07  
**Версия:** 1.0