# Защита от OOM (Out Of Memory) Killer

**Дата:** 01.10.2025
**Версия:** 1.0.0
**Статус:** ✅ РЕАЛИЗОВАНО

---

## 🚨 Проблема

При индексации больших репозиториев на VM с ограниченной памятью (32Gi) процесс `vm_rag_service.py` убивался Linux OOM killer из-за превышения лимитов памяти.

**Симптомы:**
```bash
# На VM
Killed

# В логах клиента
Circuit breaker OPEN: VM сервис недоступен
```

---

## ✅ Решение

Добавлена **динамическая подстройка batch_size** на основе текущего использования памяти.

### Реализация

**Файл:** `rag/indexer_service.py`

**Новый метод:** `_check_memory_and_adjust_batch(current_batch_size: int) -> int`

```python
def _check_memory_and_adjust_batch(self, current_batch_size: int) -> int:
    """
    Динамическая подстройка batch_size на основе памяти.
    
    Защита от OOM killer:
    - При >85% памяти: уменьшаем batch в 4 раза (минимум 32)
    - При >75% памяти: уменьшаем batch в 2 раза (минимум 64)
    - При <50% памяти: увеличиваем batch в 2 раза (максимум 512)
    """
    memory = psutil.virtual_memory()
    mem_percent = memory.percent
    
    if mem_percent > 85:
        return max(32, current_batch_size // 4)
    elif mem_percent > 75:
        return max(64, current_batch_size // 2)
    elif mem_percent < 50 and current_batch_size < 512:
        return min(512, current_batch_size * 2)
    
    return current_batch_size
```

---

## 📊 Логика работы

### Уровни защиты

| Использование памяти | Действие | Новый batch_size |
|---------------------|----------|------------------|
| **>85%** (Критично) | 🚨 Агрессивное уменьшение | `max(32, size // 4)` |
| **>75%** (Высоко) | ⚠️ Умеренное уменьшение | `max(64, size // 2)` |
| **50-75%** (Норма) | ✅ Без изменений | `size` |
| **<50%** (Низко) | 📈 Увеличение | `min(512, size * 2)` |

### Пример работы

```
Начальный batch_size: 512

Память: 45% → batch_size: 512 (норма)
Память: 76% → batch_size: 256 (уменьшение)
Память: 87% → batch_size: 64 (критично!)
Память: 48% → batch_size: 128 (восстановление)
Память: 42% → batch_size: 256 (восстановление)
```

---

## 🔧 Интеграция

### Изменения в `_index_chunks_batch()`

**Было:**
```python
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    # процесс индексации
```

**Стало:**
```python
current_batch_size = batch_size
i = 0
while i < len(chunks):
    # Проверяем память перед каждым batch
    current_batch_size = self._check_memory_and_adjust_batch(current_batch_size)
    
    batch = chunks[i:i + current_batch_size]
    # процесс индексации
    
    i += len(batch)
```

---

## 📝 Логирование

### Примеры логов

**Критический уровень:**
```
🚨 Критический уровень памяти: 87.2% (доступно: 4.1Gi).
Уменьшаем batch_size: 256 → 64
```

**Высокий уровень:**
```
⚠️ Высокий уровень памяти: 76.5% (доступно: 7.5Gi).
Уменьшаем batch_size: 512 → 256
```

**Восстановление:**
```
✅ Низкий уровень памяти: 48.3% (доступно: 16.5Gi).
Увеличиваем batch_size: 128 → 256
```

---

## 🎯 Визуализация в UI

Progress bar показывает:
```
Батч 15 (size=256, mem=76.2%)
Батч 16 (size=128, mem=82.1%)
Батч 17 (size=64, mem=88.5%)
```

---

## ⚙️ Конфигурация

