# 🔧 TIMEOUT FIX: Увеличение таймаутов для батчей 512+ чанков (03.10.2025)

**Дата применения:** 03 октября 2025, 17:55 MSK  
**Причина:** TimeoutError после 10 минут при эмбеддинге 512 чанков  
**Приоритет:** P1 - КРИТИЧЕСКИЙ FIX  
**Цель:** Обеспечить стабильную индексацию больших батчей без timeout

---

## 🎯 Проблема

**Ситуация:**
- ✅ Чанкование работает! Метрики p99 ≤768 токенов
- ⚠️ Timeout после 10 минут при эмбеддинге 512 чанков

**Лог показывает:**
```
Обработка батча из 512 чанков. Память: 0.6GB / 15.6GB (3.7%)
TimeoutError: Retry timeout: 600.0s / 600.0s после 1 попыток
```

**Анализ:**
- Память на VM: 30GB (нормально для Jina v3)
- Проблема: 512 чанков × 400 токенов = большой батч, требует >10 минут
- Пользователь готов ждать часами для полной индексации

---

## 🔧 Применённые Изменения

### 1. [`config.py`](../config.py:237) - RemoteServiceConfig

**Было:**
```python
timeout_seconds: int = 600  # HOTFIX: 10 минут (было 60s)
```

**Стало:**
```python
timeout_seconds: int = 3600  # TIMEOUT FIX: 1 час (было 600s) - для больших батчей (512+ чанков)
```

---

### 2. [`rag/retry_policy.py`](../rag/retry_policy.py:51) - RetryConfig

**Было:**
```python
timeout_seconds: float = 600.0  # HOTFIX: 10 минут общий таймаут (было 60s)
```

**Стало:**
```python
timeout_seconds: float = 3600.0  # TIMEOUT FIX: 1 час (было 600s) - для больших батчей
```

---

### 3. [`rag/remote_embedder.py`](../rag/remote_embedder.py:118) - CircuitBreakerConfig

**Было:**
```python
timeout_seconds=300.0,  # HOTFIX: 5 минут (было 60s)
```

**Стало:**
```python
timeout_seconds=1800.0,  # TIMEOUT FIX: 30 минут (было 300s) - для больших батчей
```

---

### 4. [`rag/remote_embedder.py`](../rag/remote_embedder.py:141) - Адаптивный Timeout

**Добавлено:** Расчёт timeout на основе размера батча

```python
# Адаптивный timeout на основе размера батча
batch_size = len(texts)
estimated_time_per_chunk = 1.5  # секунд на чанк (консервативно)
adaptive_timeout = max(3600, batch_size * estimated_time_per_chunk * 2)  # ×2 запас

logger.info(f"Эмбеддинг батча из {batch_size} чанков. "
            f"Ожидаемое время: {batch_size * estimated_time_per_chunk:.0f}s, "
            f"Timeout: {adaptive_timeout:.0f}s")
```

**Логика:**
- Минимальный timeout: 3600s (1 час)
- Для батча 512 чанков: 512 × 1.5 × 2 = 1536s (26 минут)
- Используется максимум из двух значений: max(3600, 1536) = 3600s

---

### 5. [`rag/remote_embedder.py`](../rag/remote_embedder.py:267) - Прогресс-индикатор эмбеддинга

**Добавлено:** Логирование прогресса при долгих запросах

```python
# Прогресс-индикатор: логируем начало обработки батча
batch_size = len(payload.get('texts', []))
max_attempts = self.retry_policy.config.max_attempts
logger.info(f"⏳ Начинаем эмбеддинг батча из {batch_size} чанков. "
            f"Максимум попыток: {max_attempts}")

# В цикле retry
async def _single_attempt():
    attempt_counter['count'] += 1
    elapsed = time.monotonic() - request_start_time
    logger.info(f"⏳ Эмбеддинг батча... Попытка {attempt_counter['count']}/{max_attempts}, "
                f"Прошло времени: {elapsed:.0f}s")
    return await self.circuit_breaker.call(self._make_single_request, payload=payload)

# После успеха
logger.info(f"✅ Эмбеддинг батча завершён успешно за {elapsed:.0f}s")
```

**Результат:** Пользователь видит, что работа идёт, а не зависла

---

### 6. [`rag/indexer_service.py`](../rag/indexer_service.py:357) - Прогресс-индикатор файлов

**Добавлено:** Прогресс-бар обработки файлов

```python
processed_count = 0
total_files = len(files)

for file_info in files:
    # ... обработка ...
    processed_count += 1
    
    # Прогресс каждые 10 файлов или на последнем файле
    if processed_count % 10 == 0 or processed_count == total_files:
        progress_percent = processed_count / total_files * 100
        bar_filled = int(processed_count * 20 / total_files)
        bar_empty = 20 - bar_filled
        logger.info(f"📂 Обработано файлов: {processed_count}/{total_files} "
                    f"({progress_percent:.1f}%) "
                    f"{'█' * bar_filled}{'░' * bar_empty}")
```

**Пример вывода:**
```
📂 Обработано файлов: 50/135 (37.0%) ████████░░░░░░░░░░░░
📂 Обработано файлов: 100/135 (74.1%) ███████████████░░░░░
📂 Обработано файлов: 135/135 (100.0%) ████████████████████
```

---

### 7. [`rag/indexer_service.py`](../rag/indexer_service.py:333) - Уменьшение batch_size

**Было:**
```python
batch_size: int = 256  # default
```

**Стало:**
```python
batch_size: int = 128  # TIMEOUT FIX: Уменьшено с 256 до 128 для безопасности
```

**Обоснование:**
- Меньший batch = меньше времени на обработку
- Меньше риск timeout
- Более частые обновления прогресса
- Лучше для стабильности на больших репозиториях

---

## 📊 Ожидаемые Результаты

### Для батча 512 чанков:

**Оценка времени:**
- 512 чанков × 1.5s = 768 секунд = **12.8 минут** (ожидаемое)
- Timeout: 3600s = **60 минут** (запас 4.7x)

**С учётом retry:**
- Максимум 5 попыток
- Общий timeout: 3600s на все попытки
- Между попытками: до 120s delay

### Для индексации репозитория 135 файлов:

**С новыми настройками:**
- Batch size: 128 чанков (вместо 256)
- Больше батчей, но меньше риск timeout
- Более частые обновления прогресса
- **Ожидаемое время:** 15-30 минут

---

## ✅ Критерии Успеха

**Минимальный успех:**
- ✅ Индексация 512 чанков завершается БЕЗ timeout
- ✅ Логи показывают прогресс (не зависло)
- ✅ Circuit Breaker остаётся CLOSED

**Полный успех:**
- ✅ Индексация 135 файлов < 30 минут
- ✅ Прогресс-бары информативны
- ✅ Нет неожиданных ошибок

---

## 🚀 Тестирование

### Шаг 1: Перезапуск VM сервиса

```powershell
# Из корня проекта
cd D:\Scripts_Python\repo_sum

# Рестарт VM сервиса (синхронизирует .env)
python vm_start.py restart
```

### Шаг 2: Перезапуск локального приложения

```powershell
# Если web_ui запущен - остановить (Ctrl+C)

# Запустить заново
python run_web.py
```

### Шаг 3: Запуск индексации

```powershell
# В веб-интерфейсе: Index → D:\Scripts_Python\repo_sum → Start
```

### Шаг 4: Мониторинг логов

**Что должны увидеть:**

```
⏳ Начинаем эмбеддинг батча из 512 чанков. Максимум попыток: 5
Эмбеддинг батча из 512 чанков. Ожидаемое время: 768s, Timeout: 3600s
⏳ Эмбеддинг батча... Попытка 1/5, Прошло времени: 2s
...
✅ Эмбеддинг батча завершён успешно за 780s

📂 Обработано файлов: 50/135 (37.0%) ████████░░░░░░░░░░░░
```

**НЕ должно быть:**
- ❌ `TimeoutError: Retry timeout: 600.0s`
- ❌ `Circuit breaker OPEN`
- ❌ `VMTimeoutError`

---

## 📝 Изменения в Коде

### Файлы изменены:

1. ✅ [`config.py`](../config.py:237) - timeout 600s → 3600s
2. ✅ [`rag/retry_policy.py`](../rag/retry_policy.py:51) - timeout 600s → 3600s
3. ✅ [`rag/remote_embedder.py`](../rag/remote_embedder.py:118) - CB timeout 300s → 1800s
4. ✅ [`rag/remote_embedder.py`](../rag/remote_embedder.py:141) - адаптивный timeout
5. ✅ [`rag/remote_embedder.py`](../rag/remote_embedder.py:267) - прогресс эмбеддинга
6. ✅ [`rag/indexer_service.py`](../rag/indexer_service.py:357) - прогресс файлов
7. ✅ [`rag/indexer_service.py`](../rag/indexer_service.py:333) - batch_size 256 → 128

### Backward Compatibility:

- ✅ Все изменения обратно совместимы
- ✅ API не изменён
- ✅ Конфигурация может быть переопределена через `.env`
- ✅ Старые значения могут быть восстановлены откатом

---

## 🔄 Откат (Rollback)

Если необходимо вернуться к старым значениям:

### [`config.py`](../config.py:237)
```python
timeout_seconds: int = 600  # Было: 3600
```

### [`rag/retry_policy.py`](../rag/retry_policy.py:51)
```python
timeout_seconds: float = 600.0  # Было: 3600.0
```

### [`rag/remote_embedder.py`](../rag/remote_embedder.py:118)
```python
timeout_seconds=300.0,  # Было: 1800.0
```

### [`rag/indexer_service.py`](../rag/indexer_service.py:333)
```python
batch_size: int = 256  # Было: 128
```

---

## 🔗 Связанные Документы

- [`HOTFIX_TIMEOUTS.md`](./HOTFIX_TIMEOUTS.md) - Предыдущий hotfix (600s)
- [`!!!!ATTENTION(02_10_2025).md`](./!!!!ATTENTION(02_10_2025).md) - Проблема памяти
- [`Technical Debt.md`](./Technical%20Debt.md) - Технический долг

---

**Статус:** ✅ ПРИМЕНЁН, готов к тестированию  
**Автор:** Claude Code (Roo)  
**Review:** Pending user testing  
**Next Step:** Тестовая индексация для проверки отсутствия timeout