# 🔧 HOTFIX: Увеличение Timeouts для Swap Thrashing (02.10.2025)

**Дата применения:** 02 октября 2025, 19:10 MSK
**Причина:** Circuit Breaker открывается из-за timeout при 99% RAM (swap thrashing)
**Приоритет:** P0 - КРИТИЧЕСКИЙ HOTFIX
**Цель:** Проверка работоспособности индексации при экстремальных условиях

---

## 🎯 Проблема

**Root cause:** НЕ память, а **TIMEOUT из-за SWAP THRASHING**

При 99% RAM:
- Jina v3 модель (15-20GB) частично в swap
- Каждый embeddings запрос → 100+ page faults
- Disk I/O: latency 500ms → **120-180 секунд**
- Старые timeout 60s → AsyncIO timeout
- 5 неудач → Circuit Breaker OPEN

**Детали:** См. [`TIMEOUT_PROBLEM_DIAGNOSIS.md`](./TIMEOUT_PROBLEM_DIAGNOSIS.md)

---

## 🔧 Применённые Изменения

### 1. `config.py` - RemoteServiceConfig

**Было:**
```python
timeout_seconds: int = 60
max_retries: int = 3
retry_delay: float = 2.0
```

**Стало:**
```python
timeout_seconds: int = 600  # 10 минут (было 60s)
max_retries: int = 5        # больше попыток (было 3)
retry_delay: float = 10.0   # больше задержка (было 2.0s)
```

---

### 2. `rag/retry_policy.py` - RetryConfig

**Было:**
```python
max_attempts: int = 3
base_delay: float = 2.0
max_delay: float = 30.0
timeout_seconds: float = 60.0
```

**Стало:**
```python
max_attempts: int = 5           # больше попыток (было 3)
base_delay: float = 10.0        # больше задержка (было 2.0s)
max_delay: float = 120.0        # до 2 минут между попытками (было 30s)
timeout_seconds: float = 600.0  # 10 минут общий таймаут (было 60s)
```

---

### 3. `rag/remote_embedder.py` - CircuitBreakerConfig

**Было:**
```python
failure_threshold=5,
timeout_seconds=60.0,
```

**Стало:**
```python
failure_threshold=10,        # 10 неудач (было 5)
timeout_seconds=300.0,       # 5 минут (было 60s)
```

---

### 4. `rag/event_loop_manager.py` - aiohttp ClientTimeout

**Было:**
```python
timeout = aiohttp.ClientTimeout(
    total=60, connect=10, sock_read=30, sock_connect=5
)
```

**Стало:**
```python
timeout = aiohttp.ClientTimeout(
    total=600, connect=30, sock_read=300, sock_connect=15  # увеличены в 10x
)
```

---

### 5. `rag/remote_vector_store.py` - run_async_safe timeouts

**Было:**
```python
initialize_collection: timeout=60
index_documents: timeout=300
search: timeout=60
search_by_text: timeout=60
health_check: timeout=30
get_collection_info: timeout=30
close_sync: timeout=10
```

**Стало:**
```python
initialize_collection: timeout=300      # 5 минут (было 60s)
index_documents: timeout=1800           # 30 минут! (было 300s)
search: timeout=300                     # 5 минут (было 60s)
search_by_text: timeout=300             # 5 минут (было 60s)
health_check: timeout=60                # 1 минута (было 30s)
get_collection_info: timeout=60         # 1 минута (было 30s)
close_sync: timeout=30                  # 30 секунд (было 10s)
```

---

## 📊 Ожидаемые Результаты

### Pessimistic (при 99% RAM с активным swap):

**Один батч embeddings (batch=32):**
- Swap-in модели: 30-60 секунд
- Inference: 30-60 секунд
- Swap-out: 10-20 секунд
- **ИТОГО:** ~60-140 секунд per batch

**Индексация 135 файлов:**
- ~4-5 батчей (batch=32)
- 4 × 120s = **480 секунд = 8 минут**
- С учётом retries: **10-15 минут**

### Optimistic (если swap stabilizes):

- Модель остаётся в RAM после первой загрузки
- Последующие батчи: 10-30 секунд
- **ИТОГО:** 3-5 минут

---

## ⚠️ Риски и Ограничения

### Риски:

1. **Очень долгое ожидание** (до 30 минут на индексацию)
2. **Плохой User Experience** - нет прогресса, кажется что зависло
3. **OOM killer всё ещё возможен** при экстремальной нагрузке
4. **Не решает корневую проблему** - swap thrashing остаётся

### Ограничения:

- ❌ Это НЕ production решение
- ❌ Производительность будет низкой
- ❌ Масштабирование невозможно

---

## ✅ Критерии Успеха

**Минимальный успех:**
- ✅ Индексация 135 файлов завершается БЕЗ ошибок
- ✅ Circuit Breaker остаётся CLOSED
- ✅ Все векторы попадают в Qdrant

**Полный успех:**
- ✅ Время индексации < 15 минут
- ✅ Поиск работает корректно
- ✅ Нет OOM событий

---

## 🚀 ПРИМЕНЕНИЕ HOTFIX (Пошаговая Инструкция)

### ✅ Шаг 1: Перезапуск VM сервиса

**Автоматическая синхронизация через vm_start.py:**

```powershell
# Из корня проекта
cd D:\Scripts_Python\repo_sum

# Рестарт VM сервиса (автоматически синхронизирует .env)
python vm_start.py restart
```

**Что произойдёт:**
1. `vm_start.py` подключится к VM через SSH
2. Прочитает локальный `.env.example` (template) + `.env` (значения)
3. Создаст `.env` на VM с новыми timeout=600
4. Перезапустит `vm_rag_service.py` на VM
5. Новый процесс подхватит новые timeout из `.env`

**Проверка на VM:**
```bash
ssh user@10.61.11.54
cd ~/repo_sum_rag/repo_sum

# Проверить .env
grep RAG_TIMEOUT .env
# Должно быть: RAG_TIMEOUT_SECONDS=600

# Проверить логи
tail -20 rag_service.log
# Должно быть: "Circuit Breaker инициализирован: failure_threshold=10, timeout=300.0s"
```

---

### ✅ Шаг 2: Перезапуск локального приложения

```powershell
# Если web_ui запущен - остановить (Ctrl+C)

# Запустить заново
python run_web.py
```

**Важно:** Python процесс должен быть **ПОЛНОСТЬЮ перезапущен**, чтобы:
- Перечитать `.env` файл
- Переинициализировать `config.py`
- Создать новые объекты `RemoteVMEmbedder` с новыми timeout

---

### ✅ Шаг 3: Проверка применения

**Локальная диагностика:**
```powershell
python scripts/check_timeouts.py
```

**Ожидаемый вывод:**
```
Environment Variables:
  RAG_TIMEOUT_SECONDS: 600  ✅

config.py (Runtime):
  timeout_seconds: 600  ✅

RemoteVMEmbedder:
  timeout_seconds: 600  ✅
  retry_policy.config.timeout_seconds: 600.0  ✅
```

**VM диагностика:**
```bash
ssh user@10.61.11.54
cd ~/repo_sum_rag/repo_sum
bash scripts/check_timeouts_vm.sh | grep -A 10 "Environment Variables"
```

---

### ✅ Шаг 4: Тестирование

**Запуск индексации:**
```powershell
python run_web.py
# В веб-интерфейсе: Index → D:\Scripts_Python\repo_sum → Start
```

**Ожидаемое поведение:**
- ✅ Индексация 135 файлов завершится **БЕЗ timeout ошибок**
- ⏱️ Время: 10-15 минут (pessimistic) или 3-5 минут (optimistic)
- ⚠️ Будет **медленно** из-за swap thrashing при 99% RAM
- ✅ Circuit Breaker должен оставаться **CLOSED**

**Мониторинг логов PC:**
```powershell
# В отдельном терминале
Get-Content D:\Scripts_Python\repo_sum\*.log -Wait
```

**Что смотреть:**
- ❌ **НЕ должно быть:** `Circuit breaker OPEN`, `TimeoutError`, `VMTimeoutError`
- ✅ **Должно быть:** медленный прогресс (1 батч каждые 2-5 минут)

**Мониторинг RAM на VM:**
```bash
ssh user@10.61.11.54
watch -n 5 'free -h'
```

**Что смотреть:**
- RAM usage: стремиться к <95% (не 99%)
- Python процесс: 5-8 GB (не 20+ GB)

---

## 🔄 Откат (Rollback)

Если HOTFIX не помогает или вызывает другие проблемы:

### `config.py`
```python
timeout_seconds: int = 60
max_retries: int = 3
retry_delay: float = 2.0
```

### `rag/retry_policy.py`
```python
max_attempts: int = 3
base_delay: float = 2.0
max_delay: float = 30.0
timeout_seconds: float = 60.0
```

### `rag/remote_embedder.py`
```python
failure_threshold=5,
timeout_seconds=60.0,
```

### `rag/event_loop_manager.py`
```python
timeout = aiohttp.ClientTimeout(
    total=60, connect=10, sock_read=30, sock_connect=5
)
```

### `rag/remote_vector_store.py`
```python
initialize_collection: timeout=60
index_documents: timeout=300
search: timeout=60
search_by_text: timeout=60
health_check: timeout=30
get_collection_info: timeout=30
close_sync: timeout=10
```

---

## 📝 Следующие Шаги

### После проверки работоспособности:

1. **Если HOTFIX работает:**
   - ✅ Подтверждаем что проблема в timeout, не в памяти
   - ➡️ Переходим к **Aggressive Batch Reduction** (Strategy 1)
   - ➡️ Планируем **Model Quantization** (Strategy 2)

2. **Если HOTFIX НЕ работает:**
   - ❌ OOM killer всё равно убивает процессы
   - ➡️ Немедленный **reboot VM** для очистки памяти
   - ➡️ Применяем **Strategy 3 (Model Offloading)** до индексации

---

## 🔗 Связанные Документы

- [`TIMEOUT_PROBLEM_DIAGNOSIS.md`](./TIMEOUT_PROBLEM_DIAGNOSIS.md) - Детальная диагностика
- [`!!!!ATTENTION(02_10_2025).md`](./!!!!ATTENTION(02_10_2025).md) - Критическая проблема памяти
- [`OOM_PROTECTION.md`](../OOM_PROTECTION.md) - OOM Protection v2.0

---

**Статус:** ✅ ПРИМЕНЁН, готов к тестированию
**Автор:** Claude Code
**Review:** Pending user testing
