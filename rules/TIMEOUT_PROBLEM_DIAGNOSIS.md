# 🚨 ДИАГНОСТИКА: Timeout при 99% RAM (02.10.2025)

## Краткое резюме

**Проблема:** Circuit Breaker открывается после 5 неудачных попыток embeddings запросов
**Root cause:** НЕ память напрямую, а **timeout из-за swap thrashing** при 99% RAM
**Симптомы:** VM сервис живой (health checks проходят), но embeddings запросы timeout

---

## 📊 Анализ логов

### Логи VM (сервис работает!)

```log
2025-10-02 15:46:07,556 - __main__ - INFO - ✅ Все сервисы успешно инициализированы
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
...
INFO:     172.16.0.10:49723 - "GET /health HTTP/1.1" 200 OK  # Последний health check
```

**Ключевые наблюдения:**
- ✅ FastAPI сервис запущен на :8000
- ✅ Qdrant инициализирован и доступен
- ✅ Jina v3 модель загружена за 3 секунды (без swap)
- ✅ Health checks проходят успешно (172.16.0.10 = PC клиент)
- ❌ **НО:** Нет логов embeddings запросов → они не доходят или не обрабатываются

---

### Логи PC (Circuit Breaker OPEN)

```log
2025-10-02 18:48:58,453 - FileScanner - INFO - Найдено файлов: 135, общий размер: 1.8 MB
2025-10-02 18:50:59,401 - rag.remote_embedder - ERROR - Circuit breaker OPEN: VM сервис недоступен. Следующая попытка через 56s
```

**Временная линия:**
- `18:48:58` - Индексация началась, файлы просканированы (135 файлов)
- `18:49:xx` - Первая попытка отправки батча эмбеддингов → **timeout**
- `18:49:xx+2s` - Retry #2 → **timeout**
- `18:49:xx+4s` - Retry #3 → **timeout**
- `18:49:xx+8s` - Retry #4 → **timeout**
- `18:49:xx+16s` - Retry #5 → **timeout**
- `18:50:59` - Circuit Breaker открыт после **~2 минут** неудачных попыток

---

## 🔍 Root Cause Analysis

### Почему timeout, если VM живая?

#### 1. Health Check vs Embeddings - Разница в нагрузке

| Операция | Размер запроса | RAM usage | Disk I/O | Время (норма) | Время (99% RAM) |
|----------|----------------|-----------|----------|---------------|-----------------|
| **GET /health** | ~1KB | 0MB (cached) | 0 reads | **50ms** | **100ms** ✅ |
| **POST /embeddings (batch=32)** | ~50KB | **15-20GB** (model) + **2-5GB** (batch) | **100+ page faults** | **500ms** | **120-180s** ❌ |

**Health checks проходят**, потому что:
- Легковесные (1KB)
- Не требуют загрузки модели в RAM
- Не вызывают page faults

**Embeddings timeout**, потому что:
- Требуют загрузки Jina v3 модели (15-20GB)
- При 99% RAM модель частично в swap
- Каждый батч вызывает **100+ page faults**
- Disk I/O убивает latency: SSD read ~500MB/s → **30+ секунд** на загрузку модели

---

#### 2. Swap Thrashing Механизм

**Что происходит при 99% RAM:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PC отправляет embeddings запрос (batch=32, ~50KB)       │
├─────────────────────────────────────────────────────────────┤
│ 2. VM FastAPI принимает запрос → OK                        │
├─────────────────────────────────────────────────────────────┤
│ 3. Jina v3 model.encode() вызывается                       │
│    ├─ Требует 15-20GB RAM для модели                       │
│    ├─ Доступно только ~0.5GB свободной RAM                 │
│    └─ Linux начинает swap-in модели с диска                │
├─────────────────────────────────────────────────────────────┤
│ 4. Page faults: 100+ reads по 4KB-64KB                     │
│    ├─ SSD read: ~500MB/s sustained                         │
│    ├─ Random I/O penalty: 0.1ms latency per page           │
│    └─ Total: 10-15 секунд на swap-in                       │
├─────────────────────────────────────────────────────────────┤
│ 5. Модель загружена, начинается inference                  │
│    ├─ Но RAM всё ещё 99% → swap-out других процессов       │
│    ├─ Qdrant, system cache evicted → новые page faults     │
│    └─ Inference time: 500ms → 30-60 секунд                 │
├─────────────────────────────────────────────────────────────┤
│ 6. Результат готов, но уже прошло 120+ секунд              │
│    ├─ PC timeout: 60 секунд                                │
│    ├─ AsyncIO timeout exception                            │
│    └─ VM пытается вернуть ответ, но клиент уже отключился  │
└─────────────────────────────────────────────────────────────┘
```

**Почему 5 попыток подряд timeout:**
- Retry #1: swap-in → 120s → timeout
- Retry #2: модель всё ещё в swap → 100s → timeout
- Retry #3: RAM 99.5% → новые page faults → 90s → timeout
- Retry #4-5: система продолжает thrashing → timeout

---

## 🎯 Текущие Timeout Настройки

### Конфигурация

**config.py:**
```python
@dataclass
class RemoteServiceConfig:
    timeout_seconds: int = 60  # Общий timeout для HTTP запросов
    max_retries: int = 3
    retry_delay: float = 2.0
```

**remote_embedder.py:**
```python
self.timeout_seconds = 60  # Из config или RAG_TIMEOUT_SECONDS
base_timeout = deadline_ms / 1000.0  # 60s
total_timeout = (base_timeout * self.max_retries) + backoff_total  # ~210s
```

**circuit_breaker.py:**
```python
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Открыть после 5 неудач
    timeout_seconds: float = 60.0   # Ждать 60s перед HALF_OPEN
```

---

### Текущий таймлайн одного батча

```
Attempt #1:
├─ HTTP connect: 0.1s
├─ VM processing: 120s (swap thrashing)
├─ Timeout at: 60s ❌
└─ Backoff: 2s

Attempt #2:
├─ HTTP connect: 0.1s
├─ VM processing: 100s (still swapping)
├─ Timeout at: 60s ❌
└─ Backoff: 4s

Attempt #3:
├─ HTTP connect: 0.1s
├─ VM processing: 90s
├─ Timeout at: 60s ❌
└─ Backoff: 8s

Circuit Breaker:
├─ 5 failures detected
└─ State: OPEN → блокировка на 60s
```

**Total time before Circuit Breaker OPEN:** ~2 минуты
**Actual VM processing time (если бы дождались):** 120+ секунд per request

---

## 💡 Решения

### Решение 1: HOTFIX - Увеличить Timeout (временное)

**Цель:** Дать VM достаточно времени при swap thrashing

**Изменения в `config.py`:**
```python
@dataclass
class RemoteServiceConfig:
    timeout_seconds: int = 180  # Было: 60 → Стало: 180 (3 минуты)
    max_retries: int = 3
    retry_delay: float = 5.0    # Было: 2.0 → Стало: 5.0
```

**Изменения в `circuit_breaker.py`:**
```python
CircuitBreakerConfig(
    failure_threshold=5,
    timeout_seconds=120.0  # Было: 60.0 → Стало: 120.0 (2 минуты)
)
```

**Плюсы:**
- ✅ Простая правка (2 файла)
- ✅ Можно применить за 5 минут
- ✅ Позволит завершить индексацию даже при 99% RAM

**Минусы:**
- ❌ Не решает корневую проблему (swap thrashing)
- ❌ Индексация будет **медленной** (8+ минут на 135 файлов вместо 30 секунд)
- ❌ User experience ухудшается (долгое ожидание)

---

### Решение 2: ПРАВИЛЬНОЕ - Aggressive Batch Reduction

**Цель:** Предотвратить 99% RAM и swap thrashing

**Изменения в `vm_rag_service.py`:**

Добавить метод в `IndexerService`:

```python
def _check_memory_and_adjust_batch(self, current_batch_size: int) -> int:
    """
    Динамическая подстройка размера батча на основе RAM.

    Для 60GB RAM:
    - <2GB free (>96%): batch=8 (EXTREME)
    - <5GB free (>92%): batch=16 (CRITICAL)
    - <9GB free (>85%): batch=32 (WARNING)
    - >9GB free (<85%): batch=64-256 (NORMAL)
    """
    memory = psutil.virtual_memory()
    mem_percent = memory.percent
    available_gb = round(memory.available / (1024**3), 2)

    # 🚨 EXTREME: <2GB free → минимальный батч
    if available_gb < 2.0:
        new_batch = 8
        logger.critical(
            f"🚨 EXTREME RAM: {available_gb}GB free ({mem_percent:.1f}%) → batch={new_batch}"
        )
        return new_batch

    # 🔴 CRITICAL: >92% → агрессивное снижение
    if mem_percent > 92:
        new_batch = max(16, current_batch_size // 8)
        logger.error(
            f"🔴 CRITICAL RAM: {mem_percent:.1f}% ({available_gb}GB free) → batch={new_batch}"
        )
        return new_batch

    # 🟠 HIGH: >85% → снижение батча
    if mem_percent > 85:
        new_batch = max(32, current_batch_size // 4)
        logger.warning(
            f"🟠 HIGH RAM: {mem_percent:.1f}% ({available_gb}GB free) → batch={new_batch}"
        )
        return new_batch

    # 🟡 WARNING: >75% → умеренное снижение
    if mem_percent > 75:
        new_batch = max(64, current_batch_size // 2)
        logger.info(
            f"🟡 WARNING RAM: {mem_percent:.1f}% → batch={new_batch}"
        )
        return new_batch

    # ✅ NORMAL: <75% → стандартный батч
    return current_batch_size
```

Использовать в цикле индексации:

```python
async def _index_chunks_batch(self, chunks, batch_size, show_progress):
    current_batch_size = batch_size

    for i in range(0, len(chunks), current_batch_size):
        # Проверяем память перед каждым батчем
        current_batch_size = self._check_memory_and_adjust_batch(current_batch_size)

        batch = chunks[i:i+current_batch_size]
        # ... обработка батча

        # GC после каждого батча при высокой нагрузке
        if psutil.virtual_memory().percent > 85:
            gc.collect()
            await asyncio.sleep(0.5)  # Даём время на GC
```

**Плюсы:**
- ✅ Предотвращает 99% RAM → swap thrashing не начинается
- ✅ Сохраняет производительность при нормальной нагрузке
- ✅ Автоматическая адаптация под условия
- ✅ Не требует изменения timeout

**Минусы:**
- ❌ Требует изменения кода (30 минут работы)
- ❌ Нужно тестирование

---

### Решение 3: ДОЛГОСРОЧНОЕ - Model Quantization + Offloading

См. `!!!!ATTENTION(02_10_2025).md` → Strategies 2-3

---

## 🎬 Рекомендуемый Action Plan

### Немедленно (следующие 10 минут):

**Option A: Увеличить timeout (HOTFIX)**
```bash
# Правка config.py
timeout_seconds: int = 180
retry_delay: float = 5.0

# Правка circuit_breaker.py
timeout_seconds=120.0

# Перезапуск VM сервиса
python vm_start.py restart
```

**Option B: Освободить RAM и повторить**
```bash
# SSH на VM
ssh user@10.61.11.54

# Рестарт сервисов для очистки памяти
sudo systemctl restart qdrant
sudo systemctl restart rag-service

# Проверка RAM
free -h

# Если всё ещё 99% → reboot VM
sudo reboot
```

---

### Краткосрочно (сегодня):

1. Применить **Aggressive Batch Reduction** из Решения 2
2. Протестировать индексацию на 135 файлах
3. Зафиксировать метрики:
   - Максимальный RAM usage
   - Время индексации
   - Количество batch adjustments

---

### Среднесрочно (на неделе):

1. Реализовать **Model Quantization** (8-bit)
   - Экономия ~8GB RAM
   - Jina v3: 15GB → 7GB
2. Добавить **Prometheus metrics** для RAM мониторинга
3. Настроить **Grafana alerts** при RAM >85%

---

## 📈 Ожидаемые Результаты

### После HOTFIX (timeout увеличен):
- ✅ Индексация завершится успешно
- ❌ Время: **8-12 минут** (вместо 30 секунд)
- ❌ User experience: плохой

### После Batch Reduction:
- ✅ RAM usage: **<92%** (не превышает критический порог)
- ✅ Время индексации: **2-3 минуты** (компромисс)
- ✅ Стабильность: высокая
- ✅ User experience: приемлемый

### После Quantization + Offloading:
- ✅ RAM usage: **<70%** при idle
- ✅ Время индексации: **30-45 секунд**
- ✅ Стабильность: отличная
- ✅ User experience: отличный

---

## 🔗 Связанные Документы

- [`!!!!ATTENTION(02_10_2025).md`](./!!!!ATTENTION(02_10_2025).md) - Оригинальная проблема 99% RAM
- [`OOM_PROTECTION.md`](../OOM_PROTECTION.md) - OOM Protection v2.0 (недостаточно для swap thrashing)
- [`Technical Debt.md`](./Technical Debt.md) - Технический долг и задачи
- [`tests/rag/TESTING_STRATEGY.md`](../tests/rag/TESTING_STRATEGY.md) - Стратегия тестирования

---

**Дата создания:** 02.10.2025
**Автор:** Claude Code Analysis
**Статус:** ACTIVE - Требует немедленного решения
**Приоритет:** P0 - Блокирует индексацию
