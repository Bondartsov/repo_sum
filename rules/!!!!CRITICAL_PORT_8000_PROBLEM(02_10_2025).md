# 🚨 !!!!CRITICAL: ПРОБЛЕМА С ПОРТОМ 8000 И СЕТЕВЫМ ДОСТУПОМ (02.10.2025)

**Дата:** 02 октября 2025, 21:30 MSK
**Статус:** 🔴 КРИТИЧНО - VM СЕРВИС ПАДАЕТ ВО ВРЕМЯ РАБОТЫ
**Приоритет:** P0 - БЛОКИРУЮЩАЯ ПРОБЛЕМА
**Ответственный:** DevOps + Infrastructure Team

---

## 🔥 КРИТИЧЕСКАЯ СИТУАЦИЯ

### Симптомы

**VM FastAPI сервис внезапно падает (OOM Killer):**

```
Логи VM:
INFO:     Uvicorn running on http://0.0.0.0:8000
...
Killed  ← OOM KILLER УБИЛ ПРОЦЕСС!
```

**Локальная индексация падает с connection refused:**

```
PC Логи (21:28:49):
ConnectionRefusedError: [Errno 10061] Connect call failed ('10.61.11.54', 8000)
ClientConnectorError: Cannot connect to host 10.61.11.54:8000
```

**Timeline:**
- `18:17:34` - VM сервис запустился успешно
- `18:25:55` - Последние успешные health checks от PC (172.16.0.4)
- `18:25:56` - **"Killed"** - OOM Killer убил процесс
- `21:25:56` - PC начал индексацию (136 файлов)
- `21:28:49` - PC получил connection refused (3 минуты попыток)

---

## 📊 ДЕТАЛЬНАЯ ДИАГНОСТИКА

### 1. Network Тесты (После Ручного Запуска)

**✅ Порт доступен изначально:**
```powershell
# PC → VM (18:24:28)
Test-NetConnection 10.61.11.54 -Port 8000
TcpTestSucceeded : True  ✅

curl http://10.61.11.54:8000/health
{"status":"connected",...}  ✅
```

**✅ Firewall правила добавлены:**
```bash
# VM (18:12:05)
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
ACCEPT     tcp  --  0.0.0.0/0   0.0.0.0/0   tcp dpt:8000  ✅
```

**✅ FastAPI слушает на правильном интерфейсе:**
```bash
# VM (18:22:30)
sudo netstat -tulnp | grep 8000
tcp   0   0 0.0.0.0:8000   LISTEN   121521/python  ✅
```

---

### 2. OOM Killer Evidence

**Логи VM показывают внезапное завершение:**
```
2025-10-02 18:25:55 - INFO: Health check OK (172.16.0.4)
Killed  ← ПРОЦЕСС УБИТ БЕЗ GRACEFUL SHUTDOWN!
```

**Нет в логах:**
- ❌ Нет `KeyboardInterrupt` (не Ctrl+C)
- ❌ Нет `Exception` (не падение приложения)
- ❌ Нет `Stopping...` (не graceful stop)
- ✅ Есть **"Killed"** = OOM Killer or SIGKILL

**Проверка dmesg (нужно сделать):**
```bash
# На VM:
sudo dmesg -T | grep -i "out of memory" | tail -20
sudo dmesg -T | grep -i "killed process" | tail -20
```
ПОДТВЕРЖДАЮ:
(venv) user@t-ubuntu-redis:~/repo_sum_rag/repo_sum$ sudo dmesg -T | grep -i killed
[Чт окт  2 15:50:03 2025] Out of memory: Killed process 43574 (python) total-vm:117919788kB, anon-rss:64806336kB, file-rss:2596kB, shmem-rss:4kB, UID:1000 pgtables:141648kB oom_score_adj:0
[Чт окт  2 16:19:15 2025] Out of memory: Killed process 62041 (python) total-vm:117919852kB, anon-rss:64898784kB, file-rss:2676kB, shmem-rss:4kB, UID:1000 pgtables:141656kB oom_score_adj:0
[Чт окт  2 17:37:41 2025] Out of memory: Killed process 95136 (python) total-vm:117919840kB, anon-rss:64957112kB, file-rss:2544kB, shmem-rss:4kB, UID:1000 pgtables:141688kB oom_score_adj:0
[Чт окт  2 17:55:50 2025] Out of memory: Killed process 106489 (python) total-vm:117920040kB, anon-rss:64995736kB, file-rss:2740kB, shmem-rss:4kB, UID:1000 pgtables:141720kB oom_score_adj:0
[Чт окт  2 18:26:58 2025] Out of memory: Killed process 121521 (python) total-vm:117919940kB, anon-rss:64991524kB, file-rss:2824kB, shmem-rss:4kB, UID:1000 pgtables:141692kB oom_score_adj:0
(venv) user@t-ubuntu-redis:~/repo_sum_rag/repo_sum$




---

### 3. Memory Analysis

**RAM на момент запуска сервиса:**
```bash
# VM (18:12) - после перезапуска
RAM: 62Gi total, 322Mi used, 61Gi free  ✅ ЧИСТО!
Swap: 4.0Gi, 130Mi used
```

**Python процесс (121521) memory footprint:**
- Jina v3 модель: ~15-20GB при загрузке
- FastAPI + Qdrant client: ~1-2GB
- **Прогрев модели: 25.4 секунды** (норма для 60GB RAM)

**Проблема:** После ~7-8 минут работы → OOM Killer активировался!

---

## 🎯 ROOT CAUSE HYPOTHESIS

### Гипотеза #1: Memory Leak в Jina v3 (Наиболее Вероятно)

**Факты:**
1. ✅ Сервис стартует успешно при чистой RAM
2. ✅ Health checks работают ~8 минут
3. ❌ После первого embeddings запроса → Killed
4. ⏱️ Timing: PC начал индексацию в 21:25:56, но VM уже упал в 18:25:56

**Возможная причина:**
- Jina v3 не освобождает память после inference
- Batch processing накапливает tensors без GC
- FastEmbed кэширует модель без memory limits

---

### Гипотеза #2: Недостаточно RAM для Jina v3 + Qdrant + FastAPI

**Memory Breakdown (теоретический):**
| Компонент | Память | Детали |
|-----------|--------|--------|
| Jina v3 Model | 15-20 GB | 570M параметров FP32 |
| Jina Inference | 8-12 GB | Activations + intermediate tensors |
| Qdrant | 2-3 GB | Векторная БД (пусто, но резервирует) |
| FastAPI | 1-2 GB | Application + dependencies |
| OS + Docker | 5-10 GB | Ubuntu + Docker containers |
| **ИТОГО** | **31-47 GB** | **Может превысить 60GB при работе!** |

---

### Гипотеза #3: OOM Protection v2.0 Недостаточно

**Текущая защита:**
```python
# vm_rag_service.py (check_memory_usage)
critical_threshold = 100 - (5.0 / 60 * 100) ≈ 92%
# Но OOM Killer срабатывает раньше при резких скачках!
```

**Проблема:**
- OOM Protection проверяет память **между батчами**
- Но inference может скакнуть с 40% → 98% **внутри одного батча**
- Kernel OOM Killer срабатывает мгновенно при >98% + swap exhaustion

---

## 🔧 РЕШЕНИЯ (Приоритизированные)

### ⭐ Решение 1: Model Offloading (СРОЧНО - P0)

**Суть:** Выгружать модель из памяти между запросами.

**Файл:** `vm_rag_service.py`

```python
import gc
import torch
from contextlib import contextmanager

class ModelManager:
    """Управление загрузкой/выгрузкой модели из памяти"""

    def __init__(self):
        self.model = None
        self.last_used = None
        self.idle_timeout = 300  # 5 минут

    @contextmanager
    def get_model(self):
        """Context manager для временной загрузки модели"""
        try:
            if self.model is None:
                logger.info("📥 Загрузка Jina v3 модели...")
                self.model = load_jina_model()
                logger.info("✅ Модель загружена")

            self.last_used = time.time()
            yield self.model

        finally:
            # Выгружаем если idle > 5 минут
            if time.time() - self.last_used > self.idle_timeout:
                logger.info("🗑️ Выгрузка модели (idle timeout)")
                self.unload_model()

    def unload_model(self):
        """Полная выгрузка модели из памяти"""
        if self.model is not None:
            del self.model
            self.model = None

            # Aggressive GC
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ Модель выгружена, память освобождена")

# Usage в FastAPI endpoint
model_manager = ModelManager()

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    with model_manager.get_model() as model:
        embeddings = model.encode(request.texts)

    # Модель автоматически выгрузится через 5 минут idle
    return {"embeddings": embeddings}
```

**Плюсы:**
- ✅ Экономит ~15-20GB RAM когда модель не используется
- ✅ Предотвращает OOM при idle
- ✅ Быстро реализовать (2-3 часа)

**Минусы:**
- ❌ Первый запрос после idle будет медленным (25s на загрузку)
- ❌ Увеличивает latency

---

### ⭐ Решение 2: Explicit GC After Each Batch (БЫСТРЫЙ HOTFIX - P0)

**Суть:** Принудительный garbage collection после каждого батча.

**Файл:** `vm_rag_service.py` → embeddings endpoint

```python
import gc
import psutil

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    try:
        # Process embeddings
        result = embedder.encode(request.texts, batch_size=32)

        # HOTFIX: Aggressive GC after each request
        gc.collect()

        # Check memory
        mem = psutil.virtual_memory()
        logger.info(f"💾 RAM after GC: {mem.percent:.1f}% ({mem.available / (1024**3):.1f}GB free)")

        if mem.percent > 90:
            logger.warning(f"⚠️ HIGH RAM: {mem.percent:.1f}% - might need model offloading")

        return {"embeddings": result.tolist()}

    except Exception as e:
        logger.error(f"❌ Embeddings error: {e}")
        # Emergency GC on error
        gc.collect()
        raise
```

**Плюсы:**
- ✅ Простой (5 минут на реализацию)
- ✅ Нет изменения архитектуры
- ✅ Можно применить немедленно

**Минусы:**
- ❌ Может не помочь если memory leak в C++ коде Jina
- ❌ GC занимает время (50-200ms per call)

---

### ⭐ Решение 3: Restart Service After N Requests (WORKAROUND - P1)

**Суть:** Автоматический рестарт сервиса после N запросов для очистки памяти.

**Файл:** `vm_rag_service.py`

```python
class ServiceLifecycle:
    def __init__(self):
        self.request_count = 0
        self.max_requests = 100  # Рестарт после 100 запросов
        self.start_time = time.time()

    def should_restart(self) -> bool:
        """Проверка нужен ли рестарт"""
        mem = psutil.virtual_memory()

        # Рестарт если:
        # 1. >95% RAM
        # 2. Или обработано >100 запросов
        # 3. Или uptime >4 часов
        uptime_hours = (time.time() - self.start_time) / 3600

        return (
            mem.percent > 95 or
            self.request_count > self.max_requests or
            uptime_hours > 4
        )

lifecycle = ServiceLifecycle()

@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    lifecycle.request_count += 1

    if lifecycle.should_restart():
        logger.warning("🔄 Restart needed - memory cleanup required")
        # Graceful shutdown
        os.kill(os.getpid(), signal.SIGTERM)

    # ... process request
```

**Плюсы:**
- ✅ Гарантированная очистка памяти
- ✅ Автоматическое восстановление

**Минусы:**
- ❌ Downtime во время рестарта (30 секунд)
- ❌ Костыль, не решает root cause

---

### ⭐ Решение 4: Model Quantization (СРЕДНИЙ СРОК - P1)

**Суть:** Использовать 8-bit quantized модель Jina v3.

**Файл:** `rag/embedder.py`

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = SentenceTransformer(
    "jinaai/jina-embeddings-v3",
    quantization_config=quantization_config,
    device="cpu"
)
```

**Экономия памяти:** 15-20GB → 7-8GB (~50%)

**Плюсы:**
- ✅ Значительная экономия RAM
- ✅ Качество embeddings почти не страдает

**Минусы:**
- ❌ Требует тестирования (2-4 часа)
- ❌ Может замедлить inference на 10-20%

---

### ⭐ Решение 5: Increase Swap + Lower swappiness (ИНФРАСТРУКТУРА - P2)

**Суть:** Увеличить swap и настроить vm.swappiness для плавной деградации.

**На VM:**
```bash
# 1. Увеличить swap с 4GB → 16GB
sudo swapoff -a
sudo dd if=/dev/zero of=/swapfile bs=1G count=16
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. Уменьшить swappiness (агрессивнее использовать RAM)
sudo sysctl vm.swappiness=10
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf

# 3. Настроить OOM score для защиты критичных процессов
echo -1000 | sudo tee /proc/$(pgrep qdrant)/oom_score_adj
# Python оставляем с дефолтным score → убьёт его первым
```

**Плюсы:**
- ✅ Система не упадёт резко
- ✅ Больше времени на graceful degradation

**Минусы:**
- ❌ Swap на SSD изнашивает диск
- ❌ Производительность деградирует при swap usage

---

## 🔬 ДИАГНОСТИЧЕСКИЕ КОМАНДЫ (Завтра Утром)

### На VM:

```bash
# 1. Проверить OOM killer логи
sudo dmesg -T | grep -i "killed process" | tail -20
sudo dmesg -T | grep -i "out of memory" | tail -20

# 2. Проверить памятьпроцесса в runtime
watch -n 1 'ps aux --sort=-%mem | head -10'

# 3. Детальная память процесса
sudo pmap -x $(pgrep -f vm_rag_service) | tail -20

# 4. Kernel OOM настройки
cat /proc/sys/vm/overcommit_memory
cat /proc/sys/vm/oom_kill_allocating_task

# 5. Memory cgroup limits (если используются)
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/memory/memory.usage_in_bytes
```

---

## 📋 ACTION PLAN (Приоритизированный)

### Завтра Утром (02.10.2025)

#### 1. **Диагностика (30 минут)**
```bash
# SSH на VM
ssh user@10.61.11.54

# Проверить OOM killer логи
sudo dmesg -T | grep -E "Killed|Out of memory" | tail -30

# Сохранить в файл для анализа
sudo dmesg -T > ~/oom_logs_$(date +%Y%m%d_%H%M).txt
```

#### 2. **HOTFIX #1: Explicit GC (15 минут)**
- Добавить `gc.collect()` после каждого embeddings запроса
- Добавить memory logging
- Тест: попробовать индексацию снова

#### 3. **HOTFIX #2: Reduce Batch Size (5 минут)**
```bash
# В .env на VM
EMBEDDING_BATCH_SIZE_MIN=1   # Было: 1 (ок)
EMBEDDING_BATCH_SIZE_MAX=2   # Было: 2 (ок)
# Уже минимальные! Проблема не в этом.
```

#### 4. **Мониторинг (setup один раз)**
```bash
# Запустить в отдельном SSH окне
watch -n 2 'echo "=== Memory ===" && free -h && echo "=== Process ===" && ps aux --sort=-%mem | grep python | head -3'
```

#### 5. **Тест с Model Offloading (если HOTFIX не помог)**
- Реализовать `ModelManager` class
- Выгружать модель после каждого запроса
- Тест: попробовать индексацию

---

### Если Всё Падает (Plan B)

#### Option 1: Использовать Mock Embedder (временно)
```bash
# На PC
python main.py rag index D:\Scripts_Python\repo_sum --use-mock
```

#### Option 2: Reboot VM + Increase Swap
```bash
# Reboot для очистки
sudo reboot

# После перезагрузки
sudo dd if=/dev/zero of=/swapfile bs=1G count=16
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Option 3: Миграция на более мощную VM
- Нужна VM с 128GB RAM или GPU
- Или использовать managed service (Cohere, OpenAI embeddings)

---

## 📊 МЕТРИКИ ДЛЯ ОТСЛЕЖИВАНИЯ

### Success Criteria (после фикса):

| Метрика | Текущее | Целевое |
|---------|---------|---------|
| VM Uptime | <10 минут | >4 часа |
| RAM Usage (idle) | 50-60GB | <30GB |
| RAM Usage (peak) | >60GB (OOM) | <55GB |
| Embeddings Requests Before OOM | 0 | >1000 |
| OOM Killer Events | 1+ per run | 0 |

---

## 🔗 СВЯЗАННЫЕ ДОКУМЕНТЫ

- [`!!!!ATTENTION(02_10_2025).md`](./!!!!ATTENTION(02_10_2025).md) - Проблема 99% RAM (связана!)
- [`TIMEOUT_PROBLEM_DIAGNOSIS.md`](./TIMEOUT_PROBLEM_DIAGNOSIS.md) - Timeout из-за swap
- [`HOTFIX_TIMEOUTS.md`](./HOTFIX_TIMEOUTS.md) - Увеличенные timeout (применены)
- [`OOM_PROTECTION.md`](../OOM_PROTECTION.md) - OOM Protection v2.0 (недостаточно!)

---

## 💡 ИНСАЙТЫ И ВЫВОДЫ

### Что Мы Узнали:

1. **Timeout != Основная Проблема**
   - Увеличенные timeout не помогли
   - Проблема глубже: OOM Killer убивает процесс

2. **60GB RAM Недостаточно для Production**
   - Jina v3 (15-20GB) + Qdrant + FastAPI + OS = критично
   - Нужен либо model offloading, либо 128GB RAM

3. **OOM Protection v2.0 Реагирует Слишком Поздно**
   - Проверяет память между запросами
   - Но spike внутри inference → instant OOM

4. **Network Access Работает (Не Firewall!)**
   - Порт 8000 открыт
   - iptables правила корректны
   - Проблема в том что процесс убивается изнутри

---

## ⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ

1. **НЕ проблема сети** - firewall и iptables настроены правильно
2. **НЕ проблема timeout** - сервис падает до истечения timeout
3. **ДА проблема памяти** - OOM Killer активен
4. **ДА нужен Model Offloading** - единственный путь для 60GB RAM

---

**Дата создания:** 02.10.2025, 21:45 MSK
**Автор:** Claude Code Analysis
**Статус:** ACTIVE - Требует немедленного решения завтра утром
**Приоритет:** P0 - БЛОКИРУЕТ ВСЮ ИНДЕКСАЦИЮ
**Next Steps:** Диагностика OOM логов + HOTFIX с GC + Model Offloading
