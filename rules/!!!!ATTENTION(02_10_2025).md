# 🚨 !!!!ATTENTION: КРИТИЧЕСКАЯ ПРОБЛЕМА ПАМЯТИ VM (02.10.2025)

**Дата:** 02 октября 2025, 18:45 MSK
**Статус:** 🔴 КРИТИЧНО - ТРЕБУЕТ НЕМЕДЛЕННОГО ВНИМАНИЯ
**Приоритет:** P0 - БЛОКИРУЮЩАЯ ПРОБЛЕМА
**Ответственный:** DevOps + Performance Team

---

## 🔥 КРИТИЧЕСКАЯ СИТУАЦИЯ

### Симптомы

**VM достигла критического уровня использования памяти:**

```
Memory Usage: 62.68 GB / 62.79 GB (99.8%)
Available: ~100 MB
Status: 🔴 EXTREME DANGER - OOM Killer активен
```

**Последствия:**
- ❌ Индексация repo_sum (135 файлов) НЕВОЗМОЖНА
- ❌ Риск падения VM сервиса от OOM killer
- ❌ Circuit breaker постоянно OPEN
- ❌ TimeoutError на всех операциях
- ⚠️ **СИСТЕМА НА ГРАНИ КРАХА**

---

## 📊 ДИАГНОСТИКА

### Memory Breakdown (приблизительный)

| Компонент | Память | Процент | Детали |
|-----------|--------|---------|--------|
| **Jina v3 модель** | 15-20 GB | ~30% | 570M параметров FP32 + inference буферы |
| **Qdrant векторы** | 8-12 GB | ~18% | Векторная БД с индексами |
| **Batch processing** | 10-15 GB | ~20% | Временные данные во время индексации |
| **OS + процессы** | 2-3 GB | ~5% | Ubuntu + Python + FastAPI |
| **Прочее** | 15-20 GB | ~27% | Кэши, буферы, фрагментация |
| **ИТОГО** | **~60-62 GB** | **~99%** | **КРИТИЧНО!** |

### Почему так много памяти?

#### 1. **Jina v3 - тяжеловес (15-20 GB)**

```python
Model: jinaai/jina-embeddings-v3
Parameters: 570M
Precision: FP32 (4 bytes per parameter)
Model size: 570M * 4 = 2.3 GB (только веса)

Runtime overhead:
- Activations: ~3-5 GB
- Attention buffers: ~2-4 GB
- Pooling layers: ~1-2 GB
- Inference cache: ~2-3 GB
- PyTorch overhead: ~2-3 GB
----------------------------------------
TOTAL: ~15-20 GB 🚨
```

#### 2. **Batch Processing - память растёт с batch (10-15 GB)**

```python
# Текущая логика
batch_size = 128-256 (стартовый)

При обработке батча:
- Input tensors: batch_size * 1024 * 4 bytes = ~500KB-1MB
- Intermediate activations: ~2-4 GB на батч
- Output embeddings: batch_size * 1024 * 4 = ~500KB-1MB
- Temporary arrays: ~2-3 GB
- GC overhead: ~2-4 GB (фрагментация)
----------------------------------------
TOTAL per batch: ~10-15 GB 🚨
```

#### 3. **Qdrant - растёт с данными (8-12 GB)**

```python
Vectors stored: ~10,000-50,000
Dimension: 1024
Storage: vectors + HNSW index + metadata

Calculation:
- Vectors: 50k * 1024 * 4 bytes = ~200 MB
- HNSW index: ~5x overhead = ~1 GB
- Scalar quantization: ~2x overhead = ~2 GB
- Metadata: ~100-200 MB
- Query cache: ~1-2 GB
- OS page cache: ~3-5 GB
----------------------------------------
TOTAL: ~8-12 GB
```

---

## 🎯 СТРАТЕГИИ РЕШЕНИЯ

### ⭐ Стратегия 1: AGGRESSIVE BATCH REDUCTION (HOTFIX - 10 минут)

**Приоритет:** 🔴 P0 - НЕМЕДЛЕННО

**Проблема текущей логики:**

```python
# rag/indexer_service.py - ТЕКУЩИЙ КОД
def _check_memory_and_adjust_batch(self, current_batch_size: int) -> int:
    if mem_percent > 85:
        return max(32, current_batch_size // 4)  # ⚠️ Минимум 32 - СЛИШКОМ МНОГО
```

**При 99% RAM и batch=32:**
- Одновременно в памяти: 32 текста * ~100KB = ~3MB (input)
- Inference память: ~8-10GB (Jina v3 activations)
- **ИТОГО: ~10GB spike при каждом батче** → OOM killer

---

**РЕШЕНИЕ v3.0: Экстремально малые батчи**

**Файл:** `rag/indexer_service.py`

**Метод:** `_check_memory_and_adjust_batch()`

```python
def _check_memory_and_adjust_batch(self, current_batch_size: int) -> int:
    """
    Динамическая подстройка batch_size с защитой от OOM.

    v3.0: Добавлена экстремальная защита для 60GB RAM
    - При <2GB свободно: batch = 8 (экстремальный режим)
    - При >92%: batch = 16 (очень малый)
    - При >85%: batch = 32 (малый)
    """
    try:
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        available_gb = round(memory.available / (1024**3), 2)

        # 🚨 ЭКСТРЕМАЛЬНАЯ ЗАЩИТА: <2GB свободно
        if available_gb < 2.0:
            if current_batch_size != 8:
                logger.critical(
                    f"🚨🚨🚨 EXTREME: {available_gb:.2f}GB свободно! "
                    f"Переход на минимальный batch: {current_batch_size} → 8"
                )
            return 8

        # 🔴 КРИТИЧЕСКИЙ УРОВЕНЬ: >92% использовано
        if mem_percent > 92:
            new_batch_size = max(16, current_batch_size // 8)
            if new_batch_size != current_batch_size:
                logger.error(
                    f"🔴 Критический уровень памяти: {mem_percent:.1f}% "
                    f"(доступно: {available_gb:.1f}GB). "
                    f"Batch: {current_batch_size} → {new_batch_size}"
                )
            return new_batch_size

        # 🟠 ВЫСОКИЙ УРОВЕНЬ: >85% использовано
        if mem_percent > 85:
            new_batch_size = max(32, current_batch_size // 4)
            if new_batch_size != current_batch_size:
                logger.warning(
                    f"🟠 Высокий уровень памяти: {mem_percent:.1f}% "
                    f"(доступно: {available_gb:.1f}GB). "
                    f"Batch: {current_batch_size} → {new_batch_size}"
                )
            return new_batch_size

        # 🟡 УМЕРЕННЫЙ УРОВЕНЬ: >75% использовано
        elif mem_percent > 75:
            new_batch_size = max(64, current_batch_size // 2)
            if new_batch_size != current_batch_size:
                logger.info(
                    f"🟡 Умеренный уровень памяти: {mem_percent:.1f}%. "
                    f"Batch: {current_batch_size} → {new_batch_size}"
                )
            return new_batch_size

        # 🟢 НИЗКИЙ УРОВЕНЬ: <50% - можем увеличить
        elif mem_percent < 50 and current_batch_size < 512:
            new_batch_size = min(512, current_batch_size * 2)
            if new_batch_size != current_batch_size:
                logger.info(
                    f"🟢 Низкий уровень памяти: {mem_percent:.1f}%. "
                    f"Увеличиваем batch: {current_batch_size} → {new_batch_size}"
                )
            return new_batch_size

        return current_batch_size

    except Exception as e:
        logger.error(f"Ошибка проверки памяти: {e}")
        return max(32, current_batch_size)  # Fallback на безопасное значение
```

**ДОПОЛНИТЕЛЬНО: Force GC каждые N батчей**

```python
# Добавить в _index_chunks_batch() после обработки каждого батча

batch_counter = 0
for batch in batches:
    # ... обработка батча ...

    batch_counter += 1

    # Force GC каждые 5 батчей при высокой памяти
    if batch_counter % 5 == 0:
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            logger.info(f"Force GC: memory at {memory.percent:.1f}%")
            gc.collect()
            await asyncio.sleep(0.5)  # Даём время на GC
```

**Ожидаемый результат:**
- ✅ Batch size автоматически уменьшится до 8-16 при >92% RAM
- ✅ Индексация станет медленнее, но **НЕ УПАДЁТ**
- ✅ Memory spikes будут меньше (<5GB вместо 10GB)
- ⏱️ Время индексации: увеличится в 2-4 раза (приемлемо для стабильности)

---

### ⭐⭐ Стратегия 2: MODEL QUANTIZATION (КРАТКОСРОЧНО - 1-2 часа)

**Приоритет:** 🟠 P1 - ВЫСОКИЙ

**Идея:** Уменьшить размер модели с FP32 до INT8 (квантование)

**Преимущества:**
- ✅ **Экономия ~50% памяти** (15GB → 7-8GB)
- ✅ Небольшая потеря качества (<2%)
- ✅ Ускорение inference (~30-40%)

**Недостатки:**
- ⚠️ Требует `bitsandbytes` библиотеку
- ⚠️ Первая загрузка медленнее (квантование на лету)
- ⚠️ Поддержка только CUDA/ROCm (для CPU нужен ONNX)

---

**РЕШЕНИЕ 2.1: 8-bit quantization (NVIDIA GPU required)**

**Файл:** `rag/embedder.py`

**Требования:**
```bash
pip install bitsandbytes>=0.41.0
```

**Код:**
```python
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

def load_jina_v3_quantized():
    """
    Загрузка Jina v3 с 8-bit квантованием.

    Экономия памяти: ~50% (15GB → 7-8GB)
    """
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        device="cuda",  # Требуется GPU
        model_kwargs={
            "quantization_config": quantization_config
        }
    )

    logger.info("✅ Jina v3 загружена с 8-bit квантованием")
    return model
```

**АЛЬТЕРНАТИВА 2.2: ONNX quantization (CPU compatible)**

**Файл:** `rag/embedder.py`

**Требования:**
```bash
pip install optimum[onnxruntime]>=1.14.0
pip install onnxruntime>=1.16.0
```

**Код:**
```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

def load_jina_v3_onnx():
    """
    Загрузка Jina v3 в ONNX формате с квантованием.

    Преимущества:
    - Работает на CPU
    - Экономия памяти ~40%
    - Ускорение inference ~50%
    """
    model = ORTModelForFeatureExtraction.from_pretrained(
        "jinaai/jina-embeddings-v3",
        export=True,  # Конвертация в ONNX
        provider="CPUExecutionProvider"
    )

    # Квантование динамическое
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input="model.onnx",
        model_output="model_quantized.onnx",
        weight_type=QuantType.QUInt8
    )

    logger.info("✅ Jina v3 загружена в ONNX INT8")
    return model
```

**Интеграция в существующий код:**

```python
# rag/embedder.py - обновить __init__

class CPUEmbedder:
    def __init__(self, config: EmbeddingConfig):
        # ... existing code ...

        # Выбираем квантованную версию если доступно
        if os.getenv("USE_QUANTIZED_MODEL", "1") == "1":
            try:
                self.model = load_jina_v3_quantized()  # Или load_jina_v3_onnx()
                logger.info("✅ Используем квантованную модель")
            except Exception as e:
                logger.warning(f"Квантование недоступно: {e}. Fallback на FP32")
                self.model = self._load_default_model()
        else:
            self.model = self._load_default_model()
```

**Ожидаемый результат:**
- ✅ Memory usage: 15GB → 7-8GB (экономия ~8GB)
- ✅ Доступная память: 62.68GB → 54GB используется (86% вместо 99%)
- ✅ Inference speed: +30-40% быстрее
- ⚠️ Quality: -1-2% (приемлемо)

---

### ⭐⭐⭐ Стратегия 3: MODEL OFFLOADING (ДОЛГОСРОЧНО - 3-5 дней)

**Приоритет:** 🟡 P2 - СРЕДНИЙ

**Идея:** Выгружать модель из памяти когда она не используется

**Преимущества:**
- ✅ **Экономия ~15-20GB** когда модель не нужна
- ✅ Защита от memory leaks
- ✅ Graceful degradation

**Недостатки:**
- ⚠️ Задержка при первом запросе после выгрузки (~10-15 секунд)
- ⚠️ Сложность реализации (управление lifecycle)

---

**РЕШЕНИЕ: ModelManager с lazy loading**

**Файл:** `vm_rag_service.py`

**Новый класс:**
```python
import threading
import time
import gc
from typing import Optional

class ModelManager:
    """
    Управление lifecycle модели Jina v3.

    Фичи:
    - Lazy loading: загрузка по требованию
    - Auto-offload: выгрузка после idle timeout
    - Thread-safe: mutex для concurrent доступа
    - Memory monitoring: отслеживание использования
    """

    def __init__(self, idle_timeout: int = 300):
        """
        Args:
            idle_timeout: Время бездействия до выгрузки (секунды)
                         Default: 300s (5 минут)
        """
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.last_used: float = 0
        self.idle_timeout = idle_timeout
        self.lock = threading.Lock()
        self.is_loading = False

        # Запускаем фоновый мониторинг
        self._start_monitor_thread()

    def get_model(self):
        """
        Получить модель (загрузить если нужно).

        Thread-safe, блокирующий метод.
        """
        with self.lock:
            if self.model is None:
                self._load_model()

            self.last_used = time.time()
            return self.model, self.tokenizer

    def _load_model(self):
        """Загрузка модели в память"""
        if self.is_loading:
            logger.info("Модель уже загружается, ожидаем...")
            return

        self.is_loading = True
        logger.info("🔄 Загрузка Jina v3 модели...")
        start_time = time.time()

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                "jinaai/jina-embeddings-v3",
                trust_remote_code=True
            )
            self.tokenizer = self.model.tokenizer

            elapsed = time.time() - start_time
            memory = psutil.virtual_memory()

            logger.info(
                f"✅ Модель загружена за {elapsed:.2f}s. "
                f"Memory: {memory.percent:.1f}% "
                f"(available: {memory.available / (1024**3):.1f}GB)"
            )

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
        finally:
            self.is_loading = False

    def _offload_model(self):
        """Выгрузка модели из памяти"""
        with self.lock:
            if self.model is None:
                return

            logger.info("🗑️ Выгрузка модели (idle timeout reached)")

            # Удаляем модель
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()

            memory = psutil.virtual_memory()
            logger.info(
                f"✅ Модель выгружена. "
                f"Memory: {memory.percent:.1f}% "
                f"(freed: ~15-20GB)"
            )

    def _start_monitor_thread(self):
        """Запуск фонового мониторинга для auto-offload"""
        def monitor():
            while True:
                time.sleep(60)  # Проверка каждую минуту

                if self.model is None:
                    continue

                idle_time = time.time() - self.last_used

                if idle_time > self.idle_timeout:
                    logger.info(
                        f"⏰ Idle timeout: {idle_time:.0f}s > {self.idle_timeout}s"
                    )
                    self._offload_model()

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info(f"✅ Model monitor started (idle_timeout={self.idle_timeout}s)")

    def force_offload(self):
        """Принудительная выгрузка (для admin API)"""
        self._offload_model()

    def get_stats(self) -> dict:
        """Статистика ModelManager"""
        with self.lock:
            is_loaded = self.model is not None
            idle_time = time.time() - self.last_used if self.last_used > 0 else None

            return {
                "model_loaded": is_loaded,
                "last_used_ago_seconds": idle_time,
                "idle_timeout": self.idle_timeout,
                "is_loading": self.is_loading
            }


# Глобальный instance
model_manager = ModelManager(idle_timeout=300)


# Использование в endpoints
@app.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    memory_check_middleware()

    # Получаем модель через manager
    model, tokenizer = model_manager.get_model()

    # ... existing embedding logic ...
```

**Новый admin endpoint:**
```python
@app.post("/admin/model/offload")
async def admin_offload_model():
    """
    Принудительная выгрузка модели.

    Полезно для освобождения памяти перед maintenance.
    """
    model_manager.force_offload()
    return {"status": "model offloaded"}

@app.get("/admin/model/stats")
async def admin_model_stats():
    """Статистика модели"""
    return model_manager.get_stats()
```

**Ожидаемый результат:**
- ✅ Модель выгружается после 5 минут бездействия
- ✅ Экономия ~15-20GB когда не используется
- ✅ Автоматическая загрузка при новых запросах
- ⚠️ Первый запрос после offload: +10-15s задержка

---

## 📋 ПЛАН ДЕЙСТВИЙ

### 🔴 НЕМЕДЛЕННО (следующие 30 минут)

**Действие 1: HOTFIX - Aggressive Batch Reduction**

```bash
# 1. Остановить текущую индексацию (если запущена)
# В web UI: нажать Cancel / Stop

# 2. Применить изменения в indexer_service.py
# Обновить метод _check_memory_and_adjust_batch() (код выше)

# 3. Commit + Push
git add rag/indexer_service.py
git commit -m "HOTFIX: Aggressive batch reduction для 99% RAM"
git push origin refactor_tests

# 4. Deploy на VM
python vm_start.py update --branch refactor_tests

# 5. Restart сервиса
python vm_start.py stop
python vm_start.py start

# 6. Проверка
curl http://10.61.11.54:8000/health
```

**Ожидаемое время:** 15-20 минут

---

**Действие 2: Force GC + Restart**

```bash
# SSH на VM
ssh user@10.61.11.54

# Проверка памяти
free -h

# Restart Python сервиса (очистит память)
cd ~/repo_sum_rag/repo_sum
pkill -f vm_rag_service.py
sleep 5
nohup python vm_rag_service.py > rag_service.log 2>&1 &

# Проверка после restart
free -h
tail -f rag_service.log
```

**Ожидаемый результат:** Memory usage падает до ~40-50GB

---

**Действие 3: Тестовая индексация с малым батчем**

```bash
# В локальном web_ui.py или CLI
# Попробовать индексацию с явно указанным малым batch

python main.py rag index /path/to/small/repo --batch-size 16
```

**Критерии успеха:**
- ✅ Memory не превышает 90%
- ✅ Индексация завершается без ошибок
- ✅ Circuit breaker остаётся CLOSED

---

### 🟠 КРАТКОСРОЧНО (следующие 2-4 часа)

**Действие 4: Мониторинг и метрики**

```bash
# Установить monitoring скрипт
# Создать: scripts/monitor_vm_memory.py

import psutil
import time
import requests

while True:
    memory = psutil.virtual_memory()

    if memory.percent > 90:
        print(f"🚨 WARNING: Memory at {memory.percent:.1f}%")

        # Опционально: send alert
        # requests.post("https://alerts.example.com/webhook", ...)

    time.sleep(30)
```

---

**Действие 5: Начать подготовку квантования**

```bash
# Исследование доступных опций
pip list | grep -i quantiz
pip list | grep -i onnx

# Тестирование на dev окружении
# (не на production VM!)
```

---

### 🟡 СРЕДНЕСРОЧНО (следующие 1-2 дня)

**Действие 6: Внедрение квантования**

- Выбрать подход: bitsandbytes vs ONNX
- Тестирование на staging
- Бенчмарки качества (precision/recall)
- Deploy на production

**Ожидаемая экономия:** ~8GB RAM

---

**Действие 7: Memory profiling**

```bash
# Установить memory_profiler
pip install memory-profiler

# Профилирование критичных функций
@profile
def embed_texts(...):
    ...

python -m memory_profiler vm_rag_service.py
```

**Цель:** Найти memory leaks и оптимизировать

---

### 🔵 ДОЛГОСРОЧНО (следующая неделя)

**Действие 8: Model Offloading**

- Реализовать ModelManager
- Тестирование lifecycle
- Мониторинг latency
- Настройка idle_timeout

**Ожидаемая экономия:** ~15-20GB когда модель не используется

---

**Действие 9: Infrastructure upgrade**

**Варианты:**

1. **Увеличить RAM до 128GB** 💰
   - Pros: Проблема решена полностью
   - Cons: Дорого (~$50-100/мес)

2. **Использовать GPU instance** 🚀
   - Pros: Быстрее + квантование эффективнее
   - Cons: Очень дорого (~$200-500/мес)

3. **Распределённая архитектура** 🏗️
   - Pros: Масштабируемость
   - Cons: Сложность, latency

---

## 📊 МЕТРИКИ И МОНИТОРИНГ

### Что отслеживать

**Critical metrics:**

```python
# Memory
memory.percent < 90%  # ⚠️ Warning
memory.percent < 95%  # 🚨 Critical
memory.available > 3GB  # Минимум

# Performance
embedding_latency_p95 < 500ms
indexing_rate > 5 files/sec
circuit_breaker_state == "CLOSED"

# Errors
oom_killer_invocations == 0
http_507_errors < 1%
timeouts < 2%
```

---

### Dashboard (рекомендации)

**Grafana panels:**

1. **Memory Usage Timeline**
   - Line chart: memory.percent за последние 24h
   - Alert threshold: 90%

2. **Batch Size Adaptation**
   - Line chart: current_batch_size в реальном времени
   - Показывает как система адаптируется

3. **Indexing Performance**
   - Gauge: files indexed per minute
   - Target: >5 files/min

4. **Circuit Breaker State**
   - Status panel: CLOSED / HALF_OPEN / OPEN
   - Alert on: OPEN state

5. **Model Lifecycle** (после offloading)
   - Timeline: model load/offload events
   - Gauge: idle time

---

## 🎯 КРИТЕРИИ УСПЕХА

### Минимальные требования (MUST HAVE)

- ✅ **Memory usage <90%** во время индексации
- ✅ **Индексация завершается** без OOM/timeout
- ✅ **Circuit breaker CLOSED** большую часть времени
- ✅ **Zero OOM killer invocations**

### Желаемые цели (NICE TO HAVE)

- 🎯 **Memory usage <85%** в пике
- 🎯 **Indexing speed >5 files/sec**
- 🎯 **Latency p95 <300ms**
- 🎯 **Uptime >99.5%**

---

## ⚠️ РИСКИ И МИТИГАЦИЯ

### Риск 1: OOM Killer убьёт VM сервис

**Вероятность:** 🔴 ВЫСОКАЯ (сейчас 99% RAM)

**Последствия:**
- Сервис упадёт
- Потеря данных в процессе индексации
- Downtime 5-10 минут (restart)

**Митигация:**
- ✅ Применить HOTFIX немедленно
- ✅ Мониторинг каждые 30 секунд
- ✅ Auto-restart скрипт

---

### Риск 2: Квантование ухудшит качество

**Вероятность:** 🟡 СРЕДНЯЯ

**Последствия:**
- Precision/recall -1-3%
- Пользователи заметят худшие результаты поиска

**Митигация:**
- ✅ A/B тестирование FP32 vs INT8
- ✅ Метрики качества (NDCG, MRR)
- ✅ Rollback план

---

### Риск 3: Model offloading увеличит latency

**Вероятность:** 🟢 НИЗКАЯ

**Последствия:**
- Первый запрос после offload: +10-15s
- Пользователи недовольны

**Митигация:**
- ✅ Настройка idle_timeout (300s → 600s)
- ✅ Pre-warming при predictable traffic
- ✅ Loading indicator в UI

---

## 📞 КОНТАКТЫ И ЭСКАЛАЦИЯ

### Кто принимает решения

**P0 (HOTFIX):** Можно применять сразу
**P1 (Квантование):** Согласовать с Tech Lead
**P2 (Offloading):** Обсудить с командой

### Если проблема критична

1. Остановить индексацию
2. Restart VM сервиса
3. Уведомить команду
4. Применить HOTFIX

---

## 📝 CHANGELOG

### 02.10.2025 18:45 - Создание документа

- Задокументирована критическая проблема 99% RAM
- Описаны 3 стратегии решения
- Создан план действий с приоритетами
- Добавлены примеры кода для всех стратегий

---

## 🔗 СВЯЗАННЫЕ ДОКУМЕНТЫ

- [OOM_PROTECTION.md](../OOM_PROTECTION.md) - История защиты от OOM (v1.0, v2.0)
- [Technical Architecture.md](Technical Architecture.md) - Общая архитектура
- [Technical Debt.md](Technical Debt.md) - Технический долг
- [Development Roadmap.md](Development Roadmap.md) - Дорожная карта

---

**Автор:** DevOps Team
**Последнее обновление:** 02 октября 2025, 18:45 MSK
**Статус:** 🔴 ACTIVE - ТРЕБУЕТ ВНИМАНИЯ
**Next Review:** 03 октября 2025
