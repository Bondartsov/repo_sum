# 🚨 !!!!ATTENTION: КРИТИЧЕСКАЯ ПРОБЛЕМА ПАМЯТИ VM (02.10.2025)

**Дата:** 02 октября 2025, 18:45 MSK (обновлено 19:00)
**Статус:** 🔴 КРИТИЧНО - ТРЕБУЕТ НЕМЕДЛЕННОГО ВНИМАНИЯ
**Приоритет:** P0 - БЛОКИРУЮЩАЯ ПРОБЛЕМА
**Ответственный:** AI Agent

---

## 🔥 КРИТИЧЕСКАЯ СИТУАЦИЯ

### ⚡ ОБНОВЛЕНИЕ (19:00): Root Cause Найден!

**Проблема НЕ в памяти напрямую, а в TIMEOUT из-за SWAP THRASHING!**

**Ключевое открытие:**
- ✅ VM сервис **ЖИВОЙ** - health checks проходят (`GET /health` → 200 OK)
- ❌ Embeddings запросы **TIMEOUT** - не успевают обработаться за 60 секунд
- 🎯 **Root cause:** При 99% RAM модель Jina v3 частично в swap → disk I/O → latency 500ms → 120+ секунд


---

### Симптомы

**VM достигла критического уровня использования памяти:**

```
Memory Usage: 62.68 GB / 62.79 GB (99.8%)
Available: ~100 MB
Status: 🔴 EXTREME DANGER - Swap thrashing активен
```

**Последствия:**
- ❌ Индексация repo_sum (135 файлов) НЕВОЗМОЖНА
- ❌ Circuit breaker открыт после 5 timeout подряд
- ❌ Embeddings запросы занимают 120+ секунд (норма: 500ms)
- ❌ Swap thrashing: 100+ page faults per request
- ⚠️ **Health checks OK, но реальная работа БЛОКИРОВАНА**

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

### ⚠️ ВЫБОР РЕШЕНИЯ:

#### **Подход 1: HOTFIX - Увеличить Timeout (5 минут)** реализован, информация в **[rules\HOTFIX_TIMEOUTS.md](rules\HOTFIX_TIMEOUTS.md)**

**Цель:** Дать VM достаточно времени завершить swap-in/out

---

#### **Подход 2: Рефакторинг системы и оптимизация модели** ВЫБРАН КАК ОСНОВНОЙ И ПРИОРИТЕТНЫЙ, информация в **[rules\rerfactor_oom.md](rules\rerfactor_oom.md)**

**Цель:** устранить OOM и стабилизировать индексацию без заметной деградации качества поиска кода. Без жёсткой квантизации качества (INT8 — опционально, по флагу).
