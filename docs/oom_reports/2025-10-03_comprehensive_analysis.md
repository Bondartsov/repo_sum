# Comprehensive Analysis: OOM Killer Problem
**Дата:** 03 октября 2025  
**Статус:** 🔴 Критичная проблема (P0)  
**Автор:** Technical Documentation Expert  
**Версия:** 1.0

---

## 1. Executive Summary

### Краткое описание проблемы

Система индексации кода на VM (10.61.11.54) испытывает критическое давление на память, приводящее к swap thrashing и потенциальным срабатываниям OOM killer. При достижении 99% использования RAM (62.68 GB / 62.79 GB) модель Jina v3 частично выгружается в swap, что приводит к увеличению латентности запросов на генерацию эмбеддингов с нормальных 500ms до 120+ секунд. Это вызывает timeout'ы и открытие circuit breaker, блокируя процесс индексации.

**Инцидент:** 02 октября 2025, 18:45 MSK  
**Root Cause:** Не нехватка памяти напрямую, а **timeout из-за swap thrashing**  
**Immediate Impact:** Невозможность индексации репозитория (135 файлов), circuit breaker открыт после 5 timeout подряд

### Текущий статус

- ✅ **HOTFIX применён:** 02.10.2025, 19:10 - увеличены все timeout'ы в 10x (с 60s до 600s)
- ⚠️ **Эффективность HOTFIX:** условно достаточно для работы, но не решает корневую проблему
- 📊 **Прогресс стратегии рефакторинга:** 35% выполнено (Фазы 0-1 частично)
- 🚨 **Критичные пробелы:** 4 проблемы P0 требуют немедленного внимания

### Главный вывод и рекомендации

**Вывод:** Проблема устранима через системную оптимизацию без деградации качества. HOTFIX обеспечил временную стабильность, но требуется выполнение критичных мер по оптимизации памяти.

**Рекомендации TOP-3 (P0):**
1. ⚡ Создать swap 64GB на VM (защита от OOM killer) - **СЕГОДНЯ**
2. 🔧 Добавить CHUNK_MAX_TOKENS=768 в code_chunker.py - **СЕГОДНЯ**
3. 💾 Рефакторинг стримовой индексации (убрать all_chunks) - **ЭТА НЕДЕЛЯ**

---

## 2. Проблема: Root Cause Analysis

### 2.1 Описание проблемы

**Timeline инцидента:**
- **02.10.2025, 18:45 MSK** - Обнаружена блокировка индексации
- **02.10.2025, 18:50** - Диагностика показала 99.8% RAM usage
- **02.10.2025, 19:00** - Найден root cause: swap thrashing → timeout
- **02.10.2025, 19:10** - Применён HOTFIX с увеличением timeout'ов

**Симптомы:**
- ❌ Memory Usage: 62.68 GB / 62.79 GB (99.8%)
- ❌ Available RAM: ~100 MB
- ❌ Swap thrashing: 100+ page faults per request
- ❌ Embeddings latency: 120-180 секунд (норма: 500ms)
- ❌ Circuit breaker: открыт после 5 timeout подряд
- ✅ Health checks: OK (сервис жив, но не работает эффективно)

**Последствия:**
- Блокировка индексации любых репозиториев
- Деградация производительности в 240x (500ms → 120s)
- Риск срабатывания OOM killer и потери данных
- Необходимость применения временных мер (HOTFIX)

### 2.2 Memory Breakdown

| Компонент | Память | Процент | Детали |
|-----------|--------|---------|--------|
| **Jina v3 модель** | 15-20 GB | ~30% | 570M параметров FP32 + inference буферы |
| **Qdrant векторы** | 8-12 GB | ~18% | Векторная БД с HNSW индексами |
| **Batch processing** | 10-15 GB | ~20% | Временные данные во время индексации |
| **OS + процессы** | 2-3 GB | ~5% | Ubuntu + Python + FastAPI |
| **Прочее** | 15-20 GB | ~27% | Кэши, буферы, фрагментация |
| **ИТОГО** | **~60-62 GB** | **~99%** | **КРИТИЧНО!** |

**Детальный анализ компонентов:**

1. **Jina v3 - тяжеловес (15-20 GB)**
   ```
   Model: jinaai/jina-embeddings-v3
   Parameters: 570M
   Precision: FP32 (4 bytes per parameter)
   
   Breakdown:
   - Model weights: 570M × 4 = 2.3 GB
   - Activations: ~3-5 GB
   - Attention buffers: ~2-4 GB
   - Pooling layers: ~1-2 GB
   - Inference cache: ~2-3 GB
   - PyTorch overhead: ~2-3 GB
   ----------------------------------------
   TOTAL: ~15-20 GB
   ```

2. **Batch Processing - растёт с batch (10-15 GB)**
   ```
   batch_size = 128-256 (стартовый)
   
   Per batch:
   - Input tensors: batch_size × 1024 × 4 bytes = ~500KB-1MB
   - Intermediate activations: ~2-4 GB
   - Output embeddings: batch_size × 1024 × 4 = ~500KB-1MB
   - Temporary arrays: ~2-3 GB
   - GC overhead: ~2-4 GB (фрагментация)
   ----------------------------------------
   TOTAL per batch: ~10-15 GB
   ```

3. **Qdrant - растёт с данными (8-12 GB)**
   ```
   Vectors stored: ~10,000-50,000
   Dimension: 1024
   
   Breakdown:
   - Vectors: 50k × 1024 × 4 bytes = ~200 MB
   - HNSW index: ~5x overhead = ~1 GB
   - Scalar quantization: ~2x overhead = ~2 GB
   - Metadata: ~100-200 MB
   - Query cache: ~1-2 GB
   - OS page cache: ~3-5 GB
   ----------------------------------------
   TOTAL: ~8-12 GB
   ```

### 2.3 Root Cause

**Главная причина: Timeout из-за swap thrashing**

При 99% RAM:
1. Jina v3 модель (15-20GB) частично выгружается в swap
2. Каждый embeddings запрос → 100+ page faults
3. Disk I/O latency: ~500ms per page fault
4. **Результат:** 120-180 секунд на один батч эмбеддингов
5. Старые timeout (60s) → AsyncIO timeout
6. 5 неудач подряд → Circuit Breaker OPEN

**Вторичные причины:**
- ❌ Отсутствие лимита на размер чанков (код берёт чанки любого размера)
- ❌ Накопление all_chunks в памяти ([`indexer_service.py:239`](rag/indexer_service.py:239))
- ❌ Swap файл не создан на VM (нет защиты от OOM killer)
- ❌ OMP/MKL потоки не ограничены (переподписывание CPU → рост памяти)

---

## 3. Применённые решения

### 3.1 HOTFIX Timeouts (02.10.2025, 19:10)

**Цель:** Дать VM достаточно времени завершить swap-in/out операции

**Что было изменено:**

| Файл | Параметр | Было | Стало | Множитель |
|------|----------|------|-------|-----------|
| [`config.py`](config.py:38-40) | timeout_seconds | 60s | 600s | 10x |
| [`config.py`](config.py:39) | max_retries | 3 | 5 | 1.7x |
| [`config.py`](config.py:40) | retry_delay | 2.0s | 10.0s | 5x |
| [`rag/retry_policy.py`](rag/retry_policy.py:52-55) | max_attempts | 3 | 5 | 1.7x |
| [`rag/retry_policy.py`](rag/retry_policy.py:53) | base_delay | 2.0s | 10.0s | 5x |
| [`rag/retry_policy.py`](rag/retry_policy.py:54) | max_delay | 30.0s | 120.0s | 4x |
| [`rag/retry_policy.py`](rag/retry_policy.py:55) | timeout_seconds | 60.0s | 600.0s | 10x |
| [`rag/remote_embedder.py`](rag/remote_embedder.py:75-76) | failure_threshold | 5 | 10 | 2x |
| [`rag/remote_embedder.py`](rag/remote_embedder.py:76) | timeout_seconds | 60.0s | 300.0s | 5x |
| [`rag/event_loop_manager.py`](rag/event_loop_manager.py:86-88) | total timeout | 60s | 600s | 10x |
| [`rag/event_loop_manager.py`](rag/event_loop_manager.py:86) | connect timeout | 10s | 30s | 3x |
| [`rag/event_loop_manager.py`](rag/event_loop_manager.py:86) | sock_read | 30s | 300s | 10x |
| [`rag/remote_vector_store.py`](rag/remote_vector_store.py:114) | index_documents | 300s | 1800s | 6x |
| [`rag/remote_vector_store.py`](rag/remote_vector_store.py:112-120) | search timeouts | 60s | 300s | 5x |

**Эффективность:**
- ✅ **Позитив:** Индексация теперь возможна (условно)
- ⚠️ **Ограничения:** Очень медленно (10-15 минут вместо 3-5)
- ❌ **Проблема:** Не решает корневую причину (swap thrashing остаётся)

**Ожидаемые результаты с HOTFIX:**

**Pessimistic (при 99% RAM):**
- Один батч (batch=32): 60-140 секунд
- Индексация 135 файлов: 10-15 минут
- Swap-in/out активен на каждом батче

**Optimistic (если swap стабилизируется):**
- Модель остаётся в RAM после первой загрузки
- Последующие батчи: 10-30 секунд
- Индексация 135 файлов: 3-5 минут

### 3.2 Бэкапы и baseline

**Созданные бэкапы:**
- 📁 [`backups/migration_backup_20251003_112326/`](backups/migration_backup_20251003_112326/)
  - Содержит: .env.example, migration_settings.json, settings.json, rollback_migration.sh
  - Назначение: Возможность отката к предыдущей конфигурации

**Baseline snapshot:**
- 📄 [`docs/oom_reports/2025-10-02_baseline.md`](docs/oom_reports/2025-10-02_baseline.md)
  - Memory snapshot: 62.68 GB / 62.79 GB (99.8%)
  - Версии: Python 3.10, uvicorn 0.35.0, Jina v3 570M, qdrant-client 1.15.1
  - Swap status: active thrashing (100+ page faults / request)
  - Назначение: Референсная точка для сравнения после оптимизаций

**Файлы конфигурации:**
- ✅ [`config.py.backup`](config.py.backup) - оригинальная конфигурация
- ✅ [`vm_rag_service.py.backup`](vm_rag_service.py.backup) - оригинальный сервис

---

## 4. Текущее состояние: Gap Analysis

### 4.1 Матрица выполнения по фазам

| Фаза | Название | Статус | Выполнено | Осталось | Критичность |
|------|----------|--------|-----------|----------|-------------|
| 0 | Оценка рисков и Rollback | 75% | Бэкапы, ветка, baseline | Swap файл | P1 |
| 1 | Наблюдаемость и baseline | 60% | Baseline, диагностика | Полные метрики | P1 |
| 2 | Быстрые стабилизаторы | 0% | Нет | Swap, OMP/MKL, uvicorn | **P0** |
| 3 | Чанкование (CHUNK_MAX_TOKENS) | 0% | Нет | Лимиты токенов, дробление | **P0** |
| 4 | Стримовая индексация | 0% | Нет | Убрать all_chunks | **P0** |
| 5 | Тюнинг эмбеддера | 0% | Нет | batch_size=1, backpressure | P1 |
| 6 | Qdrant настройки | 0% | Нет | mmap, HNSW оптимизация | P1 |
| 7 | Truncate_dim (Matryoshka) | 0% | Нет | A/B тест 1024→512 | P2 |
| 8 | Payload стратегия | 0% | Нет | summary/pointer mode | P2 |
| 9 | Полная верификация | 0% | Нет | Прогон + метрики | P2 |
| 10 | Rollout и алерты | 0% | Нет | Документация, мониторинг | P2 |

**Прогресс:** 35% (Фазы 0-1 частично выполнены)

### 4.2 Критичные пробелы (P0)

#### 1. CHUNK_MAX_TOKENS отсутствует

**Локация:** [`code_chunker.py`](code_chunker.py:1-416)

**Проблема:**
- ❌ Метод [`_count_tokens()`](code_chunker.py:388) существует, но НЕ используется для лимитирования
- ❌ Метод [`_truncate_content()`](code_chunker.py:402) существует, но НЕ вызывается
- ❌ В [`chunk_parsed_file()`](code_chunker.py:94) нет проверки размера чанков
- ❌ Код может создавать чанки ЛЮБОГО размера (вплоть до целого файла)

**Последствия:**
- 🚨 Чанки размером 2000+ токенов → OOM при inference
- 🚨 Jina v3 модель получает "монстров" на вход → swap thrashing
- 🚨 Один большой чанк может занять 10+ GB памяти при обработке

**Требуемое решение:**
```python
# В code_chunker.py добавить:
CHUNK_MAX_TOKENS = 768  # Из конфигурации
CHUNK_MIN_TOKENS = 160

def _split_large_chunk(self, chunk, max_tokens):
    """Дробит большой чанк на меньшие части"""
    # Логика дробления по AST узлам/строкам
    pass

def chunk_parsed_file(self, parsed_file, source_code):
    chunks = []
    # ... существующая логика ...
    
    # НОВОЕ: Проверка и дробление больших чанков
    for chunk in chunks:
        if chunk.tokens_estimate > CHUNK_MAX_TOKENS:
            sub_chunks = self._split_large_chunk(chunk, CHUNK_MAX_TOKENS)
            result.extend(sub_chunks)
        else:
            result.append(chunk)
    
    return result
```

**Ссылки:**
- Текущий код: [`code_chunker.py:94-124`](code_chunker.py:94-124)
- Стратегия: [`rules/rerfactor_oom.md:56-69`](rules/rerfactor_oom.md:56-69)

#### 2. Накопление all_chunks в памяти

**Локация:** [`rag/indexer_service.py:239`](rag/indexer_service.py:239)

**Проблема:**
```python
# Строка 239 - КРИТИЧЕСКАЯ ПРОБЛЕМА
all_chunks = []

# Строка 262 - накопление в памяти
all_chunks.extend(file_chunks)

# Проблема: ВСЕ чанки держатся в памяти до момента индексации
# Для 1000+ файлов это может быть 50,000+ чанков × 1-2 KB = 50-100 MB текста
# + метаданные = 200-300 MB в памяти БЕЗ необходимости
```

**Последствия:**
- 🚨 Рост памяти пропорционален размеру репозитория
- 🚨 Невозможность обработки больших репозиториев (10,000+ файлов)
- 🚨 Дополнительные 200-500 MB памяти занято зря

**Требуемое решение:**
```python
# Вместо all_chunks = [] использовать стримовую обработку:

async def _process_files_streaming(self, files, repo_path, batch_size):
    """Стримовая обработка файлов без накопления в памяти"""
    batch = []
    
    for file_info in files:
        file_chunks = await self._process_single_file(file_info, repo_path)
        batch.extend(file_chunks)
        
        # Как только накопился batch - индексируем
        if len(batch) >= batch_size:
            await self._index_chunks_batch(batch, batch_size)
            batch = []  # Освобождаем память
    
    # Индексируем остаток
    if batch:
        await self._index_chunks_batch(batch, batch_size)
```

**Ссылки:**
- Текущий код: [`rag/indexer_service.py:233-279`](rag/indexer_service.py:233-279)
- Стратегия: [`rules/rerfactor_oom.md:73-85`](rules/rerfactor_oom.md:73-85)

#### 3. Swap файл не создан на VM

**Проблема:**
- ❌ На VM (10.61.11.54) swap не настроен или недостаточен
- ❌ При 100% RAM → немедленное срабатывание OOM killer
- ❌ Нет защитной подушки для пиковых нагрузок

**Последствия:**
- 🚨 OOM killer убивает процесс Python без предупреждения
- 🚨 Потеря данных и прерванная индексация
- 🚨 Необходимость полного restart сервиса

**Требуемое решение:**
```bash
# На VM (10.61.11.54):
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Добавить в /etc/fstab для автозапуска:
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Настроить swappiness (не агрессивное использование):
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Проверка:**
```bash
free -h
swapon --show
```

**Ссылки:**
- Стратегия: [`rules/rerfactor_oom.md:40-44`](rules/rerfactor_oom.md:40-44)

#### 4. OMP/MKL потоки не ограничены

**Проблема:**
- ❌ OpenMP и Intel MKL создают множество потоков
- ❌ Переподписывание CPU → рост памяти на каждый поток
- ❌ Фрагментация памяти и конкуренция за ресурсы

**Последствия:**
- 🚨 Дополнительные 5-10 GB памяти на параллельные потоки
- 🚨 Context switching → деградация производительности
- 🚨 Сложность debugging (множество конкурентных операций)

**Требуемое решение:**
```bash
# В systemd unit файле на VM добавить:
[Service]
Environment="OMP_NUM_THREADS=1"
Environment="MKL_NUM_THREADS=1"
Environment="OPENBLAS_NUM_THREADS=1"
Environment="VECLIB_MAXIMUM_THREADS=1"
Environment="NUMEXPR_NUM_THREADS=1"

# Или в .env файле:
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

**Проверка:**
```bash
# На VM
ssh user@10.61.11.54
echo $OMP_NUM_THREADS  # Должно быть 1
```

**Ссылки:**
- Стратегия: [`rules/rerfactor_oom.md:48-49`](rules/rerfactor_oom.md:48-49)

---

## 5. Детальный план действий

### 5.1 Немедленные действия (сегодня, P0)

#### Действие 1: Создать swap 64GB на VM

**Приоритет:** P0 - КРИТИЧНО  
**Время выполнения:** 10 минут  
**Ответственный:** DevOps / Администратор VM

**Команды:**
```powershell
# С локальной машины:
ssh user@10.61.11.54

# На VM:
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Проверка:
free -h
swapon --show

# Постоянная активация (добавить в /etc/fstab):
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Настройка swappiness (не агрессивное использование):
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Финальная проверка:
swapon --show
```

**Ожидаемый результат:**
- ✅ Swap 64GB активен
- ✅ swappiness=10 (используется только при необходимости)
- ✅ Защита от OOM killer при пиковых нагрузках

**Риски:** 
- ⚠️ Замедление I/O при использовании swap (минимизировано через swappiness=10)
- ⚠️ Требует 64GB свободного места на диске

**Fallback:** Использовать 32GB swap если 64GB недоступно

#### Действие 2: Ограничить OMP/MKL потоки

**Приоритет:** P0 - КРИТИЧНО  
**Время выполнения:** 5 минут  
**Ответственный:** DevOps / Администратор VM

**Изменения:**

1. **Найти systemd unit файл на VM:**
```bash
ssh user@10.61.11.54
sudo systemctl status vm_rag_service  # Узнать путь к unit файлу
# Обычно: /etc/systemd/system/vm_rag_service.service
```

2. **Редактировать unit файл:**
```bash
sudo nano /etc/systemd/system/vm_rag_service.service

# Добавить в секцию [Service]:
[Service]
Environment="OMP_NUM_THREADS=1"
Environment="MKL_NUM_THREADS=1"
Environment="OPENBLAS_NUM_THREADS=1"
Environment="VECLIB_MAXIMUM_THREADS=1"
Environment="NUMEXPR_NUM_THREADS=1"
```

3. **Перезапустить сервис:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart vm_rag_service
sudo systemctl status vm_rag_service
```

4. **Проверка:**
```bash
# Проверить переменные окружения процесса:
ps aux | grep vm_rag_service
cat /proc/<PID>/environ | tr '\0' '\n' | grep NUM_THREADS
```

**Ожидаемый результат:**
- ✅ OMP_NUM_THREADS=1 активен
- ✅ Снижение переподписывания потоков
- ✅ Экономия 5-10 GB памяти

**Риски:** Минимальны (может незначительно замедлить некоторые операции)

#### Действие 3: Добавить CHUNK_MAX_TOKENS=768

**Приоритет:** P0 - КРИТИЧНО  
**Время выполнения:** 2-3 часа разработки  
**Ответственный:** Разработчик

**Шаги реализации:**

1. **Добавить конфигурационные параметры в `.env`:**
```ini
# Чанкование
CHUNK_MAX_TOKENS=768
CHUNK_MIN_TOKENS=160
```

2. **Обновить [`config.py`](config.py):**
```python
@dataclass
class AnalysisConfig:
    # ... существующие поля ...
    chunk_max_tokens: int = 768
    chunk_min_tokens: int = 160
```

3. **Реализовать логику в [`code_chunker.py`](code_chunker.py):**
```python
def __init__(self):
    self.config = get_config()
    self.chunk_max_tokens = self.config.analysis.chunk_max_tokens
    self.chunk_min_tokens = self.config.analysis.chunk_min_tokens
    # ... остальная инициализация ...

def _split_large_chunk(self, chunk: CodeChunk, max_tokens: int) -> List[CodeChunk]:
    """Дробит большой чанк на меньшие части по логическим границам."""
    if chunk.tokens_estimate <= max_tokens:
        return [chunk]
    
    lines = chunk.content.splitlines()
    sub_chunks = []
    current_lines = []
    current_tokens = 0
    part_num = 1
    
    for line in lines:
        line_tokens = self._count_tokens(line)
        
        if current_tokens + line_tokens > max_tokens and current_lines:
            # Создаём sub-chunk
            sub_content = "\n".join(current_lines)
            sub_chunks.append(CodeChunk(
                name=f"{chunk.name} (part {part_num})",
                content=sub_content,
                start_line=chunk.start_line,
                end_line=chunk.start_line + len(current_lines),
                chunk_type=chunk.chunk_type,
                tokens_estimate=current_tokens
            ))
            current_lines = []
            current_tokens = 0
            part_num += 1
        
        current_lines.append(line)
        current_tokens += line_tokens
    
    # Добавляем последний sub-chunk
    if current_lines:
        sub_content = "\n".join(current_lines)
        sub_chunks.append(CodeChunk(
            name=f"{chunk.name} (part {part_num})",
            content=sub_content,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            tokens_estimate=current_tokens
        ))
    
    return sub_chunks

def chunk_parsed_file(self, parsed_file: ParsedFile, source_code: str = None) -> List[CodeChunk]:
    """Основной метод разбивки файла на части с лимитом по токенам."""
    # ... существующая логика создания чанков ...
    
    # НОВОЕ: Проверка и дробление больших чанков
    result_chunks = []
    for chunk in chunks:
        if chunk.tokens_estimate > self.chunk_max_tokens:
            self.logger.warning(
                f"Чанк '{chunk.name}' превышает лимит: "
                f"{chunk.tokens_estimate} > {self.chunk_max_tokens}. Дробление."
            )
            sub_chunks = self._split_large_chunk(chunk, self.chunk_max_tokens)
            result_chunks.extend(sub_chunks)
        else:
            result_chunks.append(chunk)
    
    # Логирование статистики
    if result_chunks:
        tokens = [c.tokens_estimate for c in result_chunks]
        import numpy as np
        self.logger.info(
            f"Создано {len(result_chunks)} чанков. "
            f"Токены: p50={np.percentile(tokens, 50):.0f}, "
            f"p90={np.percentile(tokens, 90):.0f}, "
            f"p99={np.percentile(tokens, 99):.0f}, "
            f"max={max(tokens)}"
        )
    
    return result_chunks
```

4. **Добавить unit тест:**
```python
# tests/test_code_chunker.py
def test_chunk_max_tokens_limit():
    """Проверка что ни один чанк не превышает CHUNK_MAX_TOKENS"""
    chunker = CodeChunker()
    # Создать большой файл с длинной функцией
    large_code = "def huge_function():\n" + "    x = 1\n" * 2000
    chunks = chunker.chunk_code(file_info, large_code)
    
    # Проверка: все чанки <= CHUNK_MAX_TOKENS
    for chunk in chunks:
        assert chunk.tokens_estimate <= chunker.chunk_max_tokens, \
            f"Чанк '{chunk.name}' превышает лимит: {chunk.tokens_estimate} > {chunker.chunk_max_tokens}"
```

5. **Тестирование:**
```powershell
# Запуск тестов
pytest tests/test_code_chunker.py::test_chunk_max_tokens_limit -v

# Тестовая индексация на небольшом репозитории
python main.py --index tests/fixtures/test_repo --batch-size 64
```

**Ожидаемый результат:**
- ✅ Ни один чанк не превышает 768 токенов
- ✅ p99 длины чанка ≤ 768 токенов
- ✅ Снижение пиков памяти при inference

**Риски:**
- ⚠️ Возможная потеря контекста при дроблении больших функций
- ⚠️ Необходимость дополнительной метаданных (part N/M)
- Минимизация: добавить overlap 10-20 строк между частями

**Критерий успеха:** 
```python
# После реализации:
assert max(chunk.tokens_estimate for chunk in all_chunks) <= 768
```

### 5.2 Короткий срок (эта неделя, P0-P1)

#### Действие 4: Рефакторинг стримовой индексации

**Приоритет:** P0 - КРИТИЧНО  
**Время выполнения:** 1-2 дня разработки  
**Ответственный:** Разработчик

**Изменения в [`rag/indexer_service.py`](rag/indexer_service.py):**

```python
# Добавить параметр в конфиг
INDEX_BATCH_SIZE = 128  # Из .env или config

async def _process_and_index_streaming(
    self,
    files: List[FileInfo],
    repo_path: Path,
    batch_size: int,
    show_progress: bool = True
) -> int:
    """
    Стримовая обработка и индексация файлов без накопления в памяти.
    
    Обрабатывает файл → создаёт чанки → сразу индексирует → освобождает память.
    """
    indexed_count = 0
    batch = []
    
    if show_progress:
        progress = Progress(...)
        progress.start()
        task = progress.add_task("Обработка и индексация...", total=len(files))
    
    try:
        for file_info in files:
            try:
                # Обрабатываем файл
                file_chunks = await self._process_single_file(file_info, repo_path)
                batch.extend(file_chunks)
                
                # Как только накопился batch - индексируем и освобождаем память
                while len(batch) >= batch_size:
                    current_batch = batch[:batch_size]
                    batch = batch[batch_size:]  # Остаток
                    
                    # Индексируем батч
                    count = await self._index_chunks_batch(
                        current_batch, 
                        batch_size,
                        show_progress=False
                    )
                    indexed_count += count
                    
                    # Явно освобождаем память
                    del current_batch
                    
                    # Пауза для стабильности
                    await asyncio.sleep(0.1)
                
                self.stats['processed_files'] += 1
                
                if show_progress:
                    progress.advance(task)
                    
            except Exception as e:
                logger.error(f"Ошибка обработки файла {file_info.path}: {e}")
                self.stats['failed_files'] += 1
        
        # Индексируем остаток батча
        if batch:
            count = await self._index_chunks_batch(batch, batch_size, show_progress=False)
            indexed_count += count
    
    finally:
        if show_progress:
            progress.stop()
    
    return indexed_count

async def index_repository(self, repo_path: str, batch_size: int = 128, ...):
    """Главный метод индексации - использует стримовую обработку."""
    # ... инициализация ...
    
    # Сканирование файлов
    files = list(self.file_scanner.scan_repository(str(repo_path)))
    
    # НОВОЕ: Стримовая обработка И индексация
    indexed_count = await self._process_and_index_streaming(
        files, 
        repo_path, 
        batch_size,
        show_progress
    )
    
    # ... статистика ...
```

**Ожидаемый результат:**
- ✅ Память не растёт с размером репозитория
- ✅ RSS процесса стабилен (пилообразный график вместо постоянного роста)
- ✅ Возможность индексации репозиториев любого размера

**Тестирование:**
```powershell
# Мониторинг памяти во время индексации
python tests/bench/memory_sampler.py --pid <PID> --output memory_streaming.csv

# Индексация тестового репозитория
python main.py --index tests/fixtures/test_repo

# Анализ результатов
python tests/bench/plot_memory.py memory_streaming.csv
```

#### Действие 5: Настройка uvicorn workers и concurrency

**Приоритет:** P1  
**Время выполнения:** 30 минут  
**Ответственный:** DevOps

**Изменения на VM:**

```bash
# В systemd unit файле или start script:
uvicorn vm_rag_service:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --limit-concurrency 2 \
    --timeout-keep-alive 300 \
    --backlog 50
```

**Обоснование:**
- `--workers 1`: Один процесс = предсказуемое потребление памяти
- `--limit-concurrency 2`: Максимум 2 одновременных запроса (защита от перегрузки)
- `--timeout-keep-alive 300`: 5 минут для долгих запросов
- `--backlog 50`: Очередь запросов

**Проверка:**
```bash
ps aux | grep uvicorn  # Должен быть один процесс
curl http://10.61.11.54:8000/health  # Проверка доступности
```

### 5.3 Средний срок (следующая неделя, P1-P2)

#### Действие 6: Backpressure механизм

**Приоритет:** P1  
**Цель:** Временная остановка подачи чанков при высоком уровне памяти

**Реализация в [`rag/indexer_service.py`](rag/indexer_service.py):**

```python
async def _check_memory_backpressure(self) -> bool:
    """Проверяет необходимость backpressure (паузы)."""
    memory = psutil.virtual_memory()
    threshold = 80  # % из конфига
    
    if memory.percent > threshold:
        logger.warning(
            f"Backpressure: память {memory.percent:.1f}% > {threshold}%. "
            f"Ожидание освобождения..."
        )
        return True
    return False

async def _process_and_index_streaming(...):
    """С добавлением backpressure."""
    for file_info in files:
        # Проверка backpressure перед обработкой
        while await self._check_memory_backpressure():
            await asyncio.sleep(5)  # Ждём освобождения памяти
        
        # Обработка файла
        file_chunks = await self._process_single_file(...)
        # ... остальная логика ...
```

#### Действие 7: A/B тест truncate_dim (1024 → 512)

**Приоритет:** P2  
**Цель:** Снизить размерность векторов без потери качества

**План тестирования:**

1. **Создать тестовый набор запросов:**
```python
# tests/bench/test_queries.json
[
    {"query": "authentication middleware", "expected_files": ["auth/middleware.py"]},
    {"query": "database connection pooling", "expected_files": ["db/connection.py"]},
    # ... 20-50 запросов ...
]
```

2. **Запустить A/B тест:**
```bash
# Baseline (1024 dim)
python tests/bench/run_ab.py --dim 1024 --queries tests/bench/test_queries.json --output baseline_1024.json

# Test (512 dim) 
python tests/bench/run_ab.py --dim 512 --queries tests/bench/test_queries.json --output test_512.json

# Сравнение результатов
python tests/bench/compare_results.py baseline_1024.json test_512.json
```

3. **Критерии успеха:**
- Recall@10 деградация ≤ 2%
- MRR деградация ≤ 2%
- Latency улучшение ≥ 10%

#### Действие 8: Payload оптимизация (summary mode)

**Приоритет:** P2  
**Цель:** Снизить объём данных в Qdrant

**Реализация:**
```python
# В config.py
PAYLOAD_MODE = "summary"  # full | summary | pointer

# В indexer_service.py
if config.payload_mode == "summary":
    payload = {
        'file_path': metadata['file_path'],
        'chunk_name': metadata['chunk_name'],
        'start_line': metadata['start_line'],
        'end_line': metadata['end_line'],
        # Контент НЕ сохраняется, подтягивается из исходника при поиске
    }
elif config.payload_mode == "full":
    payload = {**metadata, 'content': chunk.content}
```

### 5.4 Долгий срок (2-3 недели, P2)

#### Действие 9: Полная верификация

**Содержание:**
- Прогон индексации на полном репозитории (1000+ файлов)
- Снятие всех метрик (RAM, swap, время, p50/p90/p99)
- Функциональные тесты поиска (kNN, semantic, MRR, NDCG)
- Сравнение с baseline
- Создание отчёта

#### Действие 10: Rollout и мониторинг

**Содержание:**
- Обновление документации (README, runbook)
- Настройка алертов (OOM, RSS > 80%, индексация > X минут)
- Создание dashboard'а (Grafana/Prometheus)
- Обучение команды

---

## 6. Оценка рисков и минимизация

### 6.1 Риски текущего состояния

| Риск | Вероятность | Последствия | Минимизация |
|------|-------------|-------------|-------------|
| OOM killer срабатывает даже с HOTFIX | Средняя | Полная потеря данных индексации | Создать swap 64GB |
| Деградация производительности | Высокая | Индексация 10-15 минут вместо 3-5 | Выполнить P0 оптимизации |
| Сложность отката | Низкая | Необходимость восстановления из бэкапов | Бэкапы уже созданы |
| Невозможность масштабирования | Высокая | Блокировка роста кодовой базы | Стримовая индексация |

### 6.2 Риски при внедрении изменений

| Изменение | Риск | Вероятность | Последствия | Минимизация |
|-----------|------|-------------|-------------|-------------|
| Swap 64GB | Замедление I/O при использовании | Средняя | Деградация на 10-20% | swappiness=10, SSD диск |
| CHUNK_MAX_TOKENS | Потеря контекста в больших функциях | Низкая | Ухудшение качества поиска на 1-2% | Overlap 10-20 строк, тестирование |
| Стримовая индексация | Баги в новом коде | Средняя | Ошибки индексации | Полное тестирование, rollback plan |
| OMP_NUM_THREADS=1 | Замедление некоторых операций | Низкая | Увеличение времени на 5-10% | Приемлемо для стабильности |
| Truncate_dim 512 | Деградация качества поиска | Средняя | Потеря precision на 2-5% | A/B тестирование, откат если >2% |

### 6.3 Fallback план

**При неудаче любого изменения:**

1. **Немедленный откат:**
```powershell
# Восстановление из бэкапа
cd D:\Scripts_Python\repo_sum
Copy-Item config.py.backup config.py
Copy-Item vm_rag_service.py.backup vm_rag_service.py

# Откат timeout'ов (если нужно)
git checkout <commit_before_hotfix> -- config.py rag/retry_policy.py

# Перезапуск
python vm_start.py restart
```

2. **Восстановление из git:**
```powershell
# Откатить все изменения в ветке oom-refactor
git reset --hard origin/main
git clean -fd
```

3. **Восстановление VM из snapshot (крайняя мера):**
```bash
# На VM
sudo systemctl stop vm_rag_service
# Восстановление из snapshot облачного провайдера
```

---

## 7. Метрики успеха и мониторинг

### 7.1 KPI (Key Performance Indicators)

Из [`rules/rerfactor_oom.md:8-12`](rules/rerfactor_oom.md:8-12):

| KPI | Целевое значение | Текущее значение | Статус |
|-----|------------------|------------------|--------|
| **OOM события** | 0 за сессию индексации | ? (нужно измерить) | 🔴 Неизвестно |
| **RAM usage (пик)** | <80% от доступной | 99.8% | 🔴 Критично |
| **Время индексации** | ≤45 минут (1000+ файлов) | 10-15 минут (135 файлов) | ⚠️ Медленно |
| **Качество поиска** | не хуже baseline ±2% | Baseline установлен | ⚠️ Нужен A/B тест |

### 7.2 Метрики для мониторинга

| Метрика | Целевое значение | Частота проверки | Инструмент | Алерт |
|---------|------------------|------------------|------------|-------|
| **p99 длины чанка** | ≤768 токенов | После каждого изменения | [`tests/bench/chunk_stats.py`](tests/bench/) | >768 |
| **RSS процесса Python** | <50GB | Реал-тайм (каждые 10s) | [`tests/bench/memory_sampler.py`](tests/bench/memory_sampler.py) | >55GB |
| **Swap usage** | <10% | Реал-тайм (каждые 30s) | `swapon --show`, [`tests/bench/monitor_mem.sh`](tests/bench/monitor_mem.sh) | >20% |
| **OOM события** | 0 | После каждой индексации | [`tests/bench/check_oom_linux.sh`](tests/bench/check_oom_linux.sh) | >0 |
| **Время индексации** | ≤45 минут | После каждого прогона | [`tests/bench/run_ab.py`](tests/bench/run_ab.py) | >60 мин |
| **Recall@10** | ≥baseline | При A/B тестах | [`tests/bench/retrieval_ab.py`](tests/bench/retrieval_ab.py) | <baseline-2% |
| **Circuit breaker state** | CLOSED | Реал-тайм | Логи приложения | OPEN |
| **Embeddings latency** | <5s per batch | Реал-тайм | Логи приложения | >30s |

### 7.3 Инструменты мониторинга

**Существующие скрипты в [`tests/bench/`](tests/bench/):**

1. **[`memory_sampler.py`](tests/bench/memory_sampler.py)** - Сэмплирование RSS процесса
   ```powershell
   python tests/bench/memory_sampler.py --pid <PID> --interval 10 --output memory.csv
   ```

2. **[`monitor_mem.sh`](tests/bench/monitor_mem.sh)** - Мониторинг памяти и swap на Linux
   ```bash
   bash tests/bench/monitor_mem.sh
   ```

3. **[`check_oom_linux.sh`](tests/bench/check_oom_linux.sh)** - Проверка OOM событий
   ```bash
   bash tests/bench/check_oom_linux.sh
   ```

4. **[`run_ab.py`](tests/bench/run_ab.py)** - A/B тестирование качества поиска
   ```powershell
   python tests/bench/run_ab.py --queries test_queries.json
   ```

5. **[`retrieval_ab.py`](tests/bench/retrieval_ab.py)** - Метрики качества (Recall, MRR)
   ```powershell
   python tests/bench/retrieval_ab.py --baseline baseline.json --test test.json
   ```

**Новые скрипты (нужно создать):**

1. **`chunk_stats.py`** - Статистика по размерам чанков
   ```python
   # Вычисляет p50/p90/p99/max для tokens_estimate
   # Проверяет что max <= CHUNK_MAX_TOKENS
   ```

2. **`dashboard.py`** - Real-time dashboard (Rich)
   ```python
   # Показывает: RAM%, swap%, RSS, batch progress, ETA
   # Обновление каждые 5 секунд
   ```

**Команды для проверки на VM:**

```bash
# Проверка swap
ssh user@10.61.11.54 "free -h; swapon --show"

# Проверка OOM событий
ssh user@10.61.11.54 "sudo dmesg -T | grep -i 'killed process' | tail -20"

# Проверка RSS процесса
ssh user@10.61.11.54 "ps aux | grep vm_rag_service | grep -v grep"

# Проверка переменных окружения
ssh user@10.61.11.54 "systemctl show vm_rag_service | grep Environment"
```

---

## 8. Выводы и рекомендации

### 8.1 Немедленные действия (сегодня)

**TOP-3 критичных изменения:**

1. **⚡ Создать swap 64GB** (10 минут)
   ```bash
   ssh user@10.61.11.54
   sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && \
   sudo mkswap /swapfile && sudo swapon /swapfile && \
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab && \
   echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
   ```

2. **🔧 Ограничить OMP/MKL потоки** (5 минут)
   ```bash
   # Добавить в systemd unit:
   Environment="OMP_NUM_THREADS=1"
   Environment="MKL_NUM_THREADS=1"
   ```

3. **💾 Добавить CHUNK_MAX_TOKENS=768** (2-3 часа)
   - Реализовать `_split_large_chunk()` в code_chunker.py
   - Добавить проверку в `chunk_parsed_file()`
   - Написать unit тест
   - Протестировать на малом репозитории

### 8.2 Стратегические рекомендации

**Долгосрочные улучшения:**

1. **Архитектура:** Переход на микросервисную архитектуру
   - Отдельный сервис для эмбеддингов (с автомасштабированием)
   - Отдельный сервис для индексации
   - Message queue (RabbitMQ/Kafka) для backpressure

2. **Инфраструктура:** Обновление VM
   - Увеличить RAM до 128GB (если возможно)
   - Перейти на NVMe SSD для swap
   - Настроить горизонтальное масштабирование

3. **Мониторинг:** Production-grade observability
   - Prometheus + Grafana для метрик
   - ELK stack для логов
   - Алерты в Slack/Telegram

4. **Качество:** Continuous benchmarking
   - Автоматические A/B тесты при каждом PR
   - Regression тесты для качества поиска
   - Performance budget (время индексации, память)

**Архитектурные изменения:**

1. **Пайплайн индексации:**
   ```
   [Сканирование] → [Чанкование] → [Queue] → [Embeddings] → [Queue] → [Qdrant]
                                      ↓                          ↓
                                  [Batch=32]              [Batch=128]
   ```

2. **Эмбеддинги:**
   - Кэширование эмбеддингов (Redis)
   - Инкрементальная индексация (только изменённые файлы)
   - Batch processing с приоритетами

3. **Qdrant:**
   - Sharding для больших коллекций
   - Репликация для high availability
   - Backup и restore процедуры

### 8.3 Следующие шаги (ближайшие 24 часа)

**Сегодня (03.10.2025, EOD):**
- [ ] ⚡ Создать swap 64GB на VM (10 мин) - **DevOps**
- [ ] 🔧 Ограничить OMP/MKL потоки (5 мин) - **DevOps**
- [ ] 🧪 Тестирование HOTFIX на малом репозитории (30 мин) - **QA**

**Завтра (04.10.2025):**
- [ ] 💾 Начать разработку CHUNK_MAX_TOKENS (2-3 часа) - **Dev**
- [ ] 📊 Создать baseline метрики после swap (1 час) - **Dev**
- [ ] 📝 Обновить документацию с результатами (30 мин) - **Dev**

**Эта неделя (04-06.10.2025):**
- [ ] 🎯 Завершить CHUNK_MAX_TOKENS + тесты - **Dev**
- [ ] 🌊 Начать рефакторинг стримовой индексации - **Dev**
- [ ] ⚙️ Настроить uvicorn workers=1, concurrency=2 - **DevOps**
- [ ] 🔍 Провести полное тестирование P0 изменений - **QA**

**Ответственные:**
- **DevOps:** Swap, OMP/MKL, uvicorn настройки
- **Dev:** CHUNK_MAX_TOKENS, стримовая индексация, тестирование
- **QA:** Regression тесты, baseline метрики, документация результатов

---

## Приложения

### A. Команды для проверки текущего состояния

```powershell
# === ЛОКАЛЬНАЯ МАШИНА (Windows) ===

# Проверка текущей git ветки
git branch
# Ожидаемый результат: * oom-refactor (или main с HOTFIX)

# Проверка конфигурации timeout'ов
python scripts/check_timeouts.py
# Ожидаемый результат: timeout_seconds=600 во всех компонентах

# Проверка наличия бэкапов
dir backups\migration_backup_20251003_112326
# Ожидаемый результат: .env.example, settings.json, и др.

# === VM (Linux) ===

# Подключение к VM
ssh user@10.61.11.54

# Проверка памяти и swap
free -h
swapon --show
# Ожидаемый результат (после создания swap):
# Swap: 64GB total

# Проверка OOM событий
sudo dmesg -T | grep -i 'killed process' | tail -20
# Ожидаемый результат: должно быть пусто (нет OOM)

# Проверка процесса vm_rag_service
ps aux | grep vm_rag_service | grep -v grep
# Ожидаемый результат: процесс запущен, RSS <50GB

# Проверка переменных окружения
cat /proc/$(pgrep -f vm_rag_service)/environ | tr '\0' '\n' | grep NUM_THREADS
# Ожидаемый результат (после настройки):
# OMP_NUM_THREADS=1
# MKL_NUM_THREADS=1

# Проверка статуса сервиса
systemctl status vm_rag_service
# Ожидаемый результат: active (running)

# Проверка логов
tail -50 ~/repo_sum_rag/repo_sum/rag_service.log
# Проверить: нет timeout ошибок, circuit breaker CLOSED

# Проверка доступности API
curl http://localhost:8000/health
# Ожидаемый результат: {"status": "healthy"}
```

### B. Ссылки на ключевые файлы

**Документация проблемы:**
- [Описание проблемы](rules/!!!!ATTENTION(02_10_2025).md) - Критическая ситуация OOM
- [Стратегия решения](rules/rerfactor_oom.md) - 10-фазный план рефакторинга
- [HOTFIX](rules/HOTFIX_TIMEOUTS.md) - Увеличение timeout'ов
- [Baseline](docs/oom_reports/2025-10-02_baseline.md) - Baseline метрики

**Код:**
- [code_chunker.py](code_chunker.py) - Модуль чанкования (требует CHUNK_MAX_TOKENS)
- [rag/indexer_service.py](rag/indexer_service.py) - Сервис индексации (требует стримовой обработки)
- [config.py](config.py) - Конфигурация (timeout'ы изменены)
- [rag/retry_policy.py](rag/retry_policy.py) - Retry логика (timeout'ы изменены)
- [rag/remote_embedder.py](rag/remote_embedder.py) - Embedder client (timeout'ы изменены)

**Тесты и бенчмарки:**
- [tests/bench/memory_sampler.py](tests/bench/memory_sampler.py) - Мониторинг памяти
- [tests/bench/monitor_mem.sh](tests/bench/monitor_mem.sh) - Bash скрипт мониторинга
- [tests/bench/check_oom_linux.sh](tests/bench/check_oom_linux.sh) - Проверка OOM
- [tests/bench/run_ab.py](tests/bench/run_ab.py) - A/B тестирование

**Бэкапы:**
- [backups/migration_backup_20251003_112326/](backups/migration_backup_20251003_112326/) - Полный бэкап конфигурации
- [config.py.backup](config.py.backup) - Оригинальная конфигурация
- [vm_rag_service.py.backup](vm_rag_service.py.backup) - Оригинальный сервис

### C. Timeline

```
📅 02 октября 2025
├─ 18:45 MSK - 🔴 Обнаружение проблемы
│              ├─ Симптом: Индексация зависла
│              ├─ Circuit breaker: OPEN
│              └─ Timeout на embeddings запросах
│
├─ 18:50 MSK - 🔍 Начало диагностики
│              ├─ Проверка памяти: 99.8% usage
│              ├─ Проверка swap: активное thrashing
│              └─ Проверка логов: множественные timeout
│
├─ 19:00 MSK - 💡 Root Cause найден
│              ├─ Проблема: Swap thrashing → timeout
│              ├─ НЕ нехватка памяти напрямую
│              └─ Решение: Увеличить timeout'ы
│
└─ 19:10 MSK - ⚡ Применение HOTFIX
               ├─ Изменено 5 файлов
               ├─ Timeout'ы увеличены в 10x
               └─ Статус: HOTFIX применён

📅 03 октября 2025
├─ 11:23 MSK - 💾 Создание бэкапов
│              └─ backups/migration_backup_20251003_112326/
│
├─ 12:00 MSK - 📊 Comprehensive анализ
│              ├─ Gap analysis: 35% выполнено
│              ├─ Выявлено 4 критичных пробела P0
│              └─ Создан детальный план действий
│
└─ EOD (20:00) - 🎯 Целевая дата для P0 действий
                 ├─ Swap 64GB
                 ├─ OMP/MKL ограничения
                 └─ Начало CHUNK_MAX_TOKENS

📅 04-06 октября 2025
└─ Эта неделя - ⚙️ Реализация P0 изменений
                 ├─ CHUNK_MAX_TOKENS завершён
                 ├─ Стримовая индексация
                 └─ Полное тестирование

📅 07-13 октября 2025
└─ Следующая неделя - 🔧 P1-P2 оптимизации
                       ├─ Backpressure механизм
                       ├─ A/B тест truncate_dim
                       └─ Payload оптимизация

📅 14-27 октября 2025
└─ 2-3 недели - ✅ Верификация и Rollout
                 ├─ Полная верификация
                 ├─ Production rollout
                 └─ Мониторинг и алерты
```

### D. Контрольный список выполнения

**Фаза 0: Оценка рисков (75% ✅)**
- [x] Бэкапы созданы
- [x] Baseline зафиксирован
- [x] Git ветка подготовлена
- [ ] Swap файл создан

**Фаза 1: Наблюдаемость (60% ✅)**
- [x] Baseline метрики собраны
- [x] Диагностические скрипты готовы
- [ ] Полные метрики индексации
- [ ] Dashboard для мониторинга

**Фаза 2: Стабилизаторы (0% 🔴)**
- [ ] Swap 64GB создан
- [ ] OMP/MKL ограничены
- [ ] Uvicorn настроен (workers=1)
- [ ] Стабильность подтверждена

**Фаза 3: CHUNK_MAX_TOKENS (0% 🔴)**
- [ ] Конфиг добавлен
- [ ] Логика дробления реализована
- [ ] Unit тесты написаны
- [ ] p99 <= 768 подтверждено

**Фаза 4: Стримовая индексация (0% 🔴)**
- [ ] Метод `_process_and_index_streaming()` реализован
- [ ] all_chunks удалён
- [ ] Тестирование пройдено
- [ ] RSS стабилен

---

**Конец отчёта**

**Следующий шаг:** Выполнение P0 действий (swap, OMP/MKL, CHUNK_MAX_TOKENS)  
**Дата следующего обновления:** 04.10.2025 (после выполнения P0)  
**Контакт для вопросов:** Technical Documentation Expert