# 🔥 PRODUCTION-READY RAG СИСТЕМА (ОБНОВЛЕНО: 11.08.2025)

СТАТУС: Активная реализация. Этот документ синхронизирован с новым планом: обновлён стек зависимостей, расширена конфигурация, определены компоненты эмбеддера, векторного хранилища, поискового движка и интеграции RAG в текущий пайплайн (CLI + Web UI). 

Документ — часть Memory Bank, служит «истиной» для проектирования и реализации.

---

## 🔍 ЭКСПЕРТНЫЙ АУДИТ: КРИТИЧЕСКИЕ ТЕХНИЧЕСКИЕ ДЕТАЛИ (11.08.2025)

> **Источник**: Глубокий технический аудит от экспертной модели. Эти детали критичны для корректной имплементации и должны быть учтены при реализации.

### ⚠️ КРИТИЧЕСКИЕ ПРАВКИ ПО ИНСТРУМЕНТАМ И API

#### 1.1 Квантование в sentence-transformers (ВАЖНО!)
**ПРОБЛЕМА**: В текущих версиях **НЕТ** API вида `SentenceTransformer(...).quantize('int8')`

**ПРАВИЛЬНЫЕ ПОДХОДЫ**:
- **Квантование эмбеддингов**: `encode(..., precision='binary'|'int8')` 
- **Модельное ускорение**: экспорт в ONNX/OpenVINO через `export_dynamic_quantized_onnx_model`, `export_static_quantized_openvino_model`
- **Альтернатива**: FastEmbed с готовыми квантованными ONNX моделями

#### 1.2 FastEmbed как CPU-бэкэнд по умолчанию
**РЕКОМЕНДАЦИЯ**: Использовать `FastEmbed` от Qdrant для минимальных зависимостей:
- ONNX Runtime внутри, квантованные веса, CPU-first
- Готовые модели: `Qdrant/bge-small-en-v1.5-onnx-Q` (384d)
- Интеграция: `qdrant-client[fastembed]`

**ДВА ПРОФИЛЯ СБОРКИ**:
- *Лёгкая (рекомендуется)*: `fastembed` + `qdrant-client[fastembed]`, без `torch`
- *Расширенная*: `sentence-transformers` + опциональный ONNX/OpenVINO экспорт

#### 1.3 Qdrant API актуализация
**ВАЖНО**: Использовать актуальные типы клиента:
- `HnswConfigDiff`, `OptimizersConfigDiff` в `create_collection`
- `VectorParams`: `datatype='float16'`, `on_disk=True` для CPU-friendly профиля
- `SearchParams`: `hnsw_ef`, `exact`, `quantization`, `indexed_only`
- **Квантование**: `ScalarQuantization`, `ProductQuantization`, `BinaryQuantization`

#### 1.4 Гибридный поиск (dense + sparse)  
**ВОЗМОЖНОСТЬ**: Qdrant нативно поддерживает sparse-вектора (BM25/SPLADE)
- Создание коллекции с `vectors_config` (dense) + `sparse_vectors_config` (BM25/SPLADE)
- Гибридный фьюжн результатов повышает качество по коду

#### 1.5 CPU Rerankers
**РЕКОМЕНДАЦИИ** для быстрых CPU-кейсов:
- `jinaai/jina-reranker-v1-tiny-en` (≈33M параметров, быстрый, контекст до 8k)
- `BAAI/bge-reranker-v2-m3` (мульти-язычный, лёгкий)
- **Двухступенчатый поиск**: ANN@Qdrant → top-K rerank (CPU)

### 🏗️ АРХИТЕКТУРНЫЕ УЛУЧШЕНИЯ

#### Модельный стек (реалистично для 2025)
- **По умолчанию**: `BAAI/bge-small-en-v1.5` (384d) через FastEmbed
- **Альтернативы**: `intfloat/multilingual-e5-small` для i18n
- **Зависимости**: torch актуальная ветка **2.4+** (НЕ 2.0.0!), faiss-cpu **1.7.4+**

#### CPU-профиль коллекции Qdrant
```python
VectorParams(
    size=384, distance=COSINE, on_disk=True, datatype='float16',
    hnsw_config=HnswConfigDiff(m=16..32, ef_construct=64..128),
    quantization_config=ScalarQuantization|ProductQuantization|BinaryQuantization
)
```

#### Адаптивные батчи и управление RAM
- **Калибровочная стратегия**: стартуем с минимального → увеличиваем до предельного по p95-latency
- **Контроль планировщика**: `torch.set_num_threads(k)` для SBERT; FastEmbed использует ONNX Runtime
- **Graceful degradation**: на OOM → поэлементная обработка с принудительным GC

#### Обсервабилити и SLO
**Метрики уровня сервиса**:
- p50/p95 encode latency, p95 Qdrant search, ingest throughput
- cache hit-rate, recall@k/nDCG@10 (оффлайн-замер)
- доля обновлённых чанков при инкременте

**Трассировка**: OpenTelemetry вокруг encode/search/rerank
**Алёрты**: деградация recall, рост p95 поиска, рост indexed_only miss

#### Безопасность контента
- Перед инжестом: секрет-скан (gitleaks/trufflehog), фильтр PII/лицензий  
- В payload: права доступа/тенанты, фильтрация точек по ACL

---

## 1) Обновление стека зависимостей

Рекомендованные версии и новые пакеты. Обновить requirements.txt до (CPU-first, без GPU):

```txt
# Core LLM / Embeddings
openai>=1.99.6
sentence-transformers~=5.1.0  # поддерживает precision='int8'
torch>=2.4.0  # CPU-only; настройки потоков управляются конфигом
tiktoken>=0.8.0

# Векторные БД и альтернативы
qdrant-client>=1.15.1
faiss-cpu>=1.7.4  # опционально как локальная альтернатива

# Научный стек и утилиты
numpy>=1.24.0
psutil>=5.9.5            # мониторинг RAM/CPU
cachetools>=5.3.0        # LRU/TTL кэш для QueryEngine

# API / Web UI / Frameworks
fastapi>=0.104.0         # опционально, если нужен REST API
uvicorn>=0.24.0          # опционально для FastAPI
streamlit>=1.46.0
click>=8.1.8
rich>=14.0.0
python-dotenv>=1.0.0
chardet>=5.2.0

# Тестирование
pytest>=8.3.4
pytest-asyncio>=1.1.0
hypothesis>=6.124.9
```

Замечания:
- Если выбран fastembed + ONNX Runtime, добавить:
  - fastembed>=0.3.0
  - onnxruntime>=1.16.0
- Для CPU производительности: управлять числом потоков через (OMP_NUM_THREADS, MKL_NUM_THREADS) и torch.set_num_threads().

---

## 2) Расширение конфигурации проекта (config.py)

Добавить новые dataclass’ы и секции. Существующие классы сохраняются (OpenAIConfig, AnalysisConfig, FileScannerConfig и т.д.). Новые:

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"  # "sentence-transformers" | "fastembed"
    model_name: str = "intfloat/e5-small-v2"
    precision: str = "int8"                  # "int8" | "float32"
    truncate_dim: int = 384                  # 256–384; Matryoshka/truncate
    batch_size_min: int = 8
    batch_size_max: int = 128
    normalize_embeddings: bool = True
    device: str = "cpu"                      # CPU-first
    warmup_enabled: bool = True
    num_workers: int = 4                     # воркеры подготовки текстов (не threads модели)

@dataclass
class VectorStoreConfig:
    host: str = "localhost"
    port: int = 6333
    prefer_grpc: bool = True
    collection_name: str = "code_chunks"
    vector_size: int = 384
    distance: str = "cosine"
    # HNSW
    hnsw_m: int = 24
    hnsw_ef_construct: int = 128
    search_hnsw_ef: int = 256
    # Квантование
    quantization_type: str = "SQ"            # "SQ" | "PQ"
    enable_quantization: bool = True
    # Репликация и консистентность
    replication_factor: int = 2
    write_consistency_factor: int = 1
    # Хранилище
    mmap: bool = True

@dataclass
class QueryEngineConfig:
    max_results: int = 10
    rrf_enabled: bool = True
    use_hybrid: bool = True                  # dense + sparse
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    cache_ttl_seconds: int = 300             # TTL для горячих запросов
    cache_max_entries: int = 1000
    # Параметры параллелизма
    concurrent_users_target: int = 20
    search_workers: int = 4
    embed_workers: int = 4

@dataclass
class ParallelismConfig:
    torch_num_threads: int = 4
    omp_num_threads: int = 4
    mkl_num_threads: int = 4
```

В `Config` добавить новые поля (embedding, vector_store, query_engine, parallelism). Валидация: корректность значений, проверка доступности промпта, приведение типов.

---

## 3) CPU‑оптимизированный эмбеддер (embedder.py)

Цель: быстрые, нормализованные, компактные эмбеддинги кода (CPU-first).

Интерфейс:
```python
# rag/embedder.py
from typing import List
import numpy as np

class CPUEmbedder:
    def __init__(self, cfg: EmbeddingConfig, par: ParallelismConfig):
        # Lazy init модели; установка потоков torch/OMP/MKL согласно cfg/par
        ...

    def warmup(self) -> None:
        # Прогрев модели: один dummy-encode для JIT/инициализации
        ...

    def calculate_batch_size(self, queue_len: int) -> int:
        # Учитывает psutil.virtual_memory().available и размер очереди
        # Возвращает значение в [batch_size_min, batch_size_max]
        ...

    def embed_texts(self, texts: List[str], deadline_ms: int = 1500) -> np.ndarray:
        # Бэтчевое кодирование с контролем времени отклика
        # precision='int8', normalize_embeddings=True (для ST v5.1.0)
        ...
```

Провайдеры:
- sentence-transformers v5.1.0 (основной, precision='int8', normalize_embeddings=True).
- fastembed (опционально; ONNX Runtime, quantized веса).

---

## 4) Векторное хранилище (vector_store.py, Qdrant)

Задачи:
- Инкапсулировать клиента Qdrant и операции коллекции.
- Создание/миграция коллекции: m=24, ef_construct=128, distance=cosine, mmap=true.
- Включать Scalar Quantization (SQ) или Product Quantization (PQ) для больших коллекций.
- Параметры репликации и консистентности: replication_factor=2, write_consistency_factor=1–2.

Интерфейс:
```python
# rag/vector_store.py
from typing import List, Dict, Optional
import numpy as np

class QdrantVectorStore:
    def __init__(self, vcfg: VectorStoreConfig):
        ...

    async def initialize_collection(self, recreate: bool = False) -> None:
        ...

    async def index_documents(self, points: List[Dict]) -> int:
        """
        points[i] ~ {
          'id': str,
          'vector': np.ndarray (shape=[vector_size]),
          'payload': {
            'file': 'path/to/file',
            'chunk_id': '...',
            'hash': 'sha256...',
            'lang': 'python',
            'group': 'auth/db/...', # опционально
            'line_start': 10,
            'line_end': 50,
            'ts': 'iso'
          }
        }
        Загружает батчами по 512–1024.
        """

    async def update_document(self, pid: str, vector: np.ndarray, payload: Dict) -> bool:
        ...

    async def delete_document(self, pid: str) -> bool:
        ...

    async def search(self,
                     query_vector: np.ndarray,
                     top_k: int,
                     filters: Optional[Dict] = None,
                     use_hybrid: bool = False) -> List[Dict]:
        """
        При use_hybrid=True — гибридный поиск dense+sparse (BM25/SPLADE) и фьюжн (RRF).
        Возвращает список результатов с метаданными и score.
        """
```

Ресурсы:
- Хранение на диске с mmap=true.
- При множестве коллекций — subgroup-oriented configuration/кэширование активных сегментов (см. qdrant.tech).

---

## 5) Индексация и переиндексация

Сервис индексации (скрипт/команда):
- Сканирует репозиторий (file_scanner.py).
- Разбивает файлы на чанки (code_chunker.py).
- Генерирует эмбеддинги для каждого чанка (CPUEmbedder).
- Заливает в Qdrant (QdrantVectorStore.index_documents).

Инкрементальность:
- Сверка SHA256 (utils.compute_file_hash) с .repo_sum/index.json.
- Обновление/удаление записей в Qdrant при изменениях/удалениях файлов.

Планирование:
- CI/CD после коммита/мержа.
- cron (например, ежедневно).
- По требованию пользователя (CLI команда).

Метрики качества индекса:
- recall@k, MRR@k на эталонных наборах запросов.

---

## 6) Поисковый движок (query_engine.py)

Задачи:
- Метод `search(query: str, max_results: int)`:
  - Эмбеддинг запроса.
  - Поиск в Qdrant с опциональным гибридным режимом (dense + sparse; BM25/SPLADE).
  - RRF (Reciprocal Rank Fusion) для фьюжна результатов.
  - MMR-переранжирование (или diverse beam search) для разнообразия и борьбы с дубликатами.
- LRU-кэш с TTL для горячих запросов (cachetools).
- Параллельная обработка запросов: asyncio + пул (4–8 workers), целевая нагрузка ~20 пользователей.

Интерфейс (набросок):
```python
# rag/query_engine.py
from typing import List, Dict, Optional

class CPUQueryEngine:
    def __init__(self, embedder: CPUEmbedder, store: QdrantVectorStore, qcfg: QueryEngineConfig):
        ...

    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Возвращает список результатов с payload (файл, строки) + score.
        Кэширует после фьюжна/переранжирования.
        """
```

---

## 7) Интеграция RAG в текущий поток (промпты, OpenAIManager, CLI, Web UI)

Промпты:
- Обновить `prompts/code_analysis_prompt.md`: 
  - «Сначала изучите retrieved контекст (список фрагментов), затем проанализируйте код файла…».
  - Следовать лимиту токенов (8–12k для GPT‑4o), ограничивать число retrieved chunks.

OpenAIManager:
- Расширить `GPTAnalysisRequest` полем `context_chunks: List[str]`.
- Перед вызовом OpenAI:
  - Объединить retrieved фрагменты с кодом файла (сортировать по релевантности, обрезать менее значимые части).
  - Формировать prompt из контекста + кода файла.

CLI:
- Новые команды:
  - `python main.py index /path/to/repo` — индексация в Qdrant.
  - `python main.py search "query" -k 10` — поиск по векторному индексу.
  - `python main.py analyze-with-rag /path/to/repo -o ./docs` — анализ файлов с использованием retrieved контекста.
- Существующие команды сохраняются (analyze, stats, clear-cache, token-stats).

Web UI (Streamlit):
- Новая вкладка «Поиск»:
  - Поле запроса, параметры top_k/use_hybrid.
  - Вывод фрагментов со ссылками на исходники.
  - Метрики ответа (время, количество найденных).
- Аналитический режим: анализ файла + просмотр retrieved контекста.

---

## 8) Тестирование и валидация

Unit-тесты (pytest, pytest-asyncio):
- `tests/rag/test_embedder.py` — батчевый encode, warmup, adaptive batch size, precision/normalize.
- `tests/rag/test_vector_store.py` — инициалицация коллекции, index/update/delete/search, quantization флаги.
- `tests/rag/test_query_engine.py` — гибридный поиск, RRF, MMR, LRU/TTL кэш.
- `tests/rag/test_rag_integration.py` — e2e: индексация → поиск → анализ с контекстом.
- Нагрузочные: имитация 20 пользователей (5–10 запросов/мин), оценка латентности/ошибок/ресурсов.

Метрики качества:
- MRR@k, Recall@k — на размеченных наборах запросов.
- Эксперименты с hnsw_ef и m для баланса точность/скорость.

---

## 9) Документация

README.md:
- Как поднять Qdrant (локально/Docker).
- Как выполнить индексацию/поиск.
- Как использовать «analyze-with-rag» и Web UI «Поиск».
- Производительные настройки для ЦОД: потоки, batch size, quantization, hnsw_ef, mmap.
- Политика секретов и включение санитайзинга (analysis.sanitize_enabled).

Корпоративная инструкция:
- Обновление индекса после коммитов (CI/CD).
- Требования к API-ключам.
- Правила обхода и валидации загружаемых артефактов.

---

## 10) Развёртывание в дата‑центре

Docker-compose (план):
- Контейнер приложения (Python + зависимости).
- Контейнер Qdrant (или кластер).
- Контейнер FastAPI (если нужен REST API) и Streamlit (UI).
- Том для данных Qdrant и снапшотов.

CI/CD:
- Автотесты, сборка, линтеры (flake8/black), mypy (опционально).
- Деплой на staging → production.

Мониторинг:
- Prometheus/Grafana: метрики Qdrant и приложения (CPU, RAM, latency, throughput).
- Алерты на превышение латентности/ошибки.

Безопасность:
- TLS между сервисами.
- Firewall/VPN, контроль доступа.
- Ротация API-ключей.
- Санитайзинг секретов перед отправкой в LLM (analysis.sanitize_enabled=true при необходимости).

---

## Контрольные точки (milestones)

- M1: Базовая индексация + CPUEmbedder (int8, normalize), Qdrant коллекция, простой dense-поиск, CLI `index`/`search`. 
- M2: Гибридный поиск (BM25/SPLADE) + RRF, MMR, TTL-кэш, параллелизм на 20 пользователей.
- M3: Интеграция в OpenAIManager/промпты (context_chunks) и `analyze-with-rag`, вкладка «Поиск» в Web UI.
- M4: Нагрузочные тесты, мониторинг/алерты, документация для DC, CI/CD.

---

## Примечания по совместимости с существующим кодом

- Ничего не ломать в текущем пайплайне analyze/stats/clear-cache/token-stats.
- Новые модули предлагаются в пакете `rag/`: 
  ```
  rag/
  ├── __init__.py
  ├── embedder.py
  ├── vector_store.py
  ├── query_engine.py
  └── (при необходимости: hybrid_search.py, code_context.py)
  ```
- Расширение `utils.GPTAnalysisRequest` полем `context_chunks`.
- Обновление `config.Config`: добавить `embedding`, `vector_store`, `query_engine`, `parallelism`.

---

Этот документ — руководство к действию для реализации RAG‑ядра с CPU‑оптимизацией и интеграцией в существующий анализатор. Все описанные параметры и интерфейсы являются целевыми; в процессе реализации допустимы уточнения, но принципиальные решения (CPU-first, Qdrant, гибридный поиск, MMR, TTL‑кэш, параллелизм под 20 пользователей) считаются финализированными.
