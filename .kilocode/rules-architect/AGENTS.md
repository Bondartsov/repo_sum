# Architect Mode — Архитектурные ограничения для repo_sum

## 🏗️ АРХИТЕКТУРНЫЕ ПРИНЦИПЫ

### Modular Monolith — НЕ microservices
```
repo_sum/
├── Core System/        # Анализ кода и документация
├── RAG System/         # Семантический поиск
├── Parsers System/     # Языковые парсеры
├── UI System/          # CLI + Web интерфейсы
└── Testing System/     # Комплексное тестирование
```

### CPU-First Architecture
- FastEmbed с ONNX Runtime — GPU НЕ требуется
- HNSW параметры: m=24, ef_construct=128
- Управление потоками: OMP_NUM_THREADS, MKL_NUM_THREADS

---

## 🚨 КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ

### Зависимости — проверка через скрипт
```bash
# ❌ НЕВЕРНО
pip check

# ✅ ПРАВИЛЬНО
python scripts/verify_requirements.py
```
**Причина**: Кастомный MODULE_TO_PKG маппинг.

### Qdrant — environment variables
```python
# ❌ НЕ хардкодить
host = "localhost"

# ✅ ПРАВИЛЬНО
host = os.getenv("QDRANT_HOST", "localhost")
port = int(os.getenv("QDRANT_PORT", "6333"))
```

### Health Check — стабильный метод
```python
# ❌ НЕ использовать
client.get_cluster_info()

# ✅ ПРАВИЛЬНО
client.get_collections()
```

---

## 📊 RAG АРХИТЕКТУРА

### Компоненты
```
rag/
├── embedder.py         # CPUEmbedder — ЕДИНСТВЕННЫЙ интерфейс эмбеддинга
├── vector_store.py     # QdrantVectorStore — get_collections() для health
├── query_engine.py     # CPUQueryEngine — RRF + MMR
├── search_service.py   # SearchService — высокоуровневый поиск
├── indexer_service.py  # IndexerService — оркестрация индексации
└── sparse_encoder.py   # SparseEncoder — BM25/SPLADE (M2)
```

### Гибридный поиск (M2)
```
[Query] → ├─[Dense Embedder] → Dense Search
          └─[Sparse Encoder] → Sparse Search (BM25/SPLADE)
                                     ↓
                               [RRF Fusion]
                                     ↓
                               [MMR Re-ranking]
```

---

## 🧪 ТЕСТИРОВАНИЕ

### Категоризация (pytest.ini)
| Маркер | Описание | Сеть |
|--------|----------|------|
| Без маркера | Unit тесты | ❌ |
| `@pytest.mark.integration` | OpenAI, Qdrant | ✅ |
| `@pytest.mark.functional` | CLI/subprocess | ✅ |

### Offline режим
```bash
OFFLINE_MODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest
```

---

## 📁 ИСТОЧНИКИ ИСТИНЫ

| Что | Где |
|-----|-----|
| Текущий статус | [`.clinerules/activeContext.md`](.clinerules/activeContext.md) |
| Технический стек | [`.clinerules/techContext.md`](.clinerules/techContext.md) |
| Архитектурные паттерны | [`.clinerules/systemPatterns.md`](.clinerules/systemPatterns.md) |
| Roadmap | [`ROADMAP.md`](ROADMAP.md) |

---

## 🚫 ЗАПРЕЩЁННО

1. Microservices — только Modular Monolith
2. GPU зависимости — только CPU-first
3. `pip check` — только `scripts/verify_requirements.py`
4. Хардкод адресов Qdrant
5. `get_cluster_info()` — только `get_collections()`
6. Тесты с сетью без маркера `@pytest.mark.integration`
