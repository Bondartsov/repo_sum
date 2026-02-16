# Code Mode — Правила кодирования для repo_sum

## 🚨 КРИТИЧЕСКИЕ ПРАВИЛА

### Зависимости
```bash
# ❌ НЕВЕРНО
pip check

# ✅ ПРАВИЛЬНО
python scripts/verify_requirements.py
```

### Эмбеддинги — ТОЛЬКО через CPUEmbedder
```python
# ❌ НЕВЕРНО — прямые вызовы
from fastembed import TextEmbedding
embedder = TextEmbedding(model_name="...")

# ✅ ПРАВИЛЬНО — через абстракцию
from rag.embedder import CPUEmbedder
embedder = CPUEmbedder(config)
```

### Qdrant — environment variables
```python
# ❌ НЕ хардкодить
client = QdrantClient(host="localhost", port=6333)

# ✅ ПРАВИЛЬНО
import os
host = os.getenv("QDRANT_HOST", "localhost")
port = int(os.getenv("QDRANT_PORT", "6333"))
client = QdrantClient(host=host, port=port)
```

---

## 🧪 ТЕСТИРОВАНИЕ

### Unit тесты — offline обязательно
```bash
# ✅ ПРАВИЛЬНО
pytest -m "not integration and not functional" --disable-socket --allow-unix-socket -v
```

### RAG тесты — только через раннер
```bash
python tests/rag/run_rag_tests.py smoke
```

### Mock объекты — правило `.to()`
```python
# ❌ НЕВЕРНО
def to(self, device):
    return None

# ✅ ПРАВИЛЬНО
def to(self, device):
    return self
```

---

## 📁 СТРУКТУРА КОМПОНЕНТОВ

### RAG система
```
rag/
├── embedder.py         # CPUEmbedder — ЕДИНСТВЕННЫЙ способ эмбеддинга
├── vector_store.py     # QdrantVectorStore — использовать get_collections()
├── query_engine.py     # CPUQueryEngine — RRF + MMR
├── search_service.py   # SearchService — высокоуровневый поиск
└── sparse_encoder.py   # SparseEncoder — BM25/SPLADE
```

### Offline режим
```python
# При OFFLINE_MODE=1:
# CPUEmbedder.provider_name == "offline"
# Эмбеддинги = np.zeros(...)
```

---

## 🚫 ЗАПРЕЩЁННО

1. Прямые импорты FastEmbed/SentenceTransformers в обход CPUEmbedder
2. Хардкод `localhost:6333` для Qdrant
3. Использование `get_cluster_info()` — только `get_collections()`
4. Тесты с сетью без маркера `@pytest.mark.integration`
5. `pip check` вместо `scripts/verify_requirements.py`
