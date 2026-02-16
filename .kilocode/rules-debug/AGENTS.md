# Debug Mode — Правила отладки для repo_sum

## 🚨 КРИТИЧЕСКИЕ ИНСТРУМЕНТЫ

### RAG тесты — ТОЛЬКО через раннер
```bash
# ❌ Может пропустить проблемы
pytest tests/rag/

# ✅ ПРАВИЛЬНО — с таймаутом 300 сек
python tests/rag/run_rag_tests.py smoke   # быстрая проверка
python tests/rag/run_rag_tests.py all     # полный прогон
```

### Offline тестирование
```bash
# Обязательные переменные для offline режима
OFFLINE_MODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest
```

---

## 🔍 ДИАГНОСТИКА КОМПОНЕНТОВ

### QdrantVectorStore — health check
```python
# ❌ НЕ использовать — нестабильный метод
client.get_cluster_info()

# ✅ ПРАВИЛЬНО — стабильный метод
client.get_collections()
```

### CPUEmbedder — диагностика
```python
# Проверка offline режима
assert embedder.provider_name == "offline"  # при OFFLINE_MODE=1

# Проверка эмбеддингов
vectors = embedder.embed_texts(["test"])
assert vectors.shape[1] == 384  # BAAI/bge-small-en-v1.5
```

### Ошибки — централизованная обработка
```python
# Все RAG ошибки в rag/exceptions.py
from rag.exceptions import (
    RAGError,           # Базовое исключение
    EmbeddingError,     # Ошибки эмбеддинга
    VectorStoreError,   # Ошибки Qdrant
    QueryError          # Ошибки поиска
)
```

---

## 🧪 ТЕСТОВЫЕ СЦЕНАРИИ

### Запуск конкретного теста
```bash
# Один тест
pytest tests/test_config.py::test_specific -v

# С выводом print
pytest tests/test_config.py::test_specific -v -s

# С отладкой
pytest tests/test_config.py::test_specific -v --pdb
```

### Категории тестов
```bash
# Unit (без сети)
pytest -m "not integration and not functional" --disable-socket -v

# Integration (с сетью)
pytest -m "integration" -v

# Functional (CLI)
pytest -m "functional" -v
```

---

## 📊 ЛОГИРОВАНИЕ

### Структура логов
```
rag/
├── embedder.py      → "CPUEmbedder" logger
├── vector_store.py  → "QdrantVectorStore" logger
├── query_engine.py  → "CPUQueryEngine" logger
└── search_service.py → "SearchService" logger
```

### Уровни логирования
```python
import logging
logging.getLogger("CPUEmbedder").setLevel(logging.DEBUG)
logging.getLogger("QdrantVectorStore").setLevel(logging.DEBUG)
```

---

## 🚫 ЗАПРЕЩЁННО

1. Использовать `get_cluster_info()` для диагностики Qdrant
2. Запускать RAG тесты без `run_rag_tests.py`
3. Игнорировать `--disable-socket` для unit тестов
4. Хардкодить адреса Qdrant при отладке
