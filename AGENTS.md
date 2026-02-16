# AGENTS.md

**НЕОЧЕВИДНЫЕ правила проекта repo_sum** — только то, что нельзя угадать из стандартных практик.

---

## 🚨 КРИТИЧЕСКИЕ GOTCHAS

### 1. Зависимости — НЕ использовать `pip check`
```bash
# ❌ НЕВЕРНО
pip check

# ✅ ПРАВИЛЬНО — кастомный скрипт с маппингом модулей
python scripts/verify_requirements.py
```
**Причина**: Скрипт содержит собственный `MODULE_TO_PKG` маппинг (например, `dotenv` → `python-dotenv`).

### 2. RAG тесты — ТОЛЬКО через кастомный раннер
```bash
# ❌ Может пропустить проблемы
pytest tests/rag/

# ✅ ПРАВИЛЬНО — с таймаутом 300 сек и категоризацией
python tests/rag/run_rag_tests.py smoke   # быстрая проверка
python tests/rag/run_rag_tests.py all     # полный прогон
```

### 3. Unit тесты — ОБЯЗАТЕЛЬНО `--disable-socket`
```bash
# ✅ Unit тесты (без сети)
pytest -m "not integration and not functional and not e2e" --disable-socket --allow-unix-socket -v

# ✅ Integration тесты (с сетью)
pytest -m "integration" -v

# ✅ Functional тесты (subprocess/CLI)
pytest -m "functional" -v
```
**Причина**: Без `--disable-socket` unit тесты могут случайно использовать сеть.

### 4. Offline режим — переменные окружения
```bash
# Для offline тестирования ОБЯЗАТЕЛЬНЫ:
OFFLINE_MODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest
```
**Проверка**: `tests/test_offline_no_network.py` — эталонный тест.

---

## 🔧 НЕОЧЕВИДНЫЕ ПАТТЕРНЫ КОДА

### Health Check в QdrantVectorStore
```python
# ❌ НЕ использовать — нестабильный метод
client.get_cluster_info()

# ✅ ПРАВИЛЬНО — стабильный метод
client.get_collections()
```
**Файл**: [`rag/vector_store.py`](rag/vector_store.py)

### Mock объекты — правило `.to()`
```python
# ❌ НЕВЕРНО — вернёт None
def to(self, device):
    return None

# ✅ ПРАВИЛЬНО — должен вернуть self
def to(self, device):
    return self
```
**Файл**: [`.clinerules/MOCKS_RULES.md`](.clinerules/MOCKS_RULES.md)

### CPUEmbedder в offline режиме
```python
# При OFFLINE_MODE=1:
embedder.provider_name == "offline"  # НЕ "fastembed"
vectors == np.zeros(...)             # Нулевые эмбеддинги
```

### Environment variables для Qdrant
```python
# ❌ НЕ хардкодить
host = "localhost"

# ✅ ПРАВИЛЬНО
host = os.getenv("QDRANT_HOST", "localhost")
port = int(os.getenv("QDRANT_PORT", "6333"))
```

---

## 📋 КАТЕГОРИЗАЦИЯ ТЕСТОВ (pytest.ini)

| Маркер | Описание | Сеть |
|--------|----------|------|
| Без маркера | Unit тесты | ❌ |
| `@pytest.mark.integration` | OpenAI, Qdrant, filesystem | ✅ |
| `@pytest.mark.functional` | CLI/subprocess тесты | ✅ |
| `@pytest.mark.e2e` | End-to-end сценарии | ✅ |
| `@pytest.mark.property` | Hypothesis тесты | ❌ |

**Результат**: 149 passed, 3 skipped при корректной категоризации.

---

## 🗂️ ИСТОЧНИКИ ИСТИНЫ

| Что | Где |
|-----|-----|
| Текущий статус | [`.clinerules/activeContext.md`](.clinerules/activeContext.md) |
| Технический стек | [`.clinerules/techContext.md`](.clinerules/techContext.md) |
| Mock правила | [`.clinerules/MOCKS_RULES.md`](.clinerules/MOCKS_RULES.md) |
| Стратегия тестирования | [`tests/rag/TESTING_STRATEGY.md`](tests/rag/TESTING_STRATEGY.md) |
| Roadmap | [`ROADMAP.md`](ROADMAP.md) |

---

## ⚡ БЫСТРЫЕ КОМАНДЫ

```bash
# Запуск одного теста
pytest tests/test_config.py::test_specific -v

# Проверка зависимостей
python scripts/verify_requirements.py

# RAG smoke тесты
python tests/rag/run_rag_tests.py smoke

# Offline unit тесты
pytest -m "not integration and not functional" --disable-socket -v

# Веб-интерфейс
python run_web.py
```

---

## 🚫 ЗАПРЕЩЁННЫЕ ДЕЙСТВИЯ

1. **НЕ** использовать `pip check` — только `scripts/verify_requirements.py`
2. **НЕ** хардкодить `localhost:6333` — использовать env variables
3. **НЕ** использовать `get_cluster_info()` — только `get_collections()`
4. **НЕ** писать тесты с сетью без маркера `@pytest.mark.integration`
5. **НЕ** возвращать `None` из mock метода `.to()`
