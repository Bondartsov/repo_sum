# Ask Mode — Контекст документации для repo_sum

## 🗂️ ИСТОЧНИКИ ИСТИНЫ

| Что | Где |
|-----|-----|
| Текущий статус | [`.clinerules/activeContext.md`](.clinerules/activeContext.md) |
| Технический стек | [`.clinerules/techContext.md`](.clinerules/techContext.md) |
| Продуктовый контекст | [`.clinerules/productContext.md`](.clinerules/productContext.md) |
| Архитектурные паттерны | [`.clinerules/systemPatterns.md`](.clinerules/systemPatterns.md) |
| Mock правила | [`.clinerules/MOCKS_RULES.md`](.clinerules/MOCKS_RULES.md) |
| Стратегия тестирования | [`tests/rag/TESTING_STRATEGY.md`](tests/rag/TESTING_STRATEGY.md) |
| Roadmap | [`ROADMAP.md`](ROADMAP.md) |

---

## 📚 СТРУКТУРА ДОКУМЕНТАЦИИ

### Memory Bank (.clinerules/)
```
.clinerules/
├── activeContext.md     # Текущий статус, активные задачи
├── techContext.md       # Технологический стек, компоненты
├── productContext.md    # Продуктовый контекст
├── systemPatterns.md    # Архитектурные паттерны
├── progress.md          # История развития
├── projectbrief.md      # Суть проекта
└── MOCKS_RULES.md       # Правила mock объектов
```

### Техническая документация
```
tests/rag/TESTING_STRATEGY.md  # Стратегия тестирования (741 строка)
ROADMAP.md                     # Roadmap проекта
rules/                         # Дополнительные правила
```

---

## 🔍 ГДЕ ИСКАТЬ ИНФОРМАЦИЮ

### По компонентам
| Компонент | Документация |
|-----------|--------------|
| CPUEmbedder | `rag/embedder.py` docstrings |
| QdrantVectorStore | `rag/vector_store.py` docstrings |
| CPUQueryEngine | `rag/query_engine.py` docstrings |
| SearchService | `rag/search_service.py` docstrings |
| SparseEncoder | `rag/sparse_encoder.py` docstrings |

### По темам
| Тема | Источник |
|------|----------|
| Тестирование | `tests/rag/TESTING_STRATEGY.md` |
| Конфигурация | `config.py`, `settings.json` |
| CLI команды | `main.py` |
| Web UI | `web_ui.py`, `run_web.py` |

---

## ⚡ БЫСТРЫЕ КОМАНДЫ

```bash
# Веб-интерфейс
python run_web.py

# Статус RAG
python main.py rag status

# Поиск по коду
python main.py rag search "query"

# Проверка зависимостей
python scripts/verify_requirements.py
```

---

## 🚨 КРИТИЧЕСКИЕ GOTCHAS

### При ответах учитывать:
1. **Зависимости**: НЕ `pip check`, а `scripts/verify_requirements.py`
2. **Qdrant**: НЕ `get_cluster_info()`, а `get_collections()`
3. **Тесты**: Unit с `--disable-socket`, integration с маркером
4. **Offline**: `OFFLINE_MODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
