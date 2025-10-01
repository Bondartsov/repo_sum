# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Обзор проекта

**repo_sum** - это RAG-as-a-Service система для автоматического анализа кода с использованием AI. Проект использует революционную архитектуру, где тяжёлые вычисления (Jina v3 embeddings, 570M параметров) выполняются на удалённой VM, а локально работают только HTTP клиенты.

### Ключевые особенности
- **RAG-as-a-Service архитектура** с VM-based вычислениями (10.61.11.54:8000)
- **Jina v3 embeddings** (1024d vectors) для семантического поиска
- **Гибридный поиск** (Dense + Sparse vectors с RRF fusion)
- **CPU-first оптимизация** - не требует GPU
- **Поддержка 5+ языков** (Python, JavaScript, TypeScript, C++, C#)

## Основные команды

### Запуск приложения
```bash
# Веб-интерфейс (рекомендуется)
python run_web.py

# CLI команды
python main.py analyze /path/to/repo
python main.py rag search "query"
python main.py rag status
```

### VM Service Management
```bash
# Полная автоматизация VM развертывания
python vm_start.py start

# Проверка статуса
python vm_start.py status

# Остановка сервисов
python vm_start.py stop

# Health check VM сервиса
curl http://10.61.11.54:8000/health
```

### Тестирование

```bash
# Unit тесты (изолированные, offline)
pytest -m "not integration and not functional and not e2e" --disable-socket -v

# Integration тесты (требуют внешних сервисов)
pytest -m "integration" -v

# Functional тесты (subprocess/CLI)
pytest -m "functional" -v

# Все RAG тесты
pytest tests/rag/ -v
python tests/rag/run_rag_tests.py all

# Property-based тесты
pytest tests/ -k "property"

# С покрытием
pytest --cov=. tests/ --cov-report=html
```

### Миграция и валидация
```bash
# Миграция на Jina v3
python scripts/migrate_to_jina_v3.py

# Валидация VM окружения
python scripts/validate_vm_env.py

# Проверка зависимостей
python scripts/verify_requirements.py
```

## Архитектура системы

### Компонентная структура
```
repo_sum/
├── Core System/          # Анализ кода и документация
│   ├── main.py          # Точка входа и CLI
│   ├── file_scanner.py  # Сканирование файлов
│   ├── code_chunker.py  # Разбиение на чанки
│   └── openai_integration.py  # OpenAI интеграция
│
├── RAG System/          # Семантический поиск (VM-based)
│   ├── embedder.py      # Гибридный эмбеддер
│   ├── vector_store.py  # Qdrant интеграция
│   ├── query_engine.py  # Поисковый движок
│   ├── search_service.py     # Высокоуровневый поиск
│   ├── indexer_service.py    # Индексация
│   ├── remote_embedder.py    # HTTP клиент для VM
│   └── remote_vector_store.py # HTTP клиент для VM
│
├── Parsers System/      # Языковые парсеры
│   ├── base_parser.py   # Базовый интерфейс
│   ├── python_parser.py # Python AST парсинг
│   ├── javascript_parser.py
│   ├── typescript_parser.py
│   ├── cpp_parser.py
│   └── csharp_parser.py
│
├── UI System/           # Интерфейсы
│   ├── web_ui.py        # Streamlit веб-интерфейс
│   └── run_web.py       # Запуск веб-приложения
│
└── Testing System/      # 5872+ строк тестов
    ├── tests/rag/       # RAG тесты
    └── tests/fixtures/  # Тестовые данные
```

### RAG-as-a-Service Pipeline
```
Локальная машина ←──→ HTTP REST API ←──→ VM (10.61.11.54:8000)
├─ repo_sum CLI                          ├─ FastAPI :8000
├─ Web UI                                ├─ Jina v3 (570M)
├─ OpenAI анализ                         ├─ Qdrant :6333
└─ HTTP клиенты                          └─ Гибридный поиск
```

### Ключевые потоки данных

1. **Анализ кода**: Scan → Parse → Chunk → Embed (VM) → OpenAI Analysis → Generate Docs
2. **Индексация**: Scan → Parse → Chunk → Embed (VM) → Index to Qdrant (VM)
3. **Поиск**: Query → Embed (VM) → Dense+Sparse Search (VM) → RRF Fusion → MMR Rerank → Results

## Конфигурация

### Основные файлы конфигурации
- `settings.json` - основная конфигурация системы
- `.env` - переменные окружения (API ключи, VM credentials)
- `config.py` - dataclass конфигурационные модели

### Критические переменные окружения
```bash
# OpenAI API
OPENAI_API_KEY=sk-your-key

# VM RAG Service
VM_HOST=10.61.11.54
VM_USER=user
VM_PASSWORD=your_password

# Qdrant (на VM)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Jina v3 Configuration
EMB_MODEL_ID=jinaai/jina-embeddings-v3
EMB_DIM=1024
EMB_TASK_QUERY=retrieval.query
EMB_TASK_PASSAGE=retrieval.passage
EMB_TRUST_REMOTE_CODE=true
```

## Правила разработки

### Языковые требования
- **ВСЕ** комментарии и документация **ТОЛЬКО на русском языке**
- **ВСЕ** docstrings и технические комментарии на русском
- **НИКОГДА** не использовать транслитерацию
**Команды в теримнале запускать под powershell, мы работаем в среде windows**


### Стандарты кода
- **PEP8** строгое соблюдение
- **SOLID принципы** обязательны
- **DRY (Don't Repeat Yourself)** - избегать дублирования
- **Строгая типизация** (type hints везде)
- **Fail-fast валидация** - ошибки выявляются рано

### Архитектурные принципы
1. **CPU-First Architecture** - оптимизация для CPU, GPU не требуется
2. **RAG-as-a-Service** - тяжёлые вычисления на VM, локально HTTP клиенты
3. **Configuration-Driven** - всё через settings.json и .env (никакого хардкода!)
4. **Modular Design** - слабосвязанные компоненты с чёткими интерфейсами
5. **Lazy Loading** - инициализация по требованию

### Обязательное обновление документации

При любых изменениях **ОБЯЗАТЕЛЬНО** обновлять:
- `rules/Technical Debt.md` - технический долг и статусы задач
- `rules/Development Roadmap.md` - прогресс выполнения
- `rules/Technical Architecture.md` - архитектурные изменения
- **НЕ создавать новые файлы в папке rules/** - работать только с существующими

### Тестирование

#### Категоризация тестов
- **Unit тесты** (без маркеров) - изолированные, работают с `--disable-socket`
- **Integration тесты** (`@pytest.mark.integration`) - требуют OpenAI API, Qdrant, filesystem
- **Functional тесты** (`@pytest.mark.functional`) - subprocess, CLI команды
- **E2E тесты** (`@pytest.mark.e2e`) - полные пользовательские сценарии

#### Требования к тестам
- **Offline/mock режим обязателен** для unit тестов
- **Все сетевые вызовы должны быть замоканы**
- **Property-based тесты** через `hypothesis`
- **Минимум 90% покрытия** для unit тестов

#### Mock правила
- Все моки должны возвращать корректные объекты
- Метод `.to()` у моков обязан возвращать `self`
- `MockSparseModel` должен наследоваться от `torch.nn.Module`
- Async методы мокировать через async функции, не через Mock(return_value=...)

## Технологический стек

### Core
- Python 3.8+
- OpenAI API (gpt-4)
- Click (CLI framework)
- Rich (CLI UI)
- Streamlit (Web UI)

### RAG System
- Jina Embeddings v3 (jinaai/jina-embeddings-v3, 1024d, 570M параметров)
- Sentence Transformers ≥3.0.0 (для Jina v3)
- Qdrant ≥1.15.1 (векторная БД)
- FastAPI ≥0.115.0 (VM сервис)
- aiohttp ≥3.10.0 (HTTP клиент)

### Hybrid Search
- Dense vectors (Jina v3)
- Sparse vectors (SPLADE)
- RRF (Reciprocal Rank Fusion)
- MMR (Maximal Marginal Relevance)

### VM Infrastructure
- Intel Xeon Gold 6248R, 31GB RAM
- Ubuntu 22.04.4 LTS
- FastAPI service на 10.61.11.54:8000
- SSH automation через paramiko

## Важные технические детали

### Jina v3 Configuration
- **Размерность**: 1024d (единый стандарт для всей системы)
- **Trust Remote Code**: обязателен для загрузки модели
- **Dual Task**: отдельные embeddings для query/passage
- **Pooling**: mean pooling
- **Normalization**: L2 нормализация включена

### Векторное хранилище (Qdrant)
- **Квантование**: Scalar Quantization (SQ) по умолчанию
- **HNSW параметры**: m=24, ef_construct=128
- **Репликация**: factor=2
- **mmap**: включён для экономии RAM

### Performance Targets
- **Latency**: <200ms (cached), <500ms (cold)
- **Indexing**: >8 файлов/секунду
- **Concurrency**: 20+ параллельных пользователей
- **Memory**: ~100MB локально (99% экономия)

## Частые проблемы и решения

### VM недоступен
```bash
python vm_start.py status
python vm_start.py restart
curl http://10.61.11.54:8000/health
```

### OpenAI API ошибки
```bash
echo $OPENAI_API_KEY
python main.py token-stats
```

### Поиск не работает
```bash
python main.py rag status --detailed
python main.py rag index /path/to/repo --recreate
```

### Тесты падают
- Проверить категоризацию теста (unit/integration/functional)
- Убедиться что моки корректны для async методов
- Проверить environment variables в тесте
- Использовать `os.getenv()` вместо хардкода localhost

## Ссылки на документацию

### Основная документация
- [README.md](../README.md) - главная документация
- [rules/AGENTS.md](../rules/AGENTS.md) - правила работы агентов
- [rules/Development Roadmap.md](../rules/Development Roadmap.md) - дорожная карта

### Техническая документация
- [rules/Technical Architecture.md](../rules/Technical Architecture.md) - архитектура
- [rules/Technical Debt.md](../rules/Technical Debt.md) - технический долг
- [tests/rag/TESTING_STRATEGY.md](../tests/rag/TESTING_STRATEGY.md) - стратегия тестирования

### Workflow
1. Изучить [rules/AGENTS.md](../rules/AGENTS.md) перед началом работы
2. Проверить [rules/Technical Debt.md](../rules/Technical Debt.md) для актуальных задач
3. Обновлять документацию параллельно с кодом
4. Писать тесты для всех изменений
5. Обновлять статус в Technical Debt.md после завершения

## Контакты и поддержка

- **GitHub**: https://github.com/Bondartsov/repo_sum.git
- **VM Service**: 10.61.11.54:8000
- **Qdrant DB**: 10.61.11.54:6333

## Стиль работы

@.claude/output-styles/DEV.md

## Дополнительные правила проекта

Специфичные правила проекта находятся в `rules/AGENTS.md`.