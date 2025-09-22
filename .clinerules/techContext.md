# Technical Context: Repository Analyzer

**Дата:** 22 сентября 2025  
**Статус:** M2.5 VM Migration завершена - RAG-as-a-Service production-ready  
**Версия:** 0.7.1 (M2.5 VM Migration COMPLETE, async/sync fixes реализованы)

---

## 🏗️ Технологический стек

### Основные технологии:
- **Python 3.8+**: Основной язык разработки
- **OpenAI GPT API** >= 1.99.6 - ИИ-анализ кода
- **Qdrant** >= 1.15.1 - Enterprise-ready векторная база данных с квантованием
- **FastEmbed** >= 0.3.6 - CPU-оптимизированные эмбеддинги (ONNX Runtime)
- **Sentence-Transformers** >= 3.0.0 - основной провайдер эмбеддингов + Jina v3 поддержка (trust_remote_code)
- **Jina Embeddings v3** - революционная модель с dual task архитектурой (570M параметров, 1024d)
- **Streamlit** >= 1.46.0 - веб-интерфейс с RAG интеграцией
- **AST (Abstract Syntax Tree)** - парсинг кода всех языков
- **Markdown** - формат выходной документации

### RAG система (Production-Ready):
```
rag/
├── embedder.py         # CPU-оптимизированный эмбеддер
├── vector_store.py     # Qdrant интеграция с квантованием
├── query_engine.py     # Поисковый движок с RRF/MMR
├── indexer_service.py  # Сервис индексации репозиториев
├── search_service.py   # Высокоуровневый поиск с фильтрацией
├── sparse_encoder.py   # BM25/SPLADE кодирование
└── exceptions.py       # Система исключений
```

### Зависимости (VM + автоматизация):
```txt
# Core dependencies
openai>=1.99.6                    # OpenAI API клиент
streamlit>=1.46.0                 # Web UI
python-dotenv>=1.0.0              # Environment variables + VM config
click>=8.1.8                      # CLI framework
rich>=14.0.0                      # CLI UI library

# RAG System (VM-ready + Jina v3)
qdrant-client[fastembed]>=1.15.1  # FastEmbed + Qdrant клиент
sentence-transformers>=3.0        # Jina v3 требует >=3.0 для trust_remote_code
transformers>=4.35.0              # Современная версия для Jina v3 support
numpy>=1.24.0                     # Векторные операции
psutil>=5.9.5                     # RAM мониторинг (критично для VM)
cachetools>=5.3.0                 # LRU/TTL кэширование

# VM Automation (NEW)
paramiko>=4.0.0                   # SSH автоматизация для VM
fastapi>=0.104.0                  # RAG-as-a-Service API (планируется)
uvicorn>=0.24.0                   # ASGI server для FastAPI

# Hybrid Search (M2)
rank-bm25>=0.2.2                  # BM25 алгоритм
nltk>=3.8                         # Токенизация
datasets>=2.21.0                  # Вспомогательные утилиты

# Testing & Development
pytest>=8.3.4                     # Тестирование
pytest-asyncio>=1.1.0             # Асинхронные тесты
```

---

## 🏛️ Фундаментальные архитектурные принципы

### 1. CPU-First Architecture с Jina v3 Integration
**Принцип**: Оптимизация для широкой совместимости без требования GPU
**Применение**:
- **Jina v3 Dual Task**: sentence-transformers с task-specific LoRA адаптерами
- **Adaptive HNSW**: динамические параметры (m=24, ef_construct=128 для 1024d)
- **Trust Remote Code**: безопасная загрузка jinaai/jina-embeddings-v3
- FastEmbed fallback с ONNX Runtime для совместимости
- Управление потоками через OMP_NUM_THREADS, MKL_NUM_THREADS
- Адаптивные батчи в зависимости от доступной RAM и размерности векторов

### 2. Modular Architecture Pattern
**Принцип**: Разделение системы на слабосвязанные, независимые модули
**Структура**:
```
repo_sum/
├── Core System/        # Анализ кода и документация
├── RAG System/         # Семантический поиск
├── Parsers System/     # Языковые парсеры
├── UI System/          # CLI + Web интерфейсы  
└── Testing System/     # Комплексное тестирование
```

**Преимущества**:
- Независимая разработка компонентов
- Простота тестирования и отладки
- Возможность замены компонентов без влияния на систему
- Чёткое разделение ответственностей

### 3. Configuration-Driven Development
**Принцип**: Централизованное управление поведением через конфигурацию
**Реализация**:
- `settings.json` - основная конфигурация
- `.env` - environment variables для production
- `@dataclass` конфигурационные классы с валидацией
- Типизированные конфиги с дефолтными значениями

**Пример (Jina v3)**:
```python
@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"
    model_name: str = "jinaai/jina-embeddings-v3"
    vector_size: int = 1024
    batch_size_max: int = 512
    normalize_embeddings: bool = True
    trust_remote_code: bool = True
    task_query: str = "retrieval.query"
    task_passage: str = "retrieval.passage"
```

---

## ⚙️ Ключевые компоненты

### Core система:
1. **RepositoryAnalyzer** (main.py) - основной координатор анализа
2. **FileScanner** (file_scanner.py) - сканирование и фильтрация файлов
3. **ParserRegistry** (parsers/) - выбор парсера по типу файла
4. **CodeChunker** (code_chunker.py) - разбивка кода на логические части
5. **OpenAIManager** (openai_integration.py) - интеграция с OpenAI API
6. **DocumentationGenerator** (doc_generator.py) - генерация Markdown отчетов

### RAG система (Production-Ready):
1. **CPUEmbedder** (rag/embedder.py) - CPU-оптимизированный эмбеддер
2. **QdrantVectorStore** (rag/vector_store.py) - векторное хранилище
3. **CPUQueryEngine** (rag/query_engine.py) - поисковый движок с RRF + MMR
4. **IndexerService** (rag/indexer_service.py) - сервис индексации
5. **SearchService** (rag/search_service.py) - высокоуровневый поиск
6. **SparseEncoder** (rag/sparse_encoder.py) - BM25/SPLADE векторы (M2)

### Поддерживаемые языки:
- Python (.py)
