# Technical Context: Repository Analyzer

**Дата:** 15 августа 2025  
**Статус:** Production-Ready с RAG системой  
**Версия:** 1.5.0

## 🏗️ Технологический стек

### Основные технологии:
- **Python 3.8+**: Основной язык разработки
- **OpenAI GPT API** >= 1.99.6 - ИИ-анализ кода
- **Qdrant** >= 1.15.1 - Enterprise-ready векторная база данных с квантованием
- **FastEmbed** >= 0.3.6 - CPU-оптимизированные эмбеддинги (ONNX Runtime)
- **Sentence-Transformers** >= 5.1.0 - fallback провайдер эмбеддингов
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
└── exceptions.py       # Система исключений
```

### Зависимости (актуализированные):
```txt
# Core dependencies
openai>=1.99.6                    # OpenAI API клиент
streamlit>=1.46.0                 # Web UI
python-dotenv>=1.0.0              # Environment variables
click>=8.1.8                      # CLI framework
rich>=14.0.0                      # CLI UI library

# RAG System (CPU-first)
qdrant-client[fastembed]>=1.15.1  # FastEmbed + Qdrant клиент
sentence-transformers>=5.1.0      # Fallback эмбеддинги
numpy>=1.24.0                     # Векторные операции
psutil>=5.9.5                     # RAM мониторинг
cachetools>=5.3.0                 # LRU/TTL кэширование

# Testing & Development
pytest>=8.3.4                     # Тестирование
pytest-asyncio>=1.1.0             # Асинхронные тесты
```

## ⚙️ Ключевые компоненты

### Core система:
1. **RepositoryAnalyzer** (main.py) - основной координатор анализа
2. **FileScanner** (file_scanner.py) - сканирование и фильтрация файлов
3. **ParserRegistry** (parsers/) - выбор парсера по типу файла
4. **CodeChunker** (code_chunker.py) - разбивка кода на логические части
5. **OpenAIManager** (openai_integration.py) - интеграция с OpenAI API
6. **DocumentationGenerator** (doc_generator.py) - генерация Markdown отчетов

### ✅ RAG система (Production-Ready):
1. **CPUEmbedder** (rag/embedder.py) - CPU-оптимизированный эмбеддер
2. **QdrantVectorStore** (rag/vector_store.py) - векторное хранилище
3. **CPUQueryEngine** (rag/query_engine.py) - поисковый движок с RRF + MMR
4. **IndexerService** (rag/indexer_service.py) - сервис индексации
5. **SearchService** (rag/search_service.py) - высокоуровневый поиск

### Поддерживаемые языки:
- Python (.py) - полный AST анализ
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- Java (.java), C++ (.cpp), C# (.cs)
- Go (.go), Rust (.rs), PHP (.php), Ruby (.rb)

## 🏛️ Архитектурные паттерны

### 1. Модульная архитектура
**Применение**: Разделение системы на независимые модули
- Core система: анализ и документация
- RAG система: семантический поиск
- Parsers: расширяемая система парсеров языков
- Tests: комплексная система тестирования

### 2. Plugin Architecture
**Применение**: Расширяемая система парсеров
```python
class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> ParsedData
```

### 3. Configuration-Driven Development
**Применение**: Централизованное управление через settings.json
```python
@dataclass
class EmbeddingConfig:
    provider: str = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size_max: int = 512
```

### 4. Strategy Pattern
**Применение**: Различные стратегии chunking'а кода
- logical: по функциям/классам
- size: по размеру в токенах
- lines: по количеству строк

## 🔄 Паттерны обработки данных

### 1. Pipeline Pattern
**Применение**: Последовательная обработка данных
```
scan_files() → chunk_code() → analyze_with_gpt() → generate_docs()
```

### 2. Batch Processing Pattern
**Применение**: Группировка файлов для эффективного API использования
- Адаптивные размеры батчей: от 2 до 8 файлов в зависимости от размера проекта

### 3. Caching Pattern
**Применение**: Кэширование результатов анализа
- Hash-based кэширование по содержимому файла
- TTL (Time To Live) для автоочистки
- Инвалидация при изменении файла

### 4. RAG Search Caching (TTL)
**Применение**: Кэширование поисковых запросов
```python
from cachetools import TTLCache
search_cache = TTLCache(maxsize=1000, ttl=300)
```

### 5. Reciprocal Rank Fusion (RRF)
**Применение**: Фьюжен dense и sparse результатов поиска
```python
def rrf(lists, k=60):
    fused = defaultdict(float)
    for lst in lists:
        for rank, (pid, _) in enumerate(lst, start=1):
            fused[pid] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
```

### 6. MMR Re-ranking
**Применение**: Диверсификация результатов поиска
- Балансировка релевантности и разнообразия
- Борьба с дубликатами в результатах

## 🚀 Паттерны производительности

### 1. Lazy Loading Pattern
**Применение**: Загрузка парсеров только при необходимости
```python
class ParserRegistry:
    def get_parser(self, language: str) -> BaseParser:
        if language not in self._loaded_parsers:
            self._loaded_parsers[language] = self._load_parser(language)
        return self._loaded_parsers[language]
```

### 2. Resource Pooling Pattern
**Применение**: Переиспользование OpenAI клиента
```python
class OpenAIManager:
    def __init__(self):
        self._client = OpenAI()  # Единый клиент
        self._session_cache = {}
```

### 3. Adaptive Batch Encoding
**Применение**: Адаптивный размер батча эмбеддингов
```python
import psutil

def calc_batch_size(q_len, cfg):
    avail = psutil.virtual_memory().available
    if avail > 8 * 1024**3:
        return min(cfg.batch_size_max, max(16, q_len))
    return max(cfg.batch_size_min, min(cfg.batch_size_max, q_len // 2))
```

### 4. Parallelism Threads Configuration
**Применение**: Управление потоками для CPU производительности
```python
def configure_threads(par):
    torch.set_num_threads(par.torch_num_threads)
    os.environ["OMP_NUM_THREADS"] = str(par.omp_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(par.mkl_num_threads)
```

## 🔒 Паттерны безопасности

### 1. Sanitization Pattern
**Применение**: Очистка чувствительных данных перед отправкой в LLM
```python
SANITIZE_PATTERNS = [
    r"(?i)api_key\s*[:=]\s*['\"][^'\"]+['\"]",
    r"(?i)password\s*[:=]\s*['\"][^'\"]+['\"]"
]
```

### 2. Environment-based Configuration
**Применение**: API ключи только через переменные окружения
```python
@property
def api_key(self) -> str:
    return os.getenv(self.api_key_env_var)
```

### 3. Validation Pattern
**Применение**: Валидация всех входных данных
- Размер файлов (защита от DoS)
- Path traversal protection
- Валидация типов файлов

## 📊 Конфигурационная система

### RAG конфигурация (settings.json):
```json
{
  "rag": {
    "embeddings": {
      "provider": "fastembed",
      "model_name": "BAAI/bge-small-en-v1.5",
      "batch_size_max": 512,
      "normalize_embeddings": true
    },
    "vector_store": {
      "host": "localhost",
      "port": 6333,
      "collection_name": "code_chunks",
      "quantization_type": "SQ"
    },
    "query_engine": {
      "max_results": 10,
      "rrf_enabled": true,
      "mmr_enabled": true,
      "cache_ttl_seconds": 300
    }
  }
}
```

### Dataclass конфигурации (config.py):
```python
@dataclass
class EmbeddingConfig:
    provider: str = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size_max: int = 512

@dataclass
class VectorStoreConfig:
    host: str = "localhost"
    port: int = 6333
    quantization_type: str = "SQ"

@dataclass
class QueryEngineConfig:
    max_results: int = 10
    rrf_enabled: bool = True
    mmr_enabled: bool = True
    cache_ttl_seconds: int = 300
```

## ⚡ Производительность и оптимизация

### Достигнутые показатели (Production SLO):
- **Латентность поиска**: <200ms p95 ✅
- **Скорость индексации**: >10 файлов/сек ✅
- **Использование памяти**: <500MB для 1000 документов ✅
- **Конкурентность**: до 20 пользователей ✅

### CPU-оптимизации:
- FastEmbed с ONNX Runtime
- HNSW параметры для CPU (m=16-32, ef_construct=64-128)
- Управление потоками (OMP_NUM_THREADS, torch.set_num_threads)
- Адаптивные батчи в зависимости от RAM

### Кэширование:
- LRU кэш с TTL для горячих запросов
- Hit rate >80% для повторяющихся запросов
- Автоматическая инвалидация при обновлении индекса

## 🧪 Тестирование (5872+ строк)

### Типы тестов:
- **Unit тесты** - изолированное тестирование компонентов
- **Интеграционные тесты** - взаимодействие компонентов
- **E2E CLI тесты** - команды в реальных условиях
- **Performance тесты** - метрики и стресс-тесты

### Структура тестов:
```
tests/
├── test_rag_*.py              # Unit тесты RAG компонентов
├── rag/                       # Комплексные RAG тесты
│   ├── test_rag_integration.py
│   ├── test_rag_e2e_cli.py
│   └── test_rag_performance.py
└── fixtures/test_repo/        # 1743 строки реального кода
```

## 📈 Мониторинг и метрики

### Логируемые метрики:
- Время анализа файла/проекта
- Количество токенов на запрос/ответ
- Частота кэш-попаданий
- Ошибки API и retry статистика
- Метрики RAG системы (латентность поиска, hit rate)

### Мониторинг RAG:
```python
from prometheus_client import Counter, Histogram
qdrant_requests_total = Counter("qdrant_requests_total", ["op"])
qdrant_search_latency = Histogram("qdrant_search_latency_seconds")
```

## 🔄 CI/CD и развертывание

### Поддерживаемые платформы:
- **Windows** - основная платформа разработки
- **Linux** - серверное развертывание
- **macOS** - desktop использование
- **Docker** - контейнеризация (готов к M4)

### Развертывание:
1. **Локальное**: pip install + streamlit
2. **Production**: VPS с systemd
3. **Enterprise** (M4): Docker-compose с Qdrant кластером

## 🔧 Принципы качества кода

### SOLID Principles:
- **SRP**: Каждый класс имеет единую ответственность
- **OCP**: Расширяемость через наследование (парсеры)
- **LSP**: Взаимозаменяемость компонентов
- **ISP**: Интерфейсы под конкретные потребности
- **DIP**: Зависимость от абстракций, не реализаций

### Best Practices:
- **DRY** - переиспользование компонентов
- **Explicit is better than implicit** - типизация, валидация
- **Fail-fast principle** - валидация при старте

---

**Техническая архитектура готова к дальнейшему развитию**: milestone M2-M4 имеют прочную основу для реализации гибридного поиска, enterprise развёртывания и производственного мониторинга.
