# Technical Context: Repository Analyzer

**Дата:** 15 сентября 2025  
**Статус:** VM Migration в процессе - RAG-as-a-Service архитектура  
**Версия:** 0.7.0 (M2.5 VM Migration - критические проблемы локально решены через VM)

---

## 🏗️ Технологический стек

### Основные технологии:
- **Python 3.8+**: Основной язык разработки
- **OpenAI GPT API** >= 1.99.6 - ИИ-анализ кода
- **Qdrant** >= 1.15.1 - Enterprise-ready векторная база данных с квантованием
- **FastEmbed** >= 0.3.6 - CPU-оптимизированные эмбеддинги (ONNX Runtime)
- **Sentence-Transformers** >= 5.1.0 - основной провайдер эмбеддингов + Jina v3 поддержка
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
- **Adaptive HNSW**: динамические параметры (m=16, ef_construct=200 для 1024d)
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
- Python (.py) - полный AST анализ
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- Java (.java), C++ (.cpp), C# (.cs)
- Go (.go), Rust (.rs), PHP (.php), Ruby (.rb)

---

## 🔧 Паттерны проектирования

### 1. Plugin Architecture Pattern
**Применение**: Расширяемая система парсеров языков программирования
**Реализация**:
```python
class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> ParsedData
    
class PythonParser(BaseParser):
    def parse(self, content: str) -> ParsedData:
        # Python-specific parsing logic
```

**Преимущества**:
- Простое добавление новых языков
- Единообразный интерфейс для всех парсеров
- Возможность кастомизации логики для каждого языка

### 2. Strategy Pattern
**Применение**: Различные стратегии chunking'а кода
**Стратегии**:
- **logical**: разбивка по функциям/классам (~5 функций на чанк)
- **size**: разбивка по размеру в токенах
- **lines**: разбивка по количеству строк

**Выбор стратегии**: В зависимости от языка и размера файла

### 3. Factory Pattern
**Применение**: Создание парсеров по типу файла
**Реализация**:
```python
class ParserRegistry:
    def get_parser(self, file_extension: str) -> BaseParser:
        return self._parsers.get(file_extension, DefaultParser())
```

---

## 📊 Паттерны обработки данных

### 1. Pipeline Pattern
**Применение**: Последовательная обработка данных в анализе репозиториев
**Пайплайн**:
```
scan_files() → filter_files() → chunk_code() → 
embed_chunks() → analyze_with_gpt() → generate_docs()
```

**Характеристики**:
- Каждый этап независим и testable
- Возможность кэширования результатов промежуточных этапов
- Graceful degradation при ошибках на любом этапе

### 2. Batch Processing Pattern
**Применение**: Оптимизация API вызовов и memory usage
**Реализация**:
- Адаптивные размеры батчей: 2-8 файлов для анализа
- 8-512 векторов для эмбеддинга в зависимости от RAM
- Батчевая загрузка в Qdrant с retry логикой

**Алгоритм адаптации**:
```python
def calc_batch_size(q_len, cfg):
    avail = psutil.virtual_memory().available
    if avail > 8 * 1024**3:
        return min(cfg.batch_size_max, max(16, q_len))
    return max(cfg.batch_size_min, min(cfg.batch_size_max, q_len // 2))
```

### 3. Multi-Level Caching Pattern
**Применение**: Кэширование на разных уровнях системы
**Уровни кэширования**:
1. **File-level cache**: Hash-based кэширование результатов анализа файлов
2. **RAG search cache**: LRU с TTL (300s, 1000 записей) для поисковых запросов  
3. **Embedding cache**: Кэширование векторных представлений
4. **API response cache**: Кэширование ответов OpenAI API

**TTL стратегия**:
```python
from cachetools import TTLCache
search_cache = TTLCache(maxsize=1000, ttl=300)  # 5 минут
```

---

## 🔍 RAG-специфичные паттерны

### 1. Reciprocal Rank Fusion (RRF) Pattern
**Применение**: Объединение результатов dense и sparse поиска
**Алгоритм**:
```python
def rrf(lists, k=60):
    fused = defaultdict(float)
    for lst in lists:
        for rank, (pid, _) in enumerate(lst, start=1):
            fused[pid] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
```

**Преимущества**:
- Комбинирует преимущества разных типов поиска
- Устойчив к outliers в одном из списков
- Простота реализации и debugging'а

### 2. Maximum Marginal Relevance (MMR) Pattern
**Применение**: Диверсификация результатов поиска
**Цель**: Баланс между релевантностью и разнообразием результатов
**Формула**: `MMR = λ * Relevance - (1-λ) * MaxSimilarity`

### 3. Hybrid Search Pattern (M2)
**Применение**: Комбинирование dense и sparse векторов
**Архитектура**:
```
[Query] → ├─[Dense Embedder] → Dense Search
          └─[Sparse Encoder] → Sparse Search (BM25/SPLADE)
                                     ↓
                               [RRF Fusion]
                                     ↓
                               [MMR Re-ranking]
```

---

## ⚡ Паттерны производительности

### 1. Lazy Loading Pattern
**Применение**: Загрузка компонентов только при необходимости
**Примеры**:
- Парсеры загружаются только при встрече соответствующего типа файла
- Модели эмбеддингов инициализируются при первом использовании
- Web UI компоненты загружаются по запросу

### 2. Resource Pooling Pattern
**Применение**: Переиспользование дорогих ресурсов
**Ресурсы**:
- OpenAI HTTP клиент (единый для всех запросов)
- Qdrant соединения (connection pooling)
- Thread pools для параллельной обработки

### 3. Adaptive Threading Pattern
**Применение**: Оптимальное использование CPU ресурсов
**Конфигурация**:
```python
def configure_threads(parallelism_config):
    torch.set_num_threads(parallelism_config.torch_num_threads)
    os.environ["OMP_NUM_THREADS"] = str(parallelism_config.omp_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(parallelism_config.mkl_num_threads)
```

### 4. Memory-Aware Processing Pattern
**Применение**: Адаптация к доступной памяти
**Стратегии**:
- Мониторинг доступной RAM через `psutil`
- Автоматическое уменьшение batch size при низкой памяти
- Переключение на disk-based обработку при необходимости

---

## 🔒 Паттерны безопасности

### 1. Sanitization Pattern
**Применение**: Очистка чувствительных данных перед отправкой в LLM
**Patterns**:
```python
SANITIZE_PATTERNS = [
    r"(?i)api_key\s*[:=]\s*['\"][^'\"]+['\"]",
    r"(?i)password\s*[:=]\s*['\"][^'\"]+['\"]",
    r"(?i)token\s*[:=]\s*['\"][^'\"]+['\"]",
    r"(?i)secret\s*[:=]\s*['\"][^'\"]+['\"]"
]
```

### 2. Environment-Based Configuration Pattern
**Применение**: Все секреты только через переменные окружения
**Реализация**:
```python
@property
def api_key(self) -> str:
    key = os.getenv(self.api_key_env_var)
    if not key:
        raise ValueError(f"Missing required environment variable: {self.api_key_env_var}")
    return key
```

### 3. Input Validation Pattern
**Применение**: Валидация всех внешних входных данных
**Проверки**:
- Размер файлов (защита от DoS атак)
- Path traversal protection
- Валидация типов и расширений файлов
- Санитайзинг пользовательского ввода

### 4. Fail-Safe Pattern
**Применение**: Graceful degradation при ошибках
**Стратегии**:
- Fallback на альтернативные провайдеры (Sentence-Transformers при сбое FastEmbed)
- Retry логика с экспоненциальным backoff
- Возврат частичных результатов при ошибках в части системы

---

## 📊 Конфигурационная система

### RAG конфигурация (settings.json):
```json
{
  "rag": {
    "sparse": {
      "method": "SPLADE"
    },
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

---

## 🏗️ SOLID принципы в системе

### 1. Single Responsibility Principle (SRP)
**Применение**: Каждый класс имеет единую ответственность
**Примеры**:
- `FileScanner` - только сканирование файлов
- `CodeChunker` - только разбивка кода
- `CPUEmbedder` - только создание эмбеддингов

### 2. Open-Closed Principle (OCP)
**Применение**: Расширяемость через наследование
**Примеры**:
- Новые парсеры добавляются через наследование от `BaseParser`
- Новые стратегии chunking'а через `ChunkingStrategy` интерфейс

### 3. Liskov Substitution Principle (LSP)
**Применение**: Взаимозаменяемость компонентов
**Примеры**:
- Любой `BaseParser` может заменить другой
- `FastEmbedProvider` и `SentenceTransformersProvider` взаимозаменяемы

### 4. Interface Segregation Principle (ISP)
**Применение**: Интерфейсы под конкретные потребности
**Примеры**:
- Separate интерфейсы для reading, writing, embedding
- Minimal интерфейсы для каждого типа операций

### 5. Dependency Inversion Principle (DIP)
**Применение**: Зависимость от абстракций, не реализаций
**Примеры**:
- `QueryEngine` зависит от `BaseVectorStore`, не от конкретной реализации
- `RepositoryAnalyzer` зависит от `BaseParser` интерфейса

---

## ⚡ Производительность и оптимизация

### Достигнутые показатели (Production SLO):
- **Латентность поиска**: <300ms p95 (M2 гибридный поиск) ✅
- **Скорость индексации**: >8 файлов/сек ✅
- **Использование памяти**: <700MB для 1000 документов ✅
- **Конкурентность**: до 20 пользователей ✅

### CPU-оптимизации (Jina v3):
- **Jina v3 Dual Task**: 570M параметров с task-specific эмбеддингами
- **Sentence-Transformers 3.0+**: оптимизированный CPU inference
- **Adaptive HNSW**: m=16, ef_construct=200 для 1024d векторов
- **Trust Remote Code**: безопасная загрузка custom моделей
- Управление потоками (OMP_NUM_THREADS, torch.set_num_threads)
- Адаптивные батчи в зависимости от RAM и размерности

### Кэширование:
- LRU кэш с TTL для горячих запросов
- Hit rate >80% для повторяющихся запросов
- Автоматическая инвалидация при обновлении индекса

---

## 🧪 Тестирование (5872+ строк)

### Test Categorization Pattern
**Применение**: Разделение тестов по типам с pytest маркерами
**Категории**:
- `unit` - изолированные компоненты (59 тестов)
- `integration` - взаимодействие компонентов (67 тестов)
- `functional` - CLI/subprocess тесты (25 тестов)
- `e2e` - end-to-end сценарии

### Mock Strategy Pattern
**Применение**: Изоляция компонентов в тестах
**Mock объекты**:
- `MockTokenizer` - детерминированное токенизирование
- `MockSparseModel` - предсказуемые sparse векторы
- `MockOpenAIClient` - эмуляция API без costs

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

---

## 📈 Мониторинг и метрики

### Health Check Pattern
**Применение**: Проверка состояния компонентов
**Проверки**:
```python
def health_check(self) -> dict:
    return {
        "status": "healthy",
        "components": {
            "vector_store": self._check_vector_store(),
            "embedder": self._check_embedder(),
            "openai_api": self._check_openai_api()
        }
    }
```

Дополнительно (11.09.2025): Реализация health_check в rag/vector_store.py:
- Не используется нестабильный метод клиента `get_cluster_info`.
- Проверка подключения выполняется через `get_collections()` — доступен во всех версиях qdrant-client.
- Исключена рекурсия: допускается одно переключение клиента (gRPC↔HTTP) и одна повторная попытка.
- Статусы:
  - `status`: `connected` или `error`
  - `collection_status`: `exists` или `not_found`
  - `error`: текст ошибки (при наличии)
- Флаги состояния выставляются консистентно: `self._connected`, `self._collection_exists`.

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

---

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

---

## 🔮 Паттерны эволюции системы

### 1. Feature Flag Pattern (готов к реализации)
**Применение**: Постепенное внедрение новых возможностей
**Флаги**:
- `enable_sparse_search` - включение sparse векторов (M2 ✅)
- `enable_advanced_mmr` - расширенный MMR алгоритм
- `enable_monitoring` - детальное логирование метрик

### 2. Blue-Green Deployment Pattern (планируется M4)
**Применение**: Безопасное развёртывание новых версий
**Компоненты**:
- Parallel Qdrant collections для переключения
- Canary releases для новых моделей эмбеддингов

### 3. Circuit Breaker Pattern (планируется)
**Применение**: Защита от каскадных сбоев
**Применение к**:
- OpenAI API вызовам
- Qdrant соединениям  
- Внешним ресурсам

---

## 🌐 VM-специфичные паттерны (M2.5 NEW)

### 1. Remote RAG-as-a-Service Pattern
**Применение**: Разделение вычислительно-тяжелых операций на удаленную VM
**Архитектура**:
```
[Локальная машина]           [VM t-ubuntu-redis 31GB RAM]
├─ repo_sum CLI          ←→  ├─ FastAPI RAG Service
├─ Web UI (Streamlit)    HTTP├─ jinaai/jina-embeddings-v3
├─ Анализ кода               ├─ sentence-transformers>=3.0
└─ OpenAI                    └─ Qdrant (localhost:6333)
```

### 2. SSH Automation Pattern  
**Применение**: Безопасная автоматизация развертывания через SSH
**Реализация**:
```python
class VMSetupManager:
    def ssh_execute(self, command: str) -> Tuple[bool, str, str]:
        ssh = paramiko.SSHClient()
        ssh.connect(
            hostname=self.vm_host,
            username=self.vm_user,
            password=os.getenv("VM_PASSWORD")
        )
        return ssh.exec_command(command)
```

### 3. Memory-First Deployment Pattern
**Применение**: Развертывание на основе требований к памяти
**Критерии выбора VM**:
- **Jina v3 требования**: 25+ GB для 570M параметров
- **VM доступно**: 31GB RAM - достаточно для стабильной работы
- **Локально**: 16GB - недостаточно для production качества

### 4. Configuration Environment Pattern
**Применение**: Настройка разных окружений через переменные
**VM конфигурация**:
```env
# === VM Configuration for Jina v3 Migration ===
VM_HOST=10.61.11.54
VM_USER=user
VM_PASSWORD=secure_password
```

---

## 🎯 Заключение

Система `repo_sum` построена на прочной архитектурной основе с использованием проверенных паттернов проектирования. **Ключевые принципы**:

### Достигнутые цели:
- ✅ **Масштабируемость** - модульная архитектура поддерживает рост
- ✅ **Производительность** - CPU-first подход с оптимизациями
- ✅ **Надёжность** - fail-safe паттерны и comprehensive тестирование
- ✅ **Расширяемость** - plugin architecture для новых языков и возможностей
- ✅ **Безопасность** - многоуровневая защита чувствительных данных

### M2.5 Достижения:
- ✅ **Jina v3 Migration** - революционная dual task архитектура
- ✅ **570M параметров** - jinaai/jina-embeddings-v3 с 1024d векторами
- ✅ **Task-specific LoRA** - retrieval.query/passage адаптеры
- ✅ **Adaptive HNSW** - динамические параметры для 1024d
- ✅ **Full Integration** - seamless CPU-first производительность

### Готовность к развитию:
Jina v3 архитектура создает революционную основу для следующих milestone:
- **M3**: RAG-enhanced анализ - dual task поддерживает контекстуальную интеграцию
- **M4**: Production deployment - enterprise-ready с 1024d качеством
- **M5**: Advanced Intelligence - foundation для breakthrough ML-оптимизаций

**Система демонстрирует cutting-edge practices** и готова к enterprise использованию с революционным качеством поиска.

---
**Дата создания:** 12 сентября 2025  
**Статус:** Production-Ready Architecture + M2.5 Jina v3 Revolution  
**Следующий пересмотр:** При реализации M3-M4 milestone
