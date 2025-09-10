# System Patterns: Repository Analyzer

**Дата:** 10 сентября 2025  
**Статус:** Production-Ready система с установленными архитектурными паттернами  
**Версия:** 1.8.0 (M2 завершён)

---

## 🏛️ Фундаментальные архитектурные принципы

### 1. CPU-First Architecture
**Принцип**: Оптимизация для широкой совместимости без требования GPU
**Применение**:
- FastEmbed с ONNX Runtime вместо GPU-зависимых решений
- HNSW параметры настроены для CPU (m=16-32, ef_construct=64-128)
- Управление потоками через OMP_NUM_THREADS, MKL_NUM_THREADS
- Адаптивные батчи в зависимости от доступной RAM

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

**Пример**:
```python
@dataclass
class EmbeddingConfig:
    provider: str = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size_max: int = 512
    normalize_embeddings: bool = True
```

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

### 3. Hybrid Search Pattern
**Применение**: Комбинирование dense и sparse векторов
**Архитектура**:
```
[Query] → ├─[Dense Embedder] → Dense Search
          └─[Sparse Encoder] → Sparse Search
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

## 🔄 Паттерны интеграции и коммуникации

### 1. Event-Driven Pattern (частично)
**Применение**: Обработка lifecycle событий
**События**:
- File scanned → trigger parsing
- Chunk created → trigger embedding  
- Index updated → invalidate cache

### 2. Command Pattern
**Применение**: CLI команды как объекты
**Команды**:
```python
class AnalyzeCommand:
    def execute(self, args): ...
    
class RAGSearchCommand:
    def execute(self, args): ...
```

### 3. Observer Pattern (планируется)
**Применение**: Мониторинг изменений файлов для инкрементальной индексации
**Наблюдатели**:
- File system watcher → trigger re-indexing
- Config change watcher → reload configuration

---

## 📊 Метрики и мониторинг паттерны

### 1. Metrics Collection Pattern
**Применение**: Сбор метрик производительности
**Метрики**:
- Latency histograms для поиска
- Counter для количества запросов
- Gauge для использования памяти
- Timer для длительности операций

### 2. Health Check Pattern
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

---

## 🧪 Паттерны тестирования

### 1. Test Categorization Pattern
**Применение**: Разделение тестов по типам с pytest маркерами
**Категории**:
- `unit` - изолированные компоненты (59 тестов)
- `integration` - взаимодействие компонентов (67 тестов)
- `functional` - CLI/subprocess тесты (25 тестов)
- `e2e` - end-to-end сценарии

### 2. Mock Strategy Pattern
**Применение**: Изоляция компонентов в тестах
**Mock объекты**:
- `MockTokenizer` - детерминированное токенизирование
- `MockSparseModel` - предсказуемые sparse векторы
- `MockOpenAIClient` - эмуляция API без costs

### 3. Fixture Pattern
**Применение**: Переиспользование тестовых данных
**Фикстуры**:
- `test_repo` - 1743 строки реального кода для семантического тестирования
- `mock_config` - типовые конфигурации для разных сценариев

---

## 🔮 Паттерны эволюции системы

### 1. Feature Flag Pattern (готов к реализации)
**Применение**: Постепенное внедрение новых возможностей
**Флаги**:
- `enable_sparse_search` - включение sparse векторов
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

## 🎯 Заключение

Система `repo_sum` построена на прочной архитектурной основе с использованием проверенных паттернов проектирования. **Ключевые принципы**:

### Достигнутые цели:
- ✅ **Масштабируемость** - модульная архитектура поддерживает рост
- ✅ **Производительность** - CPU-first подход с оптимизациями
- ✅ **Надёжность** - fail-safe паттерны и comprehensive тестирование
- ✅ **Расширяемость** - plugin architecture для новых языков и возможностей
- ✅ **Безопасность** - многоуровневая защита чувствительных данных

### Готовность к развитию:
Архитектурные паттерны создают прочную основу для реализации следующих milestone:
- **M3**: RAG-enhanced анализ - архитектура поддерживает интеграцию
- **M4**: Production deployment - паттерны готовы к масштабированию  
- **M5**: Advanced Intelligence - foundation для ML-оптимизаций

**Система демонстрирует industry best practices** и готова к enterprise использованию.

---
**Дата создания:** 10 сентября 2025  
**Статус:** Production-Ready Architecture  
**Следующий пересмотр:** При реализации M3-M4 milestone
