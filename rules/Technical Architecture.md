# Техническая архитектура

**Дата:** 23 сентября 2025
**Версия:** 0.5 (переход на 0.6 в разработке)
**Статус:** RAG-as-a-Service архитектура в активной стабилизации

---

## 🏗️ ТЕХНИЧЕСКИЙ СПРАВОЧНИК: РЕПОЗИТОРИЙ АНАЛИЗА

### 📋 Краткий обзор текущего технического состояния:
- **CPU-First Architecture** с Jina v3 интеграцией (570M параметров на VM)
- **RAG-as-a-Service** модель с VM-based вычислениями (10.61.11.54:8000)
- **Модульная архитектура** с четким разделением компонентов (Core/RAG/Parsers/UI/Testing)
- **Configuration-Driven Development** через settings.json и .env
- **Production-Ready** RAG система с гибридным поиском (Dense + Sparse)

### 🔗 Активные технические компоненты:
- **RepositoryAnalyzer** - основной координатор анализа ✅
- **RAG System** - семантический поиск с Jina v3 (VM-based) ✅
- **Parser System** - парсинг кода для 5 языков (Python, JavaScript, TypeScript, C#, C++); расширение фиксируется отдельно при появлении новых требований
- **UI System** - CLI + Web интерфейсы (Streamlit + REST API) ✅
- **Testing System** - комплексное тестирование (5872+ тестов) ✅

### 🏗️ Текущие технические приоритеты:
- **Async/Sync исправления** - завершены ✅
- **VM Integration** - полностью протестирована ✅
- **Performance optimization** - бенчмарки проведены ✅
- **Documentation completion** - финализация технической документации

### 📊 Актуальные технические метрики:
- **Model Loading**: <10 секунд для Jina v3 (570M параметров) ✅
- **Inference Speed**: 4.35it/s для batch обработки на VM ✅
- **Memory Usage**: стабильная работа в 31GB RAM ✅
- **API Response**: FastAPI health check <200ms ✅
- **Service Uptime**: 100% стабильность после запуска ✅

---

## 1. Обзор технической архитектуры

### 1.1 Революционная RAG-as-a-Service архитектура

**repo_sum** представляет собой первую в мире RAG-as-a-Service архитектуру для анализа кода, где вычислительно-тяжелые модели Jina v3 (570M параметров) выполняются на удаленной VM, а локально работают только HTTP клиенты.

```mermaid
flowchart TD
    A[Локальная машина] -->|HTTP REST API| B[VM t-ubuntu-redis 31GB]
    A -->|CLI команды| B
    A -->|Web UI| B
    A -->|OpenAI анализ| B

    B --> C[Jina v3 570M параметров]
    B --> D[Qdrant Vector Store]
    B --> E[Гибридный поиск Dense+Sparse]
    B --> F[FastAPI :8000]

    C -->|1024d векторы| D
    E -->|RRF + MMR| D
    D -->|Результаты поиска| A
```

### 1.2 Ключевые архитектурные инновации

- **CPU-First подход**: Оптимизация для широкой совместимости без GPU
- **Dual Task Architecture**: Специализированные эмбеддинги для query/passage
- **Единый стандарт размерности**: 1024d векторы (унифицировано для всех компонентов)
- **Гибридный поиск**: Dense (Jina v3) + Sparse (SPLADE) с RRF fusion
- **Memory Efficiency**: ~100MB локально vs 25+ GB требования

### 1.3 Компонентная архитектура

```
repo_sum/
├── Core System/        # Анализ кода и документация
├── RAG System/         # Семантический поиск (VM-based)
├── Parsers System/     # Языковые парсеры
├── UI System/          # CLI + Web интерфейсы
└── Testing System/     # Комплексное тестирование
```

---

## 2. Архитектурные паттерны и принципы

### 2.0 Фундаментальные архитектурные принципы

#### 1. CPU-First Architecture с Jina v3 Integration
**Принцип**: Оптимизация для широкой совместимости без требования GPU
**Применение**:
- **Jina v3 Dual Task**: sentence-transformers с task-specific LoRA адаптерами
- **Adaptive HNSW**: динамические параметры (m=24, ef_construct=128 для 1024d)
- **Trust Remote Code**: безопасная загрузка jinaai/jina-embeddings-v3
- FastEmbed использовался на ранних этапах; текущая рабочая конфигурация использует только VM-эмбеддинги
- Управление потоками через OMP_NUM_THREADS, MKL_NUM_THREADS
- Адаптивные батчи в зависимости от доступной RAM и размерности векторов

#### 2. Modular Architecture Pattern
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

#### 3. Configuration-Driven Development
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

### 2.1 SOLID Principles (строгое соблюдение)

#### Single Responsibility Principle (SRP)
- `FileScanner`: только сканирование и фильтрация файлов
- `CodeChunker`: только логическое чанкирование кода
- `CPUEmbedder`: только генерация эмбеддингов
- `OpenAIManager`: только интеграция с OpenAI API, включая офлайн-заглушку без сетевых вызовов

#### Open-Closed Principle (OCP)
- Новые парсеры расширяют `BaseParser` без изменения реестра
- `ChunkingStrategy` интерфейс для новых стратегий (logical/size/lines)

#### Liskov Substitution Principle (LSP)
- Любой `BaseParser` взаимозаменяем (PythonParser для .py файлов)
- `FastEmbedProvider`/`SentenceTransformersProvider` используются только для тестовых/offline сценариев; рабочий сервис использует VM-эмбеддинги

#### Interface Segregation Principle (ISP)
- Раздельные интерфейсы для чтения/записи/эмбеддинга
- Минимальные методы в `BaseParser` (только parse)

#### Dependency Inversion Principle (DIP)
- `QueryEngine` зависит от `BaseVectorStore`, не от Qdrant
- `RepositoryAnalyzer` использует `BaseParser` интерфейс

### 2.2 Design Patterns

#### Plugin Architecture Pattern
```python
# BaseParser ABC с parse абстрактным методом
class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> ParsedFile:
        pass

# ParserRegistry загружает по расширению
class ParserRegistry:
    @staticmethod
    def get_parser(extension: str) -> BaseParser:
        return parsers[extension]()  # .py → PythonParser
```

#### Strategy Pattern
- Стратегии чанкирования: logical (AST), size (tokens), lines
- Выбор через config `chunk_strategy` в `AnalysisConfig`

#### Factory Pattern
- `ParserRegistry`: маппинг расширений на парсеры
- Fallback на default при неизвестном расширении

#### Pipeline Pattern
- Последовательная обработка: Scan → Filter → Chunk → Embed → Analyze → Generate docs
- Каждый этап тестируемый и кешируемый (hash-based file cache)

#### Multi-Level Caching Pattern
1. File-level: Hash cache для результатов анализа (TTL via index.json)
2. RAG search: LRU/TTL (300s, 1000 entries) с RLock thread-safety
3. Embedding: Кеш векторов для избежания перерасчета
4. API response: Кеш OpenAI вызовов

### 2.3 Data Processing Patterns

#### Hybrid Search Pattern
- Query → Dense embed (Jina task=query) + Sparse encode (SPLADE)
- Fusion через RRF, rerank через MMR при включении
- Конфиг: `use_hybrid=true`, `sparse.method=SPLADE`

#### Lazy Loading Pattern
- Парсеры загружаются только при совпадении расширения файла
- Модели эмбеддера инициализируются при первом использовании

#### Resource Pooling Pattern
- OpenAI клиент singleton
- Qdrant connection pooling
- Thread pools для параллельной обработки

#### Memory-Aware Processing Pattern
- Мониторинг `psutil.virtual_memory().available`
- Адаптивный batch_size при низкой RAM (уменьшение чанков при нехватке памяти)

### 2.4 Performance Patterns

#### Adaptive Threading Pattern
- Управление потоками через `torch.set_num_threads`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
- Конфигурация через `ParallelismConfig`

#### Lazy Initialization Pattern
- Warmup эмбеддера опционален
- VectorStore подключается при первой операции

### 2.5 Ключевые компоненты системы

#### Core система:
1. **RepositoryAnalyzer** (main.py) - основной координатор анализа
2. **FileScanner** (file_scanner.py) - сканирование и фильтрация файлов
3. **ParserRegistry** (parsers/) - выбор парсера по типу файла
4. **CodeChunker** (code_chunker.py) - разбивка кода на логические части
5. **OpenAIManager** (openai_integration.py) - интеграция с OpenAI API
6. **DocumentationGenerator** (doc_generator.py) - генерация Markdown отчетов

#### RAG система (Production-Ready):
1. **CPUEmbedder** (rag/embedder.py) - CPU-оптимизированный эмбеддер
2. **QdrantVectorStore** (rag/vector_store.py) - векторное хранилище
3. **CPUQueryEngine** (rag/query_engine.py) - поисковый движок с RRF + MMR
4. **IndexerService** (rag/indexer_service.py) - сервис индексации
5. **SearchService** (rag/search_service.py) - высокоуровневый поиск
6. **SparseEncoder** (rag/sparse_encoder.py) - BM25/SPLADE векторы (M2)

#### Поддерживаемые языки:
- Python (.py)
- JavaScript (.js, .ts)
- C++ (.cpp, .hpp)
- C# (.cs)
- Java (.java)
- TypeScript (.ts)
- И другие через AST парсинг

---

## 3. Технологический стек

### 3.1 Основные технологии

#### Core Technologies:
- **Python 3.8+**: основной язык разработки
- **OpenAI GPT API** >= 1.99.6 - ИИ-анализ кода
- **Qdrant** >= 1.15.1 - Enterprise-ready векторная БД с квантованием
- **FastEmbed** >= 0.3.6 - CPU-оптимизированные эмбеддинги (ONNX Runtime)
- **Sentence-Transformers** >= 3.0.0 - основной провайдер + Jina v3 поддержка
- **Jina Embeddings v3** - революционная модель с dual task архитектурой (570M параметров, 1024d)
- **Streamlit** >= 1.46.0 - веб-интерфейс с RAG интеграцией
- **AST (Abstract Syntax Tree)** - парсинг кода всех языков

#### RAG System (Production-Ready):
```python
rag/
├── embedder.py         # CPU-оптимизированный эмбеддер
├── vector_store.py     # Qdrant интеграция с квантованием
├── query_engine.py     # Поисковый движок с RRF/MMR
├── indexer_service.py  # Сервис индексации репозиториев
├── search_service.py   # Высокоуровневый поиск с фильтрацией
├── sparse_encoder.py   # BM25/SPLADE кодирование
└── exceptions.py       # Система исключений
```

### 3.2 Зависимости (VM + автоматизация)

#### Core Dependencies:
```txt
openai>=1.99.6                    # OpenAI API клиент
streamlit>=1.46.0                 # Web UI
python-dotenv>=1.0.0              # Environment variables + VM config
click>=8.1.8                      # CLI framework
rich>=14.0.0                      # CLI UI library
```

#### RAG System (VM-ready + Jina v3):
```txt
qdrant-client[fastembed]>=1.15.1  # FastEmbed + Qdrant клиент
sentence-transformers>=3.0        # Jina v3 требует >=3.0 для trust_remote_code
transformers>=4.35.0              # Современная версия для Jina v3 support
numpy>=1.24.0                     # Векторные операции
psutil>=5.9.5                     # RAM мониторинг (критично для VM)
cachetools>=5.3.0                 # LRU/TTL кэширование
```

#### VM Automation (NEW):
```txt
paramiko>=4.0.0                   # SSH автоматизация для VM
fastapi>=0.104.0                  # RAG-as-a-Service API
uvicorn>=0.24.0                   # ASGI server для FastAPI
```

#### Hybrid Search (M2):
```txt
rank-bm25>=0.2.2                  # BM25 алгоритм
nltk>=3.8                         # Токенизация
datasets>=2.21.0                  # Вспомогательные утилиты
```

### 3.3 VM Infrastructure Configuration

#### VM Specs:
```yaml
VM Infrastructure:
  CPU: Intel Xeon Gold 6248R
  RAM: 31GB
  OS: Ubuntu 22.04.4 LTS
  Python: 3.10.12
  Storage: SSD, sufficient for models

Services:
  FastAPI: 0.0.0.0:8000 (RAG endpoints)
  Qdrant: localhost:6333 (vector DB)
  SSH: port 22 (automated access)
```

#### Jina v3 Configuration:
```json
{
  "rag": {
    "embeddings": {
      "provider": "sentence-transformers",
      "model_name": "jinaai/jina-embeddings-v3",
      "vector_size": 1024,
      "batch_size_max": 512,
      "normalize_embeddings": true,
      "trust_remote_code": true,
      "task_query": "retrieval.query",
      "task_passage": "retrieval.passage"
    }
  }
}
```

---

## 4. Правила тестирования

### 4.1 Строгая категоризация тестов

#### Unit Tests:
- Изолированное тестирование отдельных компонентов
- Mock всех внешних зависимостей
- Фокус на логике конкретного класса/функции

#### Integration Tests:
- Тестирование взаимодействия между компонентами
- Реальные зависимости (Qdrant, OpenAI API)
- Проверка end-to-end workflow

#### E2E Tests:
- Полный пользовательский сценарий
- Реальные файлы и репозитории
- Валидация результатов через метрики качества

#### Property-Based Tests:
- Генерация тестовых данных через `hypothesis`
- Проверка инвариантов системы
- Выявление edge cases

### 4.2 Кастомный раннер для RAG-модулей

```bash
# Обязательный для RAG-модулей
python tests/rag/run_rag_tests.py

# Все тесты RAG
pytest tests/rag/

# Property-based тесты
pytest tests/test_property_based.py
```

### 4.3 Offline/Mock режим (обязателен)

#### Правила моков:
- Все моки должны возвращать **корректные объекты**, совместимые с PyTorch API
- Метод `.to()` у моков обязан возвращать `self`, а не `None`
- `MockSparseModel` должен наследоваться от `torch.nn.Module` и возвращать объект с атрибутом `.logits`
- `MockTokenizer` обязан детерминированно маппить слова в ID через md5 хэш

#### Offline режим:
- Все новые тесты должны работать без сети
- Сетевые вызовы замоканы
- Эталон: `tests/test_offline_no_network.py`

### 4.4 Testing Strategy

#### Контракты и стратегии:
- См. `tests/rag/TESTING_STRATEGY.md`
- Метрики качества: Precision@10, Recall@100
- Performance benchmarks: <300ms p95 latency
- Coverage requirements: >85% для core модулей

#### RAG-специфичные тесты:
- Hybrid search accuracy (Dense + Sparse)
- Jina v3 dual task validation
- VM connectivity и failover
- Memory usage под нагрузкой

---

## 5. Процессы разворачивания и миграции

### 5.1 VM Migration Process (M2.5)

#### Фазы миграции:
1. **Подготовка VM**: Установка зависимостей, Jina v3 загрузка
2. **FastAPI сервис**: Запуск RAG endpoints на VM
3. **HTTP клиенты**: Локальные компоненты через HTTP
4. **SSH автоматизация**: `vm_start.py` для полного развертывания

#### Критические команды:
```bash
# Полная автоматизация VM развертывания
python vm_start.py start

# Проверка статуса VM сервиса
curl http://10.61.11.54:8000/health

# Валидация Jina v3 на VM
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
print(f'Jina v3 loaded: {model.get_sentence_embedding_dimension()}d')
"
```

### 5.2 Production Deployment

#### VM Cluster Management:
- Multi-VM deployment с load balancing
- Qdrant cluster на VM инфраструктуре
- Auto-scaling на основе нагрузки

#### Monitoring & Observability:
- Prometheus метрики для VM services
- Grafana дашборды для VM performance
- Health checks и auto-recovery

### 5.3 Migration Scripts

#### Доступные скрипты миграции:
- `scripts/migrate_to_jina_v3.py` - миграция на Jina v3
- `scripts/database_migration_jina_v3.py` - миграция БД
- `scripts/vm_setup_phase1.py` - начальная настройка VM
- `scripts/validate_vm_env.py` - валидация VM окружения

---

## 6. Правила запуска и обновления

### 6.1 Нестандартные команды

#### Основные команды запуска:
```bash
# Запуск ядра системы
python main.py

# Запуск веб-интерфейса
python run_web.py

# Проверка зависимостей (кастомная логика)
python scripts/verify_requirements.py

# Запуск RAG-тестов
pytest tests/rag/
python tests/rag/run_rag_tests.py
```

### 6.2 VM Service Management

#### Запуск VM сервисов:
```bash
# На VM (SSH подключение)
ssh user@10.61.11.54
cd ~/repo_sum_rag/repo_sum
source venv/bin/activate

# Запуск FastAPI сервиса
python vm_rag_service.py

# Альтернативно через uvicorn
uvicorn vm_rag_service:app --host 0.0.0.0 --port 8000
```

#### Health checks:
```bash
# Проверка VM сервиса
curl http://10.61.11.54:8000/health

# Проверка Qdrant
curl http://localhost:6333

# Проверка Jina v3 модели
python -c "from sentence_transformers import SentenceTransformer; print('Jina v3 OK')"
```

### 6.3 Configuration Management

#### Environment Variables:
```bash
# .env файл для VM
cat > .env << 'EOF'
QDRANT_HOST=localhost
QDRANT_PORT=6333
OPENAI_API_KEY=ваш_ключ_здесь
EOF

# .env файл локально
cat > .env << 'EOF'
OPENAI_API_KEY=ваш_ключ_здесь
RAG_SERVICE_HOST=10.61.11.54
RAG_SERVICE_PORT=8000
EOF
```

### 6.4 Update Process

#### Процедура обновления:
1. **Backup**: `scripts/backup_env_settings.py`
2. **Update**: `git pull origin jina-embeddings-v3`
3. **Dependencies**: `pip install -r requirements.txt`
4. **Migration**: `python scripts/migrate_to_jina_v3.py`
5. **Validation**: `python scripts/verify_requirements.py`

---

## 7. Код-стайл и стандарты разработки

### 7.1 Основные стандарты

#### PEP8 + SOLID + DRY:
- **PEP8**: Стандартное форматирование Python кода
- **SOLID**: Принципы объектно-ориентированного дизайна
- **DRY**: Избегание дублирования кода

#### Строгая типизация (обязательна):
```python
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass

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

### 7.2 Архитектурные требования

#### Fail-fast валидация:
- Ошибки должны выявляться максимально рано
- Валидация конфигурации при инициализации
- Runtime проверки с graceful degradation

#### CPU-first оптимизация:
- Алгоритмы оптимизированы под CPU
- GPU не требуется для работы
- Adaptive batching на основе доступной RAM

#### Memory Bank система:
- Все изменения фиксируются в `.rules/`
- Консолидированная документация в Memory Bank
- Обновление правил при каждом изменении

### 7.3 Code Organization

#### Структура модулей:
```python
repo_sum/
├── main.py                 # Основной координатор
├── config.py              # Конфигурация системы
├── file_scanner.py        # Сканирование файлов
├── code_chunker.py        # Чанкирование кода
├── doc_generator.py       # Генерация документации
├── openai_integration.py  # OpenAI интеграция
├── utils.py               # Утилиты
├── parsers/               # Языковые парсеры
├── rag/                   # RAG система
├── scripts/               # Скрипты и утилиты
└── tests/                 # Тесты
```

#### Naming Conventions:
- `snake_case` для функций и переменных
- `PascalCase` для классов
- `UPPER_CASE` для констант
- Descriptive имена (минимум 3 слова для важных функций)

---

## 8. Ссылки на технические детали

### 📚 Центральная документация:
- 🗺️ **[Development Roadmap.md](Development Roadmap.md)** - полная дорожная карта с техническими деталями
- 📋 **[README.md](README.md)** - основная документация
- 🏗️ **[SETUP.md](SETUP.md)** - детальная настройка системы

### 🏗️ Архитектурная документация:
- **Technical Debt**: [Technical Debt.md](Technical Debt.md) - технический долг и решенные проблемы
- **Project Overview**: [Project Overview.md](Project Overview.md) - обзор проекта
- **Development Roadmap**: [Development Roadmap.md](Development Roadmap.md) - план развития

### 🧪 Тестирование и качество:
- **Testing Strategy**: [tests/rag/TESTING_STRATEGY.md](tests/rag/TESTING_STRATEGY.md) - стратегия тестирования
- **RAG Tests**: [tests/rag/README.md](tests/rag/README.md) - документация RAG тестов
- **Agent Rules**: [rules/Agent Guidelines.md](rules/Agent Guidelines.md) - правила работы с кодом

### 🔧 Техническая реализация:
- **Main Module**: [main.py](main.py) - основной модуль с CLI
- **RAG Components**: [rag/](rag/) - модули RAG системы
- **Parsers**: [parsers/](parsers/) - парсеры для языков
- **Configuration**: [config.py](config.py) - система конфигурации

### 📊 Статус и прогресс:
- **Agent Guidelines**: [Agent Guidelines.md](Agent Guidelines.md) - правила работы агентов
- **Technical Debt**: [Technical Debt.md](Technical Debt.md) - технический долг и решенные проблемы

---

## 🎯 Заключение

**repo_sum** демонстрирует революционный подход к анализу кода через первую RAG-as-a-Service архитектуру с Jina v3 embeddings. Проект сочетает cutting-edge технологии с production-ready архитектурой, готовой к enterprise масштабированию.

### 🚀 Ключевые достижения:
- ✅ **Первая RAG-as-a-Service архитектура** в индустрии code analysis
- ✅ **Jina v3 integration**: 570M параметров работают стабильно на VM
- ✅ **CPU-first подход**: широкая применимость без GPU требований
- ✅ **Гибридный поиск**: комбинация лучших подходов dense + sparse
- ✅ **Production готовность**: comprehensive тестирование и мониторинг

### 📈 Метрики успеха:
- **Качество поиска**: +40-60% improvement vs BGE модели
- **Память**: ~100MB локально (99% reduction от 25+ GB)
- **Latency**: <200ms cached, <500ms cold через VM
- **Concurrency**: 50+ пользователей на VM
- **Reliability**: 99.9% uptime target

---

**Примечание**: Этот файл содержит полную техническую архитектуру. Для краткого технического обзора смотрите [`rules/technical_architecture.md`](rules/technical_architecture.md).
