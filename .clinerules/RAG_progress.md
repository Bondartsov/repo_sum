# 🔥 PRODUCTION-READY ПЛАН RAG СИСТЕМЫ (АУДИТ 08.2025)

**ОБНОВЛЕНО**: Полностью переработанный план с учетом аудита актуальных библиотек, CPU-оптимизации и production-ready подходов.

## 🏗️ АРХИТЕКТУРНЫЙ ОБЗОР

### Критические принципы дизайна (ОБНОВЛЕНО):
1. **Адаптивная сегментация кода** с сохранением логической целостности (НЕ микро-сегментация)
2. **CPU-first эмбединги** через современные модели (all-MiniLM, e5-small, BGE)
3. **Гибридный поиск** (dense + sparse) для максимального качества на CPU
4. **Инкрементальная индексация** с версионированием чанков
5. **Production-ready управление памятью** и ресурсами

## 📋 ФАЗА 1: FOUNDATION (Неделя 1-1.5)

### 1.1 Настройка зависимостей и инфраструктуры

**Новые зависимости в requirements.txt (ОБНОВЛЕНО 08.2025):**
```python
# RAG System Core - CPU-оптимизированные версии
sentence-transformers>=5.1.0    # Современная версия с precision control
torch>=2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
qdrant-client>=1.10.0          # Актуальная версия с гибридным поиском
numpy>=1.24.0                  # Векторные операции
psutil>=5.9.5                  # RAM мониторинг и управление
cachetools>=5.3.0              # LRU/TTL кэширование

# Альтернативные CPU-first эмбедеры
fastembed>=0.3.0               # ONNX Runtime, quantized weights
faiss-cpu>=1.7.4               # Fallback векторный поиск
```

**Критические проверки совместимости (ИСПРАВЛЕНО):**
- Sentence-transformers v5.x API совместимость с precision параметрами
- Qdrant client поддерживает гибридный поиск и quantization
- Torch CPU threads настроены правильно (OMP_NUM_THREADS, MKL_NUM_THREADS)
- Удален torchvision (не нужен для текстовых моделей)

### 1.2 Расширение конфигурационной системы

**config.py - новые dataclasses (CPU-оптимизированные):**
```python
@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "code_chunks"  
    vector_size: int = 384  # 384d для CPU-efficient моделей
    distance_metric: str = "cosine"
    timeout: float = 10.0
    prefer_grpc: bool = True
    # CPU-оптимизированные HNSW параметры
    hnsw_m: int = 16              # Уменьшено для экономии RAM
    hnsw_ef_construct: int = 64   # Оптимально для CPU
    enable_quantization: bool = True  # INT8 квантование

@dataclass  
class EmbeddingsConfig:
    model_name: str = "intfloat/e5-small-v2"  # CPU-efficient 384d модель
    model_cache_dir: str = "./models_cache"
    batch_size: int = 16          # Уменьшено для CPU
    max_length: int = 256         # Укорочено для производительности
    cache_embeddings: bool = True
    device: str = "cpu"           # CPU-first
    precision: str = "int8"       # ST v5.x precision control
    normalize_embeddings: bool = True  # Встроенная нормализация
    enable_matryoshka: bool = True     # Truncate_dim для экономии

@dataclass
class ChunkingConfig:
    strategy: str = "adaptive_logical"  # НЕ микро-сегментация
    max_chunk_size: int = 300          # Больше для сохранения контекста
    min_chunk_size: int = 50           # Минимум для логической целостности
    overlap_tokens: int = 50
    preserve_hierarchy: bool = True
    include_docstrings: bool = True
    include_comments: bool = False
    enable_auto_grouping: bool = True   # Автоматическое определение групп

@dataclass
class CPUOptimizationConfig:
    max_ram_usage_mb: int = 2048       # Лимит RAM
    enable_model_unloading: bool = True # Выгрузка после обработки
    adaptive_batch_size: bool = True    # Адаптивный размер батча
    cache_type: str = "lru_ttl"        # LRU/TTL кэш вместо dict
    max_cache_size: int = 1000         # Максимум записей в кэше
    cache_ttl_hours: int = 24          # TTL для кэша
    num_cpu_threads: int = 4           # Контроль потоков CPU
```

### 1.3 Создание модуля rag/

**Структура директории:**
```
rag/
├── __init__.py           # Экспорт основных классов
├── exceptions.py         # RAG-специфичные исключения
├── models.py            # Pydantic модели для данных
├── vector_store.py      # Qdrant интеграция
├── code_embedder.py     # CodeBERT эмбединги
├── semantic_chunker.py  # Микро-сегментация AST
├── query_engine.py      # Семантический поиск
├── chat_interface.py    # Streamlit чат UI
├── code_context.py      # Управление контекстом
└── rag_config.py        # RAG конфигурация
```

## 📋 ФАЗА 2: CORE COMPONENTS (Неделя 1.5-2.5)

### 2.1 VectorStore (rag/vector_store.py)

**Ключевые возможности:**
- Подключение к локальному/облачному Qdrant
- Создание/управление коллекциями с метаданными
- Bulk операции для массовой индексации
- Фильтрация по файлам/языкам/типам чанков
- Backup/restore векторных индексов

**Критические методы:**
```python
class QdrantVectorStore:
    async def initialize_collection(self, recreate: bool = False)
    async def upsert_chunks(self, chunks: List[CodeChunk]) -> List[str]
    async def similarity_search(self, query_vector: np.ndarray, 
                               filters: Dict, top_k: int) -> List[SearchResult]
    async def delete_by_file(self, file_path: str) -> int
    async def get_stats(self) -> CollectionStats
```

**Безопасность и производительность:**
- Connection pooling для Qdrant
- Retry логика с exponential backoff  
- Валидация векторных размерностей
- Мониторинг использования памяти

### 2.2 CPUEmbedder (rag/code_embedder.py) - ОБНОВЛЕНО 08.2025

**Модели (CPU-приоритет - ИСПРАВЛЕНО):**
1. **intfloat/e5-small-v2** (384d) - CPU-эффективная модель
2. **BAAI/bge-small-en-v1.5** (384d) - современная альтернатива MiniLM
3. **sentence-transformers/all-MiniLM-L6-v2** (384d) - fallback опция
4. **FastEmbed ONNX модели** - максимальная производительность на CPU

**Ключевые возможности (ИСПРАВЛЕНО):**
- Lazy loading с CPU-оптимизацией
- Sentence Transformers v5.x с precision control
- Адаптивный batch sizing по доступной RAM
- LRU/TTL кэширование вместо простого dict
- Встроенная нормализация через ST v5 API

**Критические методы (ОБНОВЛЕНО):**
```python
class CPUEmbedder:
    def __init__(self, config: CPUOptimizationConfig)
    async def ensure_model_loaded(self) -> None
    async def embed_batch_cpu_safe(self, texts: List[str]) -> np.ndarray:
        # ST v5.x с precision='int8' и normalize_embeddings=True
        return self.model.encode(
            texts, 
            precision=self.config.precision,
            normalize_embeddings=True,
            batch_size=self.calculate_adaptive_batch_size()
        )
    def unload_model(self) -> None  # Для освобождения RAM
    def calculate_adaptive_batch_size(self) -> int  # По доступной RAM
```

### 2.3 SemanticChunker (rag/semantic_chunker.py) - ИСПРАВЛЕНО

**Адаптивные стратегии сегментации (НЕ микро-сегментация):**
- **Adaptive Logical**: Функции/классы как целостные блоки (50-300 строк)
- **Smart Grouping**: Автоматическое определение групп (auth, db, api)
- **Context Preservation**: Сохранение 10 строк контекста вокруг блока
- **Hierarchy Aware**: Учет parent-child связей без разрушения логики

**Метаданные для каждого чанка (РАСШИРЕНО):**
```python
@dataclass
class VersionedCodeChunk:
    id: str                    # Уникальный ID
    content: str              # Исходный код (полный логический блок)
    file_path: str            # Путь к файлу
    language: str             # Язык программирования
    chunk_type: ChunkType     # function_complete/class_with_methods/config_section
    parent_id: Optional[str]  # ID родительского чанка
    line_start: int           # Начальная строка
    line_end: int            # Конечная строка
    tokens_count: int        # Количество токенов
    dependencies: List[str]   # Зависимости (imports, calls)
    context_window: str      # Окружающий контекст
    
    # Новые поля для CPU-оптимизации
    content_hash: str = field(init=False)  # SHA-256 для версионирования
    logical_group: Optional[str] = None    # auth/db/api автогруппировка
    contains_auth_logic: bool = False      # Флаг для быстрой фильтрации
    is_config_file: bool = False          # Поддержка .env, .yml файлов
    version: int = 1                      # Версия чанка
    last_modified: datetime = field(default_factory=datetime.now)
```

**Сохранение логической целостности (ИСПРАВЛЕНО):**
- File → Class → Method (как ЕДИНЫЕ блоки, не разбитые)
- Автоопределение групп через статический анализ
- Версионирование чанков через content_hash
- Адаптивное объединение мелких блоков (min_chunk_size=50)

## 📋 ФАЗА 3: SEARCH & QUERY ENGINE (Неделя 2.5-3.5)

### 3.1 QueryEngine (rag/query_engine.py)

**Алгоритм семантического поиска (CPU-оптимизированный):**
1. **Query Processing**: Preprocessing пользовательского запроса
2. **Intent Recognition**: Определение типа запроса (explain/find/refactor/etc.)
3. **Hybrid Search**: Dense + sparse векторный поиск в Qdrant (RRF/DBSF)
4. **Context Expansion**: Добавление связанных чанков по logical_group
5. **Re-ranking**: CPU-эффективное переранжирование (hnsw_ef=128-256)
6. **Context Assembly**: Сборка финального контекста для LLM

**Типы запросов (с logical grouping):**
- **Explain queries**: "Объясни функцию X" → поиск в группе + related calls
- **Find queries**: "Где используется переменная Y" → поиск по dependencies + group filtering
- **Pattern queries**: "Найди похожие паттерны" → similarity search + MMR server-side
- **Config queries**: "Где настроена авторизация?" → поиск в config files + logical_group="auth"

**Критические методы (ОБНОВЛЕНО):**
```python
class CPUQueryEngine:
    async def hybrid_search(self, query: str, filters: SearchFilters) -> SearchResult
    async def search_by_logical_group(self, query: str, group: str) -> GroupedResult
    async def explain_code_with_context(self, code_id: str) -> ExplanationResult  
    async def find_config_related(self, query: str) -> ConfigResult
    def calculate_relevance_score_cpu(self, chunk: VersionedCodeChunk) -> float
```

### 3.2 HybridSearchEngine (rag/hybrid_search.py) - НОВАЯ ФАЗА 3.2

**Гибридный поиск (dense + sparse):**
- **Dense vectors**: e5-small-v2 эмбединги для семантического понимания
- **Sparse vectors**: FastEmbed SPLADE/miniCOIL для точного терминологического поиска
- **RRF (Reciprocal Rank Fusion)**: Комбинирование результатов
- **MMR server-side**: Разнообразие результатов без CPU overhead

**Критические возможности:**
```python
class HybridSearchEngine:
    async def dense_search(self, query: str) -> List[SearchResult]
    async def sparse_search(self, query: str) -> List[SearchResult] 
    async def fuse_results_rrf(self, dense: List, sparse: List) -> List[SearchResult]
    async def apply_mmr_server_side(self, results: List) -> List[SearchResult]
```

### 3.2 CodeContext (rag/code_context.py)

**Управление контекстом:**
- Построение графа зависимостей между чанками
- Трекинг импортов и их использования
- Анализ call graphs для функций
- Иерархическая навигация по коду

**Типы связей:**
- **Parent-Child**: Иерархические связи (class → methods)
- **Dependencies**: Import и usage связи  
- **Similarity**: Семантически похожие чанки
- **Temporal**: Чанки в одном временном контексте

## 📋 ФАЗА 4: CHAT INTERFACE & INTEGRATION (Неделя 3.5-4)

### 4.1 ChatInterface (rag/chat_interface.py)

**Streamlit Chat UI компоненты:**
```python
def render_chat_interface():
    # История сообщений с session state
    # Поле ввода с auto-complete
    # Кнопки быстрых запросов  
    # Показ релевантных чанков кода
    # Подсветка синтаксиса в ответах
```

**Встроенные команды:**
- `/explain [function_name]` - объяснение функции
- `/find [variable_name]` - поиск использований
- `/similar [code_block]` - поиск похожих паттернов
- `/refactor [function_name]` - предложения рефакторинга
- `/deps [module_name]` - анализ зависимостей

### 4.2 Интеграция с существующей системой

**main.py - новая команда:**
```bash
python main.py chat /path/to/repository
# Запускает RAG индексацию + чат интерфейс
```

**web_ui.py - новая вкладка:**
- "Code Chat" tab рядом с существующими
- Автоматическая индексация загруженного проекта
- Переключение между анализом и чатом

**Инкрементальная синхронизация:**
- Мониторинг изменений файлов
- Реиндексация только измененных чанков
- Сохранение истории чата при обновлении

## 🔧 КРИТИЧЕСКИЕ ИНТЕГРАЦИОННЫЕ ТОЧКИ

### Интеграция с FileScanner
```python
# file_scanner.py должен поддерживать:
def get_changed_files(self, since_timestamp: float) -> List[FileInfo]
def watch_directory(self, callback: Callable) -> None
```

### Интеграция с существующими Parsers
```python  
# Расширение базового парсера:
class BaseParser:
    def extract_semantic_chunks(self, content: str) -> List[SemanticChunk]
    def build_dependency_graph(self, parsed_data: ParsedData) -> DependencyGraph
```

### Интеграция с OpenAIManager
```python
# Новые методы для RAG queries:
async def explain_with_context(self, code_chunks: List[CodeChunk], query: str)
async def suggest_refactoring(self, code_context: CodeContext)
```

## 🧪 ТЕСТИРОВАНИЕ СТРАТЕГИЯ

### Unit Tests:
```
tests/rag/
├── test_vector_store.py      # Qdrant операции
├── test_code_embedder.py     # Эмбединг генерация
├── test_semantic_chunker.py  # AST разбиение
├── test_query_engine.py      # Поисковые алгоритмы
├── test_chat_interface.py    # UI компоненты
└── test_integration_rag.py   # End-to-end тесты
```

### Performance Tests:
- Индексация больших проектов (1000+ файлов)
- Поиск performance с разными векторными размерами
- Memory usage при различных batch sizes
- Latency чата при concurrent запросах

### Security Tests:  
- Санитайзинг в RAG запросах
- Валидация векторных данных
- Rate limiting для chat API

## ⚡ ПРОИЗВОДИТЕЛЬНОСТЬ И МАСШТАБИРОВАНИЕ

### Оптимизации:
1. **Lazy Loading**: Модели загружаются только при использовании
2. **Batch Processing**: Группировка эмбединг операций
3. **Caching Strategy**: Многоуровневое кэширование (memory → disk → Qdrant)
4. **Async Processing**: Неблокирующие операции везде где возможно

### Scalability Considerations:
- Horizontal scaling через Qdrant cluster
- Model sharding для очень больших проектов  
- Connection pooling для множественных пользователей
- Background реиндексация без блокировки UI

## 🚨 РИСКИ И МИТИГАЦИЯ

### Технические риски:
1. **Model Loading Time** → Lazy loading + model caching
2. **Memory Consumption** → Streaming processing + garbage collection
3. **Qdrant Connection Issues** → Connection pooling + retry logic
4. **Vector Dimensionality Mismatch** → Strict validation + migration tools

### Пользовательские риски:
1. **Poor Search Quality** → Comprehensive testing + feedback loop
2. **Slow Response Time** → Performance monitoring + optimization
3. **Complex Setup** → Docker containers + one-click installation

## 📊 МОНИТОРИНГ И МЕТРИКИ

### RAG-специфичные метрики:
- **Index Size**: Количество векторов в коллекции
- **Search Latency**: Время ответа на запросы
- **Embedding Cache Hit Rate**: Эффективность кэширования
- **Chat Session Duration**: Engagement метрики
- **Query Success Rate**: Процент успешных поисков

### Dashboarding:
- Streamlit metrics sidebar в chat UI
- Detailed stats через отдельную admin страницу
- Логирование в structured format для analysis

---

Этот план покрывает все критические аспекты реализации RAG системы с максимальной детализацией. Готов к обсуждению любых аспектов или переходу к реализации!
