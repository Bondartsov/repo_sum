# System Patterns: Repository Analyzer

## 🏗️ Архитектурные паттерны

### 1. Модульная архитектура (Modular Architecture)
**Применение**: Разделение системы на независимые, взаимозаменяемые модули
**Реализация**:
```python
# Core система
main.py          # Координация и CLI
file_scanner.py  # Сканирование файлов
openai_integration.py # API взаимодействие
doc_generator.py # Генерация документации

# 🔥 RAG система (новый модуль)
embedder.py      # CPU-оптимизированный эмбеддер
vector_store.py  # Qdrant интеграция
query_engine.py  # Гибридный поиск
```

**Преимущества**:
- Простота тестирования каждого компонента
- Легкое добавление новой функциональности
- Независимые обновления модулей
- Изоляция RAG компонентов от основной системы

### 2. Plugin Architecture для парсеров
**Применение**: Расширяемая система парсеров языков программирования
**Реализация**:
```python
# parsers/base_parser.py - базовый интерфейс
class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> ParsedData

# parsers/python_parser.py - конкретная реализация
class PythonParser(BaseParser):
    def parse(self, content: str) -> ParsedData:
        # Python-специфичная логика
```

**Преимущества**:
- Легкое добавление новых языков
- Унифицированный интерфейс
- Изоляция языко-специфичной логики

### 3. Configuration-Driven Development
**Применение**: Централизованное управление настройками через JSON
**Реализация**:
```python
# settings.json - единый источник конфигурации
# config.py - типизированная загрузка настроек
@dataclass
class Config:
    openai: OpenAIConfig
    analysis: AnalysisConfig
    file_scanner: FileScannerConfig
```

**Преимущества**:
- Изменение поведения без изменения кода
- Типизированная валидация настроек
- Централизованное управление параметрами

### 4. Strategy Pattern для chunking
**Применение**: Различные стратегии разбивки кода на части
**Реализация**:
```python
# code_chunker.py
class ChunkStrategy:
    LOGICAL = "logical"    # По функциям/классам
    SIZE = "size"         # По размеру в токенах  
    LINES = "lines"       # По количеству строк
```

**Преимущества**:
- Гибкость выбора подходящей стратегии
- Легкое добавление новых стратегий
- Адаптация под разные типы проектов

## 🔄 Паттерны обработки данных

### 1. Pipeline Pattern
**Применение**: Последовательная обработка данных через этапы
**Реализация**:
```python
# Основной pipeline в main.py
scan_files() → chunk_code() → analyze_with_gpt() → generate_docs()
```

**Этапы пайплайна**:
1. **File Discovery**: Сканирование и фильтрация файлов
2. **Code Parsing**: Извлечение структуры кода
3. **Chunking**: Разбивка на логические части
4. **AI Analysis**: Анализ через OpenAI API
5. **Documentation**: Генерация Markdown отчетов

### 2. Batch Processing Pattern
**Применение**: Группировка файлов для эффективного API использования
**Реализация**:
```python
# Адаптивные размеры батчей
def get_batch_size(file_count: int) -> int:
    if file_count <= 10: return 2
    elif file_count <= 50: return 3
    elif file_count <= 200: return 5
    else: return 8
```

**Преимущества**:
- Оптимизация API вызовов
- Снижение стоимости анализа
- Лучшая производительность

### 3. Caching Pattern
**Применение**: Кэширование результатов анализа по hash содержимого
**Реализация**:
```python
# cache/
├── file_hash_1.json  # Кэшированный анализ файла
├── file_hash_2.json
└── metadata.json     # Метаданные кэша
```

**Механизм**:
- Hash содержимого файла как ключ кэша
- TTL (Time To Live) для автоочистки
- Инвалидация при изменении файла

### 4. Search Result Caching (TTL) — RAG
**Применение**: Кэширование результатов поисковых запросов (dense/hybrid) для популярных запросов.
**Реализация**:
- LRU/TTL через `cachetools`; ключ — нормализованный текст запроса + флаги (`use_hybrid`, `top_k`, фильтры).
- Инвалидация по TTL и принудительная инвалидация при переиндексации.
```python
from cachetools import TTLCache
search_cache = TTLCache(maxsize=1000, ttl=300)

def cached_search(key, compute):
    if key in search_cache:
        return search_cache[key]
    res = compute()
    search_cache[key] = res
    return res
```

### 5. Reciprocal Rank Fusion (RRF) Pattern
**Применение**: Фьюжн dense и sparse выдач для повышения точности.
**Реализация**:
```python
from collections import defaultdict

def rrf(lists, k=60):
    fused = defaultdict(float)
    for lst in lists:  # lst: [(id, score), ...] в порядке ранга
        for rank, (pid, _) in enumerate(lst, start=1):
            fused[pid] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
```

### 6. MMR Re-ranking Pattern
**Применение**: Диверсификация результатов и борьба с дубликатами.
**Реализация**:
```python
import numpy as np

def mmr(query_vec, cand_vecs, lambda_=0.7, top_k=10):
    selected, remaining = [], list(range(len(cand_vecs)))
    sims = lambda a,b: float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
    while remaining and len(selected) < top_k:
        best, best_score = None, -1e9
        for i in remaining:
            rel = sims(query_vec, cand_vecs[i])
            div = 0.0 if not selected else max(sims(cand_vecs[i], cand_vecs[j]) for j in selected)
            score = lambda_*rel - (1-lambda_)*div
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        remaining.remove(best)
    return selected
```

## 🔒 Паттерны безопасности

### 1. Sanitization Pattern
**Применение**: Очистка чувствительных данных перед отправкой в LLM
**Реализация**:
```python
# Regex паттерны для маскировки секретов
SANITIZE_PATTERNS = [
    r"(?i)api_key\s*[:=]\s*['\"][^'\"]+['\"]",
    r"(?i)password\s*[:=]\s*['\"][^'\"]+['\"]"
]

def sanitize_code(content: str) -> str:
    # Замена секретов на [MASKED]
```

### 2. Environment-based Configuration
**Применение**: API ключи только через переменные окружения
**Реализация**:
```python
# .env файл для локальной разработки
OPENAI_API_KEY=sk-...

# config.py - загрузка из environment
@property
def api_key(self) -> str:
    return os.getenv(self.api_key_env_var)
```

### 3. Validation Pattern
**Применение**: Валидация всех входных данных
**Реализация**:
```python
# Валидация размера файлов
if file.stat().st_size > self.config.max_file_size:
    raise ValueError("File too large")

# Валидация путей файлов (path traversal protection)
if ".." in str(path) or not path.is_relative_to(base_path):
    raise SecurityError("Unsafe path")
```

## 🚀 Паттерны производительности

### 1. Lazy Loading Pattern
**Применение**: Загрузка парсеров только при необходимости
**Реализация**:
```python
# parsers/__init__.py
class ParserRegistry:
    def get_parser(self, language: str) -> BaseParser:
        if language not in self._loaded_parsers:
            self._loaded_parsers[language] = self._load_parser(language)
        return self._loaded_parsers[language]
```

### 2. Resource Pooling Pattern
**Применение**: Переиспользование OpenAI клиента
**Реализация**:
```python
# openai_integration.py
class OpenAIManager:
    def __init__(self):
        self._client = OpenAI()  # Единый клиент
        self._session_cache = {}  # Кэш сессий
```

### 3. Progress Tracking Pattern
**Применение**: Отслеживание прогресса для длительных операций
**Реализация**:
```python
# Streamlit progress bars
progress_bar = st.progress(0)
for i, batch in enumerate(batches):
    process_batch(batch)
    progress_bar.progress((i + 1) / len(batches))
```

### 4. Adaptive Batch Encoding Pattern
**Применение**: Адаптивный размер батча эмбеддингов с учётом свободной RAM и длины очереди.
**Реализация**:
```python
import psutil

def calc_batch_size(q_len, cfg):
    avail = psutil.virtual_memory().available
    # Сервер с большим RAM → стремимся к верхней границе
    if avail > 8 * 1024**3:
        return min(cfg.batch_size_max, max(16, q_len))
    # Дефолтная эвристика
    return max(cfg.batch_size_min, min(cfg.batch_size_max, max(cfg.batch_size_min, q_len // 2)))
```

### 5. Parallelism Threads Configuration Pattern
**Применение**: Управление потоками для CPU‑производительности.
**Реализация**:
```python
import os, torch
def configure_threads(par):
    torch.set_num_threads(par.torch_num_threads)
    os.environ["OMP_NUM_THREADS"] = str(par.omp_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(par.mkl_num_threads)
```

## 🔄 Паттерны интеграции

### 1. Adapter Pattern для UI
**Применение**: Единый backend с множественными интерфейсами
**Реализация**:
```python
# main.py - CLI интерфейс
class RepositoryAnalyzer:
    def analyze(self, path: Path) -> AnalysisResult

# web_ui.py - Web интерфейс (адаптер)
def run_analysis():
    analyzer = RepositoryAnalyzer()
    result = analyzer.analyze(path)  # Тот же backend
```

### 2. Observer Pattern для логирования
**Применение**: Множественные обработчики событий анализа
**Реализация**:
```python
# Логирование в файл + консоль + progress bar
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.addHandler(progress_handler)
```

### 3. Factory Pattern для создания отчетов
**Применение**: Создание отчетов разных форматов
**Реализация**:
```python
# doc_generator.py
class ReportFactory:
    @staticmethod
    def create_report(format: str) -> BaseReportGenerator:
        if format == "markdown":
            return MarkdownReportGenerator()
        elif format == "html":
            return HTMLReportGenerator()  # Планируется
```

### 4. CLI Commands Pattern (RAG)
**Применение**: Единый UX для индексации/поиска/анализа с контекстом.
**Команды**:
```bash
python main.py index /path/to/repo
python main.py search "find auth tokens" -k 10 --hybrid
python main.py analyze-with-rag /path/to/repo -o ./docs
```

## 📊 Паттерны мониторинга

### 1. Metrics Collection Pattern
**Применение**: Сбор метрик производительности
**Реализация**:
```python
# utils.py
@dataclass
class AnalysisMetrics:
    files_processed: int
    tokens_used: int
    api_calls_made: int
    cache_hits: int
    total_time: float
```

### 2. Structured Logging Pattern
**Применение**: Структурированные логи для анализа
**Реализация**:
```python
logger.info("Analysis completed", extra={
    "files_count": len(files),
    "total_tokens": metrics.tokens_used,
    "duration_sec": metrics.total_time
})
```

### 3. Vector DB Monitoring Pattern (Qdrant)
**Применение**: Мониторинг производительности и ошибок векторного поиска.
**Реализация**:
- Метрики: количество точек/сегментов, latency поиска, error rates.
- Prometheus экспонирование гистограмм и счётчиков.
```python
from prometheus_client import Counter, Histogram
qdrant_requests_total = Counter("qdrant_requests_total", "Qdrant requests", ["op"])
qdrant_search_latency = Histogram("qdrant_search_latency_seconds", "Search latency")
```

### 4. Alerting Pattern
**Применение**: Алерты на превышение латентности/ошибок.
**Реализация**: Правила в Prometheus Alertmanager; пороги SLA по p95/p99.

## 🔧 Принципы качества кода

### 1. SOLID Principles
- **SRP**: Каждый класс имеет единую ответственность
- **OCP**: Расширяемость через наследование (парсеры)
- **LSP**: Взаимозаменяемость парсеров
- **ISP**: Интерфейсы под конкретные потребности
- **DIP**: Зависимость от абстракций, не реализаций

### 2. DRY (Don't Repeat Yourself)
- Общие утилиты в `utils.py`
- Базовые классы для парсеров
- Переиспользование конфигурации

### 3. Explicit is better than implicit
- Типизация через `dataclasses` и `typing`
- Явная валидация входных данных
- Четкие имена переменных и функций

### 4. Fail-fast principle
- Валидация конфигурации при старте
- Проверка API ключей перед анализом
- Немедленное прерывание при критических ошибках
