# System Patterns: Repository Analyzer

## 🏗️ Архитектурные паттерны

### 1. Модульная архитектура (Modular Architecture)
**Применение**: Разделение системы на независимые, взаимозаменяемые модули
**Реализация**:
```python
# Каждый модуль имеет четкую ответственность
main.py          # Координация и CLI
file_scanner.py  # Сканирование файлов
openai_integration.py # API взаимодействие
doc_generator.py # Генерация документации
```

**Преимущества**:
- Простота тестирования каждого компонента
- Легкое добавление новой функциональности
- Независимые обновления модулей

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
