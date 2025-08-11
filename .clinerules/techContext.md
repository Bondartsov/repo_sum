# Technical Context: Repository Analyzer

## 🏗️ Технологический стек

### Основные технологии:
- **Python 3.8+**: Основной язык разработки
- **OpenAI GPT API**: ИИ-анализ кода (gpt-4o-mini по умолчанию)
- **Qdrant**: Векторная база данных для RAG системы
- **CodeBERT/GraphCodeBERT**: Специализированные эмбединги для кода (бесплатно)
- **Streamlit**: Веб-интерфейс для пользователей + Chat UI
- **AST (Abstract Syntax Tree)**: Парсинг кода всех языков
- **Pathlib**: Работа с файловой системой
- **JSON**: Конфигурация и кэширование
- **Markdown**: Формат выходной документации

### Внешние зависимости:
```
# Core dependencies
openai>=1.0.0          # OpenAI API клиент
streamlit>=1.28.0      # Веб-интерфейс
python-dotenv>=1.0.0   # Управление переменными окружения
pytest>=7.4.0          # Тестирование
pathspec>=0.11.0       # Git-style ignore паттерны

# RAG System dependencies  
qdrant-client>=1.8.0   # Qdrant векторная БД
transformers>=4.35.0   # Hugging Face трансформеры
torch>=2.0.0          # PyTorch для эмбедингов
sentence-transformers>=2.2.0  # Sentence embeddings
numpy>=1.24.0         # Векторные операции
faiss-cpu>=1.7.4      # Опциональный fallback для векторного поиска
```

## 🏛️ Архитектурные принципы

### Модульная архитектура:
```
repo_sum/
├── main.py                 # CLI интерфейс и основной координатор
├── web_ui.py              # Streamlit веб-интерфейс с Chat UI
├── config.py              # Централизованная конфигурация
├── file_scanner.py        # Сканирование и фильтрация файлов
├── openai_integration.py  # OpenAI API с кэшированием и retry
├── doc_generator.py       # Генерация Markdown отчетов
├── code_chunker.py        # Разбивка кода на логические части
├── utils.py               # Утилиты и структуры данных
├── parsers/               # Парсеры для разных языков
│   ├── base_parser.py     # Базовый интерфейс парсера
│   ├── python_parser.py   # Python AST парсер
│   └── [language]_parser.py
└── rag/                   # 🔥 RAG система (новый модуль)
    ├── __init__.py        # RAG модуль инициализация
    ├── vector_store.py    # Qdrant интеграция и векторное хранилище
    ├── code_embedder.py   # CodeBERT/GraphCodeBERT эмбединги
    ├── semantic_chunker.py # Микро-сегментация кода (AST-based)
    ├── query_engine.py    # Семантический поиск и ранжирование
    ├── chat_interface.py  # Диалоговый интерфейс для кода
    ├── code_context.py    # Управление контекстом кода
    └── rag_config.py      # Специальная конфигурация RAG
```

### Принципы проектирования:
1. **Separation of Concerns**: Каждый модуль отвечает за одну область
2. **Plugin Architecture**: Расширяемая система парсеров
3. **Configuration-driven**: Все настройки выведены в settings.json
4. **Fail-safe**: Graceful degradation при ошибках API
5. **Resource Management**: Кэширование и оптимизация вызовов API

## ⚙️ Ключевые компоненты

### 1. RepositoryAnalyzer (main.py)
**Назначение**: Основной координатор анализа
**Функции**:
- Оркестрация всего процесса анализа
- CLI интерфейс с argparse
- Инкрементальный анализ с индексированием
- Логирование и error handling

### 2. FileScanner (file_scanner.py)
**Назначение**: Поиск и фильтрация файлов для анализа
**Функции**:
- Рекурсивное сканирование директорий
- Фильтрация по расширениям и exclude-спискам
- Валидация размера файлов
- Git-style ignore patterns

### 3. ParserRegistry (parsers/)
**Назначение**: Выбор подходящего парсера по типу файла
**Поддерживаемые языки**:
```python
SUPPORTED_EXTENSIONS = {
    ".py": "python",     # Python AST парсер
    ".js": "javascript", # JavaScript парсер  
    ".ts": "typescript", # TypeScript парсер
    ".java": "java",     # Java парсер
    ".cpp": "cpp",       # C++ парсер
    ".cs": "csharp",     # C# парсер
    ".go": "go",         # Go парсер
    ".rs": "rust",       # Rust парсер
    ".php": "php",       # PHP парсер
    ".rb": "ruby"        # Ruby парсер
}
```

### 4. CodeChunker (code_chunker.py)
**Назначение**: Интеллектуальная разбивка больших файлов
**Стратегии**:
- **logical**: По функциям/классам/модулям
- **size**: По размеру в токенах
- **lines**: По количеству строк

### 5. OpenAIManager (openai_integration.py)
**Назначение**: Управление взаимодействием с OpenAI API
**Возможности**:
- Батчевая обработка файлов
- Кэширование результатов (JSON файлы)
- Retry механизм с экспоненциальным backoff
- Санитайзинг секретов перед отправкой
- Подсчет и оптимизация токенов

### 6. DocumentationGenerator (doc_generator.py)
**Назначение**: Создание финальных Markdown отчетов
**Функции**:
- Генерация структурированных отчетов по файлам
- Создание индексного файла README.md
- Сохранение иерархии директорий
- Шаблонизация отчетов

### 7. 🔥 RAG System (rag/ модуль) - НОВЫЙ КРИТИЧЕСКИЙ КОМПОНЕНТ

#### 7.1 VectorStore (rag/vector_store.py)
**Назначение**: Управление Qdrant векторной базой данных
**Функции**:
- Подключение к Qdrant (локальный/облачный)
- Создание коллекций для разных проектов
- Bulk индексация микро-чанков кода
- Семантический поиск по векторам
- Метаданные и фильтрация результатов

#### 7.2 CodeEmbedder (rag/code_embedder.py) 
**Назначение**: Создание векторных представлений кода
**Модели (бесплатные)**:
- **microsoft/codebert-base**: Универсальные эмбединги для кода
- **microsoft/graphcodebert-base**: Граф-aware эмбединги
- **Salesforce/codet5p-220m**: Компактная модель для кода
**Функции**:
- Загрузка предобученных моделей
- Батчевое создание эмбедингов
- Нормализация и оптимизация векторов
- Кэширование эмбедингов

#### 7.3 SemanticChunker (rag/semantic_chunker.py)
**Назначение**: Микро-сегментация кода до атомарного уровня
**Подход**:
- AST-based разбиение до statement/expression уровня
- Сохранение иерархии: file → class → method → statement
- Метаданные для каждого чанка (тип, родитель, позиция)
- Overlap стратегия для сохранения контекста
**Типы чанков**:
- Function definitions
- Class definitions  
- Import statements
- Variable assignments
- Control flow blocks
- Individual expressions

#### 7.4 QueryEngine (rag/query_engine.py)
**Назначение**: Семантический поиск и ранжирование фрагментов
**Алгоритм**:
- Векторизация пользовательского запроса
- Поиск ближайших соседей в Qdrant
- Ре-ранжирование по релевантности и контексту
- Комбинирование связанных фрагментов
- Формирование контекста для LLM

#### 7.5 ChatInterface (rag/chat_interface.py)
**Назначение**: Диалоговый интерфейс для взаимодействия с кодом
**Интеграция**: Streamlit chat компонент
**Возможности**:
- "Объясни эту функцию [имя]"
- "Найди все места использования переменной X"
- "Как связаны модули A и B?"
- "Предложи рефакторинг для этого кода"
- "Найди похожие паттерны в проекте"
- История диалога с контекстом

#### 7.6 CodeContext (rag/code_context.py)
**Назначение**: Управление контекстом и связями между фрагментами
**Функции**:
- Построение граф связей между компонентами
- Трекинг зависимостей и импортов
- Иерархическая навигация по коду
- Контекстно-зависимый поиск

## 🔧 Конфигурационная система

### Структура settings.json:
```json
{
  "openai": {
    "max_tokens_per_chunk": 4000,
    "max_response_tokens": 5000, 
    "temperature": 0.1,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "analysis": {
    "chunk_strategy": "logical",
    "min_chunk_size": 100,
    "languages_priority": ["python", "javascript", "java"],
    "sanitize_enabled": false,
    "sanitize_patterns": [
      "(?i)api_key\\s*[:=]\\s*['\"][^'\"]+['\"]"
    ]
  },
  "file_scanner": {
    "max_file_size": 10485760,
    "excluded_directories": [".git", "node_modules"],
    "supported_extensions": {...}
  },
  "output": {
    "default_output_dir": "./docs",
    "file_template": "minimal_file.md"
  },
  "rag": {
    "qdrant": {
      "host": "localhost",
      "port": 6333,
      "collection_name": "code_chunks",
      "vector_size": 768,
      "distance_metric": "cosine"
    },
    "embeddings": {
      "model_name": "microsoft/codebert-base",
      "batch_size": 32,
      "max_length": 512,
      "cache_embeddings": true
    },
    "chunking": {
      "strategy": "semantic_ast",
      "max_chunk_size": 200,
      "min_chunk_size": 20,
      "overlap_tokens": 50,
      "preserve_hierarchy": true
    },
    "search": {
      "top_k": 10,
      "score_threshold": 0.7,
      "rerank_enabled": true,
      "context_window": 5
    },
    "chat": {
      "max_context_chunks": 20,
      "response_max_tokens": 2000,
      "history_length": 10
    }
  }
}
```

## 🚀 Производительность и масштабируемость

### Оптимизации:
1. **Адаптивная батчевая обработка**:
   - Малые проекты (≤10 файлов): батчи по 2 файла
   - Средние проекты (11-50 файлов): батчи по 3 файла
   - Большие проекты (51-200 файлов): батчи по 5 файлов
   - Очень большие (200+ файлов): батчи по 8 файлов

2. **Умное кэширование**:
   - Хеширование содержимого файлов
   - Кэш с истечением срока (7 дней)
   - Инвалидация при изменении файла

3. **Инкрементальный анализ**:
   - Индексный файл `./.repo_sum/index.json`
   - Анализ только измененных файлов
   - Сравнение по hash и timestamp

### Ограничения:
- **Максимальный размер файла**: 10MB
- **Token limit**: 4000 токенов на chunk
- **API rate limits**: Зависят от плана OpenAI
- **Memory usage**: ~100MB для проекта из 1000 файлов

## 🔒 Безопасность и конфиденциальность

### Защитные меры:
1. **API ключи**: Только через переменные окружения
2. **Санитайзинг секретов**: Маскировка перед отправкой в LLM
3. **Path traversal protection**: Валидация путей файлов
4. **File size limits**: Защита от DoS атак
5. **Логирование**: Только метаданные без содержимого

### Regex паттерны для санитайзинга:
```python
SANITIZE_PATTERNS = [
    r"(?i)api_key\s*[:=]\s*['\"][^'\"]+['\"]",
    r"(?i)password\s*[:=]\s*['\"][^'\"]+['\"]",  
    r"(?i)secret\s*[:=]\s*['\"][^'\"]+['\"]",
    r"(?i)token\s*[:=]\s*['\"][^'\"]+['\"]"
]
```

## 🧪 Тестирование

### Покрытие тестами:
```
tests/
├── test_config.py              # Тестирование конфигурации
├── test_file_scanner.py        # Тестирование сканера файлов  
├── test_code_chunker.py        # Тестирование chunker'а
├── test_openai_integration.py  # Mock тесты OpenAI API
├── test_doc_generator.py       # Тестирование генератора документации
├── test_parsers.py            # Тестирование парсеров
├── test_integration_full_cycle.py # Интеграционные тесты
└── test_property_based.py      # Property-based тесты
```

### Типы тестов:
- **Unit tests**: Изолированное тестирование компонентов
- **Integration tests**: Полный цикл анализа
- **Mock tests**: Тестирование без реальных API вызовов
- **Property-based tests**: Fuzz testing с Hypothesis

## 📈 Мониторинг и метрики

### Логируемые метрики:
- Время анализа файла/проекта
- Количество токенов на запрос/ответ
- Размер батчей и эффективность
- Частота кэш-попаданий
- Ошибки API и retry статистика
- Размер выходной документации

### Уровни логирования:
- **ERROR**: Критические ошибки, прерывающие работу
- **WARNING**: Проблемы, не блокирующие выполнение
- **INFO**: Основные события (начало/конец анализа)
- **DEBUG**: Детальная информация для разработки

## 🔄 CI/CD и развертывание

### Поддерживаемые платформы:
- **Windows**: Основная платформа разработки
- **Linux**: Серверное развертывание  
- **macOS**: Desktop использование
- **Docker**: Контейнеризация (планируется)

### Развертывание:
1. **Локальное**: `pip install -r requirements.txt`
2. **Веб-сервер**: Streamlit на порту 8501
3. **Cloud**: VPS с публичным доступом
4. **Systemd**: Автозапуск через service unit
