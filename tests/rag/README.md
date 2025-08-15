# Тесты RAG системы

Комплексная система тестирования для RAG (Retrieval-Augmented Generation) компонентов.

## Структура тестов

```
tests/
├── test_rag_imports.py          # Базовые тесты импортов RAG
├── test_vector_store_basic.py   # Основные тесты векторного хранилища
├── test_cpu_query_engine.py     # Тесты поискового движка
├── test_simple_cpu_query_engine.py # Упрощённые тесты query engine
├── rag/                         # Комплексные тесты RAG
│   ├── __init__.py                 # Модуль тестов RAG
│   ├── conftest.py                 # Общие фикстуры и утилиты
│   ├── run_rag_tests.py           # Скрипт запуска тестов
│   ├── test_rag_integration.py    # Интеграционные тесты компонентов
│   ├── test_rag_e2e_cli.py       # End-to-End тесты CLI команд
│   ├── test_rag_performance.py   # Тесты производительности
│   └── README.md                  # Данная документация
└── fixtures/                    # Тестовые данные
    └── test_repo/
```

## Типы тестов

### 1. Базовые тесты компонентов (уровень `tests/`)

#### Базовые импорты (`test_rag_imports.py`)
Проверяют корректность импортов и базовую инициализацию:
- Импорт конфигурационных классов
- Импорт RAG модуля и проверка версии
- Импорт исключений RAG системы
- Инициализация CPUEmbedder
- Проверка заглушек VectorStore и QueryEngine
- Совместимость конфигурации

```bash
# Запуск базовых тестов импортов
pytest tests/test_rag_imports.py -v
```

#### Тесты векторного хранилища (`test_vector_store_basic.py`)
Проверяют основную функциональность QdrantVectorStore:
- Импорты qdrant-client и векторного хранилища
- Создание конфигурации VectorStoreConfig
- Инициализация QdrantVectorStore
- Генерация конфигурации коллекции (с разными типами квантования)
- Валидация точек и векторов

```bash
# Запуск тестов векторного хранилища
pytest tests/test_vector_store_basic.py -v
```

#### Тесты поискового движка (`test_cpu_query_engine.py`)
Проверяют интеграцию CPUQueryEngine:
- Импорты всех RAG компонентов
- Инициализация CPUQueryEngine с реальными компонентами
- Тестирование базовой функциональности (статистика, кэш, health check)
- Проверка совместимости конфигурации

```bash
# Запуск тестов поискового движка
pytest tests/test_cpu_query_engine.py -v
```

#### Упрощённые тесты (`test_simple_cpu_query_engine.py`)
Легковесные тесты CPUQueryEngine с заглушками:
- Проверка импортов компонентов
- Загрузка конфигурации
- Создание CPUEmbedder
- Работа с MockVectorStore
- Асинхронные операции

```bash
# Запуск упрощённых тестов
pytest tests/test_simple_cpu_query_engine.py -v
```

### 2. Интеграционные тесты (`tests/rag/test_rag_integration.py`)

Проверяют взаимодействие RAG компонентов:
- **CPUEmbedder** - генерация эмбеддингов с FastEmbed/SentenceTransformers
- **QdrantVectorStore** - векторное хранилище с квантованием
- **IndexerService** - сервис индексации репозиториев  
- **SearchService** - семантический поиск
- **CPUQueryEngine** - полноценный поисковый движок с RRF и MMR

```bash
# Запуск интеграционных тестов
pytest tests/rag/test_rag_integration.py -v
```

### 2. End-to-End тесты CLI (`test_rag_e2e_cli.py`)

Тестируют CLI команды в реальных условиях:
- `rag index` - индексация репозитория
- `rag search` - семантический поиск по коду
- `rag status` - статус RAG системы

```bash  
# Запуск E2E тестов
pytest tests/rag/test_rag_e2e_cli.py -v
```

### 3. Тесты производительности (`test_rag_performance.py`)

Измеряют производительность системы:
- Скорость индексации (файлов/сек)
- Время отклика поиска (латентность p95)
- Использование памяти
- Пропускную способность (запросов/сек)
- Конкурентную обработку (до 20 пользователей)

```bash
# Быстрые тесты производительности
pytest tests/rag/test_rag_performance.py -v -m "not slow and not stress"

# Полные тесты производительности (медленные)
pytest tests/rag/test_rag_performance.py -v

# Только стресс-тесты
pytest tests/rag/test_rag_performance.py -v -m "stress"
```

## Запуск тестов

### Через скрипт run_rag_tests.py

```bash
# Дымовые тесты (быстрая проверка)
python tests/rag/run_rag_tests.py smoke

# Unit тесты
python tests/rag/run_rag_tests.py unit

# Интеграционные тесты
python tests/rag/run_rag_tests.py integration

# E2E тесты
python tests/rag/run_rag_tests.py e2e

# Тесты производительности
python tests/rag/run_rag_tests.py performance

# Стресс-тесты
python tests/rag/run_rag_tests.py stress

# Все тесты
python tests/rag/run_rag_tests.py all
```

### Через pytest напрямую

```bash
# Все тесты RAG системы
pytest tests/test_*rag*.py tests/test_*cpu*.py tests/test_*vector*.py tests/rag/ -v

# Только базовые компонентные тесты
pytest tests/test_rag_imports.py tests/test_vector_store_basic.py tests/test_cpu_query_engine.py tests/test_simple_cpu_query_engine.py -v

# Только комплексные RAG тесты
pytest tests/rag/ -v

# Только быстрые тесты
pytest tests/rag/ -v -m "not slow and not stress"

# Тесты с mock объектами (без внешних зависимостей)
pytest tests/rag/ -v -m "mock"

# Тесты с реальными компонентами (требует Qdrant)
pytest tests/rag/ -v -m "real"

# Конкретный тест
pytest tests/rag/test_rag_integration.py::TestRAGIntegration::test_full_rag_pipeline -v
```

## Маркеры тестов

- `@pytest.mark.rag` - все RAG тесты
- `@pytest.mark.unit` - быстрые unit тесты
- `@pytest.mark.integration_rag` - интеграционные тесты RAG
- `@pytest.mark.e2e_rag` - End-to-End тесты RAG
- `@pytest.mark.perf_rag` - тесты производительности RAG
- `@pytest.mark.slow` - медленные тесты (>5 секунд)
- `@pytest.mark.stress` - стресс-тесты и нагрузочное тестирование
- `@pytest.mark.benchmark` - бенчмарки производительности
- `@pytest.mark.mock` - тесты с mock объектами
- `@pytest.mark.real` - тесты с реальными компонентами

## Требования для тестов

### Минимальные (mock тесты)
- Python 3.8+
- pytest
- numpy
- unittest.mock

### Полные (real тесты)
- Все минимальные требования
- Запущенный Qdrant на localhost:6333
- FastEmbed или Sentence Transformers
- Доступ к интернету для загрузки моделей

## Конфигурация

Тесты используют изолированную конфигурацию через фикстуры:
- `test_rag_config` - базовая конфигурация
- `minimal_rag_config` - минимальная конфигурация для быстрых тестов
- `test_rag_settings_file` - временный файл настроек

## Тестовые данные

### Тестовый репозиторий
Расположен в `tests/fixtures/test_repo/` со структурой:
```
test_repo/
├── auth/
│   ├── middleware.py    # Аутентификация, JWT токены
│   └── user.py         # Управление пользователями
├── db/
│   ├── models.py       # SQLAlchemy модели
│   └── connection.py   # Подключение к БД
└── utils/
    ├── helpers.py      # Утилиты, форматирование
    └── validators.py   # Валидаторы данных
```

### Типы контента для тестирования
- Python функции и классы
- JavaScript/TypeScript код
- SQL запросы
- Документация и комментарии
- Различные размеры файлов и чанков

## Метрики производительности

### Целевые показатели
- **Индексация**: >10 файлов/сек
- **Поиск**: <200ms латентность p95
- **Память**: <500MB для 1000 документов  
- **Пропускная способность**: >20 запросов/сек
- **Конкурентность**: 20 одновременных пользователей

### Измеряемые метрики
- Время выполнения операций
- Пропускная способность (операций/сек)
- Потребление памяти (пиковое и среднее)
- Загрузка CPU
- Количество обработанных элементов
- Процент ошибок

## Отчеты

После выполнения тестов создаются отчеты:
- Сводка успешности всех тестов
- Метрики производительности
- Детализация ошибок
- Рекомендации по оптимизации

## Устранение неполадок

### Ошибки подключения к Qdrant
```bash
# Запуск Qdrant в Docker
docker run -p 6333:6333 qdrant/qdrant

# Проверка доступности
curl http://localhost:6333/health
```

### Ошибки установки FastEmbed
```bash
pip install fastembed
# или
pip install sentence-transformers
```

### Недостаток памяти
Уменьшите размеры батчей в конфигурации:
```json
{
  "rag": {
    "embeddings": {
      "batch_size_max": 32
    }
  }
}
```

## Примеры использования

### Быстрая проверка системы
```bash
python tests/rag/run_rag_tests.py smoke
```

### Полное тестирование перед релизом
```bash
python tests/rag/run_rag_tests.py all
```

### Тестирование конкретного компонента
```bash
pytest tests/rag/test_rag_integration.py::TestRAGIntegration::test_embedder_initialization -v
```

### Профилирование производительности
```bash
pytest tests/rag/test_rag_performance.py -v --durations=10