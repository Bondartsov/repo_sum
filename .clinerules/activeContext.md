# Active Context: Repository Analyzer с RAG-системой

> Обновление 2025-08-11: Синхронизация с RAG_progress.md. Ключевые изменения:
> - Обновлён стек: openai>=1.99.6, sentence-transformers~=5.1.0 (precision='int8'), torch>=2.4.0 (CPU), qdrant-client>=1.15.1, faiss-cpu, numpy, psutil, cachetools, fastapi/uvicorn (опц.).
> - Новые конфиги в config.py: EmbeddingConfig, VectorStoreConfig, QueryEngineConfig, ParallelismConfig.
> - План модулей rag/: embedder.py (CPUEmbedder + warmup + adaptive batch), vector_store.py (Qdrant HNSW m=24/ef=128, SQ/PQ, репликация), query_engine.py (hybrid search + RRF + MMR + TTL-кэш).
> - Интеграция: новые CLI команды index/search/analyze-with-rag; Web UI вкладка «Поиск»; адаптация промптов под retrieved context; расширение GPTAnalysisRequest контекстом.
> - Тестирование: unit для rag/*, e2e индексация→поиск→анализ, нагрузка на 20 пользователей, метрики MRR@k/Recall@k.

## 🎯 Текущий фокус разработки

### Основное направление разработки:
Проект активно реализует **продвинутую RAG-систему** с CPU-оптимизированным векторным поиском, гибридными алгоритмами и enterprise-готовой архитектурой для анализа и поиска по кодовой базе.

### Приоритетные задачи (текущий спринт):

#### 1. 🔥 Продвинутая RAG система (АКТИВНАЯ РЕАЛИЗАЦИЯ)
- **Статус**: Активная разработка с CPU-оптимизацией
- **Цель**: Enterprise-готовая система поиска и анализа кода
- **Ключевые компоненты**:
  - `embedder.py` - CPU-оптимизированный эмбеддер с FastEmbed (ONNX)
  - `vector_store.py` - Qdrant интеграция с квантованием и репликацией
  - `query_engine.py` - гибридный поиск (dense + sparse), MMR переранжирование
  - Обновленный `config.py` - расширенная конфигурация для RAG
  - Интеграция в существующий поток анализа
- **Архитектура** ✅ ФИНАЛИЗИРОВАНА:
  - **CPU-first подход**: FastEmbed с BAAI/bge-small-en-v1.5 (384d), ONNX Runtime
  - **Qdrant кластер**: m=24, ef_construct=128, ScalarQuantization, replication_factor=2
  - **Гибридный поиск**: dense + sparse (BM25/SPLADE) + RRF + MMR
  - **SLO**: hot cache <200ms, cold search p50 <500ms, p95 <1500ms
  - **Production-ready**: мониторинг, безопасность, масштабирование до 20 пользователей

#### 2. 🚀 Обновление технологического стека
- **Статус**: В процессе
- **Цель**: Современные зависимости для RAG
- **Компоненты**:
  - `requirements.txt` - openai>=1.95.0, sentence-transformers~=5.1.0, qdrant-client>=1.15.0
  - `torch>=2.7.0` (CPU), `faiss-cpu`, `psutil`, `cachetools`
  - **Новые конфиги**: EmbeddingConfig, VectorStoreConfig, QueryEngineConfig

#### 3. 📊 Индексация и переиндексация
- **Статус**: Проектирование
- **Цель**: Эффективная индексация кодовой базы
- **Возможности**:
  - Автоматическая индексация после коммитов
  - Инкрементальная переиндексация по хешам файлов
  - Планирование задач (cron, CI/CD, по требованию)
  - Контроль качества индекса (recall@k, MRR)

#### 4. 💡 Интеграция в существующий workflow
- **Статус**: Планирование
- **Цель**: RAG-enhanced анализ кода
- **Подход**:
  - Адаптация промптов с учетом retrieved context
  - Модификация OpenAIManager для работы с context_chunks
  - Новые CLI команды: index, search, analyze-with-rag
  - Обновление Streamlit UI с поисковой страницей

## 🔧 Активные технические задачи

### CPU-оптимизированная RAG система:
1. **CPUEmbedder с умной батчевой обработкой**
   - 🔄 В разработке: Ленивая инициализация модели с precision='int8'
   - 🔄 В разработке: calculate_batch_size() с учетом свободной RAM (psutil)
   - 🔄 В разработке: embed_texts() с контролем времени отклика (<1-2 сек)
   - 📋 План: warmup() для предварительного прогрева модели

2. **Qdrant векторное хранилище**
   - 🔄 В разработке: Конфигурация с m=24, ef_construct=128
   - 🔄 В разработке: ScalarQuantization для экономии памяти
   - 🔄 В разработке: replication_factor=2, write_consistency_factor=1-2
   - 📋 План: Операции index_documents(), update_document(), search()

3. **Гибридный поисковый движок**
   - 🔄 В разработке: Комбинация dense + sparse векторов
   - 🔄 В разработке: MMR переранжирование для разнообразия результатов
   - 🔄 В разработке: LRU кэш с TTL для горячих запросов
   - 📋 План: Параллельная обработка 20 пользователей через asyncio

### Мониторинг и надежность:
4. **Production-ready инфраструктура**
   - 📋 План: Prometheus/Grafana мониторинг Qdrant метрик
   - 📋 План: Docker-compose с контейнерами приложения, Qdrant, FastAPI
   - 📋 План: CI/CD pipeline с автотестами и деплойментом
   - 📋 План: TLS шифрование, контроль доступа, ротация API-ключей

5. **Тестирование и валидация**
   - 📋 План: Unit-тесты для embedder, vector_store, query_engine
   - 📋 План: Нагрузочное тестирование 20 параллельных пользователей
   - 📋 План: Метрики качества поиска (MRR@k, Recall@k)
   - 📋 План: Функциональные тесты с pytest-asyncio

## 🌐 Пользовательский интерфейс

### Расширенный Streamlit Web UI:
- **Статус**: Активное расширение для RAG
- **Файл**: `web_ui.py`
- **Текущие возможности**:
  - ✅ Drag & Drop загрузка файлов/архивов
  - ✅ Real-time прогресс анализа  
  - ✅ Валидация безопасности загружаемых файлов
  - ✅ Предварительный просмотр статистики проекта
- **Новые RAG возможности (в разработке)**:
  - 🔄 Поисковая страница с векторным поиском по коду
  - 🔄 Отображение retrieved фрагментов с ссылками на исходники
  - 🔄 Аналитика поиска (время ответа, количество найденных документов)
  - 🔄 RAG-enhanced анализ с контекстными фрагментами

### Расширенный CLI интерфейс:
- **Статус**: Расширение для RAG команд
- **Файл**: `main.py`
- **Существующие команды**: `analyze`, `stats`, `clear-cache`, `token-stats`
- **Новые RAG команды (в разработке)**:
  - 🔄 `index` - индексация репозитория в Qdrant
  - 🔄 `search` - поиск по векторному индексу
  - 🔄 `analyze-with-rag` - анализ с использованием retrieved context
  - 🔄 `reindex` - переиндексация измененных файлов

## 📝 Система документации и контекста

### RAG-enhanced архитектура отчетов:
- **Формат**: Markdown с retrieved контекстом
- **Шаблонизация**: Обновленные промпты в `prompts/code_analysis_prompt.md` с учетом RAG
- **Иерархия**: Сохранение структуры + векторные связи между компонентами
- **Индексирование**: Автогенерация README.md + векторный индекс в Qdrant

### Новые возможности (в разработке):
- 🔄 Retrieved context в промптах ("Изучите следующие фрагменты, затем проанализируйте...")
- 🔄 Семантические связи между файлами через векторный поиск
- 🔄 Контекстно-зависимые рекомендации по рефакторингу
- 📋 Интерактивный граф связей между компонентами кода

### Планируемые форматы:
- 📋 HTML экспорт с векторным поиском
- 📋 PDF с семантическими аннотациями
- 📋 Wiki-интеграция с поисковыми возможностями

## 🧪 Качество и тестирование

### Расширенное покрытие тестами:
```
tests/
├── test_config.py ✅           # Конфигурационная система
├── test_file_scanner.py ✅     # Сканирование файлов  
├── test_code_chunker.py ✅     # Chunking логика
├── test_openai_integration.py ✅ # Mock тесты API
├── test_doc_generator.py ✅    # Генерация документации
├── test_parsers.py ✅          # Языковые парсеры
├── test_integration_full_cycle.py ✅ # End-to-end тесты
└── rag_tests/                  # 🔄 Новые RAG тесты
    ├── test_embedder.py        # Тестирование эмбеддера
    ├── test_vector_store.py    # Qdrant операции
    ├── test_query_engine.py    # Поисковые алгоритмы
    ├── test_rag_integration.py # RAG интеграция
    └── test_load_testing.py    # Нагрузочные тесты
```

### RAG-специфичное тестирование:
- 🔄 **В разработке**: Unit-тесты для embedder, vector_store, query_engine
- 🔄 **В разработке**: Нагрузочное тестирование 20 параллельных пользователей
- 🔄 **В разработке**: Тестирование качества поиска (MRR@k, Recall@k)
- 🔄 **В разработке**: Функциональные тесты с pytest-asyncio

### Production-ready тестирование:
- 📋 **План**: Stress testing с реальными репозиториями
- 📋 **План**: A/B тестирование алгоритмов поиска
- 📋 **План**: Мониторинг quality drift при обновлении моделей

## 🔒 Безопасность и соответствие

### Enterprise-готовая безопасность:
- ✅ API ключи только через переменные окружения
- ✅ Path traversal protection при загрузке файлов  
- ✅ Валидация размеров файлов (защита от DoS)
- ✅ Санитайзинг секретов перед отправкой в LLM
- 🔄 **В разработке**: TLS шифрование между сервисами
- 🔄 **В разработке**: Контроль доступа (Firewall, VPN)
- 🔄 **В разработке**: Ротация API-ключей

### RAG-специфичная безопасность:
- 🔄 **В разработке**: Изоляция векторных данных по проектам
- 🔄 **В разработке**: Аудит поисковых запросов
- 🔄 **В разработке**: Контроль утечек через retrieved context
- 📋 **План**: Compliance с корпоративными политиками данных

### Enterprise соответствие:
- 📋 **План**: SOC2 Type II сертификация
- 📋 **План**: GDPR compliance для европейских клиентов
- 📋 **План**: Аудит третьих сторон (Qdrant, OpenAI)

## 🚀 Production Deployment и infrastructure

### Текущий статус развертывания:
- ✅ Локальное развертывание: pip install + streamlit
- ✅ Manual cloud deployment: VPS with systemd  
- 🔄 **В разработке**: Docker-compose с множественными сервисами
- 🔄 **В разработке**: CI/CD pipeline с автотестами

### Продвинутая архитектура:
- **Development**: Windows 11 + локальный Qdrant
- **Staging**: Docker-compose (приложение + Qdrant + FastAPI + Streamlit)
- **Production**: Kubernetes кластер с репликацией Qdrant
- **Enterprise**: Высокодоступный кластер с балансировкой нагрузки

### Container стратегия:
- 🔄 **В разработке**: Контейнер приложения с Python зависимостями
- 🔄 **В разработке**: Qdrant кластер с персистентными томами
- 🔄 **В разработке**: FastAPI контейнер для REST API
- 🔄 **В разработке**: Nginx reverse proxy с rate limiting

## 📊 Мониторинг и аналитика

### Production-ready мониторинг:
- ✅ Token usage tracking (daily limits)
- ✅ File processing statistics  
- ✅ API call success/failure rates
- ✅ Cache hit ratios
- 🔄 **В разработке**: Prometheus метрики для Qdrant
- 🔄 **В разработке**: Grafana дашборды для RAG системы
- 🔄 **В разработке**: Alerting на превышение латентности

### RAG-специфичные метрики:
- 🔄 **В разработке**: Поисковая латентность (цель <200ms)
- 🔄 **В разработке**: Качество поиска (MRR@k, Recall@k)
- 🔄 **В разработке**: Утилизация векторного индекса
- 🔄 **В разработке**: Throughput на 20 пользователей

### Бизнес-аналитика:
- 📋 **План**: Анонимная аналитика использования
- 📋 **План**: A/B тесты алгоритмов поиска  
- 📋 **План**: Cost optimization insights с учетом RAG
- 📋 **План**: ROI анализ для enterprise клиентов

## 🔄 Интеграции и экосистема

### Расширенные интеграции:
- ✅ OpenAI GPT API (upgraded to 1.95.0)
- ✅ Git repository parsing  
- ✅ Multi-language AST parsing
- 🔄 **В разработке**: Qdrant векторная БД (v1.15.0+)
- 🔄 **В разработке**: Sentence-transformers 5.1.0 с precision control
- 🔄 **В разработке**: FastAPI для REST API

### Планируемые enterprise интеграции:
- 📋 **План**: Alternative LLM providers (Anthropic Claude, Azure OpenAI)
- 📋 **План**: IDE plugins с RAG поиском (VSCode, JetBrains)
- 📋 **План**: CI/CD integration (GitHub Actions, GitLab CI)
- 📋 **План**: JIRA/Confluence integration для документации

### Векторные модели и провайдеры:
- 🔄 **В разработке**: Sentence-transformers с CPU-оптимизацией
- 📋 **План**: Специализированные code embeddings (CodeT5, GraphCodeBERT)  
- 📋 **План**: ONNX Runtime для максимальной производительности
- 📋 **План**: Локальные модели без внешних API

## 🎯 Следующие milestone'ы

### Milestone 1: Production-Ready RAG Core (ТЕКУЩИЙ - АКТИВНАЯ РЕАЛИЗАЦИЯ)
- **Timeline**: 2-3 недели  
- **Goals**:
  - ✅ CPU-оптимизированный embedder с precision='int8'
  - 🔄 Qdrant интеграция с квантованием и репликацией
  - 🔄 Гибридный поиск (dense + sparse) с MMR
  - 🔄 LRU кэш с TTL для горячих запросов
  - 🔄 Базовая интеграция в существующий workflow

### Milestone 2: Enterprise Infrastructure (следующий)
- **Timeline**: 3-4 недели
- **Goals**:  
  - Docker-compose развертывание
  - Prometheus/Grafana мониторинг
  - CI/CD pipeline с автотестами
  - Нагрузочные тесты на 20 пользователей
  - TLS безопасность и контроль доступа

### Milestone 3: Advanced RAG Features (2-3 месяца)
- **Timeline**: 2-3 месяца
- **Goals**:
  - Специализированные code embeddings (CodeT5+)
  - Семантический граф зависимостей кода
  - A/B тестирование поисковых алгоритмов
  - IDE плагины с RAG интеграцией
  - SOC2 compliance и enterprise сертификация

### Milestone 4: AI-Powered Code Intelligence (будущее)
- **Timeline**: 6+ месяцев
- **Goals**:
  - Автоматические рекомендации по рефакторингу через RAG
  - Предиктивный анализ технического долга
  - Интеграция с корпоративными knowledge base
  - Мультимодальный поиск (код + документация + диаграммы)
