# Code Mappings: Repository Analyzer

**Дата обновления:** 22 сентября 2025
**Статус:** M2.5 VM Migration - 80% ЗАВЕРШЕНО
**Версия:** 0.7.1 (M2.5 VM Migration SUCCESS + async/sync fixes required)

---

## 🔗 КРОСС-РЕФЕРЕНСЫ МЕЖДУ ДОКУМЕНТАЦИЕЙ И КОДОМ

### **ЦЕНТРАЛЬНАЯ СИСТЕМА СВЯЗЕЙ:**

---

## 📋 ДОКУМЕНТАЦИЯ → КОД

### **ROADMAP.md → Реализация:**

#### **M1: Production-Ready RAG Core**
- **"CPU-оптимизированная RAG"** → [`rag/embedder.py`](rag/embedder.py) - FastEmbed интеграция
- **"Qdrant векторная БД"** → [`rag/vector_store.py`](rag/vector_store.py) - Qdrant клиент
- **"Гибридный поиск"** → [`rag/query_engine.py`](rag/query_engine.py) - RRF + MMR алгоритмы
- **"CLI + Web UI интеграция"** → [`main.py`](main.py), [`web_ui.py`](web_ui.py)

#### **M2: Hybrid Search Enhancement**
- **"Sparse vectors (BM25 + SPLADE)"** → [`rag/sparse_encoder.py`](rag/sparse_encoder.py)
- **"RRF fusion + MMR re-ranking"** → [`rag/search_service.py`](rag/search_service.py)
- **"Code tokenization"** → [`rag/sparse_encoder.py`](rag/sparse_encoder.py) - токенизация кода

#### **M2.5: Jina v3 VM Migration**
- **"VM Infrastructure"** → [`vm_start.py`](vm_start.py) - SSH автоматизация
- **"Jina v3 Success"** → [`rag/remote_embedder.py`](rag/remote_embedder.py) - HTTP клиент
- **"FastAPI Service"** → [`vm_rag_service.py`](vm_rag_service.py) - VM сервис
- **"Remote клиенты"** → [`rag/remote_vector_store.py`](rag/remote_vector_store.py)

---

## 🏗️ КОД → ДОКУМЕНТАЦИЯ

### **Основные модули системы:**

#### **Core System:**
- **[`main.py`](main.py)** → [activeContext.md](activeContext.md) - текущий фокус
- **[`config.py`](config.py)** → [techContext.md](techContext.md) - конфигурационная система
- **[`file_scanner.py`](file_scanner.py)** → [progress.md](progress.md) - история сканирования
- **[`openai_integration.py`](openai_integration.py)** → [techContext.md](techContext.md) - OpenAI интеграция

#### **RAG System:**
- **[`rag/embedder.py`](rag/embedder.py)** → [techContext.md](techContext.md) - CPU эмбеддер
- **[`rag/vector_store.py`](rag/vector_store.py)** → [progress.md](progress.md) - Qdrant интеграция
- **[`rag/query_engine.py`](rag/query_engine.py)** → [techContext.md](techContext.md) - поисковый движок
- **[`rag/search_service.py`](rag/search_service.py)** → [active_tasks.md](active_tasks.md) - текущие проблемы
- **[`rag/sparse_encoder.py`](rag/sparse_encoder.py)** → [progress.md](progress.md) - SPLADE интеграция

#### **VM Migration:**
- **[`vm_start.py`](vm_start.py)** → [project_status.md](project_status.md) - статус VM
- **[`vm_rag_service.py`](vm_rag_service.py)** → [techContext.md](techContext.md) - VM сервис
- **[`rag/remote_embedder.py`](rag/remote_embedder.py)** → [active_tasks.md](active_tasks.md) - async проблемы
- **[`rag/remote_vector_store.py`](rag/remote_vector_store.py)** → [active_tasks.md](active_tasks.md) - remote клиент

#### **Parser System:**
- **[`parsers/base_parser.py`](parsers/base_parser.py)** → [techContext.md](techContext.md) - базовый парсер
- **[`parsers/python_parser.py`](parsers/python_parser.py)** → [progress.md](progress.md) - Python парсинг
- **[`parsers/javascript_parser.py`](parsers/javascript_parser.py)** → [progress.md](progress.md) - JS парсинг

#### **UI System:**
- **[`web_ui.py`](web_ui.py)** → [active_tasks.md](active_tasks.md) - Web UI тестирование
- **[`unified_launcher.py`](unified_launcher.py)** → [progress.md](progress.md) - launcher система

---

## 🧪 ТЕСТЫ → ФУНКЦИОНАЛЬНОСТЬ

### **Test Coverage Mapping:**

#### **Unit Tests (59 тестов):**
- **[`tests/test_config.py`](tests/test_config.py)** → [`config.py`](config.py) - конфигурация
- **[`tests/test_file_scanner.py`](tests/test_file_scanner.py)** → [`file_scanner.py`](file_scanner.py) - сканирование
- **[`tests/test_openai_integration.py`](tests/test_openai_integration.py)** → [`openai_integration.py`](openai_integration.py) - OpenAI
- **[`tests/rag/test_sparse_encoder.py`](tests/rag/test_sparse_encoder.py)** → [`rag/sparse_encoder.py`](rag/sparse_encoder.py) - sparse

#### **Integration Tests (67 тестов):**
- **[`tests/rag/test_rag_integration.py`](tests/rag/test_rag_integration.py)** → [`rag/`](rag/) - RAG система
- **[`tests/test_main.py`](tests/test_main.py)** → [`main.py`](main.py) - основной CLI
- **[`tests/test_web_ui.py`](tests/test_web_ui.py)** → [`web_ui.py`](web_ui.py) - Web UI

#### **Functional Tests (25 тестов):**
- **[`tests/rag/test_rag_e2e_cli.py`](tests/rag/test_rag_e2e_cli.py)** → [`main.py`](main.py) - CLI команды
- **[`tests/e2e/test_e2e_cli_analyze_generate_docs.py`](tests/e2e/test_e2e_cli_analyze_generate_docs.py)** → E2E workflow

---

## 📚 ДОКУМЕНТАЦИЯ → ТРЕБОВАНИЯ

### **Технические требования:**

#### **VM Infrastructure:**
- **[`vm_start.py`](vm_start.py)** → [techContext.md](techContext.md) - VM автоматизация
- **[`scripts/vm_setup_phase1.py`](scripts/vm_setup_phase1.py)** → [progress.md](progress.md) - VM setup
- **[`scripts/validate_vm_env.py`](scripts/validate_vm_env.py)** → [active_tasks.md](active_tasks.md) - валидация

#### **Dependencies:**
- **[`requirements.txt`](requirements.txt)** → [techContext.md](techContext.md) - зависимости
- **[`scripts/verify_requirements.py`](scripts/verify_requirements.py)** → [progress.md](progress.md) - верификация
- **[`.env.example`](.env.example)** → [techContext.md](techContext.md) - переменные окружения

#### **Configuration:**
- **[`settings.json`](settings.json)** → [techContext.md](techContext.md) - настройки
- **[`config.py`](config.py)** → [techContext.md](techContext.md) - конфигурационные классы
- **[`pytest.ini`](pytest.ini)** → [progress.md](progress.md) - тестовые настройки

---

## 🔧 ИНСТРУМЕНТЫ → ФУНКЦИИ

### **CLI Commands:**
- **`python main.py`** → [activeContext.md](activeContext.md) - основной координатор
- **`python run_web.py`** → [progress.md](progress.md) - веб-интерфейс
- **`python vm_start.py`** → [project_status.md](project_status.md) - VM управление
- **`python scripts/verify_requirements.py`** → [progress.md](progress.md) - проверка зависимостей

### **Development Tools:**
- **[`clean_pycache.py`](clean_pycache.py)** → [progress.md](progress.md) - очистка кэша
- **[`scripts/backup_env_settings.py`](scripts/backup_env_settings.py)** → [progress.md](progress.md) - бэкапы
- **[`scripts/cleanup_old_collections.py`](scripts/cleanup_old_collections.py)** → [progress.md](progress.md) - очистка

---

## 📊 МЕТРИКИ → ИМПЛЕМЕНТАЦИЯ

### **Performance Metrics:**
- **"<300ms p95 латентность"** → [`rag/query_engine.py`](rag/query_engine.py) - LRU кэш
- **">8 файлов/сек индексация"** → [`rag/indexer_service.py`](rag/indexer_service.py) - батчевая обработка
- **"<700MB памяти"** → [`rag/embedder.py`](rag/embedder.py) - CPU оптимизация
- **"20+ пользователей"** → [`config.py`](config.py) - ParallelismConfig

### **Quality Metrics:**
- **"Precision@10 +15-20%"** → [`rag/search_service.py`](rag/search_service.py) - RRF алгоритм
- **"Recall@100 +25-30%"** → [`rag/sparse_encoder.py`](rag/sparse_encoder.py) - SPLADE интеграция
- **"+40-60% vs BGE"** → [`rag/remote_embedder.py`](rag/remote_embedder.py) - Jina v3

---

## 🐛 ПРОБЛЕМЫ → РЕШЕНИЯ

### **Критические проблемы M2.5:**

#### **Async/Sync Mismatch:**
- **Проблема:** [`rag/remote_embedder.py`](rag/remote_embedder.py) - async методы
- **Решение:** [active_tasks.md](active_tasks.md) - sync wrapper
- **Тесты:** [`tests/test_remote_clients.py`](tests/test_remote_clients.py)

#### **SearchService Error:**
- **Проблема:** [`rag/search_service.py`](rag/search_service.py) - coroutine ошибки
- **Решение:** [active_tasks.md](active_tasks.md) - integration testing
- **Тесты:** [`tests/rag/test_rag_integration.py`](tests/rag/test_rag_integration.py)

#### **Web UI Issues:**
- **Проблема:** [`web_ui.py`](web_ui.py) - RAG функции
- **Решение:** [active_tasks.md](active_tasks.md) - UI testing
- **Тесты:** [`tests/test_web_ui.py`](tests/test_web_ui.py)

---

## 📈 ПРОГРЕСС → РЕАЛИЗАЦИЯ

### **Milestone Progress:**

#### **M1 (100% ЗАВЕРШЁН):**
- **[`rag/embedder.py`](rag/embedder.py)** → CPU-оптимизированный эмбеддер
- **[`rag/vector_store.py`](rag/vector_store.py)** → Qdrant интеграция
- **[`rag/query_engine.py`](rag/query_engine.py)** → гибридный поиск

#### **M2 (100% ЗАВЕРШЁН):**
- **[`rag/sparse_encoder.py`](rag/sparse_encoder.py)** → BM25/SPLADE
- **[`rag/search_service.py`](rag/search_service.py)** → RRF/MMR
- **[`tests/rag/test_splade_encoder.py`](tests/rag/test_splade_encoder.py)** → тесты

#### **M2.5 (80% ЗАВЕРШЕНО):**
- **[`vm_start.py`](vm_start.py)** → VM автоматизация ✅
- **[`vm_rag_service.py`](vm_rag_service.py)** → FastAPI сервис ✅
- **[`rag/remote_embedder.py`](rag/remote_embedder.py)** → HTTP клиент ❌
- **[`rag/remote_vector_store.py`](rag/remote_vector_store.py)** → remote store ❌

---

## 🔍 АУДИТ → ИСПРАВЛЕНИЯ

### **Memory Bank Audit Results:**

#### **✅ 12 пунктов OK:**
- **[`config.py`](config.py)** → [progress.md](progress.md) - конфигурация
- **[`main.py`](main.py)** → [activeContext.md](activeContext.md) - CLI
- **[`web_ui.py`](web_ui.py)** → [progress.md](progress.md) - Web UI

#### **⚠️ 4 пункта PARTIAL:**
- **[`rag/search_service.py`](rag/search_service.py)** → [active_tasks.md](active_tasks.md) - min_score
- **[`openai_integration.py`](openai_integration.py)** → [progress.md](progress.md) - статистика
- **[`requirements.txt`](requirements.txt)** → [techContext.md](techContext.md) - версии

#### **❌ 4 пункта MISMATCH:**
- **[`rag/query_engine.py`](rag/query_engine.py)** → [active_tasks.md](active_tasks.md) - health_check
- **[`README.md`](README.md)** → [progress.md](progress.md) - устаревшая информация
- **[`config.py`](config.py)** → [active_tasks.md](active_tasks.md) - пороги релевантности

---

## 📝 ДОКУМЕНТАЦИЯ → ОБНОВЛЕНИЯ

### **Обновляемые файлы:**

#### **Основная документация:**
- **[`README.md`](README.md)** → [completed_features.md](completed_features.md) - общая информация
- **[`SETUP.md`](SETUP.md)** → [active_tasks.md](active_tasks.md) - инструкции по настройке
- **[`ROADMAP.md`](ROADMAP.md)** → [project_status.md](project_status.md) - план развития

#### **Техническая документация:**
- **[`rules/techContext.md`](rules/techContext.md)** → [`config.py`](config.py) - архитектура
- **[`rules/progress.md`](rules/progress.md)** → [`main.py`](main.py) - история
- **[`rules/activeContext.md`](rules/activeContext.md)** → [`vm_start.py`](vm_start.py) - фокус

#### **Специализированная документация:**
- **[`.clinerules/QUICK_START_RAG_ported.md`](.clinerules/QUICK_START_RAG_ported.md)** → [`rag/`](rag/) - RAG quick start
- **[`.clinerules/RAG_architecture.md`](.clinerules/RAG_architecture.md)** → [`rag/`](rag/) - RAG архитектура
- **[`.clinerules/TODO_SPLADE.md`](.clinerules/TODO_SPLADE.md)** → [`rag/sparse_encoder.py`](rag/sparse_encoder.py) - SPLADE задачи

---

## 🎯 СЛЕДУЮЩИЕ ШАГИ

### **Приоритетные обновления:**

#### **Критические:**
- **[`rag/remote_embedder.py`](rag/remote_embedder.py)** → [active_tasks.md](active_tasks.md) - async fix
- **[`rag/remote_vector_store.py`](rag/remote_vector_store.py)** → [active_tasks.md](active_tasks.md) - sync wrapper
- **[`tests/test_remote_clients.py`](tests/test_remote_clients.py)** → [active_tasks.md](active_tasks.md) - тесты

#### **Важные:**
- **[`web_ui.py`](web_ui.py)** → [active_tasks.md](active_tasks.md) - UI тестирование
- **[`README.md`](README.md)** → [completed_features.md](completed_features.md) - документация
- **[`SETUP.md`](SETUP.md)** → [active_tasks.md](active_tasks.md) - инструкции

#### **Плановые:**
- **[`scripts/`](scripts/)** → [progress.md](progress.md) - утилиты
- **[`tests/`](tests/)** → [progress.md](progress.md) - тестовое покрытие
- **[`docs/`](docs/)** → [completed_features.md](completed_features.md) - документация

---

**Дата создания:** 22 сентября 2025
**Статус:** Code mappings established
**Следующее обновление:** При изменении кода или документации
