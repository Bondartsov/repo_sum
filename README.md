# 🤖 repo_sum: RAG-as-a-Service Анализатор Кодовых Репозиториев

**Автоматический анализатор кода с ИИ и семантическим поиском на базе RAG (Retrieval Augmented Generation). RAG-as-a-Service архитектура с Jina v3 embeddings на удаленной VM для беспрецедентного качества поиска и анализа кода.**

---

## ⚠️ ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ: ЧАСТНАЯ СОБСТВЕННОСТЬ

**🚫 ПРОЕКТ ЯВЛЯЕТСЯ ЧАСТНОЙ СОБСТВЕННОСТЬЮ**  
**🚫 ЗАПРЕЩЕНО КОММЕРЧЕСКОЕ ИСПОЛЬЗОВАНИЕ**  
**🚫 ЛЮБОЕ ИСПОЛЬЗОВАНИЕ ТРЕБУЕТ ПИСЬМЕННОГО РАЗРЕШЕНИЯ**

Этот проект представляет собой интеллектуальную собственность и не может быть использован в коммерческих целях без явного письменного разрешения правообладателя. Любое несанкционированное использование будет преследоваться по закону.

---

## 🚀 Возможности

### 🎯 Основные Функции
- **🔍 Семантический поиск по коду** - поиск по смыслу, а не только по тексту
- **🧠 ИИ-анализ кода** - генерация детальной документации через OpenAI GPT
- **📊 Интеллектуальное чанкирование** - разбиение кода на логические блоки
- **🌐 Веб-интерфейс** - drag&drop загрузка и интерактивный поиск
- **⚡ CLI команды** - автоматизация анализа и поиска

### 🔥 RAG-as-a-Service Архитектура
- **🏗️ VM-based вычисления** - Jina v3 (570M параметров) на удалённом сервере
- **🔗 HTTP-first интеграция** - локально только HTTP клиенты
- **⚡ CPU-оптимизация** - работает без GPU на любом сервере
- **📈 Enterprise масштабирование** - до 50+ пользователей одновременно
- **🔒 SSH автоматизация** - полное развертывание одной командой

### 🎨 Поддержка Языков Программирования
- **Python** (.py) - полный AST анализ
- **JavaScript/TypeScript** (.js, .ts, .jsx, .tsx)
- **Java** (.java)
- **C++** (.cpp, .cc, .cxx, .h, .hpp)
- **C#** (.cs)
- **Go** (.go)
- **Rust** (.rs)
- **PHP** (.php)
- **Ruby** (.rb)

---

## 🛠 Технические Характеристики

### 🚀 Производительность
- **Поиск**: <200ms (кэшированный), <500ms (холодный)
- **Индексация**: >8 файлов/секунду
- **Конкурентность**: 20+ параллельных пользователей
- **Память**: ~100MB локально (99% экономия)

### 🧠 ИИ Модели
- **Jina v3**: 570M параметров, dual task (retrieval.query/passage)
- **Matryoshka Compression**: 1024d → 384d сжатие
- **Гибридный поиск**: Dense (Jina v3) + Sparse (SPLADE) векторы
- **RRF Fusion**: Reciprocal Rank Fusion для ранжирования

### 🏗️ Инфраструктура
- **VM Server**: Intel Xeon Gold 6248R, 31GB RAM, Ubuntu 22.04.4
- **Векторная БД**: Qdrant с квантованием и репликацией
- **API**: FastAPI на порту 8000 с health checks
- **SSH Automation**: полная автоматизация развертывания

---

## ⚡ Быстрый Старт

### 1. Клонирование и Установка
```bash
git clone <repository-url>
cd repo_sum
pip install -r requirements.txt
```

### 2. Настройка Переменных Окружения
```bash
# Создайте .env файл
cp .env.example .env
```

**Обязательные переменные:**
```env
# OpenAI API
OPENAI_API_KEY=sk-your-api-key-here

# VM RAG Service
VM_HOST=10.61.11.54
VM_USER=user
VM_PASSWORD=your_vm_password

# Qdrant (локальный или облачный)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Запуск VM Инфраструктуры
```bash
# Полная автоматизация развертывания
python vm_start.py start

# Проверка статуса
python vm_start.py status

# Остановка сервисов
python vm_start.py stop
```

### 4. Использование Системы

#### 🌐 Веб-Интерфейс (Рекомендуется)
```bash
python run_web.py
# Откройте http://localhost:8501
```

**Возможности веб-интерфейса:**
- ✅ Drag&drop загрузка репозиториев
- ✅ Семантический поиск по коду
- ✅ Интерактивный Q&A с кодовой базой
- ✅ Генерация детальной документации
- ✅ Real-time статистика и метрики

#### 💻 CLI Команды

**Анализ репозитория:**
```bash
# Базовый анализ
python main.py analyze /path/to/repository

# С RAG индексацией
python main.py analyze /path/to/repository --with-rag

# Инкрементальный анализ
python main.py analyze /path/to/repository --incremental
```

**Семантический поиск:**
```bash
# Простой поиск
python main.py rag search "authentication middleware"

# Поиск с фильтрами
python main.py rag search "database connection" --lang python --top-k 5

# Поиск по типу контента
python main.py rag search "error handling" --chunk-type class
```

**Управление системой:**
```bash
# Статус RAG системы
python main.py rag status

# Индексация репозитория
python main.py rag index /path/to/repository

# Очистка кэша
python main.py clear-cache
```

---

## 🏗️ Архитектура Системы

### 🎯 Революционная RAG-as-a-Service Модель
```
[Локальная машина]     HTTP REST API     [VM t-ubuntu-redis 31GB]
├─ repo_sum CLI    ←─────────────→       ├─ FastAPI :8000 ✅
├─ Web UI          ←─────────────→       ├─ Jina v3 (570M) ✅
├─ OpenAI анализ   ←─────────────→       ├─ Qdrant :6333 ✅
└─ HTTP клиенты    ←─────────────→       └─ sentence-transformers>=3.0 ✅
```

### 🧩 Ключевые Компоненты

#### Локальные Компоненты:
- **RepositoryAnalyzer** - координация анализа кода
- **FileScanner** - поиск и фильтрация файлов
- **CodeChunker** - интеллектуальное разбиение кода
- **OpenAIManager** - интеграция с OpenAI API
- **DocumentationGenerator** - создание MD отчетов

#### VM Компоненты (RAG-as-a-Service):
- **FastAPI Service** - REST API для эмбеддингов и поиска
- **Jina v3 Embedder** - 570M параметрова модель
- **Qdrant Vector Store** - векторная база данных
- **Hybrid Search Engine** - комбинация dense + sparse поиска

#### HTTP Клиенты:
- **RemoteVMEmbedder** - HTTP клиент для VM эмбеддингов
- **RemoteVectorStore** - HTTP клиент для VM поиска
- **VMRAGService** - координация VM сервисов

### 🔄 Рабочий Процесс

1. **Анализ кода** → локальная обработка через OpenAI
2. **Индексация** → HTTP запросы к VM для эмбеддингов
3. **Поиск** → комбинированный dense + sparse поиск на VM
4. **Генерация** → создание документации с RAG контекстом

---

## 📊 Статус Разработки

### ✅ **Текущий Статус: M2.5 VM Migration - 80% ЗАВЕРШЕНО**

#### **Достигнутые Результаты:**
- ✅ **VM Infrastructure**: Xeon Gold 6248R, 31GB RAM, Ubuntu 22.04.4
- ✅ **Jina v3 Success**: 570M параметров работают стабильно
- ✅ **FastAPI Service**: запущен на 10.61.11.54:8000, health check "healthy"
- ✅ **SSH Automation**: полная автоматизация через vm_start.py
- ✅ **Performance**: 4.35it/s inference, <10s model loading

#### **Критические Задачи (финальные исправления):**
- ❌ **Async/Sync Fix**: исправление корутин в remote клиентах
- ❌ **Integration Testing**: полный workflow тестирование
- ❌ **Web UI Testing**: Streamlit RAG функции

### 🎯 **Следующие Этапы:**
- **M3 (Ноябрь 2025)**: RAG-Enhanced Analysis - интеграция VM RAG в OpenAI анализ
- **M4 (Декабрь 2025)**: Production Deployment & Scaling - VM кластер
- **M5 (Q2 2026)**: Advanced Intelligence - ML оптимизации

---

## 📋 Системные Требования

### Минимальные Требования:
- **Python**: 3.8+
- **RAM**: 4GB+ (рекомендуется 8GB+)
- **CPU**: любой современный (GPU НЕ требуется)
- **OS**: Windows, macOS, Linux
- **Интернет**: для загрузки моделей и OpenAI API

### Внешние Зависимости:
- **OpenAI API** - для анализа кода
- **VM Server** - для RAG вычислений (предоставляется)
- **Qdrant** - векторная база данных

---

## 🧪 Тестирование

### RAG Система:
```bash
# Все RAG тесты
python tests/rag/run_rag_tests.py all

# Быстрая проверка
python tests/rag/run_rag_tests.py quick

# Интеграционные тесты
pytest tests/rag/ -v
```

### Основные Функции:
```bash
# Все тесты
pytest tests/test_*.py -v

# С покрытием
pytest --cov=. tests/ --cov-report=html
```

---

## 🐛 Устранение Проблем

### Частые Проблемы:

**VM Недоступен:**
```bash
# Проверьте подключение
python vm_start.py status

# Перезапустите VM сервисы
python vm_start.py restart
```

**OpenAI API Ошибки:**
```bash
# Проверьте API ключ
echo $OPENAI_API_KEY

# Проверьте квоты
python main.py token-stats
```

**Поиск Не Работает:**
```bash
# Проверьте RAG статус
python main.py rag status --detailed

# Переиндексируйте репозиторий
python main.py rag index /path/to/repository --recreate
```

---

## 📚 Документация и Ресурсы

### Техническая Документация:
- 🗺️ **[Development Roadmap.md](rules/Development Roadmap.md)** - полная дорожная карта развития
- 📋 **[.clinerules/](.clinerules/)** - система памяти проекта
- 🏗️ **[SETUP.md](SETUP.md)** - детальная инструкция по настройке
- 🧪 **[AGENTS.md](AGENTS.md)** - правила работы с кодом

### Архитектурная Документация:
- **RAG Architecture**: `.clinerules/RAG_architecture.md`
- **VM Migration**: `.clinerules/QUICK_START_RAG_ported.md`
- **Testing Strategy**: `tests/rag/TESTING_STRATEGY.md`

---

## 📚 Консолидированная документация

### 📋 Основная документация проекта:
- **[📖 Обзор проекта](rules/projectContext.md)** - Общее назначение и функционал системы
- **[🏗️ Техническая архитектура](rules/technical_architecture.md)** - Детальное описание архитектуры системы
- **[🗺️ Дорожная карта](rules/roadmap.md)** - Roadmap развития проекта и планы на будущее
- **[💳 Технический долг](rules/technical_debt.md)** - Список накопленного технического долга и приоритеты
- **[✅ Активные задачи](rules/active_tasks.md)** - Текущие задачи и статус их выполнения
- **[👥 Инструкции для агентов](rules/agents.md)** - Правила и инструкции для AI агентов

### 📋 Документация соответствия:
- **[📋 Roadmap соответствия](rules/compliance_roadmap.md)** - План обеспечения соответствия стандартам и требованиям

---

## 🤝 Контакты и Поддержка

### Разработка:
- **GitHub**: https://github.com/Bondartsov/repo_sum.git
- **Issues**: Создавайте Issues для багов и предложений
- **Pull Requests**: Добро пожаловать для улучшений

### VM Инфраструктура:
- **RAG Service**: 10.61.11.54:8000
- **Qdrant DB**: 10.61.11.54:6333
- **SSH Access**: автоматизировано через vm_start.py

### Поддержка:
При возникновении проблем:
1. Проверьте логи с флагом `-v` (verbose mode)
2. Укажите версию Python и ОС
3. Приложите содержимое `.env` (без API ключей)
4. Создайте Issue в GitHub

---

## 📄 Лицензия

**🚫 ЧАСТНАЯ СОБСТВЕННОСТЬ**  
**🚫 ЗАПРЕЩЕНО КОММЕРЧЕСКОЕ ИСПОЛЬЗОВАНИЕ**  
**✅ ЛЮБОЕ ИСПОЛЬЗОВАНИЕ ТРЕБУЕТ РАЗРЕШЕНИЯ**  
**thèque OKPGG, N0. 0002859886624400
