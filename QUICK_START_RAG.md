# 🚀 БЫСТРЫЙ СТАРТ: ПРОВЕРКА RAG СИСТЕМЫ

## 📋 Требования
- Python 3.8+
- 4GB+ RAM (рекомендуется 8GB)
- Docker (для Qdrant) или локальный Qdrant
- Интернет для скачивания моделей FastEmbed

## ⚡ За 5 минут до работы

### Установка зависимостей

```bash
# Шаг 1: Клонирование и установка
git clone <repo>
cd repo_sum
pip install -r requirements.txt

# Шаг 2: Проверка базовых импортов
python -c "from rag import CPUEmbedder, QdrantVectorStore; print('✅ RAG импорты работают')"
```

### Запуск Qdrant

#### Вариант A: Docker (рекомендуется)
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

#### Вариант B: Qdrant Cloud
Создать кластер на https://cloud.qdrant.io
Обновить settings.json с API ключом

---

## 🧪 ПРОВЕРКА ПО ШАГАМ

### Шаг 1: Проверка базовых тестов
```bash
# Базовые компоненты
pytest tests/test_rag_imports.py tests/test_vector_store_basic.py -v

# Ожидаемый результат: все тесты пройдены ✅
```

### Шаг 2: Проверка подключения к Qdrant
```bash
python main.py rag status

# Ожидаемый вывод:
# 🟢 Qdrant: подключение успешно (localhost:6333)
# 📊 Коллекция code_chunks: не создана
```

### Шаг 3: Индексация тестового репозитория
```bash
python main.py rag index tests/fixtures/test_repo --batch-size 32

# Ожидаемый вывод:
# 🔄 Индексация репозитория: tests/fixtures/test_repo
# 📁 Найдено файлов: 6
# 🧩 Создано чанков: ~25
# ⚡ Генерация эмбеддингов...
# 📊 Индексировано: 25/25 [100%]
# ✅ Завершено за ~10-30s
```

### Шаг 4: Проверка поиска
```bash
python main.py rag search "user authentication" --top-k 3

# Ожидаемый вывод:
# 🔍 Поиск: "user authentication"  
# 📊 Найдено результатов: 3
#
# 🎯 1. tests/fixtures/test_repo/auth/user.py:15-25 (score: 0.85+)
#    class User:
#        def authenticate(self, password):
#            ...
```

### Шаг 5: Простой нагрузочный тест
```bash
# Несколько поисковых запросов подряд
python main.py rag search "database connection" --top-k 5
python main.py rag search "validation functions" --top-k 5  
python main.py rag search "helper utilities" --top-k 5

# Каждый запрос должен выполняться за <200ms
```

### Шаг 6: Комплексные тесты
```bash
# Интеграционные тесты
pytest tests/rag/test_rag_integration.py -v

# E2E тесты CLI
pytest tests/rag/test_rag_e2e_cli.py::TestRagCliE2E::test_rag_search_command -v

# Быстрая проверка всех RAG тестов
python tests/rag/run_rag_tests.py smoke
```

---

## 🐛 РЕШЕНИЕ ПРОБЛЕМ

### Проблема: "Connection refused" к Qdrant
**Решение:**
```bash
# Проверить что Qdrant запущен
docker ps | grep qdrant

# Перезапустить если нужно
docker stop $(docker ps -q --filter ancestor=qdrant/qdrant)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Проблема: "FastEmbed model download failed"
**Решение:**
```bash
# Проверить интернет и повторить
python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Альтернативно - использовать sentence-transformers
# Обновить в settings.json: "provider": "sentence-transformers"
```

### Проблема: Медленная индексация
**Решение:**
```bash
# Увеличить batch size
python main.py rag index /path/to/repo --batch-size 64

# Проверить использование CPU/RAM
python main.py rag status --detailed
```

---

## ✅ КРИТЕРИИ УСПЕХА

### Функциональность:
- [ ] ✅ Qdrant подключается (rag status)
- [ ] ✅ Индексация работает (rag index) 
- [ ] ✅ Поиск возвращает релевантные результаты
- [ ] ✅ Все базовые тесты проходят

### Производительность:
- [ ] ✅ Поиск выполняется за <200ms
- [ ] ✅ Индексация >5 файлов/сек
- [ ] ✅ Память <500MB для тестового репозитория

### Качество поиска:
- [ ] ✅ "authentication" находит auth модули
- [ ] ✅ "database" находит db модули
- [ ] ✅ "validation" находит validators.py

---

## 🎯 ЧТО ДАЛЬШЕ?

### Для продакшн использования:
1. **Индексируйте ваш репозиторий:**
   ```bash
   python main.py rag index /path/to/your/project --batch-size 512
   ```

2. **Используйте семантический поиск:**
   ```bash
   python main.py rag search "OAuth integration" --lang python --top-k 10
   ```

3. **Мониторьте производительность:**
   ```bash
   python main.py rag status --detailed
   ```

### Для разработки дальше:
- M2: Гибридный поиск (BM25 + dense)
- M3: Web UI интеграция
- M4: Production развёртывание

---

## 📚 ДОПОЛНИТЕЛЬНЫЕ РЕСУРСЫ

- [Архитектура RAG системы](.clinerules/RAG_architecture.md)
- [Подробные тесты](tests/rag/README.md)
- [Отчёт о тестировании](tests/rag/test_report.md)
- [Основная документация](README.md)

**Время выполнения:** ~10-15 минут  
**Статус:** Готово к использованию ✅