# 🚀 Быстрый старт: Проверка RAG системы (портировано в .clinerules)

## 📋 Требования
- Python 3.8+
- 4GB+ RAM (рекомендуется 8GB)
- Docker (для Qdrant) или локальный Qdrant
- Интернет для скачивания моделей FastEmbed

## ⚡ За 5 минут до работы

### Установка зависимостей
```bash
git clone <repo>
cd repo_sum
pip install -r requirements.txt
python -c "from rag import CPUEmbedder, QdrantVectorStore; print('✅ RAG импорты работают')"
```

### Запуск Qdrant
- **Docker (рекомендуется)**:
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```
- **Qdrant Cloud**: создать кластер на https://cloud.qdrant.io и обновить settings.json с API ключом

---

## 🧪 Проверка по шагам
1. **Базовые тесты**:
```bash
pytest tests/test_rag_imports.py tests/test_vector_store_basic.py -v
```
2. **Проверка подключения к Qdrant**:
```bash
python main.py rag status
```
3. **Индексация тестового репозитория**:
```bash
python main.py rag index tests/fixtures/test_repo --batch-size 32
```
4. **Проверка поиска**:
```bash
python main.py rag search "user authentication" --top-k 3
```
5. **Нагрузочный тест**:
```bash
python main.py rag search "database connection" --top-k 5
```
6. **Комплексные тесты**:
```bash
pytest tests/rag/test_rag_integration.py -v
pytest tests/rag/test_rag_e2e_cli.py::TestRagCliE2E::test_rag_search_command -v
python tests/rag/run_rag_tests.py smoke
```

---

## 🐛 Решение проблем
- **Connection refused**: проверить, что Qdrant запущен (`docker ps | grep qdrant`)
- **FastEmbed model download failed**: проверить интернет или использовать sentence-transformers
- **Медленная индексация**: увеличить batch size и проверить CPU/RAM

---

## ✅ Критерии успеха
- Qdrant подключается
- Индексация работает
- Поиск возвращает релевантные результаты
- Все базовые тесты проходят
- Поиск <200ms, индексация >5 файлов/сек, память <500MB

---

## 🎯 Что дальше?
- Индексируйте ваш репозиторий: `python main.py rag index /path/to/your/project --batch-size 512`
- Используйте семантический поиск: `python main.py rag search "OAuth integration" --lang python --top-k 10`
- Мониторьте производительность: `python main.py rag status --detailed`

---

## 📚 Дополнительные ресурсы
- [Архитектура RAG системы](.clinerules/RAG_architecture.md)
- [Подробные тесты](tests/rag/README.md)
- [Отчёт о тестировании](tests/rag/test_report.md)
- [Основная документация](README.md)

**Время выполнения:** ~10-15 минут  
**Статус:** Готово к использованию ✅
