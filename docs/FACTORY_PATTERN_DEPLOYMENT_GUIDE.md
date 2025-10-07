# 🚀 Deployment Guide: Factory Pattern на VM сервере

**Дата:** 7 октября 2025  
**Версия:** 2.0  
**Цель:** Развёртывание Factory Pattern решения для устранения рекурсии

---

## 📋 Pre-deployment Checklist

- [ ] Код Factory Pattern развёрнут на VM
- [ ] Unit тесты пройдены (32/32)
- [ ] Integration тесты пройдены (8/8)
- [ ] Backup текущей версии создан
- [ ] План отката подготовлен

---

## 🔄 Deployment Steps

### Шаг 1: Backup текущей версии

```bash
# На VM сервере
cd /path/to/repo_sum

# Создать backup
tar -czf backups/pre_factory_pattern_$(date +%Y%m%d_%H%M%S).tar.gz \
    rag/__init__.py \
    rag/indexer_service.py \
    rag/search_service.py \
    vm_rag_service.py

# Проверить backup
ls -lh backups/
```

### Шаг 2: Остановка сервиса

```bash
# Остановить текущий VM сервис
python vm_rag_service.py stop

# Проверить что сервис остановлен
python vm_rag_service.py status
```

### Шаг 3: Развёртывание новых файлов

```bash
# Копирование новых файлов (если развёртывание с другой машины)
# Или git pull если используется git

# Убедиться что новые файлы на месте
ls -l rag/context.py
ls -l rag/factory.py

# Проверить что изменения применены
grep "RAGFactory" rag/indexer_service.py
grep "RAGFactory" rag/search_service.py
```

### Шаг 4: Удаление временного решения

```bash
# Удалить переменную окружения из .bashrc / .profile
sed -i '/FORCE_LOCAL_VECTOR_STORE/d' ~/.bashrc
source ~/.bashrc

# Проверить что переменная удалена
echo $FORCE_LOCAL_VECTOR_STORE
# Должно быть пусто
```

### Шаг 5: Запуск сервиса с новым кодом

```bash
# Запуск VM сервиса
python vm_rag_service.py start

# Ожидаемый вывод в логах:
# ✅ RAG контекст установлен явно: vm
# 🔧 VM контекст установлен явно - все компоненты будут локальными
# ✅ Factory: Создан локальный QdrantVectorStore (VM контекст)
# ✅ Factory: Создан локальный CPUEmbedder (VM контекст)
```

### Шаг 6: Проверка health endpoint

```bash
# Проверить что сервис работает
curl http://localhost:8000/health

# Ожидаемый ответ:
# {
#   "status": "connected",
#   "services": {
#     "embedder": {"status": "connected", ...},
#     "vector_store": {"status": "connected", ...}
#   }
# }
```

### Шаг 7: Тестирование индексации

```bash
# Создать тестовый файл с документами
cat > test_docs.json << 'EOF'
{
  "documents": [
    {
      "id": "test1",
      "text": "Test document content",
      "metadata": {"source": "test"}
    }
  ],
  "batch_size": 512,
  "recreate": false
}
EOF

# Отправить запрос на индексацию
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d @test_docs.json

# Проверить логи
tail -f logs/diagnostics.log | grep "Factory\|VM:"

# Ожидается:
# ✅ Factory: Создан локальный QdrantVectorStore (VM контекст)
# 📥 VM: Получено 1 документов
# ✅ Батч 1: проиндексировано 1 точек
# 
# НЕ должно быть повторных "📥 VM: Получено 1 документов"
```

---

## ✅ Критерии успеха

### 1. Логи сервиса

```bash
grep "Factory" logs/diagnostics.log
```

**Ожидается:**
```
✅ RAG контекст установлен явно: vm
✅ Factory: Создан локальный QdrantVectorStore (VM контекст)
✅ Factory: Создан локальный CPUEmbedder (VM контекст)
```

**НЕ должно быть:**
```
❌ Factory: Создан RemoteVMVectorStore (CLIENT контекст)
```

### 2. Отсутствие рекурсии

```bash
# Мониторинг индексации
tail -f logs/diagnostics.log | grep "📥 VM:"

# При индексации батча из N документов должно быть:
# ✅ ОДНО сообщение "📥 VM: Получено N документов"
# 
# НЕ должно быть повторений (признак рекурсии)
```

### 3. Успешная индексация

```bash
# Проверка коллекции
curl http://localhost:8000/collection/info

# Ожидается рост vector_count после индексации
```

---

## 🔍 Диагностика проблем

### Проблема: Контекст определяется как CLIENT вместо VM

**Симптомы:**
```
✅ Factory: Создан RemoteVMVectorStore (CLIENT контекст)
```

**Решение 1: Явная установка через env**
```bash
export RAG_EXECUTION_CONTEXT=vm
python vm_rag_service.py start
```

**Решение 2: Проверить детекцию**
```python
# В Python консоли на VM
from rag.context import get_context_info
info = get_context_info()
print(info)

# Проверить:
# - qdrant_local_available должен быть True
# - hostname должен содержать 'vm' или 'ubuntu'
```

### Проблема: Рекурсия всё ещё происходит

**Диагностика:**
```bash
# Проверить тип vector_store
grep "type(vector_store)" logs/diagnostics.log

# Должно быть: QdrantVectorStore
# НЕ должно быть: RemoteVMVectorStore
```

**Решение:**
```bash
# Проверить что vm_rag_service.py устанавливает контекст
grep "RAGFactory.set_context" vm_rag_service.py

# Должно быть:
# RAGFactory.set_context(ExecutionContext.VM)
```

---

## 🔄 План отката (Rollback)

Если что-то пошло не так:

### Вариант A: Быстрый откат

```bash
# Остановить сервис
python vm_rag_service.py stop

# Восстановить из backup
tar -xzf backups/pre_factory_pattern_YYYYMMDD_HHMMSS.tar.gz

# Установить временное решение
export FORCE_LOCAL_VECTOR_STORE=true
echo 'export FORCE_LOCAL_VECTOR_STORE=true' >> ~/.bashrc

# Запустить сервис
python vm_rag_service.py start
```

### Вариант B: Полный откат через git

```bash
# Если используется git
git checkout <previous_commit_hash>

# Перезапуск
python vm_rag_service.py stop
python vm_rag_service.py start
```

---

## 📊 Мониторинг после deployment

### Логи для мониторинга

```bash
# 1. Factory логи
tail -f logs/diagnostics.log | grep "Factory"

# 2. Контекст детекция
tail -f logs/diagnostics.log | grep "Контекст:"

# 3. Индексация
tail -f logs/diagnostics.log | grep "📥 VM:"

# 4. Ошибки
tail -f logs/diagnostics.log | grep "ERROR\|❌"
```

### Метрики производительности

```bash
# Статистика сервиса
curl http://localhost:8000/stats

# Ожидается:
# - Нормальное время индексации
# - Отсутствие timeout ошибок
# - Линейный рост indexed_count
```

---

## 🎯 Post-deployment Tasks

### Немедленно (критично)

- [ ] Проверить логи на наличие "Factory: Создан локальный"
- [ ] Выполнить тест индексации (небольшой батч)
- [ ] Убедиться что нет рекурсии

### В течение часа

- [ ] Мониторинг логов на ошибки
- [ ] Проверка производительности
- [ ] Тест полной индексации репозитория

### В течение дня

- [ ] Удалить временное решение FORCE_LOCAL_VECTOR_STORE из кода
- [ ] Удалить прототип файлы (*_prototype.py)
- [ ] Обновить README с новым API

### В течение недели

- [ ] Мониторинг стабильности
- [ ] Сбор метрик производительности
- [ ] Обратная связь от пользователей

---

## 📞 Контакты

**При проблемах:**
1. Проверьте логи: `logs/diagnostics.log`
2. Используйте `get_context_info()` для диагностики
3. Проверьте health endpoint: `curl http://localhost:8000/health`
4. При необходимости - rollback к временному решению

**Документация:**
- Technical Spec: [`docs/RECURSION_FIX_FACTORY_PATTERN_SPEC.md`](RECURSION_FIX_FACTORY_PATTERN_SPEC.md)
- Основная документация: [`rules/RECURSION_FIX_2025_10_07.md`](../rules/RECURSION_FIX_2025_10_07.md)

---

**Подготовлено:** Roo (Code Mode)  
**Дата:** 2025-10-07  
**Статус:** Готово к deployment ✅