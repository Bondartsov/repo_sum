# VM RAG Service - CLI интерфейс

**Дата:** 01.10.2025
**Статус:** ✅ РЕАЛИЗОВАНО

---

## Проблема

При запуске `python vm_rag_service.py status` на VM возникала ошибка:
```
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```

**Причина:** Скрипт не поддерживал аргументы командной строки и всегда пытался запустить сервер заново.

---

## Решение

Добавлен полноценный CLI интерфейс с поддержкой команд `start`, `stop`, `status`.

---

## Использование

### 1. Запуск сервиса

```bash
# Вариант 1 (по умолчанию)
python vm_rag_service.py

# Вариант 2 (явно)
python vm_rag_service.py start
```

**Что делает:**
- Инициализирует все сервисы (Jina v3, Qdrant, SearchService, IndexerService)
- Запускает FastAPI сервер на `0.0.0.0:8000`
- Прогревает модель (если enabled в конфиге)

---

### 2. Проверка статуса

```bash
python vm_rag_service.py status
```

**Что делает:**
- Выполняет HTTP GET запрос к `http://localhost:8000/health`
- Показывает статус всех компонентов
- Отображает количество векторов в коллекции

**Пример вывода:**
```
✅ Сервис работает
📊 Статус: connected
🕐 Время: 2025-10-01T19:41:27.123456+00:00

📦 Компоненты:
  ✅ embedder: connected
  ✅ vector_store: connected

📚 Векторов в коллекции: 1234
```

**Exit codes:**
- `0` - сервис работает корректно
- `1` - сервис недоступен или ошибка

---

### 3. Остановка сервиса

```bash
python vm_rag_service.py stop
```

**Что делает:**
- Ищет процесс на порту 8000 (через `lsof -ti:8000`)
- Останавливает процесс через `SIGTERM`
- Fallback: если `lsof` недоступен, использует `ps aux | grep`

**Пример вывода:**
```
🔍 Найдено процессов: 1
✅ Процесс 3342267 остановлен
✅ Сервис остановлен
```

---

## Реализованные функции

### `check_service_status()` ✅
- HTTP запрос к `/health` endpoint
- Парсинг и красивый вывод статуса
- Обработка ConnectionError (сервис не запущен)

### `stop_service()` ✅
- Поиск процесса через `lsof` (основной метод)
- Fallback через `ps aux | grep` (если lsof недоступен)
- Graceful shutdown через `SIGTERM`
- Обработка ProcessLookupError

### `start_service()` ✅
- Запуск uvicorn сервера
- Логирование старта
- Обработка ошибок запуска

---

## Интеграция с vm_start.py

Теперь `vm_start.py status` корректно проверяет статус:

```bash
python vm_start.py status
```

**Проверяет:**
1. SSH подключение к VM ✅
2. Python, память, репозиторий ✅
3. Qdrant статус ✅
4. **RAG Service статус** ✅ (через HTTP GET /health, не запуская новый процесс)

---

## Технические детали

### Зависимости
- `requests` - для HTTP запросов в `status`
- `argparse` - для CLI парсинга
- `subprocess` - для `lsof` и `ps`
- `signal` - для graceful shutdown

### Безопасность
- Все команды работают локально (localhost)
- `SIGTERM` для graceful shutdown (не `SIGKILL`)
- Timeout 5 секунд для HTTP запросов

### Кроссплатформенность
- Основной метод: `lsof` (Linux/Unix)
- Fallback: `ps aux` (широкая совместимость)
- Windows: не поддерживается (VM работает на Ubuntu)

---

## Тестирование

### Сценарий 1: Первый запуск
```bash
# На VM
python vm_rag_service.py status  # → ❌ Сервис не запущен
python vm_rag_service.py start   # → Запуск сервера
# (в новом терминале)
python vm_rag_service.py status  # → ✅ Сервис работает
```

### Сценарий 2: Остановка и перезапуск
```bash
python vm_rag_service.py stop    # → ✅ Сервис остановлен
python vm_rag_service.py status  # → ❌ Сервис не запущен
python vm_rag_service.py start   # → Запуск сервера
```

### Сценарий 3: Проверка через vm_start.py
```bash
# С локальной машины
python vm_start.py status        # → Все компоненты ✅
```

---

## Известные ограничения

1. **Windows не поддерживается** - команды `lsof` и `ps` специфичны для Unix
2. **Порт захардкожен** - всегда используется порт 8000
3. **Один экземпляр** - не поддерживается запуск нескольких инстансов

---

## Будущие улучшения

1. ⏳ Добавить `restart` команду
2. ⏳ Поддержка конфигурируемого порта
3. ⏳ Логирование в файл (не только console)
4. ⏳ Systemd unit файл для автозапуска

---

**Автор:** AI Assistant  
**Дата реализации:** 01.10.2025, 22:58  
**Файлы изменены:** `vm_rag_service.py` (+137 строк)
