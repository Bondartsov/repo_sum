# Фаза 2: Быстрые стабилизаторы - Руководство по развёртыванию

**Дата:** 03 октября 2025  
**Критичность:** P0 (Критическая)  
**VM:** 10.61.11.54 (t-ubuntu-redis, 60GB RAM)  
**Проблема:** OOM killer при 99% RAM, swap thrashing

---

## 📋 Обзор

Данное руководство содержит пошаговые инструкции для выполнения **Фазы 2 - Быстрые стабилизаторы** на VM для решения критической проблемы с OOM killer.

### Что будет сделано:
1. ✅ Создание swap файла 64GB с swappiness=10
2. ✅ Настройка переменных окружения для ограничения CPU потоков
3. ✅ Конфигурация uvicorn для ограничения конкурентности
4. ✅ Верификация изменений

---

## 🚀 Метод 1: Автоматическое развёртывание (Рекомендуется)

### Шаг 1: Копирование скрипта на VM

```bash
# На локальной машине
scp scripts/vm_phase2_setup.sh user@10.61.11.54:~/vm_phase2_setup.sh
```

### Шаг 2: Запуск скрипта на VM

```bash
# Подключение к VM
ssh user@10.61.11.54

# Сделать скрипт исполняемым
chmod +x ~/vm_phase2_setup.sh

# Запустить скрипт
sudo bash ~/vm_phase2_setup.sh
```

Скрипт интерактивно проведёт через все шаги Фазы 2.

---

## 🛠️ Метод 2: Ручное развёртывание

Если автоматический скрипт не работает, выполните команды вручную.

### Шаг 1: Диагностика текущего состояния

```bash
# Подключитесь к VM
ssh user@10.61.11.54

# Проверка памяти
free -h

# Проверка swap
swapon --show
cat /proc/swaps

# Проверка доступного места на диске
df -h /

# Проверка swappiness
cat /proc/sys/vm/swappiness
```

**Ожидаемый результат:**
- Swap должен отсутствовать или быть недостаточным
- Доступно место на диске: минимум 65GB
- swappiness: обычно 60 (по умолчанию)

---

### Шаг 2: Создание swap файла 64GB

```bash
# Создание swap файла
sudo fallocate -l 64G /swapfile

# Установка правильных прав доступа
sudo chmod 600 /swapfile

# Форматирование как swap
sudo mkswap /swapfile

# Активация swap
sudo swapon /swapfile

# Проверка
swapon --show
free -h
```

**Добавление в /etc/fstab для автозапуска:**

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Настройка swappiness=10:**

```bash
# Установить swappiness на текущую сессию
sudo sysctl vm.swappiness=10

# Добавить в конфигурацию для постоянного применения
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Проверка
cat /proc/sys/vm/swappiness
```

---

### Шаг 3: Поиск systemd unit файла сервиса

```bash
# Поиск сервисов FastAPI/uvicorn
systemctl list-units --type=service | grep -E 'fastapi|uvicorn|embedder|rag'

# Если ничего не найдено, проверьте все активные сервисы
systemctl list-units --type=service --state=running

# Просмотр конфигурации найденного сервиса (замените на ваше имя)
systemctl cat <service-name>
```

**Если сервис найден:**

```bash
# Пример: embedder.service или rag-service.service
SERVICE_NAME="<your-service-name>"

# Получение пути к unit файлу
UNIT_FILE=$(systemctl show -p FragmentPath $SERVICE_NAME | cut -d= -f2)
echo "Unit файл: $UNIT_FILE"

# Создание бэкапа
sudo cp "$UNIT_FILE" "${UNIT_FILE}.backup_$(date +%Y%m%d_%H%M%S)"

# Редактирование unit файла (добавьте переменные окружения в секцию [Service])
sudo nano "$UNIT_FILE"
```

**Добавьте эти строки в секцию `[Service]`:**

```ini
Environment="OMP_NUM_THREADS=1"
Environment="MKL_NUM_THREADS=1"
Environment="TORCH_NUM_THREADS=1"
Environment="OPENBLAS_NUM_THREADS=1"
```

**Применение изменений:**

```bash
# Перезагрузка конфигурации systemd
sudo systemctl daemon-reload

# Перезапуск сервиса
sudo systemctl restart $SERVICE_NAME

# Проверка статуса
sudo systemctl status $SERVICE_NAME

# Проверка переменных окружения
sudo systemctl show $SERVICE_NAME --property=Environment | grep -E "OMP|MKL|TORCH|OPENBLAS"
```

---

### Шаг 4: Обновление параметров uvicorn в vm_rag_service.py

**На VM, перейдите в директорию репозитория:**

```bash
cd ~/repo_sum_rag/repo_sum  # или путь к вашему репозиторию
```

**Создайте бэкап:**

```bash
cp vm_rag_service.py vm_rag_service.py.backup_phase2
```

**Применение патча (если доступен файл патча):**

```bash
# Если патч скопирован на VM
patch -p0 < scripts/vm_rag_service_phase2.patch
```

**Или отредактируйте вручную:**

```bash
nano vm_rag_service.py
```

Найдите функцию `start_service()` (строка ~558) и измените `uvicorn.run()`:

```python
def start_service():
    """Запуск сервиса"""
    logger.info("🚀 Запуск RAG-as-a-Service на VM...")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=1,  # ФАЗА 2: Один worker
            limit_concurrency=2,  # ФАЗА 2: Ограничение конкурентности
            timeout_keep_alive=600,  # ФАЗА 2: Увеличенный timeout
            limit_max_requests=1000,  # ФАЗА 2: Защита от утечек памяти
            backlog=10,  # ФАЗА 2: Ограничение очереди
            log_level="info",
            access_log=True,
            use_colors=False  # ФАЗА 2: Для systemd логов
        )
    except Exception as e:
        logger.error(f"❌ Ошибка запуска сервиса: {e}")
        sys.exit(1)
```

**Перезапуск сервиса:**

```bash
# Если через systemd
sudo systemctl restart <service-name>

# Или если запускается напрямую
pkill -f vm_rag_service
python vm_rag_service.py start
```

---

### Шаг 5: Верификация изменений

```bash
# 1. Проверка swap
echo "=== SWAP STATUS ==="
swapon --show
free -h

# 2. Проверка swappiness
echo "=== VM SWAPPINESS ==="
cat /proc/sys/vm/swappiness

# 3. Проверка переменных окружения сервиса
echo "=== SERVICE ENVIRONMENT ==="
sudo systemctl show <service-name> --property=Environment | grep -E "OMP|MKL|TORCH|OPENBLAS"

# 4. Проверка процессов uvicorn
echo "=== UVICORN PROCESSES ==="
ps aux | grep uvicorn | grep -v grep

# 5. Проверка использования памяти
echo "=== MEMORY USAGE ==="
free -h | grep "Mem:"

# 6. Проверка health endpoint
echo "=== SERVICE HEALTH ==="
curl -s http://localhost:8000/health | jq .
```

**Ожидаемые результаты:**
- ✅ Swap: 64GB активен, использование минимальное
- ✅ swappiness: 10
- ✅ Переменные окружения: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, etc.
- ✅ Процесс uvicorn: один worker процесс
- ✅ Память: <90% использования
- ✅ Health endpoint: status="connected"

---

## 📊 Мониторинг после развёртывания

### Проверка логов

```bash
# Systemd логи (если через systemd)
sudo journalctl -u <service-name> -f

# Или логи приложения
tail -f ~/repo_sum_rag/repo_sum/rag_service.log
```

### Мониторинг памяти в реальном времени

```bash
# Каждые 2 секунды
watch -n 2 'free -h; echo "---"; swapon --show'

# Или с более детальной информацией
watch -n 2 'free -h; echo "---"; ps aux | grep -E "python|uvicorn" | grep -v grep | head -5'
```

### Проверка swap использования

```bash
# Должно быть минимальное использование swap при нормальной работе
cat /proc/swaps
```

---

## 🔄 Rollback инструкции

Если что-то пошло не так:

### Откат swap

```bash
# Отключить swap
sudo swapoff /swapfile

# Удалить swap файл
sudo rm /swapfile

# Удалить запись из /etc/fstab
sudo nano /etc/fstab  # Удалите строку с /swapfile

# Восстановить старый swappiness (обычно 60)
sudo sysctl vm.swappiness=60
```

### Откат systemd конфигурации

```bash
# Восстановить из бэкапа
sudo cp "${UNIT_FILE}.backup_*" "$UNIT_FILE"
sudo systemctl daemon-reload
sudo systemctl restart <service-name>
```

### Откат vm_rag_service.py

```bash
cd ~/repo_sum_rag/repo_sum
cp vm_rag_service.py.backup_phase2 vm_rag_service.py
# Перезапустить сервис
```

---

## ✅ Критерии успеха Фазы 2

- [x] Swap 64GB создан и активен
- [x] swappiness=10 установлен
- [x] Переменные окружения OMP/MKL/TORCH установлены
- [x] uvicorn настроен с workers=1, limit_concurrency=2
- [x] Сервис стабильно работает
- [x] Память не превышает 90% в обычном режиме
- [x] Отсутствуют OOM события в dmesg
- [x] Документация обновлена

---

## 📞 Поддержка

При проблемах:

1. Проверьте логи: `sudo journalctl -u <service> -n 100`
2. Проверьте dmesg на OOM: `sudo dmesg -T | grep -i "killed\|oom"`
3. Проверьте здоровье сервиса: `curl http://localhost:8000/health`
4. См. документацию: [`rules/rerfactor_oom.md`](../rules/rerfactor_oom.md)

---

## 📝 Следующие шаги

После успешного завершения Фазы 2:

1. Мониторинг стабильности 24-48 часов
2. Переход к Фазе 3: Чанкование кода (CHUNK_MAX_TOKENS=768)
3. Фаза 4: Стримовая индексация
4. Обновление baseline метрик

---

**Дата создания:** 03.10.2025  
**Версия:** 1.0  
**Статус:** Ready for deployment