#!/bin/bash
################################################################################
# Скрипт настройки VM - Фаза 2: Быстрые стабилизаторы
# Задача: Создание swap, настройка потоков, ограничение uvicorn
# 
# ВНИМАНИЕ: Выполнять на VM 10.61.11.54 от пользователя с sudo правами!
################################################################################

set -e  # Остановка при ошибке

echo "=============================================================="
echo "ФАЗА 2: БЫСТРЫЕ СТАБИЛИЗАТОРЫ - ДИАГНОСТИКА И НАСТРОЙКА"
echo "=============================================================="
echo ""

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция логирования
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

################################################################################
# ШАГ 1: ДИАГНОСТИКА ТЕКУЩЕГО СОСТОЯНИЯ
################################################################################

echo ""
log_info "Шаг 1: Диагностика текущего состояния системы"
echo "--------------------------------------------------------------"

echo ""
echo "[ПАМЯТЬ]"
free -h

echo ""
echo "[SWAP - ТЕКУЩЕЕ СОСТОЯНИЕ]"
swapon --show || echo "(swap не активен)"

echo ""
echo "[SWAP - /proc/swaps]"
cat /proc/swaps || echo "(swap не настроен)"

echo ""
echo "[ДИСК - КОРЕНЬ]"
df -h /

echo ""
echo "[VM SWAPPINESS]"
cat /proc/sys/vm/swappiness

echo ""
echo "[ПРОЦЕССЫ PYTHON/UVICORN]"
ps aux | grep -E "python|uvicorn" | grep -v grep || echo "(процессы не найдены)"

echo ""
read -p "Продолжить с настройкой? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_warn "Настройка отменена пользователем"
    exit 0
fi

################################################################################
# ШАГ 2: СОЗДАНИЕ SWAP 64GB
################################################################################

echo ""
log_info "Шаг 2: Создание swap файла 64GB"
echo "--------------------------------------------------------------"

# Проверка существования swap
if [ -f /swapfile ]; then
    log_warn "Файл /swapfile уже существует"
    read -p "Пересоздать swap? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Отключение существующего swap..."
        sudo swapoff /swapfile 2>/dev/null || true
        sudo rm /swapfile
    else
        log_info "Пропуск создания swap"
        SKIP_SWAP=1
    fi
fi

if [ -z "$SKIP_SWAP" ]; then
    # Проверка свободного места
    FREE_SPACE=$(df / | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=$((64 * 1024 * 1024))  # 64GB в KB
    
    if [ "$FREE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        log_error "Недостаточно места на диске! Требуется ~64GB, доступно: $(($FREE_SPACE / 1024 / 1024))GB"
        log_warn "Попытка создать swap меньшего размера (32GB)..."
        SWAP_SIZE="32G"
    else
        SWAP_SIZE="64G"
    fi
    
    log_info "Создание swap файла размером $SWAP_SIZE..."
    sudo fallocate -l $SWAP_SIZE /swapfile
    
    log_info "Установка прав доступа 600..."
    sudo chmod 600 /swapfile
    
    log_info "Форматирование swap..."
    sudo mkswap /swapfile
    
    log_info "Активация swap..."
    sudo swapon /swapfile
    
    log_info "Добавление записи в /etc/fstab для автозапуска..."
    if ! grep -q "/swapfile" /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    
    log_info "Swap создан и активирован!"
fi

# Настройка swappiness
log_info "Настройка vm.swappiness=10 (минимальное использование swap)..."
sudo sysctl vm.swappiness=10

log_info "Добавление в /etc/sysctl.conf для постоянной настройки..."
if ! grep -q "vm.swappiness=10" /etc/sysctl.conf; then
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
fi

echo ""
echo "[SWAP - ПОСЛЕ НАСТРОЙКИ]"
swapon --show
free -h

################################################################################
# ШАГ 3: ПОИСК И НАСТРОЙКА SYSTEMD UNIT ФАЙЛА
################################################################################

echo ""
log_info "Шаг 3: Поиск и настройка systemd unit файла сервиса"
echo "--------------------------------------------------------------"

# Поиск FastAPI/uvicorn сервисов
log_info "Поиск сервисов FastAPI/uvicorn/embedder..."
SERVICES=$(systemctl list-units --type=service --all | grep -E 'fastapi|uvicorn|embedder|rag' | awk '{print $1}' || true)

if [ -z "$SERVICES" ]; then
    log_warn "Сервисы не найдены по ключевым словам"
    log_info "Список всех активных сервисов:"
    systemctl list-units --type=service --state=running | grep -v "^●" | head -20
    
    echo ""
    read -p "Введите имя сервиса для настройки (или Enter для пропуска): " SERVICE_NAME
    
    if [ -z "$SERVICE_NAME" ]; then
        log_warn "Настройка systemd unit пропущена"
        SKIP_SYSTEMD=1
    else
        SERVICES="$SERVICE_NAME"
    fi
fi

if [ -z "$SKIP_SYSTEMD" ]; then
    for service in $SERVICES; do
        log_info "Найден сервис: $service"
        
        # Получение пути к unit файлу
        UNIT_FILE=$(systemctl show -p FragmentPath $service | cut -d= -f2)
        
        if [ -n "$UNIT_FILE" ] && [ -f "$UNIT_FILE" ]; then
            log_info "Unit файл: $UNIT_FILE"
            
            echo ""
            echo "Текущая конфигурация [Service]:"
            sudo grep -A 20 "^\[Service\]" "$UNIT_FILE" || true
            
            echo ""
            read -p "Добавить переменные окружения для ограничения потоков? (y/N): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Создание бэкапа
                BACKUP_FILE="${UNIT_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
                log_info "Создание бэкапа: $BACKUP_FILE"
                sudo cp "$UNIT_FILE" "$BACKUP_FILE"
                
                # Добавление переменных окружения
                log_info "Добавление переменных окружения..."
                
                # Проверка, есть ли уже эти переменные
                if sudo grep -q "OMP_NUM_THREADS" "$UNIT_FILE"; then
                    log_warn "Переменные окружения уже присутствуют, пропуск..."
                else
                    # Добавление после [Service]
                    sudo sed -i '/^\[Service\]/a \
Environment="OMP_NUM_THREADS=1"\
Environment="MKL_NUM_THREADS=1"\
Environment="TORCH_NUM_THREADS=1"\
Environment="OPENBLAS_NUM_THREADS=1"' "$UNIT_FILE"
                    
                    log_info "Переменные окружения добавлены!"
                fi
                
                # Перезагрузка конфигурации
                log_info "Перезагрузка systemd daemon..."
                sudo systemctl daemon-reload
                
                echo ""
                read -p "Перезапустить сервис $service? (y/N): " -n 1 -r
                echo
                
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    log_info "Перезапуск сервиса..."
                    sudo systemctl restart $service
                    
                    sleep 2
                    
                    log_info "Статус сервиса:"
                    sudo systemctl status $service --no-pager || true
                fi
                
                log_info "Сервис $service настроен!"
            fi
        else
            log_warn "Unit файл не найден для $service"
        fi
    done
fi

################################################################################
# ШАГ 4: НАСТРОЙКА ПАРАМЕТРОВ UVICORN
################################################################################

echo ""
log_info "Шаг 4: Проверка параметров запуска uvicorn"
echo "--------------------------------------------------------------"

log_info "Поиск vm_start.py..."
if [ -f "vm_start.py" ]; then
    log_info "Найден vm_start.py в текущей директории"
    echo ""
    echo "Текущие параметры uvicorn в vm_start.py:"
    grep -A 5 "uvicorn" vm_start.py || true
    
    log_warn "Для изменения параметров uvicorn отредактируйте vm_start.py вручную"
    log_info "Целевые параметры:"
    echo "  --workers 1"
    echo "  --limit-concurrency 2"
    echo "  --timeout-keep-alive 600"
else
    log_warn "vm_start.py не найден в текущей директории"
fi

echo ""
log_info "Текущие процессы uvicorn:"
ps aux | grep uvicorn | grep -v grep || echo "(процессы не найдены)"

################################################################################
# ШАГ 5: ВЕРИФИКАЦИЯ
################################################################################

echo ""
log_info "Шаг 5: Верификация изменений"
echo "--------------------------------------------------------------"

echo ""
echo "[SWAP АКТИВЕН]"
swapon --show

echo ""
echo "[ПАМЯТЬ]"
free -h

echo ""
echo "[VM.SWAPPINESS]"
cat /proc/sys/vm/swappiness

echo ""
echo "[ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ СЕРВИСОВ]"
if [ -n "$SERVICES" ] && [ -z "$SKIP_SYSTEMD" ]; then
    for service in $SERVICES; do
        echo "Сервис: $service"
        sudo systemctl show $service --property=Environment | grep -E "OMP|MKL|TORCH|OPENBLAS" || echo "  (переменные не найдены)"
    done
fi

################################################################################
# ЗАВЕРШЕНИЕ
################################################################################

echo ""
echo "=============================================================="
log_info "ФАЗА 2: НАСТРОЙКА ЗАВЕРШЕНА!"
echo "=============================================================="
echo ""
log_info "Следующие шаги:"
echo "  1. Проверьте, что swap активен и используется минимально"
echo "  2. Убедитесь, что сервис перезапущен с новыми переменными"
echo "  3. Отредактируйте vm_start.py для настройки uvicorn (если необходимо)"
echo "  4. Обновите документацию rules/rerfactor_oom.md"
echo ""
log_warn "ВАЖНО: Мониторьте использование памяти после изменений!"
echo ""