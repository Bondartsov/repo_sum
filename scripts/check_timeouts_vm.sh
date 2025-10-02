#!/bin/bash
# Скрипт диагностики timeout параметров на VM

echo "================================================================================"
echo "🔍 ДИАГНОСТИКА TIMEOUT И RETRY ПАРАМЕТРОВ НА VM"
echo "================================================================================"

# Переходим в директорию проекта
cd ~/repo_sum_rag/repo_sum || exit 1

# 1. Environment Variables
echo ""
echo "================================================================================"
echo "1. Environment Variables"
echo "================================================================================"
if [ -f .env ]; then
    echo "📄 Найден .env файл:"
    grep -E "RAG_TIMEOUT|RAG_MAX_RETRIES|RAG_RETRY_DELAY|TIMEOUT|RETRY" .env 2>/dev/null || echo "❌ Нет переменных timeout/retry в .env"
else
    echo "❌ .env файл не найден"
fi

echo ""
echo "Текущие environment variables в shell:"
env | grep -E "RAG_TIMEOUT|RAG_MAX_RETRIES|RAG_RETRY_DELAY|TIMEOUT|RETRY" || echo "❌ Нет установленных переменных"

# 2. Config файлы
echo ""
echo "================================================================================"
echo "2. Config Files"
echo "================================================================================"

if [ -f config.py ]; then
    echo "📄 config.py - RemoteServiceConfig:"
    grep -A 3 "class RemoteServiceConfig" config.py
    echo ""
    grep -E "timeout_seconds|max_retries|retry_delay" config.py | grep -v "^#" | head -20
else
    echo "❌ config.py не найден"
fi

echo ""
if [ -f settings.json ]; then
    echo "📄 settings.json - remote_service:"
    python3 -c "import json; data=json.load(open('settings.json')); print(json.dumps(data.get('rag', {}).get('remote_service', {}), indent=2))" 2>/dev/null || echo "Ошибка чтения settings.json"
else
    echo "❌ settings.json не найден"
fi

# 3. Running Process Info
echo ""
echo "================================================================================"
echo "3. Running Service Process"
echo "================================================================================"

RAG_PID=$(pgrep -f "vm_rag_service.py" | head -1)

if [ -n "$RAG_PID" ]; then
    echo "✅ RAG Service запущен (PID: $RAG_PID)"
    echo ""
    echo "Process info:"
    ps aux | grep "$RAG_PID" | grep -v grep

    echo ""
    echo "Environment variables процесса:"
    cat /proc/$RAG_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "RAG_TIMEOUT|RAG_MAX_RETRIES|RAG_RETRY_DELAY|TIMEOUT|RETRY" || echo "❌ Нет timeout/retry переменных"

    echo ""
    echo "Start time процесса:"
    ps -p $RAG_PID -o lstart= 2>/dev/null || echo "N/A"
else
    echo "❌ RAG Service не запущен"
fi

# 4. Log файлы
echo ""
echo "================================================================================"
echo "4. Recent Log Entries (timeout related)"
echo "================================================================================"

if [ -f rag_service.log ]; then
    echo "📄 rag_service.log - последние упоминания timeout:"
    grep -i "timeout" rag_service.log | tail -10 || echo "Нет упоминаний timeout"
else
    echo "❌ rag_service.log не найден"
fi

# 5. System Resources
echo ""
echo "================================================================================"
echo "5. System Resources (Memory)"
echo "================================================================================"

echo "RAM usage:"
free -h

echo ""
echo "Top memory consumers:"
ps aux --sort=-%mem | head -6

# 6. Network Connectivity
echo ""
echo "================================================================================"
echo "6. Network & Service Status"
echo "================================================================================"

echo "FastAPI service (port 8000):"
netstat -tuln | grep ":8000" || echo "❌ Port 8000 не слушается"

echo ""
echo "Qdrant service (port 6333):"
netstat -tuln | grep ":6333" || echo "❌ Port 6333 не слушается"

echo ""
echo "Health check test:"
curl -s -o /dev/null -w "Status: %{http_code}, Time: %{time_total}s\n" http://localhost:8000/health 2>/dev/null || echo "❌ Health check failed"

# 7. Recommendations
echo ""
echo "================================================================================"
echo "7. Рекомендации"
echo "================================================================================"

# Проверяем config.py timeout
TIMEOUT_IN_CONFIG=$(grep "timeout_seconds.*=" config.py | grep -v "^#" | head -1 | grep -oP '\d+' | head -1)

if [ "$TIMEOUT_IN_CONFIG" = "600" ]; then
    echo "✅ HOTFIX применён в config.py (timeout_seconds = 600)"
elif [ "$TIMEOUT_IN_CONFIG" = "60" ]; then
    echo "❌ HOTFIX НЕ применён! config.py имеет старые значения (60s)"
    echo "   Рекомендация:"
    echo "   1. Обновить config.py с новыми значениями"
    echo "   2. Перезапустить сервис: pkill -f vm_rag_service.py && python3 vm_rag_service.py &"
else
    echo "⚠️  Timeout в config.py: $TIMEOUT_IN_CONFIG (ожидалось 600 или 60)"
fi

echo ""
if [ -n "$RAG_PID" ]; then
    START_TIME=$(ps -p $RAG_PID -o lstart= 2>/dev/null)
    echo "ℹ️  Service start time: $START_TIME"
    echo "   Если HOTFIX был применён ПОСЛЕ этого времени - требуется рестарт!"
fi

echo ""
echo "================================================================================"
echo "Диагностика завершена!"
echo "================================================================================"
