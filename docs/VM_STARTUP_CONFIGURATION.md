# Конфигурация запуска сервиса на VM

## Скрипт запуска с переменными окружения

Для ограничения использования CPU потоков Jina v3 embedder необходимо создать скрипт запуска с переменными окружения.

### Создание скрипта

На VM выполните:

```bash
cat > ~/repo_sum_rag/repo_sum/start_vm_rag.sh << 'EOF'
#!/bin/bash
cd ~/repo_sum_rag/repo_sum
source venv/bin/activate

# Ограничение потоков для Jina v3 (Фаза 2 OOM refactor)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Запуск сервиса
nohup python vm_rag_service.py > rag_service.log 2>&1 &
echo $! > rag_service.pid

echo "VM RAG Service started with PID: $(cat rag_service.pid)"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "TORCH_NUM_THREADS=$TORCH_NUM_THREADS"
EOF

chmod +x ~/repo_sum_rag/repo_sum/start_vm_rag.sh
```

### Запуск сервиса

```bash
cd ~/repo_sum_rag/repo_sum
./start_vm_rag.sh
```

### Остановка сервиса

```bash
kill $(cat ~/repo_sum_rag/repo_sum/rag_service.pid)
```

### Перезапуск сервиса

```bash
cd ~/repo_sum_rag/repo_sum
kill $(cat rag_service.pid) 2>/dev/null
sleep 2
./start_vm_rag.sh
```

### Проверка логов

```bash
tail -f ~/repo_sum_rag/repo_sum/rag_service.log
```

### Верификация переменных окружения

После запуска проверьте что переменные применены:

```bash
PID=$(cat ~/repo_sum_rag/repo_sum/rag_service.pid)
cat /proc/$PID/environ | tr '\0' '\n' | grep -E 'OMP|MKL|TORCH'
```

Ожидаемый вывод:
```
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TORCH_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

## Интеграция с vm_start.py

Скрипт `start_vm_rag.sh` создаётся локально на VM и не хранится в git. 

После обновления кода через `vm_start.py`:
1. Скрипт `start_vm_rag.sh` останется на месте (не в git)
2. Используйте `./start_vm_rag.sh` для запуска с правильными переменными

## Swap конфигурация

Swap файл создан на VM и настроен системно:
- Размер: 32GB
- Расположение: `/swapfile`
- swappiness: 10
- Автозапуск: `/etc/fstab`

Эти настройки не зависят от git и сохраняются между перезагрузками.

## Troubleshooting

### Проблема: Скрипт не найден после git pull

**Решение:** Скрипт в `.gitignore`, создайте заново следуя инструкциям выше.

### Проблема: Переменные не применяются

**Решение:** 
1. Остановите старый процесс
2. Запустите через `./start_vm_rag.sh` (не `python vm_rag_service.py` напрямую)
3. Проверьте переменные через `/proc/PID/environ`

### Проблема: Permission denied

**Решение:**
```bash
chmod +x ~/repo_sum_rag/repo_sum/start_vm_rag.sh