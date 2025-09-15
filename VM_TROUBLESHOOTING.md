# 🔧 Troubleshooting: VM Migration для Jina v3

**Дата:** 15 сентября 2025  
**Назначение:** Решение проблем при миграции Jina v3 на VM

---

## 🎯 Общие проблемы и решения

### 🔐 SSH и подключение

#### ❌ "Permission denied (publickey,password)"
**Решение:**
```bash
# Проверьте SSH ключи
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
ssh-copy-id user@10.61.11.54

# Или используйте пароль в .env файле:
VM_PASSWORD=your_actual_password
```

#### ❌ "Connection refused"
**Причины и решения:**
1. **VM выключена** - убедитесь что t-ubuntu-redis запущена
2. **Firewall блокирует SSH** - проверьте порт 22
3. **Неправильный IP** - убедитесь что 10.61.11.54 доступен

#### ❌ "Host key verification failed"
**Решение:**
```bash
ssh-keyscan -H 10.61.11.54 >> ~/.ssh/known_hosts
# Или используйте флаг -o StrictHostKeyChecking=no (не рекомендуется для production)
```

---

## 🐍 Python и зависимости

#### ❌ "ensurepip is not available" (Ubuntu 22.04)
**Причина:** Отсутствует python3.10-venv пакет
**Решение:**
```bash
sudo apt update
sudo apt install -y python3.10-venv python3.10-distutils python3-pip
```

#### ❌ "ModuleNotFoundError: No module named 'venv'"
**Решение:**
```bash
# Альтернативный способ создания venv
python3 -m venv venv --without-pip
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
pip install --upgrade pip setuptools wheel
```

#### ❌ "pip: command not found"
**Решение:**
```bash
# Установка pip
sudo apt install -y python3-pip
# Или через curl
curl https://bootstrap.pypa.io/get-pip.py | python3
```

---

## 🧠 Jina v3 специфичные проблемы

#### ❌ "trust_remote_code=True не работает"
**Проверка версии sentence-transformers:**
```bash
pip install sentence-transformers>=3.0
python -c "import sentence_transformers; print(sentence_transformers.__version__)"
```

#### ❌ "No module named 'transformers_modules'"
**Очистка кэша HuggingFace:**
```bash
rm -rf ~/.cache/huggingface/transformers
rm -rf ~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3
```

#### ❌ "[Errno 2] No such file or directory: 'block.py'"
**Принудительная перезагрузка модели:**
```bash
python -c "
import shutil
shutil.rmtree('~/.cache/huggingface', ignore_errors=True)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True, force_download=True)
print('✅ Модель загружена успешно')
"
```

#### ❌ "CUDA out of memory" (несмотря на CPU режим)
**Принудительный CPU режим:**
```bash
export CUDA_VISIBLE_DEVICES=""
python -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')"
```

---

## 💾 Проблемы памяти

#### ❌ "MemoryError" или "out of memory"
**Проверка доступной памяти:**
```bash
free -h
cat /proc/meminfo | grep MemAvailable
```

**Оптимизация:**
```bash
# Уменьшите batch size
export EMBEDDING_BATCH_SIZE_MAX=16
export TORCH_NUM_THREADS=2
```

#### ❌ "Process killed" при загрузке модели
**Проверка OOM killer:**
```bash
dmesg | grep -i "killed process"
# Если есть записи - нужно больше RAM или swap
```

**Создание swap файла:**
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 🗄️ Qdrant проблемы

#### ❌ "Connection refused" к Qdrant
**Проверка статуса:**
```bash
curl http://localhost:6333/collections
# Или через netstat
netstat -tulpn | grep 6333
```

**Перезапуск Qdrant:**
```bash
# Если через Docker
docker restart $(docker ps -q --filter ancestor=qdrant/qdrant)

# Если systemd
sudo systemctl restart qdrant
```

#### ❌ "Collection not found"
**Проверка коллекций:**
```bash
curl http://localhost:6333/collections | jq
```

**Пересоздание коллекции:**
```bash
python main.py rag index /path/to/repo --recreate
```

---

## 🌐 Сетевые проблемы

#### ❌ "Timeout downloading model"
**Проверка интернета:**
```bash
ping huggingface.co
curl -I https://huggingface.co/jinaai/jina-embeddings-v3
```

**Настройка proxy (если нужно):**
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

#### ❌ "SSL certificate verify failed"
**Обход SSL (временно):**
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org sentence-transformers
```

---

## 🔬 Диагностические команды

### Системная информация
```bash
# Общая информация о системе
uname -a
cat /etc/os-release
python3 --version
pip --version

# Память и CPU
free -h
nproc
cat /proc/cpuinfo | grep "model name" | head -1
```

### Python окружение
```bash
# Установленные пакеты
pip list | grep -E "(torch|transformers|sentence|qdrant|paramiko)"

# Путь к Python и venv
which python3
echo $VIRTUAL_ENV

# Проверка импортов
python -c "import sentence_transformers, transformers, qdrant_client; print('✅ Все модули импортируются')"
```

### Тест загрузки Jina v3
```bash
# Минимальный тест
python -c "
import torch
print(f'🔧 PyTorch: {torch.__version__}')
print(f'💾 CUDA доступна: {torch.cuda.is_available()}')

from sentence_transformers import SentenceTransformer
print('📥 Загружаем Jina v3...')
model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
print(f'✅ Размерность: {model.get_sentence_embedding_dimension()}d')
print('🎉 Тест успешен!')
"
```

---

## 📋 Чеклист диагностики

### Перед обращением за помощью:
- [ ] ✅ VM доступна по SSH
- [ ] ✅ Python 3.9+ установлен
- [ ] ✅ pip и venv работают
- [ ] ✅ Интернет доступен для загрузки моделей
- [ ] ✅ Достаточно RAM (>25GB свободно)
- [ ] ✅ sentence-transformers>=3.0 установлен
- [ ] ✅ HuggingFace кэш очищен
- [ ] ✅ Qdrant сервер запущен и доступен
- [ ] ✅ .env файл настроен с VM_PASSWORD

### Сбор информации для отчета об ошибке:
```bash
# Системная информация
echo "=== System Info ===" > debug_info.txt
uname -a >> debug_info.txt
free -h >> debug_info.txt
python3 --version >> debug_info.txt

# Версии пакетов
echo "=== Python Packages ===" >> debug_info.txt
pip list | grep -E "(torch|transformers|sentence|qdrant)" >> debug_info.txt

# Последние логи
echo "=== Recent Logs ===" >> debug_info.txt
tail -50 /var/log/syslog >> debug_info.txt

# Отправьте debug_info.txt вместе с описанием проблемы
```

---

## 🚀 Быстрые решения (FAQ)

### Q: Jina v3 не загружается - "File not found"
**A:** Очистите кэш HuggingFace и переустановите:
```bash
rm -rf ~/.cache/huggingface
pip install --upgrade --force-reinstall sentence-transformers
```

### Q: VM тормозит при загрузке модели
**A:** Проверьте swap и уменьшите batch size:
```bash
sudo swapon --show
export TORCH_NUM_THREADS=2
```

### Q: "Error 403" при загрузке с HuggingFace
**A:** Авторизуйтесь в HuggingFace:
```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

### Q: Paramiko "Authentication failed"
**A:** Проверьте пароль в .env и права:
```bash
# Проверьте переменную
echo $VM_PASSWORD
# Проверьте пароль напрямую
ssh user@10.61.11.54 "echo 'success'"
```

---

## 📞 Контакты

**VM Details:**
- Host: t-ubuntu-redis (10.61.11.54)
- User: user
- RAM: 31GB
- CPU: Intel Xeon Gold 6248R

**Документация:**
- План миграции: `JINA_V3_VM_MIGRATION_PLAN.md`
- Ручная настройка: `VM_MANUAL_SETUP.md`
- Быстрый старт: `VM_MIGRATION_QUICKSTART.md`

**Статус:** 🚀 Ready for critical Jina v3 test!
