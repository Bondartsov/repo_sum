# Dependencies: Repository Analyzer

**Дата обновления:** 22 сентября 2025
**Статус:** M2.5 VM Migration - 80% ЗАВЕРШЕНО
**Версия:** 0.7.1 (M2.5 VM Migration SUCCESS + async/sync fixes required)

---

## 📦 ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ И ЗАВИСИМОСТИ

### **СИСТЕМНЫЕ ТРЕБОВАНИЯ:**

---

## 🖥️ АППАРАТНЫЕ ТРЕБОВАНИЯ

### **Локальная машина (минимальные):**
- **CPU:** 4+ ядер (рекомендуется 8+)
- **RAM:** 8GB (рекомендуется 16GB+)
- **Storage:** 2GB свободного места
- **OS:** Windows 10/11, macOS 10.15+, Linux Ubuntu 18.04+

### **VM Infrastructure (Jina v3):**
- **CPU:** Intel Xeon Gold 6248R (или аналогичный)
- **RAM:** 31GB (минимум 25GB для Jina v3)
- **Storage:** 100GB+ SSD
- **OS:** Ubuntu 22.04.4 LTS
- **Network:** SSH доступ + HTTP порты

---

## 🐍 PYTHON И ЗАВИСИМОСТИ

### **Python Version:**
- **Python:** 3.8+ (рекомендуется 3.10+)
- **pip:** Latest version
- **venv/virtualenv:** Для изоляции окружения

### **Core Dependencies (requirements.txt):**

#### **🤖 AI/ML:**
```txt
openai>=1.99.6                    # OpenAI API клиент
sentence-transformers>=3.0        # Jina v3 требует >=3.0 для trust_remote_code
transformers>=4.35.0              # Современная версия для Jina v3 support
torch>=2.7.0                      # CPU optimized
numpy>=1.24.0                     # Векторные операции
```

#### **🗄️ Database & Storage:**
```txt
qdrant-client[fastembed]>=1.15.1  # FastEmbed + Qdrant клиент
cachetools>=5.3.0                 # LRU/TTL кэширование
```

#### **🌐 Web & API:**
```txt
streamlit>=1.46.0                 # Web UI
fastapi>=0.115.0                  # RAG-as-a-Service API
uvicorn>=0.30.0                   # ASGI server для FastAPI
aiohttp>=3.10.0                   # HTTP клиент для VM API
```

#### **🔧 System & Utils:**
```txt
python-dotenv>=1.0.0              # Environment variables
click>=8.1.8                      # CLI framework
rich>=14.0.0                      # CLI UI library
psutil>=5.9.5                     # RAM мониторинг
paramiko>=4.0.0                   # SSH автоматизация для VM
```

#### **🔍 Search & NLP:**
```txt
rank-bm25>=0.2.2                  # BM25 алгоритм
nltk>=3.8                         # Токенизация
datasets>=2.21.0                  # Вспомогательные утилиты
```

#### **🧪 Testing:**
```txt
pytest>=8.3.4                     # Тестирование
pytest-asyncio>=1.1.0             # Асинхронные тесты
```

---

## 🔧 КОНФИГУРАЦИОННЫЕ ФАЙЛЫ

### **Основные конфигурации:**

#### **Environment Variables (.env):**
```env
# === OpenAI Configuration ===
OPENAI_API_KEY=your_openai_api_key_here

# === VM Configuration for Jina v3 Migration ===
VM_HOST=10.61.11.54
VM_USER=user
VM_PASSWORD=secure_password

# === Qdrant Configuration ===
QDRANT_HOST=localhost
QDRANT_PORT=6333

# === Embedding Configuration ===
EMBEDDING_DIMENSION=1024
EMBEDDING_MODEL=jinaai/jina-embeddings-v3
```

#### **Settings (settings.json):**
```json
{
  "rag": {
    "sparse": {
      "method": "SPLADE"
    },
    "embeddings": {
      "provider": "sentence-transformers",
      "model_name": "jinaai/jina-embeddings-v3",
      "vector_size": 1024,
      "batch_size_max": 512
    },
    "vector_store": {
      "host": "localhost",
      "port": 6333,
      "collection_name": "code_chunks",
      "quantization_type": "SQ"
    }
  }
}
```

#### **Python Path Configuration:**
- **PYTHONPATH:** Должен включать корневую директорию проекта
- **OMP_NUM_THREADS:** Оптимально 4-8 для CPU inference
- **MKL_NUM_THREADS:** Соответствует OMP_NUM_THREADS

---

## 🏗️ VM СПЕЦИФИЧНЫЕ ТРЕБОВАНИЯ

### **VM Infrastructure Setup:**

#### **Hardware Requirements:**
- **CPU:** 8+ ядер с поддержкой AVX2
- **RAM:** 31GB (25GB минимум для Jina v3)
- **Storage:** 100GB+ SSD для моделей и данных
- **Network:** Статический IP + SSH доступ

#### **Software Requirements (VM):**
```bash
# OS & Python
Ubuntu 22.04.4 LTS
Python 3.10.12

# System packages
build-essential
python3-dev
python3-venv

# Libraries for ML
libopenblas-dev
liblapack-dev
libjpeg-dev
zlib1g-dev
```

#### **VM Services:**
- **FastAPI:** 0.0.0.0:8000 (RAG endpoints)
- **Qdrant:** localhost:6333 (vector DB)
- **SSH:** port 22 (automated access)
- **Uvicorn:** ASGI server для FastAPI

---

## 📊 РЕСУРСНЫЕ ТРЕБОВАНИЯ

### **Память (RAM):**

#### **Локальная машина:**
- **Базовая система:** ~100MB
- **Анализ небольшого проекта:** 500MB-1GB
- **Анализ большого проекта:** 2GB-4GB
- **RAG операции:** +500MB для эмбеддингов

#### **VM (Jina v3):**
- **Базовая система:** ~2GB
- **Jina v3 модель:** ~25GB (570M параметров)
- **Qdrant:** ~1GB + данные
- **Batch processing:** +2-4GB для inference

### **Storage:**

#### **Локальная машина:**
- **Исходный код:** Зависит от проекта
- **Кэш файлов:** ~100MB (hash-based)
- **Логи:** ~50MB (ротация)
- **Конфигурация:** ~1MB

#### **VM:**
- **Jina v3 модель:** ~2GB на диске
- **Qdrant данные:** Зависит от индексируемых проектов
- **Логи:** ~100MB (ротация)
- **Временные файлы:** ~1GB

### **Network:**

#### **API Calls:**
- **OpenAI API:** ~100KB per request (токены)
- **VM API:** ~1MB per batch (эмбеддинги)
- **Qdrant:** ~10KB per operation

#### **Bandwidth:**
- **Разработка:** 10-50 MB/час
- **Production:** 100-500 MB/час (зависит от использования)

---

## 🔍 ВАЛИДАЦИЯ ЗАВИСИМОСТЕЙ

### **Автоматическая проверка:**

#### **System Requirements Check:**
```bash
python scripts/verify_requirements.py
```

#### **VM Environment Validation:**
```bash
python scripts/validate_vm_env.py
```

#### **Configuration Validation:**
```python
# В config.py
Config.validate()  # Проверка всех настроек
```

### **Manual Verification:**

#### **Python Environment:**
```bash
python --version  # 3.8+
pip list | grep -E "(openai|qdrant-client|sentence-transformers)"
```

#### **System Resources:**
```bash
# RAM
python -c "import psutil; print(f'Available: {psutil.virtual_memory().available/1024**3:.1f}GB')"

# CPU
python -c "import os; print(f'CPU cores: {os.cpu_count()}')"
```

#### **VM Connectivity:**
```bash
# Health check VM
curl http://10.61.11.54:8000/health

# SSH connectivity
ssh user@10.61.11.54 "echo 'VM accessible'"
```

---

## 🛠️ УСТАНОВКА И НАСТРОЙКА

### **Шаг 1: Python Environment**
```bash
# Создание виртуального окружения
python -m venv repo_sum_env
source repo_sum_env/bin/activate  # Linux/macOS
# или
repo_sum_env\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### **Шаг 2: Environment Configuration**
```bash
# Копирование шаблона
cp .env.example .env

# Редактирование .env
nano .env  # или любой редактор
```

### **Шаг 3: VM Setup (если используется)**
```bash
# Запуск VM автоматизации
python vm_start.py start

# Проверка VM состояния
python vm_start.py status
```

### **Шаг 4: Verification**
```bash
# Проверка зависимостей
python scripts/verify_requirements.py

# Проверка конфигурации
python -c "from config import Config; print('Config OK')"

# Проверка VM (если используется)
curl http://10.61.11.54:8000/health
```

---

## 🔧 ТРАБЛШУТИНГ ЗАВИСИМОСТЕЙ

### **Common Issues:**

#### **Memory Issues:**
```bash
# Проверить доступную память
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available/1024**3:.1f}GB')"

# Решение: Увеличить batch_size_min в конфигурации
```

#### **Import Errors:**
```bash
# Переустановка проблемных пакетов
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### **VM Connection Issues:**
```bash
# Проверка сети
ping 10.61.11.54

# Проверка SSH
ssh -v user@10.61.11.54

# Проверка сервисов
curl http://10.61.11.54:8000/health
```

#### **Jina v3 Issues:**
```bash
# Проверка версии sentence-transformers
python -c "import sentence_transformers; print(sentence_transformers.__version__)"

# Требуется >=3.0 для trust_remote_code
pip install --upgrade sentence-transformers
```

---

## 📈 МОНИТОРИНГ РЕСУРСОВ

### **System Monitoring:**
```python
# Встроенный мониторинг
import psutil

def get_system_info():
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_available': psutil.virtual_memory().available / 1024**3,
        'disk_usage': psutil.disk_usage('/').percent
    }
```

### **Application Metrics:**
```python
# Метрики в логах
logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024**3:.1f}GB")
logger.info(f"Batch size: {current_batch_size}")
```

### **VM Monitoring:**
```bash
# VM системные метрики
ssh user@10.61.11.54 "free -h"
ssh user@10.61.11.54 "df -h"
ssh user@10.61.11.54 "top -b -n1 | head -10"
```

---

## 🔄 ОБНОВЛЕНИЕ ЗАВИСИМОСТЕЙ

### **Safe Update Process:**

#### **Development:**
```bash
# Проверка уязвимостей
pip list --outdated
pip audit

# Тестирование обновлений
pip install --upgrade <package> --dry-run
```

#### **Production:**
```bash
# Создание backup
cp requirements.txt requirements.txt.backup

# Обновление с фиксацией версий
pip install --upgrade <package>
pip freeze > requirements.txt
```

#### **VM Dependencies:**
```bash
# Обновление VM зависимостей
ssh user@10.61.11.54 "pip install --upgrade <package>"

# Перезапуск VM сервисов
ssh user@10.61.11.54 "sudo systemctl restart vm-rag-service"
```

---

## 🎯 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ

### **Для лучшей производительности:**

#### **CPU Optimization:**
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

#### **Memory Optimization:**
```python
# В config.py
@dataclass
class EmbeddingConfig:
    batch_size_max: int = 256  # Уменьшить для слабых машин
    precision: str = "int8"    # Квантование для памяти
```

#### **Storage Optimization:**
```bash
# Очистка кэша
python clean_pycache.py

# Очистка старых коллекций
python scripts/cleanup_old_collections.py
```

---

**Дата создания:** 22 сентября 2025
**Статус:** Dependencies documented
**Следующее обновление:** При изменении требований или добавлении новых зависимостей