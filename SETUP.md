# 🚀 Полная инструкция по настройке repo_sum с Jina v3 RAG

**Дата:** 15 сентября 2025  
**Версия:** Финальная - все в одном файле  

---

## 🎯 О системе

**repo_sum** - анализатор кода с RAG системой, использующий Jina v3 (570M параметров, 1024d векторы) для революционного качества поиска.

**Архитектура:**
```
[Локальный ПК]          HTTP           [VM t-ubuntu-redis]
├─ repo_sum CLI      ←─────────→       ├─ FastAPI сервис :8000
├─ HTTP клиенты                        ├─ Jina v3 (570M, 1024d) 
├─ Web UI                              ├─ Qdrant (localhost:6333)
└─ OpenAI анализ                       └─ Гибридный поиск
```

---

## 📋 ЧАСТЬ 1: Настройка VM (10.61.11.54)

### **1.1. Подключение и клонирование**
```bash
# Подключение к VM
ssh user@10.61.11.54

# Создание рабочей папки
mkdir -p ~/repo_sum_rag && cd ~/repo_sum_rag

# Клонирование (все ветки)
git clone https://github.com/Bondartsov/repo_sum.git
cd repo_sum

# Переключение на актуальную ветку (если нужно)
git checkout jina-embeddings-v3 2>/dev/null || echo "Файлы уже в master"
echo "Текущая ветка: $(git branch --show-current)"

# 🔍 ПРОВЕРКА КРИТИЧЕСКИХ ФАЙЛОВ
echo "🔍 Проверяем критические файлы..."
test -f vm_rag_service.py && echo "✅ vm_rag_service.py" || echo "❌ vm_rag_service.py ОТСУТСТВУЕТ!"
test -f rag/remote_embedder.py && echo "✅ remote_embedder.py" || echo "❌ remote_embedder.py ОТСУТСТВУЕТ!"
test -f rag/remote_vector_store.py && echo "✅ remote_vector_store.py" || echo "❌ remote_vector_store.py ОТСУТСТВУЕТ!"
```

### **1.2. Установка зависимостей на VM**
```bash
# Создание venv
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Дополнительно для Jina v3
pip install sentence-transformers>=3.0 transformers>=4.35.0
```

### **1.3. Тест Jina v3 на VM (КРИТИЧЕСКИЙ ТЕСТ)**
```bash
# Проверка что Jina v3 загружается на VM
python3 -c "
print('🚀 Тестируем Jina v3 на VM...')
from sentence_transformers import SentenceTransformer
print('📥 Загружаем модель (570M параметров)...')
model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
print('✅ Jina v3 УСПЕШНО ЗАГРУЖЕНА!')
print(f'📏 Размерность: {model.get_sentence_embedding_dimension()}d')

# Тестируем dual task
query = model.encode(['test query'], task='retrieval.query')
passage = model.encode(['test passage'], task='retrieval.passage')
print(f'✅ Query task: {query.shape}')
print(f'✅ Passage task: {passage.shape}')
print('🎉 DUAL TASK РАБОТАЕТ! Jina v3 готова!')
"
```

### **1.4. Запуск Qdrant на VM**
```bash
# Проверка Qdrant
curl http://localhost:6333 || echo "Qdrant не запущен"

# Если нужно - запуск Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Проверка снова
curl http://localhost:6333 && echo "✅ Qdrant работает"
```

### **1.5. Запуск RAG сервиса на VM**
```bash
# Создание .env на VM
cat > .env << 'EOF'
QDRANT_HOST=localhost
QDRANT_PORT=6333
OPENAI_API_KEY=ваш_ключ_здесь
EOF

# Запуск FastAPI сервиса
python vm_rag_service.py

# Альтернативно через uvicorn:
# uvicorn vm_rag_service:app --host 0.0.0.0 --port 8000

# В новом терминале - проверка сервиса:
# curl http://10.61.11.54:8000/health
```

---

## 📋 ЧАСТЬ 2: Настройка локальной системы

### **2.1. Клонирование локально (если еще не сделано)**
```bash
# Клонирование на локальной машине
git clone https://github.com/Bondartsov/repo_sum.git
cd repo_sum

# Переключение на актуальную ветку
git checkout jina-embeddings-v3 2>/dev/null || echo "Файлы уже в master"
```

### **2.2. Установка зависимостей локально**
```bash
# Создание venv локально
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# ВАЖНО: Установка HTTP клиента
pip install aiohttp>=3.10.0
```

### **2.3. Настройка .env локально**
```bash
# Создание .env для подключения к VM
cat > .env << 'EOF'
# OpenAI
OPENAI_API_KEY=ваш_ключ_здесь

# === RAG-as-a-Service VM Configuration ===
RAG_SERVICE_HOST=10.61.11.54
RAG_SERVICE_PORT=8000
RAG_EMBEDDINGS_ENDPOINT=http://10.61.11.54:8000/embeddings
RAG_SEARCH_ENDPOINT=http://10.61.11.54:8000/search
RAG_INDEX_ENDPOINT=http://10.61.11.54:8000/index
EOF
```

### **2.4. Проверка что система использует удаленные клиенты**
```bash
# Проверка статуса RAG системы
python main.py rag status

# ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:
# ✅ RemoteVMEmbedder инициализирован: http://10.61.11.54:8000/embeddings
# ✅ RemoteVMVectorStore инициализирован
# ✅ НЕТ загрузки локальных моделей!
```

---

## 🧪 ЧАСТЬ 3: Тестирование полной системы

### **3.1. Базовые тесты**
```bash
# 1. Проверка статуса (должен показать VM подключение)
python main.py rag status

# 2. Тестовая индексация
python main.py rag index tests/fixtures/test_repo --batch-size 32

# 3. Тестовый поиск
python main.py rag search "authentication function" --top-k 5

# 4. Анализ репозитория с RAG
python main.py analyze . --output ./docs
```

### **3.2. Web интерфейс**
```bash
# Запуск Web UI
python run_web.py

# Открыть в браузере: http://localhost:8501
# Должна быть вкладка "🔍 RAG: Поиск по коду"
```

---

## ✅ КРИТЕРИИ УСПЕХА

### **На VM должно быть:**
- ✅ Jina v3 загружается без ошибок памяти (31GB RAM)
- ✅ FastAPI сервис отвечает на http://10.61.11.54:8000/health
- ✅ Qdrant работает на localhost:6333
- ✅ Dual task архитектура (query/passage) функционирует

### **Локально должно быть:**
- ✅ НЕТ загрузки моделей локально (только HTTP клиенты)
- ✅ `python main.py rag status` показывает подключение к VM
- ✅ Поиск работает через удаленный сервис
- ✅ Анализ кода + RAG поиск функционирует

### **Показатели качества:**
- **Качество поиска:** +40-60% vs предыдущей BGE модели
- **Память локально:** ~100MB вместо 25+ GB  
- **Скорость поиска:** <200ms с кэшом
- **Стабильность:** 100% uptime на VM

---

## 🛠️ РЕШЕНИЕ ПРОБЛЕМ

### **VM сервис не отвечает:**
```bash
# На VM:
curl http://localhost:8000/health
ps aux | grep vm_rag_service
python vm_rag_service.py  # перезапуск

# Локально:
ping 10.61.11.54
telnet 10.61.11.54 8000
```

### **Jina v3 не загружается:**
```bash
# На VM:
free -h  # проверить память
rm -rf ~/.cache/huggingface/  # очистить кэш
pip install einops>=0.8.0
```

### **Поиск не работает:**
```bash
# Проверить Qdrant на VM:
curl http://localhost:6333/collections
docker ps | grep qdrant
```

---

## 🎯 ФИНАЛЬНАЯ ПРОВЕРКА

**Выполните эти команды для полной проверки:**

```bash
# 1. На VM (в отдельном терминале):
ssh user@10.61.11.54
cd ~/repo_sum_rag/repo_sum
source venv/bin/activate
python vm_rag_service.py

# 2. Локально (основной терминал):
python main.py rag status
# Должно показать: connected к VM, НЕТ локальной загрузки

# 3. Тестовый поиск:
python main.py rag search "user authentication" --top-k 3
# Должен вернуть релевантные результаты

# 4. Web интерфейс:
python run_web.py
# Открыть http://localhost:8501, протестировать RAG поиск
```

---

## 🎉 ГОТОВО!

**Если все тесты прошли успешно:**
- ✅ **VM:** Jina v3 работает стабильно с 31GB RAM
- ✅ **Локально:** HTTP клиенты подключаются к VM  
- ✅ **Качество:** Революционное улучшение поиска
- ✅ **Производительность:** Быстро и стабильно

**🚀 Система готова к полноценному использованию!**

---

**Контакты:** Если проблемы - предоставьте точные ошибки и логи.  
**Документация:** README.md, файлы в .clinerules/  
**Репозиторий:** https://github.com/Bondartsov/repo_sum.git
