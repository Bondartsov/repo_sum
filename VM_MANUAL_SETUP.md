# 🚀 Ручная настройка VM для Jina v3 (финальные шаги)

**✅ Статус:** SSH работает, 31GB RAM готовы, Python 3.10.12 установлен

## 📋 Выполните на VM (копируйте по блокам):

### 1. Создание рабочей папки и venv (обход ensurepip проблемы)
```bash
# Создаем папку проекта
mkdir -p ~/repo_sum_rag && cd ~/repo_sum_rag

# Альтернативный способ создания venv (обход ensurepip)
sudo apt install -y python3.10-venv python3.10-distutils python3-pip git
python3 -m venv venv --without-pip
source venv/bin/activate

# Устанавливаем pip в venv напрямую
curl https://bootstrap.pypa.io/get-pip.py | python
pip install --upgrade pip setuptools wheel
```

### 2. Клонирование репозитория
```bash
# Клонируем repo_sum
git clone https://github.com/Bondartsov/repo_sum.git src
cd src

# Устанавливаем зависимости
pip install -r requirements.txt

# Дополнительно для Jina v3
pip install sentence-transformers>=3.0
pip install transformers>=4.35.0
```

### 3. 🎯 КРИТИЧЕСКИЙ ТЕСТ: Загрузка Jina v3 (570M параметров)
```bash
# Тестируем загрузку Jina v3 с trust_remote_code
python3 -c "
print('🚀 Тестируем загрузку jinaai/jina-embeddings-v3...')
from sentence_transformers import SentenceTransformer
import torch
print(f'💾 Доступно RAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else '💾 CPU режим')
print('📥 Загружаем модель (570M параметров)...')
model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
print('✅ Jina v3 УСПЕШНО ЗАГРУЖЕНА!')
print(f'📏 Размерность: {model.get_sentence_embedding_dimension()}d')
print('🧪 Тестируем dual task...')
# Тестируем retrieval.query
test_query = model.encode(['test query'], task='retrieval.query')
print(f'✅ Query task: {test_query.shape}')
# Тестируем retrieval.passage  
test_passage = model.encode(['test passage'], task='retrieval.passage')
print(f'✅ Passage task: {test_passage.shape}')
print('🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Jina v3 готова к работе!')
"
```

### 4. Если тест прошел успешно 🎉
```bash
# Создаем .env файл на VM
cat > .env << 'EOF'
QDRANT_HOST=localhost
QDRANT_PORT=6333
OPENAI_API_KEY=sk-proj-ваш_ключ_здесь
EOF

# Тестируем полную RAG систему
python3 main.py rag status
```

---

## 🎯 Ожидаемые результаты:

**✅ При успехе:**
- Jina v3 загружается без ошибок памяти
- Dual task (query/passage) работает корректно
- 1024d векторы генерируются стабильно
- Качество поиска +40-60% vs BGE

**❌ При проблемах:**
- Недостаток памяти (но маловероятно с 31GB)
- Проблемы с trust_remote_code
- Сетевые ошибки при загрузке модели

---

## 📞 Поддержка:
- Если все работает: переходим к созданию RAG-as-a-Service API
- Если проблемы: отчитайтесь с точными ошибками
- Следующий этап: FastAPI сервис для удаленного доступа

**🚀 Главная цель:** Убедиться что Jina v3 загружается на VM с 31GB RAM!
