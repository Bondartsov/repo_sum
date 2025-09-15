# 🚀 Команды для исправления Jina v3 на VM

**Выполните эти команды на VM по порядку:**

## Команда 1: Очистка кэша HuggingFace
```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/jinaai
```

## Команда 2: Полная очистка кэша (если нужно)
```bash
rm -rf ~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3
```

## Команда 3: Загрузка Jina v3 (после очистки кэша)
```bash
python3 -c 'from sentence_transformers import SentenceTransformer; print("Загружаем Jina v3..."); model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True); print("Jina v3 успешно загружена"); print(f"Размерность: {model.get_sentence_embedding_dimension()}d")'
```

## Команда 4: Тест dual task архитектуры
```bash
python3 -c 'from sentence_transformers import SentenceTransformer; model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True); query = model.encode(["test query"], task="retrieval.query"); passage = model.encode(["test passage"], task="retrieval.passage"); print("DUAL TASK РАБОТАЕТ"); print(f"Query: {query.shape}"); print(f"Passage: {passage.shape}")'
```

## Команда 5: Проверка памяти (опционально)
```bash
free -h
```

## Команда 6: Если все работает - создание .env
```bash
cat > .env << 'EOF'
QDRANT_HOST=localhost
QDRANT_PORT=6333
OPENAI_API_KEY=sk-proj-ваш_ключ_здесь
EOF
```

## Команда 7: Тест полной RAG системы
```bash
python3 main.py rag status
```

---

**После выполнения всех команд сообщите результат!**
