# 🏗️ Архитектура RAG системы (с SPLADE)

## 📋 Общая схема потоков

```mermaid
flowchart TD
    A[Исходный код] --> B[FileScanner]
    B --> C[CodeChunker]
    C --> D[CPUEmbedder (FastEmbed)]
    C --> E[SparseEncoder (BM25/SPLADE)]
    D --> F[Qdrant Vector Store]
    E --> F[Qdrant Vector Store]
    F --> G[QueryEngine (RRF + MMR)]
    G --> H[SearchService]
    H --> I[OpenAIManager (RAG-enhanced prompts)]
    I --> J[Документация / Web UI / CLI]
```

---

## 🔑 Компоненты

### 1. FileScanner
- Рекурсивное сканирование файлов
- Фильтрация по расширениям и размеру

### 2. CodeChunker
- Логическое чанкирование (по функциям/классам)
- Поддержка AST для Python и других языков

### 3. CPUEmbedder (FastEmbed)
- Dense векторы (BAAI/bge-small-en-v1.5)
- CPU-first оптимизация (int8, normalize_embeddings)

### 4. SparseEncoder
- **BM25** (baseline sparse поиск)
- **SPLADE** (Production Default)
  - HuggingFace модель: `naver/splade-cocondenser-ensembledistil`
  - PyTorch + transformers
  - Expansion токенов для улучшения поиска

### 5. Qdrant Vector Store
- Хранение dense + sparse векторов
- ScalarQuantization для оптимизации
- Репликация и mmap режим

### 6. QueryEngine
- Reciprocal Rank Fusion (RRF) для объединения dense + sparse
- Maximum Marginal Relevance (MMR) для диверсификации

### 7. SearchService
- Высокоуровневый API для поиска
- Production Defaults: всегда SPLADE
- Фильтрация по языкам и типам кода

### 8. OpenAIManager
- Интеграция RAG контекста в промпты
- Smart chunking (~8-12k токенов)
- Контекстуальный анализ связанных компонентов

### 9. Интерфейсы
- CLI (`main.py`)
- Web UI (Streamlit)
- REST API (готовность к интеграции)

---

## ⚙️ Production Defaults

```json
"sparse": {
  "method": "SPLADE"
}
```

- Пользователи всегда работают с SPLADE
- Переключение доступно только разработчикам через конфигурацию
- BM25 остаётся fallback для тестов и отладки

---

## ✅ Критерии успеха интеграции SPLADE
- Улучшение Precision@10 и Recall@100 на 20-30%
- Латентность поиска <300ms p95
- Полная совместимость с существующим пайплайном
- Изолированные тесты для SPLADE
- Обновлённый Memory Bank (.clinerules)
