# 🏗️ Архитектура RAG системы (Jina v3 + Dual Task)

## 📋 Общая схема потоков

```mermaid
flowchart TD
    A[Исходный код] --> B[FileScanner]
    B --> C[CodeChunker]
    C --> D[CPUEmbedder (Jina v3 Dual Task)]
    C --> E[SparseEncoder (BM25/SPLADE)]
    D --> F[Qdrant Vector Store (1024d)]
    E --> F[Qdrant Vector Store (1024d)]
    F --> G[QueryEngine (RRF + MMR + Adaptive HNSW)]
    G --> H[SearchService]
    H --> I[OpenAIManager (RAG-enhanced prompts)]
    I --> J[Документация / Web UI / CLI]
    
    D1[Query Task: retrieval.query] --> D
    D2[Passage Task: retrieval.passage] --> D
    D --> D3[570M параметров, 1024d векторы]
```

---

## 🔑 Компоненты

### 1. FileScanner
- Рекурсивное сканирование файлов
- Фильтрация по расширениям и размеру

### 2. CodeChunker
- Логическое чанкирование (по функциям/классам)
- Поддержка AST для Python и других языков

### 3. CPUEmbedder (Jina v3 Dual Task)
- **Jina v3 модель**: jinaai/jina-embeddings-v3 (570M параметров, 1024d векторы)
- **Dual Task Architecture**: retrieval.query/passage с task-specific LoRA адаптерами
- **CPU-First 1024d**: sentence-transformers>=3.0, trust_remote_code=True
- **FastEmbed fallback**: BAAI/bge-small-en-v1.5 для совместимости

### 4. SparseEncoder
- **BM25** (baseline sparse поиск)
- **SPLADE** (Production Default)
  - HuggingFace модель: `naver/splade-cocondenser-ensembledistil`
  - PyTorch + transformers
  - Expansion токенов для улучшения поиска

### 5. Qdrant Vector Store (1024d)
- **Adaptive HNSW**: динамические параметры (m=16, ef_construct=200 для 1024d)
- Хранение dense + sparse векторов с поддержкой 1024d
- ScalarQuantization для оптимизации high-dimensional векторов
- Репликация и mmap режим для производительности

### 6. QueryEngine (Enhanced)
- **Adaptive HNSW Search**: оптимизированный поиск для 1024d векторов
- Reciprocal Rank Fusion (RRF) для объединения dense + sparse
- Maximum Marginal Relevance (MMR) для диверсификации результатов
- Task-aware поиск с учётом dual task архитектуры

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
