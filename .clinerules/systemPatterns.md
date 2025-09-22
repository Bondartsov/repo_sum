# Architectural Patterns: Repository Analyzer

**Дата:** 22 сентября 2025
**Статус:** M2.5 completed patterns documentation
**Версия:** 1.0

---

## 🎯 SOLID Principles

### Single Responsibility Principle (SRP)
**Description:** Each class has a single responsibility.
**Implementation:**
- `FileScanner`: File scanning and filtering.
- `CodeChunker`: Code chunking logic.
- `CPUEmbedder`: Embedding generation.
- `OpenAIManager`: OpenAI API integration.

### Open-Closed Principle (OCP)
**Description:** Open for extension, closed for modification.
**Implementation:**
- New parsers extend `BaseParser` without modifying registry.
- `ChunkingStrategy` interface for new strategies (logical/size/lines).

### Liskov Substitution Principle (LSP)
**Description:** Subtypes interchangeable with base.
**Implementation:**
- Any `BaseParser` substitutable (PythonParser for .py).
- `FastEmbedProvider`/`SentenceTransformersProvider` interchangeable in embedder.

### Interface Segregation Principle (ISP)
**Description:** Clients not forced to depend on unneeded interfaces.
**Implementation:**
- Separate interfaces for reading/writing/embedding.
- Minimal methods in `BaseParser` (parse only).

### Dependency Inversion Principle (DIP)
**Description:** Depend on abstractions, not concretions.
**Implementation:**
- `QueryEngine` depends on `BaseVectorStore`, not Qdrant.
- `RepositoryAnalyzer` uses `BaseParser` interface.

## 🔧 Design Patterns

### Plugin Architecture Pattern
**Description:** Extensible system for parsers.
**Implementation:**
- `BaseParser` ABC with `parse` abstract method.
- `ParserRegistry.get_parser` loads by extension (e.g., PythonParser for .py).
- Add new languages by subclassing, no core changes.

### Strategy Pattern
**Description:** Define algorithms, encapsulate, interchangeable.
**Implementation:**
- Chunking strategies: logical (AST functions/classes), size (tokens), lines.
- Selected via config `chunk_strategy` in `AnalysisConfig`.

### Factory Pattern
**Description:** Create objects without specifying class.
**Implementation:**
- `ParserRegistry`: Maps extensions to parsers (e.g., .py → PythonParser).
- Fallback to default if unknown.

### Pipeline Pattern
**Description:** Sequential data processing stages.
**Implementation:**
- Scan → Filter → Chunk → Embed → Analyze → Generate docs.
- Each stage testable, cacheable (hash-based file cache).

### Batch Processing Pattern
**Description:** Process data in batches for efficiency.
**Implementation:**
- Adaptive batch size (min 8, max 512) based on RAM (`psutil`).
- Parallel via `asyncio.gather` in `RepositoryAnalyzer._analyze_files_batch`.

### Multi-Level Caching Pattern
**Description:** Cache at multiple levels for performance.
**Implementation:**
1. File-level: Hash cache for analysis results (TTL via index.json).
2. RAG search: LRU/TTL (300s, 1000 entries) with RLock thread-safety.
3. Embedding: Cache vectors to avoid recompute.
4. API response: Cache OpenAI calls.

### Reciprocal Rank Fusion (RRF) Pattern
**Description:** Fuse rankings from multiple sources.
**Implementation:**
- Hybrid search: Dense (Jina v3) + Sparse (SPLADE/BM25).
- Formula: 1 / (k + rank) summed, k=60.

### Maximum Marginal Relevance (MMR) Pattern
**Description:** Balance relevance and diversity.
**Implementation:**
- Post-RRF rerank: MMR = λ * relevance - (1-λ) * max_similarity (λ=0.7).
- Similarity: Set-based text overlap.

## 📊 Data Processing Patterns

### Hybrid Search Pattern
**Description:** Combine dense + sparse for better recall/precision.
**Implementation:**
- Query → Dense embed (Jina task=query) + Sparse encode (SPLADE).
- Fuse with RRF, rerank MMR if enabled.
- Config: `use_hybrid=true`, `sparse.method=SPLADE`.

### Lazy Loading Pattern
**Description:** Load components on demand.
**Implementation:**
- Parsers loaded only if file extension matches.
- Embedder models initialized on first use (`SentenceTransformer`).

### Resource Pooling Pattern
**Description:** Reuse expensive resources.
**Implementation:**
- OpenAI client singleton.
- Qdrant connection pooling.
- Thread pools for parallel processing.

### Memory-Aware Processing Pattern
**Description:** Adapt to available memory.
**Implementation:**
- Monitor `psutil.virtual_memory().available`.
- Adjust batch_size if low RAM (fallback smaller chunks).

## 🔒 Security Patterns

### Environment-Based Configuration Pattern
**Description:** Secrets via env vars, no hardcodes.
**Implementation:**
- API keys from `os.getenv` (OPENAI_API_KEY, VM_PASSWORD).
- Validate presence in `Config.validate()`.

### Input Validation Pattern
**Description:** Validate external inputs.
**Implementation:**
- File size <10MB (`max_file_size`).
- Path traversal protection in scanner.
- Config dims/host/port checks in `validate()`.

### Fail-Safe Pattern
**Description:** Graceful degradation on errors.
**Implementation:**
- Fallback: BGE local if VM down (np.zeros offline).
- Retry: Exponential backoff for API calls.
- Partial results on partial failures.

## ⚡ Performance Patterns

### Adaptive Threading Pattern
**Description:** Optimal CPU usage.
**Implementation:**
- Set `torch.set_num_threads`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` via `ParallelismConfig`.

### Lazy Initialization Pattern
**Description:** Delay heavy initialization.
**Implementation:**
- Embedder warmup optional.
- VectorStore connects on first operation.

## 📈 Monitoring Patterns

### Health Check Pattern
**Description:** Component status verification.
**Implementation:**
- Sync: `Config.validate()` for dims/env.
- Async runtime: `QdrantVectorStore.health_check` (get_collections for connection, /health for remote).

---

**References:** SOLID from techContext.md, patterns from code (config.py, rag/*, main.py). All aligned with M2.5 architecture.
