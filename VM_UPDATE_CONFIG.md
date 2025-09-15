# 🔄 Обновление конфигурации для Jina v3 на VM

**Jina v3 успешно работает! Теперь обновим конфигурацию repo_sum:**

## Команда 1: Создание правильного settings.json
```bash
cat > settings.json << 'EOF'
{
  "analysis": {
    "chunk_strategy": "logical",
    "min_chunk_size": 100,
    "enable_fallback": true,
    "languages_priority": ["python", "javascript", "java"],
    "sanitize_enabled": true
  },
  "file_scanner": {
    "max_file_size": 10485760,
    "excluded_directories": [
      ".git", ".svn", ".hg",
      "node_modules", "venv", ".venv",
      "__pycache__", ".pytest_cache",
      "target", "build", "dist",
      ".idea", ".vscode",
      "logs", "tmp", "temp"
    ],
    "supported_extensions": {
      ".py": "python",
      ".js": "javascript",
      ".ts": "typescript",
      ".jsx": "javascript",
      ".tsx": "typescript",
      ".java": "java",
      ".cpp": "cpp",
      ".cc": "cpp",
      ".cxx": "cpp",
      ".h": "cpp",
      ".hpp": "cpp",
      ".cs": "csharp",
      ".go": "go",
      ".rs": "rust",
      ".php": "php",
      ".rb": "ruby"
    }
  },
  "output": {
    "default_output_dir": "./docs",
    "file_template": "minimal_file.md",
    "index_template": "index_template.md"
  },
  "prompts": {
    "code_analysis_prompt_file": "prompts/code_analysis_prompt.md"
  },
  "rag": {
    "sparse": {
      "method": "SPLADE"
    },
    "embeddings": {
      "model_name": "jinaai/jina-embeddings-v3",
      "provider": "sentence-transformers",
      "normalize_embeddings": true,
      "warmup_enabled": true,
      "batch_size_min": 8,
      "batch_size_max": 64
    },
    "vector_store": {
      "collection_name": "repo_sum_jina_v3_vm",
      "vector_size": 1024,
      "distance": "cosine",
      "hnsw_m": 16,
      "hnsw_ef_construct": 200,
      "search_hnsw_ef": 128,
      "quantization_type": "SQ",
      "enable_quantization": true,
      "mmap": true
    },
    "query_engine": {
      "rrf_enabled": true,
      "use_hybrid": true,
      "mmr_enabled": true,
      "mmr_lambda": 0.7,
      "concurrent_users_target": 50,
      "search_workers": 8,
      "embed_workers": 8,
      "cache_ttl_seconds": 300,
      "cache_max_entries": 1000,
      "max_results": 10,
      "score_threshold": 0.0
    },
    "parallelism": {
      "torch_num_threads": 8,
      "omp_num_threads": 8,
      "mkl_num_threads": 8
    }
  }
}
EOF
```

## Команда 2: Тест обновленной конфигурации
```bash
python3 main.py rag status
```

## Команда 3: Если все правильно - тест индексации
```bash
python3 main.py rag index ../test_files --batch-size 16
```

---

**Ожидаемый результат:**
- Система покажет: "jinaai/jina-embeddings-v3", "1024d", "sentence-transformers"
- ✅ Полные 1024d векторы вместо сжатых 384d!
