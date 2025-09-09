# Project Debug Rules (Non-Obvious Only)

- RAG-тесты запускать через [`tests/rag/run_rag_tests.py`](../../tests/rag/run_rag_tests.py).
- Ошибки обрабатываются централизованно в [`rag/exceptions.py`](../../rag/exceptions.py).
- Fail-fast: ошибки должны выявляться на старте.