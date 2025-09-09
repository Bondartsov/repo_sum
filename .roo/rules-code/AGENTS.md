# Project Code Rules (Non-Obvious Only)

- Использовать [`scripts/verify_requirements.py`](../../scripts/verify_requirements.py) вместо стандартного `pip check`.
- Для эмбеддингов использовать [`rag/embedder.py`](../../rag/embedder.py), а не прямые вызовы OpenAI.
- Для sparse-поиска использовать [`rag/sparse_encoder.py`](../../rag/sparse_encoder.py), а не внешние движки.
- Все изменения фиксировать в `.clinerules/`.
- Все тесты должны поддерживать **offline/mock режим**:
  - Сетевые вызовы замоканы.
  - В CI запрещены реальные сетевые запросы.
  - Эталон: [`tests/test_offline_no_network.py`](../../tests/test_offline_no_network.py).