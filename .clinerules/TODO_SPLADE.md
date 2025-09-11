# TODO: Интеграция SPLADE (сентябрь 2025)

## Документация
- [x] Перенести полезные куски из `QUICK_START_RAG.md` в `.clinerules/`
- [x] Создать `.clinerules/RAG_architecture.md` со схемой потоков
- [x] Удалить устаревший `QUICK_START_RAG.md`

## Зависимости
- [x] Обновить `requirements.txt` (transformers, datasets, torch)

## Конфигурация
- [x] Добавить `SparseConfig` в `config.py`
- [x] Добавить `"sparse": { "method": "SPLADE" }` в `settings.json` (Production Defaults)

## Реализация SPLADE
- [x] Реализовать `SpladeModelWrapper` и `encode_splade` в `rag/sparse_encoder.py`
- [x] Интегрировать SPLADE в `rag/search_service.py`

## Тестирование
- [x] Создать `tests/rag/test_splade_encoder.py` (изолированные тесты)
- [x] Обновить `tests/rag/test_search_service_min_score_zero.py`
- [x] Прогнать все тесты через `tests/rag/run_rag_tests.py` (см. также целевые unit-тесты)

## Memory Bank
- [x] Обновить `.clinerules/techContext.md`
- [x] Обновить `.clinerules/progress.md`
- [x] Обновить `.clinerules/projectContext.md`
- [x] Зафиксировать Production Defaults и SPLADE в `.clinerules/RAG_architecture.md`
