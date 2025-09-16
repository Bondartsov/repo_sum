# Fix TODO Codex Checklist

- [x] Привести `RemoteVMEmbedder` к целевой схеме: вынести HTTP-запрос в `_async_embed_texts`, реализовать синхронный wrapper с корректными тайм-аутами/ретраями, поддержать Matryoshka-транкцию и обновление статистики.
- [x] Актуализировать `RemoteVMVectorStore`: добавить синхронные обёртки для `search`/`index`/`health`, передавать реальные dense/sparse-вектора, обеспечить гибридный поиск и обработку ошибок/ретраев.
- [x] Согласовать `SearchService` и `IndexerService` с удалёнными классами: работать через синхронные обёртки, гарантировать dual-task, гибридную логику и корректную обработку исключений.
- [x] Расширить загрузку конфигурации: учесть секцию `remote_service`, унифицировать размерности (1024d ↔ 384d), проверить `.env` и `vm_start.py` на актуальность параметров VM.
- [x] Усилить логирование и устойчивость: добавить ASCII-only fallback для Windows, обеспечить информативный вывод ошибок и graceful degradation при сбоях VM.
- [x] Обновить и дополнить тесты: покрыть RemoteVMEmbedder/VectorStore, SearchService/IndexerService, проверить offline/mock режимы и отсутствие coroutine warning’ов.
- [x] Провести сценарные проверки: CLI (`rag index/search/status`), Web UI, ручные вызовы FastAPI (`/health`, `/embeddings`, `/search`, `/index`), замерить показатели Jina v3 vs BGE. (см. `SCENARIO_VALIDATION_NOTES.md`)
- [x] Финализировать документацию и Memory Bank по результатам: обновить README/SETUP, `.clinerules/*`, ASYNC_SYNC_FIX_SPEC, зафиксировать статус M2.5.


