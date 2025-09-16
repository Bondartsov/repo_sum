# Scenario Validation Notes

- [ ] `python main.py rag status --detailed` (с подключением к VM) — проверить health-check, убедиться, что embedder и vector store в состоянии `healthy`.
- [ ] `python main.py rag index <repo_path> --batch-size 128 --no-progress` — протестировать индексацию на небольшом репозитории; зафиксировать время и количество проиндексированных чанков.
- [ ] `python main.py rag search "authentication middleware" --top-k 5 --no-content` — проверить выдачу, убедиться, что результаты ранжируются корректно.
- [ ] Web UI: запустить `python run_web.py`, выполнить поиск из вкладки "RAG: поиск по проекту" (проверить отображение контента и время ответа).
- [ ] REST API: `curl http://10.61.11.54:8000/health`, `curl -X POST http://10.61.11.54:8000/embeddings`, `curl -X POST http://10.61.11.54:8000/search` — убедиться, что сервис отвечает с кодом 200 и корректным payload.
- [ ] Jina v3 vs BGE: прогнать бенчмарк (например, `tests/test_remote_clients.py` + замер latency ответов VM) и задокументировать показатели.

> ⚠️ На текущем стенде удалённый сервис недоступен, поэтому проверки необходимо выполнить в рабочем окружении, где VM присоединена. Результаты фиксировать в `SCENARIO_VALIDATION_NOTES.md`.
