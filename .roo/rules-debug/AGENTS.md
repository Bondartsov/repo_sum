# Project Debug Rules (Non-Obvious Only)

- CLI падает без OPENAI_API_KEY; для оффлайна — флаги offline (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/USE_MOCK_EMBEDDER). См. [main.py](main.py:365-387)
- Windows‑кодировки закрыты: форс UTF‑8 stdout/stderr и отключены emoji — не убирать обёртку. См. [main.py](main.py:37-50)
- Веб‑UI сам ставит зависимости и проверяет порт; при занятом порте — жёсткий отказ. См. [run_web.py](run_web.py:119-154,163-183)
- Для диагностики VM используйте статус сервиса и метрики Prometheus. См. [vm_rag_service.py](vm_rag_service.py:351-395,1197)
- Код 507 означает триггер защитного middleware по памяти (>~92% RAM). См. [vm_rag_service.py](vm_rag_service.py:239-261)
- Для отображения удалённого статуса задайте RAG_HEALTH_ENDPOINT (печать статуса в UI). См. [run_web.py](run_web.py:39-89)
- Unified launcher уже фильтрует шум логов; отлаживайте по фазам setup→start→monitor. См. [unified_launcher.py](unified_launcher.py:70,174,285)