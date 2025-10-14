# AGENTS.md
This file provides guidance to agents when working with code in this repository.

- CLI требует OPENAI_API_KEY (префикс sk-); offline включает HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/USE_MOCK_EMBEDDER; есть fail‑fast проверка ключа — см. [cli()](main.py:340), [offline flags](main.py:365), [fail-fast key](main.py:375)
- Веб‑UI: авто‑установка deps; принудительный UTF‑8 и headless Streamlit; жёсткий отказ при занятом порте; статус удалённых сервисов через RAG_HEALTH_ENDPOINT — см. [install_requirements()](run_web.py:119), [port check](run_web.py:150), [env](run_web.py:163), [show_remote_status()](run_web.py:39)
- Unified launcher: единый «setup→start→monitor», фильтрация шума логов — см. [setup_vm()](unified_launcher.py:70), [start_web_app()](unified_launcher.py:174), [run_full_workflow()](unified_launcher.py:285)
- Автоматизация VM: выбор ветки зависит от GitHub API; .env на VM переопределяет хосты; health‑check с ретраями; фоновый сервис с PID и быстрой диагностикой — см. [branches](vm_start.py:219), [env overrides](vm_start.py:675), [health verify](vm_start.py:895), [start service](vm_start.py:959)
- Сервис на VM: CLI start/status/stop; status делает HTTP health‑check — см. [check_service_status()](vm_rag_service.py:1197), [start_service()](vm_rag_service.py:1295)
- Обязателен VM Factory Context до создания FastAPI app — [RAGFactory.set_context(ExecutionContext.VM)](vm_rag_service.py:98)
- В CLI импорты RAG — ленивые внутри команд (не поднимать наверх) — см. [lazy import note](main.py:32)
- Наблюдаемость: JSON‑логер с безопасными полями; Prometheus (гистограммы, gauge памяти); нормализация путей /v1/* — см. [logger](vm_rag_service.py:328), [metrics](vm_rag_service.py:351)
- Память: при >~92% RAM — gc + HTTP 507 — см. [check_memory_usage()](vm_rag_service.py:179), [middleware 507](vm_rag_service.py:239)
- Контракт /v1/index: X‑API‑Contract v1.0.0 обязателен; batch 1..128; строгая sha256 опционально; dropped_documents_total — см. [index](vm_rag_service.py:917)
- Поиск /v1/search_v2: обязателен протокол (dense/sparse), проверка размерности (fallback=1024), запрет NaN/Inf — см. [search_v2](vm_rag_service.py:831)
- Windows консоль: форс UTF‑8; отключены emoji — см. [stdout/stderr UTF-8](main.py:37), [emoji=False](main.py:50)
- Стиль: Ruff только lint; «ruff format» не используется; глобальные игноры E402/E722/E731/F401/F841; исключены legacy/** и tests/bench/**; per‑file послабления для tests/mocks/** и имён с "_" — см. [.ruff.toml](.ruff.toml)
- requirements.txt: конфликтующие rich (>=14 и >=13) — возможны проблемы резолвера — см. [requirements.txt](requirements.txt)
- Тесты: маркеры и asyncio_mode=strict настроены; известное предупреждение подавлено; юнит‑тесты оффлайн; нестандартного способа запуска одного теста не обнаружено — см. [pytest.ini](pytest.ini), [.claude/CLAUDE.md](.claude/CLAUDE.md:193)
- Правила ассистентов: писать по‑русски; команды — PowerShell (Windows); не создавать новые файлы в [rules/](rules/AGENTS.md:42); RAG‑раннеры и VM workflow каноничны — запуск через [vm_start.py](vm_start.py) — см. [.claude/CLAUDE.md](.claude/CLAUDE.md:162), [rules/AGENTS.md](rules/AGENTS.md:245)