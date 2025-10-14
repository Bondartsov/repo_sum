# Project Documentation Rules (Non-Obvious Only)

- Писать по‑русски; команды — PowerShell (Windows). См. [.claude/CLAUDE.md](.claude/CLAUDE.md:162-169)
- Не создавать новые файлы в [rules/](rules/AGENTS.md:42); обновлять существующие. Подтверждение: [.claude/CLAUDE.md](.claude/CLAUDE.md:185-190)
- Единые источники правил: [rules/AGENTS.md](rules/AGENTS.md) и [.claude/CLAUDE.md](.claude/CLAUDE.md); при расхождениях синхронизировать обе версии в одном PR
- Включать только не‑очевидное, подтверждённое чтением кода; удалять дефолт/догадки; все упоминания файлов делать кликабельными формата [name](path:line)
- Для RAG‑сценариев опираться на фактические контракты API и ограничения: search_v2 и index. См. [vm_rag_service.py](vm_rag_service.py:831-846), [vm_rag_service.py](vm_rag_service.py:917-1004)
- Не противоречить архитектурным инвариантам в тексте (Factory Context на VM, lazy‑импорты в CLI) — давайте ссылки на исходники: [vm_rag_service.py](vm_rag_service.py:98), [main.py](main.py:32-36)