# AGENTS.md

Этот файл фиксирует только проектно-специфичные и неочевидные правила для режима `architect`.

- Архитектура должна быть CPU-first.  
- Использовать гибридный поиск (dense + sparse).  
- Жёсткая связка локального и удалённого RAG обязательна.  
- Интеграция с [`rules/Technical Debt.md`](rules/Technical%20Debt.md) обязательна.  
- Все сценарии должны запускаться через [`unified_launcher.py`](unified_launcher.py).  