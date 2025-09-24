# AGENTS.md

Этот файл фиксирует только проектно-специфичные и неочевидные правила для режима `code`.

- Все переменные окружения должны считываться только через функции `safe_int`, `safe_float`, `safe_bool` из [`config.py`](config.py).
  Это гарантирует валидацию значений, работу с fallback и fail-fast при ошибках.
- Строгие диапазоны параметров (`truncate_dim`, `mmr_lambda`) должны соблюдаться.
- Конфигурация должна быть fail-fast: при ошибке немедленно завершать выполнение.
- Поддерживать dual-task embeddings (раздельно для query и passage).
- Обеспечивать жёсткую связку локального и удалённого RAG.
- Все изменения в коде должны сопровождаться обновлением документации в папке [`rules/`](rules/).
- Перед началом работы агент обязан изучить [`Development Roadmap.md`](rules/Development%20Roadmap.md), [`Technical Architecture.md`](rules/Technical%20Architecture.md) и актуальные задачи в [`Technical Debt.md`](rules/Technical%20Debt.md).
- Все новые задачи фиксируются в [`Technical Debt.md`](rules/Technical%20Debt.md) и сопровождаются тестами (unit, integration, e2e).