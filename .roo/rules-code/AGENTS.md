# Project Coding Rules (Non-Obvious Only)

- Обязательно устанавливать контекст фабрики на VM до создания приложения (иначе поднимутся неверные реализации). См. [vm_rag_service.py](vm_rag_service.py:98)
- Не убирать ленивые импорты RAG из CLI; оставлять импорты внутри команд (ускоряет старт и исключает тяжёлые зависимости). См. [main.py](main.py:32-36)
- Соблюдать контракт /v1/index: заголовок X-API-Contract v1.0.0 обязателен; batch 1..128; строгая sha256 при включении; учитывать метрику dropped_documents_total. См. [vm_rag_service.py](vm_rag_service.py:917-1004)
- Для /v1/search_v2: явно указывать протокол (dense/sparse); валидировать размерность вектора (fallback=1024); запрещать NaN/Inf. См. [vm_rag_service.py](vm_rag_service.py:650-717,831-846)
- Логи только через JSON-логер; не логировать содержимое документов; нормализовать пути /v1/* в метриках. См. [vm_rag_service.py](vm_rag_service.py:328-348,397)
- Ruff используется только для lint; «ruff format» не подключать без решения команды. См. [.ruff.toml](.ruff.toml)