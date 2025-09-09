# Project Architect Rules (Non-Obvious Only)

- Архитектура построена вокруг гибридного поиска (dense + sparse).
- Все алгоритмы оптимизированы под CPU-first.
- Изменения в архитектуре должны фиксироваться в `.clinerules/`.
- Тестирование строго категоризировано (unit, integration, e2e, rag, property-based).