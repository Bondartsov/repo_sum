# SLA и SLO для RAG‑as‑a‑Service на VM

Документ задаёт цели уровня сервиса (SLO) и соглашение об уровне сервиса (SLA) для основных эндпоинтов сервиса, а также метрики, алерты и процедуры контроля.

Область охвата
- Эндпоинты: /v1/index, /v1/search, /v1/embeddings, /v1/health, /metrics
- Среда: VM инстанс, FastAPI + Qdrant, клиентские вызовы через RemoteVMVectorStore
- Наблюдаемость: Prometheus метрики из [vm_rag_service.py](vm_rag_service.py:261), логирование JSON, heartbeat

Цели SLO (рассчитываются помесячно)
- Доступность сервиса: 99.9% (ошибки 5xx и таймауты вычитаются)
- Поиск /v1/search:
  - p50 ≤ 1s
  - p95 ≤ 10s
  - p99 ≤ 15s
  - Ошибки (5xx + 422 по входу не учитываются в доступность, но мониторятся)
- Индексация /v1/index:
  - p95 указывается как бюджет: base + batch_size*step (см. timeout_profiles)
  - Доля отбракованных документов: ≤ 1% за сутки (reason=empty_text и пр.)
- Здоровье /v1/health: p99 ≤ 2s; error rate ≤ 1%

SLA (для внешних пользователей)
- Мы целью считаем: доступность 99.9% и соблюдение указанных SLO
- Нарушение SLA фиксируется, если за отчётный период нарушены и доступность, и ключевые SLO (поиск p95)

Метрики (Prometheus)
- rag_request_duration_seconds{endpoint,status} — Histogram
- rag_requests_total{endpoint,status} — Counter
- rag_inprogress_requests{endpoint} — Gauge
- rag_dropped_documents_total{reason} — Counter
- rag_timeouts_total{endpoint} — Counter
- rag_memory_usage_bytes — Gauge
- TODO: retries_total — счётчик ретраев (планируется)

Примеры PromQL (латентность)
- p95 для /v1/search за 5 минут:
  - histogram_quantile(0.95, sum(rate(rag_request_duration_seconds_bucket{endpoint="/v1/search"}[5m])) by (le))
- p50/p99 — аналогично, подставив 0.5/0.99

Примеры PromQL (ошибки/таймауты/дропы)
- Ошибки 5xx по /v1/search за 5 минут:
  - sum(rate(rag_requests_total{endpoint="/v1/search",status=~"5.."}[5m]))
- Таймауты поиска:
  - sum(rate(rag_timeouts_total{endpoint="/v1/search"}[5m]))
- Отброшенные документы (любые причины):
  - sum(rate(rag_dropped_documents_total[5m]))
- Отброшенные по empty_text:
  - sum(rate(rag_dropped_documents_total{reason="empty_text"}[5m]))

Алерты (уровни и окна)
- SearchLatencyP95High (warning): p95 /v1/search > 10s в течение 5м
- SearchLatencyP95Critical: p95 /v1/search > 15s в течение 5м
- HealthP99Slow: p99 /v1/health > 2s в течение 10м
- TimeoutsSpike: rag_timeouts_total по /v1/search > 5/мин в течение 5м
- DropsSpike: rag_dropped_documents_total > 1% от потока за 10м
- MemoryHigh: rag_memory_usage_bytes растёт > 95% RAM в течение 10м

Дашборды (Grafana)
- Обзор:
  - Карточки: Availability (30д), Error budget burn (24ч), Requests RPS, In‑progress
- Поиск:
  - Latency: p50/p95/p99 по /v1/search (стэки линий)
  - Error rate 5xx, таймауты
- Индексация:
  - Доля dropped по reason, общая скорость индекса (docs/s)
  - Потребление памяти (Gauge) и корреляция со скоростью

Источники метрик и трассировка
- Метрики отдаются на /metrics (см. [vm_rag_service.py](vm_rag_service.py:405))
- JSON‑логи с trace_id/batch_id на каждый запрос (см. middleware в [vm_rag_service.py](vm_rag_service.py:334))
- Клиентский heartbeat с коротким таймаутом ≤2s (см. [python.RemoteVMVectorStore.heartbeat()](rag/remote_vector_store.py:225))

Инцидент‑менеджмент (высокоуровневый)
- Диагностика доступности:
  - Проверить /v1/health локально/удалённо, логи VM
  - Проверить сеть/порт и firewall
- Диагностика поиска:
  - Посмотреть p95/p99, таймауты, нагрузку индексации
  - При необходимости ограничить параллельную индексацию
- Диагностика индексации:
  - Посмотреть долю dropped по reason, batch_size, память
  - Перезапуск проблемного батча (см. re‑delivery в клиенте)
- См. оперативные команды: [VM_SERVICE_CLI.md](VM_SERVICE_CLI.md:1)

SLI/отчётность
- Период: сутки/неделя/месяц
- Ключевые SLI:
  - Availability (1 - errors/requests) за период
  - Search p95 (из histogram), Health p99
  - Индексация: dropped share, docs/s

Соответствие плану (rules/all_refactor.md)
- SLO для /v1/search: p95 ≤ 10s — поддерживается метриками и алертами
- /v1/health ≤ 2s p99 — поддерживается
- dropped_documents_total{reason} — на дашборде и в алертах
- retries_total — TODO (после внедрения счётчика)

Приложение: чек‑лист готовности
- /metrics доступен и метрики публикуются
- Дашборд отображает latency, errors, timeouts, drops
- Настроены алерты SearchLatencyP95High/Critical, HealthP99Slow, TimeoutsSpike
- Heartbeat работает и отображается в UI/оперативной панели
- Процедуры: быстрая проверка [run_web.py](run_web.py:1), CLI [VM_SERVICE_CLI.md](VM_SERVICE_CLI.md:1)