# Implementation Plan

[Overview]
Цель: реализовать Фазу 1.2 (критические баги) — унифицировать порог релевантности (0.5), корректно обрабатывать min_score=0.0 в поиске, выровнять контракт статистики токенов с UI и актуализировать README.

Фаза 1.2 направлена на устранение несоответствий, выявленных аудитом и зафиксированных в .clinerules/activeContext.md и .clinerules/audit_results.md. Эти изменения не добавляют новую функциональность, а ремонтируют важные контракты и дефолты, влияющие на воспроизводимость результатов (score_threshold), стабильность UI (ключи статистики токенов) и качество диагностики (полнота статистики поиска). Корректная обработка min_score=0.0 устраняет edge-case, когда «ложное» значение 0.0 подменялось дефолтом, и улучшает экспериментирование с порогами. Синхронизация дефолтов на 0.5 предотвращает дрейф конфигураций и расхождения результатов между окружениями. Актуализация README устраняет устаревшие параметры и несуществующие команды, снижая support-нагрузку и ошибки пользователей. Все исправления назад-совместимы и не меняют общих API поверхностей, кроме расширения словарей статистики (добавлены новые ключи).

[Types]  
Типовые изменения минимальные; расширяются структуры словарей статистик.

- TokenUsageStats (dict), возвращается OpenAIManager.get_token_usage_stats():
  - used_today: int — обязательный ключ (для UI совместимости)
  - requests_today: int — обязательный ключ (для UI совместимости)
  - average_per_request: float — обязательный ключ (для UI совместимости)
  - total_requests: int — обратная совместимость
  - total_tokens: int — обратная совместимость
  - average_tokens_per_request: float — обратная совместимость  
  В первой итерации значения могут быть 0 (заглушка), контракт унифицирован.

- SearchStats (dict), возвращается SearchService.get_search_stats():
  - score_threshold: float — новый обязательный ключ; берется из config.rag.query_engine.score_threshold
  - Остальные поля не меняются (total_queries, cache_hits, cache_misses, avg_search_time, cache_size, cache_max_size, last_query_time и т.п.)

[Files]
Правки только в существующих файлах; новых модулей не добавляется.

- Модифицируемые файлы:
  1) rag/search_service.py
     - search(): заменить дефолт для порога релевантности:
       было: raw_results → _process_search_results(raw_results, min_score or 0.5)
       станет: raw_results → _process_search_results(raw_results, (min_score if min_score is not None else self.config.rag.query_engine.score_threshold))
     - _generate_cache_key(): корректно кодировать min_score=0.0:
       было: str(min_score) if min_score else ''
       станет: '' if min_score is None else str(min_score)
     - get_search_stats(): добавить stats["score_threshold"] = self.config.rag.query_engine.score_threshold
     - Обновить docstring/комментарии к соответствующим методам.

  2) config.py
     - dataclass QueryEngineConfig: сменить default для score_threshold:
       было: score_threshold: float = field(default_factory=lambda: safe_float("SEARCH_SCORE_THRESHOLD", "0.7"))
       станет: score_threshold: float = field(default_factory=lambda: safe_float("SEARCH_SCORE_THRESHOLD", "0.5"))

  3) openai_integration.py
     - OpenAIManager.get_token_usage_stats(): вернуть унифицированный контракт:
       {
         "used_today": 0,
         "requests_today": 0,
         "average_per_request": 0.0,
         "total_requests": 0,
         "total_tokens": 0,
         "average_tokens_per_request": 0.0
       }
       (минимальная унификация контракта; без точного дневного учета на первом шаге)

  4) README.md
     - Удалить устаревшие упоминания openai.max_tokens_per_chunk / max_response_tokens.
     - Убрать несуществующие CLI: rag clear, rag index --incremental, rag search --use-mmr/--diversity-lambda.
     - Синхронизировать описание структуры RAG-конфигурации с текущей (rag.vector_store/query_engine/parallelism вместо qdrant/search/indexing).

- Документация к обновлению ПОСЛЕ внедрения (в рамках завершения задачи):
  - .clinerules/activeContext.md — отметить чекбоксы Фазы 1.2 как выполненные.
  - .clinerules/audit_results.md — указать закрытые пункты (16, 17, 18 частично/полностью; 11 частично).
  - MILESTONE_M2_IMPLEMENTATION_PLAN.md — отметить подпункты Фазы 1.2 как выполненные.

[Functions]
Изменения точечные в существующих функциях; новых функций не добавляется.

- Новые функции: нет
- Модифицированные функции:
  - rag/search_service.py
    - async def search(...): заменить дефолт порога релевантности на min_score if min_score is not None else self.config.rag.query_engine.score_threshold.
    - def _generate_cache_key(...): заменить кодирование min_score → '' если None, иначе str(min_score).
    - def get_search_stats(self) -> Dict[str, Any]: добавить ключ score_threshold.
  - openai_integration.py
    - def get_token_usage_stats(self) -> Dict: вернуть used_today, requests_today, average_per_request + сохранить существующие ключи (total_requests, total_tokens, average_tokens_per_request).
- Удаляемые функции: нет

[Classes]
Единственная правка класса — дефолтное поле конфигурации.

- Модифицированные классы:
  - config.QueryEngineConfig: поле score_threshold — default "0.5" вместо "0.7".
- Новые/удаляемые классы: отсутствуют.

[Dependencies]
Изменения зависимостей не требуются в рамках Фазы 1.2.

- Будущие обновления (вне этой фазы): fastembed>=0.3.6, sentence-transformers>=5.1.0 (см. .clinerules/audit_results.md), после совместимых smoke-тестов.

[Testing]
Подход: добавить минимальные тесты на критические контракты; существующие тесты должны продолжать проходить без изменений.

- Новые тесты (предложение):
  1) tests/rag/test_search_service_min_score_zero.py
     - Мок vector_store.search (async) для выдачи N результатов (включая с score<0.5).
     - Проверка, что при min_score=None применяется конфигурируемый порог (0.5), а при min_score=0.0 фильтрация не подменяется дефолтом.
     - Косвенная проверка кэш-ключей: два вызова search c min_score=None и min_score=0.0 должны кешироваться по разным ключам.
  2) tests/test_token_stats_contract.py
     - Инициализировать OpenAIManager (с замоканным OpenAI при необходимости).
     - Проверить наличие ключей used_today, requests_today, average_per_request в get_token_usage_stats().
  3) tests/rag/test_search_stats_contains_threshold.py
     - Инициализировать SearchService (или через __new__ и проставление зависимостей) и проверить наличие score_threshold в get_search_stats().

- Smoke:
  - python main.py rag status --detailed → наличие score_threshold в статистике поиска.
  - python main.py token_stats → без исключений, корректный вывод used_today.
  - python main.py rag search "x" --min-score 0.0 → отсутствие подмены порога на дефолт.

[Implementation Order]
Сначала конфиг и поиск, затем контракт токенов и документация; после — тесты и smoke.

1) config.py: QueryEngineConfig.score_threshold default → 0.5.
2) rag/search_service.py:
   - search(): дефолт порога → min_score if min_score is not None else config.rag.query_engine.score_threshold
   - _generate_cache_key(): '' if min_score is None else str(min_score)
   - get_search_stats(): добавить score_threshold
3) openai_integration.py: get_token_usage_stats() — добавить used_today/requests_today/average_per_request с сохранением старых ключей.
4) README.md: удалить устаревшие параметры и опции CLI; синхронизировать структуру RAG-конфигурации.
5) Тесты: добавить предлагаемые тесты; запустить pytest селективно.
6) Smoke CLI: rag status --detailed; token_stats; rag search с --min-score 0.0.
7) Обновить документацию в .clinerules/* и MILESTONE_M2_IMPLEMENTATION_PLAN.md — отметить выполненные пункты Фазы 1.2 как закрытые.
