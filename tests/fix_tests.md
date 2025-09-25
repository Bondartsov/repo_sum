# Fix Tests Report

## tests/e2e/test_e2e_cli_analyze_generate_docs.py
- **Test**: `test_e2e_cli_analyze_generates_docs_without_openai`
  - **Назначение**: Проверяет, что команда `analyze` без реального OpenAI ключа создаёт отчёт и не падает.
  - **Причина**: CLI перенастраивает `sys.stdout`/`sys.stderr` на UTF-8, тогда как `subprocess.run(..., text=True)` в Windows декодирует пайпы через cp1251. Эмодзи и другие UTF-8 байты дают `UnicodeDecodeError` в потоке чтения, поэтому `proc.stdout`/`proc.stderr` остаются `None`, и проверка `proc.stdout + proc.stderr` рушится.
  - **Как починить**: Не переопределять кодировку stdout/stderr, либо запускать подпроцессы с `encoding="utf-8"` (или через `PYTHONIOENCODING=utf-8`). Дополнительно можно убрать эмодзи.

## tests/rag/test_jina_v3_performance_impact.py
- **Test**: `TestJinaV3PerformanceImpact::test_embedding_latency_impact`
  - **Назначение**: Сравнивает латентность батчей между BGE-small и Jina v3.
  - **Причина**: В `PerformanceBenchmarker._measure_embedding_performance` вызывается `await monitor.start_monitoring()`, хотя `start_monitoring` синхронный и возвращает `None`.
  - **Как починить**: Сделать `start_monitoring` асинхронным либо убрать `await` и вызывать метод синхронно.
- **Test**: `TestJinaV3PerformanceImpact::test_search_performance_regression`
  - **Назначение**: Измеряет регрессию поиска при переходе на Jina v3.
  - **Причина**: Та же ошибка с `await monitor.start_monitoring()` в `_measure_search_performance`.
  - **Как починить**: Аналогично убрать `await` или сделать метод асинхронным.
- **Test**: `TestJinaV3PerformanceImpact::test_memory_usage_scaling`
  - **Назначение**: Проверяет, что рост пикового потребления памяти Jina v3 не превышает допуски относительно BGE.
  - **Причина**: RSS-показатель `psutil` для BGE почти не меняется (≈0 МБ), из-за чего отношение `jina_memory_usage / bge_memory_usage` становится >7x.
  - **Как починить**: Ввести минимальный порог для знаменателя, отдельно прогревать модель или мерить память через snaphot-разницу, чтобы не делить на почти ноль.

## tests/rag/test_query_engine_health.py
- **Test**: `TestQueryEngineHealthFixes::test_check_vector_store_success`
  - **Назначение**: Убеждается, что `_check_vector_store` дергает health-check и возвращает `True` при статусе ok.
  - **Причина**: `_check_vector_store` вызывает `self.vector_store.health_check()`, но у стора есть метод `check_health`, поэтому возникает `AttributeError`.
  - **Как починить**: Заменить вызов на `await self.vector_store.check_health()` (и корректно разобрать ответ).
- **Test**: `...::test_check_vector_store_disconnected`
  - **Назначение**: Проверяет отрицательный сценарий health-check.
  - **Причина**: Та же неверная сигнатура.
  - **Как починить**: Аналогично исправить вызов.
- **Test**: `...::test_check_vector_store_exception`
  - **Назначение**: Убеждается, что исключение при health-check приводит к `False`.
  - **Причина**: `AttributeError` из-за неправильного вызова.
  - **Как починить**: После замены метода исключение будет перехвачено блоком try/except.
- **Test**: `...::test_ensure_embeddings_fallback_default_dimension`
  - **Назначение**: Гарантирует, что fallback создаёт векторы нужной размерности (1024d).
  - **Причина**: В ветке fallback используется значение по умолчанию 384, когда у `QueryEngineConfig` нет `vector_store`.
  - **Как починить**: Определять размерность по `self.vector_store.config.vector_size` или по `self.embedder.embedding_dim`.
- **Test**: `...::test_health_check_integration`
  - **Назначение**: Комплексно проверяет `health_check` и `_check_vector_store`.
  - **Причина**: Та же ошибка вызова `health_check()`.
  - **Как починить**: После исправления вызова тест пройдёт.

## tests/rag/test_rag_e2e_cli.py
- **Test**: `TestRAGCliE2E::test_rag_search_command_basic`
  - **Назначение**: Энд-ту-энд проверка `rag search` с моками Qdrant и embedder.
  - **Причина**: Настройки (settings.json + `.env`) форсируют `remote-vm` провайдер, из-за чего CLI всё равно обращается к VM 10.61.11.54 и падает по таймауту.
  - **Как починить**: Добавить тестовый профиль/флаг, который принудительно включает локальный `CPUEmbedder`/mock Qdrant, либо уважать `--config` и не перетирать провайдер из `.env`.
- **Test**: `TestRAGCliE2E::test_rag_performance_cli_metrics`
  - **Назначение**: Проверяет SLA `rag index` и `rag search`.
  - **Причина**: Те же удалённые зависимости (таймауты ~30 c), поэтому проверки по времени проваливаются.
  - **Как починить**: В тестовом режиме отключать network, уменьшать таймауты или подставлять быстрые моки.

## tests/rag/test_rag_integration.py
- **Test**: `TestRAGIntegration::test_search_service_integration`
  - **Назначение**: Интеграционный тест `SearchService` с моками.
  - **Причина**: Конфиг по умолчанию ставит `embeddings.provider="remote-vm"`, поэтому сервис строит `RemoteVMEmbedder` и падает `VectorStoreException` при поиске.
  - **Как починить**: Использовать конфиг без overrides из `.env` или внедрять mock через DI/флаг.
- **Test**: `...::test_full_rag_pipeline`
  - **Назначение**: Проверка полного пайплайна индексирования и поиска.
  - **Причина**: Те же обращения к удалённым сервисам → `VectorStoreConnectionError`.
  - **Как починить**: Перевести пайплайн на offline-моки.
- **Test**: `...::test_rag_concurrent_operations`
  - **Назначение**: Оценивает конкурентные запросы.
  - **Причина**: Все запросы падают из-за network, в итоге условие по числу успешных ответов не выполняется.
  - **Как починить**: После внедрения локальных моков ошибки исчезнут.

## tests/rag/test_rag_performance.py
- **Test**: `TestRAGPerformance::test_search_performance`
  - **Назначение**: Снимает метрики поиска.
  - **Причина**: `SearchService` обращается к реальному стораджу → `VectorStoreException`.
  - **Как починить**: Использовать mock vector store.
- **Test**: `...::test_full_pipeline_performance`
  - **Назначение**: Проверяет полный RAG пайплайн.
  - **Причина**: Ошибки из-за недоступных удалённых сервисов накапливаются и тест падает.
  - **Как починить**: Использовать offline-моки/режим без сети.
- **Test**: `...::test_stress_concurrent_users`
  - **Назначение**: Нагрузочный сценарий.
  - **Причина**: Все попытки завершаются ошибками соединения (100% failure).
  - **Как починить**: Подставить локальные моки и настроить graceful-обработку connection errors.

## tests/test_additional_config.py
- **Test**: `test_t006_missing_required_openai_api_key`
  - **Назначение**: Проверяет, что CLI сообщает об отсутствии `OPENAI_API_KEY`.
  - **Причина**: `load_dotenv()` подхватывает prod `.env` с реальным ключом и remote-настройками, поэтому команда `analyze` зависает на сетевых вызовах и вылетает по таймауту.
  - **Как починить**: Подменить `.env` на тестовый (без ключа) или добавить флаг `--offline/--no-openai` для мгновенного отказа.

## tests/test_new_functional.py
- **Test**: `test_cli_stats_outputs_tables`
  - **Назначение**: Проверяет вывод `stats`.
  - **Причина**: Из-за перекодировки stdout/stderr в UTF-8 поток чтения падает (`stdout` становится `None`).
  - **Как починить**: Не менять кодировку stdout/stderr или запускать подпроцесс с `encoding="utf-8"`; убрать эмодзи.
- **Test**: `test_cli_token_stats_handles_error_gracefully`
  - **Назначение**: Проверяет graceful-сообщение от `token-stats`.
  - **Причина**: Та же проблема с декодированием.
  - **Как починить**: То же решение.
- **Test**: `test_cli_subcommands_help`
  - **Назначение**: Проверяет `--help` всех подкоманд.
  - **Причина**: UnicodeDecodeError → `stdout` равен `None`.
  - **Как починить**: То же решение.
- **Test**: `test_cli_settings_validation_error`
  - **Назначение**: Проверяет сообщение об ошибке при битом JSON.
  - **Причина**: CLI пишет UTF-8, подпроцесс читает cp1251, текст превращается в кракозябры и assert не находит строку.
  - **Как починить**: Согласовать кодировки.

## tests/test_new_integration.py
- **Test**: `test_cli_clear_cache_integration`
  - **Назначение**: Интеграционный тест `clear-cache`.
  - **Причина**: Та же несогласованная кодировка, из-за чего сообщение "Очищено" превращается в набор символов.
  - **Как починить**: Исправить настройку stdout/stderr или декодирование в тесте.

## tests/vm/test_vm_firewall_config.py
- **Test**: `TestVMFirewallConfig::test_port_listening_on_vm`
  - **Назначение**: Убедиться, что сервис на 10.61.11.54:8000 доступен.
  - **Причина**: На тестовом стенде VM/порт недоступны, поэтому проверка слушателя проваливается.
  - **Как починить**: Предоставить рабочую VM или пропускать тесты без `VM_PASSWORD`/доступа.
- **Test**: `TestVMFirewallConfig::test_comprehensive_firewall_suite`
  - **Назначение**: Полный аудит фаервола.
  - **Причина**: Сабтест внешней коннективности падает по той же причине.
  - **Как починить**: Аналогично — доступная VM или условный skip.

## tests/rag/test_sparse_encoder.py
- **Test**: `test_initialization`
  - **Назначение**: Проверяет, что `SparseEncoder` поднимает токенайзер и модель.
  - **Причина**: Конструктор вызывает `AutoTokenizer/AutoModel.from_pretrained(..., local_files_only=True)` без локального кэша и без fallback.
  - **Как починить**: Обернуть загрузку в try/except и при `OSError` использовать `MockTokenizer`/`MockSparseModel`.
- **Test**: `test_encode_nonempty`
  - **Назначение**: Убедиться, что `encode` возвращает список словарей.
  - **Причина**: Следствие той же ошибки инициализации.
  - **Как починить**: После добавления fallback тест пройдёт.
- **Test**: `test_encode_stability`
  - **Назначение**: Проверяет детерминированность `encode`.
  - **Причина**: Encoder не инициализировался, поэтому падает ещё до assert.
  - **Как починить**: Та же обработка offline.
- **Test**: `test_tokenization_different_words`
  - **Назначение**: Проверяет различие токенов для разных слов.
  - **Причина**: Encoder не создал токенайзер из-за отсутствия модели.
  - **Как починить**: Как выше.
- **Test**: `test_encode_multiple_sentences`
  - **Назначение**: Проверяет пакетную обработку.
  - **Причина**: Тот же `OSError`.
  - **Как починить**: Добавить fallback.
- **Test**: `test_sparse_vector_nonnegative_and_normalized`
  - **Назначение**: Проверяет неотрицательность и нормировку весов.
  - **Причина**: Конструктор не завершился, `encode` не работает.
  - **Как починить**: Как выше.
- **Test**: `test_property_based_encode_returns_valid_dicts`
  - **Назначение**: Property-based проверка корректности `encode`.
  - **Причина**: Та же невозможность загрузить модель.
  - **Как починить**: После добавления fallback всё заработает.

## Итог
25 падений и 7 ошибок приходятся на: (1) рассинхрон кодировок CLI, (2) зависимость от production RAG/OpenAI/VM, (3) отсутствие offline fallback для HuggingFace модели. Исправление этих блоков стабилизирует прогон.
