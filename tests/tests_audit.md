# Аудит тестов проекта repo_sum (обновлено: 12.08.2025, E2E добавлены)

Документ фиксирует актуальную картину по тестам: список и описание сценариев, распределение по типам тестов, матрицу покрытия, а также планы по достижению целевой «пирамиды тестирования».

## Сводка (после добавления интеграционных, функциональных и E2E тестов)

- Тестовые модули: 20 (tests/test_*.py, включая tests/e2e/*)
- Явные pytest-тест-функции: 30
  - Параметризованный тест (на 5 языков) генерирует 5 прогонов
  - Property-based тест генерирует множество случайных прогонов
- Диагностический скрипт: tests/test_api_init.py (не pytest-кейс)
- Распределение типов (по функциям):
  - Unit: 12/30 ≈ 40% (включая 1 property-based)
  - Integration: 8/30 ≈ 27%
  - Functional/Smoke: 8/30 ≈ 27%
  - End-to-End (E2E): 2/30 ≈ 6.7%
- Маркеры pytest (pytest.ini): asyncio, integration, functional, smoke, e2e, property

Комментарий:
- Добавлены E2E-тесты (2 шт.), покрывающие полный запуск CLI analyze в двух реалистичных режимах (без подгона кода под тесты). E2E-доля ≈ 6.7% — входит в целевой «верх пирамиды» (5–10%). Доля unit низкая (≈40%) — ее планово поднимать за счёт тестов для RAG и core-утилит.

---

## Подробный разбор по тестам (существовавшие ранее)

### tests/test_config.py::test_get_config_default
- Тип: Unit
- Сценарий: get_config() возвращает объект Config с корректными полями.
- Как работает: вызывает get_config(), проверяет тип и поля (например, openai.api_key_env_var).

### tests/test_code_chunker.py::test_chunk_python_code
- Тип: Unit
- Сценарий: CodeChunker корректно разбивает Python-код на чанки и сохраняет содержимое функций.
- Как работает: FileInfo + код; chunk_code(); проверка «def foo()» и «def bar()».

### tests/test_doc_generator.py::test_generate_markdown
- Тип: Unit
- Сценарий: DocumentationGenerator создаёт валидный Markdown (импорты/комментарии).
- Как работает: ParsedFile; generate_markdown(); assert «os» и «Test comment».

### tests/test_error_handling.py::test_openai_manager_no_api_key
- Тип: Unit (негативный)
- Сценарий: При отсутствии OPENAI_API_KEY OpenAIManager выбрасывает ValueError.
- Как работает: monkeypatch.delenv(); ожидается ValueError.

### tests/test_error_handling.py::test_python_parser_syntax_error
- Тип: Unit
- Сценарий: PythonParser корректно отражает синтаксическую ошибку.
- Как работает: broken.py; parse_file(); parsed.parse_errors.

### tests/test_error_handling.py::test_openai_manager_network_error
- Тип: Integration/Functional
- Сценарий: Обработка сетевой ошибки OpenAI.
- Как работает: scan → parse → chunk → analyze (mock _call_openai_api → OpenAIError); возвращается result.error.

### tests/test_file_scanner.py::test_scan_python_files
- Тип: Unit
- Сценарий: FileScanner находит .py и игнорирует неподдерживаемые.
- Как работает: a.py + b.txt; scan_repository(); assert.

### tests/test_integration_full_cycle.py::test_full_analysis_cycle
- Тип: Integration (asyncio)
- Сценарий: Полный цикл: scan → parse → chunk → analyze(mock) → markdown.
- Как работает: foo.py; мок OpenAI; проверка содержимого.

### tests/test_main.py::test_main_cli_help
- Тип: Functional/Smoke (CLI)
- Сценарий: --help на main.py.
- Как работает: subprocess.run(); код 0; вывод включает help.

### tests/test_markdown_report.py::test_markdown_report_sections
- Тип: Unit
- Сценарий: Markdown-отчёт содержит секции, нет заглушек.
- Как работает: generate_markdown(); assert «Импорты», «Комментарии»; отсутствие «Ошибка анализа» и «{code_content}».

### tests/test_openai_integration.py::test_analyze_chunk_with_mock
- Тип: Unit
- Сценарий: analyze_chunk возвращает результат при mock-ответе OpenAI.
- Как работает: CodeChunk + GPTAnalysisRequest; mock OpenAI; assert not None.

### tests/test_parsers.py::test_parser_extracts_classes (param)
- Тип: Unit
- Сценарий: Парсеры Python/Cpp/CSharp/TS/JS извлекают классы.
- Как работает: parametrized; parse_file(); assert e.type=="class".

### tests/test_property_based.py::test_python_parser_property_based
- Тип: Unit (Property-based)
- Сценарий: Устойчивость PythonParser к случайным строкам.
- Как работает: Hypothesis; parse_file(); без необработанных исключений.

### tests/test_readme.py::test_readme_exists_and_content
- Тип: Functional/Smoke
- Сценарий: README.md существует и содержит «OpenAI» или «GPT».

### tests/test_run_web.py::test_run_web_import
- Тип: Functional/Smoke
- Сценарий: run_web.py импортируется без ошибок.

### tests/test_utils.py::test_ensure_directory_exists
- Тип: Unit
- Сценарий: ensure_directory_exists создаёт директорию.

### tests/test_web_ui.py::test_web_ui_import
- Тип: Functional/Smoke
- Сценарий: web_ui.py импортируется.

### tests/test_api_init.py (диагностический скрипт)
- Тип: Diagnostic
- Сценарий: Ручная проверка загрузки .env, config и инициализации OpenAIManager.

---

## Новые добавленные тесты (integration/functional)

Файл: tests/test_new_integration.py
- test_openai_cache_hit_on_second_call (integration, asyncio)
  - Кэширование OpenAI-анализа: второй вызов из кэша, OpenAI вызывается ровно 1 раз.
- test_full_cycle_multiple_files (integration, asyncio)
  - Сквозной цикл для нескольких файлов (.py/.js/.ts); chunk_parsed_file(parsed, code).
- test_openai_retries_and_error_propagation (integration, asyncio)
  - Ретраи и корректное сообщение об ошибке.
- test_cli_clear_cache_integration (integration)
  - CLI clear-cache очищает cache/*.json в корне проекта.
- test_async_concurrent_analysis (integration, asyncio)
  - Параллельный анализ 5 запросов через asyncio.gather.
- test_incremental_analysis_skips_unchanged (integration, asyncio)
  - analyze_repository с incremental=True пропускает неизменённые файлы.

Файл: tests/test_new_functional.py
- test_cli_analyze_incremental_no_changes_success (functional)
  - CLI analyze в инкрементальном режиме без изменений; успешное завершение.
- test_cli_stats_outputs_tables (functional)
  - CLI stats печатает таблицы.
- test_cli_token_stats_handles_error_gracefully (functional)
  - CLI token-stats корректно обрабатывает несовпадение ключей статистики.
- test_cli_subcommands_help (functional/smoke)
  - --help для analyze/stats/clear-cache/token-stats.
- test_cli_settings_validation_error (functional)
  - Некорректный settings.json (-c) → код 1 и сообщение об ошибке.

---

## E2E тесты (добавлены и запущены)

Файлы: tests/e2e/*
- test_e2e_cli_analyze_incremental.py::test_e2e_cli_analyze_incremental_skip
  - Тип: E2E + Integration
  - Сценарий: Полный запуск CLI analyze в инкрементальном режиме при отсутствии изменений.
  - Реальная картина: Загружается settings.json из корня проекта; инициализируется OpenAIManager (OPENAI_API_KEY=fake), но анализ не выполняется — ранний выход («Нет изменений — отчёты актуальны»), успешное завершение без сетевых вызовов.
- test_e2e_cli_analyze_generate_docs.py::test_e2e_cli_analyze_generates_docs_without_openai
  - Тип: E2E + Integration
  - Сценарий: Полный запуск CLI analyze без инкремента и без сети (OPENAI_API_KEY=fake).
  - Реальная картина: Команда завершается успешно; создаётся SUMMARY_REPORT_<repo>/README.md; формируются отчёты по файлам. В условиях отсутствия доступа к OpenAI часть отчётов может содержать «Ошибка анализа» (фиксируем это как ожидаемое поведение), но при этом README/структура отчётов создаются корректно.

Результат прогонов E2E:
- Запуск: `pytest -q -m "e2e"` — 2 passed.

Важно:
- Код приложения не подгонялся под тесты. Мы зафиксировали «как есть» сценарии: без сетевого доступа к OpenAI анализ либо пропускается (incremental skip), либо генерирует отчёты с возможной «Ошибка анализа» в содержимом, при этом вся файловая структура документации создаётся корректно.

---

## Матрица покрытия (области → тесты)

- Конфигурация: test_config.py; (diag) test_api_init.py; test_error_handling.py::test_openai_manager_no_api_key; test_new_functional.py::test_cli_settings_validation_error
- Сканирование: test_file_scanner.py; test_integration_full_cycle.py; test_new_integration.py::test_full_cycle_multiple_files
- Парсеры: test_parsers.py; test_property_based.py; test_error_handling.py::test_python_parser_syntax_error; test_integration_full_cycle.py; test_new_integration.py::test_full_cycle_multiple_files
- Чанкёр: test_code_chunker.py; test_integration_full_cycle.py; test_new_integration.py::test_full_cycle_multiple_files
- OpenAI интеграция: test_openai_integration.py; test_error_handling.py::test_openai_manager_network_error; test_integration_full_cycle.py; test_new_integration.py::{cache_hit,retries,concurrency}
- Кэширование/инкрементальность: test_new_integration.py::{cache_hit,incremental_skip}; test_new_functional.py::test_cli_analyze_incremental_no_changes_success; E2E incremental_skip
- Генерация документации: test_doc_generator.py; test_markdown_report.py; test_integration_full_cycle.py; test_new_integration.py::test_full_cycle_multiple_files; E2E generate_docs_without_openai
- CLI: test_main.py::help; test_new_functional.py::{analyze,stats,token-stats,help,settings}; test_new_integration.py::clear-cache; E2E CLI analyze (2 сценария)
- Web/Streamlit: test_web_ui.py (import), test_run_web.py (import)
- Документация/репо: test_readme.py

---

## Пирамида тестирования — текущее vs целевое

Текущее:
- Unit ≈ 40%
- Integration ≈ 27%
- Functional/Smoke ≈ 27%
- E2E ≈ 6.7% (входит в целевой диапазон 5–10%)

Целевое:
- Unit: 70–80%
- Integration/Functional: 15–20%
- E2E: 5–10%

План выравнивания:
- Нарастить unit-основание (RAG-модули: embedder, vector_store, query_engine; utils: sanitize_text, GPTCache TTL/истечение, truncate_to_tokens; unit для CLI-хелперов).
- Сохранить/укрепить E2E на 2–4 теста (сейчас 2 ок).
- Поддерживать интеграционные/функциональные в рамках 15–20% за счёт переноса части логики в unit.

---

## TODO (актуализировано)

- [ ] E2E: добавить RAG index→search (после реализации CLI/интеграции с Qdrant), со скипом при недоступности сервиса.
- [ ] E2E: web_ui smoke через requests/playwright (условный запуск в CI).
- [ ] Unit: добить core и RAG (embedder/vector_store/query_engine) + utils (sanitize_text, GPTCache, truncate_to_tokens).
- [ ] pytest-cov: включить сбор покрытия (цели: core ≥80%, RAG ≥70%).
- [ ] tests/conftest.py: фикстура fake_openai_key для унификации тестов, где требуется OPENAI_API_KEY.
