# Implementation Plan

## [Overview]
Переход от mock-архитектуры к правильной категоризации тестов с использованием pytest маркеров для стабильной работы CI.

Проблема заключается в неправильной категоризации тестов в CI пайплайне. Этап "Run unit tests (offline)" использует флаг `--disable-socket`, который блокирует сетевые вызовы, но многие тесты, являющиеся по сути интеграционными или функциональными, не имеют соответствующих маркеров. В результате они запускаются вместе с unit тестами, пытаются выполнить запрещенные операции и падают с SocketBlockedError.

Решение: правильно классифицировать каждый тест с помощью pytest маркеров (@pytest.mark.integration, @pytest.mark.functional), чтобы CI мог разделить их выполнение на соответствующие этапы с правильными настройками сети.

## [Types]  
Добавление pytest маркеров для категоризации тестов без изменения типов данных.

Используются стандартные pytest маркеры:
- `@pytest.mark.functional` - для тестов, использующих subprocess и CLI интерфейсы
- `@pytest.mark.integration` - для тестов, работающих с файловой системой, OpenAI API, внешними зависимостями
- Без маркеров остаются только изолированные unit тесты

## [Files]
Модификация существующих тестовых файлов для добавления правильных pytest маркеров.

**Файлы, требующие изменений:**

**FUNCTIONAL маркеры (subprocess/CLI тесты):**
- `tests/test_additional_cli.py` - добавить `@pytest.mark.functional` к классу TestAdditionalCLI
- `tests/test_additional_config.py` - добавить `@pytest.mark.functional` к test_t003_cli_port_priority_over_env, test_t006_missing_required_openai_api_key, test_t007_invalid_env_type_validation
- `tests/test_additional_scanner.py` - добавить `@pytest.mark.functional` к test_main_analyze_command_integration
- `tests/test_additional_verify_requirements.py` - добавить `@pytest.mark.functional` к классу TestVerifyRequirements
- `tests/test_additional_web.py` - добавить `@pytest.mark.functional` к test_t004_web_occupied_port, test_t005_web_ui_404_unknown_route
- `tests/test_main.py` - добавить `@pytest.mark.functional` к test_main_cli_help

**INTEGRATION маркеры (файловая система/внешние зависимости):**
- `tests/test_additional_docgen.py` - добавить `@pytest.mark.integration` к test_t015_markdown_header_collisions, test_t016_long_lines_tables_lists
- `tests/test_additional_openai.py` - добавить `@pytest.mark.integration` к классам TestOpenAIRateLimit, TestOpenAIConnectionErrors
- `tests/test_additional_scanner.py` - добавить `@pytest.mark.integration` к остальным методам класса TestFileScannerAdditional
- `tests/test_config.py` - добавить `@pytest.mark.integration` к test_get_config_default
- `tests/test_error_handling.py` - добавить `@pytest.mark.integration` к test_openai_manager_no_api_key, test_openai_manager_network_error
- `tests/test_file_scanner.py` - добавить `@pytest.mark.integration` к test_scan_python_files
- `tests/test_integration_full_cycle.py` - добавить `@pytest.mark.integration` к test_full_analysis_cycle
- `tests/test_openai_integration.py` - добавить `@pytest.mark.integration` к test_analyze_chunk_with_mock

**Файлы без изменений (корректные unit тесты):**
- `tests/test_additional_chunker.py`, `tests/test_additional_parsers.py`, `tests/test_additional_utils.py`
- `tests/test_code_chunker.py`, `tests/test_doc_generator.py`, `tests/test_markdown_report.py`
- `tests/test_parsers.py`, `tests/test_property_based.py`, `tests/test_readme.py`
- `tests/test_run_web.py`, `tests/test_utils.py`, `tests/test_web_ui.py`
- `tests/e2e/*` (уже имеют корректные маркеры)
- `tests/test_new_functional.py`, `tests/test_new_integration.py` (уже имеют маркеры)

## [Functions]
Добавление декораторов к существующим тестовым функциям без изменения их логики.

**Новые функции:** Нет  
**Удаляемые функции:** Нет  
**Модифицируемые функции:**

**В tests/test_additional_cli.py:**
- Класс `TestAdditionalCLI` - добавить декоратор `@pytest.mark.functional`

**В tests/test_additional_config.py:**
- `test_t003_cli_port_priority_over_env()` - добавить `@pytest.mark.functional`
- `test_t006_missing_required_openai_api_key()` - добавить `@pytest.mark.functional`  
- `test_t007_invalid_env_type_validation()` - добавить `@pytest.mark.functional`

**В tests/test_additional_docgen.py:**
- `test_t015_markdown_header_collisions()` - добавить `@pytest.mark.integration`
- `test_t016_long_lines_tables_lists()` - добавить `@pytest.mark.integration`

**В tests/test_additional_openai.py:**
- Класс `TestOpenAIRateLimit` - добавить `@pytest.mark.integration`
- Класс `TestOpenAIConnectionErrors` - добавить `@pytest.mark.integration`

**В tests/test_additional_scanner.py:**
- Все методы класса `TestFileScannerAdditional` кроме `test_main_analyze_command_integration` - добавить `@pytest.mark.integration`
- `test_main_analyze_command_integration()` - добавить `@pytest.mark.functional`

**В tests/test_additional_verify_requirements.py:**
- Класс `TestVerifyRequirements` - добавить `@pytest.mark.functional`

**В tests/test_additional_web.py:**
- Все тестовые методы - добавить `@pytest.mark.functional`

**В остальных файлах:**
- По одной функции на файл с соответствующим маркером

## [Classes]
Добавление декораторов к тестовым классам без изменения их структуры.

**Новые классы:** Нет  
**Удаляемые классы:** Нет  
**Модифицируемые классы:**

- `TestAdditionalCLI` в tests/test_additional_cli.py - добавить `@pytest.mark.functional`
- `TestOpenAIRateLimit` в tests/test_additional_openai.py - добавить `@pytest.mark.integration`  
- `TestOpenAIConnectionErrors` в tests/test_additional_openai.py - добавить `@pytest.mark.integration`
- `TestVerifyRequirements` в tests/test_additional_verify_requirements.py - добавить `@pytest.mark.functional`
- Остальные модификации на уровне отдельных методов

## [Dependencies]
Никаких изменений в зависимостях не требуется.

pytest и все необходимые маркеры уже определены в pytest.ini. Конфигурация CI в .github/workflows/ci.yml уже правильно настроена для разделения тестов по маркерам.

## [Testing]
Поэтапная проверка правильности категоризации через локальное тестирование.

**Стратегия тестирования:**
1. **Локальная проверка unit тестов:** `pytest --disable-socket -m "not integration and not functional and not e2e" -v`
2. **Локальная проверка integration тестов:** `pytest -m "integration" -v` 
3. **Локальная проверка functional тестов:** `pytest -m "functional" -v`
4. **Проверка что ни один тест не запускается в нескольких категориях**
5. **Финальная проверка готовности к CI**

**Критерии успеха:**
- Unit тесты проходят с `--disable-socket` (без попыток сетевых подключений)
- Integration тесты запускаются только с доступом к сети
- Functional тесты корректно выполняют subprocess операции
- Нет тестов без категории, которые могут попасть не в ту группу

## [Implementation Order]
Пошаговая реализация с проверкой каждого этапа.

**Шаг 1: Критически важные FUNCTIONAL тесты (subprocess/CLI)**
- Модифицировать `tests/test_additional_cli.py` - добавить маркер к классу
- Модифицировать `tests/test_additional_config.py` - добавить маркеры к трём функциям
- Модифицировать `tests/test_main.py` - добавить маркер к CLI тесту
- **Проверка:** `pytest --disable-socket -m "not integration and not functional and not e2e" -v` не должен запускать эти тесты

**Шаг 2: Критически важные INTEGRATION тесты (файловая система/OpenAI)**  
- Модифицировать `tests/test_additional_openai.py` - добавить маркеры к классам
- Модифицировать `tests/test_config.py` - добавить маркер к функции чтения settings.json
- Модифицировать `tests/test_error_handling.py` - добавить маркеры к OpenAI тестам
- **Проверка:** Unit тесты больше не должны пытаться работать с OpenAI или файловой системой

**Шаг 3: Оставшиеся FUNCTIONAL тесты**
- Модифицировать `tests/test_additional_scanner.py` - добавить functional маркер к subprocess тесту
- Модифицировать `tests/test_additional_verify_requirements.py` - добавить маркер к классу
- Модифицировать `tests/test_additional_web.py` - добавить маркеры к веб-тестам
- **Проверка:** `pytest -m "functional" -v` должен запустить все CLI/subprocess тесты

**Шаг 4: Оставшиеся INTEGRATION тесты**
- Модифицировать `tests/test_additional_docgen.py` - добавить маркеры к тестам с файловой системой
- Модифицировать `tests/test_additional_scanner.py` - добавить integration маркеры к остальным тестам
- Модифицировать остальные integration файлы
- **Проверка:** `pytest -m "integration" -v` должен запустить все тесты с внешними зависимостями

**Шаг 5: Финальная верификация**
- Запустить полный набор тестов локально по категориям
- Убедиться что каждый тест попадает только в одну категорию
- Проверить что unit тесты работают с `--disable-socket`
- Подготовить к деплою в CI

**Шаг 6: CI проверка**
- Зафиксировать изменения в git
- Запустить GitHub Actions CI
- Убедиться что все этапы (unit/integration/functional) проходят успешно
- При необходимости скорректировать маркеры на основе результатов CI
