@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo ========================================
echo    ЗАПУСК ТЕСТОВ REPO_SUM
echo ========================================
echo.

REM Проверяем аргументы
if "%1"=="" goto menu
if "%1"=="all" goto all_tests
if "%1"=="rag" goto rag_tests
if "%1"=="debug" goto debug_tests
if "%1"=="web" goto web_tests
if "%1"=="unit" goto unit_tests
if "%1"=="integration" goto integration_tests
if "%1"=="e2e" goto e2e_tests
goto custom_test

:menu
echo Выберите режим запуска:
echo.
echo [1] Все тесты (tests/)
echo [2] RAG тесты (tests/rag/)
echo [3] Debug тесты (tests/test_debug_*)
echo [4] Web тесты (tests/test_*_web.py)
echo [5] Unit тесты (категория unit)
echo [6] Integration тесты (категория integration)
echo [7] E2E тесты (категория e2e)
echo [8] Указать путь вручную
echo [0] Выход
echo.
set /p choice="Ваш выбор (0-8): "

if "%choice%"=="1" goto all_tests
if "%choice%"=="2" goto rag_tests
if "%choice%"=="3" goto debug_tests
if "%choice%"=="4" goto web_tests
if "%choice%"=="5" goto unit_tests
if "%choice%"=="6" goto integration_tests
if "%choice%"=="7" goto e2e_tests
if "%choice%"=="8" goto manual_path
if "%choice%"=="0" goto end
goto menu

:all_tests
echo.
echo [INFO] Запуск ВСЕХ тестов в папке tests/
echo.
set TEST_PATH=tests/
goto run_tests

:rag_tests
echo.
echo [INFO] Запуск RAG тестов (tests/rag/)
echo.
set TEST_PATH=tests/rag/
goto run_tests

:debug_tests
echo.
echo [INFO] Запуск Debug тестов
echo.
set TEST_PATH=tests/test_debug_ascii.py tests/test_debug_simple.py
goto run_tests

:web_tests
echo.
echo [INFO] Запуск Web тестов
echo.
set TEST_PATH=tests/test_additional_web.py tests/test_web_ui_vm_rag.py
goto run_tests

:unit_tests
echo.
echo [INFO] Запуск Unit тестов (маркер: unit)
echo.
set TEST_PATH=tests/ -m unit
goto run_tests

:integration_tests
echo.
echo [INFO] Запуск Integration тестов (маркер: integration)
echo.
set TEST_PATH=tests/ -m integration
goto run_tests

:e2e_tests
echo.
echo [INFO] Запуск E2E тестов (маркер: e2e)
echo.
set TEST_PATH=tests/ -m e2e
goto run_tests

:manual_path
echo.
set /p TEST_PATH="Введите путь к тесту (например: tests/test_debug_ascii.py::test_main_py_exists): "
if "%TEST_PATH%"=="" goto menu
goto run_tests

:custom_test
set TEST_PATH=%*
goto run_tests

:run_tests
echo ========================================
echo Команда: pytest %TEST_PATH%
echo Лог: log_tests.txt
echo ========================================
echo.

REM Запускаем pytest с подробным выводом
python -m pytest %TEST_PATH% -vv --tb=long --capture=no --durations=10 --showlocals 2>&1 | python -c "import sys; [print(line.rstrip()) for line in sys.stdin]" > log_tests.txt

REM Показываем результат в консоли
type log_tests.txt

echo.
echo ========================================
echo Тесты завершены!
echo Полный лог сохранен в: log_tests.txt
echo ========================================
echo.

REM Спрашиваем, запустить ли еще тесты
set /p again="Запустить другие тесты? (y/n): "
if /i "%again%"=="y" goto menu

goto end

:end
echo.
echo Завершение работы...
exit /b 0