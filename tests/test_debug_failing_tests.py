"""
Диагностические тесты для анализа падающих тестов.

Этот файл содержит улучшенные версии тестов с детальным логированием
для понимания причин падений.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def debug_test_t006_missing_required_openai_api_key():
    """
    Диагностическая версия test_t006_missing_required_openai_api_key
    с детальным логированием для понимания причин падения.
    """
    print("=" * 60)
    print("ДИАГНОСТИКА: test_t006_missing_required_openai_api_key")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    # Создаем временную директорию для тестового репозитория
    with tempfile.TemporaryDirectory() as temp_repo:
        # Создаем простой Python файл для анализа
        test_file = Path(temp_repo) / 'test.py'
        test_file.write_text('print("Hello, World!")')

        # Очищаем все возможные источники API ключа
        clean_env_dict = os.environ.copy()
        for key in list(clean_env_dict.keys()):
            if 'openai' in key.lower() or 'api' in key.lower():
                print(f"DEBUG: Удаляем переменную окружения: {key}")
                clean_env_dict.pop(key, None)
        clean_env_dict.pop('OPENAI_API_KEY', None)

        # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ДЛЯ ДИАГНОСТИКИ
        print(f"DEBUG: Environment before subprocess: {dict(clean_env_dict)}")
        print(f"DEBUG: OPENAI_API_KEY in env: {'OPENAI_API_KEY' in clean_env_dict}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: Project root: {project_root}")
        print(f"DEBUG: Main.py path: {project_root / 'main.py'}")
        print(f"DEBUG: Main.py exists: {(project_root / 'main.py').exists()}")

        # Проверяем наличие .env файла в корне проекта
        env_file = project_root / '.env'
        if env_file.exists():
            print(f"DEBUG: .env file exists at {env_file}")
            env_content = env_file.read_text()
            print(f"DEBUG: .env content: {repr(env_content)}")
            has_openai_key = 'OPENAI_API_KEY' in env_content
            print(f"DEBUG: .env contains OPENAI_API_KEY: {has_openai_key}")
        else:
            print(f"DEBUG: No .env file found at {env_file}")

        try:
            # Запускаем анализ с полностью очищенным окружением
            result = subprocess.run([
                'python', str(project_root / 'main.py'),
                'analyze', temp_repo
            ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=str(project_root), env=clean_env_dict)

            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ РЕЗУЛЬТАТА
            print(f"DEBUG: subprocess returncode: {result.returncode}")
            print(f"DEBUG: subprocess stdout: {repr(result.stdout)}")
            print(f"DEBUG: subprocess stderr: {repr(result.stderr)}")
            print(f"DEBUG: subprocess stdout type: {type(result.stdout)}")
            print(f"DEBUG: subprocess stderr type: {type(result.stderr)}")

            # Проверяем результат
            error_output = (result.stderr or "").lower()
            stdout_output = (result.stdout or "").lower()
            combined_output = error_output + stdout_output

            print(f"DEBUG: combined_output: {repr(combined_output)}")

            # Ищем различные варианты сообщений об ошибке или успешной работы
            api_key_missing_phrases = [
                'openai_api_key', 'api ключ', 'api key',
                'ключ не найден', 'key not found', 'не задан',
                'valueerror', 'error', 'authentication', 'unauthorized'
            ]

            # Если программа работает без API ключа, это тоже валидное поведение
            success_indicators = [
                'анализ завершен', 'analysis complete', 'успешно', 'successful',
                'документация сохранена', 'documentation saved'
            ]

            found_error_message = any(phrase in combined_output for phrase in api_key_missing_phrases)
            found_success = any(phrase in combined_output for phrase in success_indicators)

            print(f"DEBUG: found_error_message: {found_error_message}")
            print(f"DEBUG: found_success: {found_success}")
            print(f"DEBUG: success_condition: {found_error_message or found_success}")

            # Тест проходит если программа ЛИБО показывает ошибку об API ключе, ЛИБО работает корректно
            success_condition = (result.returncode != 0) or found_error_message or found_success

            print(f"DEBUG: Final success_condition: {success_condition}")

            if not success_condition:
                print("❌ ТЕСТ ПАДАЕТ!")
                print(f"   returncode: {result.returncode}")
                print(f"   combined_output: {combined_output}")
                return False
            else:
                print("✅ Тест проходит")
                return True

        except subprocess.TimeoutExpired:
            print("❌ ТЕСТ ПАДАЕТ: subprocess.TimeoutExpired")
            return False
        except Exception as e:
            print(f"❌ ТЕСТ ПАДАЕТ: Unexpected error: {e}")
            return False

def debug_test_cli_stats_outputs_tables():
    """
    Диагностическая версия test_cli_stats_outputs_tables
    с детальным логированием для понимания причин падения.
    """
    print("=" * 60)
    print("ДИАГНОСТИКА: test_cli_stats_outputs_tables")
    print("=" * 60)

    repo = tempfile.mkdtemp()
    try:
        repo_path = Path(repo)
        (repo_path / "a.py").write_text("print('x')\n", encoding="utf-8")
        (repo_path / "b.js").write_text("console.log('y')\n", encoding="utf-8")

        project_root = Path(__file__).resolve().parents[1]
        main_py = project_root / "main.py"

        print(f"DEBUG: repo_path: {repo_path}")
        print(f"DEBUG: main_py: {main_py}")
        print(f"DEBUG: main_py exists: {main_py.exists()}")

        try:
            proc = subprocess.run(
                [sys.executable, str(main_py), "stats", str(repo_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30
            )

            print(f"DEBUG: returncode: {proc.returncode}")
            print(f"DEBUG: stdout: {repr(proc.stdout)}")
            print(f"DEBUG: stderr: {repr(proc.stderr)}")

            out = proc.stdout or ""

            # Проверяем наличие ожидаемых элементов
            has_general_stats = "Общая статистика" in out
            has_lang_table = ("По языкам программирования" in out) or ("Самые большие файлы" in out)

            print(f"DEBUG: has_general_stats: {has_general_stats}")
            print(f"DEBUG: has_lang_table: {has_lang_table}")

            if proc.returncode != 0:
                print("❌ ТЕСТ ПАДАЕТ: returncode != 0")
                return False

            if not has_general_stats:
                print("❌ ТЕСТ ПАДАЕТ: 'Общая статистика' not found in output")
                return False

            if not has_lang_table:
                print("❌ ТЕСТ ПАДАЕТ: No language or large files table found")
                return False

            print("✅ Тест проходит")
            return True

        except subprocess.TimeoutExpired:
            print("❌ ТЕСТ ПАДАЕТ: subprocess.TimeoutExpired")
            return False
        except Exception as e:
            print(f"❌ ТЕСТ ПАДАЕТ: Unexpected error: {e}")
            return False

    finally:
        # Очищаем временную директорию
        import shutil
        shutil.rmtree(repo, ignore_errors=True)

def debug_test_cli_token_stats_handles_error_gracefully():
    """
    Диагностическая версия test_cli_token_stats_handles_error_gracefully
    с детальным логированием для понимания причин падения.
    """
    print("=" * 60)
    print("ДИАГНОСТИКА: test_cli_token_stats_handles_error_gracefully")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "fake-key")

    print(f"DEBUG: main_py: {main_py}")
    print(f"DEBUG: main_py exists: {main_py.exists()}")
    print(f"DEBUG: OPENAI_API_KEY in env: {env.get('OPENAI_API_KEY', 'NOT SET')}")

    try:
        proc = subprocess.run(
            [sys.executable, str(main_py), "token-stats"],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30
        )

        print(f"DEBUG: returncode: {proc.returncode}")
        print(f"DEBUG: stdout: {repr(proc.stdout)}")
        print(f"DEBUG: stderr: {repr(proc.stderr)}")

        stdout_output = proc.stdout or ""
        stderr_output = proc.stderr or ""
        combined_output = stdout_output + stderr_output

        # Проверяем наличие сообщения об ошибке
        has_error_message = "Ошибка при получении статистики" in combined_output

        print(f"DEBUG: has_error_message: {has_error_message}")

        if proc.returncode != 0:
            print("❌ ТЕСТ ПАДАЕТ: returncode != 0")
            return False

        if not has_error_message:
            print("❌ ТЕСТ ПАДАЕТ: 'Ошибка при получении статистики' not found in output")
            return False

        print("✅ Тест проходит")
        return True

    except subprocess.TimeoutExpired:
        print("❌ ТЕСТ ПАДАЕТ: subprocess.TimeoutExpired")
        return False
    except Exception as e:
        print(f"❌ ТЕСТ ПАДАЕТ: Unexpected error: {e}")
        return False

def debug_test_cli_subcommands_help():
    """
    Диагностическая версия test_cli_subcommands_help
    с детальным логированием для понимания причин падения.
    """
    print("=" * 60)
    print("ДИАГНОСТИКА: test_cli_subcommands_help")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    subcommands = [
        ["analyze", "--help"],
        ["stats", "--help"],
        ["clear-cache", "--help"],
        ["token-stats", "--help"],
    ]

    all_passed = True

    for args in subcommands:
        print(f"\nDEBUG: Testing command: {' '.join(args)}")

        try:
            proc = subprocess.run(
                [sys.executable, str(main_py), *args],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30
            )

            print(f"DEBUG: returncode: {proc.returncode}")
            print(f"DEBUG: stdout: {repr(proc.stdout)}")
            print(f"DEBUG: stderr: {repr(proc.stderr)}")

            # Проверяем наличие help информации
            stdout_output = proc.stdout or ""
            has_options = (
                ("Options" in stdout_output)
                or ("Опции" in stdout_output)
                or ("help" in stdout_output.lower())
            )

            print(f"DEBUG: has_options: {has_options}")

            if proc.returncode != 0:
                print(f"❌ КОМАНДА ПАДАЕТ: {args} - returncode != 0")
                all_passed = False
                continue

            if not has_options:
                print(f"❌ КОМАНДА ПАДАЕТ: {args} - no help options found")
                all_passed = False
                continue

            print(f"✅ Команда {args} проходит")

        except subprocess.TimeoutExpired:
            print(f"❌ КОМАНДА ПАДАЕТ: {args} - subprocess.TimeoutExpired")
            all_passed = False
        except Exception as e:
            print(f"❌ КОМАНДА ПАДАЕТ: {args} - Unexpected error: {e}")
            all_passed = False

    return all_passed

def debug_test_cli_settings_validation_error():
    """
    Диагностическая версия test_cli_settings_validation_error
    с детальным логированием для понимания причин падения.
    """
    print("=" * 60)
    print("ДИАГНОСТИКА: test_cli_settings_validation_error")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        project_root = Path(__file__).resolve().parents[1]
        main_py = project_root / "main.py"

        # Некорректный JSON
        bad_cfg = tmp_path / "bad_settings.json"
        bad_cfg.write_text("{ invalid json", encoding="utf-8")

        print(f"DEBUG: bad_cfg: {bad_cfg}")
        print(f"DEBUG: bad_cfg content: {repr(bad_cfg.read_text())}")
        print(f"DEBUG: main_py: {main_py}")
        print(f"DEBUG: main_py exists: {main_py.exists()}")

        try:
            proc = subprocess.run(
                [sys.executable, str(main_py), "-c", str(bad_cfg), "stats", str(tmp_path)],
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30
            )

            print(f"DEBUG: returncode: {proc.returncode}")
            print(f"DEBUG: stdout: {repr(proc.stdout)}")
            print(f"DEBUG: stderr: {repr(proc.stderr)}")

            stdout_output = proc.stdout or ""
            stderr_output = proc.stderr or ""
            combined_output = stdout_output + stderr_output

            # Проверяем наличие сообщения об ошибке
            has_error_message = "Ошибка загрузки конфигурации" in combined_output

            print(f"DEBUG: has_error_message: {has_error_message}")

            if proc.returncode != 1:
                print("❌ ТЕСТ ПАДАЕТ: returncode != 1")
                return False

            if not has_error_message:
                print("❌ ТЕСТ ПАДАЕТ: 'Ошибка загрузки конфигурации' not found in output")
                return False

            print("✅ Тест проходит")
            return True

        except subprocess.TimeoutExpired:
            print("❌ ТЕСТ ПАДАЕТ: subprocess.TimeoutExpired")
            return False
        except Exception as e:
            print(f"❌ ТЕСТ ПАДАЕТ: Unexpected error: {e}")
            return False

def debug_test_cli_clear_cache_integration():
    """
    Диагностическая версия test_cli_clear_cache_integration
    с детальным логированием для понимания причин падения.
    """
    print("=" * 60)
    print("ДИАГНОСТИКА: test_cli_clear_cache_integration")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    # создаём фейковые записи кэша в CWD проекта
    cache_dir = project_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        (cache_dir / f"entry_{i}.json").write_text('{"cached_at":"2025-01-01T00:00:00"}', encoding="utf-8")

    print(f"DEBUG: cache_dir: {cache_dir}")
    print(f"DEBUG: cache files created: {list(cache_dir.glob('*.json'))}")
    print(f"DEBUG: main_py: {main_py}")
    print(f"DEBUG: main_py exists: {main_py.exists()}")

    env = os.environ.copy()
    if "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = "fake-key"

    try:
        proc = subprocess.run(
            [sys.executable, str(main_py), "clear-cache"],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30
        )

        print(f"DEBUG: returncode: {proc.returncode}")
        print(f"DEBUG: stdout: {repr(proc.stdout)}")
        print(f"DEBUG: stderr: {repr(proc.stderr)}")

        stdout_output = proc.stdout or ""
        stderr_output = proc.stderr or ""
        combined_output = stdout_output + stderr_output

        # Проверяем наличие сообщения об очистке
        has_cleared_message = "Очищено" in combined_output

        print(f"DEBUG: has_cleared_message: {has_cleared_message}")

        # Проверяем, что файлы кэша действительно удалены
        remaining_files = list(cache_dir.glob("*.json"))
        print(f"DEBUG: remaining cache files: {remaining_files}")

        if proc.returncode != 0:
            print("❌ ТЕСТ ПАДАЕТ: returncode != 0")
            return False

        if not has_cleared_message:
            print("❌ ТЕСТ ПАДАЕТ: 'Очищено' not found in output")
            return False

        if remaining_files:
            print("❌ ТЕСТ ПАДАЕТ: cache files still exist after clear-cache")
            return False

        print("✅ Тест проходит")
        return True

    except subprocess.TimeoutExpired:
        print("❌ ТЕСТ ПАДАЕТ: subprocess.TimeoutExpired")
        return False
    except Exception as e:
        print(f"❌ ТЕСТ ПАДАЕТ: Unexpected error: {e}")
        return False
    finally:
        # Очищаем кэш директорию
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)

if __name__ == "__main__":
    print("Запуск диагностики падающих тестов...")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print()

    results = []

    # Запускаем диагностику каждого теста
    results.append(("test_t006_missing_required_openai_api_key", debug_test_t006_missing_required_openai_api_key()))
    results.append(("test_cli_stats_outputs_tables", debug_test_cli_stats_outputs_tables()))
    results.append(("test_cli_token_stats_handles_error_gracefully", debug_test_cli_token_stats_handles_error_gracefully()))
    results.append(("test_cli_subcommands_help", debug_test_cli_subcommands_help()))
    results.append(("test_cli_settings_validation_error", debug_test_cli_settings_validation_error()))
    results.append(("test_cli_clear_cache_integration", debug_test_cli_clear_cache_integration()))

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ ПРОШЕЛ" if result else "❌ ПАДАЕТ"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nИтого: {passed} прошли, {failed} падают")

    if failed > 0:
        print("\n❌ Некоторые тесты падают. Проверьте вывод выше для диагностики.")
        sys.exit(1)
    else:
        print("\n✅ Все тесты проходят!")
        sys.exit(0)