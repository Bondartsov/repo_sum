"""
ASCII диагностика падающих тестов.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

import pytest

def check_env_file():
    """Проверяем .env файл"""
    print("=" * 50)
    print("ПРОВЕРКА .ENV ФАЙЛА")
    print("=" * 50)

    env_file = Path("d:/Scripts_Python/repo_sum/.env")
    if env_file.exists():
        print(f"[OK] .env файл найден: {env_file}")
        try:
            with open(env_file, 'rb') as f:
                content = f.read()
                print(f"Размер файла: {len(content)} байт")

            for encoding in ['utf-8', 'cp1251', 'latin1']:
                try:
                    decoded = content.decode(encoding)
                    print(f"[OK] Успешно декодирован как {encoding}")
                    if 'OPENAI_API_KEY' in decoded:
                        print("[OK] Найден OPENAI_API_KEY в .env файле!")
                        return True
                    print("[FAIL] OPENAI_API_KEY НЕ найден в .env файле")
                    return False
                except UnicodeDecodeError:
                    print(f"[FAIL] Не удалось декодировать как {encoding}")
                    continue

            print("[FAIL] Не удалось декодировать .env файл ни в одной кодировке")
            return False
        except Exception as exc:
            print(f"[ERROR] Ошибка чтения .env файла: {exc}")
            return False
    else:
        print("[FAIL] .env файл НЕ найден")
        return False

def _run_cli(args, *, timeout=10):
    """Запускает subprocess с безопасной UTF-8 декодировкой."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout
    )

def test_subprocess_basic():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА SUBPROCESS")
    print("=" * 50)

    try:
        result = _run_cli(['python', '--version'])
    except Exception as exc:
        pytest.fail(f"[ERROR] Ошибка subprocess: {exc}")

    stdout_output = result.stdout or ""
    print(f"[OK] subprocess работает: {stdout_output.strip()}")
    assert result.returncode == 0, 'Команда python --version завершилась с ошибкой'
    assert stdout_output.strip(), 'Команда python --version не вернула вывод'

def test_main_py_exists():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА MAIN.PY")
    print("=" * 50)

    main_py = Path("d:/Scripts_Python/repo_sum/main.py")
    if main_py.exists():
        print(f"[OK] main.py найден: {main_py}")
    else:
        pytest.fail(f"[FAIL] main.py НЕ найден: {main_py}")

def test_openai_api_key_in_env():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА OPENAI_API_KEY В ОКРУЖЕНИИ")
    print("=" * 50)

    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"[OK] OPENAI_API_KEY найден в окружении: {api_key[:10]}...")
    else:
        pytest.fail("[FAIL] OPENAI_API_KEY НЕ найден в окружении")

def test_cli_help():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА CLI --HELP")
    print("=" * 50)

    main_py = "d:/Scripts_Python/repo_sum/main.py"

    try:
        result = _run_cli(['python', main_py, '--help'])
    except Exception as exc:
        pytest.fail(f"[ERROR] Ошибка при тестировании --help: {exc}")

    print(f"returncode: {result.returncode}")
    print(f"stdout: {repr(result.stdout)}")
    print(f"stderr: {repr(result.stderr)}")

    assert result.returncode == 0, 'Команда --help завершилась с ошибкой'
    stdout_output = result.stdout or ""
    assert 'Options' in stdout_output, 'Вывод --help не содержит блока Options'
    print("[OK] --help работает корректно")

def test_cli_stats():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА CLI STATS")
    print("=" * 50)

    main_py = "d:/Scripts_Python/repo_sum/main.py"

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        (repo_path / "test.py").write_text("print('hello')")

        try:
            result = _run_cli(['python', main_py, 'stats', str(repo_path)], timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail('[FAIL] stats зависает (timeout)')
        except Exception as exc:
            pytest.fail(f"[ERROR] Ошибка при тестировании stats: {exc}")

        print(f"returncode: {result.returncode}")
        print(f"stdout: {repr(result.stdout)}")
        print(f"stderr: {repr(result.stderr)}")

        assert result.returncode == 0, 'Команда stats завершилась с ошибкой'
        stdout_output = result.stdout or ""
        assert 'Общая статистика' in stdout_output, 'Вывод stats не содержит ожидаемого текста'
        print("[OK] stats работает корректно")

def main():
    print("DIAGNOSTIC: FAILING TESTS")
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")

    tests = [
        ("Проверка .env файла", check_env_file),
        ("Базовая проверка subprocess", test_subprocess_basic),
        ("Проверка main.py", test_main_py_exists),
        ("Проверка OPENAI_API_KEY в env", test_openai_api_key_in_env),
        ("Проверка CLI --help", test_cli_help),
        ("Проверка CLI stats", test_cli_stats),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\nTEST: {test_name}...")
        try:
            test_func()
            results.append((test_name, True))
        except AssertionError as err:
            print(f"[ASSERTION FAILED] {err}")
            results.append((test_name, False))
        except Exception as err:
            print(f"[CRITICAL ERROR] in test {test_name}: {err}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed

    for test_name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n[FAIL] Problems detected!")
        print("Possible causes:")
        print("1. Encoding issues with .env file")
        print("2. Subprocess issues (hangs, None in stdout)")
        print("3. Missing or invalid OPENAI_API_KEY")
        print("4. Issues with main.py or its dependencies")
        return False

    print("\n[SUCCESS] All basic checks passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
