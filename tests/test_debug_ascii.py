"""
ASCII диагностика падающих тестов.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

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

            # Пробуем разные кодировки
            for encoding in ['utf-8', 'cp1251', 'latin1']:
                try:
                    decoded = content.decode(encoding)
                    print(f"[OK] Успешно декодирован как {encoding}")
                    if 'OPENAI_API_KEY' in decoded:
                        print("[OK] Найден OPENAI_API_KEY в .env файле!")
                        return True
                    else:
                        print("[FAIL] OPENAI_API_KEY НЕ найден в .env файле")
                        return False
                except UnicodeDecodeError:
                    print(f"[FAIL] Не удалось декодировать как {encoding}")
                    continue

            print("[FAIL] Не удалось декодировать .env файл ни в одной кодировке")
            return False
        except Exception as e:
            print(f"[ERROR] Ошибка чтения .env файла: {e}")
            return False
    else:
        print("[FAIL] .env файл НЕ найден")
        return False

def test_subprocess_basic():
    """Базовая проверка subprocess"""
    print("\n" + "=" * 50)
    print("ПРОВЕРКА SUBPROCESS")
    print("=" * 50)

    try:
        result = subprocess.run(
            ['python', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"[OK] subprocess работает: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка subprocess: {e}")
        return False

def test_main_py_exists():
    """Проверяем существование main.py"""
    print("\n" + "=" * 50)
    print("ПРОВЕРКА MAIN.PY")
    print("=" * 50)

    main_py = Path("d:/Scripts_Python/repo_sum/main.py")
    if main_py.exists():
        print(f"[OK] main.py найден: {main_py}")
        return True
    else:
        print(f"[FAIL] main.py НЕ найден: {main_py}")
        return False

def test_openai_api_key_in_env():
    """Проверяем наличие API ключа в окружении"""
    print("\n" + "=" * 50)
    print("ПРОВЕРКА OPENAI_API_KEY В ОКРУЖЕНИИ")
    print("=" * 50)

    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"[OK] OPENAI_API_KEY найден в окружении: {api_key[:10]}...")
        return True
    else:
        print("[FAIL] OPENAI_API_KEY НЕ найден в окружении")
        return False

def test_cli_help():
    """Тестируем --help команды"""
    print("\n" + "=" * 50)
    print("ПРОВЕРКА CLI --HELP")
    print("=" * 50)

    main_py = "d:/Scripts_Python/repo_sum/main.py"

    try:
        result = subprocess.run(
            ['python', main_py, '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )

        print(f"returncode: {result.returncode}")
        print(f"stdout: {repr(result.stdout)}")
        print(f"stderr: {repr(result.stderr)}")

        if result.returncode == 0 and 'Options' in result.stdout:
            print("[OK] --help работает корректно")
            return True
        else:
            print("[FAIL] --help НЕ работает корректно")
            return False

    except Exception as e:
        print(f"[ERROR] Ошибка при тестировании --help: {e}")
        return False

def test_cli_stats():
    """Тестируем команду stats"""
    print("\n" + "=" * 50)
    print("ПРОВЕРКА CLI STATS")
    print("=" * 50)

    main_py = "d:/Scripts_Python/repo_sum/main.py"

    # Создаем временный репозиторий
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        (repo_path / "test.py").write_text("print('hello')")

        try:
            result = subprocess.run(
                ['python', main_py, 'stats', str(repo_path)],
                capture_output=True,
                text=True,
                timeout=30
            )

            print(f"returncode: {result.returncode}")
            print(f"stdout: {repr(result.stdout)}")
            print(f"stderr: {repr(result.stderr)}")

            if result.returncode == 0 and 'Общая статистика' in result.stdout:
                print("[OK] stats работает корректно")
                return True
            else:
                print("[FAIL] stats НЕ работает корректно")
                return False

        except subprocess.TimeoutExpired:
            print("[FAIL] stats зависает (timeout)")
            return False
        except Exception as e:
            print(f"[ERROR] Ошибка при тестировании stats: {e}")
            return False

def main():
    """Главная функция"""
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
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[CRITICAL ERROR] in test {test_name}: {e}")
            results.append((test_name, False))

    # Выводим итоги
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n[FAIL] Problems detected!")
        print("Possible causes:")
        print("1. Encoding issues with .env file")
        print("2. Subprocess issues (hangs, None in stdout)")
        print("3. Missing or invalid OPENAI_API_KEY")
        print("4. Issues with main.py or its dependencies")
        return False
    else:
        print("\n[SUCCESS] All basic checks passed!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)