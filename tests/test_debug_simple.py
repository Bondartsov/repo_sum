"""
Упрощенная диагностика падающих тестов.
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
        print(f"✅ .env файл найден: {env_file}")
        try:
            with open(env_file, 'rb') as f:
                content = f.read()
                print(f"Размер файла: {len(content)} байт")

            for encoding in ['utf-8', 'cp1251', 'latin1']:
                try:
                    decoded = content.decode(encoding)
                    print(f"✅ Успешно декодирован как {encoding}")
                    if 'OPENAI_API_KEY' in decoded:
                        print("✅ Найден OPENAI_API_KEY в .env файле!")
                        return True
                    print("❌ OPENAI_API_KEY НЕ найден в .env файле")
                    return False
                except UnicodeDecodeError:
                    print(f"❌ Не удалось декодировать как {encoding}")
                    continue

            print("❌ Не удалось декодировать .env файл ни в одной кодировке")
            return False
        except Exception as exc:
            print(f"❌ Ошибка чтения .env файла: {exc}")
            return False
    else:
        print("❌ .env файл НЕ найден")
        return False

def _run_cli(args, *, timeout=10):
    """Запускает subprocess с безопасной UTF-8 декодировкой."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=timeout
    )

def test_subprocess_basic():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА SUBPROCESS")
    print("=" * 50)

    try:
        result = _run_cli(['python', '--version'])
    except Exception as exc:
        pytest.fail(f"❌ Ошибка subprocess: {exc}")

    print(f"✅ subprocess работает: {result.stdout.strip()}")
    assert result.returncode == 0, 'Команда python --version завершилась с ошибкой'
    assert result.stdout.strip(), 'Команда python --version не вернула вывод'

def test_main_py_exists():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА MAIN.PY")
    print("=" * 50)

    main_py = Path("d:/Scripts_Python/repo_sum/main.py")
    if main_py.exists():
        print(f"✅ main.py найден: {main_py}")
    else:
        pytest.fail(f"❌ main.py НЕ найден: {main_py}")

def test_openai_api_key_in_env():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА OPENAI_API_KEY В ОКРУЖЕНИИ")
    print("=" * 50)

    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OPENAI_API_KEY найден в окружении: {api_key[:10]}...")
    else:
        pytest.fail("❌ OPENAI_API_KEY НЕ найден в окружении")

def test_cli_help():
    print("\n" + "=" * 50)
    print("ПРОВЕРКА CLI --HELP")
    print("=" * 50)

    main_py = "d:/Scripts_Python/repo_sum/main.py"

    try:
        result = _run_cli(['python', main_py, '--help'])
    except Exception as exc:
        pytest.fail(f"❌ Ошибка при тестировании --help: {exc}")

    print(f"returncode: {result.returncode}")
    print(f"stdout: {repr(result.stdout)}")
    print(f"stderr: {repr(result.stderr)}")

    assert result.returncode == 0, 'Команда --help завершилась с ошибкой'
    assert 'Options' in result.stdout, 'Вывод --help не содержит блока Options'
    print("✅ --help работает корректно")

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
            pytest.fail("❌ stats зависает (timeout)")
        except Exception as exc:
            pytest.fail(f"❌ Ошибка при тестировании stats: {exc}")

        print(f"returncode: {result.returncode}")
        print(f"stdout: {repr(result.stdout)}")
        print(f"stderr: {repr(result.stderr)}")

        assert result.returncode == 0, 'Команда stats завершилась с ошибкой'
        assert 'Общая статистика' in result.stdout, 'Вывод stats не содержит ожидаемого текста'
        print("✅ stats работает корректно")

def main():
    print("🔍 ДИАГНОСТИКА ПАДАЮЩИХ ТЕСТОВ")
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
        print(f"\n🧪 {test_name}...")
        try:
            test_func()
            results.append((test_name, True))
        except AssertionError as err:
            print(f"❌ Проверка не пройдена: {err}")
            results.append((test_name, False))
        except Exception as err:
            print(f"❌ Критическая ошибка в тесте {test_name}: {err}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
    print("=" * 50)

    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed

    for test_name, ok in results:
        status = "✅ ПРОШЕЛ" if ok else "❌ ПАДАЕТ"
        print(f"{test_name}: {status}")

    print(f"\nИтого: {passed} прошли, {failed} падают")

    if failed > 0:
        print("\n❌ Обнаружены проблемы!")
        print("Возможные причины:")
        print("1. Проблемы с кодировкой .env файла")
        print("2. Проблемы с subprocess (зависания, None в stdout)")
        print("3. Отсутствие или некорректный OPENAI_API_KEY")
        print("4. Проблемы с main.py или его зависимостями")
        return False

    print("\n✅ Все базовые проверки прошли!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
