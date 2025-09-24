"""
Диагностика проблем с кодировкой в тестах.
Проверяет основные источники UnicodeDecodeError и проблем с proc.stdout.
"""

import os
import sys
import subprocess
import tempfile
import locale
from pathlib import Path

def diagnose_env_file_encoding():
    """Диагностика кодировки .env файла"""
    print("=" * 60)
    print("🔍 ДИАГНОСТИКА КОДИРОВКИ .ENV ФАЙЛА")
    print("=" * 60)

    env_file = Path(__file__).parent.parent / '.env'
    if not env_file.exists():
        print("❌ .env файл не найден")
        return False

    print(f"📁 Путь к .env файлу: {env_file}")

    # Проверяем системную кодировку
    print(f"🖥️ Системная кодировка: {locale.getpreferredencoding()}")
    print(f"🖥️ Кодировка stdout: {sys.stdout.encoding}")
    print(f"🖥️ Кодировка stderr: {sys.stderr.encoding}")

    # Читаем файл в бинарном режиме
    try:
        with open(env_file, 'rb') as f:
            raw_content = f.read()
        print(f"📏 Размер файла: {len(raw_content)} байт")

        # Пробуем разные кодировки
        encodings_to_try = ['utf-8', 'cp1251', 'latin1', 'ascii']

        for encoding in encodings_to_try:
            try:
                decoded = raw_content.decode(encoding)
                print(f"✅ Успешно декодирован как {encoding}")

                # Проверяем наличие русских символов
                has_cyrillic = any(ord(c) > 127 for c in decoded)
                print(f"   📝 Содержит кириллицу: {has_cyrillic}")

                # Проверяем наличие OPENAI_API_KEY
                has_api_key = 'OPENAI_API_KEY' in decoded
                print(f"   🔑 Содержит OPENAI_API_KEY: {has_api_key}")

                if has_cyrillic:
                    print(f"⚠️ ВНИМАНИЕ: Найдены русские символы в {encoding} кодировке!")
                    return False

            except UnicodeDecodeError as e:
                print(f"❌ Ошибка декодирования как {encoding}: {e}")
                continue

        return True

    except Exception as e:
        print(f"❌ Критическая ошибка чтения .env файла: {e}")
        return False

def diagnose_subprocess_encoding():
    """Диагностика проблем с кодировкой в subprocess"""
    print("\n" + "=" * 60)
    print("🔍 ДИАГНОСТИКА SUBPROCESS КОДИРОВКИ")
    print("=" * 60)

    main_py = Path(__file__).parent.parent / 'main.py'
    if not main_py.exists():
        print("❌ main.py не найден")
        return False

    print(f"📁 Путь к main.py: {main_py}")

    # Создаем тестовый репозиторий
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        (repo_path / "test.py").write_text('print("Hello, мир!")')  # Содержит русский текст

        print(f"📁 Тестовый репозиторий: {repo_path}")

        # Тест 1: capture_output=True, text=True (текущий способ)
        print("\n🧪 ТЕСТ 1: capture_output=True, text=True")
        try:
            result = subprocess.run(
                [sys.executable, str(main_py), 'stats', str(repo_path)],
                capture_output=True,
                text=True,
                timeout=10
            )

            print(f"   ✅ returncode: {result.returncode}")
            print(f"   📝 stdout type: {type(result.stdout)}")
            print(f"   📝 stderr type: {type(result.stderr)}")

            if result.stdout:
                print(f"   📝 stdout length: {len(result.stdout)}")
                print(f"   📝 stdout preview: {repr(result.stdout[:100])}")
            else:
                print("   ⚠️ stdout is None!")

            if result.stderr:
                print(f"   📝 stderr length: {len(result.stderr)}")
                print(f"   📝 stderr preview: {repr(result.stderr[:100])}")
            else:
                print("   ⚠️ stderr is None!")

        except UnicodeDecodeError as e:
            print(f"   ❌ UnicodeDecodeError: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Другая ошибка: {e}")

        # Тест 2: capture_output=True, text=True, encoding='utf-8'
        print("\n🧪 ТЕСТ 2: capture_output=True, text=True, encoding='utf-8'")
        try:
            result = subprocess.run(
                [sys.executable, str(main_py), 'stats', str(repo_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10
            )

            print(f"   ✅ returncode: {result.returncode}")
            print(f"   📝 stdout type: {type(result.stdout)}")
            print(f"   📝 stderr type: {type(result.stderr)}")

            if result.stdout:
                print(f"   📝 stdout length: {len(result.stdout)}")
                print(f"   📝 stdout preview: {repr(result.stdout[:100])}")
            else:
                print("   ⚠️ stdout is None!")

        except UnicodeDecodeError as e:
            print(f"   ❌ UnicodeDecodeError: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Другая ошибка: {e}")

        # Тест 3: capture_output=True, text=True, encoding='utf-8', errors='replace'
        print("\n🧪 ТЕСТ 3: capture_output=True, text=True, encoding='utf-8', errors='replace'")
        try:
            result = subprocess.run(
                [sys.executable, str(main_py), 'stats', str(repo_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )

            print(f"   ✅ returncode: {result.returncode}")
            print(f"   📝 stdout type: {type(result.stdout)}")
            print(f"   📝 stderr type: {type(result.stderr)}")

            if result.stdout:
                print(f"   📝 stdout length: {len(result.stdout)}")
                print(f"   📝 stdout preview: {repr(result.stdout[:100])}")
            else:
                print("   ⚠️ stdout is None!")

        except UnicodeDecodeError as e:
            print(f"   ❌ UnicodeDecodeError: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Другая ошибка: {e}")

    return True

def diagnose_rich_console_encoding():
    """Диагностика проблем с Rich Console"""
    print("\n" + "=" * 60)
    print("🔍 ДИАГНОСТИКА RICH CONSOLE КОДИРОВКИ")
    print("=" * 60)

    try:
        from rich.console import Console
        console = Console()

        # Тестируем вывод русских символов
        test_text = "Привет мир! Hello world! 🚀"
        print(f"🧪 Тестируем вывод: {test_text}")

        try:
            console.print(f"[bold blue]{test_text}[/bold blue]")
            print("✅ Rich Console работает корректно")
            return True
        except Exception as e:
            print(f"❌ Ошибка Rich Console: {e}")
            return False

    except ImportError:
        print("❌ Rich не установлен")
        return False

def main():
    """Главная диагностическая функция"""
    print("🚀 ДИАГНОСТИКА ПРОБЛЕМ С КОДИРОВКОЙ")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"CWD: {os.getcwd()}")

    results = []

    # Запускаем диагностику
    results.append(("Кодировка .env файла", diagnose_env_file_encoding()))
    results.append(("Кодировка subprocess", diagnose_subprocess_encoding()))
    results.append(("Кодировка Rich Console", diagnose_rich_console_encoding()))

    # Выводим итоги
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
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
        print("\n❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ С КОДИРОВКОЙ!")
        print("\nВозможные причины:")
        print("1. Проблемы с кодировкой .env файла (русские комментарии)")
        print("2. Проблемы с subprocess (UnicodeDecodeError)")
        print("3. Проблемы с Rich Console")
        print("4. Несоответствие кодировок stdout/stderr")
        return False
    else:
        print("\n✅ Все проверки кодировки прошли!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)