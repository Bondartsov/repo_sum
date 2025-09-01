# tests/test_main.py
# Тесты для main.py (CLI-интерфейс)
import pytest
import subprocess
import sys

# Пример smoke-теста для запуска CLI

@pytest.mark.functional
def test_main_cli_help():
    """
    Проверяет, что при запуске main.py с --help выводится справка.
    """
    result = subprocess.run([sys.executable, 'main.py', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'помощь' in result.stdout.lower() or 'help' in result.stdout.lower()
