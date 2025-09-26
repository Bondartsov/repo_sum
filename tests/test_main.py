# tests/test_main.py
# Тесты для main.py (CLI-интерфейс)
import pytest
import sys

from tests.utils_cli import run_cli

# Пример smoke-теста для запуска CLI

@pytest.mark.functional
def test_main_cli_help():
    """
    Проверяет, что при запуске main.py с --help выводится справка.
    """
    result = run_cli(["--help"], use_test_config=False)
    assert result.returncode == 0
    assert 'помощь' in result.stdout.lower() or 'help' in result.stdout.lower()
