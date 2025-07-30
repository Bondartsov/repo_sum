# tests/test_utils.py
# Тесты для модуля utils.py
import pytest
from utils import *

# Пример теста для функции ensure_directory_exists

def test_ensure_directory_exists(tmp_path):
    """
    Проверяет, что функция создаёт директорию, если её не существует.
    """
    test_dir = tmp_path / "new_dir"
    assert not test_dir.exists()
    ensure_directory_exists(str(test_dir))
    assert test_dir.exists()

# Добавьте аналогично тесты для других функций utils.py