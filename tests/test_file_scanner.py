# tests/test_file_scanner.py
# Тесты для модуля file_scanner.py
import pytest
from file_scanner import FileScanner

# Пример теста для сканирования файлов

@pytest.mark.integration
def test_scan_python_files(tmp_path):
    """
    Проверяет, что FileScanner находит Python-файлы в директории.
    """
    # Создаём временные файлы
    (tmp_path / "a.py").write_text("print('hello')")
    (tmp_path / "b.txt").write_text("text")
    scanner = FileScanner()
    files = list(scanner.scan_repository(str(tmp_path)))
    assert any(f.path.endswith('.py') for f in files)
    assert not any(f.path.endswith('.txt') for f in files)
