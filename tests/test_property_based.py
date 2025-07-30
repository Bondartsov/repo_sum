import pytest
from hypothesis import given, strategies as st
from parsers.python_parser import PythonParser
from utils import FileInfo

import tempfile
import os

@given(code=st.text(alphabet=st.characters(blacklist_categories=["Cs", "Cc"]), min_size=1, max_size=200))
def test_python_parser_property_based(code):
    """
    Property-based тест: парсер должен устойчиво обрабатывать любые строки как Python-код.
    Не должно быть необработанных исключений, даже на случайном входе.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "random.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(code)
        file_info = FileInfo(
            path=str(test_file),
            name="random.py",
            size=os.path.getsize(test_file),
            language="python",
            extension=".py",
            modified_time="2025-01-01T00:00:00",
            encoding="utf-8"
        )
        parser = PythonParser()
        try:
            parsed = parser.parse_file(file_info)
            # Проверяем, что возвращается ParsedFile и нет необработанных исключений
            assert hasattr(parsed, "elements")
        except Exception:
            pytest.fail("Parser упал на случайном входе")
