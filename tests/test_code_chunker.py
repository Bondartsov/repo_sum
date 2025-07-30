# tests/test_code_chunker.py
# Тесты для модуля code_chunker.py
import pytest
from code_chunker import CodeChunker
from utils import FileInfo

# Пример теста для разбиения кода на чанки

def test_chunk_python_code():
    """
    Проверяет, что CodeChunker корректно разбивает Python-код на чанки.
    """
    code = """
def foo():
    pass

def bar():
    pass
"""
    file_info = FileInfo(
        path="test.py",
        name="test.py",
        size=len(code.encode("utf-8")),
        language="python",
        extension=".py",
        modified_time="2025-01-01T00:00:00",
        encoding="utf-8"
    )
    chunker = CodeChunker()
    chunks = chunker.chunk_code(file_info, code)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    # Проверяем, что обе функции присутствуют в чанке(ах)
    all_code = "\n".join(chunk.content for chunk in chunks)
    assert "def foo()" in all_code
    assert "def bar()" in all_code
