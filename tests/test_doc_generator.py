# tests/test_doc_generator.py
# Тесты для модуля doc_generator.py
import pytest
from doc_generator import DocumentationGenerator
from utils import ParsedFile, FileInfo

# Пример теста для генерации документации

def test_generate_markdown():
    """
    Проверяет, что DocumentationGenerator создаёт Markdown-документ.
    """
    parsed_file = ParsedFile(
        file_info=FileInfo(
            path="test.py",
            name="test.py",
            size=0,
            language="python",
            extension=".py",
            modified_time="2025-01-01T00:00:00",
            encoding="utf-8"
        ),
        elements=[],
        imports=["os"],
        global_comments=["Test comment"],
        parse_errors=[]
    )
    doc_gen = DocumentationGenerator()
    md = doc_gen.generate_markdown(parsed_file)
    assert isinstance(md, str)
    assert "os" in md
    assert "Test comment" in md
