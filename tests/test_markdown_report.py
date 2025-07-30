import pytest
from doc_generator import DocumentationGenerator
from utils import ParsedFile, FileInfo

def test_markdown_report_sections(tmp_path):
    """
    Проверяет, что Markdown-отчёт содержит ключевые секции и не содержит шаблонных заглушек.
    """
    file_info = FileInfo(
        path="foo.py",
        name="foo.py",
        size=42,
        language="python",
        extension=".py",
        modified_time="2025-01-01T00:00:00",
        encoding="utf-8"
    )
    parsed = ParsedFile(
        file_info=file_info,
        elements=[],
        imports=["os", "sys"],
        global_comments=["Test comment"],
        parse_errors=[]
    )
    doc_gen = DocumentationGenerator()
    md = doc_gen.generate_markdown(parsed)
    # Проверяем наличие ключевых секций
    assert "Импорты" in md
    assert "Комментарии" in md
    # Проверяем отсутствие шаблонных заглушек
    assert "Ошибка анализа" not in md
    assert "{code_content}" not in md
