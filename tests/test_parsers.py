# tests/test_parsers.py
# Тесты для всех парсеров из папки parsers
import pytest
from parsers.python_parser import PythonParser
from parsers.cpp_parser import CppParser
from parsers.csharp_parser import CSharpParser
from parsers.typescript_parser import TypeScriptParser
from parsers.javascript_parser import JavaScriptParser
from utils import FileInfo

@pytest.mark.parametrize("parser_cls, code, ext", [
    (PythonParser, "class Foo:\n    pass", ".py"),
    (CppParser, "class Foo { };", ".cpp"),
    (CSharpParser, "public class Foo { }", ".cs"),
    (TypeScriptParser, "class Foo {}", ".ts"),
    (JavaScriptParser, "class Foo {}", ".js"),
])
def test_parser_extracts_classes(parser_cls, code, ext, tmp_path):
    """
    Проверяет, что каждый парсер находит хотя бы один класс в простом коде.
    """
    file_path = tmp_path / ("test" + ext)
    file_path.write_text(code, encoding="utf-8")
    parser = parser_cls()
    file_info = FileInfo(
        path=str(file_path),
        name=file_path.name,
        size=file_path.stat().st_size,
        language=ext.lstrip("."),
        extension=ext,
        modified_time="2025-01-01T00:00:00",
        encoding="utf-8"
    )
    parsed = parser.parse_file(file_info)
    assert any(e.type == "class" for e in parsed.elements)
