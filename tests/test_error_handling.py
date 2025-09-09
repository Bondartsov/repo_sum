import pytest
import os
from openai_integration import OpenAIManager, OpenAIError
from file_scanner import FileScanner
from parsers.base_parser import ParserRegistry
from utils import FileInfo

@pytest.mark.integration
def test_openai_manager_no_api_key(monkeypatch):
    """
    Проверяет, что при отсутствии API-ключа OpenAIManager выбрасывает ошибку.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIManager()

def test_python_parser_syntax_error(tmp_path):
    """
    Проверяет, что PythonParser корректно обрабатывает синтаксическую ошибку.
    """
    code = "def broken_func(:\n    pass"
    test_file = tmp_path / "broken.py"
    test_file.write_text(code, encoding="utf-8")
    file_info = FileInfo(
        path=str(test_file),
        name=test_file.name,
        size=test_file.stat().st_size,
        language="python",
        extension=".py",
        modified_time="2025-01-01T00:00:00",
        encoding="utf-8"
    )
    parser = ParserRegistry().get_parser(file_info.path)
    parsed = parser.parse_file(file_info)
    assert parsed.parse_errors, "Parser должен вернуть ошибку для некорректного кода"

@pytest.mark.integration
def test_openai_manager_network_error(monkeypatch, tmp_path):
    """
    Проверяет, что OpenAIManager корректно обрабатывает сетевую ошибку (эмулируется через mock).
    """
    from utils import GPTAnalysisRequest
    import asyncio
    from file_scanner import FileScanner
    from parsers.base_parser import ParserRegistry
    from code_chunker import CodeChunker

    # Создаём валидный Python-файл
    code = "def foo():\n    pass"
    test_file = tmp_path / "dummy.py"
    test_file.write_text(code, encoding="utf-8")

    # Сканируем и парсим файл
    scanner = FileScanner()
    files = list(scanner.scan_repository(str(tmp_path)))
    file_info = files[0]
    parser = ParserRegistry().get_parser(file_info.path)
    parsed = parser.parse_file(file_info)
    chunker = CodeChunker()
    chunks = chunker.chunk_code(file_info, code)

    async def fake_call_openai_api(*args, **kwargs):
        raise OpenAIError("Network error")

    # Добавляем фиктивное поле output_path для совместимости с кодом, если потребуется
    request = GPTAnalysisRequest(
        file_path=str(test_file),
        language="python",
        chunks=chunks,
        context=""
    )
    setattr(request, "output_path", str(tmp_path))

    manager = OpenAIManager()
    monkeypatch.setattr(manager, "_call_openai_api", fake_call_openai_api)
    result = asyncio.run(manager.analyze_code(request))
    print("DEBUG ERROR:", result.error)
    assert result.error and "Network error" in result.error
