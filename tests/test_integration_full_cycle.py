import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from config import get_config
from file_scanner import FileScanner
from parsers.base_parser import ParserRegistry
from code_chunker import CodeChunker
from openai_integration import OpenAIManager
from doc_generator import DocumentationGenerator
from utils import GPTAnalysisRequest

@pytest.mark.asyncio
async def test_full_analysis_cycle(tmp_path):
    """
    Интеграционный тест полного цикла:
    - Сканирование файлов
    - Парсинг кода
    - Разбиение на чанки
    - Анализ чанков через OpenAI (mock)
    - Генерация Markdown-отчёта
    """
    # 1. Создаём тестовый Python-файл
    code = '''
class Foo:
    """Test class"""
    def bar(self):
        pass
    '''
    test_file = tmp_path / "foo.py"
    test_file.write_text(code, encoding="utf-8")

    # 2. Сканируем файлы
    scanner = FileScanner()
    files = list(scanner.scan_repository(str(tmp_path)))
    assert len(files) == 1
    file_info = files[0]

    # 3. Парсим файл
    parser = ParserRegistry().get_parser(file_info.path)
    parsed = parser.parse_file(file_info)
    assert any(e.type == "class" for e in parsed.elements)

    # 4. Разбиваем на чанки
    chunker = CodeChunker()
    chunks = chunker.chunk_code(file_info, code)
    assert len(chunks) >= 1

    # 5. Анализируем чанки через OpenAI (mock)
    request = GPTAnalysisRequest(
        file_path=file_info.path,
        language="python",
        chunks=chunks,
        context=""
    )
    with patch("openai_integration.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Анализ: класс Foo реализует bar"))
        ]
        manager = OpenAIManager()
        result = await manager.analyze_code(request)
        assert "Foo" in result.full_text or "Анализ" in result.full_text

    # 6. Генерируем Markdown-отчёт
    doc_gen = DocumentationGenerator()
    md = doc_gen.generate_markdown(parsed)
    assert "Foo" in md or "class" in md
