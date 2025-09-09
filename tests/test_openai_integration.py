# tests/test_openai_integration.py
# Тесты для openai_integration.py с моками
import pytest
from unittest.mock import patch, MagicMock
from openai_integration import OpenAIManager
from utils import CodeChunk, GPTAnalysisRequest

# Пример теста для анализа чанка (мокаем OpenAI)

@pytest.mark.integration
def test_analyze_chunk_with_mock():
    """
    Проверяет, что OpenAIManager.analyze_chunk возвращает результат при успешном ответе OpenAI.
    """
    manager = OpenAIManager()
    chunk = CodeChunk(
        name="test.py",
        content="def foo(): pass",
        start_line=1,
        end_line=2,
        chunk_type="function"
    )
    request = GPTAnalysisRequest(
        file_path="test.py",
        language="python",
        chunks=[chunk],
        context="PROMPT"
    )
    with patch("openai_integration.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Ответ"))]
        result = manager.analyze_chunk(request)
        assert result is not None
