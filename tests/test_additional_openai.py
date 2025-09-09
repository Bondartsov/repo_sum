"""
Тесты для интеграции с OpenAI - обработка сетевых ошибок и rate limit.

T-017 - OpenAI: rate limit и повторные запросы
T-018 - OpenAI: офлайн/нет соединения
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock, call
import httpx

from openai import RateLimitError, APIConnectionError, APITimeoutError
from openai_integration import OpenAIManager
from utils import CodeChunk, GPTAnalysisRequest, GPTAnalysisResult


@pytest.fixture
def sample_request():
    """Пример запроса для тестирования"""
    chunk = CodeChunk(
        name="test_function",
        content="def test_func():\n    return True",
        start_line=1,
        end_line=2,
        chunk_type="function"
    )
    return GPTAnalysisRequest(
        file_path="test.py",
        language="python",
        chunks=[chunk],
        context="Test context"
    )


@pytest.fixture
def mock_config():
    """Мокированная конфигурация с настройками retry"""
    config = MagicMock()
    config.openai.api_key = "test_key"
    config.openai.model = "gpt-3.5-turbo"
    config.openai.temperature = 0.7
    config.openai.max_response_tokens = 1000
    config.openai.retry_attempts = 3
    config.openai.retry_delay = 1.0
    config.analysis.sanitize_enabled = False
    config.prompts.code_analysis_prompt_file = "prompts/code_analysis_prompt.md"
    return config


@pytest.mark.integration
class TestOpenAIRateLimit:
    """T-017 - OpenAI: rate limit и повторные запросы"""

    @patch('openai_integration.get_config')
    @patch('openai_integration.load_prompt_from_file')
    @patch('openai_integration.OpenAI')
    def test_rate_limit_with_retries_success(self, mock_openai_class, mock_load_prompt, mock_get_config, sample_request, mock_config):
        """
        Тест успешного восстановления после rate limit ошибок.
        
        Сценарий:
        - Первые 2 попытки возвращают RateLimitError
        - 3-я попытка успешна
        - Проверяем количество повторов и задержки
        """
        # Настройка моков
        mock_get_config.return_value = mock_config
        mock_load_prompt.return_value = "Test prompt template: {filename} {code_content}"
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Первые 2 вызова - rate limit, 3-й успешен
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "🔍 Анализ кода успешен"
        
        # Создаем mock response с status_code и headers для RateLimitError
        mock_response_obj = MagicMock(spec=httpx.Response)
        mock_response_obj.status_code = 429
        mock_response_obj.headers = {"x-request-id": "test-request-id"}
        
        side_effects = [
            RateLimitError("Rate limit exceeded", response=mock_response_obj, body=None),
            RateLimitError("Rate limit exceeded", response=mock_response_obj, body=None),
            mock_response
        ]
        mock_client.chat.completions.create.side_effect = side_effects
        
        # Мокируем asyncio.sleep для проверки задержек
        with patch('asyncio.sleep') as mock_sleep:
            manager = OpenAIManager()
            # Отключаем кэш для этого теста
            with patch.object(manager.cache, 'get_cached_result', return_value=None):
                result = asyncio.run(manager.analyze_code(sample_request))
        
        # Проверяем результат
        assert isinstance(result, GPTAnalysisResult)
        assert result.error is None
        assert "Анализ кода успешен" in result.summary
        
        # Проверяем количество вызовов API (3 попытки)
        assert mock_client.chat.completions.create.call_count == 3
        
        # Проверяем задержки (2 задержки между 3 попытками)
        assert mock_sleep.call_count == 2
        expected_calls = [call(1.0), call(1.0)]  # Фиксированная задержка из конфига
        mock_sleep.assert_has_calls(expected_calls)

    @patch('openai_integration.get_config')
    @patch('openai_integration.load_prompt_from_file') 
    @patch('openai_integration.OpenAI')
    def test_rate_limit_exhausted_retries(self, mock_openai_class, mock_load_prompt, mock_get_config, sample_request, mock_config):
        """
        Тест исчерпания лимита попыток при rate limit.
        
        Сценарий:
        - Все попытки возвращают RateLimitError
        - Проверяем что выбрасывается исходная ошибка
        - Проверяем количество попыток согласно конфигу
        """
        # Настройка моков
        mock_get_config.return_value = mock_config
        mock_load_prompt.return_value = "Test prompt template: {filename} {code_content}"
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Все вызовы возвращают rate limit error
        mock_response_obj = MagicMock(spec=httpx.Response)
        mock_response_obj.status_code = 429
        mock_response_obj.headers = {"x-request-id": "test-request-id"}
        rate_limit_error = RateLimitError("Rate limit exceeded", response=mock_response_obj, body=None)
        mock_client.chat.completions.create.side_effect = rate_limit_error
        
        with patch('asyncio.sleep') as mock_sleep:
            manager = OpenAIManager()
            # Отключаем кэш для этого теста
            with patch.object(manager.cache, 'get_cached_result', return_value=None):
                result = asyncio.run(manager.analyze_code(sample_request))
        
        # Проверяем что возвращается результат с ошибкой (не исключение)
        assert isinstance(result, GPTAnalysisResult)
        assert result.error is not None
        assert "Rate limit exceeded" in result.error
        assert result.summary == "Ошибка анализа"
        
        # Проверяем количество попыток (согласно конфигу retry_attempts = 3)
        assert mock_client.chat.completions.create.call_count == 3
        
        # Проверяем задержки (2 задержки между 3 попытками)
        assert mock_sleep.call_count == 2

    @patch('openai_integration.get_config')
    @patch('openai_integration.load_prompt_from_file')
    @patch('openai_integration.OpenAI')
    def test_rate_limit_retry_configuration(self, mock_openai_class, mock_load_prompt, mock_get_config, sample_request):
        """
        Тест настройки количества повторов и задержек из конфигурации.
        """
        # Конфиг с другими параметрами retry
        config = MagicMock()
        config.openai.api_key = "test_key"
        config.openai.model = "gpt-3.5-turbo" 
        config.openai.temperature = 0.7
        config.openai.max_response_tokens = 1000
        config.openai.retry_attempts = 5  # Больше попыток
        config.openai.retry_delay = 2.0   # Больше задержка
        config.analysis.sanitize_enabled = False
        config.prompts.code_analysis_prompt_file = "prompts/code_analysis_prompt.md"
        
        mock_get_config.return_value = config
        mock_load_prompt.return_value = "Test prompt: {filename} {code_content}"
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Все вызовы возвращают rate limit
        mock_response_obj = MagicMock(spec=httpx.Response)
        mock_response_obj.status_code = 429
        mock_response_obj.headers = {"x-request-id": "test-request-id"}
        rate_limit_error = RateLimitError("Rate limit exceeded", response=mock_response_obj, body=None)
        mock_client.chat.completions.create.side_effect = rate_limit_error
        
        with patch('asyncio.sleep') as mock_sleep:
            manager = OpenAIManager()
            # Отключаем кэш для этого теста
            with patch.object(manager.cache, 'get_cached_result', return_value=None):
                result = asyncio.run(manager.analyze_code(sample_request))
        
        # Проверяем количество попыток согласно новому конфигу
        assert mock_client.chat.completions.create.call_count == 5
        
        # Проверяем задержки с новым значением
        assert mock_sleep.call_count == 4  # 4 задержки между 5 попытками
        expected_calls = [call(2.0)] * 4
        mock_sleep.assert_has_calls(expected_calls)


@pytest.mark.integration
class TestOpenAIConnectionErrors:
    """T-018 - OpenAI: офлайн/нет соединения"""

    @patch('openai_integration.get_config')
    @patch('openai_integration.load_prompt_from_file')
    @patch('openai_integration.OpenAI')
    def test_connection_error_handling(self, mock_openai_class, mock_load_prompt, mock_get_config, sample_request, mock_config):
        """
        Тест обработки ошибок подключения (нет интернета).
        
        Сценарий:
        - API возвращает APIConnectionError
        - Проверяем повторные попытки
        - Проверяем итоговую обработку ошибки
        """
        # Настройка моков
        mock_get_config.return_value = mock_config
        mock_load_prompt.return_value = "Test prompt template: {filename} {code_content}"
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Все вызовы возвращают connection error
        mock_request = MagicMock(spec=httpx.Request)
        connection_error = APIConnectionError(message="Connection failed", request=mock_request)
        mock_client.chat.completions.create.side_effect = connection_error
        
        with patch('asyncio.sleep') as mock_sleep:
            manager = OpenAIManager()
            # Отключаем кэш для этого теста
            with patch.object(manager.cache, 'get_cached_result', return_value=None):
                result = asyncio.run(manager.analyze_code(sample_request))
        
        # Проверяем результат с ошибкой
        assert isinstance(result, GPTAnalysisResult)
        assert result.error is not None
        assert "Connection failed" in result.error
        assert result.summary == "Ошибка анализа"
        
        # Проверяем количество попыток
        assert mock_client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch('openai_integration.get_config')
    @patch('openai_integration.load_prompt_from_file')
    @patch('openai_integration.OpenAI')
    def test_timeout_error_handling(self, mock_openai_class, mock_load_prompt, mock_get_config, sample_request, mock_config):
        """
        Тест обработки таймаутов.
        
        Сценарий:
        - API возвращает APITimeoutError
        - Проверяем повторные попытки и контролируемую обработку
        """
        # Настройка моков
        mock_get_config.return_value = mock_config
        mock_load_prompt.return_value = "Test prompt template: {filename} {code_content}"
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Все вызовы возвращают timeout error
        mock_request = MagicMock(spec=httpx.Request)
        timeout_error = APITimeoutError(request=mock_request)
        mock_client.chat.completions.create.side_effect = timeout_error
        
        with patch('asyncio.sleep') as mock_sleep:
            manager = OpenAIManager()
            # Отключаем кэш для этого теста
            with patch.object(manager.cache, 'get_cached_result', return_value=None):
                result = asyncio.run(manager.analyze_code(sample_request))
        
        # Проверяем результат с ошибкой
        assert isinstance(result, GPTAnalysisResult)
        assert result.error is not None
        assert "Request timed out" in result.error or "timeout" in result.error.lower()
        assert result.summary == "Ошибка анализа"
        
        # Проверяем количество попыток
        assert mock_client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch('openai_integration.get_config')
    @patch('openai_integration.load_prompt_from_file')
    @patch('openai_integration.OpenAI')
    def test_mixed_network_errors_recovery(self, mock_openai_class, mock_load_prompt, mock_get_config, sample_request, mock_config):
        """
        Тест восстановления после смешанных сетевых ошибок.
        
        Сценарий:
        - 1-я попытка: ConnectionError
        - 2-я попытка: TimeoutError  
        - 3-я попытка: успех
        """
        # Настройка моков
        mock_get_config.return_value = mock_config
        mock_load_prompt.return_value = "Test prompt template: {filename} {code_content}"
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Смешанные ошибки, потом успех
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "🔍 Восстановленный анализ"
        
        mock_request = MagicMock(spec=httpx.Request)
        side_effects = [
            APIConnectionError(message="No internet", request=mock_request),
            APITimeoutError(request=mock_request),
            mock_response
        ]
        mock_client.chat.completions.create.side_effect = side_effects
        
        with patch('asyncio.sleep') as mock_sleep:
            manager = OpenAIManager()
            # Отключаем кэш для этого теста
            with patch.object(manager.cache, 'get_cached_result', return_value=None):
                result = asyncio.run(manager.analyze_code(sample_request))
        
        # Проверяем успешный результат
        assert isinstance(result, GPTAnalysisResult)
        assert result.error is None
        assert "Восстановленный анализ" in result.summary
        
        # Проверяем количество вызовов
        assert mock_client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch('openai_integration.get_config')
    @patch('openai_integration.load_prompt_from_file')
    @patch('openai_integration.OpenAI')
    def test_no_infinite_waiting(self, mock_openai_class, mock_load_prompt, mock_get_config, sample_request):
        """
        Тест что нет бесконечных ожиданий при сетевых проблемах.
        
        Проверяем что общее время выполнения ограничено.
        """
        # Конфиг с минимальными настройками для быстрого теста
        config = MagicMock()
        config.openai.api_key = "test_key"
        config.openai.model = "gpt-3.5-turbo"
        config.openai.temperature = 0.7
        config.openai.max_response_tokens = 1000
        config.openai.retry_attempts = 2
        config.openai.retry_delay = 0.1  # Минимальная задержка
        config.analysis.sanitize_enabled = False
        config.prompts.code_analysis_prompt_file = "prompts/code_analysis_prompt.md"
        
        mock_get_config.return_value = config
        mock_load_prompt.return_value = "Test prompt: {filename} {code_content}"
        
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Постоянные connection errors
        mock_request = MagicMock(spec=httpx.Request)
        connection_error = APIConnectionError(message="Network unreachable", request=mock_request)
        mock_client.chat.completions.create.side_effect = connection_error
        
        # Измеряем время выполнения
        start_time = time.time()
        
        with patch('asyncio.sleep') as mock_sleep:
            manager = OpenAIManager()
            # Отключаем кэш для этого теста
            with patch.object(manager.cache, 'get_cached_result', return_value=None):
                result = asyncio.run(manager.analyze_code(sample_request))
        
        execution_time = time.time() - start_time
        
        # Проверяем что выполнение завершилось быстро (без реальных задержек благодаря моку)
        assert execution_time < 1.0  # Должно быть очень быстро с моками
        
        # Проверяем что попытки ограничены
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1
        
        # Проверяем ошибку в результате
        assert result.error is not None
        assert "Network unreachable" in result.error
