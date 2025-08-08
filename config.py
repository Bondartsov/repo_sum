"""
Модуль конфигурации для анализатора репозиториев.
"""

import json
import os
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Загружаем переменные из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv не установлен, переменные окружения должны быть установлены системно


@dataclass
class OpenAIConfig:
    """Конфигурация OpenAI API"""
    api_key_env_var: str = "OPENAI_API_KEY"
    max_tokens_per_chunk: int = 4000
    max_response_tokens: int = 5000
    temperature: float = 0.1
    retry_attempts: int = 3
    retry_delay: float = 1.0

    @property
    def api_key(self) -> Optional[str]:
        """Получает API ключ из переменных окружения"""
        return os.getenv(self.api_key_env_var)

    @property
    def model(self) -> str:
        """Получает имя модели из переменных окружения"""
        return os.getenv("OPENAI_MODEL", "gpt-4.1-nano")


@dataclass
class TokenManagementConfig:
    """Конфигурация управления токенами"""
    enable_caching: bool = True
    cache_expiry_days: int = 7


@dataclass
class AnalysisConfig:
    """Конфигурация анализа кода"""
    chunk_strategy: str = "logical"
    min_chunk_size: int = 100
    enable_fallback: bool = True
    languages_priority: List[str] = field(default_factory=lambda: ["python", "javascript", "java"])
    # Новые опции расширенного анализа
    enable_advanced_scoring: bool = False  # приоритизация чанков по «важности»
    sanitize_enabled: bool = False         # санитайзинг секретов перед отправкой в LLM
    sanitize_patterns: List[str] = field(default_factory=list)  # регулярные выражения для вырезания


@dataclass
class FileScannerConfig:
    """Конфигурация сканера файлов"""
    max_file_size: int = 10485760  # 10MB
    excluded_directories: List[str] = field(default_factory=lambda: [
        ".git", ".svn", ".hg",
        "node_modules", "venv", ".venv",
        "__pycache__", ".pytest_cache",
        "target", "build", "dist",
        ".idea", ".vscode",
        "logs", "tmp", "temp"
    ])
    supported_extensions: Dict[str, str] = field(default_factory=lambda: {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
        ".rb": "ruby"
    })


@dataclass
class OutputConfig:
    """Конфигурация вывода"""
    default_output_dir: str = "./docs"
    file_template: str = "minimal_file.md"
    index_template: str = "index_template.md"
    format: str = "markdown"  # markdown|html
    templates_dir: str = "report_templates"


@dataclass
class PromptsConfig:
    """Конфигурация промптов"""
    code_analysis_prompt_file: str = "prompts/code_analysis_prompt.md"


@dataclass
class Config:
    """Основной класс конфигурации"""
    openai: OpenAIConfig
    token_management: TokenManagementConfig
    analysis: AnalysisConfig
    file_scanner: FileScannerConfig
    output: OutputConfig
    prompts: PromptsConfig

    @classmethod
    def load_from_file(cls, config_path: str = "settings.json") -> "Config":
        """Загружает конфигурацию из JSON файла"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            openai=OpenAIConfig(**data.get("openai", {})),
            token_management=TokenManagementConfig(**data.get("token_management", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
            file_scanner=FileScannerConfig(**data.get("file_scanner", {})),
            output=OutputConfig(**data.get("output", {})),
            prompts=PromptsConfig(**data.get("prompts", {}))
        )

    def validate(self, require_api_key: bool = True) -> bool:
        """Валидирует конфигурацию"""
        errors = []
        
        # Валидация OpenAI конфигурации
        if require_api_key and not self.openai.api_key:
            errors.append(f"OpenAI API ключ не найден в переменной окружения {self.openai.api_key_env_var}")
        
        if self.openai.max_tokens_per_chunk <= 0:
            errors.append("max_tokens_per_chunk должно быть положительным числом")
        
        if self.openai.max_response_tokens <= 0:
            errors.append("max_response_tokens должно быть положительным числом")
        
        if not 0 <= self.openai.temperature <= 2:
            errors.append("temperature должна быть в диапазоне 0-2")
        
        if self.openai.retry_attempts < 0:
            errors.append("retry_attempts должно быть неотрицательным числом")
        
        if self.openai.retry_delay < 0:
            errors.append("retry_delay должно быть неотрицательным числом")
        
        # Валидация управления токенами
        if self.token_management.cache_expiry_days <= 0:
            errors.append("cache_expiry_days должно быть положительным числом")
        
        # Валидация анализа
        if self.analysis.chunk_strategy not in ["logical", "size", "lines"]:
            errors.append("chunk_strategy должна быть одной из: logical, size, lines")
        
        if self.analysis.min_chunk_size <= 0:
            errors.append("min_chunk_size должно быть положительным числом")
        
        # Валидация сканера файлов
        if self.file_scanner.max_file_size <= 0:
            errors.append("max_file_size должно быть положительным числом")
        
        if not self.file_scanner.supported_extensions:
            errors.append("supported_extensions не может быть пустым")
        
        # Проверяем что все расширения начинаются с точки
        for ext in self.file_scanner.supported_extensions.keys():
            if not ext.startswith('.'):
                errors.append(f"Расширение файла должно начинаться с точки: {ext}")
        
        # Валидация исключенных директорий
        if not isinstance(self.file_scanner.excluded_directories, list):
            errors.append("excluded_directories должно быть списком")
        
        # Валидация вывода
        if not self.output.default_output_dir.strip():
            errors.append("default_output_dir не может быть пустым")
        
        # Валидация промптов
        if not self.prompts.code_analysis_prompt_file.strip():
            errors.append("code_analysis_prompt_file не может быть пустым")
        
        # Проверяем существование файла промпта
        prompt_path = Path(self.prompts.code_analysis_prompt_file)
        if not prompt_path.exists():
            errors.append(f"Файл промпта не найден: {self.prompts.code_analysis_prompt_file}")
        
        if errors:
            raise ValueError("Ошибки конфигурации:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True


# Глобальный экземпляр конфигурации
_config: Optional[Config] = None


def get_config(require_api_key: bool = False) -> Config:
    """Получает глобальный экземпляр конфигурации"""
    global _config
    if _config is None:
        _config = Config.load_from_file()
        _config.validate(require_api_key=require_api_key)
    return _config


def reload_config(config_path: str = "settings.json", require_api_key: bool = True) -> Config:
    """Перезагружает конфигурацию"""
    global _config
    logger.debug(f"reload_config: старый _config id={id(_config)}")
    _config = Config.load_from_file(config_path)
    logger.debug(f"reload_config: загружен новый _config id={id(_config)}, api_key_length={len(_config.openai.api_key) if _config.openai.api_key else 0}")
    _config.validate(require_api_key=require_api_key)
    return _config
