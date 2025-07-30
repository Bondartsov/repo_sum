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
    temperature: float = 0.1

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


@dataclass
class Config:
    """Основной класс конфигурации"""
    openai: OpenAIConfig
    token_management: TokenManagementConfig
    analysis: AnalysisConfig
    file_scanner: FileScannerConfig
    output: OutputConfig

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
            output=OutputConfig(**data.get("output", {}))
        )

    def validate(self, require_api_key: bool = True) -> bool:
        """Валидирует конфигурацию"""
        if require_api_key and not self.openai.api_key:
            raise ValueError(f"OpenAI API ключ не найден в переменной окружения {self.openai.api_key_env_var}")
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
    logger.debug(f"reload_config: загружен новый _config id={id(_config)}, api_key={_config.openai.api_key}")
    _config.validate(require_api_key=require_api_key)
    return _config
