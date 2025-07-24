"""
Утилиты и базовые структуры данных.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FileInfo:
    """Информация о файле"""
    path: str
    name: str
    size: int
    language: str
    extension: str
    modified_time: str
    encoding: str = "utf-8"


@dataclass
class CodeChunk:
    """Фрагмент кода для анализа"""
    name: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str = "unknown"  # function, class, module, etc.
    # Для совместимости с CodeChunker
    line_start: int = 0
    line_end: int = 0
    tokens_estimate: int = 0
    element_type: str = ""  # алиас для chunk_type
    
    def __post_init__(self):
        # Синхронизируем поля для совместимости
        if self.line_start == 0 and self.start_line > 0:
            self.line_start = self.start_line
        if self.line_end == 0 and self.end_line > 0:
            self.line_end = self.end_line
        if not self.element_type and self.chunk_type:
            self.element_type = self.chunk_type
        elif self.element_type and not self.chunk_type:
            self.chunk_type = self.element_type


@dataclass
class ParsedElement:
    """Элемент кода (класс, функция, переменная и т.д.)"""
    name: str
    type: str  # "class", "function", "method", "variable", "constant", etc.
    line_number: int
    signature: str = ""
    docstring: Optional[str] = None
    comments: List[str] = field(default_factory=list)


@dataclass
class ParsedFile:
    """Результат парсинга файла"""
    file_info: FileInfo
    chunks: List[CodeChunk] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    # Для совместимости с парсерами
    elements: List[ParsedElement] = field(default_factory=list)
    global_comments: List[str] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)


@dataclass
class GPTAnalysisRequest:
    """Запрос к GPT для анализа"""
    file_path: str
    language: str
    chunks: List[CodeChunk]
    context: str = ""


@dataclass
class GPTAnalysisResult:
    """
    Итог ответа GPT.
    `full_text` — полный отчёт, отображаемый пользователю.
    """
    summary: str
    key_components: List[str]
    analysis_per_chunk: Dict[str, str]
    full_text: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        self.key_components = self.key_components or []
        self.analysis_per_chunk = self.analysis_per_chunk or {}


# Исключения
class RepoSumError(Exception):
    """Базовое исключение для всех ошибок проекта"""
    pass


class FileParsingError(RepoSumError):
    """Ошибка парсинга файла"""
    pass


# Алиас для совместимости с парсерами
ParsingError = FileParsingError


class OpenAIError(RepoSumError):
    """Ошибка взаимодействия с OpenAI"""
    pass


class ConfigError(RepoSumError):
    """Ошибка конфигурации"""
    pass


# Утилитные функции
def ensure_directory_exists(path: str) -> None:
    """Создать директорию если её нет"""
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_filename(filename: str) -> str:
    """Очистить имя файла от недопустимых символов"""
    import re
    # Убираем недопустимые символы для имени файла
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Убираем множественные подчеркивания
    cleaned = re.sub(r'_+', '_', cleaned)
    # Убираем подчеркивания в начале и конце
    cleaned = cleaned.strip('_')
    return cleaned


def format_file_size(size: int) -> str:
    """Форматировать размер файла"""
    if size < 1024:
        return f"{size} bytes"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def get_language_from_extension(extension: str) -> str:
    """Определить язык программирования по расширению файла"""
    language_map = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.kt': 'kotlin',
        '.swift': 'swift',
        '.m': 'objective-c',
        '.sh': 'bash',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        '.xml': 'xml',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.txt': 'text',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp'
    }
    return language_map.get(extension.lower(), 'unknown')


def truncate_text(text: str, max_length: int = 100) -> str:
    """Обрезать текст до указанной длины"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def count_lines(content: str) -> tuple:
    """
    Подсчитать количество строк разных типов
    Возвращает (total_lines, code_lines, comment_lines, blank_lines)
    """
    lines = content.split('\n')
    total_lines = len(lines)
    blank_lines = 0
    comment_lines = 0
    code_lines = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
            comment_lines += 1
        else:
            code_lines += 1
    
    return total_lines, code_lines, comment_lines, blank_lines


# Алиас для совместимости
count_lines_in_text = count_lines


def setup_logging(level: str = "INFO") -> None:
    """Настройка логирования"""
    import logging
    import os
    
    # Создаем директорию для логов если её нет
    os.makedirs('logs', exist_ok=True)
    
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app.log', encoding='utf-8')
        ]
    )
