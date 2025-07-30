"""
Базовый абстрактный класс для всех парсеров кода.
"""

import re
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from utils import FileInfo, ParsedFile, ParsedElement, ParsingError


class BaseParser(ABC):
    """Абстрактный базовый класс для всех парсеров"""
    
    def __init__(self):
        self.comment_patterns = []  # Регулярные выражения для комментариев
        self.supported_extensions = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Проверяет, может ли парсер обработать данный файл"""
        pass
        
    @abstractmethod
    def parse_file(self, file_info: FileInfo) -> ParsedFile:
        """Основной метод парсинга файла"""
        pass
        
    @abstractmethod
    def extract_imports(self, source_code: str) -> List[str]:
        """Извлекает импорты из кода"""
        pass
        
    @abstractmethod
    def extract_classes(self, source_code: str) -> List[ParsedElement]:
        """Извлекает классы из кода"""
        pass
        
    @abstractmethod
    def extract_functions(self, source_code: str) -> List[ParsedElement]:
        """Извлекает функции из кода"""
        pass
        
    def extract_comments(self, source_code: str) -> List[str]:
        """Общий метод извлечения комментариев через регулярные выражения"""
        comments = []
        for pattern in self.comment_patterns:
            matches = re.finditer(pattern, source_code, re.MULTILINE | re.DOTALL)
            for match in matches:
                comment_text = match.group(1).strip()
                if comment_text and len(comment_text) > 2:  # Игнорируем слишком короткие комментарии
                    comments.append(comment_text)
        return comments
        
    def safe_parse(self, file_info: FileInfo) -> ParsedFile:
        """Безопасный парсинг с обработкой ошибок"""
        try:
            return self.parse_file(file_info)
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге файла {file_info.path}: {e}")
            return ParsedFile(
                file_info=file_info,
                elements=[],
                imports=[],
                global_comments=[],
                parse_errors=[str(e)]
            )
    
    def _read_file_content(self, file_info: FileInfo) -> str:
        """Читает содержимое файла с обработкой кодировки"""
        try:
            with open(file_info.path, 'r', encoding=file_info.encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Пытаемся с другими кодировками
            for encoding in ['utf-8', 'latin-1', 'cp1251']:
                try:
                    with open(file_info.path, 'r', encoding=encoding) as f:
                        content = f.read()
                        self.logger.warning(f"Использована кодировка {encoding} для {file_info.path}")
                        return content
                except UnicodeDecodeError:
                    continue
            raise ParsingError(f"Не удалось прочитать файл {file_info.path} с поддерживаемыми кодировками")
    
    def _get_line_number(self, source_code: str, match_obj) -> int:
        """Получает номер строки для найденного совпадения"""
        if match_obj is None:
            return 1
        return source_code[:match_obj.start()].count('\n') + 1
    
    def _clean_docstring(self, docstring: str) -> str:
        """Очищает докстринг от лишних символов"""
        if not docstring:
            return ""
        
        # Удаляем тройные кавычки
        docstring = re.sub(r'^[\'\"]{3}|[\'\"]{3}$', '', docstring.strip())
        
        # Удаляем лишние пробелы и переносы строк
        lines = docstring.splitlines()
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        return ' '.join(cleaned_lines)
    
    def _extract_function_signature(self, source_code: str, function_name: str) -> Optional[str]:
        """Извлекает сигнатуру функции (базовая реализация)"""
        # Простая регулярка для поиска функции
        pattern = rf'def\s+{re.escape(function_name)}\s*\([^)]*\)[^:]*:'
        match = re.search(pattern, source_code)
        if match:
            return match.group(0).replace(':', '').strip()
        return None


class ParserRegistry:
    """Централизованный реестр всех парсеров"""
    
    def __init__(self):
        self.parsers = []
        self._initialize_parsers()
        
    def _initialize_parsers(self):
        """Инициализирует доступные парсеры"""
        try:
            from .python_parser import PythonParser
            self.parsers.append(PythonParser())
        except ImportError as e:
            logging.warning(f"Не удалось загрузить Python парсер: {e}")

        try:
            from .cpp_parser import CppParser
            self.parsers.append(CppParser())
        except ImportError as e:
            logging.warning(f"Не удалось загрузить C++ парсер: {e}")

        # --- Новые парсеры ---
        try:
            from .csharp_parser import CSharpParser  # Парсер для C#
            self.parsers.append(CSharpParser())
        except ImportError as e:
            logging.warning(f"Не удалось загрузить C# парсер: {e}")

        try:
            from .typescript_parser import TypeScriptParser  # Парсер для TypeScript
            self.parsers.append(TypeScriptParser())
        except ImportError as e:
            logging.warning(f"Не удалось загрузить TypeScript парсер: {e}")

        try:
            from .javascript_parser import JavaScriptParser  # Парсер для JavaScript
            self.parsers.append(JavaScriptParser())
        except ImportError as e:
            logging.warning(f"Не удалось загрузить JavaScript парсер: {e}")
        
    def get_parser(self, file_path: str) -> Optional[BaseParser]:
        """Возвращает подходящий парсер для файла"""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
        
    def get_supported_languages(self) -> List[str]:
        """Возвращает список поддерживаемых языков"""
        languages = set()
        for parser in self.parsers:
            languages.update(parser.supported_extensions)
        return list(languages)
    
    def get_parser_count(self) -> int:
        """Возвращает количество загруженных парсеров"""
        return len(self.parsers)