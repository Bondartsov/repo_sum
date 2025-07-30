"""
Парсер для C# файлов (.cs).
"""

import re
from typing import List
from .base_parser import BaseParser
from utils import FileInfo, ParsedFile, ParsedElement

class CSharpParser(BaseParser):
    """
    Парсер для C# файлов. Извлекает классы, функции, импорты и комментарии.
    """
    def __init__(self):
        super().__init__()
        # Поддерживаемые расширения для C#
        self.supported_extensions = ['.cs']
        # Регулярные выражения для комментариев в C#
        self.comment_patterns = [
            r'//\s*(.*?)$',  # Однострочные комментарии
            r'/\*\s*(.*?)\s*\*/',  # Многострочные комментарии
        ]

    def can_parse(self, file_path: str) -> bool:
        """
        Проверяет, может ли парсер обработать данный файл по расширению.
        """
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)

    def parse_file(self, file_info: FileInfo) -> ParsedFile:
        """
        Основной метод парсинга C# файла.
        """
        source_code = self._read_file_content(file_info)
        imports = self.extract_imports(source_code)
        classes = self.extract_classes(source_code)
        functions = self.extract_functions(source_code)
        comments = self.extract_comments(source_code)
        all_elements = classes + functions
        return ParsedFile(
            file_info=file_info,
            elements=all_elements,
            imports=imports,
            global_comments=comments,
            parse_errors=[]
        )

    def extract_imports(self, source_code: str) -> List[str]:
        """
        Извлекает директивы using из C# кода.
        """
        imports = []
        pattern = r'^using\s+([\w\.]+);'
        matches = re.finditer(pattern, source_code, re.MULTILINE)
        for match in matches:
            import_name = match.group(1).strip()
            if import_name:
                imports.append(import_name)
        return imports

    def extract_classes(self, source_code: str) -> List[ParsedElement]:
        """
        Извлекает классы из C# кода.
        """
        classes = []
        # Ищем определения классов и структур
        class_pattern = r'(?:class|struct)\s+(\w+)(?:\s*:\s*[^{]*)?'
        matches = re.finditer(class_pattern, source_code, re.MULTILINE)
        for match in matches:
            class_name = match.group(1)
            line_num = self._get_line_number(source_code, match)
            docstring = self._find_preceding_comment(source_code, match.start())
            classes.append(ParsedElement(
                name=class_name,
                type="class",
                line_number=line_num,
                docstring=docstring,
                signature=match.group(0)
            ))
        return classes

    def extract_functions(self, source_code: str) -> List[ParsedElement]:
        """
        Извлекает методы и функции из C# кода.
        """
        functions = []
        # Упрощённое регулярное выражение для поиска методов
        function_pattern = r'(?:public|private|protected|internal|static|virtual|override|async|sealed|extern|unsafe|new|partial|abstract|\s)+([\w<>\[\]]+)\s+(\w+)\s*\([^)]*\)\s*\{'
        matches = re.finditer(function_pattern, source_code, re.MULTILINE)
        for match in matches:
            return_type = match.group(1)
            func_name = match.group(2)
            line_num = self._get_line_number(source_code, match)
            docstring = self._find_preceding_comment(source_code, match.start())
            signature = match.group(0).replace('{', '').strip()
            functions.append(ParsedElement(
                name=func_name,
                type="function",
                line_number=line_num,
                docstring=docstring,
                signature=signature
            ))
        return functions

    def _find_preceding_comment(self, source_code: str, position: int) -> str:
        """
        Ищет комментарий непосредственно перед указанной позицией (для классов/методов).
        """
        lines_before = source_code[:position].splitlines()
        comment_lines = []
        for line in reversed(lines_before[-5:]):  # Проверяем последние 5 строк
            line = line.strip()
            if line.startswith('//'):
                comment_lines.insert(0, line[2:].strip())
            elif line.startswith('/*') and line.endswith('*/'):
                comment_text = line[2:-2].strip()
                comment_lines.insert(0, comment_text)
            elif line == '':
                continue  # Пропускаем пустые строки
            else:
                break  # Если встретили не комментарий, прерываем
        return ' '.join(comment_lines) if comment_lines else ""
