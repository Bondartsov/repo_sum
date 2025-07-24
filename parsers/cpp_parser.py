"""
Парсер для C++ файлов (.cpp, .hpp, .h, .cc, .cxx).
"""

import re
from typing import List
from .base_parser import BaseParser
from utils import FileInfo, ParsedFile, ParsedElement


class CppParser(BaseParser):
    """Парсер для C++ файлов"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.cpp', '.hpp', '.h', '.cc', '.cxx']
        
        # Регулярные выражения для комментариев в C++
        self.comment_patterns = [
            r'//\s*(.*?)$',  # Однострочные комментарии
            r'/\*\s*(.*?)\s*\*/',  # Многострочные комментарии
        ]
    
    def can_parse(self, file_path: str) -> bool:
        """Проверяет, может ли парсер обработать данный файл"""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)
    
    def parse_file(self, file_info: FileInfo) -> ParsedFile:
        """Основной метод парсинга C++ файла"""
        source_code = self._read_file_content(file_info)
        
        # Извлекаем различные элементы
        imports = self.extract_imports(source_code)
        classes = self.extract_classes(source_code)
        functions = self.extract_functions(source_code)
        comments = self.extract_comments(source_code)
        
        # Объединяем все элементы
        all_elements = classes + functions
        
        return ParsedFile(
            file_info=file_info,
            elements=all_elements,
            imports=imports,
            global_comments=comments,
            parse_errors=[]
        )
    
    def extract_imports(self, source_code: str) -> List[str]:
        """Извлекает #include директивы из C++ кода"""
        includes = []
        
        # Регулярное выражение для #include
        include_pattern = r'#include\s*[<"](.*?)[>"]'
        matches = re.finditer(include_pattern, source_code)
        
        for match in matches:
            include_name = match.group(1).strip()
            if include_name:
                includes.append(include_name)
        
        return includes
    
    def extract_classes(self, source_code: str) -> List[ParsedElement]:
        """Извлекает классы из C++ кода"""
        classes = []
        
        # Регулярное выражение для классов и структур
        class_pattern = r'(?:class|struct)\s+(\w+)(?:\s*:\s*[^{]*)?'
        matches = re.finditer(class_pattern, source_code, re.MULTILINE)
        
        for match in matches:
            class_name = match.group(1)
            line_num = self._get_line_number(source_code, match)
            
            # Пытаемся найти комментарий над классом
            docstring = self._find_preceding_comment(source_code, match.start())
            
            classes.append(ParsedElement(
                name=class_name,
                element_type="class",
                line_number=line_num,
                docstring=docstring,
                signature=match.group(0)
            ))
        
        return classes
    
    def extract_functions(self, source_code: str) -> List[ParsedElement]:
        """Извлекает функции из C++ кода"""
        functions = []
        
        # Регулярное выражение для функций (упрощенное)
        # Ищем паттерн: тип имя_функции(параметры) {
        function_pattern = r'(?:[\w:]+(?:\s*\*|\s*&)?)\s+(\w+)\s*\([^)]*\)(?:\s*const)?\s*(?:{|;)'
        matches = re.finditer(function_pattern, source_code, re.MULTILINE)
        
        for match in matches:
            func_name = match.group(1)
            line_num = self._get_line_number(source_code, match)
            
            # Исключаем ключевые слова C++
            if func_name.lower() in ['if', 'for', 'while', 'switch', 'return', 'class', 'struct']:
                continue
            
            # Пытаемся найти комментарий над функцией
            docstring = self._find_preceding_comment(source_code, match.start())
            
            functions.append(ParsedElement(
                name=func_name,
                element_type="function",
                line_number=line_num,
                docstring=docstring,
                signature=match.group(0).replace('{', '').replace(';', '').strip()
            ))
        
        return functions
    
    def _find_preceding_comment(self, source_code: str, position: int) -> str:
        """Ищет комментарий перед указанной позицией"""
        lines_before = source_code[:position].splitlines()
        comment_lines = []
        
        # Ищем комментарии, идущие непосредственно перед элементом
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