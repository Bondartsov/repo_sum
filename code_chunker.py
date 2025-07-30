"""
Модуль разбивки кода на логические части для анализа OpenAI GPT.
"""

import logging
from typing import List, Optional

import tiktoken

from config import get_config
from utils import ParsedFile, CodeChunk, count_lines_in_text


class CodeChunker:
    """Разбивает код на логические части для анализа GPT"""
    
    def chunk_code(self, file_info, code):
        """
        Совместимость с тестами: разбивает код на чанки по FileInfo и строке кода.
        """
        from utils import ParsedFile, CodeChunk, ParsedElement, count_lines_in_text, get_language_from_extension
        # Простейший парсер: находит функции по def и создает ParsedElement
        import re
        elements = []
        lines = code.splitlines()
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("def "):
                name = re.findall(r"def\s+(\w+)", line)
                if name:
                    elements.append(ParsedElement(
                        name=name[0],
                        type="function",
                        line_number=i,
                        signature=line.strip(),
                        docstring=None
                    ))
        parsed_file = ParsedFile(
            file_info=file_info,
            elements=elements,
            imports=[],
            classes=[],
            functions=[el.name for el in elements],
            comments=[],
            total_lines=len(lines),
            code_lines=len([l for l in lines if l.strip()]),
            comment_lines=0,
            blank_lines=len([l for l in lines if not l.strip()])
        )
        return self.chunk_parsed_file(parsed_file, code)
    
    def __init__(self):
        self.config = get_config()
        self.max_tokens_per_chunk = self.config.openai.max_tokens_per_chunk
        self.min_chunk_size = self.config.analysis.min_chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Инициализируем токенизатор для подсчета токенов
        try:
            # Явно используем cl100k_base для gpt-4.1-nano и подобных моделей
            if "gpt-4.1-nano" in self.config.openai.model or "gpt-4o" in self.config.openai.model:
                self.token_encoder = tiktoken.get_encoding("cl100k_base")
            else:
                self.token_encoder = tiktoken.encoding_for_model(self.config.openai.model)
        except Exception as e:
            self.logger.warning(f"Не удалось инициализировать токенизатор: {e}")
            self.token_encoder = tiktoken.get_encoding("cl100k_base")  # fallback
        
    def chunk_parsed_file(self, parsed_file: ParsedFile, source_code: str = None) -> List[CodeChunk]:
        """Основной метод разбивки файла на части. Если source_code передан — использовать его, иначе читать с диска."""
        chunks = []
        try:
            # Используем переданный source_code, если он есть
            if source_code is None:
                with open(parsed_file.file_info.path, 'r', encoding=parsed_file.file_info.encoding) as f:
                    source_code = f.read()
            # 1. Создаем чанк для импортов и заголовка файла
            header_chunk = self._create_header_chunk(parsed_file, source_code)
            if header_chunk:
                chunks.append(header_chunk)
            # 2. Создаем отдельные чанки для классов
            for element in parsed_file.elements:
                if element.type == 'class':
                    class_chunk = self._create_class_chunk(element, parsed_file, source_code)
                    if class_chunk:
                        chunks.append(class_chunk)
            # 3. Группируем функции в чанки
            function_chunks = self._group_functions_into_chunks(parsed_file, source_code)
            chunks.extend(function_chunks)
            # 4. Создаем чанк для глобальных переменных и констант
            variables_chunk = self._create_variables_chunk(parsed_file, source_code)
            if variables_chunk:
                chunks.append(variables_chunk)
            self.logger.debug(f"Создано {len(chunks)} чанков для {parsed_file.file_info.path}")
            return chunks
        except Exception as e:
            self.logger.error(f"Ошибка при разбивке файла {parsed_file.file_info.path}: {e}")
            # Возвращаем хотя бы один чанк с основной информацией
            return [self._create_fallback_chunk(parsed_file)]
    
    def _create_header_chunk(self, parsed_file: ParsedFile, source_code: str) -> Optional[CodeChunk]:
        """Создает чанк для импортов и комментариев файла"""
        header_content = []
        
        # Добавляем импорты
        if parsed_file.imports:
            header_content.append("# Импорты:")
            header_content.extend(parsed_file.imports)
            header_content.append("")
        
        # Добавляем глобальные комментарии (первые несколько)
        if parsed_file.global_comments:
            header_content.append("# Комментарии:")
            # Берем только первые 3 комментария чтобы не перегружать
            for comment in parsed_file.global_comments[:3]:
                if len(comment) > 10:  # Игнорируем слишком короткие комментарии
                    header_content.append(f"# {comment}")
        
        if not header_content:
            return None
        
        content = "\n".join(header_content)
        tokens = self._count_tokens(content)
        
        # Если чанк слишком большой, обрезаем
        if tokens > self.max_tokens_per_chunk:
            content = self._truncate_content(content, self.max_tokens_per_chunk)
            tokens = self._count_tokens(content)
        
        return CodeChunk(
            name=f"Header of {parsed_file.file_info.name}",
            content=content,
            start_line=1,
            end_line=max(10, len(parsed_file.imports) + len(parsed_file.global_comments[:3])),
            chunk_type="file_header",
            tokens_estimate=tokens
        )
    
    def _create_class_chunk(self, class_element, parsed_file: ParsedFile, source_code: str) -> Optional[CodeChunk]:
        """Создает отдельный чанк для класса"""
        try:
            lines = source_code.splitlines()
            class_start = class_element.line_number - 1  # Преобразуем в 0-индекс
            
            # Ищем конец класса (следующий класс или функция на том же уровне отступов)
            class_end = self._find_class_end(lines, class_start)
            
            # Извлекаем код класса
            class_lines = lines[class_start:class_end + 1]
            class_content = "\n".join(class_lines)
            
            # Добавляем докстринг если есть
            content_parts = []
            if class_element.docstring:
                content_parts.append(f'"""Класс: {class_element.name}')
                content_parts.append(f'Описание: {class_element.docstring}"""')
                content_parts.append("")
            
            content_parts.append(class_content)
            content = "\n".join(content_parts)
            
            tokens = self._count_tokens(content)
            
            # Если класс слишком большой, берем только сигнатуру и докстринг
            if tokens > self.max_tokens_per_chunk:
                content = f"{class_element.signature}\n"
                if class_element.docstring:
                    content += f'    """{class_element.docstring}"""\n'
                content += "    # ... (остальной код класса сокращен)"
                tokens = self._count_tokens(content)
            
            return CodeChunk(
                name=class_element.name,
                content=content,
                start_line=class_element.line_number,
                end_line=class_end + 1,
                chunk_type="class",
                tokens_estimate=tokens
            )
            
        except Exception as e:
            self.logger.warning(f"Ошибка при создании чанка для класса {class_element.name}: {e}")
            return None
    
    def _group_functions_into_chunks(self, parsed_file: ParsedFile, source_code: str) -> List[CodeChunk]:
        """Группирует функции в чанки по лимиту токенов"""
        function_chunks = []
        current_chunk_functions = []
        current_chunk_tokens = 0
        
        # Получаем только функции (не методы классов)
        functions = [elem for elem in parsed_file.elements if elem.type == 'function']
        
        if not functions:
            return []
        
        lines = source_code.splitlines()
        
        for func in functions:
            try:
                # Получаем код функции
                func_start = func.line_number - 1
                func_end = self._find_function_end(lines, func_start)
                func_lines = lines[func_start:func_end + 1]
                func_content = "\n".join(func_lines)
                
                # Подсчитываем токены для функции
                func_tokens = self._count_tokens(func_content)
                
                # Если одна функция больше лимита, создаем отдельный чанк
                if func_tokens > self.max_tokens_per_chunk:
                    # Сначала сохраняем накопленные функции
                    if current_chunk_functions:
                        chunk = self._create_functions_chunk(current_chunk_functions, "functions")
                        if chunk:
                            function_chunks.append(chunk)
                        current_chunk_functions = []
                        current_chunk_tokens = 0
                    
                    # Создаем чанк только с сигнатурой большой функции
                    large_func_chunk = self._create_large_function_chunk(func)
                    if large_func_chunk:
                        function_chunks.append(large_func_chunk)
                    continue
                
                # Если добавление функции превысит лимит, создаем чанк
                if current_chunk_tokens + func_tokens > self.max_tokens_per_chunk and current_chunk_functions:
                    chunk = self._create_functions_chunk(current_chunk_functions, "functions")
                    if chunk:
                        function_chunks.append(chunk)
                    current_chunk_functions = []
                    current_chunk_tokens = 0
                
                # Добавляем функцию к текущему чанку
                current_chunk_functions.append({
                    'element': func,
                    'content': func_content,
                    'start_line': func.line_number,
                    'end_line': func_end + 1,
                    'tokens': func_tokens
                })
                current_chunk_tokens += func_tokens
                
            except Exception as e:
                self.logger.warning(f"Ошибка при обработке функции {func.name}: {e}")
                continue
        
        # Добавляем оставшиеся функции
        if current_chunk_functions:
            chunk = self._create_functions_chunk(current_chunk_functions, "functions")
            if chunk:
                function_chunks.append(chunk)
        
        return function_chunks
    
    def _create_variables_chunk(self, parsed_file: ParsedFile, source_code: str) -> Optional[CodeChunk]:
        """Создает чанк для глобальных переменных и констант"""
        variables = [elem for elem in parsed_file.elements if elem.type in ['variable', 'constant']]
        
        if not variables:
            return None
        
        content_parts = ["# Глобальные переменные и константы:"]
        
        for var in variables[:10]:  # Ограничиваем количество переменных
            if var.signature:
                content_parts.append(var.signature)
        
        content = "\n".join(content_parts)
        tokens = self._count_tokens(content)
        
        return CodeChunk(
            name="Global Variables",
            content=content,
            start_line=min(var.line_number for var in variables),
            end_line=max(var.line_number for var in variables),
            chunk_type="variables",
            tokens_estimate=tokens
        )
    
    def _create_functions_chunk(self, functions_data: List[dict], chunk_type: str) -> Optional[CodeChunk]:
        """Создает чанк из группы функций"""
        if not functions_data:
            return None
        
        content_parts = []
        total_tokens = 0
        min_line = float('inf')
        max_line = 0
        function_names = []
        
        for func_data in functions_data:
            content_parts.append(func_data['content'])
            content_parts.append("")  # Разделитель между функциями
            total_tokens += func_data['tokens']
            min_line = min(min_line, func_data['start_line'])
            max_line = max(max_line, func_data['end_line'])
            function_names.append(func_data['element'].name)
        
        content = "\n".join(content_parts).strip()
        
        return CodeChunk(
            name=f"Functions: {', '.join(function_names[:3])}" + ("..." if len(function_names) > 3 else ""),
            content=content,
            start_line=min_line,
            end_line=max_line,
            chunk_type=chunk_type,
            tokens_estimate=total_tokens
        )
    
    def _create_large_function_chunk(self, func_element) -> Optional[CodeChunk]:
        """Создает чанк для большой функции (только сигнатура и докстринг)"""
        content_parts = [f"# Большая функция: {func_element.name}"]
        
        if func_element.signature:
            content_parts.append(func_element.signature)
        
        if func_element.docstring:
            content_parts.append(f'    """{func_element.docstring}"""')
        
        content_parts.append("    # ... (тело функции сокращено)")
        
        content = "\n".join(content_parts)
        tokens = self._count_tokens(content)
        
        return CodeChunk(
            name=func_element.name,
            content=content,
            start_line=func_element.line_number,
            end_line=func_element.line_number + 10,  # Примерная оценка
            chunk_type="large_function",
            tokens_estimate=tokens
        )
    
    def _create_fallback_chunk(self, parsed_file: ParsedFile) -> CodeChunk:
        """Создает базовый чанк в случае ошибок"""
        content = f"Файл: {parsed_file.file_info.name}\n"
        content += f"Язык: {parsed_file.file_info.language}\n"
        content += f"Размер: {parsed_file.file_info.size} байт\n"
        content += f"Элементов кода: {len(parsed_file.elements)}\n"
        content += f"Импортов: {len(parsed_file.imports)}"
        
        return CodeChunk(
            name=f"Summary of {parsed_file.file_info.name}",
            content=content,
            start_line=1,
            end_line=count_lines_in_text(content)[0],
            chunk_type="file_summary",
            tokens_estimate=self._count_tokens(content)
        )
    
    def _find_class_end(self, lines: List[str], class_start: int) -> int:
        """Находит конец класса по отступам"""
        if class_start >= len(lines):
            return class_start
        
        # Определяем базовый отступ класса
        class_line = lines[class_start]
        base_indent = len(class_line) - len(class_line.lstrip())
        
        # Ищем следующую строку с таким же или меньшим отступом
        for i in range(class_start + 1, len(lines)):
            line = lines[i].rstrip()
            if not line:  # Пустая строка
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and not line.lstrip().startswith(('#', '"', "'")):
                return i - 1
        
        return len(lines) - 1
    
    def _find_function_end(self, lines: List[str], func_start: int) -> int:
        """Находит конец функции по отступам"""
        if func_start >= len(lines):
            return func_start
        
        # Определяем базовый отступ функции
        func_line = lines[func_start]
        base_indent = len(func_line) - len(func_line.lstrip())
        
        # Ищем следующую строку с таким же или меньшим отступом
        for i in range(func_start + 1, len(lines)):
            line = lines[i].rstrip()
            if not line:  # Пустая строка
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and not line.lstrip().startswith(('#', '"', "'")):
                return i - 1
        
        return len(lines) - 1
    
    def _count_tokens(self, text: str) -> int:
        """Подсчитывает количество токенов в тексте"""
        try:
            return len(self.token_encoder.encode(text))
        except Exception:
            # Если токенизатор не работает, используем приблизительную оценку
            return len(text.split()) * 1.3  # Примерно 1.3 токена на слово
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Обрезает содержимое до заданного количества токенов"""
        lines = content.splitlines()
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self._count_tokens(line)
            if current_tokens + line_tokens > max_tokens:
                truncated_lines.append("# ... (содержимое обрезано)")
                break
            truncated_lines.append(line)
            current_tokens += line_tokens
        
        return "\n".join(truncated_lines)
