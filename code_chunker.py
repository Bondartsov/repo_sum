"""Модуль разбивки кода на логические части для анализа OpenAI GPT."""

import logging
import math
import os
import sys
from typing import List, Optional

import tiktoken

from config import get_config
from utils import ParsedFile, CodeChunk, count_lines_in_text

# Фаза 3: OOM рефакторинг - ограничения на размер чанков
CHUNK_MAX_TOKENS = 768  # Максимальный размер чанка в токенах
CHUNK_MIN_TOKENS = 160  # Минимальный размер для группировки мелких
CHUNK_TARGET_TOKENS = 512  # Целевой размер чанка (оптимальный)


def _is_offline_mode() -> bool:
    """Определяет, нужно ли использовать офлайн-режим (без сетевых запросов)."""

    env_true = {"1", "true", "yes", "on"}

    if str(os.getenv("OFFLINE_MODE", "")).lower() in env_true:
        return True

    if "pytest_socket" in sys.modules:
        return True

    if "--disable-socket" in sys.argv:
        return True

    return False


class CodeChunker:
    """Разбивает код на логические части для анализа GPT"""
    
    def chunk_code(self, file_info, code):
        """
        Совместимость с тестами: разбивает код на чанки по FileInfo и строке кода.
        """
        from utils import ParsedFile, ParsedElement
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
            code_lines=len([line for line in lines if line.strip()]),
            comment_lines=0,
            blank_lines=len([line for line in lines if not line.strip()])
        )
        return self.chunk_parsed_file(parsed_file, code)
    
    def __init__(self):
        self.config = get_config()
        self.min_chunk_size = self.config.analysis.min_chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._offline_mode = _is_offline_mode()

        # Инициализируем токенизатор для подсчета токенов
        if self._offline_mode:
            self.token_encoder = None
            self.logger.info("CodeChunker работает в офлайн-режиме: используется эвристический подсчёт токенов")
        else:
            try:
                # Явно используем cl100k_base для gpt-4.1-nano и подобных моделей
                if "gpt-4.1-nano" in self.config.openai.model or "gpt-4o" in self.config.openai.model:
                    self.token_encoder = tiktoken.get_encoding("cl100k_base")
                else:
                    self.token_encoder = tiktoken.encoding_for_model(self.config.openai.model)
            except Exception as e:
                self.logger.warning(f"Не удалось инициализировать токенизатор: {e}")
                try:
                    self.token_encoder = tiktoken.get_encoding("cl100k_base")  # fallback
                except Exception:
                    self.logger.warning("Фоллбек токенизатора недоступен, переходим на эвристику")
                    self.token_encoder = None
        
    def chunk_parsed_file(self, parsed_file: ParsedFile, source_code: str = None) -> List[CodeChunk]:
        """Основной метод разбивки файла на части. Если source_code передан — использовать его, иначе читать с диска."""
        chunks = []
        try:
            # Используем переданный source_code, если он есть
            if source_code is None:
                with open(parsed_file.file_info.path, 'r', encoding=parsed_file.file_info.encoding) as f:
                    source_code = f.read()
            
            # 1. Создаем чанк для импортов и заголовка файла
            header_chunks = self._create_header_chunk(parsed_file, source_code)
            chunks.extend(header_chunks)
            
            # 2. Создаем отдельные чанки для классов
            for element in parsed_file.elements:
                if element.type == 'class':
                    class_chunks = self._create_class_chunk(element, parsed_file, source_code)
                    chunks.extend(class_chunks)
            
            # 3. Группируем функции в чанки
            function_chunks = self._group_functions_into_chunks(parsed_file, source_code)
            chunks.extend(function_chunks)
            
            # 4. Создаем чанк для глобальных переменных и констант
            variables_chunks = self._create_variables_chunk(parsed_file, source_code)
            chunks.extend(variables_chunks)
            
            # ФАЗА 3: Применяем дробление ко ВСЕМ чанкам ПОСЛЕ их создания
            final_chunks = []
            for chunk in chunks:
                split_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(split_chunks)
            
            self.logger.debug(f"Создано {len(final_chunks)} чанков для {parsed_file.file_info.path}")
            
            # Логируем метрики распределения
            self._log_chunk_metrics(final_chunks)
            
            return final_chunks
        except Exception as e:
            self.logger.error(f"Ошибка при разбивке файла {parsed_file.file_info.path}: {e}")
            # Возвращаем хотя бы один чанк с основной информацией
            return [self._create_fallback_chunk(parsed_file)]
    
    def _create_header_chunk(self, parsed_file: ParsedFile, source_code: str) -> List[CodeChunk]:
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
            return []
        
        content = "\n".join(header_content)
        tokens = self._count_tokens(content)
        
        chunk = CodeChunk(
            name=f"Header of {parsed_file.file_info.name}",
            content=content,
            start_line=1,
            end_line=max(10, len(parsed_file.imports) + len(parsed_file.global_comments[:3])),
            chunk_type="file_header",
            tokens_estimate=tokens
        )
        
        # ОТКАТ ФАЗЫ 3: Возвращаем чанк как есть, дробление будет в chunk_parsed_file
        return [chunk]
    
    def _create_class_chunk(self, class_element, parsed_file: ParsedFile, source_code: str) -> List[CodeChunk]:
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
            
            chunk = CodeChunk(
                name=class_element.name,
                content=content,
                start_line=class_element.line_number,
                end_line=class_end + 1,
                chunk_type="class",
                tokens_estimate=tokens
            )
            
            # ОТКАТ ФАЗЫ 3: Возвращаем чанк как есть
            return [chunk]
            
        except Exception as e:
            self.logger.warning(f"Ошибка при создании чанка для класса {class_element.name}: {e}")
            return []
    
    def _group_functions_into_chunks(self, parsed_file: ParsedFile, source_code: str) -> List[CodeChunk]:
        """Группирует функции в чанки логически"""
        function_chunks = []
        current_chunk_functions = []
        max_functions_per_chunk = 5  # Разумный лимит функций в чанке
        
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
                
                # Если накопилось достаточно функций, создаем чанк
                if len(current_chunk_functions) >= max_functions_per_chunk:
                    chunks = self._create_functions_chunk(current_chunk_functions, "functions")
                    function_chunks.extend(chunks)
                    current_chunk_functions = []
                
                # Добавляем функцию к текущему чанку
                current_chunk_functions.append({
                    'element': func,
                    'content': func_content,
                    'start_line': func.line_number,
                    'end_line': func_end + 1,
                    'tokens': func_tokens
                })
                
            except Exception as e:
                self.logger.warning(f"Ошибка при обработке функции {func.name}: {e}")
                continue
        
        # Добавляем оставшиеся функции
        if current_chunk_functions:
            chunks = self._create_functions_chunk(current_chunk_functions, "functions")
            function_chunks.extend(chunks)
        
        return function_chunks
    
    def _create_variables_chunk(self, parsed_file: ParsedFile, source_code: str) -> List[CodeChunk]:
        """Создает чанк для глобальных переменных и констант"""
        variables = [elem for elem in parsed_file.elements if elem.type in ['variable', 'constant']]
        
        if not variables:
            return []
        
        content_parts = ["# Глобальные переменные и константы:"]
        
        for var in variables[:10]:  # Ограничиваем количество переменных
            if var.signature:
                content_parts.append(var.signature)
        
        content = "\n".join(content_parts)
        tokens = self._count_tokens(content)
        
        chunk = CodeChunk(
            name="Global Variables",
            content=content,
            start_line=min(var.line_number for var in variables),
            end_line=max(var.line_number for var in variables),
            chunk_type="variables",
            tokens_estimate=tokens
        )
        
        # ОТКАТ ФАЗЫ 3: Возвращаем чанк как есть
        return [chunk]
    
    def _create_functions_chunk(self, functions_data: List[dict], chunk_type: str) -> List[CodeChunk]:
        """Создает чанк из группы функций"""
        if not functions_data:
            return []
        
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
        
        chunk = CodeChunk(
            name=f"Functions: {', '.join(function_names[:3])}" + ("..." if len(function_names) > 3 else ""),
            content=content,
            start_line=min_line,
            end_line=max_line,
            chunk_type=chunk_type,
            tokens_estimate=total_tokens
        )
        
        # ОТКАТ ФАЗЫ 3: Возвращаем чанк как есть
        return [chunk]
    
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
        if not text:
            return 0

        if self.token_encoder is None:
            return max(1, int(len(text.split()) * 1.3))

        try:
            return len(self.token_encoder.encode(text))
        except Exception:
            # Если токенизатор не работает, используем приблизительную оценку
            return max(1, int(len(text.split()) * 1.3))  # Примерно 1.3 токена на слово
    
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

    
    def _split_large_chunk(self, chunk: CodeChunk) -> List[CodeChunk]:
        """
        Дробит чанк >CHUNK_MAX_TOKENS на части с сохранением контекста
        
        Args:
            chunk: Большой чанк для дробления
            
        Returns:
            List[CodeChunk]: Список чанков ≤CHUNK_MAX_TOKENS
        """
        token_count = self._count_tokens(chunk.content)
        
        # Если чанк в пределах лимита - вернуть как есть
        if token_count <= CHUNK_MAX_TOKENS:
            return [chunk]
        
        # Чанк слишком большой - нужно дробить
        lines = chunk.content.split('\n')
        total_lines = len(lines)
        
        # Вычислить количество частей
        num_parts = math.ceil(token_count / CHUNK_TARGET_TOKENS)
        lines_per_part = max(10, total_lines // num_parts)  # Минимум 10 строк на часть
        
        parts = []
        for part_idx in range(num_parts):
            start_line = part_idx * lines_per_part
            end_line = min(start_line + lines_per_part, total_lines)
            
            if start_line >= total_lines:
                break
                
            part_lines = lines[start_line:end_line]
            part_content = '\n'.join(part_lines)
            
            # Проверить что часть не пустая
            if not part_content.strip():
                continue
            
            # Создать новый чанк для части
            part_chunk = CodeChunk(
                name=f"{chunk.name} (часть {part_idx + 1}/{num_parts})",
                content=part_content,
                start_line=chunk.start_line + start_line,
                end_line=chunk.start_line + end_line - 1,
                chunk_type=chunk.chunk_type,
                tokens_estimate=self._count_tokens(part_content)
            )
            
            # Добавляем метаданные если chunk имеет атрибут metadata
            if hasattr(chunk, 'metadata') and chunk.metadata:
                part_chunk.metadata = {
                    **chunk.metadata,
                    "part": f"{part_idx + 1}/{num_parts}",
                    "original_chunk": chunk.name,
                    "is_split": True
                }
            
            parts.append(part_chunk)
        
        return parts if parts else [chunk]
    
    def _log_chunk_metrics(self, chunks: List[CodeChunk]) -> None:
        """Логирует метрики распределения размера чанков"""
        if not chunks:
            return
        
        try:
            token_counts = [self._count_tokens(c.content) for c in chunks]
            
            # Вычислить статистики
            p50 = sorted(token_counts)[len(token_counts) // 2]
            p90 = sorted(token_counts)[int(len(token_counts) * 0.9)]
            p99 = sorted(token_counts)[int(len(token_counts) * 0.99)]
            max_tokens = max(token_counts)
            avg_tokens = sum(token_counts) / len(token_counts)
            
            # Подсчитать чанки превышающие лимит
            oversized = sum(1 for t in token_counts if t > CHUNK_MAX_TOKENS)
            
            # КРИТИЧНО: Используем logger вместо print() для VM-совместимости
            # print() может кидать "I/O operation on closed file" если stdout закрыт
            self.logger.info("📊 Метрики чанков:")
            self.logger.info(f"   Всего чанков: {len(chunks)}")
            self.logger.info(f"   Средний размер: {avg_tokens:.0f} токенов")
            self.logger.info(f"   p50: {p50} токенов")
            self.logger.info(f"   p90: {p90} токенов")
            self.logger.info(f"   p99: {p99} токенов (ЛИМИТ: {CHUNK_MAX_TOKENS})")
            self.logger.info(f"   Максимум: {max_tokens} токенов")
            self.logger.info(f"   Превышают лимит: {oversized} чанков")
            
            if p99 > CHUNK_MAX_TOKENS:
                self.logger.warning(f"⚠️ ВНИМАНИЕ: p99 ({p99}) превышает лимит ({CHUNK_MAX_TOKENS})!")
                
        except Exception as e:
            # Не даём метрикам уронить процесс разбивки
            self.logger.debug(f"Не удалось вычислить метрики чанков: {e}")
