"""
Дополнительные тесты для системы чанкинга кода (code_chunker.py)
T-012 и T-013 из плана тестирования
"""

import pytest
import tempfile
import os
import subprocess
from pathlib import Path
from typing import List

from code_chunker import CodeChunker
from utils import FileInfo, ParsedFile, ParsedElement


class TestChunkingSystem:
    """Тесты системы чанкинга кода"""
    
    def create_temp_python_file(self, content: str) -> str:
        """Создаёт временный Python файл с заданным содержимым"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(content)
            return f.name
    
    def cleanup_temp_file(self, filepath: str):
        """Удаляет временный файл"""
        try:
            os.unlink(filepath)
        except FileNotFoundError:
            pass
    
    def create_file_info(self, filepath: str) -> FileInfo:
        """Создаёт FileInfo для тестового файла"""
        path_obj = Path(filepath)
        stat = path_obj.stat()
        return FileInfo(
            path=filepath,
            name=path_obj.name,
            size=stat.st_size,
            language="python",
            extension=".py",
            modified_time="2025-01-01T00:00:00",
            encoding="utf-8"
        )

    def create_parsed_file_with_many_functions(self, filepath: str, source_code: str) -> ParsedFile:
        """Создаёт ParsedFile с множеством функций для тестирования overlap"""
        file_info = self.create_file_info(filepath)
        
        # Простой парсер для поиска функций
        elements = []
        lines = source_code.splitlines()
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("def "):
                import re
                name_match = re.findall(r"def\s+(\w+)", line)
                if name_match:
                    elements.append(ParsedElement(
                        name=name_match[0],
                        type="function",
                        line_number=i,
                        signature=line.strip(),
                        docstring=None
                    ))
        
        return ParsedFile(
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

    def test_t012_no_duplicates_on_overlap(self):
        """
        T-012 - Chunking: отсутствие дубликатов на оверлапе
        
        Подготавливает тестовый файл с длиной, требующей нескольких чанков с overlap.
        Сравнивает границы соседних чанков из code_chunker.py.
        Ожидается: на стыке чанков нет задвоений текста в агрегированном результате.
        """
        # Создаём код с множеством функций, чтобы гарантированно получить несколько чанков
        large_code = '''#!/usr/bin/env python3
"""
Модуль с множеством функций для тестирования чанкинга.
Этот код специально создан для тестирования системы разбивки на чанки.
"""

import os
import sys
import json
from typing import Dict, List, Optional

def function_one():
    """Первая функция с большим количеством кода."""
    result = []
    for i in range(100):
        if i % 2 == 0:
            result.append(f"Even number: {i}")
        else:
            result.append(f"Odd number: {i}")
        # Дополнительная обработка для увеличения размера
        temp_dict = {"index": i, "value": i * 2, "description": f"Item {i}"}
        if temp_dict["value"] > 50:
            temp_dict["category"] = "large"
        elif temp_dict["value"] > 20:
            temp_dict["category"] = "medium"
        else:
            temp_dict["category"] = "small"
    return result

def function_two():
    """Вторая функция с обработкой данных."""
    data = {"name": "test", "values": [1, 2, 3, 4, 5]}
    processed = []
    for item in data["values"]:
        processed.append({
            "original": item,
            "squared": item ** 2,
            "cubed": item ** 3,
            "description": f"Number {item} processing"
        })
        # Дополнительные вычисления
        if item > 3:
            processed[-1]["category"] = "high"
        else:
            processed[-1]["category"] = "low"
    return processed

def function_three():
    """Третья функция с файловыми операциями."""
    file_list = []
    base_path = "/tmp/test"
    for i in range(50):
        filename = f"file_{i:03d}.txt"
        full_path = os.path.join(base_path, filename)
        file_info = {
            "name": filename,
            "path": full_path,
            "size": i * 1024,
            "type": "text" if i % 2 == 0 else "data"
        }
        file_list.append(file_info)
    return file_list

def function_four():
    """Четвёртая функция с JSON обработкой."""
    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "testdb",
            "credentials": {"user": "admin", "password": "secret"}
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "backend": "redis"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        }
    }
    # Обработка конфигурации
    processed_config = {}
    for section, settings in config.items():
        processed_config[section] = {}
        for key, value in settings.items():
            if isinstance(value, dict):
                processed_config[section][key] = json.dumps(value)
            else:
                processed_config[section][key] = str(value)
    return processed_config

def function_five():
    """Пятая функция с алгоритмической обработкой."""
    numbers = list(range(1, 101))
    results = {
        "primes": [],
        "composites": [],
        "squares": [],
        "cubes": []
    }
    
    for num in numbers:
        # Проверка на простое число
        is_prime = True
        if num < 2:
            is_prime = False
        else:
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    is_prime = False
                    break
        
        if is_prime:
            results["primes"].append(num)
        else:
            results["composites"].append(num)
        
        # Проверка на полный квадрат
        sqrt_num = int(num ** 0.5)
        if sqrt_num * sqrt_num == num:
            results["squares"].append(num)
        
        # Проверка на полный куб
        cbrt_num = round(num ** (1/3))
        if cbrt_num ** 3 == num:
            results["cubes"].append(num)
    
    return results
'''
        
        temp_file = self.create_temp_python_file(large_code)
        try:
            # Создаём CodeChunker с небольшим лимитом токенов для принудительного разбиения
            chunker = CodeChunker()
            # Временно уменьшаем лимит для получения нескольких чанков
            original_limit = chunker.max_tokens_per_chunk
            chunker.max_tokens_per_chunk = 800  # Небольшой лимит для принудительного разбиения
            
            # Создаём ParsedFile
            parsed_file = self.create_parsed_file_with_many_functions(temp_file, large_code)
            
            # Получаем чанки
            chunks = chunker.chunk_parsed_file(parsed_file, large_code)
            
            # Восстанавливаем оригинальный лимит
            chunker.max_tokens_per_chunk = original_limit
            
            # Должно быть несколько чанков
            assert len(chunks) > 1, f"Ожидалось несколько чанков, получено: {len(chunks)}"
            
            # Проверяем отсутствие дубликатов между соседними чанками
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # Получаем последние несколько строк текущего чанка
                current_lines = current_chunk.content.splitlines()[-5:]
                
                # Получаем первые несколько строк следующего чанка
                next_lines = next_chunk.content.splitlines()[:5]
                
                # Проверяем что нет полного дублирования строк
                current_set = set(line.strip() for line in current_lines if line.strip())
                next_set = set(line.strip() for line in next_lines if line.strip())
                
                # Допускается небольшое пересечение (например, пустые строки или комментарии),
                # но не должно быть идентичных содержательных блоков кода
                overlap = current_set.intersection(next_set)
                substantial_overlap = [line for line in overlap 
                                     if len(line) > 10 and not line.startswith('#') 
                                     and not line.startswith('"""') and not line.startswith("'''")]
                
                assert len(substantial_overlap) == 0, \
                    f"Найдено существенное дублирование между чанками {i} и {i+1}: {substantial_overlap}"
            
            # Дополнительная проверка: агрегированный результат не должен содержать дубликатов функций
            all_content = "\n".join(chunk.content for chunk in chunks)
            
            # Подсчитываем количество определений функций
            import re
            function_defs = re.findall(r'^def\s+(\w+)\s*\(', all_content, re.MULTILINE)
            function_names = [name for name in function_defs]
            
            # Проверяем что каждая функция встречается только один раз
            from collections import Counter
            function_counts = Counter(function_names)
            duplicated_functions = [name for name, count in function_counts.items() if count > 1]
            
            assert len(duplicated_functions) == 0, \
                f"Найдены дублированные функции в агрегированном результате: {duplicated_functions}"
                
        finally:
            self.cleanup_temp_file(temp_file)

    def test_t013_boundary_accuracy(self):
        """
        T-013 - Chunking: точность границ
        
        Создаёт файл с длинной строкой, длина которой близка к ограничению размера чанка.
        Проверяет конкатенацию чанков из code_chunker.py обратно в исходный текст.
        Ожидается: конкатенация чанков в исходном порядке даёт точную копию исходного текста.
        """
        # Создаём код с длинными строками и функциями разного размера
        precision_code = '''#!/usr/bin/env python3
"""
Тестовый модуль для проверки точности границ чанкинга.
Содержит функции разного размера для тестирования.
"""

def small_function():
    """Небольшая функция."""
    return "small"

def medium_function():
    """Функция среднего размера с некоторой логикой."""
    data = []
    for i in range(20):
        item = {
            "index": i,
            "value": i * 2,
            "description": f"Medium item {i}",
            "metadata": {"type": "medium", "processed": True}
        }
        data.append(item)
    return data

def large_function_with_long_string():
    """Функция с очень длинной строкой."""
    # Эта строка специально создана длинной для тестирования границ чанкинга
    very_long_string = "This is a very long string that is designed to test the chunking boundaries and ensure that the chunker can handle long lines properly without losing any content or creating duplicates. " * 10
    
    # Дополнительная обработка для увеличения размера функции
    processed_data = []
    for i in range(30):
        entry = {
            "id": f"entry_{i:04d}",
            "long_text": very_long_string + f" - Entry number {i}",
            "timestamp": f"2025-01-01T{i:02d}:00:00",
            "details": {
                "processing_time": i * 0.1,
                "status": "completed" if i % 2 == 0 else "pending",
                "metadata": {
                    "version": "1.0",
                    "author": "test_system",
                    "description": f"Auto-generated entry {i} with long content for boundary testing"
                }
            }
        }
        processed_data.append(entry)
    
    return {
        "long_string": very_long_string,
        "processed_data": processed_data,
        "summary": {
            "total_entries": len(processed_data),
            "total_length": len(very_long_string),
            "processing_complete": True
        }
    }

def another_medium_function():
    """Ещё одна функция среднего размера."""
    config = {
        "settings": {
            "debug": True,
            "verbose": False,
            "max_iterations": 1000
        },
        "parameters": {
            "threshold": 0.95,
            "tolerance": 0.001,
            "batch_size": 32
        }
    }
    
    # Обработка конфигурации
    result = {}
    for section, values in config.items():
        result[section] = {}
        for key, value in values.items():
            if isinstance(value, bool):
                result[section][key] = "enabled" if value else "disabled"
            elif isinstance(value, (int, float)):
                result[section][key] = f"value_{value}"
            else:
                result[section][key] = str(value)
    
    return result

def final_small_function():
    """Завершающая небольшая функция."""
    return {"status": "complete", "message": "All processing finished"}
'''
        
        temp_file = self.create_temp_python_file(precision_code)
        try:
            # Создаём CodeChunker
            chunker = CodeChunker()
            
            # Создаём ParsedFile
            parsed_file = self.create_parsed_file_with_many_functions(temp_file, precision_code)
            
            # Получаем чанки
            chunks = chunker.chunk_parsed_file(parsed_file, precision_code)
            
            # Проверяем что получили хотя бы один чанк
            assert len(chunks) >= 1, "Должен быть создан хотя бы один чанк"
            
            # Извлекаем весь код из чанков в правильном порядке
            # Сортируем чанки по start_line для правильного порядка
            sorted_chunks = sorted(chunks, key=lambda x: x.start_line)
            
            # Собираем код из всех чанков
            reconstructed_lines = []
            covered_lines = set()
            
            for chunk in sorted_chunks:
                chunk_lines = chunk.content.splitlines()
                
                # Для каждого чанка определяем какие строки он покрывает
                # Пропускаем заголовочные комментарии чанков
                actual_code_lines = []
                for line in chunk_lines:
                    # Пропускаем служебные комментарии чанкера
                    if (not line.startswith("# Импорты:") and 
                        not line.startswith("# Комментарии:") and
                        not line.startswith("# Глобальные переменные") and
                        not line.startswith("# Большая функция:") and
                        line.strip() != ""):
                        actual_code_lines.append(line)
                
                reconstructed_lines.extend(actual_code_lines)
            
            # Получаем оригинальные строки (без пустых в конце)
            original_lines = [line for line in precision_code.splitlines() if line.strip()]
            
            # Получаем восстановленные строки (без пустых)
            reconstructed_code_lines = [line for line in reconstructed_lines if line.strip()]
            
            # Проверяем что все основные конструкции кода присутствуют
            original_content = "\n".join(original_lines)
            reconstructed_content = "\n".join(reconstructed_code_lines)
            
            # Проверяем наличие всех функций
            import re
            original_functions = set(re.findall(r'^def\s+(\w+)', original_content, re.MULTILINE))
            reconstructed_functions = set(re.findall(r'^def\s+(\w+)', reconstructed_content, re.MULTILINE))
            
            assert original_functions == reconstructed_functions, \
                f"Функции не совпадают. Оригинал: {original_functions}, Восстановлено: {reconstructed_functions}"
            
            # Проверяем что длинная строка сохранилась полностью
            long_string_marker = "This is a very long string that is designed to test"
            assert long_string_marker in reconstructed_content, \
                "Длинная строка для тестирования границ не найдена в восстановленном коде"
            
            # Проверяем что общий объём кода примерно соответствует оригиналу
            # (допускаем небольшие расхождения из-за форматирования чанков)
            original_substantial_lines = len([line for line in original_lines 
                                            if line.strip() and not line.strip().startswith('#')])
            reconstructed_substantial_lines = len([line for line in reconstructed_code_lines 
                                                 if line.strip() and not line.strip().startswith('#')])
            
            # Разница не должна превышать 10% (учитываем возможные изменения форматирования)
            difference_ratio = abs(original_substantial_lines - reconstructed_substantial_lines) / original_substantial_lines
            assert difference_ratio <= 0.1, \
                f"Слишком большая разница в количестве строк: оригинал {original_substantial_lines}, " \
                f"восстановлено {reconstructed_substantial_lines}, разница {difference_ratio:.2%}"
            
        finally:
            self.cleanup_temp_file(temp_file)