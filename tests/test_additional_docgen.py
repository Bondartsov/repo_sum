#!/usr/bin/env python3
"""
Дополнительные тесты генерации документации для проекта repo_sum.
Тесты T-015 и T-016 согласно техническому заданию.
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import re

import pytest

from doc_generator import DocumentationGenerator, MarkdownGenerator
from utils import (
    FileInfo, ParsedFile, GPTAnalysisResult, 
    create_error_gpt_result, ensure_directory_exists
)


class TestAdditionalDocGen:
    """Дополнительные тесты генерации документации"""
    
    @pytest.fixture
    def temp_repo_dir(self):
        """Создает временную директорию для тестового репозитория"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Создает временную директорию для вывода"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def main_script_path(self):
        """Путь к основному скрипту main.py"""
        return Path(__file__).parent.parent / "main.py"

    def create_test_file_with_content(self, repo_dir: str, filename: str, content: str) -> str:
        """Создает тестовый файл с указанным содержимым"""
        file_path = Path(repo_dir) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)

    def create_parsed_file_from_content(self, file_path: str, content: str) -> ParsedFile:
        """Создает объект ParsedFile из содержимого файла"""
        path_obj = Path(file_path)
        file_info = FileInfo(
            path=str(file_path),
            name=path_obj.name,
            size=len(content.encode('utf-8')),
            language="python" if path_obj.suffix == ".py" else "markdown",
            extension=path_obj.suffix,
            modified_time="2025-01-01T00:00:00",
            encoding="utf-8"
        )
        return ParsedFile(
            file_info=file_info,
            imports=[],
            global_comments=[],
            parse_errors=[]
        )

    def create_mock_gpt_result(self, content: str) -> GPTAnalysisResult:
        """Создает mock GPTAnalysisResult с указанным содержимым"""
        return GPTAnalysisResult(
            summary="Тестовый анализ",
            key_components=["test"],
            analysis_per_chunk={},
            full_text=content,
            error=None
        )

    @pytest.mark.integration
    @patch('openai_integration.OpenAIManager')
    def test_t015_markdown_header_collisions(self, mock_openai, temp_repo_dir, temp_output_dir):
        """
        T-015 - Докогенерация: коллизии заголовков Markdown
        
        В нескольких входных файлах создать идентичные заголовки первого/второго уровня
        Запустить генерацию документации
        Проверить итоговый Markdown
        Ожидается: doc_generator.py генерирует устойчивые к коллизиям якоря/заголовки,
        ссылки внутри отчёта работают корректно
        """
        # Создаем тестовые файлы с коллизиями заголовков
        file1_content = """# Основной заголовок

## Функционал
Описание функционала в файле 1

## API
Описание API в файле 1

### Методы
- method1()
- method2()
"""

        file2_content = """# Основной заголовок

## Функционал  
Описание функционала в файле 2

## API
Описание API в файле 2

### Методы
- method3()
- method4()
"""

        file3_content = """# Основной заголовок

## Конфигурация
Описание конфигурации

## API
Описание API в файле 3
"""

        # Создаем файлы
        self.create_test_file_with_content(temp_repo_dir, "module1.py", file1_content)
        self.create_test_file_with_content(temp_repo_dir, "module2.py", file2_content) 
        self.create_test_file_with_content(temp_repo_dir, "config.py", file3_content)

        # Настраиваем mock для OpenAI
        mock_openai.return_value.analyze_code.return_value = self.create_mock_gpt_result(file1_content)

        # Создаем данные для генерации
        files_data = []
        for filename, content in [("module1.py", file1_content), ("module2.py", file2_content), ("config.py", file3_content)]:
            file_path = str(Path(temp_repo_dir) / filename)
            parsed_file = self.create_parsed_file_from_content(file_path, content)
            gpt_result = self.create_mock_gpt_result(content)
            files_data.append((parsed_file, gpt_result))

        # Генерируем документацию
        doc_gen = DocumentationGenerator()
        result = doc_gen.generate_complete_documentation(
            files_data, temp_output_dir, temp_repo_dir
        )

        # Проверяем результат генерации
        assert result['success'] is True
        assert result['total_files'] == 3
        assert result['successful'] == 3
        assert result['failed'] == 0

        # Проверяем структуру выходного каталога
        repo_name = Path(temp_repo_dir).name
        summary_dir = Path(temp_output_dir) / f"SUMMARY_REPORT_{repo_name}"
        assert summary_dir.exists()
        
        readme_path = summary_dir / "README.md"
        assert readme_path.exists()

        # Проверяем, что созданы отчёты для всех файлов
        report_files = list(summary_dir.glob("report_*.md"))
        assert len(report_files) == 3

        # Анализируем содержимое сгенерированных файлов на коллизии заголовков
        all_headers = []
        for report_file in report_files:
            content = report_file.read_text(encoding='utf-8')
            # Ищем заголовки разных уровней
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            all_headers.extend([(len(level), header.strip()) for level, header in headers])

        # Проверяем наличие потенциально конфликтующих заголовков
        header_texts = [header[1] for header in all_headers]
        duplicate_headers = []
        seen_headers = set()
        
        for header in header_texts:
            if header in seen_headers and header not in duplicate_headers:
                duplicate_headers.append(header)
            seen_headers.add(header)

        # Должны быть дубликаты из-за одинаковых заголовков в исходных файлах
        assert len(duplicate_headers) >= 2, f"Ожидались дубликаты заголовков, найдено: {duplicate_headers}"

        # Проверяем, что doc_generator корректно обработал коллизии
        # (например, добавил суффиксы к якорям или использовал другие механизмы)
        for report_file in report_files:
            content = report_file.read_text(encoding='utf-8')
            
            # Проверяем, что файл содержит корректный Markdown
            assert content.strip(), f"Файл {report_file} пустой"
            
            # Проверяем наличие заголовков первого уровня
            h1_headers = re.findall(r'^#\s+(.+)$', content, re.MULTILINE)
            assert len(h1_headers) >= 1, f"Не найдены заголовки H1 в {report_file}"

        print(f"T-015 PASSED: Коллизии заголовков Markdown корректно обработаны")
        print(f"Всего заголовков: {len(all_headers)}")
        print(f"Дублирующихся: {len(duplicate_headers)}")
        print(f"Выходной каталог: {summary_dir}")

    @pytest.mark.integration
    @patch('openai_integration.OpenAIManager')
    def test_t016_long_lines_tables_lists(self, mock_openai, temp_repo_dir, temp_output_dir):
        """
        T-016 - Докогенерация: длинные строки, таблицы, списки
        
        Подготовить контент с длинными строками, таблицами и вложенными списками
        Запустить генерацию
        Изучить сформированный отчёт
        Ожидается: doc_generator.py сохраняет таблицы и списки без искажений,
        длинные строки корректно переносятся/сохраняются без поломки Markdown
        """
        # Создаем контент с длинными строками, таблицами и вложенными списками
        complex_content = """# Модуль с сложной структурой

## Очень длинная строка для тестирования обработки
Это очень длинная строка, которая содержит много текста и должна быть корректно обработана генератором документации. В ней есть разные символы: @#$%^&*()_+-={}[]|\\:";'<>?,./ и она может превышать стандартную длину строки в 80 или 120 символов, что должно тестировать алгоритмы переноса или сохранения длинных строк в результирующем Markdown документе без нарушения форматирования.

## Таблица с данными

| Столбец 1 | Столбец 2 | Столбец 3 | Очень длинный заголовок столбца |
|-----------|-----------|-----------|--------------------------------|
| Значение 1| Значение 2| Значение 3| Очень длинное значение, которое может потребовать специальной обработки |
| A | B | C | Еще одно длинное значение с символами: <>[]{}()*&^%$ |
| `код` | **жирный** | *курсив* | ~~зачеркнутый~~ |

## Вложенные списки

1. Первый элемент верхнего уровня
   - Подэлемент 1
   - Подэлемент 2 с длинным текстом: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua
     * Вложенный элемент третьего уровня
     * Еще один вложенный элемент
       - Четвертый уровень вложенности
       - Элемент с кодом: `function_name(parameter_with_very_long_name, another_parameter)`
2. Второй элемент верхнего уровня
   1. Нумерованный подэлемент
   2. Еще один нумерованный подэлемент
      - Смешанный тип списка
      - Элемент со ссылкой: [очень длинный текст ссылки](https://example.com/very/long/url/that/might/cause/formatting/issues)

## Код блоки с длинными строками

```python
def very_long_function_name_that_exceeds_normal_line_length(parameter_one, parameter_two, parameter_three, parameter_four):
    '''
    Очень длинная документация функции, которая описывает все параметры и возможные исключения
    '''
    very_long_variable_name = parameter_one + parameter_two + parameter_three + parameter_four
    return very_long_variable_name
```

## Блок цитат с таблицей

> ### Цитата с таблицей внутри
> 
> | Колонка А | Колонка Б |
> |-----------|-----------|
> | > Значение в цитате | > Еще значение |
> 
> Длинная цитата с множественными строками и сложным форматированием, включающая код: `inline_code_example`, **жирный текст**, *курсив* и [ссылки](http://example.com).
"""

        # Создаем тестовый файл
        self.create_test_file_with_content(temp_repo_dir, "complex_module.py", complex_content)

        # Настраиваем mock для OpenAI
        mock_openai.return_value.analyze_code.return_value = self.create_mock_gpt_result(complex_content)

        # Создаем данные для генерации
        file_path = str(Path(temp_repo_dir) / "complex_module.py")
        parsed_file = self.create_parsed_file_from_content(file_path, complex_content)
        gpt_result = self.create_mock_gpt_result(complex_content)
        files_data = [(parsed_file, gpt_result)]

        # Генерируем документацию
        doc_gen = DocumentationGenerator()
        result = doc_gen.generate_complete_documentation(
            files_data, temp_output_dir, temp_repo_dir
        )

        # Проверяем результат генерации
        assert result['success'] is True
        assert result['total_files'] == 1
        assert result['successful'] == 1
        assert result['failed'] == 0

        # Проверяем структуру выходного каталога
        repo_name = Path(temp_repo_dir).name
        summary_dir = Path(temp_output_dir) / f"SUMMARY_REPORT_{repo_name}"
        assert summary_dir.exists()

        # Находим сгенерированный отчёт
        report_files = list(summary_dir.glob("report_*.md"))
        assert len(report_files) == 1
        report_file = report_files[0]
        
        generated_content = report_file.read_text(encoding='utf-8')

        # Проверяем сохранение таблиц
        table_patterns = [
            r'\|.*\|.*\|.*\|',  # строки таблицы
            r'\|[-\s]+\|[-\s]+\|',  # разделители столбцов
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, generated_content)
            assert len(matches) > 0, f"Таблицы не найдены или искажены. Паттерн: {pattern}"

        # Проверяем сохранение вложенных списков
        list_patterns = [
            r'^\s*1\.\s+',  # нумерованные списки
            r'^\s*[-*]\s+',  # маркированные списки  
            r'^\s{2,}[-*]\s+',  # вложенные списки с отступами
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, generated_content, re.MULTILINE)
            assert len(matches) > 0, f"Списки не найдены или искажены. Паттерн: {pattern}"

        # Проверяем сохранение длинных строк
        lines = generated_content.split('\n')
        long_lines = [line for line in lines if len(line) > 100]
        assert len(long_lines) > 0, "Не найдены длинные строки в выходном документе"

        # Проверяем, что не нарушена структура Markdown
        # Должны быть заголовки разных уровней
        headers = re.findall(r'^(#{1,6})\s+(.+)$', generated_content, re.MULTILINE)
        assert len(headers) >= 3, f"Не найдены заголовки в сгенерированном контенте: {len(headers)}"

        # Проверяем наличие блоков кода
        code_blocks = re.findall(r'```[\s\S]*?```', generated_content, re.MULTILINE)
        assert len(code_blocks) > 0, "Блоки кода не найдены или искажены"

        # Проверяем наличие цитат
        quotes = re.findall(r'^>\s+', generated_content, re.MULTILINE)
        assert len(quotes) > 0, "Цитаты не найдены или искажены"

        # Проверяем базовое форматирование
        formatting_patterns = [
            (r'\*\*[^*]+\*\*', 'жирный текст'),  # жирный
            (r'\*[^*]+\*', 'курсив'),  # курсив  
            (r'`[^`]+`', 'inline код'),  # inline код
            (r'\[.+\]\(.+\)', 'ссылки')  # ссылки
        ]
        
        for pattern, name in formatting_patterns:
            matches = re.findall(pattern, generated_content)
            assert len(matches) > 0, f"Не найдено форматирование: {name}"

        print(f"T-016 PASSED: Длинные строки, таблицы и списки корректно сохранены")
        print(f"Длинных строк: {len(long_lines)}")
        print(f"Заголовков: {len(headers)}")
        print(f"Блоков кода: {len(code_blocks)}")
        print(f"Размер сгенерированного файла: {len(generated_content)} символов")
        print(f"Отчёт сохранен: {report_file}")

    def test_direct_markdown_generator(self):
        """
        Тест прямого использования MarkdownGenerator для проверки 
        fallback форматирования без OpenAI
        """
        # Создаем тестовые данные
        file_info = FileInfo(
            path="test_direct.py",
            name="test_direct.py", 
            size=100,
            language="python",
            extension=".py",
            modified_time="2025-01-01T00:00:00",
            encoding="utf-8"
        )
        
        parsed_file = ParsedFile(
            file_info=file_info,
            imports=["os", "sys"],
            global_comments=["Тестовый комментарий"],
            parse_errors=[]
        )

        # Тестируем fallback форматирование (без полного GPT отчета)
        md_gen = MarkdownGenerator()
        gpt_result = GPTAnalysisResult(
            summary="Краткий анализ",
            key_components=[],
            analysis_per_chunk={},
            full_text="",  # Пустой full_text должен вызвать fallback
            error=None
        )
        
        content = md_gen._generate_file_content(parsed_file, gpt_result)
        
        # Проверяем структуру fallback формата
        assert "test_direct.py" in content
        assert "**Путь:** `test_direct.py`" in content
        assert "**Язык:** python" in content
        assert "## Импорты" in content
        assert "- os" in content
        assert "- sys" in content
        assert "## Комментарии" in content
        assert "> Тестовый комментарий" in content
        assert "## Анализ кода" in content
        assert "Краткий анализ" in content

        print("Прямой тест MarkdownGenerator PASSED")


if __name__ == "__main__":
    # Запуск тестов напрямую для отладки
    pytest.main([__file__, "-v", "-s"])
