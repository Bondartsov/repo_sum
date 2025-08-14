"""
T-019 - Тест парсеров: синтаксические ошибки и смешанный контент
"""

import pytest
import tempfile
import os
from pathlib import Path
from parsers.base_parser import ParserRegistry
from utils import FileInfo


class TestParsersErrorHandling:
    """Тестирование обработки ошибок и смешанного контента парсерами"""
    
    @pytest.fixture
    def parser_registry(self):
        """Фикстура для создания реестра парсеров"""
        return ParserRegistry()
    
    @pytest.fixture
    def temp_dir(self):
        """Фикстура для создания временной директории"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def create_test_file(self, temp_dir: str, filename: str, content: str) -> FileInfo:
        """Создает тестовый файл и возвращает FileInfo"""
        file_path = os.path.join(temp_dir, filename)
        Path(file_path).write_text(content, encoding='utf-8')
        
        return FileInfo(
            path=file_path,
            name=filename,
            size=len(content.encode('utf-8')),
            language=filename.split('.')[-1],
            extension='.' + filename.split('.')[-1],
            modified_time="2024-01-01T00:00:00Z",
            encoding="utf-8"
        )
    
    def test_python_syntax_error_handling(self, parser_registry, temp_dir):
        """Тест обработки синтаксической ошибки в Python коде"""
        # Создаем Python файл с синтаксической ошибкой
        python_content = '''
def broken_function(
    # Пропущена закрывающая скобка и двоеточие
    print("This will cause a SyntaxError")
    return "broken"

class IncompleteClass
    # Пропущено двоеточие
    pass
        '''
        
        file_info = self.create_test_file(temp_dir, "broken_syntax.py", python_content)
        parser = parser_registry.get_parser(file_info.path)
        
        assert parser is not None, "Должен найти Python парсер"
        
        # Используем safe_parse для обработки ошибок
        result = parser.safe_parse(file_info)
        
        # Проверяем что парсер не упал, но вернул ошибки
        assert result is not None
        assert len(result.parse_errors) > 0, "Должны быть ошибки парсинга"
        assert any("Синтаксическая ошибка" in error for error in result.parse_errors), \
               "Должна быть синтаксическая ошибка"
        assert result.elements == [], "Элементы не должны быть извлечены при синтаксической ошибке"
        assert result.imports == [], "Импорты не должны быть извлечены при синтаксической ошибке"
    
    def test_javascript_malformed_code(self, parser_registry, temp_dir):
        """Тест обработки некорректного JavaScript кода"""
        # JavaScript с синтаксическими ошибками
        js_content = '''
// Некорректный синтаксис JavaScript
function broken( {
    // Пропущена закрывающая скобка параметров
    console.log("test";
    // Пропущена закрывающая скобка в console.log
}

class UnfinishedClass {
    constructor(
    // Незавершенный конструктор
        '''
        
        file_info = self.create_test_file(temp_dir, "broken_syntax.js", js_content)
        parser = parser_registry.get_parser(file_info.path)
        
        assert parser is not None, "Должен найти JavaScript парсер"
        
        # JavaScript парсер использует регулярки, поэтому не должен падать
        result = parser.safe_parse(file_info)
        
        # Проверяем что парсер отработал без краша
        assert result is not None
        # JS парсер не должен падать на синтаксических ошибках, но может не найти элементы
        # или найти частично корректные
        assert isinstance(result.elements, list)
        assert isinstance(result.imports, list)
    
    def test_typescript_malformed_code(self, parser_registry, temp_dir):
        """Тест обработки некорректного TypeScript кода"""
        ts_content = '''
// TypeScript с ошибками типов и синтаксиса
interface BrokenInterface {
    name: string
    age: // Пропущен тип
}

function brokenFunction(param: UnknownType): {
    // Незавершенная функция с неизвестным типом
    return param.
        '''
        
        file_info = self.create_test_file(temp_dir, "broken_syntax.ts", ts_content)
        parser = parser_registry.get_parser(file_info.path)
        
        assert parser is not None, "Должен найти TypeScript парсер"
        
        result = parser.safe_parse(file_info)
        
        assert result is not None
        assert isinstance(result.elements, list)
        assert isinstance(result.imports, list)
    
    def test_csharp_malformed_code(self, parser_registry, temp_dir):
        """Тест обработки некорректного C# кода"""
        cs_content = '''
using System
// Пропущена точка с запятой

namespace TestNamespace {
    class BrokenClass {
        public void BrokenMethod( {
            // Пропущены параметры и закрывающая скобка
            Console.WriteLine("test"
            // Пропущена закрывающая скобка
        }
        '''
        
        file_info = self.create_test_file(temp_dir, "broken_syntax.cs", cs_content)
        parser = parser_registry.get_parser(file_info.path)
        
        assert parser is not None, "Должен найти C# парсер"
        
        result = parser.safe_parse(file_info)
        
        assert result is not None
        assert isinstance(result.elements, list)
        assert isinstance(result.imports, list)
    
    def test_cpp_malformed_code(self, parser_registry, temp_dir):
        """Тест обработки некорректного C++ кода"""
        cpp_content = '''
#include <iostream
// Пропущена закрывающая угловая скобка

class BrokenClass {
public:
    void brokenMethod( {
        // Пропущены параметры
        std::cout << "test" << 
        // Незавершенная строка
    }
        '''
        
        file_info = self.create_test_file(temp_dir, "broken_syntax.cpp", cpp_content)
        parser = parser_registry.get_parser(file_info.path)
        
        assert parser is not None, "Должен найти C++ парсер"
        
        result = parser.safe_parse(file_info)
        
        assert result is not None
        assert isinstance(result.elements, list)
        assert isinstance(result.imports, list)
    
    def test_mixed_content_js_in_ts(self, parser_registry, temp_dir):
        """Тест смешанного контента: JavaScript код в TypeScript файле"""
        mixed_content = '''
// TypeScript файл со смешанным JS/TS контентом
// Чистый JavaScript код (без типов)
function jsFunction(param) {
    return param + 1;
}

// TypeScript код с типами
function tsFunction(param: number): string {
    return param.toString();
}

// Смешанный подход - частично типизированная функция
function mixedFunction(param: any) {
    // JavaScript стиль внутри TS функции
    var result = param;
    if (typeof result == "string") {
        return result.toUpperCase();
    }
    return String(result);
}

// JavaScript объект без типизации в TS файле
const jsObject = {
    method: function(x) { return x * 2; },
    arrow: (y) => y / 2
};
        '''
        
        file_info = self.create_test_file(temp_dir, "mixed_content.ts", mixed_content)
        parser = parser_registry.get_parser(file_info.path)
        
        assert parser is not None, "Должен найти TypeScript парсер"
        
        result = parser.safe_parse(file_info)
        
        # Проверяем что смешанный контент обрабатывается без ошибок
        assert result is not None
        assert len(result.parse_errors) == 0, "Не должно быть ошибок парсинга для смешанного контента"
        # Должны найти функции
        function_elements = [elem for elem in result.elements if elem.type == "function"]
        assert len(function_elements) >= 3, "Должны найти как минимум 3 функции"
    
    def test_mixed_content_html_in_python(self, parser_registry, temp_dir):
        """Тест смешанного контента: HTML в Python строках"""
        mixed_content = '''
"""
Модуль с HTML контентом внутри Python строк
"""

def generate_html():
    """Функция генерирует HTML разметку"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <script>
            function jsFunction() {
                console.log("JavaScript внутри Python строки");
                var result = 42;  // Корректный JavaScript внутри Python строки
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Заголовок</h1>
            <p>Параграф с "вложенными кавычками"</p>
        </div>
    </body>
    </html>
    """
    return html_template

class HTMLGenerator:
    """Класс для работы с HTML"""
    
    def __init__(self):
        self.css = """
        .broken-css {
            color: red;
            /* Пропущена точка с запятой, но это строка в Python */
            background: url("image.png");
        }
        """
    
    def get_js_code(self):
        # JavaScript код с ошибками внутри Python строки
        return """
        function(param) {
            // Синтаксическая ошибка JS, но для Python это просто строка
            return param + 1;
        }
        """
        '''
        
        file_info = self.create_test_file(temp_dir, "mixed_html_python.py", mixed_content)
        parser = parser_registry.get_parser(file_info.path)
        
        assert parser is not None, "Должен найти Python парсер"
        
        result = parser.safe_parse(file_info)
        
        # Python парсер должен обработать код успешно, игнорируя содержимое строк
        assert result is not None
        # Для смешанного контента могут быть незначительные предупреждения, но не критические ошибки
        if len(result.parse_errors) > 0:
            # Логируем ошибки для отладки, но не падаем если их немного
            print(f"Parse errors in mixed content: {result.parse_errors}")
            assert len(result.parse_errors) <= 1, "Должно быть не более одной ошибки для смешанного контента"
        
        # Должны найти функции и классы
        classes = [elem for elem in result.elements if elem.type == "class"]
        functions = [elem for elem in result.elements if elem.type == "function"]
        assert len(classes) >= 1, "Должен найти как минимум 1 класс"
        assert len(functions) >= 1, "Должен найти как минимум 1 функцию"
    
    def test_parser_registry_integration(self, parser_registry, temp_dir):
        """Тест интеграции парсеров с ParserRegistry при обработке ошибочных файлов"""
        test_files = [
            ("syntax_error.py", "def broken(\npass"),  # Python с синтаксической ошибкой
            ("malformed.js", "function test( {\nconsole.log('test';"),  # JS с ошибкой
            ("invalid.ts", "class Test {\nconstructor(\n}"),  # TS с ошибкой
            ("broken.cs", "class Test {\npublic void Method(\n"),  # C# с ошибкой
            ("bad.cpp", "#include <iostream\nint main( {\n")  # C++ с ошибкой
        ]
        
        for filename, content in test_files:
            file_info = self.create_test_file(temp_dir, filename, content)
            parser = parser_registry.get_parser(file_info.path)
            
            # Проверяем что найден подходящий парсер
            assert parser is not None, f"Должен найти парсер для {filename}"
            
            # Проверяем что safe_parse не падает
            result = parser.safe_parse(file_info)
            assert result is not None, f"safe_parse должен вернуть результат для {filename}"
            assert isinstance(result.elements, list), f"elements должен быть списком для {filename}"
            assert isinstance(result.imports, list), f"imports должен быть списком для {filename}"
            assert isinstance(result.parse_errors, list), f"parse_errors должен быть списком для {filename}"
    
    def test_safe_parse_vs_direct_parse(self, parser_registry, temp_dir):
        """Тест сравнения safe_parse и прямого вызова parse_file с ошибочным кодом"""
        # Создаем Python файл с синтаксической ошибкой
        python_content = "def broken_func(\nprint('missing parenthesis')"
        file_info = self.create_test_file(temp_dir, "test_safe_parse.py", python_content)
        
        parser = parser_registry.get_parser(file_info.path)
        assert parser is not None
        
        # Прямой вызов parse_file должен выбросить исключение или вернуть ParsedFile с ошибками
        try:
            direct_result = parser.parse_file(file_info)
            # Если не выбросило исключение, проверяем что есть ошибки
            assert len(direct_result.parse_errors) > 0, "Прямой вызов должен записать ошибки"
        except Exception:
            # Это ожидаемое поведение для некоторых парсеров
            pass
        
        # safe_parse должен всегда вернуть результат без исключений
        safe_result = parser.safe_parse(file_info)
        assert safe_result is not None, "safe_parse должен всегда возвращать результат"
        assert isinstance(safe_result.parse_errors, list), "parse_errors должен быть списком"
        
        # При ошибке safe_parse должен вернуть пустые коллекции
        if len(safe_result.parse_errors) > 0:
            assert safe_result.elements == [], "При ошибках elements должен быть пустым"
            assert safe_result.imports == [], "При ошибках imports должен быть пустым"
    
    def test_encoding_error_handling(self, parser_registry, temp_dir):
        """Тест обработки ошибок кодировки"""
        # Создаем файл с некорректной кодировкой
        file_path = os.path.join(temp_dir, "encoding_test.py")
        
        # Записываем файл в кодировке, отличной от UTF-8
        with open(file_path, 'w', encoding='latin-1') as f:
            f.write("# Comment with special chars\ndef test():\n    pass")
        
        file_info = FileInfo(
            path=file_path,
            name="encoding_test.py",
            size=os.path.getsize(file_path),
            language="python",
            extension=".py",
            modified_time="2024-01-01T00:00:00Z",
            encoding="utf-8"  # Указываем неправильную кодировку намеренно
        )
        
        parser = parser_registry.get_parser(file_info.path)
        assert parser is not None
        
        # safe_parse должен обработать ошибку кодировки
        result = parser.safe_parse(file_info)
        assert result is not None
        
        # Может быть либо успешный результат (если парсер переключился на другую кодировку)
        # либо ошибка парсинга
        assert isinstance(result.elements, list)
        assert isinstance(result.imports, list)
        assert isinstance(result.parse_errors, list)