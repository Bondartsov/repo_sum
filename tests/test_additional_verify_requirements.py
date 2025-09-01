"""
Тесты для проверки зависимостей проекта - T-020

Проверяет скрипт scripts/verify_requirements.py, который анализирует соответствие
импортов в коде и пакетов в requirements.txt.
"""

import pytest
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest import mock

# Импортируем функции из скрипта для прямого тестирования
import importlib.util
spec = importlib.util.spec_from_file_location("verify_requirements",
                                              Path(__file__).parent.parent / "scripts" / "verify_requirements.py")
verify_requirements = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_requirements)
collect_top_imports = verify_requirements.collect_top_imports
parse_requirements = verify_requirements.parse_requirements
main = verify_requirements.main
MODULE_TO_PKG = verify_requirements.MODULE_TO_PKG


@pytest.mark.functional
class TestVerifyRequirements:
    """Тесты для скрипта проверки зависимостей"""

    def create_temp_project(self, temp_dir, requirements_content, python_files):
        """
        Создает временную структуру проекта для тестирования
        
        Args:
            temp_dir: Временная директория
            requirements_content: Содержимое requirements.txt
            python_files: Словарь {путь_к_файлу: содержимое_файла}
        """
        temp_path = Path(temp_dir)
        
        # Создаем requirements.txt
        req_file = temp_path / "requirements.txt"
        req_file.write_text(requirements_content, encoding="utf-8")
        
        # Создаем Python файлы
        for file_path, content in python_files.items():
            full_path = temp_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            
        return temp_path

    def test_parse_requirements_basic(self):
        """Тест парсинга requirements.txt с базовыми пакетами"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            req_file = temp_path / "requirements.txt"
            req_content = """
# Комментарий
openai>=1.99.6
tiktoken>=0.8.0
pytest==8.3.4

# Еще комментарий
click
"""
            req_file.write_text(req_content, encoding="utf-8")
            
            packages = parse_requirements(req_file)
            expected = {"openai", "tiktoken", "pytest", "click"}
            assert packages == expected

    def test_parse_requirements_with_complex_versions(self):
        """Тест парсинга requirements.txt со сложными версиями"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            req_file = temp_path / "requirements.txt"
            req_content = """
qdrant-client[fastembed]>=1.15.1
sentence-transformers>=3.0.0,<4.0.0
torch==2.4.0+cpu
numpy>=1.24.0
"""
            req_file.write_text(req_content, encoding="utf-8")
            
            packages = parse_requirements(req_file)
            expected = {"qdrant-client[fastembed]", "sentence-transformers", "torch", "numpy"}
            assert packages == expected

    def test_collect_top_imports_basic(self):
        """Тест сбора импортов из Python файлов"""
        with tempfile.TemporaryDirectory() as temp_dir:
            python_files = {
                "main.py": """
import openai
import tiktoken
from pathlib import Path
from config import settings  # локальный импорт - должен игнорироваться
import sys
""",
                "utils.py": """
import click
from rich.console import Console
import json
"""
            }
            
            temp_path = self.create_temp_project(temp_dir, "", python_files)
            modules = collect_top_imports(temp_path)
            
            expected = ["click", "json", "openai", "pathlib", "rich", "sys", "tiktoken"]
            assert modules == expected

    def test_collect_top_imports_ignores_local_modules(self):
        """Тест что локальные модули проекта игнорируются"""
        with tempfile.TemporaryDirectory() as temp_dir:
            python_files = {
                "main.py": """
import config
import utils
import openai_integration
import parsers
from tests import helpers
import web_ui
""",
                "test_file.py": """
import pytest
import openai
"""
            }
            
            temp_path = self.create_temp_project(temp_dir, "", python_files)
            modules = collect_top_imports(temp_path)
            
            # Все локальные модули должны быть проигнорированы
            assert "config" not in modules
            assert "utils" not in modules
            assert "openai_integration" not in modules
            assert "parsers" not in modules
            assert "tests" not in modules
            assert "web_ui" not in modules
            
            # Но внешние модули должны остаться
            assert "pytest" in modules
            assert "openai" in modules

    def test_script_success_all_dependencies_present(self):
        """Тест успешного выполнения когда все зависимости присутствуют"""
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_content = """
openai>=1.99.6
tiktoken>=0.8.0
pytest>=8.3.4
"""
            python_files = {
                "main.py": """
import openai
import tiktoken
""",
                "test_main.py": """
import pytest
"""
            }
            
            temp_path = self.create_temp_project(temp_dir, requirements_content, python_files)
            
            # Мокируем глобальные переменные скрипта
            with mock.patch.object(verify_requirements, 'PROJECT_ROOT', temp_path), \
                 mock.patch.object(verify_requirements, 'REQ_FILE', temp_path / "requirements.txt"):
                
                # Запускаем main() напрямую вместо subprocess
                with mock.patch('builtins.print') as mock_print, \
                     mock.patch('sys.exit') as mock_exit:
                    
                    main()
                    
                    # Проверяем, что exit не был вызван (успех)
                    mock_exit.assert_not_called()
                    
                    # Проверяем успешное сообщение
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    success_msg = any("Requirements look consistent" in call for call in print_calls)
                    assert success_msg

    def test_script_failure_missing_dependencies(self):
        """Тест T-020: обнаружение отсутствующих пакетов"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # requirements.txt не содержит openai, но код его импортирует
            requirements_content = """
tiktoken>=0.8.0
pytest>=8.3.4
"""
            python_files = {
                "main.py": """
import openai  # этот пакет отсутствует в requirements.txt
import tiktoken
""",
                "test_main.py": """
import pytest
"""
            }
            
            temp_path = self.create_temp_project(temp_dir, requirements_content, python_files)
            
            # Мокируем глобальные переменные скрипта
            with mock.patch.object(verify_requirements, 'PROJECT_ROOT', temp_path), \
                 mock.patch.object(verify_requirements, 'REQ_FILE', temp_path / "requirements.txt"):
                
                # Запускаем main() напрямую
                with mock.patch('builtins.print') as mock_print, \
                     mock.patch('sys.exit') as mock_exit:
                    
                    main()
                    
                    # Должен вызвать sys.exit(1)
                    mock_exit.assert_called_once_with(1)
                    
                    # Должен отчётливо указать отсутствующие пакеты
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    missing_info = any("MISSING packages" in call for call in print_calls)
                    assert missing_info
                    openai_info = any("openai -> openai" in call for call in print_calls)
                    assert openai_info

    def test_script_with_extra_dependencies(self):
        """Тест обнаружения потенциально лишних пакетов"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # requirements.txt содержит пакеты, которые не импортируются
            requirements_content = """
openai>=1.99.6
tiktoken>=0.8.0
unused-package>=1.0.0
another-unused>=2.0.0
"""
            python_files = {
                "main.py": """
import openai
import tiktoken
"""
            }
            
            temp_path = self.create_temp_project(temp_dir, requirements_content, python_files)
            
            # Мокируем глобальные переменные скрипта
            with mock.patch.object(verify_requirements, 'PROJECT_ROOT', temp_path), \
                 mock.patch.object(verify_requirements, 'REQ_FILE', temp_path / "requirements.txt"):
                
                # Запускаем main() напрямую
                with mock.patch('builtins.print') as mock_print, \
                     mock.patch('sys.exit') as mock_exit:
                    
                    main()
                    
                    # При лишних пакетах код выхода должен быть 0 (только предупреждение)
                    mock_exit.assert_not_called()
                    
                    # Должен показать потенциально лишние пакеты
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    extra_info = any("POTENTIALLY EXTRA packages" in call for call in print_calls)
                    assert extra_info
                    unused_info = any("unused-package" in call for call in print_calls)
                    assert unused_info
                    another_unused_info = any("another-unused" in call for call in print_calls)
                    assert another_unused_info
                    success_msg = any("Requirements look consistent" in call for call in print_calls)
                    assert success_msg

    def test_script_missing_and_extra_dependencies(self):
        """Тест с одновременно отсутствующими и лишними пакетами"""
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_content = """
tiktoken>=0.8.0
unused-package>=1.0.0
"""
            python_files = {
                "main.py": """
import openai  # отсутствует в requirements
import tiktoken
"""
            }
            
            temp_path = self.create_temp_project(temp_dir, requirements_content, python_files)
            
            # Мокируем глобальные переменные скрипта
            with mock.patch.object(verify_requirements, 'PROJECT_ROOT', temp_path), \
                 mock.patch.object(verify_requirements, 'REQ_FILE', temp_path / "requirements.txt"):
                
                # Запускаем main() напрямую
                with mock.patch('builtins.print') as mock_print, \
                     mock.patch('sys.exit') as mock_exit:
                    
                    main()
                    
                    # При отсутствующих пакетах код выхода должен быть 1
                    mock_exit.assert_called_once_with(1)
                    
                    # Должен показать и отсутствующие, и лишние пакеты
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    missing_info = any("MISSING packages" in call for call in print_calls)
                    assert missing_info
                    openai_info = any("openai -> openai" in call for call in print_calls)
                    assert openai_info
                    extra_info = any("POTENTIALLY EXTRA packages" in call for call in print_calls)
                    assert extra_info
                    unused_info = any("unused-package" in call for call in print_calls)
                    assert unused_info

    def test_direct_function_calls_with_mock(self):
        """Тест прямых вызовов функций с мокированием"""
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_content = "openai>=1.99.6\ntiktoken>=0.8.0"
            python_files = {
                "main.py": "import openai\nimport missing_pkg"
            }
            
            temp_path = self.create_temp_project(temp_dir, requirements_content, python_files)
            req_file = temp_path / "requirements.txt"
            
            # Тестируем функции напрямую
            modules = collect_top_imports(temp_path)
            packages = parse_requirements(req_file)
            
            assert "openai" in modules
            assert "missing_pkg" in modules
            assert "openai" in packages
            assert "tiktoken" in packages
            
            # Мокируем sys.exit для тестирования main()
            with mock.patch('sys.exit') as mock_exit, \
                 mock.patch.object(verify_requirements, 'PROJECT_ROOT', temp_path), \
                 mock.patch.object(verify_requirements, 'REQ_FILE', req_file), \
                 mock.patch('builtins.print') as mock_print:
                
                # Добавляем missing_pkg в MODULE_TO_PKG для теста
                with mock.patch.dict(verify_requirements.MODULE_TO_PKG, {"missing_pkg": "missing-package"}):
                    main()
                    
                    # Должен вызвать sys.exit(1) из-за отсутствующего пакета
                    mock_exit.assert_called_once_with(1)
                    
                    # Должен напечатать информацию об отсутствующем пакете
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    missing_info = any("MISSING packages" in call for call in print_calls)
                    assert missing_info

    def test_empty_requirements_file(self):
        """Тест с пустым файлом requirements.txt"""
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_content = "# Только комментарии\n\n"
            python_files = {
                "main.py": "import openai"
            }
            
            temp_path = self.create_temp_project(temp_dir, requirements_content, python_files)
            
            # Мокируем глобальные переменные скрипта
            with mock.patch.object(verify_requirements, 'PROJECT_ROOT', temp_path), \
                 mock.patch.object(verify_requirements, 'REQ_FILE', temp_path / "requirements.txt"):
                
                # Запускаем main() напрямую
                with mock.patch('builtins.print') as mock_print, \
                     mock.patch('sys.exit') as mock_exit:
                    
                    main()
                    
                    # Должен найти отсутствующий пакет
                    mock_exit.assert_called_once_with(1)
                    
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    missing_info = any("MISSING packages" in call for call in print_calls)
                    assert missing_info

    def test_no_python_files(self):
        """Тест с проектом без Python файлов"""
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_content = "openai>=1.99.6"
            python_files = {}  # Нет Python файлов
            
            temp_path = self.create_temp_project(temp_dir, requirements_content, python_files)
            
            result = subprocess.run([
                sys.executable, 
                str(Path(__file__).parent.parent / "scripts" / "verify_requirements.py")
            ], 
            cwd=str(temp_path), 
            capture_output=True, 
            text=True
            )
            
            # Без Python файлов нет импортов, поэтому все пакеты будут "лишние"
            assert result.returncode == 0
            assert "POTENTIALLY EXTRA packages" in result.stdout
            assert "openai" in result.stdout
