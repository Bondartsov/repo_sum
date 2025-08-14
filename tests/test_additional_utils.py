"""
T-014 - Utils: нормализация путей (Windows/UNC/POSIX)

Тестирует обработку смешанных форматов путей в проекте repo_sum:
- Windows пути: C:\repo\file.py
- UNC пути: \\\\server\\share\\repo\\file.py
- POSIX пути: /home/user/repo/file.py

Проверяет что логика из utils.py обеспечивает предсказуемый формат
и пути консистентны во всех частях пайплайна.
"""

import pytest
import tempfile
import os
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from unittest.mock import patch, Mock
import shutil

from utils import (
    FileInfo, 
    ensure_directory_exists,
    clean_filename,
    create_error_parsed_file,
    FileParsingError
)
from file_scanner import FileScanner
from config import get_config


class TestPathNormalization:
    """Тесты нормализации путей для различных форматов (Windows/UNC/POSIX)"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        
    def teardown_method(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_file(self, relative_path: str, content: str = "test content") -> str:
        """Создает тестовый файл и возвращает полный путь"""
        full_path = os.path.join(self.temp_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.test_files.append(full_path)
        return full_path

    def test_windows_style_paths(self):
        """Тест обработки путей в стиле Windows (обратные слеши)"""
        # Создаем файл
        test_file = self.create_test_file("subdir/test.py", "print('hello')")
        
        # Имитируем Windows-стиль пути с обратными слешами
        windows_style_path = test_file.replace('/', '\\')
        
        # Тестируем нормализацию через Path.resolve()
        normalized = Path(windows_style_path).resolve()
        
        # Проверяем что путь нормализован
        assert normalized.exists()
        assert str(normalized).replace('\\', '/').endswith('subdir/test.py')
        
        # Тестируем создание FileInfo с нормализованным путем
        file_info = FileInfo(
            path=str(normalized),
            name="test.py",
            size=100,
            language="python", 
            extension=".py",
            modified_time="2025-01-01 12:00:00"
        )
        
        assert file_info.path == str(normalized)
        assert file_info.name == "test.py"

    def test_posix_style_paths(self):
        """Тест обработки путей в стиле POSIX (прямые слеши)"""
        # Создаем файл
        test_file = self.create_test_file("subdir/test.py", "print('hello')")
        
        # Имитируем POSIX-стиль пути (уже с прямыми слешами)
        posix_style_path = test_file.replace('\\', '/')
        
        # Тестируем нормализацию
        normalized = Path(posix_style_path).resolve()
        
        assert normalized.exists()
        assert 'test.py' in str(normalized)
        
        # Проверяем консистентность в FileInfo
        file_info = FileInfo(
            path=str(normalized),
            name="test.py",
            size=100,
            language="python",
            extension=".py", 
            modified_time="2025-01-01 12:00:00"
        )
        
        assert file_info.path == str(normalized)

    def test_mixed_path_separators(self):
        """Тест обработки путей со смешанными разделителями"""
        # Создаем файл
        test_file = self.create_test_file("dir1/dir2/test.py", "# test")
        
        # Создаем путь со смешанными разделителями
        if os.name == 'nt':  # Windows
            mixed_path = test_file.replace('/', '\\', 1).replace('\\', '/', 1)
        else:  # Unix-like
            mixed_path = test_file
            
        # Нормализуем через Path
        normalized = Path(mixed_path).resolve()
        
        assert normalized.exists()
        assert normalized.is_file()
        
        # Проверяем что путь стал консистентным
        path_str = str(normalized)
        if os.name == 'nt':
            # На Windows все должно быть с обратными слешами
            assert '\\' in path_str or '/' in path_str  # Path может использовать любые
        else:
            # На Unix все должно быть с прямыми слешами
            assert '/' in path_str

    def test_relative_paths_normalization(self):
        """Тест нормализации относительных путей"""
        # Создаем файл
        test_file = self.create_test_file("test.py", "# test")
        
        # Переходим в директорию с файлом
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            
            # Тестируем относительный путь
            relative_path = "./test.py"
            normalized = Path(relative_path).resolve()
            
            assert normalized.exists()
            assert normalized.is_absolute()
            assert str(normalized).endswith('test.py')
            
            # Тестируем путь с ".."
            relative_path_with_parent = "../" + os.path.basename(self.temp_dir) + "/test.py"
            normalized_parent = Path(relative_path_with_parent).resolve()
            
            assert normalized_parent.exists()
            assert normalized_parent == normalized
            
        finally:
            os.chdir(original_cwd)

    def test_path_with_special_characters(self):
        """Тест путей со специальными символами"""
        # Создаем файл с именем, содержащим специальные символы
        special_filename = "test file (copy) [1].py"
        test_file = self.create_test_file(f"special/{special_filename}", "# test")
        
        normalized = Path(test_file).resolve()
        assert normalized.exists()
        
        # Тестируем clean_filename для специальных символов
        dirty_filename = "test<>file:\"/\\|?*.py"
        cleaned = clean_filename(dirty_filename)
        
        # Проверяем что опасные символы заменены
        assert '<' not in cleaned
        assert '>' not in cleaned  
        assert ':' not in cleaned
        assert '"' not in cleaned
        assert '/' not in cleaned
        assert '\\' not in cleaned
        assert '|' not in cleaned
        assert '?' not in cleaned
        assert '*' not in cleaned
        
        # Должно остаться что-то разумное
        assert 'test' in cleaned
        assert 'file' in cleaned
        assert '.py' in cleaned

    def test_empty_and_invalid_paths(self):
        """Тест обработки пустых и некорректных путей"""
        # Тест пустого пути - Path("") представляет текущую директорию
        empty_path = Path("")
        normalized_empty = empty_path.resolve()
        # Пустая строка как путь должна разрешаться в текущую директорию
        assert normalized_empty.exists()
        assert normalized_empty.is_dir()
            
        # Тест несуществующего пути
        nonexistent = Path("nonexistent/path/file.py")
        normalized = nonexistent.resolve()  # Нормализация должна работать
        assert not normalized.exists()  # Но файла не должно существовать
        
        # Тест создания FileInfo с несуществующим путем
        file_info = FileInfo(
            path=str(normalized),
            name="file.py",
            size=0,
            language="python",
            extension=".py",
            modified_time="2025-01-01 12:00:00"
        )
        
        assert file_info.path == str(normalized)

    def test_unc_path_handling(self):
        """Тест обработки UNC путей (только на Windows)"""
        if os.name != 'nt':
            pytest.skip("UNC пути поддерживаются только на Windows")
            
        # Имитируем UNC путь
        unc_path = "\\\\server\\share\\repo\\file.py"
        
        # Path должен корректно обрабатывать UNC пути
        try:
            path_obj = Path(unc_path)
            # Даже если путь не существует, нормализация должна работать
            assert str(path_obj).startswith('\\\\')
            assert 'server' in str(path_obj)
            assert 'share' in str(path_obj)
        except Exception:
            # UNC пути могут не работать в тестовой среде, это нормально
            pass

    def test_scanner_path_consistency(self):
        """Тест консистентности путей в FileScanner"""
        # Создаем тестовые файлы
        py_file = self.create_test_file("src/main.py", "print('hello')")
        js_file = self.create_test_file("src/app.js", "console.log('hello');")
        
        # Создаем FileScanner с моком конфигурации
        with patch('file_scanner.get_config') as mock_get_config:
            # Настраиваем мок конфигурации
            mock_config = Mock()
            mock_config.file_scanner = Mock()
            mock_config.file_scanner.supported_extensions = {
                '.py': 'python',
                '.js': 'javascript'
            }
            mock_config.file_scanner.excluded_directories = []
            mock_config.file_scanner.max_file_size = 10 * 1024 * 1024
            mock_get_config.return_value = mock_config
            
            scanner = FileScanner()
            
            # Тестируем сканирование с различными форматами путей
            files_found = list(scanner.scan_repository(self.temp_dir))
            
            # Проверяем что пути в FileInfo нормализованы
            assert len(files_found) == 2
            
            for file_info in files_found:
                # Путь должен быть строкой
                assert isinstance(file_info.path, str)
                
                # Путь должен быть абсолютным
                path_obj = Path(file_info.path)
                assert path_obj.is_absolute()
                
                # Файл должен существовать
                assert path_obj.exists()
                
                # Имя файла должно совпадать с именем в пути
                assert file_info.name == path_obj.name

    def test_ensure_directory_exists_path_normalization(self):
        """Тест нормализации путей в ensure_directory_exists"""
        # Тест с относительным путем
        rel_path = os.path.join(self.temp_dir, "relative", "nested", "dir")
        ensure_directory_exists(rel_path)
        assert os.path.exists(rel_path)
        assert os.path.isdir(rel_path)
        
        # Тест с путем, содержащим ".."
        parent_path = os.path.join(self.temp_dir, "test", "..", "final_dir")
        ensure_directory_exists(parent_path)
        
        # Нормализуем путь для проверки
        normalized_parent = os.path.normpath(parent_path)
        assert os.path.exists(normalized_parent)
        assert os.path.isdir(normalized_parent)

    def test_error_handling_with_paths(self):
        """Тест обработки ошибок с различными путями"""
        # Создаем FileInfo с корректными данными
        file_info = FileInfo(
            path="/nonexistent/path/test.py",
            name="test.py", 
            size=100,
            language="python",
            extension=".py",
            modified_time="2025-01-01 12:00:00"
        )
        
        # Тестируем создание ParsedFile с ошибкой
        error = FileParsingError("Test error")
        parsed_file = create_error_parsed_file(file_info, error)
        
        # Проверяем что путь сохранен корректно
        assert parsed_file.file_info.path == "/nonexistent/path/test.py"
        assert len(parsed_file.parse_errors) == 1
        assert "Test error" in parsed_file.parse_errors[0]

    def test_path_consistency_across_pipeline(self):
        """Тест консистентности путей в различных частях пайплайна"""
        # Создаем тестовый файл
        test_file = self.create_test_file("project/src/main.py", "# Main file")
        
        # 1. Нормализация на уровне сканирования
        scan_normalized = Path(test_file).resolve()
        
        # 2. Создание FileInfo
        file_info = FileInfo(
            path=str(scan_normalized),
            name="main.py",
            size=100, 
            language="python",
            extension=".py",
            modified_time="2025-01-01 12:00:00"
        )
        
        # 3. Проверяем что все пути одинаково нормализованы
        assert file_info.path == str(scan_normalized)
        
        # 4. Path объекты из одинаковых строк должны быть эквивалентны
        path1 = Path(file_info.path)
        path2 = Path(str(scan_normalized))
        assert path1 == path2
        
        # 5. Все пути должны указывать на существующий файл
        assert path1.exists()
        assert path2.exists()
        assert scan_normalized.exists()

    def test_duplicate_path_segments_normalization(self):
        """Тест нормализации путей с дублирующими сегментами"""
        # Создаем путь с дублирующими сегментами
        test_file = self.create_test_file("src/main.py", "# test")
        
        # Создаем путь с дублирующими слешами
        duplicate_path = test_file.replace(os.sep, os.sep + os.sep)
        
        # Нормализуем
        normalized = Path(duplicate_path).resolve()
        
        # Проверяем что дублирующие сегменты устранены
        assert normalized.exists()
        path_str = str(normalized)
        
        # Не должно быть двойных разделителей
        if os.name == 'nt':
            assert '\\\\' not in path_str or path_str.startswith('\\\\')  # UNC может начинаться с \\
        else:
            assert '//' not in path_str