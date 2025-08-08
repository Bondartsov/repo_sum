"""
Модуль для сканирования файловой системы и поиска файлов кода.
"""

import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional, Set

import chardet

from config import get_config
from utils import FileInfo, format_file_size


class FileScanner:
    """Класс для рекурсивного сканирования репозитория"""
    
    def __init__(self):
        self.config = get_config()
        self.scanner_config = self.config.file_scanner
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.supported_extensions = self.scanner_config.supported_extensions
        self.excluded_directories = set(self.scanner_config.excluded_directories)
        self.max_file_size = self.scanner_config.max_file_size
        
    def scan_repository(self, repo_path: str) -> Iterator[FileInfo]:
        """Основной метод сканирования репозитория"""
        repo_path = Path(repo_path).resolve()
        
        if not repo_path.exists():
            raise FileNotFoundError(f"Путь не найден: {repo_path}")
        
        if not repo_path.is_dir():
            raise ValueError(f"Путь не является директорией: {repo_path}")
        
        self.logger.info(f"Начинаем сканирование репозитория: {repo_path}")
        
        file_count = 0
        total_size = 0
        
        for file_path in self._walk_directory(repo_path):
            try:
                file_info = self._create_file_info(file_path)
                if file_info:
                    file_count += 1
                    total_size += file_info.size
                    self.logger.debug(f"Найден файл: {file_info.path} ({file_info.language})")
                    yield file_info
                    
            except Exception as e:
                self.logger.warning(f"Ошибка при обработке файла {file_path}: {e}")
                continue
        
        self.logger.info(f"Сканирование завершено. Найдено файлов: {file_count}, "
                        f"общий размер: {format_file_size(total_size)}")
    
    def _walk_directory(self, root_path: Path) -> Iterator[Path]:
        """Рекурсивно обходит директорию с фильтрацией"""
        try:
            for item in root_path.iterdir():
                if item.is_file():
                    if self._should_include_file(item):
                        yield item
                elif item.is_dir():
                    if not self._is_excluded_directory(item):
                        yield from self._walk_directory(item)
                        
        except PermissionError:
            self.logger.warning(f"Нет доступа к директории: {root_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при обходе директории {root_path}: {e}")
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Проверяет, нужно ли включить файл в анализ"""
        # Проверяем расширение
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Проверяем размер файла
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                self.logger.debug(f"Файл слишком большой: {file_path} "
                                f"({format_file_size(file_size)})")
                return False
        except OSError:
            return False
        
        # Исключаем скрытые файлы (начинающиеся с точки)
        if file_path.name.startswith('.'):
            return False
        
        # Исключаем библиотечные и сторонние файлы
        if self._is_library_file(file_path):
            return False
        
        return True
    
    def _is_excluded_directory(self, dir_path: Path) -> bool:
        """Проверяет, нужно ли исключить директорию"""
        dir_name = dir_path.name.lower()
        
        # Проверяем по точному совпадению имени
        if dir_name in self.excluded_directories:
            return True
        
        # Исключаем скрытые директории (начинающиеся с точки)
        if dir_name.startswith('.'):
            return True
        
        return False
    
    def _is_library_file(self, file_path: Path) -> bool:
        """Проверяет, является ли файл библиотечным/сторонним.
        Более консервативный фильтр, чтобы не исключать пользовательский код по случайным совпадениям.
        """
        path_parts = [p.lower() for p in file_path.parts]
        path_str = str(file_path).lower()

        # 1) Явные директории стороннего кода
        vendor_dirs = {
            'third_party', 'third-party', 'thirdparty',
            'external', 'vendor', 'vendors',
            'node_modules', 'bower_components'
        }
        if any(part in vendor_dirs for part in path_parts):
            return True

        # 2) Служебные каталоги сборки/генерации
        build_dirs = {'cmake-build-debug', 'cmake-build-release', 'build', 'dist', 'generated'}
        if any(part in build_dirs for part in path_parts):
            return True

        # 3) Специфичные include‑пути системных заголовков
        # Проверяем только паттерны include/...
        if 'include' in path_parts:
            joined = '/'.join(path_parts)
            if any(seg in joined for seg in ('include/std', 'include/bits', 'include/sys', 'include/linux')):
                return True

        # 4) Генерированные/автосгенерированные файлы по названию
        file_name = file_path.name.lower()
        generated_patterns = (
            '.pb.', '.proto.', '_pb2.', '_pb2_grpc.',
            '.generated.', '.gen.', 'generated_',
            '.moc.', '.ui.', '.qrc.',
            'moc_', 'ui_', 'qrc_'
        )
        if any(pat in file_name for pat in generated_patterns):
            return True

        # 5) Не отфильтровываем по общим словам ("json", "log", "lib" и т.п.) — слишком много ложных срабатываний
        return False
    
    def _create_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """Создает объект FileInfo для файла"""
        try:
            stat = file_path.stat()
            
            # Определяем язык программирования
            language = self._get_file_language(file_path)
            if not language:
                return None
            
            # Определяем кодировку
            encoding = self._detect_encoding(file_path)
            
            # Форматируем время модификации
            from datetime import datetime
            modified_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            return FileInfo(
                path=str(file_path),
                name=file_path.name,
                size=stat.st_size,
                language=language,
                extension=file_path.suffix.lower(),
                modified_time=modified_time,
                encoding=encoding
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании FileInfo для {file_path}: {e}")
            return None
    
    def _get_file_language(self, file_path: Path) -> Optional[str]:
        """Определяет язык программирования по расширению"""
        extension = file_path.suffix.lower()
        return self.supported_extensions.get(extension)
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Определяет кодировку файла.
        Стратегия:
        1. Попытка чтения как UTF-8
        2. Использование библиотеки chardet
        3. Fallback на latin-1
        """
        # Сначала пробуем UTF-8
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Читаем первые 1KB для проверки
            return 'utf-8'
        except UnicodeDecodeError:
            pass
        
        # Используем chardet для автоматического определения
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(min(8192, file_path.stat().st_size))  # Читаем до 8KB
            
            result = chardet.detect(raw_data)
            if result and result['confidence'] > 0.7:
                encoding = result['encoding'].lower()
                # Нормализуем название кодировки
                if 'utf-8' in encoding or 'utf8' in encoding:
                    return 'utf-8'
                elif 'windows-1251' in encoding or 'cp1251' in encoding:
                    return 'cp1251'
                elif 'iso-8859-1' in encoding or 'latin-1' in encoding:
                    return 'latin-1'
                else:
                    return result['encoding']
                    
        except Exception as e:
            self.logger.debug(f"Ошибка определения кодировки для {file_path}: {e}")
        
        # Fallback стратегия - пробуем популярные кодировки
        encodings_to_try = ['utf-8', 'cp1251', 'latin-1']
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)
                self.logger.debug(f"Использована кодировка {encoding} для {file_path}")
                return encoding
            except UnicodeDecodeError:
                continue
        
        # Последний шанс - latin-1 всегда работает
        return 'latin-1'
    
    def get_repository_stats(self, repo_path: str) -> dict:
        """Получает статистику репозитория"""
        stats = {
            'total_files': 0,
            'total_size': 0,
            'languages': {},
            'largest_files': [],
            'encoding_distribution': {}
        }
        
        files_info = []
        
        for file_info in self.scan_repository(repo_path):
            stats['total_files'] += 1
            stats['total_size'] += file_info.size
            
            # Статистика по языкам
            if file_info.language in stats['languages']:
                stats['languages'][file_info.language] += 1
            else:
                stats['languages'][file_info.language] = 1
            
            # Статистика по кодировкам
            if file_info.encoding in stats['encoding_distribution']:
                stats['encoding_distribution'][file_info.encoding] += 1
            else:
                stats['encoding_distribution'][file_info.encoding] = 1
            
            files_info.append(file_info)
        
        # Топ-10 самых больших файлов
        files_info.sort(key=lambda x: x.size, reverse=True)
        stats['largest_files'] = [
            {
                'path': f.path,
                'size': f.size,
                'language': f.language
            }
            for f in files_info[:10]
        ]
        
        return stats
    
    def count_files(self, repo_path: str) -> int:
        """Быстрый подсчет количества файлов для анализа"""
        count = 0
        for _ in self.scan_repository(repo_path):
            count += 1
        return count
