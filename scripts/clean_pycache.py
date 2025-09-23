#!/usr/bin/env python3
"""
Скрипт для рекурсивного поиска и удаления папок __pycache__ в Python проектах.

Использование:
    python clean_pycache.py [опции] [путь]

Опции:
    --dry-run       Показать что будет удалено без реального удаления
    --interactive   Запрашивать подтверждение для каждой папки
    --quiet         Минимальный вывод (только ошибки и итоговая статистика)
    --no-color      Отключить цветной вывод
    --help          Показать эту справку

Если путь не указан, используется текущая директория.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI цветовые коды для терминала"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class PyCacheManager:
    """Менеджер для поиска и удаления папок __pycache__"""
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.found_dirs = []
        self.removed_dirs = []
        self.failed_dirs = []
    
    def _colorize(self, text: str, color: str) -> str:
        """Применить цвет к тексту если включены цвета"""
        if not self.use_colors:
            return text
        return f"{color}{text}{Colors.RESET}"
    
    def find_pycache_directories(self, root_path: Path) -> List[Path]:
        """
        Рекурсивный поиск всех папок __pycache__
        
        Args:
            root_path: Корневая директория для поиска
            
        Returns:
            Список путей к найденным папкам __pycache__
        """
        pycache_dirs = []
        
        try:
            for root, dirs, files in os.walk(root_path):
                # Ищем папки __pycache__ в текущей директории
                if '__pycache__' in dirs:
                    pycache_path = Path(root) / '__pycache__'
                    pycache_dirs.append(pycache_path)
                    
        except PermissionError as e:
            print(self._colorize(f"⚠️  Нет доступа к директории: {e}", Colors.YELLOW))
        except Exception as e:
            print(self._colorize(f"❌ Ошибка при поиске: {e}", Colors.RED))
        
        self.found_dirs = pycache_dirs
        return pycache_dirs
    
    def remove_directory(self, dir_path: Path, interactive: bool = False) -> bool:
        """
        Безопасное удаление директории __pycache__
        
        Args:
            dir_path: Путь к директории для удаления
            interactive: Запрашивать подтверждение
            
        Returns:
            True если удаление прошло успешно, False если нет
        """
        try:
            if interactive:
                response = input(f"Удалить {dir_path}? [y/N]: ").strip().lower()
                if response not in ['y', 'yes', 'да', 'д']:
                    print(self._colorize("  ⏭️  Пропущено", Colors.YELLOW))
                    return False
            
            # Дополнительная проверка безопасности
            if dir_path.name != '__pycache__':
                print(self._colorize(f"⚠️  ВНИМАНИЕ: '{dir_path.name}' не является папкой __pycache__!", Colors.RED))
                return False
            
            # Удаляем директорию и все содержимое
            shutil.rmtree(dir_path)
            self.removed_dirs.append(dir_path)
            return True
            
        except PermissionError:
            error_msg = f"❌ Нет прав на удаление: {dir_path}"
            print(self._colorize(error_msg, Colors.RED))
            self.failed_dirs.append((dir_path, "Permission denied"))
            return False
            
        except FileNotFoundError:
            # Папка уже была удалена
            print(self._colorize(f"⚠️  Папка уже не существует: {dir_path}", Colors.YELLOW))
            return False
            
        except Exception as e:
            error_msg = f"❌ Ошибка при удалении {dir_path}: {e}"
            print(self._colorize(error_msg, Colors.RED))
            self.failed_dirs.append((dir_path, str(e)))
            return False
    
    def clean_pycache(self, root_path: Path, dry_run: bool = False, 
                     interactive: bool = False, quiet: bool = False) -> Tuple[int, int, int]:
        """
        Основная функция очистки __pycache__ папок
        
        Args:
            root_path: Корневая директория
            dry_run: Только показать что будет удалено
            interactive: Запрашивать подтверждение
            quiet: Минимальный вывод
            
        Returns:
            Кортеж (найдено, удалено, ошибок)
        """
        if not quiet:
            print(self._colorize(f"🔍 Поиск папок __pycache__ в: {root_path}", Colors.CYAN))
        
        # Поиск всех папок __pycache__
        pycache_dirs = self.find_pycache_directories(root_path)
        
        if not pycache_dirs:
            if not quiet:
                print(self._colorize("✨ Папки __pycache__ не найдены!", Colors.GREEN))
            return 0, 0, 0
        
        # Показать найденные папки
        if not quiet:
            print(self._colorize(f"\n📁 Найдено {len(pycache_dirs)} папок __pycache__:", Colors.BLUE))
            for i, dir_path in enumerate(pycache_dirs, 1):
                relative_path = dir_path.relative_to(root_path)
                print(f"  {i:2d}. {relative_path}")
        
        if dry_run:
            if not quiet:
                print(self._colorize(f"\n🧪 DRY RUN: Было бы удалено {len(pycache_dirs)} папок", Colors.MAGENTA))
            return len(pycache_dirs), 0, 0
        
        # Удаление папок
        if not quiet:
            print(self._colorize(f"\n🗑️  Начинаю удаление...", Colors.YELLOW))
        
        removed_count = 0
        for i, dir_path in enumerate(pycache_dirs, 1):
            if not quiet:
                relative_path = dir_path.relative_to(root_path)
                print(f"  {i:2d}/{len(pycache_dirs)} {relative_path} ... ", end="")
                
            if self.remove_directory(dir_path, interactive):
                if not quiet:
                    print(self._colorize("✅ Удалено", Colors.GREEN))
                removed_count += 1
            elif not quiet:
                print()  # Новая строка после сообщения об ошибке
        
        return len(pycache_dirs), removed_count, len(self.failed_dirs)
    
    def print_summary(self, found: int, removed: int, failed: int, quiet: bool = False):
        """Вывести итоговую статистику"""
        if quiet and found == 0:
            return
            
        print(self._colorize("\n" + "="*50, Colors.BOLD))
        print(self._colorize("📊 ИТОГОВАЯ СТАТИСТИКА", Colors.BOLD))
        print(self._colorize("="*50, Colors.BOLD))
        
        print(f"📁 Найдено папок:     {self._colorize(str(found), Colors.BLUE)}")
        print(f"✅ Удалено успешно:   {self._colorize(str(removed), Colors.GREEN)}")
        
        if failed > 0:
            print(f"❌ Ошибок удаления:   {self._colorize(str(failed), Colors.RED)}")
            
            if not quiet and self.failed_dirs:
                print(self._colorize("\n🚨 Ошибки удаления:", Colors.RED))
                for dir_path, error in self.failed_dirs:
                    print(f"  • {dir_path}: {error}")
        
        if removed > 0:
            freed_space = self._estimate_freed_space()
            if freed_space:
                print(f"💾 Примерно освобождено: {self._colorize(freed_space, Colors.MAGENTA)}")
        
        print(self._colorize("="*50 + "\n", Colors.BOLD))
    
    def _estimate_freed_space(self) -> str:
        """Примерная оценка освобожденного места"""
        # Очень приблизительная оценка: каждая папка __pycache__ ~ 100KB - 2MB
        removed_count = len(self.removed_dirs)
        if removed_count == 0:
            return ""
        
        # Консервативная оценка в 500KB на папку
        estimated_kb = removed_count * 500
        
        if estimated_kb < 1024:
            return f"{estimated_kb} KB"
        elif estimated_kb < 1024 * 1024:
            return f"{estimated_kb / 1024:.1f} MB"
        else:
            return f"{estimated_kb / (1024 * 1024):.1f} GB"


def create_argument_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Рекурсивно найти и удалить все папки __pycache__ в Python проекте.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                          # Очистить текущую директорию
  %(prog)s /path/to/project         # Очистить указанную директорию
  %(prog)s --dry-run                # Показать что будет удалено
  %(prog)s --interactive            # Запрашивать подтверждение
  %(prog)s --quiet /path/to/project # Минимальный вывод
        """
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Путь к директории для очистки (по умолчанию: текущая директория)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Показать какие папки будут удалены, но не удалять их'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Запрашивать подтверждение перед удалением каждой папки'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Минимальный вывод (только ошибки и итоговая статистика)'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Отключить цветной вывод'
    )
    
    return parser


def main():
    """Основная функция"""
    try:
        # Парсинг аргументов
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Проверка пути
        root_path = Path(args.path).resolve()
        if not root_path.exists():
            print(f"❌ Ошибка: Путь '{args.path}' не существует!", file=sys.stderr)
            sys.exit(1)
        
        if not root_path.is_dir():
            print(f"❌ Ошибка: '{args.path}' не является директорией!", file=sys.stderr)
            sys.exit(1)
        
        # Определить использование цветов
        use_colors = not args.no_color and (
            sys.stdout.isatty() and os.getenv('TERM') != 'dumb'
        )
        
        # Создать менеджер и выполнить очистку
        manager = PyCacheManager(use_colors=use_colors)
        
        if not args.quiet:
            print(manager._colorize("🧹 Python __pycache__ Cleaner", Colors.BOLD + Colors.CYAN))
            print(manager._colorize("="*50, Colors.CYAN))
        
        found, removed, failed = manager.clean_pycache(
            root_path,
            dry_run=args.dry_run,
            interactive=args.interactive,
            quiet=args.quiet
        )
        
        # Показать итоговую статистику
        manager.print_summary(found, removed, failed, args.quiet)
        
        # Код возврата: 0 если все OK, 1 если были ошибки
        sys.exit(1 if failed > 0 else 0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
