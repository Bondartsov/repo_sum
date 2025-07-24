#!/usr/bin/env python3
"""
Главный модуль анализатора репозиториев с генерацией MD документации через OpenAI GPT.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import click
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table

from config import get_config, reload_config
from file_scanner import FileScanner
from parsers.base_parser import ParserRegistry
from code_chunker import CodeChunker
from openai_integration import OpenAIManager
from doc_generator import DocumentationGenerator
from utils import setup_logging, FileInfo, ParsedFile, GPTAnalysisRequest, GPTAnalysisResult, ensure_directory_exists


class RepositoryAnalyzer:
    """Главный класс анализатора репозиториев"""
    
    def __init__(self):
        self.config = get_config()
        self.console = Console()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Инициализируем компоненты
        self.file_scanner = FileScanner()
        self.parser_registry = ParserRegistry()
        self.code_chunker = CodeChunker()
        self.openai_manager = OpenAIManager()
        self.doc_generator = DocumentationGenerator()
        
    async def analyze_repository(self, repo_path: str, output_path: str, show_progress: bool = True) -> dict:
        """Основной метод анализа репозитория"""
        self.logger.info(f"Начинаем анализ репозитория: {repo_path}")
        self.logger.info(f"Результат будет сохранен в: {output_path}")
        
        # Создаем выходную директорию
        ensure_directory_exists(output_path)
        
        # Сканируем файлы
        self.console.print("[bold blue]Сканирование файлов...[/bold blue]")
        files_to_analyze = list(self.file_scanner.scan_repository(repo_path))
        
        if not files_to_analyze:
            self.console.print("[bold red]Не найдено файлов для анализа![/bold red]")
            return {'success': False, 'error': 'Не найдено файлов для анализа'}
        
        self.console.print(f"[green]Найдено {len(files_to_analyze)} файлов для анализа[/green]")
        
        # Отображаем статистику по языкам
        self._show_language_statistics(files_to_analyze)
        
        # Анализируем файлы с прогресс-баром
        if show_progress:
            analyzed_files = await self._analyze_files_with_progress(files_to_analyze)
        else:
            analyzed_files = await self._analyze_files_simple(files_to_analyze)
        
        # Генерируем документацию
        self.console.print("[bold blue]Генерация документации...[/bold blue]")
        result = self.doc_generator.generate_complete_documentation(
            analyzed_files, output_path, repo_path
        )
        
        # Показываем финальную статистику
        self._show_final_statistics(result)
        
        # Показываем статистику токенов
        token_stats = self.openai_manager.get_token_usage_stats()
        self._show_token_statistics(token_stats)
        
        return result
    
    async def _analyze_files_with_progress(self, files: List[FileInfo]) -> List[Tuple[ParsedFile, GPTAnalysisResult]]:
        """Анализирует файлы с отображением прогресса"""
        analyzed_files = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Анализ файлов...", total=len(files))
            
            for file_info in files:
                try:
                    # Обновляем описание задачи
                    progress.update(task, description=f"Анализ: {Path(file_info.path).name}")
                    
                    # Анализируем файл
                    parsed_file, gpt_result = await self._analyze_single_file(file_info)
                    analyzed_files.append((parsed_file, gpt_result))
                    
                    progress.advance(task)
                    
                except Exception as e:
                    self.logger.error(f"Ошибка при анализе {file_info.path}: {e}")
                    # Создаем пустой результат для файла с ошибкой
                    empty_parsed = ParsedFile(file_info, [], [], [], [str(e)])
                    empty_gpt = GPTAnalysisResult("", [], {}, f"Ошибка анализа: {e}")
                    analyzed_files.append((empty_parsed, empty_gpt))
                    progress.advance(task)
        
        return analyzed_files
    
    async def _analyze_files_simple(self, files: List[FileInfo]) -> List[Tuple[ParsedFile, GPTAnalysisResult]]:
        """Простой анализ файлов без прогресс-бара"""
        analyzed_files = []
        
        for i, file_info in enumerate(files, 1):
            self.console.print(f"[dim]Анализ {i}/{len(files)}: {Path(file_info.path).name}[/dim]")
            
            try:
                parsed_file, gpt_result = await self._analyze_single_file(file_info)
                analyzed_files.append((parsed_file, gpt_result))
            except Exception as e:
                self.logger.error(f"Ошибка при анализе {file_info.path}: {e}")
                empty_parsed = ParsedFile(file_info, [], [], [], [str(e)])
                empty_gpt = GPTAnalysisResult("", [], {}, f"Ошибка анализа: {e}")
                analyzed_files.append((empty_parsed, empty_gpt))
        
        return analyzed_files
    
    async def _analyze_single_file(self, file_info: FileInfo) -> Tuple[ParsedFile, GPTAnalysisResult]:
        """Анализирует один файл"""
        # Получаем парсер
        parser = self.parser_registry.get_parser(file_info.path)
        if not parser:
            raise ValueError(f"Не найден парсер для файла {file_info.path}")
        
        # Парсим файл
        parsed_file = parser.safe_parse(file_info)
        
        # Разбиваем на чанки
        chunks = self.code_chunker.chunk_parsed_file(parsed_file)
        
        # Создаем запрос к GPT
        gpt_request = GPTAnalysisRequest(
            file_path=file_info.path,
            language=file_info.language,
            chunks=chunks
        )
        
        # Анализируем через GPT
        gpt_result = await self.openai_manager.analyze_code(gpt_request)
        
        return parsed_file, gpt_result
    
    def _show_language_statistics(self, files: List[FileInfo]) -> None:
        """Показывает статистику по языкам программирования"""
        languages = {}
        total_size = 0
        
        for file_info in files:
            lang = file_info.language
            languages[lang] = languages.get(lang, {'count': 0, 'size': 0})
            languages[lang]['count'] += 1
            languages[lang]['size'] += file_info.size
            total_size += file_info.size
        
        table = Table(title="Статистика по языкам")
        table.add_column("Язык", style="cyan", no_wrap=True)
        table.add_column("Файлов", justify="right", style="green")
        table.add_column("Размер", justify="right", style="yellow")
        table.add_column("%", justify="right", style="blue")
        
        for lang in sorted(languages.keys()):
            stats = languages[lang]
            size_str = self._format_size(stats['size'])
            percentage = (stats['size'] / total_size * 100) if total_size > 0 else 0
            
            table.add_row(
                lang.title(),
                str(stats['count']),
                size_str,
                f"{percentage:.1f}%"
            )
        
        self.console.print(table)
    
    def _show_final_statistics(self, result: dict) -> None:
        """Показывает финальную статистику генерации"""
        table = Table(title="Результат анализа")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", justify="right", style="green")
        
        table.add_row("Всего файлов", str(result.get('total_files', 0)))
        table.add_row("Успешно проанализировано", str(result.get('successful', 0)))
        
        if result.get('failed', 0) > 0:
            table.add_row("С ошибками", str(result['failed']), style="red")
        
        table.add_row("Директория вывода", str(result.get('output_directory', '')))
        
        self.console.print(table)
    
    def _show_token_statistics(self, token_stats: dict) -> None:
        """Показывает статистику использования токенов"""
        table = Table(title="Использование токенов OpenAI")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", justify="right", style="yellow")
        
        table.add_row("Использовано сегодня", str(token_stats.get('used_today', 0)))
        
        self.console.print(table)
    
    def _format_size(self, size_bytes: int) -> str:
        """Форматирует размер файла"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"


# CLI команды
@click.group()
@click.option('--config', '-c', default='settings.json', help='Путь к файлу конфигурации')
@click.option('--verbose', '-v', is_flag=True, help='Подробный вывод')
@click.option('--quiet', '-q', is_flag=True, help='Тихий режим')
@click.pass_context
def cli(ctx, config, verbose, quiet):
    """Анализатор репозиториев с генерацией MD документации через OpenAI GPT."""
    ctx.ensure_object(dict)
    
    # Настраиваем логирование
    if quiet:
        log_level = "ERROR"
    elif verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    setup_logging(log_level)
    
    # Загружаем конфигурацию
    try:
        if config != 'settings.json':
            reload_config(config)
        else:
            get_config()  # Загружаем стандартную конфигурацию
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Ошибка загрузки конфигурации: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='./docs', help='Директория для сохранения документации')
@click.option('--no-progress', is_flag=True, help='Отключить прогресс-бар')
def analyze(repo_path, output, no_progress):
    """Анализирует репозиторий и создает MD документацию."""
    analyzer = RepositoryAnalyzer()
    
    try:
        result = asyncio.run(analyzer.analyze_repository(
            repo_path, output, show_progress=not no_progress
        ))
        
        if result.get('success', True):
            console = Console()
            console.print(f"[bold green]Анализ завершен успешно![/bold green]")
            console.print(f"Документация сохранена в: [cyan]{output}[/cyan]")
            if result.get('index_file'):
                console.print(f"Главный файл: [cyan]{result['index_file']}[/cyan]")
        else:
            console = Console()
            console.print(f"[bold red]Ошибка: {result.get('error', 'Неизвестная ошибка')}[/bold red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console = Console()
        console.print("[yellow]Анализ прерван пользователем[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        console = Console()
        console.print(f"[bold red]Критическая ошибка: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
def stats(repo_path):
    """Показывает статистику репозитория без анализа."""
    console = Console()
    console.print(f"[bold blue]Сбор статистики для: {repo_path}[/bold blue]")
    
    try:
        scanner = FileScanner()
        stats = scanner.get_repository_stats(repo_path)
        
        # Общая статистика
        table = Table(title="Общая статистика")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", justify="right", style="green")
        
        table.add_row("Всего файлов", str(stats['total_files']))
        table.add_row("Общий размер", RepositoryAnalyzer()._format_size(stats['total_size']))
        
        console.print(table)
        
        # Статистика по языкам
        if stats['languages']:
            lang_table = Table(title="По языкам программирования")
            lang_table.add_column("Язык", style="cyan")
            lang_table.add_column("Файлов", justify="right", style="green")
            
            for lang, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True):
                lang_table.add_row(lang.title(), str(count))
            
            console.print(lang_table)
        
        # Самые большие файлы
        if stats['largest_files']:
            large_table = Table(title="Самые большие файлы")
            large_table.add_column("Файл", style="cyan")
            large_table.add_column("Размер", justify="right", style="yellow")
            large_table.add_column("Язык", style="green")
            
            for file_info in stats['largest_files'][:10]:
                large_table.add_row(
                    str(Path(file_info['path']).name),
                    RepositoryAnalyzer()._format_size(file_info['size']),
                    file_info['language'].title()
                )
            
            console.print(large_table)
            
    except Exception as e:
        console.print(f"[bold red]Ошибка при сборе статистики: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def clear_cache():
    """Очищает кэш OpenAI запросов."""
    try:
        manager = OpenAIManager()
        cleared = manager.clear_cache()
        
        console = Console()
        console.print(f"[green]Очищено {cleared} записей кэша[/green]")
        
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Ошибка при очистке кэша: {e}[/bold red]")


@cli.command()
def token_stats():
    """Показывает статистику использования токенов OpenAI."""
    try:
        manager = OpenAIManager()
        stats = manager.get_token_usage_stats()
        
        console = Console()
        table = Table(title="Статистика токенов OpenAI")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", justify="right", style="green")
        
        table.add_row("Использовано сегодня", str(stats['used_today']))
        
        console.print(table)
        
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Ошибка при получении статистики: {e}[/bold red]")


if __name__ == '__main__':
    cli()
