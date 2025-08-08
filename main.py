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
from utils import (
    setup_logging, FileInfo, ParsedFile, GPTAnalysisRequest, GPTAnalysisResult,
    ensure_directory_exists, create_error_parsed_file, create_error_gpt_result,
    compute_file_hash, read_index, write_index
)


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
        
    async def analyze_repository(self, repo_path: str, output_path: str, show_progress: bool = True, incremental: bool = True) -> dict:
        """Основной метод анализа репозитория"""
        self.logger.info(f"Начинаем анализ репозитория: {repo_path}")
        self.logger.info(f"Результат будет сохранен в: {output_path}")
        
        # Создаем выходную директорию
        ensure_directory_exists(output_path)
        
        # Сканируем файлы
        self.console.print("[bold blue]Сканирование файлов...[/bold blue]")
        all_files = list(self.file_scanner.scan_repository(repo_path))

        # Инкрементальный режим: отбираем только изменённые файлы
        files_to_analyze = all_files
        index_path = str(Path(repo_path) / ".repo_sum" / "index.json")
        if incremental:
            try:
                index = read_index(index_path)
                changed: List[FileInfo] = []
                for fi in all_files:
                    try:
                        h = compute_file_hash(fi.path)
                    except Exception:
                        # если не удаётся прочитать файл — считаем изменённым
                        changed.append(fi)
                        continue
                    prev = index.get(fi.path, {})
                    if prev.get("hash") != h:
                        changed.append(fi)
                if changed:
                    files_to_analyze = changed
                else:
                    self.console.print("[green]Нет изменений — отчёты актуальны[/green]")
                    return {
                        'total_files': 0,
                        'successful': 0,
                        'failed': 0,
                        'output_directory': str(Path(output_path) / f"SUMMARY_REPORT_{Path(repo_path).name}"),
                        'index_file': str(Path(output_path) / f"SUMMARY_REPORT_{Path(repo_path).name}" / "README.md"),
                        'success': True
                    }
            except Exception as e:
                self.logger.warning(f"Инкрементальный анализ отключен из-за ошибки индекса: {e}")
                files_to_analyze = all_files
        
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

        # Обновляем индекс для успешно обработанных файлов
        try:
            index = read_index(index_path)
            now = __import__('datetime').datetime.utcnow().isoformat()
            for parsed_file, _ in analyzed_files:
                try:
                    h = compute_file_hash(parsed_file.file_info.path)
                    index[parsed_file.file_info.path] = {"hash": h, "analyzed_at": now}
                except Exception:
                    continue
            write_index(index_path, index)
        except Exception as e:
            self.logger.warning(f"Не удалось обновить индекс изменений: {e}")
        
        # Показываем финальную статистику
        self._show_final_statistics(result)
        
        # Показываем статистику токенов
        token_stats = self.openai_manager.get_token_usage_stats()
        self._show_token_statistics(token_stats)
        
        return result
    
    async def _analyze_files_with_progress(self, files: List[FileInfo]) -> List[Tuple[ParsedFile, GPTAnalysisResult]]:
        """Анализирует файлы с отображением прогресса, используя батчевую обработку"""
        analyzed_files = []
        batch_size = self._get_optimal_batch_size(len(files))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Анализ файлов...", total=len(files))
            
            # Обрабатываем файлы батчами
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                
                # Обновляем описание для текущего батча
                batch_names = [Path(f.path).name for f in batch[:2]]  # Показываем первые 2 файла
                if len(batch) > 2:
                    batch_names.append(f"и еще {len(batch) - 2}")
                progress.update(task, description=f"Батч: {', '.join(batch_names)}")
                
                # Анализируем батч асинхронно
                batch_results = await self._analyze_files_batch(batch)
                analyzed_files.extend(batch_results)
                
                # Обновляем прогресс на количество обработанных файлов
                progress.advance(task, len(batch))
        
        return analyzed_files
    
    def _get_optimal_batch_size(self, total_files: int) -> int:
        """Определяет оптимальный размер батча в зависимости от количества файлов"""
        if total_files <= 10:
            return 2  # Маленькие проекты - небольшие батчи
        elif total_files <= 50:
            return 3  # Средние проекты
        elif total_files <= 200:
            return 5  # Большие проекты
        else:
            return 8  # Очень большие проекты
    
    async def _analyze_files_batch(self, batch: List[FileInfo]) -> List[Tuple[ParsedFile, GPTAnalysisResult]]:
        """Анализирует батч файлов параллельно"""
        # Создаем задачи для параллельного выполнения
        tasks = [self._analyze_single_file_safe(file_info) for file_info in batch]
        
        # Выполняем все задачи параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analyzed_files = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Обрабатываем исключение
                self.logger.error(f"Ошибка при анализе {batch[i].path}: {result}")
                error_parsed = create_error_parsed_file(batch[i], result)
                error_gpt = create_error_gpt_result(result) 
                analyzed_files.append((error_parsed, error_gpt))
            else:
                analyzed_files.append(result)
        
        return analyzed_files
    
    async def _analyze_single_file_safe(self, file_info: FileInfo) -> Tuple[ParsedFile, GPTAnalysisResult]:
        """Безопасно анализирует один файл с обработкой исключений"""
        try:
            return await self._analyze_single_file(file_info)
        except Exception as e:
            # Логируем ошибку, но не прерываем выполнение
            self.logger.error(f"Ошибка при анализе {file_info.path}: {e}")
            error_parsed = create_error_parsed_file(file_info, e)
            error_gpt = create_error_gpt_result(e)
            return (error_parsed, error_gpt)
    
    async def _analyze_files_simple(self, files: List[FileInfo]) -> List[Tuple[ParsedFile, GPTAnalysisResult]]:
        """Простой анализ файлов без прогресс-бара, тоже с батчевой обработкой"""
        analyzed_files = []
        batch_size = self._get_optimal_batch_size(len(files))
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_start = i + 1
            batch_end = min(i + batch_size, len(files))
            
            self.console.print(f"[dim]Анализ файлов {batch_start}-{batch_end}/{len(files)}[/dim]")
            
            # Используем ту же батчевую логику
            batch_results = await self._analyze_files_batch(batch)
            analyzed_files.extend(batch_results)
        
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
@click.option('--incremental/--no-incremental', default=True, help='Инкрементальный анализ только изменённых файлов')
def analyze(repo_path, output, no_progress, incremental):
    """Анализирует репозиторий и создает MD документацию."""
    analyzer = RepositoryAnalyzer()
    
    try:
        result = asyncio.run(analyzer.analyze_repository(
            repo_path, output, show_progress=not no_progress, incremental=incremental
        ))
        
        if result.get('success', True):
            console = Console()
            console.print(f"[bold green]Анализ завершен успешно![/bold green]")
            saved_dir = result.get('output_directory', output)
            console.print(f"Документация сохранена в: [cyan]{saved_dir}[/cyan]")
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
