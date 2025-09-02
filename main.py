#!/usr/bin/env python3
"""
Главный модуль анализатора репозиториев с генерацией MD документации через OpenAI GPT.
Теперь включает RAG систему для семантического поиска по коду.
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

# RAG система
from rag.indexer_service import IndexerService
from rag.search_service import SearchService
from rag.exceptions import VectorStoreException, VectorStoreConnectionError


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
    console = Console()
    
    try:
        # Fail-fast: проверяем что все компоненты могут быть инициализированы
        analyzer = RepositoryAnalyzer()
    except ValueError as e:
        # Быстрый выход при ошибках конфигурации (например, отсутствие API ключа)
        console.print(f"[bold red]Ошибка конфигурации: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        # Любая другая критическая ошибка инициализации
        logger = logging.getLogger(__name__)
        logger.error(f"Критическая ошибка инициализации: {e}", exc_info=True)
        console.print(f"[bold red]Критическая ошибка инициализации: {e}[/bold red]")
        sys.exit(1)
    
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


# RAG команды
@cli.group()
def rag():
    """Команды RAG системы для семантического поиска по коду"""
    pass


@rag.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--batch-size', default=512, help='Размер батча для индексации')
@click.option('--recreate', is_flag=True, help='Пересоздать коллекцию')
@click.option('--no-progress', is_flag=True, help='Отключить прогресс-бар')
def index(repo_path, batch_size, recreate, no_progress):
    """Индексация репозитория в векторную БД"""
    console = Console()
    
    try:
        # Получаем конфигурацию с проверкой RAG настроек
        config = get_config()
        
        # Валидируем RAG конфигурацию
        try:
            config.validate(require_api_key=False)
        except ValueError as e:
            console.print(f"[bold red]Ошибка конфигурации RAG: {e}[/bold red]")
            sys.exit(1)
        
        console.print(f"[bold blue]🔄 Индексация репозитория: {repo_path}[/bold blue]")
        console.print(f"[dim]Qdrant: {config.rag.vector_store.host}:{config.rag.vector_store.port}[/dim]")
        console.print(f"[dim]Коллекция: {config.rag.vector_store.collection_name}[/dim]")
        console.print(f"[dim]Размер батча: {batch_size}[/dim]")
        
        if recreate:
            console.print("[yellow]⚠️  Коллекция будет пересоздана[/yellow]")
        
        # Создаем сервис индексации
        indexer = IndexerService(config)
        
        # Запускаем индексацию
        result = asyncio.run(indexer.index_repository(
            repo_path=repo_path,
            batch_size=batch_size,
            recreate=recreate,
            show_progress=not no_progress
        ))
        
        if result.get('success', False):
            # Показываем результат индексации
            table = Table(title="Результат индексации")
            table.add_column("Метрика", style="cyan")
            table.add_column("Значение", justify="right", style="green")
            
            table.add_row("Всего файлов", str(result.get('total_files', 0)))
            table.add_row("Обработано файлов", str(result.get('processed_files', 0)))
            table.add_row("Всего чанков", str(result.get('total_chunks', 0)))
            table.add_row("Проиндексировано чанков", str(result.get('indexed_chunks', 0)))
            table.add_row("Время выполнения", f"{result.get('total_time', 0):.1f}s")
            table.add_row("Скорость обработки", f"{result.get('processing_rate', 0):.1f} файлов/с")
            table.add_row("Скорость индексации", f"{result.get('indexing_rate', 0):.1f} чанков/с")
            
            if result.get('failed_files', 0) > 0:
                table.add_row("С ошибками", str(result['failed_files']), style="red")
            
            console.print(table)
            console.print(f"[bold green]✅ Индексация завершена успешно![/bold green]")
        else:
            console.print(f"[bold red]❌ Ошибка индексации: {result.get('error', 'Неизвестная ошибка')}[/bold red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("[yellow]⏹️ Индексация прервана пользователем[/yellow]")
        sys.exit(1)
    except VectorStoreConnectionError as e:
        console.print(f"[bold red]❌ Ошибка подключения к Qdrant: {e}[/bold red]")
        console.print("[dim]Проверьте, что Qdrant запущен и доступен[/dim]")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Критическая ошибка индексации: {e}", exc_info=True)
        console.print(f"[bold red]❌ Критическая ошибка: {e}[/bold red]")
        sys.exit(1)


@rag.command()
@click.argument('query')
@click.option('--top-k', default=10, help='Количество результатов')
@click.option('--lang', help='Фильтр по языку программирования')
@click.option('--chunk-type', help='Фильтр по типу чанка (class, function, etc.)')
@click.option('--min-score', type=float, help='Минимальный порог релевантности (0.0-1.0)')
@click.option('--file-path', help='Фильтр по пути к файлу')
@click.option('--no-content', is_flag=True, help='Не показывать содержимое')
@click.option('--max-lines', default=10, help='Максимальное количество строк контента')
def search(query, top_k, lang, chunk_type, min_score, file_path, no_content, max_lines):
    """Семантический поиск по коду"""
    console = Console()
    
    try:
        # Получаем конфигурацию
        config = get_config()
        
        console.print(f"[bold blue]🔍 Поиск: '{query}'[/bold blue]")
        console.print(f"[dim]Результатов: {top_k}, Фильтры: язык={lang or 'все'}, тип={chunk_type or 'все'}[/dim]")
        
        # Создаем сервис поиска
        searcher = SearchService(config)
        
        # Выполняем поиск
        results = asyncio.run(searcher.search(
            query=query,
            top_k=top_k,
            language_filter=lang,
            chunk_type_filter=chunk_type,
            min_score=min_score,
            file_path_filter=file_path
        ))
        
        # Форматируем и выводим результаты
        searcher.format_search_results(
            results=results,
            show_content=not no_content,
            max_content_lines=max_lines
        )
        
        if results:
            console.print()
            console.print(f"[dim]Поиск завершен. Средний скор: {sum(r.score for r in results) / len(results):.3f}[/dim]")
        
    except KeyboardInterrupt:
        console.print("[yellow]⏹️ Поиск прерван пользователем[/yellow]")
        sys.exit(1)
    except VectorStoreConnectionError as e:
        console.print(f"[bold red]❌ Ошибка подключения к Qdrant: {e}[/bold red]")
        console.print("[dim]Проверьте, что Qdrant запущен и коллекция проиндексирована[/dim]")
        sys.exit(1)
    except VectorStoreException as e:
        console.print(f"[bold red]❌ Ошибка поиска: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Критическая ошибка поиска: {e}", exc_info=True)
        console.print(f"[bold red]❌ Критическая ошибка: {e}[/bold red]")
        sys.exit(1)


@rag.command()
@click.option('--detailed', is_flag=True, help='Подробная статистика')
def status(detailed):
    """Статус RAG системы и векторной БД"""
    console = Console()
    
    try:
        config = get_config()
        
        console.print("[bold blue]📊 Статус RAG системы[/bold blue]")
        
        # Health check индексера
        indexer = IndexerService(config)
        health = asyncio.run(indexer.health_check())
        
        # Общий статус
        status_color = "green" if health['status'] == 'healthy' else "yellow" if health['status'] == 'degraded' else "red"
        console.print(f"[bold]Общий статус: [{status_color}]{health['status'].upper()}[/{status_color}][/bold]")
        console.print()
        
        # Статус компонентов
        components_table = Table(title="Статус компонентов")
        components_table.add_column("Компонент", style="cyan")
        components_table.add_column("Статус", style="bold")
        components_table.add_column("Детали", style="dim")
        
        # Vector Store
        vs_health = health['components'].get('vector_store', {})
        vs_status = vs_health.get('status', 'unknown')
        vs_status_color = "green" if vs_status == 'connected' else "red"
        
        vs_details = ""
        if vs_health.get('collection_info'):
            coll_info = vs_health['collection_info']
            vs_details = f"Документов: {coll_info.get('points_count', 0)}, Проиндексировано: {coll_info.get('indexed_vectors_count', 0)}"
        
        components_table.add_row(
            "Qdrant Vector Store",
            f"[{vs_status_color}]{vs_status}[/{vs_status_color}]",
            vs_details
        )
        
        # Embedder
        embedder_health = health['components'].get('embedder', {})
        embedder_status = embedder_health.get('status', 'unknown')
        embedder_status_color = "green" if embedder_status == 'healthy' else "yellow"
        embedder_details = f"Модель: {embedder_health.get('model', 'неизвестно')}, Провайдер: {embedder_health.get('provider', 'неизвестно')}"
        
        components_table.add_row(
            "Embedder",
            f"[{embedder_status_color}]{embedder_status}[/{embedder_status_color}]",
            embedder_details
        )
        
        console.print(components_table)
        
        # Конфигурация
        console.print()
        config_table = Table(title="Конфигурация")
        config_table.add_column("Параметр", style="cyan")
        config_table.add_column("Значение", style="green")
        
        config_table.add_row("Хост Qdrant", f"{config.rag.vector_store.host}:{config.rag.vector_store.port}")
        config_table.add_row("Коллекция", config.rag.vector_store.collection_name)
        config_table.add_row("Размерность векторов", str(config.rag.vector_store.vector_size))
        config_table.add_row("Модель эмбеддингов", config.rag.embeddings.model_name)
        config_table.add_row("Провайдер", config.rag.embeddings.provider)
        config_table.add_row("Квантование", f"{config.rag.vector_store.quantization_type}" if config.rag.vector_store.enable_quantization else "отключено")
        
        console.print(config_table)
        
        # Подробная статистика
        if detailed:
            console.print()
            stats = asyncio.run(indexer.get_indexing_stats())
            
            # Статистика индексации
            if stats.get('indexer'):
                indexer_stats = stats['indexer']
                index_table = Table(title="Статистика индексации")
                index_table.add_column("Метрика", style="cyan")
                index_table.add_column("Значение", style="green")
                
                index_table.add_row("Всего файлов", str(indexer_stats.get('total_files', 0)))
                index_table.add_row("Обработано файлов", str(indexer_stats.get('processed_files', 0)))
                index_table.add_row("Всего чанков", str(indexer_stats.get('total_chunks', 0)))
                index_table.add_row("Ошибок", str(indexer_stats.get('failed_files', 0)))
                
                console.print(index_table)
            
            # Статистика эмбеддера
            if stats.get('embedder'):
                embedder_stats = stats['embedder']
                embed_table = Table(title="Статистика эмбеддера")
                embed_table.add_column("Метрика", style="cyan")
                embed_table.add_column("Значение", style="green")
                
                embed_table.add_row("Обработано текстов", str(embedder_stats.get('total_texts', 0)))
                embed_table.add_row("Среднее время батча", f"{embedder_stats.get('avg_batch_time', 0):.3f}s")
                embed_table.add_row("Текстов в секунду", f"{embedder_stats.get('avg_texts_per_second', 0):.1f}")
                embed_table.add_row("Текущий размер батча", str(embedder_stats.get('current_batch_size', 0)))
                embed_table.add_row("OOM fallbacks", str(embedder_stats.get('oom_fallbacks', 0)))
                
                console.print(embed_table)
            
            # Статистика поиска
            searcher = SearchService(config)
            search_stats = searcher.get_search_stats()
            
            search_table = Table(title="Статистика поиска")
            search_table.add_column("Метрика", style="cyan")
            search_table.add_column("Значение", style="green")
            
            search_table.add_row("Всего запросов", str(search_stats.get('total_queries', 0)))
            search_table.add_row("Попаданий в кэш", str(search_stats.get('cache_hits', 0)))
            search_table.add_row("Промахов кэша", str(search_stats.get('cache_misses', 0)))
            search_table.add_row("Размер кэша", str(search_stats.get('cache_size', 0)))
            search_table.add_row("Среднее время поиска", f"{search_stats.get('avg_search_time', 0):.3f}s")
            
            console.print(search_table)
        
        # Закрываем сервисы
        asyncio.run(indexer.close())
        
    except KeyboardInterrupt:
        console.print("[yellow]⏹️ Проверка статуса прервана пользователем[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Ошибка проверки статуса: {e}", exc_info=True)
        console.print(f"[bold red]❌ Ошибка проверки статуса: {e}[/bold red]")
        sys.exit(1)


if __name__ == '__main__':
    cli()
