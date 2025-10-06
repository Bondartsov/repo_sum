"""
Сервис индексации репозиториев для RAG системы.

Объединяет компоненты FileScanner, CodeChunker, CPUEmbedder и QdrantVectorStore
для полного процесса индексации кодовой базы в векторное хранилище.
"""

import asyncio
import os
import warnings
import logging
import time
import uuid
import psutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, AsyncGenerator
from datetime import datetime
import numpy as np

from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table

from config import Config
from file_scanner import FileScanner
from code_chunker import CodeChunker
from parsers.base_parser import ParserRegistry
from utils import FileInfo, ParsedFile, CodeChunk
from . import CPUEmbedder, QdrantVectorStore
from .exceptions import VectorStoreException, VectorStoreConnectionError

logger = logging.getLogger(__name__)
# Подавляем шумные SyntaxWarning (например, при парсинге файлов tests)
warnings.filterwarnings('ignore', category=SyntaxWarning)

class IndexerService:
    """
    Сервис индексации репозиториев в RAG систему.
    
    Основные возможности:
    - Сканирование файлов репозитория
    - Парсинг и разбивка кода на чанки
    - Генерация эмбеддингов
    - Сохранение в векторное хранилище
    - Инкрементальная индексация
    - Статистика и мониторинг производительности
    """
    
    def __init__(self, config: Config, silent_mode: bool = False):
        """
        Инициализация сервиса индексации.
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        # Console with emojis disabled to prevent Windows 'charmap' errors; 
        # in silent mode, route output to devnull.
        if silent_mode:
            import io, sys, os
            try:
                devnull = open(os.devnull, 'w', encoding='utf-8')
            except Exception:
                devnull = None
            self.console = Console(emoji=False, file=devnull, force_terminal=False, color_system=None) if devnull else None
        else:
            self.console = Console(emoji=False)
        
        # Инициализация компонентов
        self.file_scanner = FileScanner()
        self.parser_registry = ParserRegistry()
        self.code_chunker = CodeChunker()
        self.embedder = CPUEmbedder(
            config.rag.embeddings,
            config.rag.parallelism,
            config.rag.remote_service
        )
        import os as _os
        env_true = {'1', 'true', 'yes', 'on'}
        use_mock_vs = str(_os.getenv('USE_MOCK_VECTOR_STORE', '')).lower() in env_true or str(_os.getenv('OFFLINE_MODE', '')).lower() in env_true

        if use_mock_vs:
            try:
                from .memory_vector_store import InMemoryVectorStore
                self.vector_store = InMemoryVectorStore(config.rag.vector_store, config.rag.remote_service)
            except Exception as error:
                logger.warning(f'Не удалось инициализировать InMemoryVectorStore: {error}')
                self.vector_store = None
        else:
            self.vector_store = None

        if self.vector_store is None:
            try:
                self.vector_store = QdrantVectorStore(
                    config.rag.vector_store,
                    config.rag.remote_service
                )
            except TypeError:
                # Local QdrantVectorStore expects only one argument
                self.vector_store = QdrantVectorStore(
                    config.rag.vector_store
                )
        
        # Статистика индексации
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'indexed_chunks': 0,
            'total_time': 0.0,
            'embedding_time': 0.0,
            'indexing_time': 0.0,
            'errors': []
        }
        
        logger.info("IndexerService инициализирован")
    
    async def initialize_vector_store(self, recreate: bool = False) -> None:
        """
        Инициализирует векторное хранилище.
        
        Args:
            recreate: Пересоздать коллекцию если она существует
        """
        try:
            init_fn = getattr(self.vector_store, 'initialize_collection')
            if asyncio.iscoroutinefunction(init_fn):
                await init_fn(recreate=recreate)
            else:
                await asyncio.to_thread(init_fn, recreate=recreate)
            logger.info("Vector store ready")
        except Exception as e:
            logger.error(f"Ошибка инициализации векторного хранилища: {e}")
            raise VectorStoreConnectionError(f"Не удалось подключиться к векторному хранилищу: {e}")
    
    async def index_repository(
        self,
        repo_path: str,
        batch_size: int = 128,  # TIMEOUT FIX: Уменьшен batch_size по умолчанию для безопасности
        recreate: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Индексирует репозиторий в векторное хранилище.
        
        Args:
            repo_path: Путь к репозиторию
            batch_size: Размер батча для обработки эмбеддингов
            recreate: Пересоздать коллекцию
            show_progress: Показывать прогресс-бар
            
        Returns:
            Статистика индексации
        """
        start_time = time.time()
        repo_path = Path(repo_path).resolve()
        
        logger.info(f"Начинаем индексацию репозитория: {repo_path}")
        
        try:
            # 1. Инициализация векторного хранилища
            self.console.print("[bold blue]🔗 Инициализация векторного хранилища...[/bold blue]")
            await self.initialize_vector_store(recreate=recreate)

            health_fn = getattr(self.vector_store, 'health_check', None)
            if callable(health_fn):
                try:
                    vs_health = await health_fn() if asyncio.iscoroutinefunction(health_fn) else await asyncio.to_thread(health_fn)
                    status = None
                    if isinstance(vs_health, dict):
                        status = vs_health.get('status') or vs_health.get('state')
                    if status and status.lower() not in {'connected', 'healthy', 'ok'}:
                        raise VectorStoreConnectionError(f'VM vector store health check failed: {status}')
                except Exception as exc:
                    raise VectorStoreConnectionError(f'VM vector store health check failed: {exc}') from exc
            
            # 2. Сканирование файлов
            self.console.print("[bold blue]📁 Сканирование файлов...[/bold blue]")
            files = list(self.file_scanner.scan_repository(str(repo_path)))
            
            if not files:
                self.console.print("[bold red]❌ Не найдено файлов для индексации![/bold red]")
                return {'success': False, 'error': 'Не найдено файлов для индексации'}
            
            self.stats['total_files'] = len(files)
            self.console.print(f"[green]✓ Найдено {len(files)} файлов для индексации[/green]")
            
            # 3. Стримовая обработка файлов и индексация батчами
            self.console.print("[bold blue]🔄 Обработка файлов и индексация...[/bold blue]")
            indexed_count = 0
            total_chunks = 0
            
            async for chunk_batch in self._process_files_generator(files, repo_path, batch_size):
                # Логирование памяти
                mem_info = self._get_memory_info()
                total_chunks += len(chunk_batch)
                self.stats['total_chunks'] = total_chunks
                
                logger.info(f"Обработка батча из {len(chunk_batch)} чанков. "
                            f"Память: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB "
                            f"({mem_info['percent']:.1f}%)")
                
                if show_progress:
                    self.console.print(
                        f"[dim]Батч: {len(chunk_batch)} чанков, "
                        f"Память: {mem_info['used_gb']:.1f}GB/{mem_info['total_gb']:.1f}GB[/dim]"
                    )
                
                # Индексация батча
                batch_indexed = await self._index_chunks_batch(chunk_batch, batch_size, show_progress)
                indexed_count += batch_indexed
            
            if total_chunks == 0:
                self.console.print("[bold yellow]⚠️ Не создано ни одного чанка для индексации[/bold yellow]")
                return {'success': False, 'error': 'Не создано чанков для индексации'}
            
            # 4. Результаты
            self.console.print(f"[bold blue]🧩 Всего создано {total_chunks} чанков кода[/bold blue]")
            
            # 5. Статистика
            total_time = time.time() - start_time
            self.stats['total_time'] = total_time
            
            result = {
                'success': True,
                'repository_path': str(repo_path),
                'total_files': self.stats['total_files'],
                'processed_files': self.stats['processed_files'],
                'failed_files': self.stats['failed_files'],
                'total_chunks': self.stats['total_chunks'],
                'indexed_chunks': indexed_count,
                'total_time': total_time,
                'processing_rate': self.stats['processed_files'] / total_time if total_time > 0 else 0,
                'indexing_rate': indexed_count / total_time if total_time > 0 else 0
            }
            
            self.console.print(f"[bold green]✅ Индексация завершена за {total_time:.1f}s[/bold green]")
            return result
            
        except KeyboardInterrupt:
            logger.info("Индексация прервана пользователем")
            self.console.print("[yellow]⏹️ Индексация прервана пользователем[/yellow]")
            raise
            
        except Exception as e:
            logger.error(f"Критическая ошибка индексации: {e}")
            self.console.print(f"[bold red]❌ Ошибка индексации: {e}[/bold red]")
            raise
    
    async def _process_files_with_progress(
        self, 
        files: List[FileInfo], 
        repo_path: Path
    ) -> List[Tuple[CodeChunk, Dict[str, Any]]]:
        """Обрабатывает файлы с отображением прогресса"""
        all_chunks = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Обработка файлов...", total=len(files))
            
            for file_info in files:
                try:
                    # Обновляем описание прогресса
                    progress.update(
                        task, 
                        description=f"Обработка: {file_info.name}"
                    )
                    
                    # Обрабатываем файл
                    file_chunks = await self._process_single_file(file_info, repo_path)
                    all_chunks.extend(file_chunks)
                    
                    self.stats['processed_files'] += 1
                    
                except Exception as e:
                    logger.error(f"Ошибка обработки файла {file_info.path}: {e}")
                    self.stats['failed_files'] += 1
                    from datetime import datetime, timezone
                    self.stats['errors'].append({
                        'file': file_info.path,
                        'error': str(e),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                
                progress.advance(task)
        
        self.stats['total_chunks'] = len(all_chunks)
        return all_chunks
    
    async def _process_files_simple(
        self, 
        files: List[FileInfo], 
        repo_path: Path
    ) -> List[Tuple[CodeChunk, Dict[str, Any]]]:
        """Простая обработка файлов без прогресс-бара"""
        all_chunks = []
        
        for i, file_info in enumerate(files, 1):
            try:
                self.console.print(f"[dim]Обработка {i}/{len(files)}: {file_info.name}[/dim]")
                
                file_chunks = await self._process_single_file(file_info, repo_path)
                all_chunks.extend(file_chunks)
                
                self.stats['processed_files'] += 1
                
            except Exception as e:
                logger.error(f"Ошибка обработки файла {file_info.path}: {e}")
                self.stats['failed_files'] += 1
                from datetime import datetime, timezone
                self.stats['errors'].append({
                    'file': file_info.path,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        self.stats['total_chunks'] = len(all_chunks)
        return all_chunks
    
    async def _process_files_generator(
        self,
        files: List[FileInfo],
        repo_path: Path,
        batch_size: int = 128  # TIMEOUT FIX: Уменьшено с 256 до 128 для безопасности
    ) -> AsyncGenerator[List[Tuple[CodeChunk, Dict[str, Any]]], None]:
        """
        Генератор для потоковой обработки файлов батчами
        
        Args:
            files: Список файлов для обработки
            repo_path: Корневая директория репозитория
            batch_size: Размер батча чанков (default: 128)
            
        Yields:
            List[Tuple[CodeChunk, Dict[str, Any]]]: Батч чанков готовых к индексации
        """
        current_batch = []
        processed_count = 0
        total_files = len(files)
        
        for file_info in files:
            try:
                # Получить чанки для файла
                file_chunks = await self._process_single_file(file_info, repo_path)
                
                # Добавить в текущий батч
                current_batch.extend(file_chunks)
                
                # Обновить статистику
                self.stats['processed_files'] += 1
                processed_count += 1
                
                # Прогресс-индикатор каждые 10 файлов или на последнем файле
                if processed_count % 10 == 0 or processed_count == total_files:
                    progress_percent = processed_count / total_files * 100
                    bar_filled = int(processed_count * 20 / total_files)
                    bar_empty = 20 - bar_filled
                    logger.info(f"📂 Обработано файлов: {processed_count}/{total_files} "
                                f"({progress_percent:.1f}%) "
                                f"{'█' * bar_filled}{'░' * bar_empty}")
                
                # Если батч достиг размера - отдать его
                while len(current_batch) >= batch_size:
                    # Извлечь батч
                    batch_to_yield = current_batch[:batch_size]
                    current_batch = current_batch[batch_size:]
                    
                    # Отдать батч для индексации
                    yield batch_to_yield
                    
            except Exception as e:
                logger.error(f"Ошибка обработки файла {file_info.path}: {e}")
                self.stats['failed_files'] += 1
                from datetime import datetime, timezone
                self.stats['errors'].append({
                    'file': file_info.path,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                continue
        
        # Отдать остатки
        if current_batch:
            yield current_batch
    
    async def _process_single_file(
        self, 
        file_info: FileInfo, 
        repo_path: Path
    ) -> List[Tuple[CodeChunk, Dict[str, Any]]]:
        """
        Обрабатывает один файл: парсинг -> чанкинг -> метаданные.
        
        Args:
            file_info: Информация о файле
            repo_path: Путь к репозиторию
            
        Returns:
            Список чанков с метаданными
        """
        try:
            # 1. Получаем парсер для файла
            parser = self.parser_registry.get_parser(file_info.path)
            if not parser:
                logger.debug(f"Пропускаем файл без парсера: {file_info.path}")
                return []
            
            # 2. Парсим файл
            parsed_file = parser.safe_parse(file_info)
            
            # 3. Разбиваем на чанки
            chunks = self.code_chunker.chunk_parsed_file(parsed_file)
            
            if not chunks:
                logger.debug(f"Нет чанков для файла: {file_info.path}")
                return []
            
            # 4. Создаем метаданные для каждого чанка
            result_chunks = []
            relative_path = Path(file_info.path).relative_to(repo_path)
            
            for chunk in chunks:
                from datetime import datetime, timezone
                metadata = {
                    'file_path': str(relative_path),
                    'file_name': file_info.name,
                    'language': file_info.language,
                    'chunk_name': chunk.name,
                    'chunk_type': chunk.chunk_type,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'tokens_estimate': chunk.tokens_estimate,
                    'file_size': file_info.size,
                    'indexed_at': datetime.now(timezone.utc).isoformat(),
                    'repository': repo_path.name
                }
                
                result_chunks.append((chunk, metadata))
            
            logger.debug(f"Создано {len(chunks)} чанков для {file_info.path}")
            return result_chunks
            
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_info.path}: {e}")
            raise
    
    def _check_memory_and_adjust_batch(self, current_batch_size: int) -> int:
        """
        Динамическая подстройка batch_size на основе текущего использования памяти.
        
        Защита от OOM (Out Of Memory) killer:
        - При >85% памяти: уменьшаем batch в 4 раза (минимум 32)
        - При >75% памяти: уменьшаем batch в 2 раза (минимум 64)
        - При <50% памяти: увеличиваем batch в 2 раза (максимум 512)
        
        Args:
            current_batch_size: Текущий размер батча
            
        Returns:
            Оптимизированный размер батча
        """
        try:
            memory = psutil.virtual_memory()
            mem_percent = memory.percent
            
            # Критический уровень памяти (>85%) - агрессивное уменьшение
            if mem_percent > 85:
                new_batch_size = max(32, current_batch_size // 4)
                if new_batch_size != current_batch_size:
                    logger.warning(
                        f"🚨 Критический уровень памяти: {mem_percent:.1f}% "
                        f"(доступно: {memory.available / (1024**3):.1f}Gi). "
                        f"Уменьшаем batch_size: {current_batch_size} → {new_batch_size}"
                    )
                return new_batch_size
            
            # Высокий уровень памяти (>75%) - умеренное уменьшение
            elif mem_percent > 75:
                new_batch_size = max(64, current_batch_size // 2)
                if new_batch_size != current_batch_size:
                    logger.warning(
                        f"⚠️ Высокий уровень памяти: {mem_percent:.1f}% "
                        f"(доступно: {memory.available / (1024**3):.1f}Gi). "
                        f"Уменьшаем batch_size: {current_batch_size} → {new_batch_size}"
                    )
                return new_batch_size
            
            # Низкий уровень памяти (<50%) - можем увеличить batch обратно
            elif mem_percent < 50 and current_batch_size < 512:
                new_batch_size = min(512, current_batch_size * 2)
                if new_batch_size != current_batch_size:
                    logger.info(
                        f"✅ Низкий уровень памяти: {mem_percent:.1f}% "
                        f"(доступно: {memory.available / (1024**3):.1f}Gi). "
                        f"Увеличиваем batch_size: {current_batch_size} → {new_batch_size}"
                    )
                return new_batch_size
            
            # Нормальный уровень памяти (50-75%) - не меняем batch
            return current_batch_size
            
        except Exception as e:
            logger.error(f"Ошибка проверки памяти: {e}")
            # При ошибке возвращаем безопасный размер
            return min(128, current_batch_size)
    
    def _get_memory_info(self) -> dict:
        """Получить информацию о текущем использовании памяти"""
        try:
            process = psutil.Process()
            mem = process.memory_info()
            system_mem = psutil.virtual_memory()
            
            return {
                'used_gb': mem.rss / (1024**3),
                'total_gb': system_mem.total / (1024**3),
                'percent': (mem.rss / system_mem.total) * 100,
                'available_gb': system_mem.available / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Не удалось получить info памяти: {e}")
            return {'used_gb': 0, 'total_gb': 0, 'percent': 0, 'available_gb': 0}
    
    async def _index_chunks_batch(
        self, 
        chunks: List[Tuple[CodeChunk, Dict[str, Any]]], 
        batch_size: int,
        show_progress: bool = True
    ) -> int:
        """
        Индексирует чанки батчами с генерацией эмбеддингов.
        
        Args:
            chunks: Список чанков с метаданными
            batch_size: Размер батча
            show_progress: Показывать прогресс
            
        Returns:
            Количество проиндексированных чанков
        """
        if not chunks:
            return 0
        
        start_time = time.time()
        indexed_count = 0
        
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console
            )
            progress.start()
            task = progress.add_task("Индексация чанков...", total=len(chunks))
        else:
            progress = None
            task = None
        
        try:
            # Обрабатываем батчами с динамической подстройкой batch_size
            current_batch_size = batch_size
            i = 0
            batch_num = 0
            
            while i < len(chunks):
                # Проверяем память и корректируем batch_size перед каждым batch
                current_batch_size = self._check_memory_and_adjust_batch(current_batch_size)
                
                batch = chunks[i:i + current_batch_size]
                batch_num += 1
                
                if progress:
                    memory = psutil.virtual_memory()
                    progress.update(
                        task, 
                        description=f"Батч {batch_num} (size={current_batch_size}, mem={memory.percent:.1f}%)"
                    )
                
                # Генерируем эмбеддинги для батча с задачей retrieval.passage (Jina v3)
                texts = [chunk.content for chunk, _ in batch]
                
                embed_start = time.time()
                passage_task = getattr(self.config.rag.embeddings, 'task_passage', 'retrieval.passage')
                embeddings = await asyncio.to_thread(self.embedder.embed_texts, texts, task=passage_task)
                self.stats['embedding_time'] += time.time() - embed_start
                # Нормализация формы эмбеддингов для устойчивости пайплайна
                try:
                    embeddings = np.asarray(embeddings)
                    if embeddings.ndim == 0:
                        embeddings = []
                    elif embeddings.ndim == 1 and len(texts) == 1:
                        embeddings = embeddings.reshape(1, -1)
                except Exception as e:
                    logger.error(f"Ошибка приведения формы эмбеддингов: {e}")
                    embeddings = []
                
                if embeddings is None or len(embeddings) == 0:
                    logger.error(f"Не удалось сгенерировать эмбеддинги для батча {i//batch_size + 1}")
                    continue
                
                logger.debug(f"Сгенерированы эмбеддинги с task='{passage_task}' для батча {i//batch_size + 1}")
                
                # Подготавливаем точки для Qdrant
                points = []
                for j, ((chunk, metadata), embedding) in enumerate(zip(batch, embeddings)):
                    point_id = str(uuid.uuid4())
                    
                    points.append({
                        'id': point_id,
                        'vector': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                        'payload': {
                            **metadata,
                            'content': chunk.content,  # Сохраняем контент для поиска
                            'point_id': point_id
                        }
                    })
                
                # Индексируем в Qdrant
                index_start = time.time()
                
                # ДИАГНОСТИКА: Проверяем тип vector_store и метода
                logger.info(f"🔍 ДИАГНОСТИКА: type(vector_store) = {type(self.vector_store).__name__}")
                index_fn = getattr(self.vector_store, 'index_documents')
                logger.info(f"🔍 ДИАГНОСТИКА: asyncio.iscoroutinefunction(index_documents) = {asyncio.iscoroutinefunction(index_fn)}")
                
                # Проверяем тип функции и вызываем соответствующим образом
                if asyncio.iscoroutinefunction(index_fn):
                    # Локальный QdrantVectorStore - async функция
                    logger.info("✅ Используем await для async функции")
                    batch_indexed = await index_fn(points)
                else:
                    # Удалённый RemoteVectorStore - sync функция
                    logger.info("✅ Используем asyncio.to_thread для sync функции")
                    batch_indexed = await asyncio.to_thread(index_fn, points)
                
                logger.info(f"✅ batch_indexed = {batch_indexed}, type = {type(batch_indexed).__name__}")
                self.stats['indexing_time'] += time.time() - index_start
                
                indexed_count += batch_indexed
                
                if progress:
                    progress.advance(task, len(batch))
                
                # Двигаемся к следующему batch
                i += len(batch)
                
                # Краткая пауза между батчами для стабильности
                await asyncio.sleep(0.1)
        
        finally:
            if progress:
                progress.stop()
        
        total_time = time.time() - start_time
        logger.info(f"Индексировано {indexed_count}/{len(chunks)} чанков за {total_time:.2f}s")
        
        return indexed_count
    
    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 512,
        recreate_collection: bool = False
    ) -> int:
        """Индексация произвольных документов (id, text, metadata) напрямую в хранилище.

        Используется VM API эндпоинтом /index.
        """
        logger.info(f"📥 index_documents: получено {len(documents)} документов")
        if not documents:
            logger.warning("⚠️ Пустой список документов!")
            return 0

        await self.initialize_vector_store(recreate=recreate_collection)

        passage_task = getattr(self.config.rag.embeddings, 'task_passage', 'retrieval.passage')
        total_indexed = 0

        for i in range(0, len(documents), batch_size):
            batch_num = i // batch_size + 1
            logger.info(f"📦 Батч {batch_num}: обработка документов {i} - {min(i+batch_size, len(documents))}")
            
            batch_docs = documents[i:i + batch_size]
            texts = [doc.get('text', '') for doc in batch_docs]
            
            # Диагностика текстов
            if texts:
                first_text_preview = texts[0][:100] if len(texts[0]) > 100 else texts[0]
                logger.info(f"📝 Извлечено {len(texts)} текстов. Первый: '{first_text_preview}...' (длина: {len(texts[0])})")
            else:
                logger.warning(f"⚠️ Батч {batch_num}: список текстов пуст!")

            embeddings = await asyncio.to_thread(self.embedder.embed_texts, texts, task=passage_task)
            logger.info(f"🔢 Эмбеддинги получены: shape={getattr(embeddings, 'shape', None)}, type={type(embeddings).__name__}")
            try:
                embeddings = np.asarray(embeddings)
                if embeddings.ndim == 1 and len(texts) == 1:
                    embeddings = embeddings.reshape(1, -1)
                logger.info(f"✅ Форма эмбеддингов после reshape: {embeddings.shape}")
            except Exception as e:
                logger.error(f"❌ Ошибка приведения формы эмбеддингов: {e}", exc_info=True)
                continue

            if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
                logger.error(
                    f"❌ БАТЧ {batch_num} ОТБРОШЕН! Некорректная форма эмбеддингов: shape={embeddings.shape}, ожидалось: ({len(texts)}, ?)"
                )
                continue

            points = []
            for j, doc in enumerate(batch_docs):
                point_id = str(doc.get('id') or uuid.uuid4())
                metadata = dict(doc.get('metadata') or {})
                from datetime import datetime, timezone
                ts = doc.get('timestamp') or datetime.now(timezone.utc).isoformat()

                vec = embeddings[j]
                vec = vec.tolist() if hasattr(vec, 'tolist') else vec

                points.append({
                    'id': point_id,
                    'vector': vec,
                    'payload': {
                        **metadata,
                        'content': doc.get('text', ''),
                        'point_id': point_id,
                        'indexed_at': ts,
                    }
                })

            logger.info(f"💾 Отправка {len(points)} точек в vector_store.index_documents()")
            try:
                index_fn = getattr(self.vector_store, 'index_documents')
                logger.info(f"🔍 Тип функции index_documents: async={asyncio.iscoroutinefunction(index_fn)}")
                
                if asyncio.iscoroutinefunction(index_fn):
                    batch_indexed = await index_fn(points)
                else:
                    batch_indexed = await asyncio.to_thread(index_fn, points)
                
                logger.info(f"✅ Батч {batch_num}: проиндексировано {batch_indexed} точек (type={type(batch_indexed).__name__})")
                total_indexed += batch_indexed
            except Exception as e:
                logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА индексации батча {batch_num}: {e}", exc_info=True)
                # НЕ пробрасываем ошибку, но фиксируем в логах

        logger.info(f"🎯 ИТОГО проиндексировано: {total_indexed} из {len(documents)} документов")
        return total_indexed

    async def get_indexing_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику индексации.
        
        Returns:
            Словарь со статистикой
        """
        # Получаем статистику компонентов
        embedder_stats = self.embedder.get_stats()
        vector_store_stats = self.vector_store.get_stats()
        
        # Объединяем со статистикой индексации
        from datetime import datetime, timezone
        combined_stats = {
            'indexer': self.stats.copy(),
            'embedder': embedder_stats,
            'vector_store': vector_store_stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return combined_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверяет состояние всех компонентов индексации.
        
        Returns:
            Информация о состоянии системы
        """
        from datetime import datetime, timezone
        health_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'healthy',
            'components': {}
        }
        
        try:
            # Проверка векторного хранилища
            _vh = self.vector_store.health_check
            if asyncio.iscoroutinefunction(_vh):
                vs_health = await _vh()
            else:
                vs_health = await asyncio.to_thread(_vh)
            health_info['components']['vector_store'] = vs_health
            
            # Проверка эмбеддера
            # Добавляем прогрев перед получением статистики
            self.embedder.warmup()
            embedder_stats = self.embedder.get_stats()
            health_info['components']['embedder'] = {
                'status': 'healthy' if embedder_stats['is_warmed_up'] else 'warming_up',
                'provider': embedder_stats['provider'],
                'model': embedder_stats['model_name'],
                'stats': embedder_stats
            }
            
            # Общий статус
            if vs_health['status'].lower() not in {"connected", "healthy", "ok"}:
                health_info['status'] = 'degraded'
            
        except Exception as e:
            health_info['status'] = 'unhealthy'
            health_info['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_info

    def print_health_status(self, health_info: Dict[str, Any]) -> None:
        """Выводит таблицу статусов компонентов в консоль."""
        if not self.console:
            return

        status_colors = {
            "green": {"ok", "healthy", "connected"},
            "yellow": {"degraded", "warming", "initializing"},
            "red": {"error", "failed", "unavailable", "unhealthy"},
        }

        def colorize(status: str) -> str:
            raw = status or "unknown"
            s = raw.strip().lower()

            # Fallback: если статус "ok", всегда зелёный
            if s == "ok":
                return f"[green]{raw}[/green]"

            # Проверяем остальные статусы по категориям
            for color, states in status_colors.items():
                if s in states:
                    return f"[{color}]{raw}[/{color}]"

            # Fallback для неизвестных статусов - жёлтый (требует внимания)
            return f"[yellow]{raw}[/yellow]"

        table = Table(title="Состояние компонентов", show_header=True, header_style="bold magenta")
        table.add_column("Компонент", style="bold")
        table.add_column("Статус", style="green")
        table.add_column("Детали", style="dim")

        components = health_info.get("components", {})

        # Vector Store (Qdrant)
        vs = components.get("vector_store", {})
        if vs:
            # Реализуем правильную логику fallback для получения атрибутов из разных источников
            def get_qdrant_attribute(attr_name: str, default: str = None) -> str:
                """Получаем атрибут Qdrant с fallback из разных источников"""
                # Пробуем получить из конфига векторного хранилища
                cfg = getattr(self.vector_store, "config", None)
                if cfg:
                    value = getattr(cfg, attr_name, None)
                    if value is not None:
                        return str(value)

                # Пробуем получить из конфига RAG
                if hasattr(self.config, 'rag') and hasattr(self.config.rag, 'vector_store'):
                    value = getattr(self.config.rag.vector_store, attr_name, None)
                    if value is not None:
                        return str(value)

                # Пробуем получить напрямую из векторного хранилища
                value = getattr(self.vector_store, attr_name, None)
                if value is not None:
                    return str(value)

                return default or "-"

            # Получаем атрибуты с правильным fallback
            host = get_qdrant_attribute('host', '') or get_qdrant_attribute('service_host', '') or '-'
            collection = get_qdrant_attribute('collection_name', '') or get_qdrant_attribute('collection', '') or '-'
            dim = get_qdrant_attribute('vector_size', '') or get_qdrant_attribute('dim', '') or '-'

            # Формируем строку деталей в требуемом формате
            qdrant_details = f"Хост: {host}, Коллекция: {collection}, Размерность: {dim}"

            qdrant_status = (vs.get("status") or "unknown").strip()
            table.add_row(
                "Qdrant Vector Store",
                colorize(qdrant_status),
                qdrant_details
            )

        # Embedder
        emb = components.get("embedder", {})
        if emb:
            details = []
            if emb.get("provider"):
                details.append(f"Провайдер: {emb['provider']}")
            if emb.get("model"):
                details.append(f"Модель: {emb['model']}")
            stats = emb.get("stats", {})
            if stats.get("embedding_dim"):
                details.append(f"Размерность: {stats['embedding_dim']}")
            table.add_row("Embedder", colorize(emb.get("status")), ", ".join(details) or "-")

        self.console.print(table)
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику индексации"""
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'indexed_chunks': 0,
            'total_time': 0.0,
            'embedding_time': 0.0,
            'indexing_time': 0.0,
            'errors': []
        }
        
        # Сбрасываем статистику компонентов
        self.embedder.reset_stats()
        self.vector_store.reset_stats()
        
        logger.info("Статистика индексации сброшена")
    
    async def close(self) -> None:
        """Закрывает соединения и освобождает ресурсы"""
        try:
            await self.vector_store.close()
            logger.info("IndexerService закрыт")
        except Exception as e:
            logger.error(f"Ошибка закрытия IndexerService: {e}")
