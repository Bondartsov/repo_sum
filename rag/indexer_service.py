"""
Сервис индексации репозиториев для RAG системы.

Объединяет компоненты FileScanner, CodeChunker, CPUEmbedder и QdrantVectorStore
для полного процесса индексации кодовой базы в векторное хранилище.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console

from config import Config
from file_scanner import FileScanner
from code_chunker import CodeChunker
from parsers.base_parser import ParserRegistry
from utils import FileInfo, ParsedFile, CodeChunk
from .embedder import CPUEmbedder
from .vector_store import QdrantVectorStore
from .exceptions import VectorStoreException, VectorStoreConnectionError

logger = logging.getLogger(__name__)


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
    
    def __init__(self, config: Config):
        """
        Инициализация сервиса индексации.
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.console = Console()
        
        # Инициализация компонентов
        self.file_scanner = FileScanner()
        self.parser_registry = ParserRegistry()
        self.code_chunker = CodeChunker()
        self.embedder = CPUEmbedder(config.rag.embeddings, config.rag.parallelism)
        self.vector_store = QdrantVectorStore(config.rag.vector_store)
        
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
            await self.vector_store.initialize_collection(recreate=recreate)
            logger.info("Векторное хранилище готово к работе")
        except Exception as e:
            logger.error(f"Ошибка инициализации векторного хранилища: {e}")
            raise VectorStoreConnectionError(f"Не удалось подключиться к векторному хранилищу: {e}")
    
    async def index_repository(
        self, 
        repo_path: str, 
        batch_size: int = 512, 
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
            
            # 2. Сканирование файлов
            self.console.print("[bold blue]📁 Сканирование файлов...[/bold blue]")
            files = list(self.file_scanner.scan_repository(str(repo_path)))
            
            if not files:
                self.console.print("[bold red]❌ Не найдено файлов для индексации![/bold red]")
                return {'success': False, 'error': 'Не найдено файлов для индексации'}
            
            self.stats['total_files'] = len(files)
            self.console.print(f"[green]✓ Найдено {len(files)} файлов для индексации[/green]")
            
            # 3. Обработка файлов с прогресс-баром
            if show_progress:
                chunks = await self._process_files_with_progress(files, repo_path)
            else:
                chunks = await self._process_files_simple(files, repo_path)
            
            if not chunks:
                self.console.print("[bold yellow]⚠️ Не создано ни одного чанка для индексации[/bold yellow]")
                return {'success': False, 'error': 'Не создано чанков для индексации'}
            
            # 4. Генерация эмбеддингов и индексация
            self.console.print(f"[bold blue]🧩 Создано {len(chunks)} чанков кода[/bold blue]")
            indexed_count = await self._index_chunks_batch(chunks, batch_size, show_progress)
            
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
                    self.stats['errors'].append({
                        'file': file_info.path,
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
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
                self.stats['errors'].append({
                    'file': file_info.path,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        self.stats['total_chunks'] = len(all_chunks)
        return all_chunks
    
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
                    'indexed_at': datetime.utcnow().isoformat(),
                    'repository': repo_path.name
                }
                
                result_chunks.append((chunk, metadata))
            
            logger.debug(f"Создано {len(chunks)} чанков для {file_info.path}")
            return result_chunks
            
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_info.path}: {e}")
            raise
    
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
            # Обрабатываем батчами
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                if progress:
                    progress.update(task, description=f"Батч {i//batch_size + 1}")
                
                # Генерируем эмбеддинги для батча с задачей retrieval.passage (Jina v3)
                texts = [chunk.content for chunk, _ in batch]
                
                embed_start = time.time()
                passage_task = getattr(self.config.rag.embeddings, 'task_passage', 'retrieval.passage')
                embeddings = self.embedder.embed_texts(texts, task=passage_task)
                self.stats['embedding_time'] += time.time() - embed_start
                
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
                batch_indexed = await self.vector_store.index_documents(points)
                self.stats['indexing_time'] += time.time() - index_start
                
                indexed_count += batch_indexed
                
                if progress:
                    progress.advance(task, len(batch))
                
                # Краткая пауза между батчами для стабильности
                await asyncio.sleep(0.1)
        
        finally:
            if progress:
                progress.stop()
        
        total_time = time.time() - start_time
        logger.info(f"Индексировано {indexed_count}/{len(chunks)} чанков за {total_time:.2f}s")
        
        return indexed_count
    
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
        combined_stats = {
            'indexer': self.stats.copy(),
            'embedder': embedder_stats,
            'vector_store': vector_store_stats,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return combined_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверяет состояние всех компонентов индексации.
        
        Returns:
            Информация о состоянии системы
        """
        health_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'components': {}
        }
        
        try:
            # Проверка векторного хранилища
            vs_health = await self.vector_store.health_check()
            health_info['components']['vector_store'] = vs_health
            
            # Проверка эмбеддера
            embedder_stats = self.embedder.get_stats()
            health_info['components']['embedder'] = {
                'status': 'healthy' if embedder_stats['is_warmed_up'] else 'warming_up',
                'provider': embedder_stats['provider'],
                'model': embedder_stats['model_name'],
                'stats': embedder_stats
            }
            
            # Общий статус
            if vs_health['status'] != 'connected':
                health_info['status'] = 'degraded'
            
        except Exception as e:
            health_info['status'] = 'unhealthy'
            health_info['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_info
    
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
