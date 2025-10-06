"""
Сервис семантического поиска по коду для RAG системы.

Выполняет поиск по векторному хранилищу с фильтрацией, ранжированием
и форматированием результатов для удобного отображения.
"""

import logging
import os
import time
import asyncio
import threading
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel

from config import Config
# ✅ ИСПРАВЛЕНО: Используем remote версии через алиасы
from . import CPUEmbedder, QdrantVectorStore
from .exceptions import VectorStoreException

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат поиска по коду"""
    chunk_id: str
    file_path: str
    file_name: str
    chunk_name: str
    chunk_type: str
    language: str
    start_line: int
    end_line: int
    score: float
    content: str
    metadata: Dict[str, Any]


class SearchService:
    """
    Сервис семантического поиска по коду.
    
    Основные возможности:
    - Семантический поиск с генерацией эмбеддингов
    - Фильтрация по языкам программирования
    - Фильтрация по типам чанков
    - Ранжирование результатов
    - Форматирование для Rich UI
    - Кэширование запросов
    """
    
    def __init__(self, config: Config, silent_mode: bool = False):
        """
        Инициализация сервиса поиска.
        
        Args:
            config: Конфигурация системы
            silent_mode: Отключить консольный вывод (для web UI)
        """
        # Унификация: поддержка как Config с секцией rag, так и RagConfig без неё
        self.config = getattr(config, "rag", config)
        # Use Rich console without emojis to avoid Windows charmap issues
        self.console = Console(emoji=False) if not silent_mode else None
        self.silent_mode = silent_mode
        
        # Инициализация компонентов
        self.embedder = CPUEmbedder(
            self.config.embeddings,
            self.config.parallelism,
            self.config.remote_service
        )
        import os as _os
        env_true = {'1', 'true', 'yes', 'on'}
        use_mock_vs = str(_os.getenv('USE_MOCK_VECTOR_STORE', '')).lower() in env_true or str(_os.getenv('OFFLINE_MODE', '')).lower() in env_true

        if use_mock_vs:
            try:
                from .memory_vector_store import InMemoryVectorStore
                self.vector_store = InMemoryVectorStore(self.config.vector_store, self.config.remote_service)
            except Exception as error:
                logger.warning(f'Не удалось инициализировать InMemoryVectorStore: {error}')
                self.vector_store = None
        else:
            self.vector_store = None

        if self.vector_store is None:
            try:
                self.vector_store = QdrantVectorStore(
                    self.config.vector_store,
                    self.config.remote_service
                )
            except TypeError:
                # Local QdrantVectorStore expects only one argument
                self.vector_store = QdrantVectorStore(
                    self.config.vector_store
                )
        
        # Thread-safe кэш запросов с блокировками
        self._query_cache = {}
        self._cache_lock = threading.RLock()  # RLock для поддержки вложенных вызовов
        self._cache_max_size = self.config.query_engine.cache_max_entries
        self._cache_ttl = self.config.query_engine.cache_ttl_seconds
        
        # Thread-safe статистика поиска с блокировкой
        self._stats_lock = threading.RLock()
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_search_time': 0.0,
            'avg_results_per_query': 0.0,
            'last_query_time': None
        }
        
        logger.info("SearchService инициализирован с thread-safe поддержкой")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
        min_score: Optional[float] = None,
        file_path_filter: Optional[str] = None,
        *,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: Optional[bool] = None,
        task: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Выполняет семантический поиск по коду.
        
        Args:
            query: Поисковый запрос
            top_k: Максимальное количество результатов
            language_filter: Фильтр по языку программирования
            chunk_type_filter: Фильтр по типу чанка (class, function, etc.)
            min_score: Минимальный порог релевантности
            file_path_filter: Фильтр по пути к файлу (поддерживает подстроки)
            
        Returns:
            Список результатов поиска
        """
        start_time = time.time()
        
        try:
            # Проверяем кэш
            # Расширяем фильтры данными из словаря filters, если он задан
            extra_filters_for_cache = dict(filters) if isinstance(filters, dict) else None
            if isinstance(filters, dict) and filters:
                language_filter = language_filter or filters.get('language')
                chunk_type_filter = chunk_type_filter or filters.get('chunk_type')
                file_path_filter = file_path_filter or filters.get('file_path')

            cache_key = self._generate_cache_key(
                query,
                top_k,
                language_filter,
                chunk_type_filter,
                min_score,
                file_path_filter,
                extra_filters=extra_filters_for_cache
            )
            
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._update_stats_safely(cache_hits_incr=1)
                logger.debug(f"Получен результат из кэша для запроса: {query[:50]}...")
                return cached_result
            
            self._update_stats_safely(cache_misses_incr=1)
            
            # Генерируем эмбеддинг для запроса с задачей retrieval.query (Jina v3)
            embed_start = time.time()
            query_task = task or getattr(self.config.embeddings, 'task_query', 'retrieval.query')
            query_embeddings = await asyncio.to_thread(
                self.embedder.embed_texts,
                [query],
                task=query_task
            )
            embed_time = time.time() - embed_start
            
            if query_embeddings is None or len(query_embeddings) == 0:
                logger.error(f"Не удалось сгенерировать эмбеддинг для запроса: {query}")
                return []
            
            query_vector = query_embeddings[0]
            logger.debug(f"Эмбеддинг сгенерирован с task='{query_task}' за {embed_time:.3f}s")
            
            # Строим фильтры для Qdrant
            structured_filters = self._build_search_filters(
                language_filter, chunk_type_filter, file_path_filter
            )
            if isinstance(filters, dict) and filters:
                merged_filters = dict(structured_filters or {})
                merged_filters.update(filters)
                structured_filters = merged_filters
            
            # Выполняем поиск в векторном хранилище
            search_start = time.time()
            sparse_vector = None
            if self.config.query_engine.use_hybrid:
                try:
                    from .sparse_encoder import SparseEncoder
                    from config import get_config
                    cfg = get_config().sparse
                    encoder = SparseEncoder(method=cfg.method)
                    sparse_vector = encoder.encode([query])[0]
                except Exception as e:
                    logger.warning(f"Ошибка генерации sparse-вектора: {e}")
            hybrid_enabled = (
                use_hybrid
                if use_hybrid is not None
                else self.config.query_engine.use_hybrid
            )

            # ИСПРАВЛЕНИЕ: Проверяем тип функции search (RemoteVMVectorStore.search - синхронная)
            search_fn = getattr(self.vector_store, 'search')
            if asyncio.iscoroutinefunction(search_fn):
                raw_results = await search_fn(
                    query_vector,
                    top_k * 2,
                    structured_filters,
                    hybrid_enabled,
                    sparse_vector
                )
            else:
                raw_results = await asyncio.to_thread(
                    search_fn,
                    query_vector,
                    top_k * 2,
                    structured_filters,
                    hybrid_enabled,
                    sparse_vector
                )
            search_time = time.time() - search_start
            
            logger.debug(f"Поиск выполнен за {search_time:.3f}s, найдено {len(raw_results)} результатов")
            
            # Обрабатываем и фильтруем результаты
            effective_min_score = min_score if min_score is not None else self.config.query_engine.score_threshold
            processed_results = self._process_search_results(
                raw_results, effective_min_score
            )
            
            # Применяем MMR если включено
            if self.config.query_engine.mmr_enabled and len(processed_results) > top_k:
                processed_results = self._apply_mmr_ranking(
                    processed_results, query_vector, top_k
                )
            else:
                processed_results = processed_results[:top_k]
            
            # Сохраняем в кэш
            self._save_to_cache(cache_key, processed_results)
            
            # Thread-safe обновление статистики
            total_time = time.time() - start_time
            from datetime import datetime, timezone
            self._update_stats_safely(
                total_queries_incr=1,
                total_search_time_incr=total_time,
                last_query_time=datetime.now(timezone.utc).isoformat()
            )
            
            logger.info(
                f"Поиск завершен: '{query}' -> {len(processed_results)} результатов за {total_time:.3f}s"
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise VectorStoreException(f"Ошибка выполнения поиска: {e}")
    
    def _build_search_filters(
        self,
        language_filter: Optional[str],
        chunk_type_filter: Optional[str],
        file_path_filter: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Строит фильтры для поиска в Qdrant"""
        filters = {}
        
        if language_filter:
            filters['language'] = language_filter.lower()
        
        if chunk_type_filter:
            filters['chunk_type'] = chunk_type_filter
        
        if file_path_filter:
            # Для фильтрации по пути используем подстроку
            filters['file_path'] = file_path_filter
        
        return filters if filters else None
    
    def _process_search_results(
        self, 
        raw_results: List[Dict[str, Any]], 
        min_score: float
    ) -> List[SearchResult]:
        """Обрабатывает сырые результаты поиска в SearchResult объекты"""
        processed = []
        
        for result in raw_results:
            try:
                # Фильтруем по минимальному скору
                if result['score'] < min_score:
                    continue
                
                payload = result.get('payload', {})
                
                search_result = SearchResult(
                    chunk_id=result['id'],
                    file_path=payload.get('file_path', ''),
                    file_name=payload.get('file_name', ''),
                    chunk_name=payload.get('chunk_name', ''),
                    chunk_type=payload.get('chunk_type', ''),
                    language=payload.get('language', ''),
                    start_line=payload.get('start_line', 0),
                    end_line=payload.get('end_line', 0),
                    score=result['score'],
                    content=payload.get('content', ''),
                    metadata=payload
                )
                
                processed.append(search_result)
                
            except Exception as e:
                logger.warning(f"Ошибка обработки результата поиска: {e}")
                continue
        
        # Сортируем по релевантности
        processed.sort(key=lambda x: x.score, reverse=True)
        
        return processed
    
    def _apply_mmr_ranking(
        self,
        results: List[SearchResult],
        query_vector: np.ndarray,
        top_k: int
    ) -> List[SearchResult]:
        """
        Применяет Maximum Marginal Relevance для диверсификации результатов.
        
        Args:
            results: Список результатов поиска
            query_vector: Вектор запроса
            top_k: Количество результатов для возврата
            
        Returns:
            Переранжированный список результатов
        """
        if len(results) <= top_k:
            return results
        
        lambda_param = self.config.query_engine.mmr_lambda
        selected = []
        remaining = results.copy()
        
        # Выбираем первый результат (самый релевантный)
        selected.append(remaining.pop(0))
        
        while len(selected) < top_k and remaining:
            best_score = -1
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Релевантность к запросу
                relevance = candidate.score
                
                # Максимальное сходство с уже выбранными
                max_similarity = 0
                for selected_result in selected:
                    # Упрощенная мера сходства на основе текста
                    similarity = self._text_similarity(candidate.content, selected_result.content)
                    max_similarity = max(max_similarity, similarity)
                
                # MMR скор
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет простую текстовую схожесть между двумя текстами"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_cache_key(
        self,
        query: str,
        top_k: int,
        language_filter: Optional[str],
        chunk_type_filter: Optional[str],
        min_score: Optional[float],
        file_path_filter: Optional[str],
        extra_filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Генерирует ключ для кэширования запроса"""
        import hashlib
        
        extra_filters_part = ""
        if extra_filters:
            try:
                serialized = sorted(extra_filters.items())
            except Exception:
                serialized = []
            extra_filters_part = str(serialized)

        key_parts = [
            query,
            str(top_k),
            language_filter or '',
            chunk_type_filter or '',
            str(min_score) if min_score is not None else '',
            file_path_filter or '',
            extra_filters_part
        ]
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Thread-safe получение результата из кэша"""
        with self._cache_lock:
            if cache_key not in self._query_cache:
                return None
            
            cached_data = self._query_cache[cache_key]
            
            # Проверяем TTL
            if time.time() - cached_data['timestamp'] > self._cache_ttl:
                self._query_cache.pop(cache_key, None)  # Безопасное удаление
                return None
            
            return cached_data['results']
    
    def _update_stats_safely(self, **kwargs) -> None:
        """Thread-safe обновление статистики поиска"""
        with self._stats_lock:
            if 'cache_hits_incr' in kwargs:
                self.stats['cache_hits'] += kwargs['cache_hits_incr']
            if 'cache_misses_incr' in kwargs:
                self.stats['cache_misses'] += kwargs['cache_misses_incr']
            if 'total_queries_incr' in kwargs:
                self.stats['total_queries'] += kwargs['total_queries_incr']
            if 'total_search_time_incr' in kwargs:
                self.stats['total_search_time'] += kwargs['total_search_time_incr']
            if 'last_query_time' in kwargs:
                self.stats['last_query_time'] = kwargs['last_query_time']
            
            # Пересчитываем average results per query
            if self.stats['total_queries'] > 0:
                with self._cache_lock:  # Блокируем кэш для безопасного итерирования
                    try:
                        # Создаем копию значений для безопасного итерирования
                        cache_values = list(self._query_cache.values())
                        total_results = sum(len(data.get('results', [])) for data in cache_values)
                        self.stats['avg_results_per_query'] = total_results / self.stats['total_queries']
                    except Exception as e:
                        logger.warning(f"Ошибка пересчета avg_results_per_query: {e}")
                        self.stats['avg_results_per_query'] = 0.0

    def _save_to_cache(self, cache_key: str, results: List[SearchResult]) -> None:
        """Thread-safe сохранение результата в кэш"""
        with self._cache_lock:
            # Ограничиваем размер кэша
            if len(self._query_cache) >= self._cache_max_size:
                # Удаляем самый старый элемент
                try:
                    if self._query_cache:  # Проверяем что кэш не пустой
                        oldest_key = min(
                            self._query_cache.keys(), 
                            key=lambda k: self._query_cache[k]['timestamp']
                        )
                        self._query_cache.pop(oldest_key, None)
                except (ValueError, KeyError) as e:
                    # Если произошла ошибка, очищаем один произвольный элемент
                    logger.warning(f"Ошибка очистки кэша: {e}, очищаем произвольный элемент")
                    if self._query_cache:
                        first_key = next(iter(self._query_cache))
                        self._query_cache.pop(first_key, None)
            
            # Атомарная запись в кэш
            self._query_cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }
    
    def format_search_results(
        self, 
        results: List[SearchResult], 
        show_content: bool = True,
        max_content_lines: int = 10
    ) -> None:
        """
        Форматирует результаты поиска для вывода с помощью Rich.
        
        Args:
            results: Список результатов поиска
            show_content: Показывать содержимое чанков
            max_content_lines: Максимальное количество строк контента
        """
        # В silent режиме не выводим в консоль
        if self.silent_mode or not self.console:
            return
            
        if not results:
            self.console.print("[yellow]🔍 Результаты не найдены[/yellow]")
            return
        
        self.console.print(f"[bold green]🎯 Найдено результатов: {len(results)}[/bold green]")
        self.console.print()
        
        for i, result in enumerate(results, 1):
            # Заголовок результата
            score_color = "green" if result.score > 0.8 else "yellow" if result.score > 0.6 else "red"
            
            header = (
                f"[bold]{i}. {result.chunk_name}[/bold] "
                f"[dim]({result.file_path}:{result.start_line}-{result.end_line})[/dim] "
                f"[{score_color}]score: {result.score:.3f}[/{score_color}]"
            )
            
            # Метаданные
            metadata = (
                f"[dim]Язык: {result.language.title()}, "
                f"Тип: {result.chunk_type}, "
                f"Файл: {result.file_name}[/dim]"
            )
            
            self.console.print(header)
            self.console.print(metadata)
            
            # Содержимое
            if show_content and result.content:
                content_lines = result.content.split('\n')
                if len(content_lines) > max_content_lines:
                    content = '\n'.join(content_lines[:max_content_lines]) + '\n... (обрезано)'
                else:
                    content = result.content
                
                # Синтаксическая подсветка
                try:
                    syntax = Syntax(
                        content,
                        result.language,
                        theme="monokai",
                        line_numbers=True,
                        start_line=result.start_line
                    )
                    
                    panel = Panel(
                        syntax,
                        title=f"[bold]{result.chunk_name}[/bold]",
                        border_style="blue"
                    )
                    
                    self.console.print(panel)
                    
                except Exception:
                    # Fallback без синтаксической подсветки
                    self.console.print(Panel(content, border_style="dim"))
            
            self.console.print()
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Thread-safe возврат статистики поиска.
        
        Returns:
            Словарь со статистикой
        """
        with self._stats_lock:
            stats = self.stats.copy()
            
            # Дополнительные вычисленные метрики
            if stats['total_queries'] > 0:
                stats['avg_search_time'] = stats['total_search_time'] / stats['total_queries']
                stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']
            else:
                stats['avg_search_time'] = 0.0
                stats['cache_hit_rate'] = 0.0
        
        with self._cache_lock:
            stats['cache_size'] = len(self._query_cache)
            stats['cache_max_size'] = self._cache_max_size
        
        # Добавляем порог релевантности из конфигурации
        stats['score_threshold'] = self.config.query_engine.score_threshold
        
        return stats
    
    def clear_cache(self) -> int:
        """
        Thread-safe очистка кэша поисковых запросов.
        
        Returns:
            Количество удаленных записей
        """
        with self._cache_lock:
            cache_size = len(self._query_cache)
            self._query_cache.clear()
            logger.info(f"Очищен кэш поиска: {cache_size} записей")
            return cache_size
    
    def reset_stats(self) -> None:
        """Thread-safe сброс статистики поиска"""
        with self._stats_lock:
            self.stats = {
                'total_queries': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'total_search_time': 0.0,
                'avg_results_per_query': 0.0,
                'last_query_time': None
            }
            logger.info("Статистика поиска сброшена")
    
    async def close(self) -> None:
        """Закрывает соединения и освобождает ресурсы"""
        try:
            await self.vector_store.close()
            self.clear_cache()
            logger.info("SearchService закрыт")
        except Exception as e:
            logger.error(f"Ошибка закрытия SearchService: {e}")
