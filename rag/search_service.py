"""
Сервис семантического поиска по коду для RAG системы.

Выполняет поиск по векторному хранилищу с фильтрацией, ранжированием
и форматированием результатов для удобного отображения.
"""

import logging
import time
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel

from config import Config
from .embedder import CPUEmbedder
from .vector_store import QdrantVectorStore
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
    
    def __init__(self, config: Config):
        """
        Инициализация сервиса поиска.
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.console = Console()
        
        # Инициализация компонентов
        self.embedder = CPUEmbedder(config.rag.embeddings, config.rag.parallelism)
        self.vector_store = QdrantVectorStore(config.rag.vector_store)
        
        # Кэш запросов
        self._query_cache = {}
        self._cache_max_size = config.rag.query_engine.cache_max_entries
        self._cache_ttl = config.rag.query_engine.cache_ttl_seconds
        
        # Статистика поиска
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_search_time': 0.0,
            'avg_results_per_query': 0.0,
            'last_query_time': None
        }
        
        logger.info("SearchService инициализирован")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
        min_score: Optional[float] = None,
        file_path_filter: Optional[str] = None
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
            cache_key = self._generate_cache_key(
                query, top_k, language_filter, chunk_type_filter, min_score, file_path_filter
            )
            
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.debug(f"Получен результат из кэша для запроса: {query[:50]}...")
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # Генерируем эмбеддинг для запроса
            embed_start = time.time()
            query_embeddings = self.embedder.embed_texts([query])
            embed_time = time.time() - embed_start
            
            if query_embeddings is None or len(query_embeddings) == 0:
                logger.error(f"Не удалось сгенерировать эмбеддинг для запроса: {query}")
                return []
            
            query_vector = query_embeddings[0]
            logger.debug(f"Эмбеддинг сгенерирован за {embed_time:.3f}s")
            
            # Строим фильтры для Qdrant
            filters = self._build_search_filters(
                language_filter, chunk_type_filter, file_path_filter
            )
            
            # Выполняем поиск в векторном хранилище
            search_start = time.time()
            raw_results = await self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k * 2,  # Берем больше результатов для фильтрации
                filters=filters,
                use_hybrid=self.config.rag.query_engine.use_hybrid
            )
            search_time = time.time() - search_start
            
            logger.debug(f"Поиск выполнен за {search_time:.3f}s, найдено {len(raw_results)} результатов")
            
            # Обрабатываем и фильтруем результаты
            processed_results = self._process_search_results(
                raw_results, min_score or self.config.rag.query_engine.score_threshold
            )
            
            # Применяем MMR если включено
            if self.config.rag.query_engine.mmr_enabled and len(processed_results) > top_k:
                processed_results = self._apply_mmr_ranking(
                    processed_results, query_vector, top_k
                )
            else:
                processed_results = processed_results[:top_k]
            
            # Сохраняем в кэш
            self._save_to_cache(cache_key, processed_results)
            
            # Обновляем статистику
            total_time = time.time() - start_time
            self.stats['total_queries'] += 1
            self.stats['total_search_time'] += total_time
            self.stats['last_query_time'] = datetime.utcnow().isoformat()
            
            if self.stats['total_queries'] > 0:
                total_results = sum(len(self._get_from_cache(k) or []) for k in self._query_cache.keys())
                self.stats['avg_results_per_query'] = total_results / self.stats['total_queries']
            
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
        
        lambda_param = self.config.rag.query_engine.mmr_lambda
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
        file_path_filter: Optional[str]
    ) -> str:
        """Генерирует ключ для кэширования запроса"""
        import hashlib
        
        key_parts = [
            query,
            str(top_k),
            language_filter or '',
            chunk_type_filter or '',
            str(min_score) if min_score else '',
            file_path_filter or ''
        ]
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Получает результат из кэша"""
        if cache_key not in self._query_cache:
            return None
        
        cached_data = self._query_cache[cache_key]
        
        # Проверяем TTL
        if time.time() - cached_data['timestamp'] > self._cache_ttl:
            del self._query_cache[cache_key]
            return None
        
        return cached_data['results']
    
    def _save_to_cache(self, cache_key: str, results: List[SearchResult]) -> None:
        """Сохраняет результат в кэш"""
        # Ограничиваем размер кэша
        if len(self._query_cache) >= self._cache_max_size:
            # Удаляем самый старый элемент
            oldest_key = min(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k]['timestamp']
            )
            del self._query_cache[oldest_key]
        
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
        Возвращает статистику поиска.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.stats.copy()
        
        # Дополнительные вычисленные метрики
        if stats['total_queries'] > 0:
            stats['avg_search_time'] = stats['total_search_time'] / stats['total_queries']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']
        else:
            stats['avg_search_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        stats['cache_size'] = len(self._query_cache)
        stats['cache_max_size'] = self._cache_max_size
        
        return stats
    
    def clear_cache(self) -> int:
        """
        Очищает кэш поисковых запросов.
        
        Returns:
            Количество удаленных записей
        """
        cache_size = len(self._query_cache)
        self._query_cache.clear()
        logger.info(f"Очищен кэш поиска: {cache_size} записей")
        return cache_size
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику поиска"""
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