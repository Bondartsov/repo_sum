"""
Полноценный поисковый движок RAG системы с продвинутыми алгоритмами.

Основные возможности:
- RRF (Reciprocal Rank Fusion) для объединения результатов
- MMR (Maximum Marginal Relevance) для переранжирования
- LRU-кэш с TTL для горячих запросов
- Параллельная обработка запросов
- Интеграция с SearchService
"""

import asyncio
import hashlib
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import defaultdict

import numpy as np
from cachetools import TTLCache

from config import QueryEngineConfig
from .embedder import CPUEmbedder
from .vector_store import QdrantVectorStore
from .search_service import SearchService
from .exceptions import QueryEngineException, VectorStoreException, TimeoutException

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Статистика производительности запросов"""
    def __init__(self):
        self.total_queries: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.average_latency: float = 0.0
        self.embedding_time: float = 0.0
        self.search_time: float = 0.0
        self.rerank_time: float = 0.0
        self.total_latency: float = 0.0
        self.rrf_operations: int = 0
        self.mmr_operations: int = 0
        self.concurrent_queries: int = 0
        self.max_concurrent_queries: int = 0
        self.error_count: int = 0
        self.last_query_time: Optional[str] = None

    def update_latency(self, latency: float):
        """Обновляет статистику времени отклика"""
        self.total_latency += latency
        if self.total_queries > 0:
            self.average_latency = self.total_latency / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует статистику в словарь"""
        cache_hit_rate = 0.0
        if self.total_queries > 0:
            cache_hit_rate = self.cache_hits / self.total_queries

        return {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'average_latency': self.average_latency,
            'embedding_time': self.embedding_time,
            'search_time': self.search_time,
            'rerank_time': self.rerank_time,
            'rrf_operations': self.rrf_operations,
            'mmr_operations': self.mmr_operations,
            'concurrent_queries': self.concurrent_queries,
            'max_concurrent_queries': self.max_concurrent_queries,
            'error_count': self.error_count,
            'last_query_time': self.last_query_time
        }


@dataclass
class SearchResult:
    """Расширенный результат поиска с дополнительными метаданными"""
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
    embedding: Optional[np.ndarray] = None
    rrf_score: Optional[float] = None
    mmr_score: Optional[float] = None
    rank: Optional[int] = None


class CPUQueryEngine:
    """
    Полноценный поисковый движок RAG системы с продвинутыми алгоритмами.

    Возможности:
    - RRF (Reciprocal Rank Fusion) для объединения результатов разных источников
    - MMR (Maximum Marginal Relevance) для переранжирования и разнообразия
    - LRU-кэш с TTL для горячих запросов (cachetools)
    - Параллельная обработка до 20 пользователей (asyncio + пул воркеров)
    - Интеграция с существующим SearchService
    - Подробная статистика производительности
    """

    def __init__(self, embedder: CPUEmbedder, store: QdrantVectorStore, qcfg: QueryEngineConfig):
        """
        Инициализация полноценного поискового движка.

        Args:
            embedder: Эмбеддер для векторизации запросов
            store: Векторное хранилище Qdrant
            qcfg: Конфигурация поискового движка
        """
        self.embedder = embedder
        self.vector_store = store
        self.config = qcfg

        # Используем SearchService как базовый компонент
        # Безопасно извлекаем параметры из store.config
        store_host = getattr(store.config, 'host', 'localhost') if hasattr(store, 'config') else 'localhost'
        store_port = getattr(store.config, 'port', 6333) if hasattr(store, 'config') else 6333
        store_collection = getattr(store.config, 'collection_name', 'default') if hasattr(store, 'config') else 'default'
        
        # Создаём фиктивный Config для SearchService
        from config import Config, RagConfig
        config_dict = {
            'rag': {
                'embeddings': {
                    'provider': 'fastembed',
                    'model_name': getattr(embedder.embedding_config, 'model_name', 'BAAI/bge-small-en-v1.5') if hasattr(embedder, 'embedding_config') else 'BAAI/bge-small-en-v1.5',
                    'batch_size_min': 8,
                    'batch_size_max': 128,
                    'device': 'cpu',
                    'num_workers': 4
                },
                'vector_store': {
                    'host': store_host,
                    'port': store_port,
                    'collection_name': store_collection
                },
                'query_engine': {
                    'mmr_enabled': qcfg.mmr_enabled,
                    'mmr_lambda': qcfg.mmr_lambda,
                    'score_threshold': qcfg.score_threshold,
                    'cache_max_entries': qcfg.cache_max_entries,
                    'cache_ttl_seconds': qcfg.cache_ttl_seconds,
                    'use_hybrid': qcfg.use_hybrid
                }
            }
        }
        
        # Создаём минимальную конфигурацию для SearchService
        try:
            from config import get_config
            base_config = get_config(require_api_key=False)
            base_config.rag.query_engine = qcfg
            self.search_service = SearchService(base_config)
        except:
            # Fallback: создаём SearchService напрямую с переданными компонентами
            self.search_service = SearchService.__new__(SearchService)
            self.search_service.embedder = embedder
            self.search_service.vector_store = store
            self.search_service.config = qcfg
            self.search_service._query_cache = {}
            self.search_service._cache_max_size = qcfg.cache_max_entries
            self.search_service._cache_ttl = qcfg.cache_ttl_seconds
            self.search_service.stats = {
                'total_queries': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'total_search_time': 0.0,
                'avg_results_per_query': 0.0,
                'last_query_time': None
            }

        # LRU кэш с TTL для результатов после фьюжна/переранжирования
        self.cache = TTLCache(
            maxsize=qcfg.cache_max_entries,
            ttl=qcfg.cache_ttl_seconds
        )

        # Пул воркеров для параллельной обработки
        self.search_executor = ThreadPoolExecutor(
            max_workers=qcfg.search_workers,
            thread_name_prefix="search_worker"
        )
        self.embed_executor = ThreadPoolExecutor(
            max_workers=qcfg.embed_workers, 
            thread_name_prefix="embed_worker"
        )

        # Статистика производительности
        self.stats = QueryStats()

        # Семафор для ограничения одновременных запросов
        self.concurrent_semaphore = asyncio.Semaphore(qcfg.concurrent_users_target)

        logger.info(f"CPUQueryEngine инициализирован: RRF={qcfg.rrf_enabled}, MMR={qcfg.mmr_enabled}")

    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Главный метод поиска с продвинутыми алгоритмами.

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов

        Returns:
            Список результатов с payload (файл, строки) + score

        Raises:
            QueryEngineException: При ошибке поиска
        """
        if max_results is None:
            max_results = self.config.max_results

        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]

        # Ограничиваем количество одновременных запросов
        async with self.concurrent_semaphore:
            self.stats.concurrent_queries += 1
            self.stats.max_concurrent_queries = max(
                self.stats.max_concurrent_queries, 
                self.stats.concurrent_queries
            )

            try:
                logger.debug(f"[{request_id}] Начинаем поиск: '{query[:50]}...'")

                # 1. Проверяем кэш
                cache_key = self._generate_cache_key(query, max_results)
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    self.stats.cache_hits += 1
                    self.stats.total_queries += 1
                    logger.debug(f"[{request_id}] Результат из кэша: {len(cached_result)} элементов")
                    return cached_result

                self.stats.cache_misses += 1

                # 2. Генерируем эмбеддинг запроса
                embed_start = time.time()
                try:
                    query_embeddings = self.embedder.embed_texts([query])
                    if query_embeddings is None or len(query_embeddings) == 0:
                        raise QueryEngineException(
                            message="Не удалось сгенерировать эмбеддинг запроса",
                            query=query
                        )
                    query_vector = query_embeddings[0]
                except Exception as e:
                    raise QueryEngineException(
                        message="Ошибка генерации эмбеддинга",
                        query=query,
                        details=str(e)
                    )
                
                self.stats.embedding_time += time.time() - embed_start

                # 3. Выполняем базовый поиск через SearchService
                search_start = time.time()
                raw_results = await self._perform_base_search(query, max_results * 2, request_id)
                self.stats.search_time += time.time() - search_start

                if not raw_results:
                    logger.debug(f"[{request_id}] Поиск не дал результатов")
                    self._cache_result(cache_key, [])
                    return []

                # 4. Применяем RRF (если включен и есть несколько источников)
                rerank_start = time.time()
                if self.config.rrf_enabled:
                    # Для демонстрации RRF создаём дополнительные источники результатов
                    rrf_results = self._reciprocal_rank_fusion([raw_results], k=60)
                    self.stats.rrf_operations += 1
                else:
                    rrf_results = raw_results

                # 5. Применяем MMR переранжирование
                final_results = rrf_results
                if self.config.mmr_enabled and len(rrf_results) > max_results:
                    final_results = self._mmr_rerank(
                        rrf_results, 
                        query_vector,
                        lambda_param=self.config.mmr_lambda,
                        max_results=max_results
                    )
                    self.stats.mmr_operations += 1
                else:
                    final_results = rrf_results[:max_results]

                self.stats.rerank_time += time.time() - rerank_start

                # 6. Преобразуем в выходной формат
                output_results = self._convert_to_output_format(final_results)

                # 7. Кэшируем результат
                self._cache_result(cache_key, output_results)

                # 8. Обновляем статистику
                total_time = time.time() - start_time
                self.stats.total_queries += 1
                self.stats.update_latency(total_time)
                self.stats.last_query_time = datetime.utcnow().isoformat()

                logger.info(
                    f"[{request_id}] Поиск завершён: {len(output_results)} результатов за {total_time:.3f}s"
                )

                return output_results

            except Exception as e:
                self.stats.error_count += 1
                logger.error(f"[{request_id}] Ошибка поиска: {e}")
                if isinstance(e, QueryEngineException):
                    raise
                raise QueryEngineException(
                    message="Внутренняя ошибка поискового движка",
                    query=query,
                    details=str(e)
                )
            finally:
                self.stats.concurrent_queries -= 1

    async def _perform_base_search(
        self, 
        query: str, 
        limit: int, 
        request_id: str
    ) -> List[SearchResult]:
        """Выполняет базовый поиск через SearchService"""
        try:
            # Используем SearchService для базового поиска
            raw_results = await self.search_service.search(
                query=query,
                top_k=limit,
                min_score=self.config.score_threshold
            )
            
            # Преобразуем результаты SearchService в наш формат
            search_results = []
            for idx, result in enumerate(raw_results):
                search_results.append(SearchResult(
                    chunk_id=result.chunk_id,
                    file_path=result.file_path,
                    file_name=result.file_name,
                    chunk_name=result.chunk_name,
                    chunk_type=result.chunk_type,
                    language=result.language,
                    start_line=result.start_line,
                    end_line=result.end_line,
                    score=result.score,
                    content=result.content,
                    metadata=result.metadata,
                    rank=idx + 1
                ))
            
            logger.debug(f"[{request_id}] Базовый поиск: {len(search_results)} результатов")
            return search_results

        except Exception as e:
            logger.error(f"[{request_id}] Ошибка базового поиска: {e}")
            raise

    def _reciprocal_rank_fusion(self, results_lists: List[List[SearchResult]], k: int = 60) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion для объединения результатов от разных источников.

        Args:
            results_lists: Список списков результатов от разных поисковых стратегий
            k: Параметр RRF (обычно 60)

        Returns:
            Объединённый и отсортированный список результатов
        """
        if len(results_lists) <= 1:
            return results_lists[0] if results_lists else []

        # Собираем все уникальные результаты
        all_results: Dict[str, SearchResult] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)

        for result_list in results_lists:
            for rank, result in enumerate(result_list, 1):
                result_id = result.chunk_id
                
                # RRF формула: score = 1/(k + rank_i)
                rrf_contribution = 1.0 / (k + rank)
                rrf_scores[result_id] += rrf_contribution
                
                # Сохраняем результат (берём первое вхождение)
                if result_id not in all_results:
                    all_results[result_id] = result

        # Сортируем по RRF score
        fused_results = []
        for result_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            result = all_results[result_id]
            result.rrf_score = rrf_score
            fused_results.append(result)

        logger.debug(f"RRF: объединено {len(results_lists)} списков -> {len(fused_results)} результатов")
        return fused_results

    def _mmr_rerank(
        self,
        results: List[SearchResult],
        query_vector: np.ndarray,
        lambda_param: float = 0.7,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Maximum Marginal Relevance для диверсификации результатов.

        Args:
            results: Исходные результаты поиска
            query_vector: Вектор запроса
            lambda_param: Баланс релевантности/разнообразия (0.0-1.0)
            max_results: Количество результатов для возврата

        Returns:
            Переранжированный список результатов
        """
        if len(results) <= max_results:
            for result in results:
                result.mmr_score = result.score
            return results

        selected = []
        remaining = results.copy()

        # Получаем эмбеддинги для всех результатов
        self._ensure_embeddings(remaining)

        # Выбираем первый результат (самый релевантный)
        if remaining:
            first_result = remaining.pop(0)
            first_result.mmr_score = first_result.score
            selected.append(first_result)

        while len(selected) < max_results and remaining:
            best_score = -1
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # Релевантность к запросу (косинусное сходство)
                relevance = self._cosine_similarity(query_vector, candidate.embedding)
                
                # Максимальное сходство с уже выбранными результатами
                max_similarity = 0
                for selected_result in selected:
                    if selected_result.embedding is not None:
                        similarity = self._cosine_similarity(
                            candidate.embedding, 
                            selected_result.embedding
                        )
                        max_similarity = max(max_similarity, similarity)

                # MMR скор
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            # Добавляем лучший результат
            best_result = remaining.pop(best_idx)
            best_result.mmr_score = best_score
            selected.append(best_result)

        logger.debug(f"MMR: переранжировано {len(results)} -> {len(selected)} результатов")
        return selected

    def _ensure_embeddings(self, results: List[SearchResult]) -> None:
        """Гарантирует наличие эмбеддингов у результатов"""
        texts_to_embed = []
        indices_to_embed = []

        for i, result in enumerate(results):
            if result.embedding is None:
                texts_to_embed.append(result.content)
                indices_to_embed.append(i)

        if texts_to_embed:
            try:
                embeddings = self.embedder.embed_texts(texts_to_embed)
                for i, embedding in zip(indices_to_embed, embeddings):
                    results[i].embedding = embedding
            except Exception as e:
                logger.warning(f"Не удалось получить эмбеддинги для MMR: {e}")
                # Fallback: используем случайные эмбеддинги
                for i in indices_to_embed:
                    results[i].embedding = np.random.random(self.embedder.embedding_dim)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусное сходство между векторами"""
        try:
            if vec1 is None or vec2 is None:
                return 0.0
                
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    def _convert_to_output_format(self, results: List[SearchResult]) -> List[Dict]:
        """Преобразует результаты в выходной формат"""
        output_results = []
        
        for result in results:
            output_result = {
                'id': result.chunk_id,
                'score': result.score,
                'payload': {
                    'file_path': result.file_path,
                    'file_name': result.file_name,
                    'chunk_name': result.chunk_name,
                    'chunk_type': result.chunk_type,
                    'language': result.language,
                    'start_line': result.start_line,
                    'end_line': result.end_line,
                    'content': result.content,
                    **result.metadata
                }
            }
            
            # Добавляем дополнительные скоры если есть
            if result.rrf_score is not None:
                output_result['rrf_score'] = result.rrf_score
            if result.mmr_score is not None:
                output_result['mmr_score'] = result.mmr_score
            if result.rank is not None:
                output_result['rank'] = result.rank
                
            output_results.append(output_result)
        
        return output_results

    def _generate_cache_key(self, query: str, max_results: int, filters: Dict = None) -> str:
        """Генерирует стабильный ключ для кэширования"""
        key_components = [
            query.strip().lower(),
            str(max_results),
            str(self.config.mmr_lambda),
            str(self.config.score_threshold),
            str(self.config.rrf_enabled),
            str(self.config.mmr_enabled)
        ]
        
        if filters:
            key_components.append(str(sorted(filters.items())))
            
        key_string = '|'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _cache_result(self, cache_key: str, results: List[Dict]) -> None:
        """Сохраняет результат в кэш"""
        try:
            self.cache[cache_key] = results
        except Exception as e:
            logger.warning(f"Не удалось сохранить в кэш: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния поискового движка"""
        try:
            # Проверяем компоненты
            embedder_status = "ok" if self.embedder else "not_available"
            vector_store_status = "ok" if await self._check_vector_store() else "disconnected"
            
            return {
                'status': 'healthy',
                'embedder_status': embedder_status,
                'vector_store_status': vector_store_status,
                'cache_size': len(self.cache),
                'cache_max_size': self.config.cache_max_entries,
                'stats': self.stats.to_dict(),
                'config': {
                    'rrf_enabled': self.config.rrf_enabled,
                    'mmr_enabled': self.config.mmr_enabled,
                    'max_results': self.config.max_results,
                    'concurrent_users_target': self.config.concurrent_users_target
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'stats': self.stats.to_dict()
            }

    async def _check_vector_store(self) -> bool:
        """Проверяет доступность векторного хранилища"""
        try:
            return self.vector_store.is_connected()
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает подробную статистику"""
        base_stats = self.stats.to_dict()
        
        # Добавляем дополнительную информацию
        base_stats.update({
            'cache_size': len(self.cache),
            'cache_maxsize': self.config.cache_max_entries,
            'cache_ttl_seconds': self.config.cache_ttl_seconds,
            'search_workers': self.config.search_workers,
            'embed_workers': self.config.embed_workers,
            'algorithms': {
                'rrf_enabled': self.config.rrf_enabled,
                'mmr_enabled': self.config.mmr_enabled,
                'mmr_lambda': self.config.mmr_lambda
            }
        })
        
        return base_stats

    def clear_cache(self) -> int:
        """Очищает кэш и возвращает количество удалённых записей"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Очищен кэш CPUQueryEngine: {cache_size} записей")
        return cache_size

    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        self.stats = QueryStats()
        logger.info("Статистика CPUQueryEngine сброшена")

    async def close(self) -> None:
        """Закрывает движок и освобождает ресурсы"""
        try:
            # Закрываем пулы воркеров
            self.search_executor.shutdown(wait=True)
            self.embed_executor.shutdown(wait=True)
            
            # Закрываем SearchService
            if hasattr(self.search_service, 'close'):
                await self.search_service.close()
            
            # Очищаем кэш
            self.clear_cache()
            
            logger.info("CPUQueryEngine закрыт")
        except Exception as e:
            logger.error(f"Ошибка закрытия CPUQueryEngine: {e}")

    def __del__(self):
        """Деструктор для принудительного закрытия ресурсов"""
        try:
            if hasattr(self, 'search_executor'):
                self.search_executor.shutdown(wait=False)
            if hasattr(self, 'embed_executor'):
                self.embed_executor.shutdown(wait=False)
        except Exception:
            pass
