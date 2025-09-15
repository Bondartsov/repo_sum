"""
HTTP клиент для удалённого векторного хранилища через RAG-as-a-Service на VM.

Заменяет прямое подключение к Qdrant на HTTP запросы к FastAPI сервису на VM,
где работает Qdrant с 1024d векторами и гибридным поиском.
"""

import os
import logging
import time
import aiohttp
import asyncio
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RemoteVMVectorStore:
    """
    HTTP клиент для удалённого векторного хранилища на VM.
    
    Возможности:
    - HTTP запросы к Qdrant через FastAPI сервис на VM
    - Индексация документов через HTTP
    - Гибридный поиск (dense + sparse) через HTTP
    - Health check удалённого сервиса
    """
    
    def __init__(self, vector_store_config=None):
        """
        Инициализация удалённого векторного хранилища.
        
        Args:
            vector_store_config: Конфигурация векторного хранилища (игнорируется, для совместимости)
        """
        # Читаем конфигурацию из переменных окружения
        self.search_endpoint = os.getenv("RAG_SEARCH_ENDPOINT", "http://10.61.11.54:8000/search")
        self.index_endpoint = os.getenv("RAG_INDEX_ENDPOINT", "http://10.61.11.54:8000/index")
        self.service_host = os.getenv("RAG_SERVICE_HOST", "10.61.11.54")
        self.service_port = int(os.getenv("RAG_SERVICE_PORT", "8000"))
        
        # HTTP клиент настройки
        self.timeout = aiohttp.ClientTimeout(total=60, connect=10)  # Увеличены таймауты для индексации
        self.max_retries = 3
        self.retry_delay = 2.0
        
        # Статистика
        self.stats = {
            'total_searches': 0,
            'total_indexed': 0,
            'total_search_time': 0.0,
            'total_index_time': 0.0,
            'error_count': 0,
            'retry_count': 0
        }
        
        self._connected = False
        self._collection_exists = False
        
        logger.info(f"RemoteVMVectorStore инициализирован: поиск={self.search_endpoint}, индексация={self.index_endpoint}")
    
    async def initialize_collection(self, recreate: bool = False) -> None:
        """
        Инициализирует коллекцию через удалённый сервис.
        
        Args:
            recreate: Пересоздать коллекцию если она уже существует
        """
        try:
            health_info = await self.health_check()
            
            if health_info['status'] == 'connected':
                self._connected = True
                self._collection_exists = health_info.get('collection_status') == 'exists'
                
                if recreate and self._collection_exists:
                    logger.info("Пересоздание коллекции через удалённый сервис...")
                    # Пересоздание будет обработано на стороне VM сервиса
                    await self._recreate_collection()
                
                logger.info(f"Коллекция {'существует' if self._collection_exists else 'будет создана'} на VM")
            else:
                raise ConnectionError(f"Не удалось подключиться к VM сервису: {health_info.get('error')}")
                
        except Exception as e:
            logger.error(f"Ошибка инициализации коллекции через VM: {e}")
            raise
    
    async def _recreate_collection(self) -> None:
        """Пересоздаёт коллекцию через удалённый сервис"""
        try:
            recreate_endpoint = f"http://{self.service_host}:{self.service_port}/collection/recreate"
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(recreate_endpoint) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Коллекция пересоздана: {result}")
                        self._collection_exists = True
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка пересоздания коллекции: HTTP {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Ошибка пересоздания коллекции через VM: {e}")
            raise
    
    async def index_documents(self, points: List[Dict]) -> int:
        """
        Индексирует документы через удалённый сервис.
        
        Args:
            points: Список точек для индексации (формат: [{"id": ..., "text": ..., "metadata": ...}, ...])
            
        Returns:
            Количество успешно проиндексированных документов
        """
        if not points:
            return 0
        
        start_time = time.time()
        
        try:
            # Подготовка данных для удалённого сервиса
            payload = {
                "documents": [
                    {
                        "id": str(point.get("id", f"doc_{i}")),
                        "text": point.get("text", ""),
                        "metadata": point.get("metadata", {}),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    for i, point in enumerate(points)
                ],
                "batch_size": min(512, len(points)),  # Батчевая обработка на сервере
                "recreate": False
            }
            
            # HTTP запрос на индексацию
            indexed_count = await self._make_index_request_with_retry(payload)
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_indexed'] += indexed_count
            self.stats['total_index_time'] += elapsed_time
            
            logger.info(
                f"Индексация через VM завершена: {indexed_count}/{len(points)} документов "
                f"за {elapsed_time:.3f}s ({indexed_count/elapsed_time:.1f} док/с)"
            )
            
            return indexed_count
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Ошибка индексации документов через VM: {e}")
            raise
    
    async def _make_index_request_with_retry(self, payload: Dict[str, Any]) -> int:
        """
        Выполняет запрос на индексацию с retry логикой.
        
        Args:
            payload: Данные для индексации
            
        Returns:
            Количество проиндексированных документов
        """
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(
                        self.index_endpoint,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # Ожидаем формат: {"indexed_count": 123, "status": "success"}
                            if "indexed_count" in result:
                                return result["indexed_count"]
                            else:
                                raise ValueError(f"Неожиданный формат ответа индексации: {result.keys()}")
                        
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(
                                f"HTTP {response.status}: {error_text}"
                            )
            
            except Exception as e:
                logger.warning(f"Ошибка индексации (попытка {attempt + 1}): {e}")
                self.stats['retry_count'] += 1
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    raise  # Последняя попытка - пробрасываем ошибку
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
        sparse_vector: Optional[Dict[int, float]] = None
    ) -> List[Dict]:
        """
        Выполняет поиск через удалённый сервис.
        
        Args:
            query_vector: Вектор запроса (не используется напрямую - текст будет векторизован на VM)
            top_k: Количество результатов
            filters: Фильтры по метаданным
            use_hybrid: Использовать гибридный поиск
            sparse_vector: Разреженный вектор (не используется - обрабатывается на VM)
            
        Returns:
            Список результатов поиска
        """
        start_time = time.time()
        
        try:
            # Подготовка запроса для удалённого поиска
            # Поскольку у нас нет текста запроса здесь, используем заглушку
            # В реальном использовании query должен содержать текст
            payload = {
                "query": "search_query_placeholder",  # Будет заменен в search_service
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "filters": filters or {},
                "task": "retrieval.query"
            }
            
            # HTTP запрос на поиск
            results = await self._make_search_request_with_retry(payload)
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += elapsed_time
            
            logger.debug(f"Поиск через VM завершён: {len(results)} результатов за {elapsed_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Ошибка поиска через VM: {e}")
            return []  # Возвращаем пустой результат при ошибке
    
    async def search_by_text(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Выполняет поиск по тексту через удалённый сервис.
        
        Args:
            query_text: Текст запроса
            top_k: Количество результатов
            filters: Фильтры по метаданным
            use_hybrid: Использовать гибридный поиск
            
        Returns:
            Список результатов поиска
        """
        start_time = time.time()
        
        try:
            payload = {
                "query": query_text,
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "filters": filters or {},
                "task": "retrieval.query"
            }
            
            results = await self._make_search_request_with_retry(payload)
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += elapsed_time
            
            logger.debug(f"Текстовый поиск через VM: '{query_text[:50]}...' -> {len(results)} результатов за {elapsed_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Ошибка текстового поиска через VM: {e}")
            return []
    
    async def _make_search_request_with_retry(self, payload: Dict[str, Any]) -> List[Dict]:
        """
        Выполняет запрос на поиск с retry логикой.
        
        Args:
            payload: Данные запроса
            
        Returns:
            Список результатов поиска
        """
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(
                        self.search_endpoint,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # Ожидаем формат: {"results": [...], "query_time": 0.123}
                            if "results" in result:
                                return result["results"]
                            else:
                                raise ValueError(f"Неожиданный формат ответа поиска: {result.keys()}")
                        
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(
                                f"HTTP {response.status}: {error_text}"
                            )
            
            except Exception as e:
                logger.warning(f"Ошибка поиска (попытка {attempt + 1}): {e}")
                self.stats['retry_count'] += 1
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise  # Последняя попытка - пробрасываем ошибку
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверяет состояние удалённого векторного хранилища.
        
        Returns:
            Информация о состоянии сервиса
        """
        health_info = {
            'status': 'unknown',
            'service_host': self.service_host,
            'service_port': self.service_port,
            'search_endpoint': self.search_endpoint,
            'index_endpoint': self.index_endpoint,
            'collection_status': 'unknown',
            'error': None
        }
        
        try:
            health_endpoint = f"http://{self.service_host}:{self.service_port}/health"
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(health_endpoint) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        health_info['status'] = 'connected'
                        health_info['collection_status'] = result.get('collection_status', 'unknown')
                        health_info['qdrant_status'] = result.get('qdrant_status', 'unknown')
                        health_info['vector_count'] = result.get('vector_count', 0)
                        
                        # Обновляем внутреннее состояние
                        self._connected = True
                        self._collection_exists = health_info['collection_status'] == 'exists'
                        
                    else:
                        error_text = await response.text()
                        health_info['status'] = 'error'
                        health_info['error'] = f"HTTP {response.status}: {error_text}"
                        self._connected = False
        
        except Exception as e:
            health_info['status'] = 'error'
            health_info['error'] = str(e)
            self._connected = False
            
        return health_info
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Получает информацию о коллекции через удалённый сервис.
        
        Returns:
            Информация о коллекции
        """
        try:
            info_endpoint = f"http://{self.service_host}:{self.service_port}/collection/info"
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(info_endpoint) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {
                            'error': f"HTTP {response.status}: {error_text}",
                            'status': 'error'
                        }
        
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        stats = self.stats.copy()
        
        # Вычисляем средние показатели
        if stats['total_searches'] > 0:
            stats['avg_search_time'] = stats['total_search_time'] / stats['total_searches']
        else:
            stats['avg_search_time'] = 0.0
            
        if stats['total_indexed'] > 0 and stats['total_index_time'] > 0:
            stats['avg_index_rate'] = stats['total_indexed'] / stats['total_index_time']
        else:
            stats['avg_index_rate'] = 0.0
        
        stats.update({
            'connected': self._connected,
            'collection_exists': self._collection_exists,
            'service_url': f"http://{self.service_host}:{self.service_port}",
            'search_endpoint': self.search_endpoint,
            'index_endpoint': self.index_endpoint
        })
        
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        self.stats = {
            'total_searches': 0,
            'total_indexed': 0,
            'total_search_time': 0.0,
            'total_index_time': 0.0,
            'error_count': 0,
            'retry_count': 0
        }
        logger.info("Статистика RemoteVMVectorStore сброшена")
    
    async def close(self) -> None:
        """Закрывает соединения (для удалённого клиента не требуется)"""
        self._connected = False
        logger.info("RemoteVMVectorStore закрыт")


# Обратная совместимость: алиас для старого класса
QdrantVectorStore = RemoteVMVectorStore
