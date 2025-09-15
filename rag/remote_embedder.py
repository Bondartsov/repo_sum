"""
HTTP клиент для удалённых эмбеддингов через RAG-as-a-Service на VM.

Заменяет локальную загрузку моделей на HTTP запросы к FastAPI сервису на VM,
где работает Jina v3 с полными 1024d векторами.
"""

import os
import logging
import time
import aiohttp
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np
import json

logger = logging.getLogger(__name__)


class RemoteVMEmbedder:
    """
    HTTP клиент для получения эмбеддингов от Jina v3 сервиса на VM.
    
    Возможности:
    - HTTP запросы к FastAPI сервису на VM (10.61.11.54:8000)
    - Jina v3 dual task support (retrieval.query/passage)
    - Matryoshka сжатие (1024d → 384d)
    - Батчевая обработка через HTTP
    - Fallback и retry логика
    """
    
    def __init__(self, embedding_config=None, parallelism_config=None):
        """
        Инициализация удалённого эмбеддера.
        
        Args:
            embedding_config: Конфигурация эмбеддингов (игнорируется, для совместимости)
            parallelism_config: Конфигурация параллелизма (игнорируется, для совместимости)
        """
        # Читаем конфигурацию из переменных окружения
        self.embeddings_endpoint = os.getenv("RAG_EMBEDDINGS_ENDPOINT", "http://10.61.11.54:8000/embeddings")
        self.service_host = os.getenv("RAG_SERVICE_HOST", "10.61.11.54")
        self.service_port = int(os.getenv("RAG_SERVICE_PORT", "8000"))
        
        # Конфигурация эмбеддингов
        self.model_name = os.getenv("EMB_MODEL_ID", "jinaai/jina-embeddings-v3")
        self.embedding_dim = int(os.getenv("EMB_DIM", "1024"))
        self.truncate_dim = int(os.getenv("EMB_TRUNCATE_DIM", "384"))
        
        # HTTP клиент настройки
        self.timeout = aiohttp.ClientTimeout(total=30, connect=5)
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'total_texts': 0,
            'total_time': 0.0,
            'error_count': 0,
            'retry_count': 0,
            'avg_response_time': 0.0
        }
        
        self._is_warmed_up = False
        logger.info(f"RemoteVMEmbedder инициализирован: {self.embeddings_endpoint}")
    
    async def embed_texts(
        self, 
        texts: List[str], 
        task: Optional[str] = None,
        deadline_ms: int = 30000
    ) -> np.ndarray:
        """
        Получает эмбеддинги текстов от удалённого Jina v3 сервиса.
        
        Args:
            texts: Список текстов для эмбеддинга
            task: Задача для dual task ("retrieval.query" или "retrieval.passage")
            deadline_ms: Максимальное время ожидания в миллисекундах
            
        Returns:
            numpy массив эмбеддингов размером [len(texts), truncate_dim]
        """
        if not texts:
            return np.array([])
        
        start_time = time.time()
        
        try:
            # Подготовка запроса
            payload = {
                "texts": texts,
                "task": task or "retrieval.passage",  # По умолчанию для индексации
                "truncate_dim": self.truncate_dim,
                "normalize": True
            }
            
            # HTTP запрос к VM сервису
            embeddings = await self._make_request_with_retry(payload, deadline_ms)
            
            # Конвертируем в numpy массив
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_requests'] += 1
            self.stats['total_texts'] += len(texts)
            self.stats['total_time'] += elapsed_time
            self.stats['avg_response_time'] = self.stats['total_time'] / self.stats['total_requests']
            
            logger.debug(
                f"Получены эмбеддинги от VM: {len(texts)} текстов, "
                f"размерность: {embeddings_array.shape}, время: {elapsed_time:.3f}s"
            )
            
            return embeddings_array
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Ошибка получения эмбеддингов от VM: {e}")
            
            # Fallback: возвращаем нулевые векторы
            logger.warning(f"Используем fallback нулевые векторы для {len(texts)} текстов")
            return np.zeros((len(texts), self.truncate_dim), dtype=np.float32)
    
    async def _make_request_with_retry(
        self, 
        payload: Dict[str, Any], 
        deadline_ms: int
    ) -> List[List[float]]:
        """
        Выполняет HTTP запрос с retry логикой.
        
        Args:
            payload: Данные для отправки
            deadline_ms: Дедлайн в миллисекундах
            
        Returns:
            Список эмбеддингов
        """
        timeout = aiohttp.ClientTimeout(total=deadline_ms/1000.0)
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.embeddings_endpoint,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # Ожидаем формат: {"embeddings": [[...], [...], ...]}
                            if "embeddings" in result:
                                return result["embeddings"]
                            else:
                                raise ValueError(f"Неожиданный формат ответа: {result.keys()}")
                        
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(
                                f"HTTP {response.status}: {error_text}"
                            )
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout при запросе к VM (попытка {attempt + 1})")
                self.stats['retry_count'] += 1
                
            except Exception as e:
                logger.warning(f"Ошибка HTTP запроса (попытка {attempt + 1}): {e}")
                self.stats['retry_count'] += 1
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    raise  # Последняя попытка - пробрасываем ошибку
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверяет состояние удалённого сервиса.
        
        Returns:
            Информация о состоянии сервиса
        """
        health_info = {
            'status': 'unknown',
            'service_url': self.embeddings_endpoint,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'truncate_dim': self.truncate_dim,
            'error': None
        }
        
        try:
            # Тестовый запрос
            test_payload = {
                "texts": ["test"],
                "task": "retrieval.query",
                "truncate_dim": self.truncate_dim,
                "normalize": True
            }
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    self.embeddings_endpoint,
                    json=test_payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        if "embeddings" in result and len(result["embeddings"]) > 0:
                            actual_dim = len(result["embeddings"][0])
                            health_info['status'] = 'healthy'
                            health_info['actual_embedding_dim'] = actual_dim
                            
                            if actual_dim != self.truncate_dim:
                                health_info['warning'] = f"Размерность не соответствует ожидаемой: {actual_dim} vs {self.truncate_dim}"
                        else:
                            health_info['status'] = 'error'
                            health_info['error'] = "Некорректный формат ответа"
                    else:
                        error_text = await response.text()
                        health_info['status'] = 'error'
                        health_info['error'] = f"HTTP {response.status}: {error_text}"
        
        except Exception as e:
            health_info['status'] = 'error'
            health_info['error'] = str(e)
            
        return health_info
    
    def warmup(self) -> None:
        """
        Прогрев удалённого сервиса (опционально).
        """
        if self._is_warmed_up:
            return
            
        logger.info("Прогрев удалённого VM сервиса...")
        
        try:
            # Асинхронный прогрев в синхронном контексте
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если уже в event loop, создаем задачу
                asyncio.create_task(self._async_warmup())
            else:
                # Запускаем новый event loop
                loop.run_until_complete(self._async_warmup())
                
        except Exception as e:
            logger.warning(f"Ошибка прогрева VM сервиса: {e}")
            # Не критично, продолжаем работу
            
    async def _async_warmup(self) -> None:
        """Асинхронный прогрев сервиса"""
        try:
            health_info = await self.health_check()
            
            if health_info['status'] == 'healthy':
                self._is_warmed_up = True
                logger.info(f"VM сервис готов: {health_info.get('actual_embedding_dim', 'N/A')}d векторы")
            else:
                logger.warning(f"VM сервис не готов: {health_info.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Ошибка асинхронного прогрева: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        stats = self.stats.copy()
        stats.update({
            'service_url': self.embeddings_endpoint,
            'model_name': self.model_name,
            'is_warmed_up': self._is_warmed_up,
            'embedding_dim': self.embedding_dim,
            'truncate_dim': self.truncate_dim
        })
        
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        self.stats = {
            'total_requests': 0,
            'total_texts': 0,
            'total_time': 0.0,
            'error_count': 0,
            'retry_count': 0,
            'avg_response_time': 0.0
        }
        logger.info("Статистика RemoteVMEmbedder сброшена")


# Обратная совместимость: алиас для старого класса
CPUEmbedder = RemoteVMEmbedder
