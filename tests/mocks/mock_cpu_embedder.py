"""
Mock реализация CPUEmbedder для тестирования без сетевых соединений.

MockCPUEmbedder имитирует поведение настоящего CPUEmbedder, но:
- Не требует загрузки моделей из интернета
- Генерирует детерминированные фиктивные эмбеддинги
- Поддерживает все методы оригинального класса
- Автоматически активируется при обнаружении pytest-socket
"""

import os
import sys
import time
import logging
import hashlib
from typing import List, Optional
import numpy as np

from config import EmbeddingConfig, ParallelismConfig

logger = logging.getLogger(__name__)


def is_socket_disabled() -> bool:
    """
    Определяет, заблокированы ли сетевые соединения (pytest-socket).
    
    Returns:
        True если pytest-socket активен или если мы в CI окружении
    """
    # Проверяем наличие pytest-socket в модулях
    if 'pytest_socket' in sys.modules:
        return True
    
    # Проверяем переменные окружения CI
    ci_indicators = [
        'CI', 'GITHUB_ACTIONS', 'TRAVIS', 'CIRCLECI', 
        'JENKINS_URL', 'BUILDKITE', 'GITLAB_CI'
    ]
    
    if any(os.getenv(var) for var in ci_indicators):
        return True
    
    # Проверяем наличие --disable-socket в sys.argv
    if '--disable-socket' in sys.argv:
        return True
        
    return False


class MockCPUEmbedder:
    """
    Mock версия CPUEmbedder для тестирования без сетевых соединений.
    
    Генерирует детерминированные фиктивные эмбеддинги на основе hash текста.
    Поддерживает все методы оригинального класса для полной совместимости.
    """
    
    def __init__(self, embedding_config: EmbeddingConfig, parallelism_config: ParallelismConfig):
        """
        Инициализация mock эмбеддера.
        
        Args:
            embedding_config: Конфигурация эмбеддингов
            parallelism_config: Конфигурация параллелизма
        """
        self.embedding_config = embedding_config
        self.parallelism_config = parallelism_config
        
        # Имитируем настройку provider_name как в оригинале
        self.provider_name = "mock_" + embedding_config.provider
        self.model = None  # Mock модель отсутствует
        self._is_warmed_up = False
        self._current_batch_size = embedding_config.batch_size_min
        
        # Вычисляем размерность эмбеддингов с поддержкой Jina v3
        if embedding_config.truncate_dim > 0:
            self.embedding_dim = embedding_config.truncate_dim
        else:
            # Используем стандартные размерности для известных моделей
            model_dimensions = {
                'BAAI/bge-small-en-v1.5': 384,
                'all-MiniLM-L6-v2': 384,
                'all-mpnet-base-v2': 768,
                'jinaai/jina-embeddings-v3': 1024,  # Jina v3 поддержка
                'sentence-transformers/all-MiniLM-L6-v2': 384,
            }
            self.embedding_dim = model_dimensions.get(embedding_config.model_name, 1024)  # Дефолт 1024d для Jina v3
        
        # Статистика производительности (имитируем реальную)
        self.stats = {
            'total_texts': 0,
            'total_time': 0.0,
            'batch_count': 0,
            'oom_fallbacks': 0,
            'provider_fallbacks': 0
        }
        
        logger.info(f"Инициализирован MockCPUEmbedder (provider: {self.provider_name}, dim: {self.embedding_dim})")
    
    def _generate_deterministic_embedding(self, text: str) -> np.ndarray:
        """
        Генерирует детерминированный эмбеддинг на основе hash текста.
        
        Args:
            text: Входной текст
            
        Returns:
            numpy array размерности [embedding_dim]
        """
        # Используем hash текста как seed для воспроизводимости
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16) % (2**32)
        
        # Генерируем детерминированный вектор
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
        
        # Нормализация если требуется
        if self.embedding_config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def embed_texts(self, texts: List[str], deadline_ms: int = 1500, task: Optional[str] = None) -> np.ndarray:
        """
        Генерирует фиктивные эмбеддинги для списка текстов с поддержкой dual task (Jina v3).
        
        Args:
            texts: Список текстов для кодирования
            deadline_ms: Максимальное время на обработку (имитируется)
            task: Задача для специализированных эмбеддингов (retrieval.query/passage)
            
        Returns:
            numpy массив эмбеддингов размером [len(texts), embedding_dim]
        """
        if not texts:
            return np.array([])
        
        start_time = time.time()
        
        # Имитируем прогрев если не был выполнен
        if self.embedding_config.warmup_enabled and not self._is_warmed_up:
            self.warmup()
        
        # Имитируем dual task switching для Jina v3
        effective_task = task or getattr(self.embedding_config, 'task_passage', 'retrieval.passage')
        if hasattr(self.embedding_config, 'model_name') and 'jina' in self.embedding_config.model_name.lower():
            logger.debug(f"Mock Jina v3 dual task switching: {effective_task}")
        
        # Генерируем эмбеддинги для всех текстов с учетом task
        embeddings = []
        for text in texts:
            # Добавляем task в hash для различных эмбеддингов query vs passage
            text_with_task = f"{text}|task:{effective_task}" if effective_task else text
            embedding = self._generate_deterministic_embedding(text_with_task)
            embeddings.append(embedding)
        
        # Имитируем время обработки (пропорционально количеству текстов)
        processing_time = len(texts) * 0.001  # 1ms на текст
        if processing_time > 0.01:  # Минимальная задержка для реалистичности
            time.sleep(min(processing_time, 0.1))  # Максимум 100ms задержки
        
        result = np.array(embeddings)
        
        # Обновляем статистику
        total_time = time.time() - start_time
        self.stats['total_texts'] += len(texts)
        self.stats['total_time'] += total_time
        self.stats['batch_count'] += 1
        
        task_info = f" (task: {effective_task})" if effective_task else ""
        logger.debug(
            f"MockEmbedder обработал {len(texts)} текстов за {total_time:.3f}s "
            f"(размерность: {result.shape}){task_info}"
        )
        
        return result
    
    def calculate_batch_size(self, queue_len: int) -> int:
        """
        Имитирует вычисление адаптивного размера батча.
        
        Args:
            queue_len: Количество текстов в очереди
            
        Returns:
            Размер батча (имитированный)
        """
        # Упрощенная логика для mock'а
        min_batch = self.embedding_config.batch_size_min
        max_batch = self.embedding_config.batch_size_max
        
        if queue_len <= 10:
            return min_batch
        elif queue_len <= 100:
            return min(max_batch, min_batch + queue_len // 10)
        else:
            return max_batch
    
    def warmup(self) -> None:
        """
        Имитация прогрева модели.
        """
        if self._is_warmed_up:
            return
        
        logger.info("Прогрев mock модели...")
        start_time = time.time()
        
        # Имитируем прогрев с коротким текстом
        dummy_text = "This is a warmup text for mock model initialization."
        _ = self._generate_deterministic_embedding(dummy_text)
        
        # Небольшая задержка для реалистичности
        time.sleep(0.001)
        
        warmup_time = time.time() - start_time
        self._is_warmed_up = True
        
        logger.info(f"Прогрев mock модели завершён за {warmup_time:.3f}s")
    
    def get_stats(self) -> dict:
        """
        Возвращает статистику производительности mock эмбеддера.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.stats.copy()
        
        # Вычисляем дополнительные метрики
        if stats['total_time'] > 0:
            stats['avg_texts_per_second'] = stats['total_texts'] / stats['total_time']
        else:
            stats['avg_texts_per_second'] = 0.0
        
        if stats['batch_count'] > 0:
            stats['avg_batch_time'] = stats['total_time'] / stats['batch_count']
        else:
            stats['avg_batch_time'] = 0.0
        
        # Дополнительные поля для совместимости
        stats['provider'] = self.provider_name
        stats['model_name'] = self.embedding_config.model_name
        stats['is_warmed_up'] = self._is_warmed_up
        stats['current_batch_size'] = self._current_batch_size
        stats['embedding_dim'] = self.embedding_dim
        stats['is_mock'] = True  # Флаг что это mock
        
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику производительности"""
        self.stats = {
            'total_texts': 0,
            'total_time': 0.0,
            'batch_count': 0,
            'oom_fallbacks': 0,
            'provider_fallbacks': 0
        }
        logger.info("Статистика mock эмбеддера сброшена")
    
    def __del__(self):
        """Очистка ресурсов mock эмбеддера"""
        try:
            # Mock очистка - нечего освобождать
            logger.debug(f"MockCPUEmbedder очищен (обработано {self.stats['total_texts']} текстов)")
        except Exception as e:
            logger.error(f"Ошибка при очистке mock эмбеддера: {e}")


class MockEmbedderError(Exception):
    """Исключение для имитации ошибок эмбеддера в тестах"""
    pass


def create_mock_embedder_with_error(embedding_config: EmbeddingConfig, 
                                  parallelism_config: ParallelismConfig,
                                  error_message: str = "Mock embedder error") -> MockCPUEmbedder:
    """
    Создаёт mock эмбеддер, который будет генерировать ошибки.
    Полезно для тестирования обработки ошибок.
    
    Args:
        embedding_config: Конфигурация эмбеддингов
        parallelism_config: Конфигурация параллелизма
        error_message: Сообщение об ошибке
        
    Returns:
        MockCPUEmbedder который будет падать при embed_texts
    """
    mock_embedder = MockCPUEmbedder(embedding_config, parallelism_config)
    
    # Заменяем метод embed_texts на версию с ошибкой
    def embed_with_error(*args, **kwargs):
        raise MockEmbedderError(error_message)
    
    mock_embedder.embed_texts = embed_with_error
    
    return mock_embedder


def should_use_mock_embedder() -> bool:
    """
    Определяет, следует ли использовать mock эмбеддер вместо реального.
    
    Returns:
        True если нужно использовать mock
    """
    # Используем mock если сеть заблокирована или мы в CI
    if is_socket_disabled():
        return True
    
    # Проверяем явную переменную окружения
    if os.getenv('USE_MOCK_EMBEDDER', '').lower() in ('1', 'true', 'yes'):
        return True
    
    return False
