"""
CPU-оптимизированный эмбеддер для RAG системы.

Поддерживает FastEmbed как основной провайдер с fallback на sentence-transformers.
Адаптивные размеры батчей, управление потоками и контроль времени отклика.
"""

import os
import logging
import time
import gc
from typing import List, Optional, Union, Tuple
import numpy as np
import psutil

# Установка переменных окружения для управления потоками
# Должно быть выполнено до импорта torch и других библиотек
def _set_thread_environment(parallelism_config):
    """Устанавливает переменные окружения для управления потоками"""
    os.environ["OMP_NUM_THREADS"] = str(parallelism_config.omp_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(parallelism_config.mkl_num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(parallelism_config.omp_num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(parallelism_config.omp_num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(parallelism_config.omp_num_threads)

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import EmbeddingConfig, ParallelismConfig

logger = logging.getLogger(__name__)


class CPUEmbedder:
    """
    CPU-оптимизированный эмбеддер с поддержкой FastEmbed и Sentence Transformers.
    
    Основные возможности:
    - FastEmbed как основной провайдер (ONNX Runtime, квантованные веса)
    - Sentence Transformers как fallback с поддержкой int8 precision
    - Адаптивные размеры батчей на основе доступной памяти
    - Управление потоками для CPU оптимизации
    - Контроль времени отклика с deadline
    - Graceful degradation при нехватке ресурсов
    """
    
    def __init__(self, embedding_config: EmbeddingConfig, parallelism_config: ParallelismConfig):
        """
        Инициализация эмбеддера.
        
        Args:
            embedding_config: Конфигурация эмбеддингов
            parallelism_config: Конфигурация управления потоками
        """
        self.embedding_config = embedding_config
        self.parallelism_config = parallelism_config
        
        # Установка переменных окружения для потоков
        _set_thread_environment(parallelism_config)
        
        # Настройка torch потоков, если доступен
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            torch.set_num_threads(parallelism_config.torch_num_threads)
            # Отключаем градиенты для инференса
            torch.set_grad_enabled(False)
        
        self.model = None
        self._is_warmed_up = False
        self._current_batch_size = embedding_config.batch_size_min
        
        # Статистика производительности
        self.stats = {
            'total_texts': 0,
            'total_time': 0.0,
            'batch_count': 0,
            'oom_fallbacks': 0,
            'provider_fallbacks': 0
        }
        
        # Инициализация модели
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Инициализирует модель в зависимости от конфигурации провайдера"""
        logger.info(f"Инициализация эмбеддера с провайдером: {self.embedding_config.provider}")
        
        try:
            if self.embedding_config.provider == "fastembed" and FASTEMBED_AVAILABLE:
                self._initialize_fastembed()
            elif self.embedding_config.provider == "sentence-transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
                self._initialize_sentence_transformers()
            else:
                # Fallback логика
                if FASTEMBED_AVAILABLE:
                    logger.warning(f"Провайдер {self.embedding_config.provider} недоступен, используем FastEmbed")
                    self._initialize_fastembed()
                    self.stats['provider_fallbacks'] += 1
                elif SENTENCE_TRANSFORMERS_AVAILABLE:
                    logger.warning(f"Провайдер {self.embedding_config.provider} недоступен, используем Sentence Transformers")
                    self._initialize_sentence_transformers()
                    self.stats['provider_fallbacks'] += 1
                else:
                    raise ImportError("Ни FastEmbed, ни Sentence Transformers не доступны")
                    
        except Exception as e:
            logger.error(f"Ошибка инициализации основного провайдера: {e}")
            # Попытка fallback
            self._try_fallback_initialization()
    
    def _initialize_fastembed(self) -> None:
        """Инициализирует FastEmbed модель"""
        try:
            logger.info(f"Загрузка FastEmbed модели: {self.embedding_config.model_name}")
            
            # FastEmbed параметры
            cache_dir = os.path.expanduser("~/.cache/fastembed")
            os.makedirs(cache_dir, exist_ok=True)
            
            self.model = TextEmbedding(
                model_name=self.embedding_config.model_name,
                cache_dir=cache_dir,
                providers=["CPUExecutionProvider"],  # Принудительно CPU
                threads=self.parallelism_config.torch_num_threads
            )
            
            self.provider_name = "fastembed"
            logger.info("FastEmbed модель успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки FastEmbed: {e}")
            raise
    
    def _initialize_sentence_transformers(self) -> None:
        """Инициализирует Sentence Transformers модель"""
        try:
            logger.info(f"Загрузка Sentence Transformers модели: {self.embedding_config.model_name}")
            
            # Параметры для Sentence Transformers
            device = "cpu"  # Принудительно CPU
            
            self.model = SentenceTransformer(
                self.embedding_config.model_name,
                device=device,
                cache_folder=os.path.expanduser("~/.cache/torch/sentence_transformers")
            )
            
            # Настройка precision для SentenceTransformer v5.1.0+
            if hasattr(self.model, 'precision') and self.embedding_config.precision == "int8":
                try:
                    # Попытка установить int8 precision (доступно в новых версиях)
                    self.model.precision = 'int8'
                    logger.info("Установлена int8 precision для Sentence Transformers")
                except Exception as e:
                    logger.warning(f"Не удалось установить int8 precision: {e}")
            
            self.provider_name = "sentence-transformers"
            logger.info("Sentence Transformers модель успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Sentence Transformers: {e}")
            raise
    
    def _try_fallback_initialization(self) -> None:
        """Попытка инициализации fallback провайдера"""
        logger.warning("Пытаемся fallback инициализацию")
        
        if self.provider_name != "sentence-transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._initialize_sentence_transformers()
                self.stats['provider_fallbacks'] += 1
                return
            except Exception as e:
                logger.error(f"Fallback на Sentence Transformers не удался: {e}")
        
        if self.provider_name != "fastembed" and FASTEMBED_AVAILABLE:
            try:
                self._initialize_fastembed()
                self.stats['provider_fallbacks'] += 1
                return
            except Exception as e:
                logger.error(f"Fallback на FastEmbed не удался: {e}")
        
        raise RuntimeError("Все провайдеры эмбеддингов недоступны")
    
    def warmup(self) -> None:
        """
        Прогрев модели одним dummy-encode для JIT компиляции.
        """
        if self._is_warmed_up:
            return
            
        logger.info("Прогрев модели...")
        start_time = time.time()
        
        try:
            # Dummy текст для прогрева
            dummy_text = ["This is a warmup text for JIT compilation and model initialization."]
            
            # Выполняем dummy encode
            if self.provider_name == "fastembed":
                embeddings = list(self.model.embed(dummy_text))
                # FastEmbed возвращает generator, берём первый элемент
                _ = embeddings[0] if embeddings else None
            else:  # sentence-transformers
                _ = self.model.encode(
                    dummy_text,
                    normalize_embeddings=self.embedding_config.normalize_embeddings,
                    batch_size=1
                )
            
            warmup_time = time.time() - start_time
            self._is_warmed_up = True
            
            logger.info(f"Прогрев модели завершён за {warmup_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Ошибка прогрева модели: {e}")
            # Не критическая ошибка, продолжаем работу
    
    def calculate_batch_size(self, queue_len: int) -> int:
        """
        Вычисляет адаптивный размер батча на основе доступной RAM и длины очереди.
        
        Args:
            queue_len: Количество текстов в очереди на обработку
            
        Returns:
            Оптимальный размер батча
        """
        try:
            # Получаем информацию о памяти
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024 ** 3)
            
            # Базовый размер батча из конфигурации
            min_batch = self.embedding_config.batch_size_min
            max_batch = self.embedding_config.batch_size_max
            
            # Простая логика адаптивного батчирования для предсказуемости тестов
            if queue_len <= 50:
                target_batch = min_batch + (queue_len // 10) * 2  # Плавное увеличение
            elif queue_len <= 500:
                target_batch = min_batch + (queue_len // 50) * 8  # Среднее увеличение
            else:
                target_batch = max_batch  # Максимальный батч для больших очередей
            
            # Корректировка по доступной памяти
            if available_memory_gb < 2.0:
                target_batch = min_batch  # Критически мало памяти
            elif available_memory_gb < 4.0:
                target_batch = min(target_batch, int(max_batch * 0.5))  # Ограничиваем при нехватке памяти
            
            # Финальный размер с учетом ограничений
            calculated_size = max(min_batch, min(max_batch, target_batch))
            
            logger.debug(
                f"Размер батча: {calculated_size} "
                f"(очередь: {queue_len}, память: {available_memory_gb:.1f}GB)"
            )
            
            return calculated_size
            
        except Exception as e:
            logger.warning(f"Ошибка вычисления размера батча: {e}")
            return self.embedding_config.batch_size_min
    
    def embed_texts(self, texts: List[str], deadline_ms: int = 1500) -> np.ndarray:
        """
        Батчевое кодирование текстов в векторы с контролем времени отклика.
        
        Args:
            texts: Список текстов для кодирования
            deadline_ms: Максимальное время на обработку в миллисекундах
            
        Returns:
            numpy массив эмбеддингов размером [len(texts), embedding_dim]
        """
        if not texts:
            return np.array([])
        
        start_time = time.time()
        deadline_seconds = deadline_ms / 1000.0
        
        try:
            # Прогрев модели если не был выполнен
            if self.embedding_config.warmup_enabled and not self._is_warmed_up:
                self.warmup()
            
            # Вычисляем оптимальный размер батча
            batch_size = self.calculate_batch_size(len(texts))
            
            # Обработка батчами
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                # Проверяем deadline
                elapsed = time.time() - start_time
                if elapsed > deadline_seconds:
                    logger.warning(f"Превышен deadline {deadline_ms}ms, обработано {i}/{len(texts)} текстов")
                    # Возвращаем частичные результаты с заполнением нулями
                    partial_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
                    remaining_count = len(texts) - i
                    if remaining_count > 0 and len(partial_embeddings) > 0:
                        # Заполняем оставшиеся эмбеддинги нулями той же размерности
                        embedding_dim = partial_embeddings.shape[1]
                        zero_embeddings = np.zeros((remaining_count, embedding_dim))
                        return np.vstack([partial_embeddings, zero_embeddings])
                    break
                
                batch_texts = texts[i:i + batch_size]
                
                try:
                    # Кодирование батча
                    batch_embeddings = self._encode_batch(batch_texts)
                    all_embeddings.append(batch_embeddings)
                    
                except (RuntimeError, MemoryError) as e:
                    if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                        # OOM - пытаемся fallback на меньший батч
                        logger.warning(f"OOM при обработке батча размером {len(batch_texts)}: {e}")
                        self.stats['oom_fallbacks'] += 1
                        
                        # Уменьшаем размер батча
                        self._current_batch_size = max(1, self._current_batch_size // 2)
                        
                        # Пытаемся обработать поэлементно
                        batch_embeddings = self._encode_fallback(batch_texts)
                        all_embeddings.append(batch_embeddings)
                    else:
                        raise
            
            # Объединяем результаты
            if all_embeddings:
                result = np.vstack(all_embeddings)
                
                # Применяем truncate_dim если необходимо
                if (self.embedding_config.truncate_dim > 0 and 
                    result.shape[1] > self.embedding_config.truncate_dim):
                    result = result[:, :self.embedding_config.truncate_dim]
                
                # Обновляем статистику
                total_time = time.time() - start_time
                self.stats['total_texts'] += len(texts)
                self.stats['total_time'] += total_time
                self.stats['batch_count'] += 1
                
                logger.debug(
                    f"Обработано {len(texts)} текстов за {total_time:.3f}s "
                    f"({len(texts)/total_time:.1f} текстов/сек)"
                )
                
                return result
            else:
                # Fallback - возвращаем пустой массив нужной размерности
                embedding_dim = self.embedding_config.truncate_dim or 384
                return np.zeros((len(texts), embedding_dim))
                
        except Exception as e:
            logger.error(f"Критическая ошибка кодирования: {e}")
            # Возвращаем нулевые эмбеддинги как последний fallback
            embedding_dim = self.embedding_config.truncate_dim or 384
            return np.zeros((len(texts), embedding_dim))
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Кодирование батча текстов с использованием текущего провайдера.
        
        Args:
            texts: Список текстов для кодирования
            
        Returns:
            numpy массив эмбеддингов
        """
        try:
            if self.provider_name == "fastembed":
                # FastEmbed возвращает generator
                embeddings_gen = self.model.embed(texts)
                embeddings = list(embeddings_gen)
                return np.array(embeddings)
                
            else:  # sentence-transformers
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=self.embedding_config.normalize_embeddings,
                    batch_size=len(texts),
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return embeddings
                
        except Exception as e:
            logger.error(f"Ошибка кодирования батча с {self.provider_name}: {e}")
            raise
    
    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
        """
        Fallback кодирование - поэлементная обработка при OOM.
        
        Args:
            texts: Список текстов для кодирования
            
        Returns:
            numpy массив эмбеддингов
        """
        logger.info(f"Fallback к поэлементной обработке для {len(texts)} текстов")
        
        embeddings = []
        for text in texts:
            try:
                # Принудительная сборка мусора между итерациями
                if len(embeddings) % 10 == 0:
                    gc.collect()
                
                embedding = self._encode_batch([text])
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Ошибка кодирования текста '{text[:50]}...': {e}")
                # Добавляем нулевой вектор как fallback
                if embeddings:
                    # Используем размерность предыдущих эмбеддингов
                    zero_embedding = np.zeros_like(embeddings[-1])
                else:
                    # Используем дефолтную размерность
                    embedding_dim = self.embedding_config.truncate_dim or 384
                    zero_embedding = np.zeros((1, embedding_dim))
                embeddings.append(zero_embedding)
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def get_stats(self) -> dict:
        """
        Возвращает статистику производительности.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.stats.copy()
        
        if stats['total_time'] > 0:
            stats['avg_texts_per_second'] = stats['total_texts'] / stats['total_time']
        else:
            stats['avg_texts_per_second'] = 0.0
            
        if stats['batch_count'] > 0:
            stats['avg_batch_time'] = stats['total_time'] / stats['batch_count']
        else:
            stats['avg_batch_time'] = 0.0
        
        stats['provider'] = getattr(self, 'provider_name', 'unknown')
        stats['model_name'] = self.embedding_config.model_name
        stats['is_warmed_up'] = self._is_warmed_up
        stats['current_batch_size'] = self._current_batch_size
        
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
        logger.info("Статистика производительности сброшена")
    
    def __del__(self):
        """Очистка ресурсов при уничтожении объекта"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Очистка модели
                del self.model
                self.model = None
                
            # Принудительная сборка мусора
            gc.collect()
            
        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов эмбеддера: {e}")
