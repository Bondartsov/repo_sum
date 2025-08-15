"""
CPU-оптимизированное векторное хранилище Qdrant для RAG системы.

Основные возможности:
- CPU-профиль коллекции с квантованием
- Батчевая загрузка точек 512-1024
- CRUD операции с retry логикой
- Поиск с фильтрацией и гибридным режимом
- Health check и мониторинг производительности
"""

import logging
import uuid
import time
import asyncio
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np
from dataclasses import asdict
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CollectionInfo,
    HnswConfigDiff, OptimizersConfigDiff,
    ScalarQuantization, ProductQuantization, BinaryQuantization,
    PointStruct, SearchParams, Filter, FieldCondition, Range,
    UpdateStatus, CollectionStatus,
    MatchValue, GeoBoundingBox, GeoRadius,
    VectorsConfig, Datatype, CompressionRatio
)
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from config import VectorStoreConfig
from .exceptions import (
    VectorStoreException, 
    VectorStoreConnectionError, 
    VectorStoreConfigurationError,
    TimeoutException
)

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    CPU-оптимизированное векторное хранилище на базе Qdrant.
    
    Возможности:
    - HTTP и gRPC клиенты с автоматическим переключением
    - CPU-friendly профиль коллекции (float16, on_disk, квантование)
    - Батчевая загрузка с retry логикой
    - Поиск с фильтрацией и гибридным режимом (dense + sparse)
    - Health check и метрики производительности
    """
    
    def __init__(self, vector_config: VectorStoreConfig):
        """
        Инициализация Qdrant векторного хранилища.
        
        Args:
            vector_config: Конфигурация векторного хранилища
        """
        self.config = vector_config
        self.http_client: Optional[QdrantClient] = None
        self.grpc_client: Optional[QdrantClient] = None
        self.active_client: Optional[QdrantClient] = None
        self._connected = False
        self._collection_exists = False
        
        # Статистика производительности
        self.stats = {
            'total_points': 0,
            'total_searches': 0,
            'total_index_time': 0.0,
            'total_search_time': 0.0,
            'avg_batch_size': 0,
            'error_count': 0,
            'retry_count': 0,
            'connection_switches': 0
        }
        
        logger.info(
            f"Инициализация QdrantVectorStore: {vector_config.host}:{vector_config.port}, "
            f"коллекция: {vector_config.collection_name}"
        )
        
        # Инициализация клиентов
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Инициализирует HTTP и gRPC клиенты Qdrant"""
        try:
            # HTTP клиент (основной)
            self.http_client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                prefer_grpc=False,
                timeout=30,
                # Дополнительные параметры для стабильности
                https=False,
                api_key=None
            )
            
            # gRPC клиент (fallback)
            if self.config.prefer_grpc:
                try:
                    self.grpc_client = QdrantClient(
                        host=self.config.host,
                        port=self.config.port,
                        prefer_grpc=True,
                        timeout=30
                    )
                    self.active_client = self.grpc_client
                    logger.info("Используется gRPC клиент как основной")
                except Exception as e:
                    logger.warning(f"gRPC клиент недоступен, используется HTTP: {e}")
                    self.active_client = self.http_client
            else:
                self.active_client = self.http_client
                logger.info("Используется HTTP клиент как основной")
                
        except Exception as e:
            logger.error(f"Ошибка инициализации Qdrant клиентов: {e}")
            raise VectorStoreConnectionError(
                f"Не удалось инициализировать клиенты Qdrant: {e}"
            )
    
    def _switch_client(self) -> bool:
        """
        Переключается на резервный клиент при ошибке.
        
        Returns:
            True если переключение успешно
        """
        try:
            if self.active_client == self.http_client and self.grpc_client:
                self.active_client = self.grpc_client
                logger.info("Переключение на gRPC клиент")
            elif self.active_client == self.grpc_client and self.http_client:
                self.active_client = self.http_client
                logger.info("Переключение на HTTP клиент")
            else:
                return False
            
            self.stats['connection_switches'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Ошибка переключения клиента: {e}")
            return False
    
    def _create_collection_config(self) -> Dict[str, Any]:
        """
        Создаёт CPU-оптимизированную конфигурацию коллекции.
        
        Returns:
            Конфигурация коллекции для Qdrant
        """
        # Маппинг distance метрик
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID
        }
        
        distance = distance_map.get(self.config.distance, Distance.COSINE)
        
        # CPU-friendly параметры HNSW
        hnsw_config = HnswConfigDiff(
            m=self.config.hnsw_m,                    # 16-32 для CPU
            ef_construct=self.config.hnsw_ef_construct,  # 64-128
            full_scan_threshold=10000,                   # Полное сканирование для малых коллекций
            max_indexing_threads=4                       # Ограничение потоков индексации
        )
        
        # Настройки оптимизатора для CPU
        optimizer_config = OptimizersConfigDiff(
            deleted_threshold=0.2,      # Порог для переиндексации
            vacuum_min_vector_number=1000,  # Минимум векторов для vacuum
            default_segment_number=2,    # Количество сегментов
            max_segment_size=200000,     # Максимальный размер сегмента
            memmap_threshold=200000,     # Порог для memory mapping
            indexing_threshold=10000,    # Порог для индексации
            flush_interval_sec=30,       # Интервал записи на диск
            max_optimization_threads=2   # Ограничение потоков оптимизации
        )
        
        # Квантование для экономии памяти
        quantization_config = None
        if self.config.enable_quantization:
            try:
                if self.config.quantization_type == "SQ":  # Scalar Quantization
                    # Совместимость с различными версиями qdrant-client
                    try:
                        # Новый API (qdrant-client >= 1.15.0)
                        from qdrant_client.models import ScalarQuantizationConfig
                        quantization_config = ScalarQuantization(
                            scalar=ScalarQuantizationConfig()
                        )
                    except ImportError:
                        # Fallback для старых версий - базовая конфигурация
                        try:
                            quantization_config = ScalarQuantization(scalar={})
                        except Exception as sq_e:
                            logger.warning(f"Ошибка SQ конфигурации (старый API): {sq_e}, отключаем квантование")
                            quantization_config = None
                    except Exception as e:
                        logger.warning(f"Ошибка SQ конфигурации (новый API): {e}, отключаем квантование")
                        quantization_config = None
                elif self.config.quantization_type == "PQ":  # Product Quantization
                    quantization_config = ProductQuantization(
                        compression=CompressionRatio.X16  # 16x сжатие
                    )
                elif self.config.quantization_type == "BQ":  # Binary Quantization
                    quantization_config = BinaryQuantization()
            except Exception as e:
                logger.warning(f"Не удалось создать конфигурацию квантования {self.config.quantization_type}: {e}")
                quantization_config = None
        
        # Основные параметры векторов
        vector_params = VectorParams(
            size=self.config.vector_size,
            distance=distance,
            hnsw_config=hnsw_config,
            quantization_config=quantization_config,
            on_disk=True,                    # CPU-friendly: храним на диске
            datatype=Datatype.FLOAT16        # Экономия памяти: float16
        )
        
        return {
            "vectors_config": vector_params,
            "optimizers_config": optimizer_config,
            "replication_factor": self.config.replication_factor,
            "write_consistency_factor": self.config.write_consistency_factor,
            "shard_number": 1,  # Для начала один шард
            "on_disk_payload": True,  # Payload тоже на диск для экономии RAM
        }
    
    async def initialize_collection(self, recreate: bool = False) -> None:
        """
        Создает или пересоздает коллекцию с CPU-оптимизированными параметрами.
        
        Args:
            recreate: Пересоздать коллекцию если она уже существует
            
        Raises:
            VectorStoreException: При ошибке создания коллекции
        """
        collection_name = self.config.collection_name
        
        try:
            # Проверяем существование коллекции
            exists = await self._collection_exists_check()
            
            if exists and recreate:
                logger.info(f"Пересоздание коллекции {collection_name}")
                await self._delete_collection_safe()
                exists = False
            
            if not exists:
                logger.info(f"Создание новой коллекции {collection_name}")
                
                # Получаем конфигурацию
                config = self._create_collection_config()
                
                # Создаем коллекцию
                result = self.active_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=config["vectors_config"],
                    optimizers_config=config["optimizers_config"],
                    replication_factor=config["replication_factor"],
                    write_consistency_factor=config["write_consistency_factor"],
                    shard_number=config["shard_number"],
                    on_disk_payload=config["on_disk_payload"]
                )
                
                if not result:
                    raise VectorStoreException(
                        f"Не удалось создать коллекцию {collection_name}",
                        collection_name=collection_name,
                        operation="create_collection"
                    )
                
                logger.info(f"Коллекция {collection_name} успешно создана")
            else:
                logger.info(f"Коллекция {collection_name} уже существует")
            
            # Проверяем готовность коллекции
            await self._wait_collection_ready()
            self._collection_exists = True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации коллекции {collection_name}: {e}")
            
            # Пытаемся переключить клиент и повторить
            if self._switch_client():
                try:
                    await self.initialize_collection(recreate)
                    return
                except Exception as retry_e:
                    logger.error(f"Повторная попытка также не удалась: {retry_e}")
            
            raise VectorStoreException(
                f"Критическая ошибка инициализации коллекции: {e}",
                collection_name=collection_name,
                operation="initialize"
            )
    
    def _validate_points(self, points: List[Dict]) -> List[Dict]:
        """
        Валидирует структуру точек перед загрузкой.
        
        Args:
            points: Список точек для валидации
            
        Returns:
            Валидированный список точек
            
        Raises:
            VectorStoreConfigurationError: При некорректной структуре точек
        """
        if not points:
            return []
        
        validated_points = []
        
        for i, point in enumerate(points):
            try:
                # Проверяем обязательные поля
                if 'id' not in point:
                    point['id'] = str(uuid.uuid4())
                
                if 'vector' not in point:
                    raise VectorStoreConfigurationError(
                        f"Точка {i} не содержит поле 'vector'"
                    )
                
                # Проверяем вектор
                vector = point['vector']
                if isinstance(vector, list):
                    vector = np.array(vector, dtype=np.float32)
                elif isinstance(vector, np.ndarray):
                    vector = vector.astype(np.float32)
                else:
                    raise VectorStoreConfigurationError(
                        f"Некорректный тип вектора в точке {i}: {type(vector)}"
                    )
                
                # Проверяем размерность
                if len(vector.shape) != 1 or vector.shape[0] != self.config.vector_size:
                    raise VectorStoreConfigurationError(
                        f"Некорректная размерность вектора в точке {i}: "
                        f"ожидается {self.config.vector_size}, получено {vector.shape}"
                    )
                
                # Проверяем payload
                payload = point.get('payload', {})
                if not isinstance(payload, dict):
                    payload = {}
                
                # Добавляем timestamp если отсутствует
                if 'ts' not in payload:
                    payload['ts'] = datetime.now(timezone.utc).isoformat()
                
                validated_point = {
                    'id': str(point['id']),
                    'vector': vector.tolist(),  # Qdrant ожидает список
                    'payload': payload
                }
                
                validated_points.append(validated_point)
                
            except Exception as e:
                logger.error(f"Ошибка валидации точки {i}: {e}")
                self.stats['error_count'] += 1
                # Пропускаем некорректную точку
                continue
        
        logger.debug(f"Валидировано {len(validated_points)}/{len(points)} точек")
        return validated_points
    
    async def _batch_upsert(self, points: List[Dict], batch_size: int = 1024) -> int:
        """
        Пакетная загрузка точек с retry логикой.
        
        Args:
            points: Список точек для загрузки
            batch_size: Размер батча
            
        Returns:
            Количество успешно загруженных точек
            
        Raises:
            VectorStoreException: При критической ошибке загрузки
        """
        total_uploaded = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    # Преобразуем в формат Qdrant
                    qdrant_points = [
                        PointStruct(
                            id=point['id'],
                            vector=point['vector'],
                            payload=point['payload']
                        )
                        for point in batch
                    ]
                    
                    # Выполняем upsert
                    result = self.active_client.upsert(
                        collection_name=self.config.collection_name,
                        points=qdrant_points,
                        wait=True  # Ждем подтверждения записи
                    )
                    
                    if result and result.status == UpdateStatus.COMPLETED:
                        total_uploaded += len(batch)
                        logger.debug(f"Загружен батч {i//batch_size + 1}: {len(batch)} точек")
                        break
                    else:
                        raise VectorStoreException(
                            f"Неуспешный upsert, статус: {result.status if result else 'None'}"
                        )
                        
                except Exception as e:
                    retry_count += 1
                    self.stats['retry_count'] += 1
                    logger.warning(f"Ошибка загрузки батча {i//batch_size + 1}, попытка {retry_count}: {e}")
                    
                    if retry_count >= max_retries:
                        # Попытка переключить клиент
                        if self._switch_client():
                            retry_count = 0  # Сброс счетчика после переключения
                            continue
                        
                        logger.error(f"Критическая ошибка загрузки батча после {max_retries} попыток")
                        raise VectorStoreException(
                            f"Не удалось загрузить батч после {max_retries} попыток: {e}",
                            collection_name=self.config.collection_name,
                            operation="batch_upsert"
                        )
                    
                    # Экспоненциальная задержка
                    await asyncio.sleep(2 ** retry_count)
        
        return total_uploaded
    
    async def index_documents(self, points: List[Dict]) -> int:
        """
        Загружает документы в коллекцию батчами по 512–1024 точки.
        
        Args:
            points: Список точек для индексации
            
        Returns:
            Количество успешно проиндексированных документов
            
        Raises:
            VectorStoreException: При ошибке индексации
        """
        if not points:
            return 0
        
        start_time = time.time()
        
        try:
            # Валидация точек
            validated_points = self._validate_points(points)
            if not validated_points:
                logger.warning("Нет валидных точек для индексации")
                return 0
            
            # Определяем оптимальный размер батча
            batch_size = min(1024, max(512, len(validated_points) // 4))
            
            logger.info(f"Индексация {len(validated_points)} документов (батч: {batch_size})")
            
            # Пакетная загрузка
            uploaded_count = await self._batch_upsert(validated_points, batch_size)
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_points'] += uploaded_count
            self.stats['total_index_time'] += elapsed_time
            
            if uploaded_count > 0:
                self.stats['avg_batch_size'] = (
                    self.stats['avg_batch_size'] * 0.9 + batch_size * 0.1
                )
            
            # Безопасное деление для избежания ZeroDivisionError
            if elapsed_time > 0.001:  # Минимальный порог времени для предотвращения деления на ноль
                rate_info = f" ({uploaded_count/elapsed_time:.1f} док/с)"
            else:
                rate_info = " (мгновенно)"
            
            logger.info(
                f"Индексация завершена: {uploaded_count}/{len(points)} документов "
                f"за {elapsed_time:.3f}s{rate_info}"
            )
            
            return uploaded_count
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Критическая ошибка индексации документов: {e}")
            raise VectorStoreException(
                f"Ошибка индексации документов: {e}",
                collection_name=self.config.collection_name,
                operation="index_documents"
            )
    
    async def update_document(self, pid: str, vector: np.ndarray, payload: Dict) -> bool:
        """
        Обновляет существующий документ.
        
        Args:
            pid: ID точки для обновления
            vector: Новый вектор
            payload: Новые метаданные
            
        Returns:
            True если обновление успешно
        """
        try:
            # Валидируем вектор
            if isinstance(vector, np.ndarray):
                vector = vector.astype(np.float32)
            
            if len(vector) != self.config.vector_size:
                raise VectorStoreConfigurationError(
                    f"Некорректная размерность вектора: {len(vector)}, ожидается {self.config.vector_size}"
                )
            
            # Добавляем timestamp обновления
            payload = payload.copy()
            payload['updated_ts'] = datetime.now(timezone.utc).isoformat()
            
            # Создаем точку для обновления
            point = PointStruct(
                id=pid,
                vector=vector.tolist(),
                payload=payload
            )
            
            # Выполняем upsert
            result = self.active_client.upsert(
                collection_name=self.config.collection_name,
                points=[point],
                wait=True
            )
            
            success = result and result.status == UpdateStatus.COMPLETED
            
            if success:
                logger.debug(f"Документ {pid} успешно обновлен")
            else:
                logger.error(f"Не удалось обновить документ {pid}, статус: {result.status if result else 'None'}")
                
            return success
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Ошибка обновления документа {pid}: {e}")
            
            # Retry с переключением клиента
            if self._switch_client():
                try:
                    return await self.update_document(pid, vector, payload)
                except Exception as retry_e:
                    logger.error(f"Повторная попытка обновления не удалась: {retry_e}")
            
            return False
    
    async def delete_document(self, pid: str) -> bool:
        """
        Удаляет документ по ID.
        
        Args:
            pid: ID документа для удаления
            
        Returns:
            True если удаление успешно
        """
        try:
            result = self.active_client.delete(
                collection_name=self.config.collection_name,
                points_selector=[pid],
                wait=True
            )
            
            success = result and result.status == UpdateStatus.COMPLETED
            
            if success:
                logger.debug(f"Документ {pid} успешно удален")
            else:
                logger.error(f"Не удалось удалить документ {pid}, статус: {result.status if result else 'None'}")
                
            return success
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Ошибка удаления документа {pid}: {e}")
            
            # Retry с переключением клиента
            if self._switch_client():
                try:
                    return await self.delete_document(pid)
                except Exception as retry_e:
                    logger.error(f"Повторная попытка удаления не удалась: {retry_e}")
                    
            return False
    
    def _build_search_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """
        Создает фильтр Qdrant из словаря условий.
        
        Args:
            filters: Условия фильтрации
            
        Returns:
            Фильтр Qdrant или None
        """
        if not filters:
            return None
        
        conditions = []
        
        for field, value in filters.items():
            try:
                if isinstance(value, (str, int, float, bool)):
                    # Простое равенство
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
                elif isinstance(value, list):
                    # Список значений (OR)
                    for v in value:
                        conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=v))
                        )
                elif isinstance(value, dict):
                    # Сложные условия
                    if 'range' in value:
                        range_val = value['range']
                        conditions.append(
                            FieldCondition(
                                key=field, 
                                range=Range(
                                    gte=range_val.get('gte'),
                                    gt=range_val.get('gt'),
                                    lte=range_val.get('lte'),
                                    lt=range_val.get('lt')
                                )
                            )
                        )
                    elif 'geo_bounding_box' in value:
                        geo_box = value['geo_bounding_box']
                        conditions.append(
                            FieldCondition(
                                key=field,
                                geo_bounding_box=GeoBoundingBox(
                                    top_left=geo_box['top_left'],
                                    bottom_right=geo_box['bottom_right']
                                )
                            )
                        )
                    elif 'geo_radius' in value:
                        geo_rad = value['geo_radius'] 
                        conditions.append(
                            FieldCondition(
                                key=field,
                                geo_radius=GeoRadius(
                                    center=geo_rad['center'],
                                    radius=geo_rad['radius']
                                )
                            )
                        )
                        
            except Exception as e:
                logger.warning(f"Ошибка создания условия фильтра для поля {field}: {e}")
                continue
        
        return Filter(must=conditions) if conditions else None
    
    async def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int,
        filters: Optional[Dict] = None, 
        use_hybrid: bool = False
    ) -> List[Dict]:
        """
        Поиск с опциональным гибридным режимом (dense + sparse).
        
        Args:
            query_vector: Вектор запроса
            top_k: Количество результатов
            filters: Фильтры по метаданным
            use_hybrid: Использовать гибридный поиск
            
        Returns:
            Список результатов поиска
        """
        start_time = time.time()
        
        try:
            # Валидация вектора запроса
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.astype(np.float32)
            
            if len(query_vector) != self.config.vector_size:
                raise VectorStoreConfigurationError(
                    f"Некорректная размерность вектора запроса: {len(query_vector)}, "
                    f"ожидается {self.config.vector_size}"
                )
            
            # Создаем фильтр
            search_filter = self._build_search_filter(filters)
            
            # Параметры поиска для CPU-оптимизации
            search_params = SearchParams(
                hnsw_ef=self.config.search_hnsw_ef,
                exact=False,  # Используем индекс для скорости
                quantization=None,  # Автоматический выбор
                indexed_only=False  # Включаем неиндексированные точки
            )
            
            if use_hybrid:
                # Гибридный поиск (пока только dense, sparse добавим позже)
                logger.debug("Выполнение гибридного поиска (dense)")
                results = await self._search_dense(
                    query_vector, top_k, search_filter, search_params
                )
            else:
                # Обычный dense поиск
                results = await self._search_dense(
                    query_vector, top_k, search_filter, search_params
                )
            
            # Обновляем статистику
            elapsed_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += elapsed_time
            
            logger.debug(f"Поиск завершен: {len(results)} результатов за {elapsed_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Ошибка поиска: {e}")
            
            # Retry с переключением клиента
            if self._switch_client():
                try:
                    return await self.search(query_vector, top_k, filters, use_hybrid)
                except Exception as retry_e:
                    logger.error(f"Повторная попытка поиска не удалась: {retry_e}")
            
            # Возвращаем пустой результат при критической ошибке
            return []
    
    async def _search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int,
        search_filter: Optional[Filter],
        search_params: SearchParams
    ) -> List[Dict]:
        """
        Выполняет dense векторный поиск.
        
        Args:
            query_vector: Вектор запроса
            top_k: Количество результатов
            search_filter: Фильтр поиска
            search_params: Параметры поиска
            
        Returns:
            Список результатов
        """
        try:
            results = self.active_client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=search_filter,
                search_params=search_params,
                limit=top_k,
                with_payload=True,
                with_vectors=False  # Не возвращаем векторы для экономии трафика
            )
            
            # Преобразуем результаты в нужный формат
            formatted_results = []
            for result in results:
                formatted_result = {
                    'id': result.id,
                    'score': float(result.score),
                    'payload': result.payload or {}
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка dense поиска: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверяет состояние векторного хранилища и коллекции.
        
        Returns:
            Информация о состоянии системы
        """
        health_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'unknown',
            'client_type': 'http' if self.active_client == self.http_client else 'grpc',
            'collection_status': 'unknown',
            'error': None
        }
        
        try:
            # Проверка подключения к Qdrant
            cluster_info = self.active_client.get_cluster_info()
            health_info['status'] = 'connected'
            health_info['cluster_info'] = {
                'peer_id': getattr(cluster_info, 'peer_id', 'unknown'),
                'peers_count': len(getattr(cluster_info, 'peers', [])),
                'raft_info': getattr(cluster_info, 'raft_info', {})
            }
            
            # Проверка коллекции
            collection_info = self.active_client.get_collection(self.config.collection_name)
            if collection_info:
                health_info['collection_status'] = 'exists'
                health_info['collection_info'] = {
                    'vectors_count': getattr(collection_info, 'vectors_count', 0),
                    'indexed_vectors_count': getattr(collection_info, 'indexed_vectors_count', 0),
                    'points_count': getattr(collection_info, 'points_count', 0),
                    'status': getattr(collection_info, 'status', 'unknown')
                }
            else:
                health_info['collection_status'] = 'not_found'
            
            self._connected = True
            
        except Exception as e:
            health_info['status'] = 'error'
            health_info['error'] = str(e)
            self._connected = False
            
            logger.error(f"Health check не пройден: {e}")
            
            # Пытаемся переключить клиент и повторить
            if self._switch_client():
                try:
                    return await self.health_check()
                except Exception as retry_e:
                    logger.error(f"Health check после переключения клиента также не удался: {retry_e}")
        
        return health_info
    
    async def _collection_exists_check(self) -> bool:
        """Проверяет существование коллекции"""
        try:
            collection_info = self.active_client.get_collection(self.config.collection_name)
            return collection_info is not None
        except Exception:
            return False
    
    async def _delete_collection_safe(self) -> bool:
        """Безопасное удаление коллекции"""
        try:
            result = self.active_client.delete_collection(self.config.collection_name)
            return result is not None
        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")
            return False
    
    async def _wait_collection_ready(self, timeout: int = 60) -> None:
        """Ожидает готовности коллекции к работе"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                collection_info = self.active_client.get_collection(self.config.collection_name)
                if collection_info and getattr(collection_info, 'status', None) == CollectionStatus.GREEN:
                    logger.debug(f"Коллекция {self.config.collection_name} готова к работе")
                    return
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Ошибка проверки готовности коллекции: {e}")
                await asyncio.sleep(2)
        
        raise TimeoutException(
            operation="wait_collection_ready",
            timeout_seconds=timeout,
            elapsed_seconds=time.time() - start_time
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику производительности.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.stats.copy()
        
        # Вычисляем средние показатели
        if stats['total_searches'] > 0:
            stats['avg_search_time'] = stats['total_search_time'] / stats['total_searches']
        else:
            stats['avg_search_time'] = 0.0
            
        if stats['total_points'] > 0 and stats['total_index_time'] > 0:
            stats['avg_index_rate'] = stats['total_points'] / stats['total_index_time']
        else:
            stats['avg_index_rate'] = 0.0
        
        # Дополнительная информация
        stats.update({
            'connected': self._connected,
            'collection_exists': self._collection_exists,
            'active_client': 'http' if self.active_client == self.http_client else 'grpc',
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'collection_name': self.config.collection_name,
                'vector_size': self.config.vector_size,
                'quantization_enabled': self.config.enable_quantization,
                'quantization_type': self.config.quantization_type
            }
        })
        
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику производительности"""
        self.stats = {
            'total_points': 0,
            'total_searches': 0,
            'total_index_time': 0.0,
            'total_search_time': 0.0,
            'avg_batch_size': 0,
            'error_count': 0,
            'retry_count': 0,
            'connection_switches': 0
        }
        logger.info("Статистика QdrantVectorStore сброшена")
    
    async def close(self) -> None:
        """Закрывает соединения с Qdrant"""
        try:
            if self.http_client:
                self.http_client.close()
                self.http_client = None
                
            if self.grpc_client:
                self.grpc_client.close() 
                self.grpc_client = None
                
            self.active_client = None
            self._connected = False
            
            logger.info("Соединения с Qdrant закрыты")
            
        except Exception as e:
            logger.error(f"Ошибка закрытия соединений: {e}")
    
    def __del__(self):
        """Очистка ресурсов при уничтожении объекта"""
        try:
            if hasattr(self, 'active_client') and self.active_client:
                # Синхронное закрытие в деструкторе
                if self.http_client:
                    self.http_client.close()
                if self.grpc_client:
                    self.grpc_client.close()
        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов QdrantVectorStore: {e}")
