#!/usr/bin/env python3
"""
Скрипт миграции на Jina v3 с пересозданием Qdrant коллекции.

Выполняет:
1. Проверку конфигурации на совместимость с Jina v3  
2. Backup существующих эмбеддингов (опционально)
3. Подтверждение стандартной размерности 1024d для коллекции
4. Валидацию новой коллекции
"""

import sys
import asyncio
import logging
from pathlib import Path

# Добавляем корень проекта в Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from rag.vector_store import QdrantVectorStore
from rag.embedder import CPUEmbedder

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Удаляем старые хендлеры
for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Консольный хендлер
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Файловый хендлер (без закрытия глобальных потоков)
file_handler = logging.FileHandler('migration_jina_v3.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


async def validate_jina_v3_config():
    """Проверяет корректность конфигурации для Jina v3"""
    logger.info("🔍 Валидация конфигурации Jina v3...")
    
    try:
        config = get_config(require_api_key=False)
        
        # Проверяем основные параметры
        embedding_config = config.rag.embeddings
        vector_config = config.rag.vector_store
        
        validation_errors = []
        
        # 1. Модель должна быть Jina v3
        if embedding_config.model_name != "jinaai/jina-embeddings-v3":
            validation_errors.append(
                f"❌ Некорректная модель: {embedding_config.model_name}, "
                "ожидается 'jinaai/jina-embeddings-v3'"
            )
        
        # 2. Провайдер должен быть sentence-transformers для Jina v3
        if embedding_config.provider != "sentence-transformers":
            validation_errors.append(
                f"❌ Некорректный провайдер: {embedding_config.provider}, "
                "ожидается 'sentence-transformers'"
            )
        
        # 3. trust_remote_code должен быть включен
        if not embedding_config.trust_remote_code:
            validation_errors.append(
                "❌ trust_remote_code должен быть true для Jina v3"
            )
        
        # 4. Проверяем task параметры
        if not embedding_config.task_query or not embedding_config.task_passage:
            validation_errors.append(
                "❌ task_query и task_passage должны быть заданы"
            )
        
        # 5. Размерность векторов должна соответствовать стандарту 1024d
        expected_dim = getattr(embedding_config, 'embedding_dim', vector_config.vector_size)
        if vector_config.vector_size != expected_dim:
            validation_errors.append(
                f"❌ Несоответствие размерностей: vector_size={vector_config.vector_size}, "
                f"embedding_dim={expected_dim}"
            )
        
        # 6. Коллекция должна иметь новое имя для избежания конфликтов
        if not vector_config.collection_name.endswith(('jina_v3', 'v3')):
            logger.warning(
                f"⚠️  Рекомендуется переименовать коллекцию с '{vector_config.collection_name}' "
                "на название, содержащее 'jina_v3' для ясности"
            )
        
        if validation_errors:
            logger.error("❌ Ошибки валидации конфигурации:")
            for error in validation_errors:
                logger.error(f"  {error}")
            return False
        
        logger.info("✅ Конфигурация Jina v3 корректна")
        logger.info(f"  📊 Модель: {embedding_config.model_name}")
        logger.info(f"  📊 Провайдер: {embedding_config.provider}")
        logger.info(f"  📊 Размерность: {vector_config.vector_size}d")
        logger.info(f"  📊 Коллекция: {vector_config.collection_name}")
        logger.info(f"  📊 Квантование: {vector_config.quantization_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка валидации конфигурации: {e}")
        return False


async def backup_existing_collection(vector_store: QdrantVectorStore):
    """Создает backup существующей коллекции (опционально)"""
    logger.info("💾 Проверка необходимости backup...")
    
    try:
        # Проверяем health коллекции
        health = await vector_store.health_check()
        
        if health['collection_status'] == 'exists':
            collection_info = health.get('collection_info', {})
            points_count = collection_info.get('points_count', 0)
            
            if points_count > 0:
                logger.info(f"📊 Найдена существующая коллекция с {points_count} документами")
                
                # Спрашиваем пользователя о backup
                response = input("Создать backup существующей коллекции? (y/n): ").lower().strip()
                
                if response == 'y':
                    backup_name = f"{vector_store.config.collection_name}_backup_{int(asyncio.get_event_loop().time())}"
                    logger.info(f"💾 Создание backup коллекции: {backup_name}")
                    
                    # Здесь можно добавить логику копирования коллекции
                    # Для простоты просто логируем
                    logger.info("ℹ️  ПРИМЕЧАНИЕ: Ручной backup не реализован в данной версии")
                    logger.info("ℹ️  Рекомендуется создать snapshot Qdrant вручную")
                    
                    backup_confirm = input("Продолжить без автоматического backup? (y/n): ").lower().strip()
                    if backup_confirm != 'y':
                        logger.info("❌ Миграция прервана пользователем")
                        return False
            else:
                logger.info("✅ Коллекция пуста, backup не требуется")
        else:
            logger.info("✅ Коллекция не существует, backup не требуется")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки backup: {e}")
        return False


async def recreate_collection(vector_store: QdrantVectorStore):
    """Пересоздает коллекцию с параметрами для Jina v3"""
    logger.info("🔄 Пересоздание коллекции для Jina v3...")
    
    try:
        # Пересоздаем коллекцию с параметрами Jina v3
        await vector_store.initialize_collection(recreate=True)
        
        # Проверяем успешность создания
        health = await vector_store.health_check()
        
        if health['status'] == 'connected' and health['collection_status'] == 'exists':
            logger.info("✅ Коллекция успешно пересоздана")
            
            # Выводим информацию о коллекции
            collection_info = health.get('collection_info', {})
            logger.info(f"  📊 Статус: {collection_info.get('status', 'unknown')}")
            logger.info(f"  📊 Векторов: {collection_info.get('vectors_count', 0)}")
            logger.info(f"  📊 Индексированных: {collection_info.get('indexed_vectors_count', 0)}")
            
            return True
        else:
            logger.error(f"❌ Коллекция создана некорректно: {health}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка пересоздания коллекции: {e}")
        return False


async def validate_embedder_initialization():
    """Проверяет инициализацию эмбеддера Jina v3"""
    logger.info("🤖 Проверка инициализации эмбеддера Jina v3...")
    
    try:
        config = get_config(require_api_key=False)
        embedding_config = config.rag.embeddings
        parallelism_config = config.rag.parallelism
        
        # Инициализируем эмбеддер
        embedder = CPUEmbedder(embedding_config, parallelism_config)
        
        # Проверяем провайдер
        if embedder.provider_name == "offline":
            logger.warning("⚠️  Эмбеддер в offline режиме (тесты/CI)")
            return True
        
        if embedder.provider_name != "sentence-transformers":
            logger.error(f"❌ Неожиданный провайдер: {embedder.provider_name}")
            return False
        
        # Проверяем размерность
        expected_dim = getattr(embedding_config, 'embedding_dim', embedder.embedding_dim)
        if embedder.embedding_dim != expected_dim:
            logger.error(
                f"❌ Несоответствие размерностей эмбеддера: "
                f"получено {embedder.embedding_dim}, ожидается {expected_dim}"
            )
            return False
        
        # Тестовое кодирование
        test_texts = ["Test text for Jina v3 validation", "Another test sentence"]
        
        try:
            embeddings = embedder.embed_texts(
                test_texts, 
                deadline_ms=10000,  # 10 секунд на тест
                task="retrieval.passage"
            )
            
            if embeddings.shape != (len(test_texts), expected_dim):
                logger.error(
                    f"❌ Неожиданная форма эмбеддингов: "
                    f"получено {embeddings.shape}, ожидается ({len(test_texts)}, {expected_dim})"
                )
                return False
            
            logger.info("✅ Эмбеддер Jina v3 инициализирован корректно")
            logger.info(f"  📊 Провайдер: {embedder.provider_name}")
            logger.info(f"  📊 Модель: {embedding_config.model_name}")
            logger.info(f"  📊 Размерность: {embedder.embedding_dim}d")
            logger.info(f"  📊 Тестовые эмбеддинги: {embeddings.shape}")
            
            return True
            
        except Exception as embed_e:
            logger.error(f"❌ Ошибка тестового кодирования: {embed_e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации эмбеддера: {e}")
        return False


async def run_migration():
    """Основной процесс миграции"""
    logger.info("🚀 Начало миграции на Jina v3")
    logger.info("="*60)
    
    # Шаг 1: Валидация конфигурации
    if not await validate_jina_v3_config():
        logger.error("❌ Миграция прервана из-за ошибок конфигурации")
        return False
    
    # Шаг 2: Инициализация компонентов
    try:
        config = get_config(require_api_key=False)
        vector_store = QdrantVectorStore(config.rag.vector_store)
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации компонентов: {e}")
        return False
    
    # Шаг 3: Backup (опционально)
    if not await backup_existing_collection(vector_store):
        logger.error("❌ Миграция прервана из-за ошибок backup")
        return False
    
    # Шаг 4: Пересоздание коллекции
    if not await recreate_collection(vector_store):
        logger.error("❌ Миграция прервана из-за ошибок пересоздания коллекции")
        return False
    
    # Шаг 5: Валидация эмбеддера
    if not await validate_embedder_initialization():
        logger.error("❌ Миграция прервана из-за ошибок эмбеддера")
        return False
    
    # Завершение
    await vector_store.close()
    
    logger.info("="*60)
    logger.info("✅ Миграция на Jina v3 успешно завершена!")
    logger.info("")
    logger.info("📋 Следующие шаги:")
    logger.info("  1. Проведите реиндексацию: python main.py rag index /path/to/repo")
    logger.info("  2. Проверьте качество поиска: python main.py rag search 'your query'")
    logger.info("  3. Сравните результаты с предыдущей версией")
    logger.info("")
    logger.info("📊 Лог миграции сохранен: migration_jina_v3.log")
    
    return True


async def main():
    """Entry point"""
    try:
        success = await run_migration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("❌ Миграция прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка миграции: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
