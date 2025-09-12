#!/usr/bin/env python3
"""
Скрипт очистки старых коллекций Qdrant для миграции на Jina v3.

Этот скрипт:
- Удаляет старую коллекцию code_chunks (384d)
- Подготавливает место для новой коллекции repo_sum_v3 (1024d)
- Создаёт backup метаданных перед удалением
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Добавляем корневую директорию в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import ResponseHandlingException
except ImportError:
    print("❌ Ошибка: qdrant-client не установлен. Выполните: pip install -r requirements.txt")
    sys.exit(1)

from config import get_config
from rag.exceptions import VectorStoreException, VectorStoreConnectionError

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scripts/cleanup.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class QdrantCleanupManager:
    """Менеджер очистки коллекций Qdrant"""
    
    def __init__(self):
        """Инициализация менеджера очистки"""
        try:
            self.config = get_config(require_api_key=False)
            self.vector_config = self.config.rag.vector_store
            
            # Инициализация клиентов
            self.client = self._initialize_client()
            
            # Коллекции для очистки
            self.old_collections = ["code_chunks", "code_chunks_v1", "code_chunks_v2"]
            self.new_collection = self.vector_config.collection_name  # repo_sum_v3
            
            logger.info("QdrantCleanupManager инициализирован")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации QdrantCleanupManager: {e}")
            raise VectorStoreException(f"Не удалось инициализировать cleanup manager: {e}")
    
    def _initialize_client(self) -> QdrantClient:
        """Инициализирует Qdrant клиент"""
        try:
            client = QdrantClient(
                host=self.vector_config.host,
                port=self.vector_config.port,
                prefer_grpc=self.vector_config.prefer_grpc,
                timeout=30
            )
            logger.info(f"Подключение к Qdrant: {self.vector_config.host}:{self.vector_config.port}")
            return client
            
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            raise VectorStoreConnectionError(f"Не удалось подключиться к Qdrant: {e}")
    
    def health_check(self) -> bool:
        """Проверка подключения к Qdrant"""
        try:
            collections = self.client.get_collections()
            logger.info(f"✅ Подключение к Qdrant успешно. Доступно коллекций: {len(collections.collections)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Qdrant: {e}")
            return False
    
    def list_collections(self) -> Dict[str, Any]:
        """Список всех коллекций в Qdrant"""
        try:
            collections_response = self.client.get_collections()
            
            collections_info = {}
            for collection in collections_response.collections:
                try:
                    collection_info = self.client.get_collection(collection.name)
                    collections_info[collection.name] = {
                        'status': collection_info.status if hasattr(collection_info, 'status') else 'unknown',
                        'vectors_count': collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0,
                        'points_count': collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
                        'config': {
                            'vector_size': collection_info.config.params.vectors.size if hasattr(collection_info, 'config') else 'unknown',
                            'distance': str(collection_info.config.params.vectors.distance) if hasattr(collection_info, 'config') else 'unknown'
                        }
                    }
                except Exception as e:
                    logger.warning(f"Не удалось получить информацию о коллекции {collection.name}: {e}")
                    collections_info[collection.name] = {'error': str(e)}
            
            return collections_info
            
        except Exception as e:
            logger.error(f"Ошибка получения списка коллекций: {e}")
            return {}
    
    def backup_collection_metadata(self, collection_name: str) -> bool:
        """Создаёт backup метаданных коллекции"""
        try:
            backup_dir = Path("scripts/backups")
            backup_dir.mkdir(exist_ok=True)
            
            # Получаем информацию о коллекции
            collection_info = self.client.get_collection(collection_name)
            
            # Создаём backup файл
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{collection_name}_metadata_{timestamp}.json"
            
            backup_data = {
                'collection_name': collection_name,
                'backup_timestamp': datetime.now(timezone.utc).isoformat(),
                'status': str(collection_info.status) if hasattr(collection_info, 'status') else 'unknown',
                'vectors_count': collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0,
                'points_count': collection_info.points_count if hasattr(collection_info, 'points_count') else 0,
                'indexed_vectors_count': collection_info.indexed_vectors_count if hasattr(collection_info, 'indexed_vectors_count') else 0,
                'config': {
                    'vector_size': collection_info.config.params.vectors.size if hasattr(collection_info, 'config') else None,
                    'distance': str(collection_info.config.params.vectors.distance) if hasattr(collection_info, 'config') else None,
                } if hasattr(collection_info, 'config') else {}
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Backup метаданных коллекции {collection_name} создан: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания backup для {collection_name}: {e}")
            return False
    
    def delete_collection_safe(self, collection_name: str) -> bool:
        """Безопасное удаление коллекции с backup"""
        try:
            # Проверяем существование коллекции
            try:
                collection_info = self.client.get_collection(collection_name)
                logger.info(f"📋 Коллекция {collection_name} найдена")
                
                # Показываем информацию
                vectors_count = collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0
                points_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                logger.info(f"   📊 Векторов: {vectors_count}, Точек: {points_count}")
                
            except Exception:
                logger.info(f"ℹ️  Коллекция {collection_name} не существует, пропускаем")
                return True
            
            # Создаём backup метаданных
            if not self.backup_collection_metadata(collection_name):
                logger.warning(f"⚠️  Не удалось создать backup для {collection_name}, но продолжаем удаление")
            
            # Удаляем коллекцию
            result = self.client.delete_collection(collection_name)
            
            if result:
                logger.info(f"✅ Коллекция {collection_name} успешно удалена")
                return True
            else:
                logger.error(f"❌ Не удалось удалить коллекцию {collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка удаления коллекции {collection_name}: {e}")
            return False
    
    def cleanup_old_collections(self) -> bool:
        """Очистка всех старых коллекций"""
        success_count = 0
        
        logger.info("🧹 Начинаем очистку старых коллекций...")
        
        for collection_name in self.old_collections:
            logger.info(f"🗑️  Обрабатываем коллекцию: {collection_name}")
            
            if self.delete_collection_safe(collection_name):
                success_count += 1
                logger.info(f"✅ {collection_name} обработана успешно")
            else:
                logger.error(f"❌ Ошибка обработки {collection_name}")
        
        logger.info(f"📊 Результат очистки: {success_count}/{len(self.old_collections)} коллекций обработаны")
        
        return success_count == len(self.old_collections)
    
    def prepare_new_collection_space(self) -> bool:
        """Подготовка места для новой коллекции"""
        try:
            logger.info(f"🆕 Подготовка места для новой коллекции: {self.new_collection}")
            
            # Проверяем, не существует ли уже новая коллекция
            try:
                existing_info = self.client.get_collection(self.new_collection)
                logger.warning(f"⚠️  Коллекция {self.new_collection} уже существует!")
                
                # Получаем информацию о существующей коллекции
                vectors_count = existing_info.vectors_count if hasattr(existing_info, 'vectors_count') else 0
                vector_size = existing_info.config.params.vectors.size if hasattr(existing_info, 'config') else 'unknown'
                
                logger.info(f"   📊 Существующая коллекция: {vectors_count} векторов, размерность: {vector_size}")
                
                if vector_size == 1024:
                    logger.info(f"✅ Коллекция {self.new_collection} уже имеет правильную размерность (1024d)")
                    return True
                else:
                    logger.warning(f"⚠️  Коллекция {self.new_collection} имеет неправильную размерность: {vector_size}")
                    
                    # Спрашиваем пользователя о пересоздании
                    response = input(f"Пересоздать коллекцию {self.new_collection}? (y/N): ").strip().lower()
                    if response in ['y', 'yes', 'да']:
                        logger.info(f"🗑️  Удаляем существующую коллекцию {self.new_collection}")
                        return self.delete_collection_safe(self.new_collection)
                    else:
                        logger.info("ℹ️  Оставляем существующую коллекцию без изменений")
                        return False
                        
            except Exception:
                logger.info(f"✅ Место для коллекции {self.new_collection} свободно")
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки места для новой коллекции: {e}")
            return False
    
    def run_cleanup(self, interactive: bool = True) -> bool:
        """Запуск полной очистки"""
        logger.info("🚀 Запуск очистки коллекций Qdrant для миграции на Jina v3")
        
        try:
            # 1. Проверка подключения
            if not self.health_check():
                return False
            
            # 2. Показываем текущие коллекции
            logger.info("📋 Текущие коллекции в Qdrant:")
            collections = self.list_collections()
            for name, info in collections.items():
                if 'error' in info:
                    logger.warning(f"   ❌ {name}: {info['error']}")
                else:
                    vector_size = info.get('config', {}).get('vector_size', 'unknown')
                    points = info.get('points_count', 0)
                    logger.info(f"   📊 {name}: {points} точек, размерность: {vector_size}")
            
            # 3. Подтверждение от пользователя
            if interactive:
                print(f"\n🔄 Планируется удалить следующие коллекции: {', '.join(self.old_collections)}")
                print(f"🆕 И подготовить место для новой коллекции: {self.new_collection}")
                
                response = input("\nПродолжить очистку? (y/N): ").strip().lower()
                if response not in ['y', 'yes', 'да']:
                    logger.info("❌ Очистка отменена пользователем")
                    return False
            
            # 4. Очистка старых коллекций
            cleanup_success = self.cleanup_old_collections()
            
            # 5. Подготовка места для новой коллекции
            prepare_success = self.prepare_new_collection_space()
            
            # 6. Итоговый результат
            overall_success = cleanup_success and prepare_success
            
            if overall_success:
                logger.info("🎉 Очистка завершена успешно!")
                logger.info(f"✅ Система готова для создания коллекции {self.new_collection} с размерностью 1024d")
            else:
                logger.error("❌ Очистка завершена с ошибками")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка очистки: {e}")
            return False


def main():
    """Основная функция скрипта"""
    try:
        print("🧹 Скрипт очистки коллекций Qdrant для миграции на Jina v3")
        print("=" * 60)
        
        # Инициализация менеджера очистки
        cleanup_manager = QdrantCleanupManager()
        
        # Запуск очистки
        success = cleanup_manager.run_cleanup(interactive=True)
        
        if success:
            print("\n🎉 Очистка завершена успешно!")
            print("✅ Система готова для миграции на Jina v3 (1024d)")
            sys.exit(0)
        else:
            print("\n❌ Очистка завершена с ошибками")
            print("📋 Проверьте логи для деталей: scripts/cleanup.log")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Очистка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
