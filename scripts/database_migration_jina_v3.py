#!/usr/bin/env python3
"""
Database Migration Script для Jina v3
Полная миграция коллекций Qdrant с 384d на 1024d векторы
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Добавляем корневую директорию в Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        VectorParams, Distance, OptimizersConfig, HnswConfig,
        ScalarQuantization, ScalarQuantizationConfig, QuantizationType,
        PayloadSchemaType, CreateCollection, CollectionInfo
    )
    from config import get_config
    from rag.vector_store import QdrantVectorStore
    from rag.embedder import CPUEmbedder
    from file_scanner import FileScanner
    from code_chunker import CodeChunker
    from utils import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Make sure you're running from the project root directory")
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class MigrationConfig:
    """Конфигурация миграции"""
    old_collection: str = "code_chunks"
    new_collection: str = "repo_sum_v3"
    old_dimension: int = 384
    new_dimension: int = 1024
    backup_enabled: bool = True
    progress_reporting: bool = True
    batch_size: int = 32
    retry_attempts: int = 3

class JinaV3DatabaseMigration:
    """Класс для миграции database на Jina v3"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.client = None
        self.vector_store = None
        self.embedder = None
        
        # Инициализация компонентов
        self._initialize_components()
    
    def _initialize_components(self):
        """Инициализация Qdrant клиента и компонентов"""
        try:
            app_config = get_config()
            
            # Qdrant клиент
            self.client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
            )
            
            # Vector Store
            self.vector_store = QdrantVectorStore(app_config.rag.vector_store)
            
            # Embedder (Jina v3)
            self.embedder = CPUEmbedder(app_config.rag.embeddings)
            
            logger.info("✅ Database migration components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def check_collections_status(self) -> Dict[str, bool]:
        """Проверить статус существующих коллекций"""
        status = {}
        
        try:
            collections = self.client.get_collections().collections
            existing_names = [col.name for col in collections]
            
            status["old_collection_exists"] = self.config.old_collection in existing_names
            status["new_collection_exists"] = self.config.new_collection in existing_names
            
            logger.info(f"📊 Collection status: old={status['old_collection_exists']}, new={status['new_collection_exists']}")
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to check collections: {e}")
            raise
    
    def backup_old_collection(self) -> Optional[str]:
        """Создать backup старой коллекции"""
        if not self.config.backup_enabled:
            logger.info("⚠️ Backup disabled, skipping...")
            return None
        
        backup_name = f"{self.config.old_collection}_backup_{int(time.time())}"
        
        try:
            # Создаем snapshot коллекции
            response = self.client.create_snapshot(collection_name=self.config.old_collection)
            logger.info(f"✅ Created backup snapshot: {response}")
            return backup_name
            
        except Exception as e:
            logger.warning(f"⚠️ Backup failed (non-critical): {e}")
            return None
    
    def delete_old_collection(self) -> bool:
        """Удалить старую коллекцию"""
        try:
            status = self.check_collections_status()
            
            if not status["old_collection_exists"]:
                logger.info(f"✅ Old collection '{self.config.old_collection}' doesn't exist, nothing to delete")
                return True
            
            # Подтверждение удаления
            if not self._confirm_deletion():
                logger.info("❌ Deletion cancelled by user")
                return False
            
            # Удаление коллекции
            self.client.delete_collection(self.config.old_collection)
            logger.info(f"✅ Deleted old collection: {self.config.old_collection}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete old collection: {e}")
            return False
    
    def create_new_collection(self) -> bool:
        """Создать новую коллекцию с 1024d векторами"""
        try:
            # Оптимизированная конфигурация для 1024d
            vector_config = VectorParams(
                size=self.config.new_dimension,
                distance=Distance.COSINE,
                hnsw_config=HnswConfig(
                    m=16,  # Оптимально для 1024d
                    ef_construct=200,  # Увеличено для качества
                    full_scan_threshold=10000,  # CPU-friendly
                    max_indexing_threads=0  # Auto
                ),
                on_disk=True  # Для больших индексов
            )
            
            # Квантование для экономии памяти
            optimizers_config = OptimizersConfig(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=0,  # Auto
                max_segment_size=None,  # Auto
                memmap_threshold=None,  # Auto
                indexing_threshold=20000,
                flush_interval_sec=5,
                max_optimization_threads=1  # CPU-conservative
            )
            
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=QuantizationType.INT8,
                    quantile=0.99,
                    always_ram=False  # Диск для больших объемов
                )
            )
            
            # Создание коллекции
            self.client.create_collection(
                collection_name=self.config.new_collection,
                vectors_config=vector_config,
                optimizers_config=optimizers_config,
                quantization_config=quantization_config
            )
            
            logger.info(f"✅ Created new collection: {self.config.new_collection} (1024d)")
            
            # Проверка создания
            collection_info = self.client.get_collection(self.config.new_collection)
            logger.info(f"📊 Collection info: {collection_info.config}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create new collection: {e}")
            return False
    
    def validate_collection_structure(self) -> bool:
        """Валидация структуры новой коллекции"""
        try:
            collection_info = self.client.get_collection(self.config.new_collection)
            
            # Проверяем размерность
            vector_size = collection_info.config.params.vectors.size
            if vector_size != self.config.new_dimension:
                logger.error(f"❌ Wrong vector dimension: {vector_size}, expected: {self.config.new_dimension}")
                return False
            
            # Проверяем distance metric
            distance = collection_info.config.params.vectors.distance
            if distance != Distance.COSINE:
                logger.error(f"❌ Wrong distance metric: {distance}, expected: COSINE")
                return False
            
            # Проверяем HNSW параметры
            hnsw = collection_info.config.params.vectors.hnsw_config
            if hnsw.m != 16:
                logger.warning(f"⚠️ HNSW m parameter: {hnsw.m}, recommended: 16")
            
            logger.info("✅ Collection structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Collection validation failed: {e}")
            return False
    
    def perform_full_reindexing(self, repo_path: str) -> bool:
        """Полная переиндексация репозитория с Jina v3"""
        try:
            logger.info(f"🔄 Starting full reindexing: {repo_path}")
            
            # Сканирование файлов
            scanner = FileScanner()
            files = scanner.scan_files(repo_path)
            logger.info(f"📁 Found {len(files)} files to index")
            
            if not files:
                logger.warning("⚠️ No files found to index")
                return True
            
            # Chunking кода
            chunker = CodeChunker()
            all_chunks = []
            
            for file_info in files:
                try:
                    chunks = chunker.chunk_file(file_info)
                    all_chunks.extend(chunks)
                    
                    if self.config.progress_reporting and len(all_chunks) % 50 == 0:
                        logger.info(f"🔄 Processed {len(all_chunks)} chunks...")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to chunk file {file_info.path}: {e}")
                    continue
            
            logger.info(f"📊 Total chunks to index: {len(all_chunks)}")
            
            # Батчевая индексация с прогрессом
            success_count = 0
            failed_count = 0
            
            for i in range(0, len(all_chunks), self.config.batch_size):
                batch = all_chunks[i:i + self.config.batch_size]
                
                try:
                    # Индексация батча с retry логикой
                    self._index_batch_with_retry(batch)
                    success_count += len(batch)
                    
                    if self.config.progress_reporting:
                        progress = (i + len(batch)) / len(all_chunks) * 100
                        logger.info(f"🔄 Progress: {progress:.1f}% ({success_count}/{len(all_chunks)})")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to index batch {i//self.config.batch_size}: {e}")
                    failed_count += len(batch)
                    continue
            
            logger.info(f"✅ Reindexing completed: {success_count} success, {failed_count} failed")
            return failed_count == 0
            
        except Exception as e:
            logger.error(f"❌ Full reindexing failed: {e}")
            return False
    
    def _index_batch_with_retry(self, chunks: List) -> bool:
        """Индексация батча с retry логикой"""
        for attempt in range(self.config.retry_attempts):
            try:
                # Извлечение текста и создание embeddings
                texts = [chunk.content for chunk in chunks]
                
                # Использование task="retrieval.passage" для индексации
                embeddings = self.embedder.embed_texts(texts, task="retrieval.passage")
                
                # Подготовка точек для Qdrant
                points = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    point = {
                        "id": len(points) + int(time.time() * 1000000) + i,  # Уникальный ID
                        "vector": embedding.tolist(),
                        "payload": {
                            "content": chunk.content,
                            "file_path": chunk.file_path,
                            "language": chunk.language,
                            "chunk_type": chunk.chunk_type,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line
                        }
                    }
                    points.append(point)
                
                # Загрузка в Qdrant
                self.client.upsert(
                    collection_name=self.config.new_collection,
                    points=points
                )
                
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1}/{self.config.retry_attempts} failed: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def _confirm_deletion(self) -> bool:
        """Подтверждение удаления коллекции"""
        try:
            response = input(f"\n⚠️ This will DELETE the collection '{self.config.old_collection}'. Continue? (y/N): ")
            return response.lower() in ['y', 'yes']
        except KeyboardInterrupt:
            return False
    
    def get_migration_statistics(self) -> Dict:
        """Получить статистику миграции"""
        try:
            stats = {}
            
            # Статистика новой коллекции
            if self.config.new_collection:
                try:
                    collection_info = self.client.get_collection(self.config.new_collection)
                    stats["new_collection"] = {
                        "name": self.config.new_collection,
                        "vectors_count": collection_info.vectors_count,
                        "indexed_vectors_count": collection_info.indexed_vectors_count,
                        "points_count": collection_info.points_count,
                        "dimension": self.config.new_dimension
                    }
                except:
                    stats["new_collection"] = {"error": "Collection not found"}
            
            # Общая статистика
            stats["migration"] = {
                "from_model": "BAAI/bge-small-en-v1.5",
                "to_model": "jinaai/jina-embeddings-v3",
                "from_dimension": self.config.old_dimension,
                "to_dimension": self.config.new_dimension,
                "timestamp": time.time()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get migration statistics: {e}")
            return {"error": str(e)}

def main():
    """Main function для database migration"""
    parser = argparse.ArgumentParser(description="Jina v3 Database Migration")
    parser.add_argument("--action", choices=["migrate", "rollback", "status", "reindex"], 
                       required=True, help="Migration action")
    parser.add_argument("--repo-path", type=str, help="Repository path for reindexing")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for indexing")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--force", action="store_true", help="Force operation without confirmation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Migration config
    migration_config = MigrationConfig(
        backup_enabled=not args.no_backup,
        batch_size=args.batch_size
    )
    
    migration = JinaV3DatabaseMigration(migration_config)
    
    try:
        if args.action == "status":
            print("📊 Migration Status:")
            status = migration.check_collections_status()
            stats = migration.get_migration_statistics()
            print(json.dumps({"status": status, "stats": stats}, indent=2))
            
        elif args.action == "migrate":
            print("🚀 Starting Jina v3 Migration...")
            
            # Backup (если включен)
            migration.backup_old_collection()
            
            # Удаление старой коллекции
            if not migration.delete_old_collection():
                print("❌ Migration failed at deletion step")
                return 1
            
            # Создание новой коллекции
            if not migration.create_new_collection():
                print("❌ Migration failed at creation step")
                return 1
            
            # Валидация
            if not migration.validate_collection_structure():
                print("❌ Migration failed at validation step")
                return 1
            
            print("✅ Database migration completed successfully!")
            
            # Переиндексация (если указан путь)
            if args.repo_path:
                print(f"🔄 Starting reindexing: {args.repo_path}")
                if migration.perform_full_reindexing(args.repo_path):
                    print("✅ Reindexing completed successfully!")
                else:
                    print("⚠️ Reindexing completed with errors")
            
        elif args.action == "reindex":
            if not args.repo_path:
                print("❌ --repo-path required for reindexing")
                return 1
            
            print(f"🔄 Starting reindexing: {args.repo_path}")
            if migration.perform_full_reindexing(args.repo_path):
                print("✅ Reindexing completed successfully!")
            else:
                print("⚠️ Reindexing completed with errors")
                
        elif args.action == "rollback":
            print("🔄 Rollback not implemented in this script")
            print("🔧 Use: bash backups/migration_backup_*/rollback_migration.sh")
            
    except KeyboardInterrupt:
        print("\n❌ Migration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
