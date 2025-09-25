#!/usr/bin/env python3
"""
Базовый тест импортов и инициализации QdrantVectorStore.
"""

import sys
import traceback
import numpy as np
from dataclasses import asdict

def test_basic_imports():
    """Тестирует основные импорты"""
    print("🔄 Тестирование импортов...")
    
    # Тестирование импорта config
    print("  ➤ Импорт config...")
    from config import VectorStoreConfig
    print("    ✅ config.VectorStoreConfig")
    
    # Тестирование импорта exceptions
    print("  ➤ Импорт rag.exceptions...")
    from rag.exceptions import VectorStoreException, VectorStoreConnectionError
    print("    ✅ rag.exceptions")
    
    # Тестирование импорта основного модуля
    print("  ➤ Импорт rag.vector_store...")
    from rag.vector_store import QdrantVectorStore
    print("    ✅ rag.vector_store.QdrantVectorStore")
    
    # Тестирование Qdrant клиента (проверка доступности)
    print("  ➤ Импорт qdrant-client...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    print("    ✅ qdrant_client")
    
    # Все импорты успешны - используем assert вместо return
    assert VectorStoreConfig is not None
    assert QdrantVectorStore is not None
    assert QdrantClient is not None

def test_config_creation():
    """Тестирует создание конфигурации VectorStore"""
    from config import VectorStoreConfig
    
    config = VectorStoreConfig(
        host="localhost",
        port=6333,
        prefer_grpc=True,
        collection_name="test_collection",
        vector_size=1024,
        distance="cosine"
    )
    
    # Проверяем базовые параметры
    assert config.host == "localhost"
    assert config.port == 6333
    assert config.prefer_grpc == True
    assert config.collection_name == "test_collection"
    assert config.vector_size == 1024
    assert config.distance == "cosine"
    
    # Проверяем значения по умолчанию
    assert config.hnsw_m >= 16
    assert config.hnsw_ef_construct >= 64
    assert config.quantization_type in ["SQ", "PQ", "BQ"]
    
    # Используем assert вместо return
    assert config is not None

def test_qdrant_initialization():
    """Тестирует инициализацию QdrantVectorStore (без подключения)"""
    print("\n🔄 Тестирование инициализации QdrantVectorStore...")
    
    try:
        from config import VectorStoreConfig
        from rag.vector_store import QdrantVectorStore
        
        # Создаем тестовую конфигурацию
        config = VectorStoreConfig(
            host="localhost",  # Не подключаемся реально
            port=6333,
            collection_name="test_collection",
            vector_size=1024
        )
        
        # Инициализация (должна пройти без реального подключения)
        vector_store = QdrantVectorStore(config)
        
        print("    ✅ QdrantVectorStore инициализирован")
        print(f"    ➤ Хост: {vector_store.config.host}")
        print(f"    ➤ Порт: {vector_store.config.port}")
        print(f"    ➤ Коллекция: {vector_store.config.collection_name}")
        
        # Проверяем методы
        stats = vector_store.get_stats()
        print(f"    ➤ Статистика: {len(stats)} полей")
        
        # Используем assert вместо return
        assert vector_store is not None
        assert stats is not None
        assert len(stats) > 0
        
    except Exception as e:
        print(f"    ❌ Ошибка инициализации: {e}")
        traceback.print_exc()
        # Падаем в тесте при ошибке
        assert False, f"Инициализация не удалась: {e}"

def test_collection_config_generation():
    """Тестирует генерацию конфигурации коллекции"""
    print("\n🔄 Тестирование генерации конфигурации коллекции...")
    
    from config import VectorStoreConfig
    from rag.vector_store import QdrantVectorStore
    
    # Различные конфигурации для тестирования
    test_configs = [
        {
            "name": "SQ квантование",
            "quantization_type": "SQ",
            "vector_size": 1024,
            "hnsw_m": 16
        },
        {
            "name": "PQ квантование", 
            "quantization_type": "PQ",
            "vector_size": 256,
            "hnsw_m": 32
        },
        {
            "name": "Без квантования",
            "quantization_type": "SQ",
            "enable_quantization": False,
            "vector_size": 1024,
            "hnsw_m": 24
        }
    ]
    
    for test_config in test_configs:
        print(f"\n  ➤ Тест: {test_config['name']}")
        
        config = VectorStoreConfig(
            host="localhost",
            port=6333,
            collection_name="test",
            vector_size=test_config["vector_size"],
            quantization_type=test_config["quantization_type"],
            enable_quantization=test_config.get("enable_quantization", True),
            hnsw_m=test_config["hnsw_m"]
        )
        
        vector_store = QdrantVectorStore(config)
        collection_config = vector_store._create_collection_config()
        
        print(f"    ✅ Конфигурация сгенерирована")
        print(f"    ➤ Размер вектора: {collection_config['vectors_config'].size}")
        print(f"    ➤ Distance: {collection_config['vectors_config'].distance}")
        print(f"    ➤ On disk: {collection_config['vectors_config'].on_disk}")
        print(f"    ➤ Квантование: {'Включено' if collection_config['vectors_config'].quantization_config else 'Выключено'}")
    
    # Все конфигурации успешно сгенерированы - используем assert
    assert config is not None
    assert collection_config is not None
    assert 'vectors_config' in collection_config

def test_points_validation():
    """Тестирует валидацию точек"""
    print("\n🔄 Тестирование валидации точек...")
    
    from config import VectorStoreConfig
    from rag.vector_store import QdrantVectorStore
    
    config = VectorStoreConfig(vector_size=1024)
    vector_store = QdrantVectorStore(config)
    
    # Тестовые точки
    test_points = [
        {
            # Корректная точка
            "id": "test_1",
            "vector": np.random.random(1024).astype(np.float32),
            "payload": {
                "file": "test.py",
                "chunk_id": "chunk_1",
                "hash": "abc123"
            }
        },
        {
            # Точка без ID (должен быть сгенерирован)
            "vector": np.random.random(1024).astype(np.float32),
            "payload": {"file": "test2.py"}
        },
        {
            # Точка с некорректной размерностью (должна быть отброшена)
            "id": "bad_point",
            "vector": np.random.random(128).astype(np.float32),
            "payload": {"file": "bad.py"}
        }
    ]
    
    validated_points = vector_store._validate_points(test_points)
    
    print(f"    ✅ Валидация завершена")
    print(f"    ➤ Входных точек: {len(test_points)}")
    print(f"    ➤ Валидных точек: {len(validated_points)}")
    
    # Проверяем первую точку
    if validated_points:
        first_point = validated_points[0]
        print(f"    ➤ ID первой точки: {first_point['id']}")
        print(f"    ➤ Размер вектора: {len(first_point['vector'])}")
        print(f"    ➤ Payload полей: {len(first_point['payload'])}")
    
    # Проверяем что получили 2 валидные точки
    assert len(validated_points) == 2, f"Ожидается 2 валидные точки, получено {len(validated_points)}"
    
    # Используем assert вместо return
    assert validated_points is not None
    assert len(validated_points) == 2

def main():
    """Основная функция тестирования"""
    print("🚀 Базовое тестирование QdrantVectorStore")
    print("=" * 50)
    
    tests = [
        ("Импорты", test_basic_imports),
        ("Создание конфигурации", test_config_creation),
        ("Инициализация", test_qdrant_initialization), 
        ("Генерация конфигурации коллекции", test_collection_config_generation),
        ("Валидация точек", test_points_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            test_func()  # Просто вызываем функцию без возврата значения
            results.append((test_name, True))  # Если не было исключений, тест прошел
            print(f"  ✅ Тест '{test_name}' пройден")
        except Exception as e:
            print(f"  ❌ Ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Финальный отчет
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    
    success_count = 0
    for test_name, success in results:
        status = "✅ ПРОЙДЕН" if success else "❌ НЕ ПРОЙДЕН"
        print(f"  {test_name:<35} {status}")
        if success:
            success_count += 1
    
    print(f"\nИтого: {success_count}/{len(results)} тестов пройдено")
    
    if success_count == len(results):
        print("🎉 ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
        print("✅ QdrantVectorStore готов к использованию")
        return True
    else:
        print("⚠️  Обнаружены проблемы в реализации")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Тестирование прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка тестирования: {e}")
        traceback.print_exc()
        sys.exit(1)
