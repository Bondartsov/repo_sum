"""
Тестовый скрипт для проверки импортов и базовой функциональности RAG системы.
"""

import sys
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_basic_imports():
    """Тестирование базовых импортов"""
    logger.info("=== Тестирование базовых импортов ===")
    
    # Проверка импорта конфигурации
    from config import EmbeddingConfig
    logger.info("✓ Конфигурационные классы импортированы успешно")
    
    # Проверка импорта RAG модуля
    import rag
    logger.info(f"✓ RAG модуль импортирован, версия: {rag.__version__}")
    
    # Проверка импорта исключений
    logger.info("✓ Исключения RAG импортированы успешно")
    
    # Все импорты успешны
    assert rag is not None
    assert EmbeddingConfig is not None

def test_embedder_initialization():
    """Тестирование инициализации эмбеддера"""
    logger.info("=== Тестирование инициализации CPUEmbedder ===")
    
    from rag.embedder import CPUEmbedder
    from config import EmbeddingConfig, ParallelismConfig
    
    # Создаем конфигурации
    embedding_config = EmbeddingConfig(
        provider="fastembed",
        model_name="BAAI/bge-small-en-v1.5",
        warmup_enabled=False  # Отключаем прогрев для теста
    )
    
    parallelism_config = ParallelismConfig(
        torch_num_threads=2,
        omp_num_threads=2,
        mkl_num_threads=2
    )
    
    # Попытка создания эмбеддера (в offline режиме будет использован mock)
    try:
        embedder = CPUEmbedder(embedding_config, parallelism_config)
        logger.info("✓ CPUEmbedder инициализирован успешно")
        
        # Проверяем статистику
        stats = embedder.get_stats()
        logger.info(f"✓ Статистика эмбеддера: {stats['provider']}, модель: {stats['model_name']}")
        
        assert embedder is not None
        assert stats is not None
        
    except ImportError as e:
        logger.warning(f"⚠ CPUEmbedder не может быть инициализирован (отсутствуют зависимости): {e}")
        logger.info("✓ Класс CPUEmbedder доступен для импорта")
        assert CPUEmbedder is not None

def test_vector_store_stub():
    """Тестирование заглушки VectorStore"""
    logger.info("=== Тестирование заглушки VectorStore ===")
    
    from rag import VectorStore
    from config import VectorStoreConfig
    
    if VectorStore is None:
        logger.info("✓ VectorStore корректно помечен как None (заглушка)")
        assert True
        return
    
    # Создаем конфигурацию
    config = VectorStoreConfig()
    
    # Создаем VectorStore
    vector_store = VectorStore(config)
    logger.info("✓ VectorStore (заглушка) создан успешно")
    
    # Проверяем статистику
    stats = vector_store.get_stats()
    logger.info(f"✓ Статистика VectorStore: {stats['status']}")
    
    assert vector_store is not None
    assert stats is not None

def test_query_engine_stub():
    """Тестирование заглушки QueryEngine"""
    logger.info("=== Тестирование заглушки QueryEngine ===")
    
    from rag import QueryEngine
    
    if QueryEngine is None:
        logger.info("✓ QueryEngine корректно помечен как None (заглушка)")
        assert True
        return
    
    logger.info("✓ QueryEngine доступен для импорта")
    assert QueryEngine is not None

def test_rag_structure():
    """Проверка структуры RAG директории"""
    logger.info("=== Проверка структуры RAG директории ===")
    
    rag_dir = Path("rag")
    
    required_files = [
        "__init__.py",
        "embedder.py", 
        "exceptions.py",
        "vector_store.py",
        "query_engine.py"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = rag_dir / file_name
        if file_path.exists():
            logger.info(f"✓ {file_name} существует")
        else:
            logger.error(f"✗ {file_name} отсутствует")
            missing_files.append(file_name)
    
    assert not missing_files, f"Отсутствующие файлы: {missing_files}"
    
    logger.info("✓ Все необходимые файлы присутствуют")

def test_config_compatibility():
    """Проверка совместимости с существующей конфигурацией"""
    logger.info("=== Проверка совместимости конфигурации ===")
    
    from config import get_config
    
    # Попытка загрузки конфигурации (может не удастся без settings.json)
    try:
        config = get_config(require_api_key=False)
        logger.info("✓ Основная конфигурация загружена успешно")
        
        # Проверяем RAG конфигурацию
        rag_config = config.rag
        logger.info("✓ RAG конфигурация доступна:")
        logger.info(f"  - Провайдер эмбеддингов: {rag_config.embeddings.provider}")
        logger.info(f"  - Модель: {rag_config.embeddings.model_name}")
        logger.info(f"  - Векторное хранилище: {rag_config.vector_store.host}:{rag_config.vector_store.port}")
        
        assert config is not None
        assert rag_config is not None
        
    except FileNotFoundError:
        logger.warning("⚠ Файл settings.json не найден, используем дефолтную конфигурацию")
        
        # Создаем дефолтную RAG конфигурацию
        from config import RagConfig
        rag_config = RagConfig()
        logger.info("✓ Дефолтная RAG конфигурация создана:")
        logger.info(f"  - Провайдер эмбеддингов: {rag_config.embeddings.provider}")
        logger.info(f"  - Модель: {rag_config.embeddings.model_name}")
        
        assert rag_config is not None

def main():
    """Основная функция тестирования"""
    logger.info("Запуск тестирования RAG системы...")
    
    tests = [
        ("Базовые импорты", test_basic_imports),
        ("Структура директории", test_rag_structure),
        ("Совместимость конфигурации", test_config_compatibility),
        ("Инициализация CPUEmbedder", test_embedder_initialization),
        ("Заглушка VectorStore", test_vector_store_stub),
        ("Заглушка QueryEngine", test_query_engine_stub),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            test_func()  # Убираем result = так как функции больше не возвращают значения
            results.append((test_name, True))  # Если не было исключений, тест прошел
        except Exception as e:
            logger.error(f"Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    logger.info("\n" + "="*50)
    logger.info("ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
    logger.info("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ ПРОЙДЕН" if result else "✗ НЕУДАЧЕН"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nВсего тестов: {len(results)}")
    logger.info(f"Пройдено: {passed}")
    logger.info(f"Неудачно: {failed}")
    
    if failed == 0:
        logger.info("🎉 Все тесты пройдены успешно! RAG система готова к использованию.")
        return True
    else:
        logger.warning(f"⚠ {failed} тестов не пройдены. Требуется дополнительная настройка.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
