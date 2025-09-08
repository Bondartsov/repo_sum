# tests/conftest.py
# Общие фикстуры для pytest (если понадобятся)
import pytest
import sys
import os
from unittest.mock import patch

def pytest_addoption(parser):
    parser.addoption(
        "--run-symlink-tests",
        action="store_true",
        default=False,
        help="Явно попытаться запускать тесты, создающие symlink (Windows требует права администратора/Developer Mode)"
    )

# Здесь можно определить фикстуры для всего проекта

# Автоматический патчинг CPUEmbedder для offline тестов
def pytest_configure(config):
    """Конфигурация pytest с автоматическим патчингом для offline тестов"""
    
    # Проверяем, нужно ли использовать mock эмбеддеры
    from tests.mocks.mock_cpu_embedder import should_use_mock_embedder
    
    if should_use_mock_embedder():
        print("\n🔄 Обнаружен offline режим - активируем mock эмбеддеры")
        
        # Патчим CPUEmbedder на уровне модуля
        try:
            from tests.mocks.mock_cpu_embedder import MockCPUEmbedder
            
            # Создаем патч для rag.embedder.CPUEmbedder
            embedder_patcher = patch('rag.embedder.CPUEmbedder', MockCPUEmbedder)
            embedder_patcher.start()
            
            # Также патчим импорты из rag пакета
            rag_embedder_patcher = patch('rag.CPUEmbedder', MockCPUEmbedder) 
            rag_embedder_patcher.start()
            
            # ВАЖНО: патчим IndexerService который импортирует CPUEmbedder напрямую
            indexer_embedder_patcher = patch('rag.indexer_service.CPUEmbedder', MockCPUEmbedder)
            indexer_embedder_patcher.start()

            # Дополнительно: патчим точки прямого импорта CPUEmbedder в сервисах поиска/движке
            search_embedder_patcher = patch('rag.search_service.CPUEmbedder', MockCPUEmbedder)
            search_embedder_patcher.start()
            query_engine_embedder_patcher = patch('rag.query_engine.CPUEmbedder', MockCPUEmbedder)
            query_engine_embedder_patcher.start()
            
            # Сохраняем патчеры для отключения в конце
            if not hasattr(config, '_mock_patchers'):
                config._mock_patchers = []
            config._mock_patchers.extend([
                embedder_patcher,
                rag_embedder_patcher,
                indexer_embedder_patcher,
                search_embedder_patcher,
                query_engine_embedder_patcher,
            ])
            
            print("✅ Mock эмбеддеры активированы")
            
        except ImportError as e:
            print(f"⚠️  Не удалось активировать mock эмбеддеры: {e}")


def pytest_unconfigure(config):
    """Очистка патчеров после завершения тестов"""
    if hasattr(config, '_mock_patchers'):
        for patcher in config._mock_patchers:
            try:
                patcher.stop()
            except Exception:
                pass  # Игнорируем ошибки при остановке патчеров


@pytest.fixture(autouse=True)
def reset_embedder_environment():
    """Автоматически сбрасывает состояние эмбеддеров между тестами"""
    yield
    
    # Принудительная сборка мусора для освобождения ресурсов
    import gc
    gc.collect()


@pytest.fixture
def mock_cpu_embedder_offline():
    """
    Фикстура для принудительного использования mock эмбеддера.
    Полезна для конкретных тестов, которым нужен гарантированно mock эмбеддер.
    """
    from tests.mocks.mock_cpu_embedder import MockCPUEmbedder
    from config import EmbeddingConfig, ParallelismConfig
    
    # Создаем базовую конфигурацию для mock'а
    embedding_config = EmbeddingConfig(
        provider="fastembed",
        model_name="BAAI/bge-small-en-v1.5",
        batch_size_min=4,
        batch_size_max=16,
        warmup_enabled=False
    )
    
    parallelism_config = ParallelismConfig(
        torch_num_threads=2,
        omp_num_threads=2,
        mkl_num_threads=2
    )
    
    return MockCPUEmbedder(embedding_config, parallelism_config)
