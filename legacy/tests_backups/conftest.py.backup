# tests/conftest.py
# Общие фикстуры для pytest (если понадобятся)
#ВНИМАНИЕ!!!! ФАЙЛ ТРЕБУЕТ АКТУАЛИЗАЦИИ!!!!
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

@pytest.fixture(autouse=True)
def force_offline_env(monkeypatch):
    """Гарантирует offline-профиль по умолчанию для тестов"""
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("USE_MOCK_EMBEDDER", "1")
    monkeypatch.setenv("USE_MOCK_VECTOR_STORE", "1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", os.getenv("EMBEDDING_PROVIDER", "mock"))
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", os.getenv("VECTOR_STORE_PROVIDER", "mock"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    yield

@pytest.fixture(autouse=True)
def ensure_utf8_subprocess(monkeypatch):
    """Гарантирует корректное декодирование stdout/stderr в subprocess.run."""
    import subprocess

    original_run = subprocess.run

    def patched_run(*popenargs, **kwargs):
        if kwargs.get("capture_output"):
            kwargs.setdefault("text", True)
        if kwargs.get("text") or kwargs.get("universal_newlines"):
            kwargs.setdefault("encoding", "utf-8")
            kwargs.setdefault("errors", "replace")
        return original_run(*popenargs, **kwargs)

    monkeypatch.setattr(subprocess, "run", patched_run)
    yield

# Здесь можно определить фикстуры для всего проекта

# Автоматический патчинг CPUEmbedder для offline тестов
def pytest_configure(config):
    """Конфигурация pytest с автоматическим патчингом для offline тестов"""
    
    # Проверяем, нужно ли использовать mock эмбеддеры
    from tests.mocks.mock_cpu_embedder import should_use_mock_embedder
    
    force_mock = os.getenv("USE_MOCK_EMBEDDER", "1").lower() in ("1", "true", "yes")

    if force_mock or should_use_mock_embedder():
        print("\n[offline] Обнаружен offline режим - активируем mock эмбеддеры")
        
        # Патчим CPUEmbedder на уровне модуля
        try:
            from tests.mocks.mock_cpu_embedder import MockCPUEmbedder
            
            # ВАЖНО: патчим IndexerService который импортирует CPUEmbedder напрямую
            indexer_embedder_patcher = patch('rag.indexer_service.CPUEmbedder', MockCPUEmbedder)
            indexer_embedder_patcher.start()

            # Дополнительно: патчим точки прямого импорта CPUEmbedder в сервисах поиска/движке
            search_embedder_patcher = patch('rag.search_service.CPUEmbedder', MockCPUEmbedder)
            search_embedder_patcher.start()
            query_engine_embedder_patcher = patch('rag.query_engine.CPUEmbedder', MockCPUEmbedder)
            query_engine_embedder_patcher.start()
            from tests.mocks.mock_remote_embedder import MockRemoteEmbedder
            remote_embedder_patcher = patch('rag.remote_embedder.RemoteVMEmbedder', MockRemoteEmbedder)
            remote_embedder_patcher.start()
            
            # Сохраняем патчеры для отключения в конце
            if not hasattr(config, '_mock_patchers'):
                config._mock_patchers = []
            config._mock_patchers.extend([
                indexer_embedder_patcher,
                search_embedder_patcher,
                query_engine_embedder_patcher,
                remote_embedder_patcher,
            ])
            
            print("[offline] Mock эмбеддеры активированы")
            
        except ImportError as e:
            print(f"[offline] Не удалось активировать mock эмбеддеры: {e}")


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
        model_name="BAAI/bge-small-en-v1.5", ##мы давно отказались от этой модели и выбрали jinaai-embedding-v3 и перенесли её на виртуальную машину
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
