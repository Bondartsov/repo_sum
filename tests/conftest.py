# tests/conftest.py
# Общие фикстуры для pytest (если понадобятся)
# ФАЗА 4: КРИТИЧЕСКИЙ РЕФАКТОРИНГ - удалён глобальный патчинг, добавлены scoped фикстуры
import os
import sys
from typing import Any, Optional
from unittest.mock import patch

import pytest

# ИСПРАВЛЕНИЕ #1: Изменён дефолт USE_MOCK_EMBEDDER с "1" на "0"
os.environ.setdefault("USE_MOCK_EMBEDDER", "0")

def pytest_addoption(parser):
    """Добавляет CLI опции для управления режимами тестирования."""

    parser.addoption(
        "--run-symlink-tests",
        action="store_true",
        default=False,
        help="Явно попытаться запускать тесты, создающие symlink (Windows требует права администратора/Developer Mode)"
    )
    parser.addoption(
        "--use-mock-embedder",
        action="store_true",
        default=False,
        help="Использовать mock эмбеддер вместо реального RemoteVMEmbedder"
    )
    parser.addoption(
        "--vm-host",
        action="store",
        default=None,
        help="Хост удалённой VM для интеграционных тестов"
    )
    parser.addoption(
        "--vm-port",
        action="store",
        default=8000,
        type=int,
        help="Порт удалённой VM для интеграционных тестов"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_event_loop_policy():
    """Устанавливает WindowsSelectorEventLoopPolicy на Windows для стабильных async-тестов."""

    import asyncio

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    yield


@pytest.fixture  # ИСПРАВЛЕНИЕ #2: Убран autouse=True
def force_offline_env(monkeypatch):
    """Гарантирует offline-профиль по умолчанию для тестов"""
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("DISABLE_REAL_EMBEDDINGS", "1")  # ИСПРАВЛЕНИЕ #3: Заменён USE_MOCK_EMBEDDER
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

# ИСПРАВЛЕНИЕ #4: Удалён глобальный патчинг из pytest_configure
def pytest_configure(config):
    """Регистрирует пользовательские маркеры без глобального патчинга."""

    config.addinivalue_line("markers", "real_embedder: Тесты, требующие реальный RemoteVMEmbedder")
    config.addinivalue_line("markers", "mock_embedder: Тесты, требующие mock-эмбеддер")
    config.addinivalue_line("markers", "vm: Тесты, требующие доступности VM сервиса")


def pytest_unconfigure(config):
    """Очистка ресурсов после завершения тестов"""
    # Cleanup для session-scoped патчеров если они были применены
    if hasattr(config, '_mock_patchers'):
        for patcher in config._mock_patchers:
            try:
                patcher.stop()
            except Exception:
                pass  # Игнорируем ошибки при остановке патчеров


def check_vm_availability(host: str, port: int, timeout: float = 0.5) -> bool:
    """Проверяет доступность VM через TCP-подключение."""

    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error, OSError):
        return False


@pytest.fixture(scope="session")
def embedder_factory(request):
    """Фабрика для создания mock или реального RemoteVMEmbedder."""

    use_mock_cli = request.config.getoption("--use-mock-embedder")
    env_flag = os.getenv("USE_MOCK_EMBEDDER", "0").lower() in {"1", "true", "yes", "on"}

    def _create_embedder(
        override_mock: Optional[bool] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        transport_client: Optional[Any] = None,
        remote_service_config: Optional[Any] = None,
    ):
        should_mock = override_mock if override_mock is not None else (use_mock_cli or env_flag)

        if should_mock:
            from tests.mocks.mock_remote_embedder import MockRemoteEmbedder

            embedder = MockRemoteEmbedder()
            if model is not None:
                setattr(embedder, "model_name", model)
            if provider is not None:
                setattr(embedder, "provider_name", provider)
            return embedder

        from rag.remote_embedder import RemoteVMEmbedder

        # Создаём embedder с переданными параметрами
        embedder = RemoteVMEmbedder(
            remote_service_config=remote_service_config,
            transport_client=transport_client
        )
        if model is not None:
            setattr(embedder, "model_name", model)
        if provider is not None:
            setattr(embedder, "provider_name", provider)
        return embedder

    return _create_embedder


@pytest.fixture(scope="session")
def mock_embedder_session(request):
    """Сессионный патч RemoteVMEmbedder на mock-реализацию при необходимости."""

    env_flag = os.getenv("USE_MOCK_EMBEDDER", "0").lower() in {"1", "true", "yes", "on"}
    if not (request.config.getoption("--use-mock-embedder") or env_flag):
        yield
        return

    from tests.mocks.mock_remote_embedder import MockRemoteEmbedder

    patchers = []
    targets = [
        "rag.remote_embedder.RemoteVMEmbedder",
        "rag.CPUEmbedder",
    ]

    try:
        for target in targets:
            try:
                patcher = patch(target, MockRemoteEmbedder)
            except (AttributeError, ImportError):
                continue
            patcher.start()
            patchers.append(patcher)

        yield
    finally:
        for patcher in patchers:
            patcher.stop()


def pytest_collection_modifyitems(config, items):
    """Применяет маркеры пропуска для mock/real embedder и VM-тестов."""

    env_true = {"1", "true", "yes", "on"}
    use_mock = config.getoption("--use-mock-embedder") or os.getenv("USE_MOCK_EMBEDDER", "0").lower() in env_true

    vm_host = config.getoption("--vm-host") or os.getenv("RAG_SERVICE_HOST", "10.61.11.54")
    vm_port_option = config.getoption("--vm-port")
    vm_port_env = os.getenv("RAG_SERVICE_PORT")

    try:
        vm_port = vm_port_option or (int(vm_port_env) if vm_port_env else 8000)
    except ValueError:
        vm_port = 8000

    vm_status_cache: Optional[bool] = None

    for item in items:
        # Пропускаем real_embedder тесты в mock режиме
        if "real_embedder" in item.keywords and use_mock:
            item.add_marker(
                pytest.mark.skip(
                    reason="Требуется реальный RemoteVMEmbedder, но включён mock режим"
                )
            )

        # Пропускаем mock_embedder тесты если mock не включён
        if "mock_embedder" in item.keywords and not use_mock:
            item.add_marker(
                pytest.mark.skip(
                    reason="Тест требует mock-эмбеддер. Запустите с --use-mock-embedder или установите USE_MOCK_EMBEDDER=1"
                )
            )

        # Проверяем доступность VM для vm тестов
        if "vm" in item.keywords:
            if vm_status_cache is None:
                vm_status_cache = check_vm_availability(vm_host, vm_port, timeout=0.5)
            if not vm_status_cache:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"VM endpoint {vm_host}:{vm_port} недоступен. Запустите сервис VM или пропустите vm-тесты."
                        )
                    )
                )


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
        model_name="jinaai/jina-embeddings-v3",  # Обновлённая модель
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
