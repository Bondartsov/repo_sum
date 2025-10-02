"""Проверки для embedder_factory: маркеры и изоляция экземпляров."""

import importlib

import pytest

from tests.mocks.mock_remote_embedder import MockRemoteEmbedder


@pytest.mark.real_embedder
def test_real_embedder_marker_provides_remote_instance(embedder_factory) -> None:
    """Маркер real_embedder должен выдавать настоящий RemoteVMEmbedder."""

    embedder = embedder_factory()

    remote_module = importlib.import_module("rag.remote_embedder")
    remote_class = remote_module.RemoteVMEmbedder

    assert isinstance(embedder, remote_class), "Ожидается реальный RemoteVMEmbedder"

    stats = embedder.get_stats()
    assert stats["total_requests"] == 0, "Новые экземпляры должны иметь пустую статистику"
    assert "retry_count" in stats, "Статистика реального эмбеддера должна содержать retry_count"


@pytest.mark.mock_embedder
def test_mock_embedder_marker_provides_mock_instance(embedder_factory) -> None:
    """Маркер mock_embedder должен переключать фабрику на MockRemoteEmbedder."""

    embedder = embedder_factory()

    assert isinstance(embedder, MockRemoteEmbedder), "Должен возвращаться MockRemoteEmbedder"
    assert embedder.stats["total_requests"] == 0, "Статистика нового mock экземпляра начинается с нуля"

    embedder.embed_texts(["пример"])

    assert embedder.stats["total_requests"] >= 1, "Вызов embed_texts обновляет счётчик запросов"


def test_embedder_factory_creates_isolated_instances(embedder_factory) -> None:
    """Каждый вызов embedder_factory обязан возвращать независимые экземпляры."""

    first_embedder = embedder_factory()
    first_embedder.stats["total_requests"] = 5

    second_embedder = embedder_factory()

    assert first_embedder is not second_embedder, "Фабрика должна создавать новые объекты"
    assert first_embedder.stats is not second_embedder.stats, "Статистики не должны разделяться между экземплярами"
    assert second_embedder.stats["total_requests"] == 0, "Новый экземпляр получает чистую статистику"
