"""Контрактные тесты для EmbedderProtocol.

Автор: AI Assistant (gpt-5-codex)
Дата: 03 октября 2025
"""
from __future__ import annotations

import inspect
from typing import Dict

import numpy as np
import pytest

from rag.embedder_protocol import EmbedderProtocol
from rag.remote_embedder import RemoteVMEmbedder
from tests.mocks.mock_remote_embedder import MockRemoteEmbedder
from tests.mocks.mock_transport_client import MockTransportClient


@pytest.fixture
def transport_client() -> MockTransportClient:
    """Возвращает mock транспорт с детерминированным ответом."""
    client = MockTransportClient()
    client.response_payload = {"embeddings": [[0.1] * 1024]}
    return client


def _assert_stats_contract(stats: Dict[str, object]) -> None:
    """Проверяет структуру статистики embedder."""
    assert stats["schema_version"] == 1
    assert set(stats["requests"].keys()) == {"total", "errors", "texts"}
    assert set(stats["retry"].keys()) == {"total_retries", "attempts"}
    assert set(stats["latency"].keys()) == {"avg_ms", "total_time"}
    assert set(stats["cb"].keys()) == {"state", "failure_count"}
    for key in ("total_requests", "total_texts", "error_count", "retry_count", "avg_response_time"):
        assert key in stats


def test_remote_embedder_implements_protocol(transport_client: MockTransportClient) -> None:
    """RemoteVMEmbedder должен удовлетворять EmbedderProtocol."""
    embedder = RemoteVMEmbedder(transport_client=transport_client)
    assert isinstance(embedder, EmbedderProtocol)


def test_mock_embedder_implements_protocol() -> None:
    """MockRemoteEmbedder должен удовлетворять EmbedderProtocol."""
    embedder = MockRemoteEmbedder()
    assert isinstance(embedder, EmbedderProtocol)


def test_embedder_stats_contract_remote(transport_client: MockTransportClient) -> None:
    """RemoteVMEmbedder возвращает статистику с schema_version=1."""
    embedder = RemoteVMEmbedder(transport_client=transport_client)
    embeddings = embedder.embed_texts(["пример"])
    assert isinstance(embeddings, np.ndarray)
    stats = embedder.get_stats()
    _assert_stats_contract(stats)


def test_embedder_stats_contract_mock() -> None:
    """MockRemoteEmbedder возвращает корректную структуру статистики."""
    embedder = MockRemoteEmbedder()
    embedder.embed_texts(["пример"])
    stats = embedder.get_stats()
    _assert_stats_contract(stats)


def test_embedder_stats_documentation_mentions_schema_version() -> None:
    """Документация метода get_stats должна упоминать schema_version."""
    doc = inspect.getdoc(EmbedderProtocol.get_stats)
    assert doc is not None and "schema_version" in doc
