"""Тесты для проверки исправлений в RemoteVMEmbedder."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional

import aiohttp
import pytest

pytest.importorskip("freezegun")
from freezegun import freeze_time

from config import RemoteServiceConfig
from rag.exceptions import VMConnectionError, VMTimeoutError
from tests.mocks.mock_transport_client import MockTransportClient

pytestmark = pytest.mark.real_embedder


def create_transport_spy(
    should_fail: bool = False,
    failure_count: Optional[int] = 3,
    *,
    exception_factory: Optional[Callable[[], Exception]] = None,
    response_payload: Optional[Dict[str, Any]] = None,
):
    """Создает транспортный spy для подмены HTTP клиента."""
    spy = MockTransportClient()
    spy.should_fail = should_fail
    spy.failures_before_success = failure_count

    if exception_factory is not None:
        spy.exception_factory = exception_factory

    if response_payload is not None:
        spy.response_payload = response_payload

    def get_spy_stats() -> Dict[str, Any]:
        return {
            "call_count": spy.call_count,
            "calls_history": list(spy.calls_history),
        }

    return spy, get_spy_stats


@pytest.mark.asyncio
async def test_timeout_no_keyerror(embedder_factory):
    """При таймауте должен возникать VMTimeoutError без KeyError."""
    remote_config = RemoteServiceConfig(
        host="localhost",
        port=8000,
        timeout_seconds=1,
        max_retries=3,
        retry_delay=0.0,
    )
    spy_transport, transport_stats = create_transport_spy(
        should_fail=True,
        failure_count=remote_config.max_retries,
        exception_factory=lambda: asyncio.TimeoutError("Mock transport timeout"),
    )
    embedder = embedder_factory(
        remote_service_config=remote_config,
        transport_client=spy_transport,
    )

    with freeze_time("2024-01-01 00:00:00"):
        with pytest.raises(VMTimeoutError) as exc_info:
            await embedder._make_request_with_retry({"test": "data"})

    error = exc_info.value
    assert hasattr(error, "elapsed_seconds")
    assert error.elapsed_seconds >= 0

    stats = embedder.get_stats()
    retry_stats = stats["retry_policy_stats"]
    assert stats["retry_count"] == remote_config.max_retries - 1
    assert retry_stats["failed_executions"] == 1
    assert transport_stats()["call_count"] == remote_config.max_retries


@pytest.mark.asyncio
async def test_circuit_breaker_composition(embedder_factory):
    """Circuit breaker должен видеть каждую попытку retry."""
    remote_config = RemoteServiceConfig(
        host="localhost",
        port=8000,
        timeout_seconds=1,
        max_retries=4,
        retry_delay=0.0,
    )
    spy_transport, transport_stats = create_transport_spy(
        should_fail=True,
        failure_count=remote_config.max_retries,
        exception_factory=lambda: aiohttp.ClientError("Mock failure"),
    )
    embedder = embedder_factory(
        remote_service_config=remote_config,
        transport_client=spy_transport,
    )

    with freeze_time("2024-01-01 00:00:00"):
        with pytest.raises(VMConnectionError):
            await embedder._make_request_with_retry({"test": "data"})

    cb_stats = embedder.circuit_breaker.get_stats()
    assert cb_stats["failed_calls"] == remote_config.max_retries

    stats = embedder.get_stats()
    assert stats["retry_count"] == remote_config.max_retries - 1
    assert transport_stats()["call_count"] == remote_config.max_retries


@pytest.mark.asyncio
async def test_retry_count_metric(embedder_factory):
    """retry_count должен отражать фактическое количество retry попыток."""
    remote_config = RemoteServiceConfig(
        host="localhost",
        port=8000,
        timeout_seconds=1,
        max_retries=4,
        retry_delay=0.0,
    )
    spy_transport, transport_stats = create_transport_spy(
        should_fail=True,
        failure_count=2,
        exception_factory=lambda: aiohttp.ClientError("Temporary failure"),
        response_payload={"embeddings": [[0.1, 0.2]]},
    )
    embedder = embedder_factory(
        remote_service_config=remote_config,
        transport_client=spy_transport,
    )
    embedder.retry_policy.reset_stats()
    embedder.reset_stats()

    with freeze_time("2024-01-01 00:00:00"):
        result = await embedder._make_request_with_retry({"test": "data"})

    assert result == [[0.1, 0.2]]

    stats = embedder.get_stats()
    retry_stats = stats["retry_policy_stats"]
    assert stats["retry_count"] == 2
    assert retry_stats["total_retries"] == 2
    assert retry_stats["successful_executions"] == 1
    assert transport_stats()["call_count"] == 3
