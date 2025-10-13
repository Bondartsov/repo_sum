import math
from types import SimpleNamespace
from aiohttp import ClientTimeout

from rag.remote_vector_store import RemoteVMVectorStore


class DummyTransport:
    def __init__(self):
        self.last_timeout = None
        self.calls = []

    async def post_json(self, url, payload, timeout, headers=None):
        self.last_timeout = timeout
        self.calls.append((url, payload, timeout, headers))
        return {"results": []}


def test_search_by_text_uses_split_timeouts_for_client():
    dummy = DummyTransport()
    rvs = RemoteVMVectorStore(transport_client=dummy)
    # Настройка профиля таймаутов для теста: общий бюджет 3.0s
    rvs.search_timeout = 10  # fallback больше, чем профиль
    rvs.timeout_profiles = SimpleNamespace(search_total_p95_sec=3.0)

    # Вызов не должен кидать исключения и должен вернуть пустой список
    results = rvs.search_by_text('q', top_k=1)
    assert isinstance(results, list)
    assert results == []

    # Проверяем, что DummyTransport получил aiohttp.ClientTimeout с раздельными полями
    timeout = dummy.last_timeout
    assert isinstance(timeout, ClientTimeout)

    assert math.isclose(timeout.total, 3.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(timeout.sock_read, 3.0, rel_tol=0.0, abs_tol=1e-6)
    expected_connect = 0.6  # min(1.0, max(0.05, 3.0 * 0.2))
    assert math.isclose(timeout.connect, expected_connect, rel_tol=0.0, abs_tol=1e-6)