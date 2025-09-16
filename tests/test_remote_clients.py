import numpy as np
import pytest

from rag.remote_embedder import RemoteVMEmbedder
from rag.remote_vector_store import RemoteVMVectorStore


@pytest.mark.parametrize("texts", [["one"], ["a", "b"]])
def test_remote_embedder_sync_wrapper(monkeypatch, texts):
    embedder = RemoteVMEmbedder()

    async def fake_async_embed(texts, task=None, deadline_ms=30000):
        return np.ones((len(texts), embedder.truncate_dim), dtype=np.float32)

    monkeypatch.setattr(embedder, "_async_embed_texts", fake_async_embed)

    result = embedder.embed_texts(texts)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(texts), embedder.truncate_dim)


def test_remote_embedder_fallback(monkeypatch):
    embedder = RemoteVMEmbedder()

    async def failing_async_embed(texts, task=None, deadline_ms=30000):
        raise RuntimeError("boom")

    monkeypatch.setattr(embedder, "_async_embed_texts", failing_async_embed)

    result = embedder.embed_texts(["demo"])
    assert np.array_equal(result, np.zeros((1, embedder.truncate_dim), dtype=np.float32))
    assert embedder.stats['error_count'] >= 1


def test_remote_vector_store_search_sync(monkeypatch):
    store = RemoteVMVectorStore()

    async def fake_async_search(query_vector, top_k, filters, use_hybrid, sparse_vector):
        return [{'id': '1', 'score': 1.0, 'payload': {}}]

    monkeypatch.setattr(store, "_async_search", fake_async_search)

    results = store.search(np.zeros(1), 1, filters=None, use_hybrid=False, sparse_vector=None)
    assert isinstance(results, list)
    assert results and results[0]['score'] == 1.0


def test_remote_vector_store_health_sync(monkeypatch):
    store = RemoteVMVectorStore()

    async def fake_async_health():
        return {'status': 'connected'}

    monkeypatch.setattr(store, "_async_health_check", fake_async_health)

    health = store.health_check()
    assert health['status'] == 'connected'
