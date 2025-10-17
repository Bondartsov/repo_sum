import pytest
from rag.remote_vector_store import RemoteVMVectorStore

class DummyTransport:
    def __init__(self):
        self.last_url = None
        self.last_payload = None
        self.last_headers = None
        self.last_timeout = None
    async def post_json(self, url, payload, timeout, headers=None):
        self.last_url = url
        self.last_payload = payload
        self.last_headers = headers
        self.last_timeout = timeout
        return {"results": []}

@pytest.mark.asyncio
async def test_hybrid_protocol_v2_endpoint():
    dummy = DummyTransport()
    store = RemoteVMVectorStore(transport_client=dummy)
    dense = [0.0] * 1024
    sparse = {1: 0.6, 2: 0.4}
    await store._async_search(dense, top_k=10, filters=None, use_hybrid=True, sparse_vector=sparse)
    assert dummy.last_url.endswith("/v1/search_v2")
    p = dummy.last_payload
    assert p.get("protocol") == "hybrid"
    assert isinstance(p.get("dense_vector"), list) and len(p["dense_vector"]) == 1024
    assert p.get("use_hybrid") is True
    assert isinstance(p.get("sparse_vector"), dict)
    assert set(p["sparse_vector"].keys()) == {1, 2}

@pytest.mark.asyncio
async def test_dense_protocol_v2_endpoint():
    dummy = DummyTransport()
    store = RemoteVMVectorStore(transport_client=dummy)
    dense = [0.0] * 1024
    await store._async_search(dense, top_k=5, filters=None, use_hybrid=False, sparse_vector=None)
    assert dummy.last_url.endswith("/v1/search_v2")
    p = dummy.last_payload
    assert p.get("protocol") == "dense"
    assert "sparse_vector" not in p or p["sparse_vector"] in (None, {})
    assert isinstance(p.get("dense_vector"), list) and len(p["dense_vector"]) == 1024