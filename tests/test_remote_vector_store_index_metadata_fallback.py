import pytest
from rag.remote_vector_store import RemoteVMVectorStore

class DummyTransport:
    def __init__(self):
        self.last_url = None
        self.last_payload = None
    async def post_json(self, url, payload, timeout, headers=None):
        self.last_url = url
        self.last_payload = payload
        # /index контракт
        return { "accepted": len(payload.get("documents", [])), "rejected": 0, "elapsed_ms": 10 }

@pytest.mark.asyncio
async def test_index_metadata_fallback_from_payload_top_level():
    dummy = DummyTransport()
    store = RemoteVMVectorStore(transport_client=dummy)
    points = [{
        "id": "p1",
        "payload": {
            "content": "print('hello')",
            "file_path": "rag/event_loop_manager.py",
            "line_start": 10,
            "line_end": 20,
            "language": "python",
            "chunk_type": "function",
            "repo": "repo_sum"
        }
    }]
    accepted = await store._async_index_documents(points)
    assert accepted == 1
    docs = dummy.last_payload["documents"]
    assert len(docs) == 1
    meta = docs[0]["metadata"]
    assert meta["file_path"] == "rag/event_loop_manager.py"
    assert meta["line_start"] == 10
    assert meta["line_end"] == 20
    assert meta["language"] == "python"
    assert meta["chunk_type"] == "function"
    assert meta["repo"] == "repo_sum"
    # убеждаемся, что не проставлены дефолтные значения
    assert meta["file_path"] != "unknown"
    assert meta["line_start"] != 0
    assert meta["line_end"] != 0