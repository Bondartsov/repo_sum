import pytest
import asyncio
from types import SimpleNamespace

from rag.indexer_service import IndexerService
from config import Config


class DummyEmbedder:
    def __init__(self):
        self._stats = {
            "is_warmed_up": True,
            "provider": "mock",
            "model_name": "mock-model"
        }

    def warmup(self):
        return None

    def get_stats(self):
        return self._stats

    def reset_stats(self):
        self._stats["is_warmed_up"] = False


class DummyVectorStore:
    def __init__(self, status):
        self._status = status

    def health_check(self):
        return {"status": self._status}

    def get_stats(self):
        return {"calls": 1}

    def reset_stats(self):
        return None

    async def close(self):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("vs_status, expected", [
    ("connected", "healthy"),
    ("ok", "healthy"),
    ("healthy", "healthy"),
    ("error", "degraded"),
])
async def test_health_check_statuses(vs_status, expected):
    cfg = Config()
    service = IndexerService(cfg, silent_mode=True)

    # Подменяем зависимости
    service.vector_store = DummyVectorStore(vs_status)
    service.embedder = DummyEmbedder()

    result = await service.health_check()

    assert "status" in result
    assert result["status"] == expected
    assert "components" in result
    assert "vector_store" in result["components"]
    assert "embedder" in result["components"]