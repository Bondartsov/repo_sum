"""
Unit тесты для CPUQueryEngine: проверяем исправления health_check и fallback эмбеддингов.
"""

from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
from unittest.mock import AsyncMock, Mock, patch

from config import QueryEngineConfig, VectorStoreConfig
from rag.embedder import CPUEmbedder
from rag.query_engine import CPUQueryEngine, SearchResult
from rag.vector_store import QdrantVectorStore


@pytest.mark.unit
class TestQueryEngineHealthFixes:
    """Проверяет исправления, описанные в tests/fix_tests.md."""

    @staticmethod
    def _make_config(vector_size: int = 1024) -> QueryEngineConfig:
        cfg = QueryEngineConfig()
        cfg.vector_store = VectorStoreConfig()
        cfg.vector_store.vector_size = vector_size
        return cfg

    @staticmethod
    def _make_vector_store(payload: Any = None) -> SimpleNamespace:
        store = SimpleNamespace()
        store.config = SimpleNamespace(host="localhost", port=6333, collection_name="test")
        ok_payload: Dict[str, Any] = {
            "status": "ok",
            "components": {
                "vector_store": {
                    "collection_status": "exists"
                }
            }
        }
        value = payload if payload is not None else ok_payload
        store.check_health = AsyncMock(return_value=value)
        store.health_check = AsyncMock(return_value=value)
        return store

    @staticmethod
    def _make_query_engine(embedder: Mock, store: SimpleNamespace, cfg: QueryEngineConfig) -> CPUQueryEngine:
        with patch("rag.query_engine.SearchService"):
            return CPUQueryEngine(embedder, store, cfg)

    @pytest.mark.asyncio
    async def test_check_vector_store_success(self):
        embedder = Mock(spec=CPUEmbedder)
        store = self._make_vector_store()
        cfg = self._make_config()

        engine = self._make_query_engine(embedder, store, cfg)

        assert await engine._check_vector_store() is True
        store.check_health.assert_called_once()
        store.health_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_vector_store_disconnected(self):
        embedder = Mock(spec=CPUEmbedder)
        payload = {
            "status": "error",
            "components": {
                "vector_store": {
                    "collection_status": "not_found"
                }
            },
            "error": "Connection failed"
        }
        store = self._make_vector_store(payload)
        cfg = self._make_config()

        engine = self._make_query_engine(embedder, store, cfg)

        assert await engine._check_vector_store() is False
        store.check_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_vector_store_exception(self):
        embedder = Mock(spec=CPUEmbedder)
        store = self._make_vector_store()
        store.check_health.side_effect = RuntimeError("boom")
        store.health_check.side_effect = RuntimeError("boom")
        cfg = self._make_config()

        engine = self._make_query_engine(embedder, store, cfg)

        assert await engine._check_vector_store() is False
        store.check_health.assert_called_once()

    def test_ensure_embeddings_uses_config_dimension(self):
        embedder = Mock(spec=CPUEmbedder)
        embedder.embed_texts.side_effect = RuntimeError("fail")

        store = Mock(spec=QdrantVectorStore)
        store.config = SimpleNamespace(host="localhost", port=6333, collection_name="test")
        cfg = self._make_config(vector_size=1024)

        engine = self._make_query_engine(embedder, store, cfg)

        results = [
            SearchResult(
                chunk_id="a",
                file_path="a.py",
                file_name="a.py",
                chunk_name="func_a",
                chunk_type="function",
                language="python",
                start_line=1,
                end_line=10,
                score=0.9,
                content="def func_a(): pass",
                metadata={},
                embedding=None,
            ),
            SearchResult(
                chunk_id="b",
                file_path="b.py",
                file_name="b.py",
                chunk_name="func_b",
                chunk_type="function",
                language="python",
                start_line=5,
                end_line=12,
                score=0.8,
                content="def func_b(): return True",
                metadata={},
                embedding=None,
            ),
        ]

        engine._ensure_embeddings(results)

        for item in results:
            assert item.embedding is not None
            assert len(item.embedding) == 1024
        embedder.embed_texts.assert_called_once()

    def test_ensure_embeddings_default_dimension(self):
        embedder = Mock(spec=CPUEmbedder)
        embedder.embed_texts.side_effect = RuntimeError("fail")

        store = Mock(spec=QdrantVectorStore)
        store.config = SimpleNamespace(host="localhost", port=6333, collection_name="test")
        cfg = QueryEngineConfig()

        engine = self._make_query_engine(embedder, store, cfg)

        result = SearchResult(
            chunk_id="fallback",
            file_path="fallback.py",
            file_name="fallback.py",
            chunk_name="fallback_func",
            chunk_type="function",
            language="python",
            start_line=1,
            end_line=3,
            score=0.7,
            content="def fallback_func(): pass",
            metadata={},
            embedding=None,
        )

        engine._ensure_embeddings([result])

        assert result.embedding is not None
        assert len(result.embedding) == 1024
        embedder.embed_texts.assert_called_once()

    def test_ensure_embeddings_successful(self):
        embedder = Mock(spec=CPUEmbedder)
        embeddings = np.array([
            np.random.random(1024),
            np.random.random(1024),
        ])
        embedder.embed_texts.return_value = embeddings

        store = Mock(spec=QdrantVectorStore)
        store.config = SimpleNamespace(host="localhost", port=6333, collection_name="test")
        cfg = self._make_config(vector_size=1024)

        engine = self._make_query_engine(embedder, store, cfg)

        results = [
            SearchResult(
                chunk_id="x",
                file_path="x.py",
                file_name="x.py",
                chunk_name="func_x",
                chunk_type="function",
                language="python",
                start_line=1,
                end_line=4,
                score=0.95,
                content="def func_x(): return 1",
                metadata={},
                embedding=None,
            ),
            SearchResult(
                chunk_id="y",
                file_path="y.py",
                file_name="y.py",
                chunk_name="func_y",
                chunk_type="function",
                language="python",
                start_line=10,
                end_line=16,
                score=0.85,
                content="def func_y(): return 2",
                metadata={},
                embedding=None,
            ),
        ]

        engine._ensure_embeddings(results)

        np.testing.assert_array_equal(results[0].embedding, embeddings[0])
        np.testing.assert_array_equal(results[1].embedding, embeddings[1])
        embedder.embed_texts.assert_called_once_with([
            "def func_x(): return 1",
            "def func_y(): return 2",
        ])

    @pytest.mark.asyncio
    async def test_health_check_integration(self):
        embedder = Mock(spec=CPUEmbedder)
        store = self._make_vector_store()
        cfg = self._make_config()

        engine = self._make_query_engine(embedder, store, cfg)

        result = await engine.health_check()

        assert result["status"] == "healthy"
        assert result["embedder_status"] == "ok"
        assert result["vector_store_status"] == "ok"
        store.check_health.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
