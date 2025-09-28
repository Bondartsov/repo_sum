import asyncio
import numpy as np
from typing import List, Optional

from rag.exceptions import EmbeddingException


class MockRemoteEmbedder:
    """Простой mock-режим для RemoteVMEmbedder без сетевых вызовов."""

    def __init__(self, embedding_config=None, parallelism_config=None, remote_service_config=None):
        self.embedding_config = embedding_config
        self.parallelism_config = parallelism_config
        self.remote_service_config = remote_service_config
        self.truncate_dim = getattr(embedding_config, "truncate_dim", 1024) if embedding_config else 1024
        self.model_name = getattr(embedding_config, "model_name", "mock-remote") if embedding_config else "mock-remote"
        self.provider_name = getattr(embedding_config, "provider", "mock-remote") if embedding_config else "mock-remote"
        self.stats = {
            "total_requests": 0,
            "total_texts": 0,
            "error_count": 0,
        }

    def embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        deadline_ms: int = 30000,
    ) -> np.ndarray:
        """Синхронная обёртка, вызывающая async версию для совместимости."""
        if not texts:
            return np.zeros((0, self.truncate_dim), dtype=np.float32)
        self.stats["total_requests"] += 1
        self.stats["total_texts"] += len(texts)

        async def runner() -> np.ndarray:
            return await self._async_embed_texts(texts, task=task, deadline_ms=deadline_ms)

        try:
            return asyncio.run(runner())
        except RuntimeError as loop_error:
            message = str(loop_error)
            if "asyncio.run() cannot be called" in message:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._async_embed_texts(texts, task=task, deadline_ms=deadline_ms))
            self.stats['error_count'] += 1
            raise EmbeddingException(message, provider=self.provider_name, model_name=self.model_name) from loop_error
        except Exception as exc:
            self.stats['error_count'] += 1
            raise EmbeddingException(str(exc), provider=self.provider_name, model_name=self.model_name) from exc

    async def _async_embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        deadline_ms: int = 30000,
    ) -> np.ndarray:
        return self.embed_texts(texts, task=task, deadline_ms=deadline_ms)

    def _create_vector(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(self.truncate_dim, dtype=np.float32)

    def close(self) -> None:
        return None

    def health_check(self) -> bool:
        """Проверка доступности mock-эмбеддера (всегда True в offline-режиме)."""
        return True

    def warmup(self) -> bool:
        """Прогревочный метод для совместимости с RemoteVMEmbedder (всегда True)."""
        return True
