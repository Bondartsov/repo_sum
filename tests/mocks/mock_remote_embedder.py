"""Mock реализация RemoteVMEmbedder, соответствующая EmbedderProtocol.

Автор: AI Assistant (gpt-5-codex)
Дата: 03 октября 2025
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rag.embedder_protocol import (
    CircuitBreakerProtocol,
    EmbedderProtocol,
    RetryPolicyProtocol,
)
from rag.exceptions import EmbeddingException


@dataclass
class MockRetryPolicy(RetryPolicyProtocol):
    """Минимальная retry-политика для контрактных тестов."""

    _stats: Dict[str, Any] = field(
        default_factory=lambda: {
            "total_executions": 0,
            "total_retries": 0,
            "successful_executions": 0,
            "failed_executions": 0,
        }
    )

    async def execute_with_retry(self, func, *args, **kwargs):
        """Выполняет функцию без повторных попыток, фиксируя статистику."""
        self._stats["total_executions"] += 1
        try:
            result = await func(*args, **kwargs)
            self._stats["successful_executions"] += 1
            return result
        except Exception:
            self._stats["failed_executions"] += 1
            raise

    def record_execution(self, success: bool, retry_count: int = 0) -> None:
        """Позволяет тестам вручную обновлять статистику."""
        self._stats["total_executions"] += 1
        self._stats["total_retries"] += max(retry_count, 0)
        if success:
            self._stats["successful_executions"] += 1
        else:
            self._stats["failed_executions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает копию статистики с производными метриками."""
        stats = self._stats.copy()
        executions = stats["total_executions"]
        if executions > 0:
            stats["success_rate"] = (
                stats["successful_executions"] / executions
            ) * 100
            stats["avg_retries_per_execution"] = (
                stats["total_retries"] / executions
            )
        else:
            stats["success_rate"] = 0.0
            stats["avg_retries_per_execution"] = 0.0
        return stats

    def reset_stats(self) -> None:
        """Сбрасывает статистику."""
        self._stats = {
            "total_executions": 0,
            "total_retries": 0,
            "successful_executions": 0,
            "failed_executions": 0,
        }


class MockCircuitBreaker(CircuitBreakerProtocol):
    """Упрощённый circuit breaker с явной статистикой."""

    def __init__(self) -> None:
        self.state: str = "closed"
        self.failure_count: int = 0
        self._stats: Dict[str, Any] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
        }

    async def call(self, func, *args, **kwargs):
        """Выполняет вызов, эмулируя логику circuit breaker."""
        self._stats["total_calls"] += 1
        if self.state == "open":
            self._stats["rejected_calls"] += 1
            from rag.circuit_breaker import CircuitBreakerOpenException

            raise CircuitBreakerOpenException(
                "MockCircuitBreaker: состояние OPEN", time_until_retry=1.0
            )

        try:
            result = await func(*args, **kwargs)
            self._stats["successful_calls"] += 1
            if self.state != "closed":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception:
            self._stats["failed_calls"] += 1
            self.failure_count += 1
            if self.failure_count >= 5:
                self.state = "open"
            raise

    def get_state(self) -> Dict[str, Any]:
        """Возвращает состояние circuit breaker."""
        return {"state": self.state, "failure_count": self.failure_count}

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает агрегированную статистику вызовов."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Обнуляет статистику без изменения состояния."""
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
        }

    def reset(self) -> None:
        """Полностью сбрасывает состояние и статистику."""
        self.state = "closed"
        self.failure_count = 0
        self.reset_stats()


class MockRemoteEmbedder(EmbedderProtocol):
    """Mock RemoteVMEmbedder c расширенной статистикой."""

    def __init__(
        self,
        embedding_config: Optional[Any] = None,
        parallelism_config: Optional[Any] = None,
        remote_service_config: Optional[Any] = None,
    ) -> None:
        self.embedding_config = embedding_config
        self.parallelism_config = parallelism_config
        self.remote_service_config = remote_service_config
        self.truncate_dim = getattr(embedding_config, "truncate_dim", 1024) if embedding_config else 1024
        self.model_name = getattr(embedding_config, "model_name", "mock-remote") if embedding_config else "mock-remote"
        self.provider_name = getattr(embedding_config, "provider", "mock-remote") if embedding_config else "mock-remote"
        self.retry_policy: MockRetryPolicy = MockRetryPolicy()
        self.circuit_breaker: MockCircuitBreaker = MockCircuitBreaker()
        self._is_warmed_up: bool = False
        self._base_stats: Dict[str, Any] = {
            "total_requests": 0,
            "total_texts": 0,
            "total_time": 0.0,
            "error_count": 0,
            "avg_response_time": 0.0,
        }
        self.stats = self._base_stats

    def embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        deadline_ms: int = 30000,
    ) -> np.ndarray:
        """Синхронная обёртка над асинхронным методом."""
        if not texts:
            return np.zeros((0, self.truncate_dim), dtype=np.float32)

        self._base_stats["total_requests"] += 1
        start_time = time.perf_counter()
        async def runner() -> np.ndarray:
            return await self._async_embed_texts(texts, task=task, deadline_ms=deadline_ms)

        try:
            try:
                result = asyncio.run(runner())
            except RuntimeError as loop_error:
                message = str(loop_error)
                if "asyncio.run() cannot be called" in message:
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(
                        self._async_embed_texts(texts, task=task, deadline_ms=deadline_ms)
                    )
                else:
                    raise

            elapsed = time.perf_counter() - start_time
            self._update_success_metrics(len(texts), elapsed)
            return result
        except EmbeddingException:
            elapsed = time.perf_counter() - start_time
            self._update_failure_metrics(elapsed)
            raise
        except Exception as exc:  # pragma: no cover - защитный слой
            elapsed = time.perf_counter() - start_time
            self._update_failure_metrics(elapsed)
            raise EmbeddingException(str(exc), provider=self.provider_name, model_name=self.model_name) from exc

    async def _async_embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        deadline_ms: int = 30000,
    ) -> np.ndarray:
        """Асинхронная генерация псевдослучайных эмбеддингов."""
        if not texts:
            return np.zeros((0, self.truncate_dim), dtype=np.float32)

        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.standard_normal(self.truncate_dim, dtype=np.float32))

        return np.stack(vectors)

    def _update_success_metrics(self, text_count: int, elapsed: float) -> None:
        self._base_stats["total_texts"] += text_count
        self._base_stats["total_time"] += elapsed
        total_requests = self._base_stats["total_requests"]
        if total_requests:
            self._base_stats["avg_response_time"] = self._base_stats["total_time"] / total_requests

    def _update_failure_metrics(self, elapsed: float) -> None:
        self._base_stats["error_count"] += 1
        self._base_stats["total_time"] += elapsed
        total_requests = self._base_stats["total_requests"]
        if total_requests:
            self._base_stats["avg_response_time"] = self._base_stats["total_time"] / total_requests

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику в версионированном формате."""
        base_stats = self._base_stats.copy()
        retry_stats = self.retry_policy.get_stats()
        cb_state = self.circuit_breaker.get_state()
        cb_stats = self.circuit_breaker.get_stats()

        return {
            "schema_version": 1,
            "requests": {
                "total": base_stats.get("total_requests", 0),
                "errors": base_stats.get("error_count", 0),
                "texts": base_stats.get("total_texts", 0),
            },
            "retry": {
                "total_retries": retry_stats.get("total_retries", 0),
                "attempts": retry_stats.get("total_executions", 0),
            },
            "latency": {
                "avg_ms": base_stats.get("avg_response_time", 0.0) * 1000,
                "total_time": base_stats.get("total_time", 0.0),
            },
            "cb": {
                "state": cb_state.get("state", "unknown"),
                "failure_count": cb_state.get("failure_count", 0),
            },
            "total_requests": base_stats.get("total_requests", 0),
            "total_texts": base_stats.get("total_texts", 0),
            "error_count": base_stats.get("error_count", 0),
            "retry_count": retry_stats.get("total_retries", 0),
            "avg_response_time": base_stats.get("avg_response_time", 0.0),
            "is_warmed_up": self._is_warmed_up,
            "provider": self.provider_name,
            "model_name": self.model_name,
            "retry_policy_stats": retry_stats,
            "circuit_breaker_stats": cb_stats,
        }

    def reset_stats(self) -> None:
        """Сбрасывает статистику и состояния вспомогательных компонентов."""
        self._base_stats = {
            "total_requests": 0,
            "total_texts": 0,
            "total_time": 0.0,
            "error_count": 0,
            "avg_response_time": 0.0,
        }
        self.stats = self._base_stats
        self.retry_policy.reset_stats()
        self.circuit_breaker.reset()

    def warmup(self) -> bool:
        """Прогревает mock-компонент (no-op)."""
        self._is_warmed_up = True
        return True

    def check_health(self) -> Dict[str, Any]:
        """Возвращает статус готовности mock-компонента."""
        status = "ok" if self._is_warmed_up else "cold"
        return {
            "status": status,
            "error": None,
            "components": {
                "embedder": {
                    "model_name": self.model_name,
                    "provider": self.provider_name,
                    "embedding_dim": self.truncate_dim,
                }
            },
        }
