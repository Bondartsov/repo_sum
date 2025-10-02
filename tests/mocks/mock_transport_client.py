"""Mock-реализация транспортного клиента для тестов.

Автор: AI Assistant (gpt-5-codex)
Дата: 03 октября 2025
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from rag.embedder_protocol import TransportClientProtocol


class MockTransportClient(TransportClientProtocol):
    """Имитация HTTP-транспорта с управляемыми сценариями."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.calls_history: List[Tuple[str, Dict[str, Any], float]] = []
        self.should_fail: bool = False
        self.fail_with: Optional[Exception] = None
        self.response_payload: Optional[Dict[str, Any]] = None
        self.latency: float = 0.0

    async def post_json(self, url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Сохраняет параметры вызова и возвращает предопределённый ответ."""
        self.call_count += 1
        self.calls_history.append((url, payload, timeout))

        if self.latency > 0:
            await asyncio.sleep(self.latency)

        if self.should_fail:
            if self.fail_with:
                raise self.fail_with
            raise RuntimeError("MockTransportClient: запрошен сценарий ошибки")

        if self.response_payload is not None:
            return self.response_payload

        texts = payload.get("texts", [])
        dim = payload.get("truncate_dim", 1024)
        embeddings = [[0.0 for _ in range(dim)] for _ in texts]
        return {"embeddings": embeddings}

    def reset(self) -> None:
        """Сбрасывает накопленные данные о вызовах."""
        self.call_count = 0
        self.calls_history.clear()
        self.should_fail = False
        self.fail_with = None
        self.response_payload = None
        self.latency = 0.0
