"""Асинхронный HTTP-транспорт для RemoteVMEmbedder.

Автор: AI Assistant (gpt-5-codex)
Дата: 03 октября 2025
"""
from __future__ import annotations

import aiohttp
from typing import Any, Dict

from .embedder_protocol import TransportClientProtocol
from .event_loop_manager import get_shared_http_session


class AiohttpTransportClient(TransportClientProtocol):
    """Реализация транспорта поверх aiohttp.

    Класс оборачивает общую HTTP-сессию проекта и предоставляет
    метод post_json для выполнения запросов с единым форматированием ошибок.
    """

    async def post_json(self, url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Отправляет POST-запрос с JSON-телом и возвращает результат."""
        session = await get_shared_http_session()
        timeout_ctx = aiohttp.ClientTimeout(total=timeout)
        async with session.post(url, json=payload, timeout=timeout_ctx) as response:
            if response.status == 200:
                return await response.json()

            error_text = await response.text()
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=f"HTTP {response.status}: {error_text}",
                headers=response.headers,
            )
