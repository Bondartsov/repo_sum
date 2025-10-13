"""Асинхронный HTTP-транспорт для RemoteVMEmbedder и других клиентов.

Автор: AI Assistant (gpt-5-codex)
Дата: 03 октября 2025
"""
from __future__ import annotations

import uuid
import aiohttp
from typing import Any, Dict, Optional, Callable

from .embedder_protocol import TransportClientProtocol
from .event_loop_manager import get_shared_http_session


class AiohttpTransportClient(TransportClientProtocol):
    """Реализация транспорта поверх aiohttp.

    - Единая точка доступа к shared HTTP session
    - Централизованная работа с заголовками, включая X-Trace-Id
    - Унифицированные методы get_json и post_json
    """

    def __init__(
        self,
        default_headers: Optional[Dict[str, str]] = None,
        trace_id_provider: Optional[Callable[[], str]] = None,
        session_provider: Callable = get_shared_http_session,
    ) -> None:
        """Инициализация транспорта.

        Args:
            default_headers: Заголовки по умолчанию (сливаются с per-call headers)
            trace_id_provider: Провайдер trace-id; если не задан — используется uuid.uuid4().hex
            session_provider: Провайдер aiohttp.ClientSession (по умолчанию get_shared_http_session)
        """
        self._default_headers: Dict[str, str] = dict(default_headers) if default_headers else {}
        self._trace_id_provider: Callable[[], str] = trace_id_provider or (lambda: uuid.uuid4().hex)
        self._session_provider: Callable = session_provider

    def _merge_headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Склеивает default_headers и per-call headers с приоритетом per-call и добавляет X-Trace-Id."""
        merged: Dict[str, str] = {}
        if self._default_headers:
            merged.update(self._default_headers)
        if headers:
            merged.update(headers)

        # Автоматическое добавление X-Trace-Id, если отсутствует
        trace_header = "X-Trace-Id"
        if not merged.get(trace_header):
            try:
                merged[trace_header] = self._trace_id_provider() if self._trace_id_provider else uuid.uuid4().hex
            except Exception:
                merged[trace_header] = uuid.uuid4().hex
        return merged

    async def get_json(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> Any:
        """Выполняет GET-запрос и возвращает JSON.

        Исключения:
            Поднимает aiohttp.ClientResponseError при HTTP-ошибках (status != 200)
        """
        session = await self._session_provider()
        final_headers = self._merge_headers(headers)

        timeout_ctx: Optional[aiohttp.ClientTimeout]
        if timeout is None:
            timeout_ctx = None
        elif isinstance(timeout, (int, float)):
            timeout_ctx = aiohttp.ClientTimeout(total=float(timeout))
        else:
            timeout_ctx = timeout

        async with session.get(endpoint, params=params, headers=final_headers, timeout=timeout_ctx) as response:
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

    async def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: float,
        headers: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """Отправляет POST-запрос с JSON-телом.

        Требования Шага 3.1:
        - Сливать default_headers + headers (приоритет у headers)
        - Автоматически прокидывать X-Trace-Id (если нет — сгенерировать через trace_id_provider/uuid4)
        - Сохранить семантику исключений: поднимать aiohttp.ClientResponseError при HTTP ошибках
        - Не менять внешнее поведение сигнатуры метода
        """
        session = await self._session_provider()
        final_headers = self._merge_headers(headers)

        # Поддерживаем как числовой timeout, так и переданный ClientTimeout (для per-request таймаутов)
        if isinstance(timeout, aiohttp.ClientTimeout):
            timeout_ctx = timeout
        else:
            timeout_ctx = aiohttp.ClientTimeout(total=float(timeout)) if timeout is not None else None

        async with session.post(url, json=payload, timeout=timeout_ctx, headers=final_headers) as response:
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
