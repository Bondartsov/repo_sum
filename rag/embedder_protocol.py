"""Описание протоколов для компонентов эмбеддинга.

Документ формализует публичный контракт для embedder-реализаций
и связанных подсистем (retry политика, circuit breaker, транспорт).

Автор: AI Assistant (gpt-5-codex)
Дата: 03 октября 2025
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

try:
    # NumPy доступен в проекте, используем типизированные массивы
    from numpy.typing import NDArray  # type: ignore
    import numpy as np

    ArrayLike = Union[NDArray[np.float32], Sequence[Sequence[float]]]
except ImportError:  # pragma: no cover - fallback для теоретического случая
    ArrayLike = Sequence[Sequence[float]]


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Контракт для всех реализаций эмбеддеров.

    Каждый embedder обязан предоставлять публичные методы, описанные ниже.
    Контракт сознательно не включает приватные детали реализации, чтобы
    тесты и клиенты полагались только на стабильный API.
    """

    def embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        deadline_ms: int = 30000,
    ) -> ArrayLike:
        """Выполняет синхронное получение эмбеддингов для набора текстов."""

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает наблюдаемую статистику с ключом ``schema_version``."""

    def reset_stats(self) -> None:
        """Сбрасывает накопленную статистику embedder-компонента."""

    def warmup(self) -> Optional[bool]:
        """Запускает прогрев или проверку готовности компонента."""

    def check_health(self) -> Dict[str, Any]:
        """Возвращает агрегированное состояние сервиса/компонента."""


@runtime_checkable
class RetryPolicyProtocol(Protocol):
    """Минимальный контракт для retry-политики."""

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику retry-политики."""

    def reset_stats(self) -> None:
        """Сбрасывает статистику retry-политики."""


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Контракт для circuit breaker компонентов."""

    def get_state(self) -> Dict[str, Any]:
        """Возвращает агрегированное состояние circuit breaker."""

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает расширенную статистику circuit breaker."""

    def reset_stats(self) -> None:
        """Обнуляет метрики circuit breaker."""

    def reset(self) -> None:
        """Полностью возвращает circuit breaker в изначальное состояние."""


@runtime_checkable
class TransportClientProtocol(Protocol):
    """Контракт для асинхронного HTTP-транспорта.

    Выделение транспорта в отдельный протокол позволяет
    инжектировать mock/spy реализации без вмешательства в приватные методы
    embedder-реализаций. Это устраняет необходимость monkeypatch/patch
    в тестах и делает контракт прозрачным.
    """

    async def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Выполняет POST-запрос и возвращает JSON-ответ сервера."""
