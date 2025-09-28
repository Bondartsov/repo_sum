"""Вспомогательные функции для проверки сетевой доступности во время тестов."""

from functools import lru_cache
import socket
from typing import Optional


@lru_cache(maxsize=None)
def is_network_available(host: str = "huggingface.co", port: int = 443, timeout: float = 5.0) -> bool:
    """Проверяет доступность сети до указанного хоста и порта."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_network_or_skip(host: str, port: int, reason: Optional[str] = None) -> bool:
    """Утилита для лаконичных skipif-условий."""
    available = is_network_available(host=host, port=port)
    if not available:
        import pytest

        pytest.skip(reason or f"Нет сети до {host}:{port}")
    return True

