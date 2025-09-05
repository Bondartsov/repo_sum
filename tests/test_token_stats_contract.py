# tests/test_token_stats_contract.py
import os
import pytest

pytest.importorskip("openai")
pytest.importorskip("tiktoken")

from openai_integration import OpenAIManager


def test_token_stats_contract(monkeypatch):
    """
    Проверяет, что OpenAIManager.get_token_usage_stats() возвращает
    как новые ключи (used_today/requests_today/average_per_request),
    так и старые total_* для обратной совместимости.
    """
    # Имитируем наличие API ключа, чтобы инициализация OpenAIManager прошла
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-token")

    mgr = OpenAIManager()
    stats = mgr.get_token_usage_stats()

    # Новые ключи
    assert "used_today" in stats
    assert "requests_today" in stats
    assert "average_per_request" in stats

    # Старые ключи для обратной совместимости
    assert "total_requests" in stats
    assert "total_tokens" in stats
    assert "average_tokens_per_request" in stats

    # Согласованность значений (в заглушке равны 0)
    assert stats["used_today"] == stats["total_tokens"]
    assert stats["requests_today"] == stats["total_requests"]
    assert stats["average_per_request"] == stats["average_tokens_per_request"]
