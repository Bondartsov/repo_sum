# tests/rag/test_search_stats_contains_threshold.py
import pytest

from config import get_config
from rag.search_service import SearchService


@pytest.fixture
def search_service():
    # Создаём сервис поиска в тихом режиме (без вывода в консоль)
    cfg = get_config(require_api_key=False)
    return SearchService(cfg, silent_mode=True)


def test_get_search_stats_contains_score_threshold(search_service: SearchService):
    """
    Проверяет, что get_search_stats() содержит поле score_threshold
    и оно синхронизировано с конфигурацией.
    """
    cfg = get_config(require_api_key=False)
    stats = search_service.get_search_stats()

    assert "score_threshold" in stats, "В статистике поиска должен присутствовать ключ 'score_threshold'"
    assert stats["score_threshold"] == cfg.rag.query_engine.score_threshold, (
        "Значение score_threshold в статистике должно соответствовать конфигурации"
    )
