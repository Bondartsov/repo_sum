# tests/rag/test_search_service_min_score_zero.py
import os
import pytest

from config import get_config
from rag.search_service import SearchService


@pytest.fixture
def search_service():
    # Создаём сервис поиска. На этапе __init__ сетевые вызовы не выполняются.
    cfg = get_config(require_api_key=False)
    return SearchService(cfg, silent_mode=True)


def test_cache_key_handles_min_score_zero(search_service: SearchService):
    """
    Проверяет, что min_score=0.0 корректно участвует в генерации ключа кэша
    (не теряется как falsy значение) и отличается от случая None.
    """
    key_zero = search_service._generate_cache_key(
        query="test", top_k=5,
        language_filter=None, chunk_type_filter=None,
        min_score=0.0, file_path_filter=None
    )
    key_none = search_service._generate_cache_key(
        query="test", top_k=5,
        language_filter=None, chunk_type_filter=None,
        min_score=None, file_path_filter=None
    )
    assert key_zero != key_none, "Ключ кэша для min_score=0.0 не должен совпадать с ключом для None"


def test_process_search_results_respects_zero_threshold(search_service: SearchService):
    """
    Проверяет, что фильтрация по min_score=0.0 пропускает результаты со score=0.0,
    а при пороге 0.1 эти результаты отфильтровываются.
    """
    raw_results = [
        {
            "id": "1",
            "score": 0.0,
            "payload": {
                "file_path": "a.py",
                "file_name": "a.py",
                "chunk_name": "func_a",
                "chunk_type": "function",
                "language": "python",
                "start_line": 1,
                "end_line": 5,
                "content": "def a(): pass",
            },
        },
        {
            "id": "2",
            "score": 0.1,
            "payload": {
                "file_path": "b.py",
                "file_name": "b.py",
                "chunk_name": "func_b",
                "chunk_type": "function",
                "language": "python",
                "start_line": 10,
                "end_line": 20,
                "content": "def b(): pass",
            },
        },
    ]

    # При пороге 0.0 оба результата должны пройти
    res_zero = search_service._process_search_results(raw_results, min_score=0.0)
    assert len(res_zero) == 2, "При min_score=0.0 должны проходить результаты с score=0.0"

    # При пороге 0.1 должен остаться только второй
    res_point_one = search_service._process_search_results(raw_results, min_score=0.1)
    assert len(res_point_one) == 1 and res_point_one[0].chunk_id == "2", "При min_score=0.1 результат со score=0.0 должен быть отфильтрован"
