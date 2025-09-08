import os
import sys
import numpy as np
import pytest


def test_patch_covers_search_and_query():
    """
    Проверяет, что conftest.py подпачивает CPUEmbedder в сервисах,
    где он импортируется напрямую (SearchService и QueryEngine).
    Тест выполняется только если mock-режим активирован.
    """
    from tests.mocks.mock_cpu_embedder import should_use_mock_embedder, MockCPUEmbedder

    if not should_use_mock_embedder():
        pytest.skip("Mock embedder не активирован в этом окружении")

    import rag.search_service as ss
    import rag.query_engine as qe

    assert ss.CPUEmbedder is MockCPUEmbedder, "rag.search_service.CPUEmbedder должен быть подпачен на MockCPUEmbedder"
    assert qe.CPUEmbedder is MockCPUEmbedder, "rag.query_engine.CPUEmbedder должен быть подпачен на MockCPUEmbedder"


def test_cpu_embedder_offline_skips_model_init(monkeypatch):
    """
    Проверяет, что в offline режиме CPUEmbedder не инициализирует реальные провайдеры
    (никаких сетевых вызовов), а возвращает оффлайн-эмбеддинги корректной размерности.
    """
    import rag.embedder as r_embedder

    # Если вдруг инициализация реальных провайдеров будет вызвана — тест должен упасть
    class Raiser:
        def __init__(self, *args, **kwargs):
            raise AssertionError("НЕЛЬЗЯ инициализировать реальные провайдеры в offline режиме")

    # Подменяем потенциальные классы-провайдеры на Raiser
    monkeypatch.setattr(r_embedder, "TextEmbedding", Raiser, raising=False)
    monkeypatch.setattr(r_embedder, "SentenceTransformer", Raiser, raising=False)

    # Включаем offline режим через ENV (распознаётся в _is_offline_mode)
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from config import EmbeddingConfig, ParallelismConfig

    emb_cfg = EmbeddingConfig(
        provider="fastembed",
        model_name="BAAI/bge-small-en-v1.5",
        truncate_dim=384,
        batch_size_min=2,
        batch_size_max=8,
        normalize_embeddings=True,
        warmup_enabled=True,
    )
    par_cfg = ParallelismConfig(torch_num_threads=1, omp_num_threads=1, mkl_num_threads=1)

    embedder = r_embedder.CPUEmbedder(emb_cfg, par_cfg)

    assert getattr(embedder, "_offline_mode", False) is True, "Ожидается offline режим"
    assert embedder.provider_name == "offline", "В offline режиме provider_name должен быть 'offline'"

    vectors = embedder.embed_texts(["hello", "world"])
    assert vectors.shape == (2, emb_cfg.truncate_dim or 384)
    assert np.allclose(vectors, 0.0), "Ожидаются нулевые оффлайн-эмбеддинги без сети"
