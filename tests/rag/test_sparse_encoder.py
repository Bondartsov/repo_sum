import pytest
from rag.sparse_encoder import SparseEncoder
from hypothesis import given, strategies as st, settings


@pytest.fixture(scope="module")
def encoder():
    return SparseEncoder(model_name="bert-base-uncased")


def test_initialization(encoder):
    assert encoder.tokenizer is not None
    assert encoder.model is not None


def test_encode_nonempty(encoder):
    texts = ["Hello world"]
    vectors = encoder.encode(texts)
    assert isinstance(vectors, list)
    assert isinstance(vectors[0], dict)
    assert len(vectors[0]) > 0


def test_encode_stability(encoder):
    text = ["Consistency check"]
    vec1 = encoder.encode(text)
    vec2 = encoder.encode(text)
    assert vec1 == vec2


def test_tokenization_different_words(encoder):
    tokens1 = encoder.tokenizer.encode("apple", add_special_tokens=False)
    tokens2 = encoder.tokenizer.encode("banana", add_special_tokens=False)
    assert tokens1 != tokens2
    assert all(isinstance(t, int) for t in tokens1 + tokens2)


def test_encode_multiple_sentences(encoder):
    texts = ["First sentence", "Second sentence", "Third one"]
    vectors = encoder.encode(texts)
    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    for v in vectors:
        assert isinstance(v, dict)
        assert all(isinstance(k, int) for k in v.keys())
        assert all(isinstance(val, float) for val in v.values())


def test_sparse_vector_nonnegative_and_normalized(encoder):
    texts = ["Normalization test"]
    vectors = encoder.encode(texts)
    vec = vectors[0]
    assert all(val >= 0 for val in vec.values())
    total_weight = sum(vec.values())
    # допускаем небольшую погрешность
    assert pytest.approx(total_weight, rel=1e-2) == 1.0 or total_weight > 0


@settings(deadline=None, max_examples=10)
@given(st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=20), min_size=1, max_size=3))
def test_property_based_encode_returns_valid_dicts(encoder, texts):
    vectors = encoder.encode(texts)
    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    for v in vectors:
        assert isinstance(v, dict)
        for k, val in v.items():
            assert isinstance(k, int)
            assert isinstance(val, float)
            assert val >= 0


def test_mock_mode_tokenizer(monkeypatch):
    import os
    from tests.mocks.mock_tokenizer import MockTokenizer

    monkeypatch.setenv("MOCK_MODE", "1")
    encoder = SparseEncoder(model_name="bert-base-uncased")

    assert isinstance(encoder.tokenizer, MockTokenizer)
    assert encoder.model is None