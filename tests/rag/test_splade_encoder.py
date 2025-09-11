import pytest
import torch
from rag.sparse_encoder import SpladeModelWrapper, SparseEncoder

@pytest.mark.unit
def test_splade_model_wrapper_forward_shapes():
    model = SpladeModelWrapper()
    texts = ["def foo(): return 42", "class Bar: pass"]
    outputs = model.forward(texts)
    assert isinstance(outputs, list)
    assert all(isinstance(vec, dict) for vec in outputs)
    assert all(all(isinstance(k, int) and isinstance(v, float) for k, v in vec.items()) for vec in outputs)

@pytest.mark.unit
def test_sparse_encoder_with_splade_method():
    encoder = SparseEncoder(method="SPLADE")
    texts = ["print('hello')", "def add(a,b): return a+b"]
    outputs = encoder.encode(texts)
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    assert all(isinstance(vec, dict) for vec in outputs)
    assert all(all(isinstance(k, int) and isinstance(v, float) for k, v in vec.items()) for vec in outputs)
