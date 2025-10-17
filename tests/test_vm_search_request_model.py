import pytest
from vm_rag_service import SearchRequest

def test_vm_search_request_accepts_protocol_and_sparse():
    sr = SearchRequest(
        protocol="hybrid",
        query="Q",
        top_k=10,
        use_hybrid=True,
        sparse_vector={123: 0.5, 456: 0.5}
    )
    assert getattr(sr, "protocol", None) == "hybrid"
    assert sr.top_k == 10
    assert isinstance(sr.sparse_vector, dict)

def test_search_request_protocol_variants_init():
    # dense protocol with provided dense_vector
    sr_dense = SearchRequest(
        protocol="dense",
        dense_vector=[0.0] * 1024,
        top_k=5,
        use_hybrid=False
    )
    assert sr_dense.protocol == "dense"
    assert isinstance(sr_dense.dense_vector, list)
    assert len(sr_dense.dense_vector) == 1024

    # sparse protocol with provided sparse_vector
    sr_sparse = SearchRequest(
        protocol="sparse",
        query=None,
        top_k=3,
        use_hybrid=False,
        sparse_vector={7: 0.5}
    )
    assert sr_sparse.protocol == "sparse"
    assert isinstance(sr_sparse.sparse_vector, dict)
    assert 7 in sr_sparse.sparse_vector