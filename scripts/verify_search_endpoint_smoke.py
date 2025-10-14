import asyncio
from starlette.requests import Request
from fastapi import HTTPException
from vm_rag_service import search_documents, SearchRequest, services

class DummyVectorStore:
    async def search(self, query_vector, top_k, filters, use_hybrid, sparse_vector=None):
        return []
    async def health_check(self):
        return {"status": "connected"}

class DummySearchService:
    async def search(self, query, top_k, filters, use_hybrid, task):
        return []

async def run_case(name, req_obj, path):
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": []
    }
    http_request = Request(scope)
    try:
        resp = await search_documents(req_obj, http_request)
        print(f"{name}: OK 200 results={len(resp.results)}")
    except HTTPException as e:
        print(f"{name}: HTTP {e.status_code} {e.detail}")
    except Exception as e:
        print(f"{name}: EXC {type(e).__name__} {e}")

async def main():
    # Prepare services
    services.clear()
    services["vector_store"] = DummyVectorStore()
    services["search_service"] = DummySearchService()

    dense = [0.0] * 1024

    # 1) Vector OK
    req1 = SearchRequest(dense_vector=dense, top_k=3, use_hybrid=False, filters={}, task="retrieval.query")
    await run_case("vector_ok", req1, "/v1/search_v2")

    # 2) Empty text query &#8594; 422
    req2 = SearchRequest(query="")
    await run_case("text_empty", req2, "/search")

    # 3) v2 missing vectors &#8594; 422
    req3 = SearchRequest()
    await run_case("v2_missing_vectors", req3, "/v1/search_v2")

    # 4) Invalid top_k &#8594; 422
    req4 = SearchRequest(dense_vector=dense, top_k=0)
    await run_case("invalid_topk", req4, "/v1/search_v2")

if __name__ == "__main__":
    asyncio.run(main())