from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class MockVectorStore:
    """Простейший in-memory vector store для тестов без сети."""

    def __init__(self, *args, **kwargs):
        vector_config = kwargs.get('vector_config') if kwargs else (args[0] if args else None)
        self.vector_config = vector_config
        self.config = vector_config or SimpleNamespace(
            host='localhost',
            port=6333,
            collection_name='mock_collection',
            distance='cosine'
        )
        self.documents: List[Dict[str, Any]] = []
        self._query_count = 0

    # Collection helpers -------------------------------------------------
    async def initialize_collection(self, *args, **kwargs) -> bool:
        return True

    async def ensure_collection(self, *args, **kwargs) -> bool:
        return True

    # Health -------------------------------------------------------------
    def check_health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "components": {
                "vector_store": {
                    "collection_status": "exists"
                }
            }
        }

    health_check = check_health

    # Indexing -----------------------------------------------------------
    async def index_documents(self, documents: List[Dict[str, Any]], *args, **kwargs) -> int:
        self.documents.extend(documents)
        return len(documents)

    # Searching ----------------------------------------------------------
    async def search(
        self,
        query_vector: Any,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = False,
        sparse_vector: Optional[Dict[int, float]] = None,
        *args,
        **kwargs,
    ) -> List[SimpleNamespace]:
        self._query_count += 1
        language = (filters or {}).get('language', 'python')
        source_documents = self.documents or [
            {
                'content': 'Mock content',
                'file_path': 'mock/path.py',
                'file_name': 'path.py',
                'chunk_name': 'mock_chunk',
                'chunk_type': 'function',
                'language': language,
            }
        ]

        results: List[SimpleNamespace] = []
        for idx in range(min(top_k, len(source_documents))):
            payload = source_documents[idx % len(source_documents)].copy()
            payload.setdefault('language', language)
            payload.setdefault('start_line', 1)
            payload.setdefault('end_line', 10)
            results.append(SimpleNamespace(id=f'mock_{idx}', score=0.9 - idx * 0.05, payload=payload))
        return results

    # Statistics ---------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            'queries': self._query_count,
            'documents_indexed': len(self.documents),
            'total_points': len(self.documents)
        }

    def reset_stats(self) -> None:
        self._query_count = 0

    # Cleanup ------------------------------------------------------------
    async def close(self) -> None:
        return None
