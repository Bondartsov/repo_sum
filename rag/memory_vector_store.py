from types import SimpleNamespace
from typing import Any, Dict, List, Optional

class InMemoryVectorStore:
    """Легковесное in-memory хранилище для offline-режима."""

    def __init__(self, vector_config: Optional[Any] = None, remote_config: Optional[Any] = None):
        self.vector_config = vector_config
        self.remote_config = remote_config
        self._documents: List[Dict[str, Any]] = []
        self._stats = {
            'queries': 0,
            'documents_indexed': 0,
        }

    # Collection management -------------------------------------------------
    def initialize_collection(self, *args, **kwargs) -> bool:
        return True

    def ensure_collection(self, *args, **kwargs) -> bool:
        return True

    # Health ----------------------------------------------------------------
    def check_health(self) -> Dict[str, Any]:
        return {
            'status': 'ok',
            'components': {
                'vector_store': {
                    'collection_status': 'exists'
                }
            }
        }

    health_check = check_health

    # Indexing --------------------------------------------------------------
    def index_documents(self, documents: List[Dict[str, Any]], *args, **kwargs) -> Dict[str, Any]:
        self._documents.extend(documents)
        self._stats['documents_indexed'] = len(self._documents)
        return len(documents)

    # Searching -------------------------------------------------------------
    def search(
        self,
        query_vector: Any,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = False,
        sparse_vector: Optional[Dict[int, float]] = None,
        *args,
        **kwargs,
    ) -> List[SimpleNamespace]:
        self._stats['queries'] += 1
        language = (filters or {}).get('language', 'python')
        results: List[SimpleNamespace] = []
        base_documents = self._documents or [
            {
                'content': 'Mock content placeholder',
                'file_path': 'mock/path.py',
                'file_name': 'path.py',
                'chunk_name': 'mock_chunk',
                'chunk_type': 'function',
                'language': language,
            }
        ]
        for idx in range(min(top_k, len(base_documents))):
            payload = base_documents[idx % len(base_documents)].copy()
            payload.setdefault('language', language)
            payload.setdefault('start_line', 1)
            payload.setdefault('end_line', 10)
            results.append(SimpleNamespace(id=f'mock_{idx}', score=0.9 - idx * 0.05, payload=payload))
        return results

    # Stats ----------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats['queries'] = 0

    # Cleanup --------------------------------------------------------------
    async def close(self) -> None:
        return None
