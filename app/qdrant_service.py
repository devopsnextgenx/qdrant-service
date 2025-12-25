import logging
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.models import Prefetch, Fusion
from .config import CONFIG

logger = logging.getLogger(__name__)

QDRANT_URL = CONFIG.get("qdrant", {}).get("url", "http://localhost:6333")
QDRANT_TIMEOUT = CONFIG.get("qdrant", {}).get("timeout", 30)

class QdrantService:
    def __init__(self, url: str = QDRANT_URL, timeout: int = QDRANT_TIMEOUT):
        # qdrant-client accepts url param
        self.client = QdrantClient(url=url, timeout=timeout)
        logger.info("Connected to Qdrant at %s", url)

    def ensure_collection(self, collection_name: str, vector_size: int):
        try:
            # create collection if missing; ignore if exists
            self.client.get_collection(collection_name=collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE),
            )

    def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]):
        # points: [{'id':..., 'vector':..., 'payload': {...}}, ...]
        point_structs = [rest_models.PointStruct(id=p['id'], vector=p['vector'], payload=p['payload']) for p in points]
        self.client.upsert(collection_name=collection_name, points=point_structs)


    def search(self, collection_name: str, vector: List[float], limit: int = 10):
        hits = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
        )

        return [
            {
                "id": h.id,
                "score": h.score,
                "payload": h.payload,
            }
            for h in hits.points
        ]

    def hybrid_search(
        self, 
        collection_name: str, 
        vector: List[float],
        text_query: Optional[str] = None,
        limit: int = 10
    ):
        """
        Hybrid search with automatic fallback to dense-only search
        """
        # Check if collection has sparse vectors configured
        try:
            collection_info = self.client.get_collection(collection_name)
            has_sparse = hasattr(collection_info.config, 'sparse_vectors_config') and \
                        collection_info.config.sparse_vectors_config is not None
        except:
            has_sparse = False
        
        # If we have sparse vectors and a text query, do hybrid search
        if has_sparse and text_query:
            try:
                hits = self.client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        Prefetch(query=vector, using="dense", limit=limit * 2),
                        Prefetch(query=text_query, using="sparse", limit=limit * 2)
                    ],
                    query=Fusion.RRF,
                    limit=limit,
                )
            except Exception as e:
                print(f"Hybrid search failed: {e}, falling back to dense search")
                # Fallback to dense search
                hits = self.client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=limit,
                )
        else:
            # No sparse vectors available or no text query - use dense only
            hits = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
            )

        return [
            {
                "id": h.id,
                "score": h.score,
                "payload": h.payload,
            }
            for h in hits.points
        ]


    def searchx(self, collection_name: str, vector: List[float], limit: int = 10):
        hits = self.client.query_points(
            collection_name=collection_name,
            query={"default": vector},
            limit=limit,
        )

        results = []
        for h in hits.points:
            text = (h.payload or {}).get("text", "")
            results.append({
                "id": h.id,
                "score": h.score,
                "snippet": make_snippet(text),
                "payload": h.payload,
            })

        return results

    def health(self) -> dict:
        """Basic Qdrant connectivity check and collection list"""
        try:
            cols = self.client.get_collections()
            names = [c.name for c in cols.collections]
            return {"ok": True, "collections": names}
        except Exception as e:
            return {"ok": False, "error": str(e)}

qdrant = QdrantService()
