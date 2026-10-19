import logging
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.chunking.chunk_schema import ChunkRecord
from src.config.settings import settings

logger = logging.getLogger(__name__)


def generate_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUID from chunk_id for Qdrant compatibility."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantStore:
    """
    Manages vector indexing and retrieval in Qdrant.
    Supports local embedded on-disk storage (zero-docker) or remote Qdrant service.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        client: Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.dimension = dimension or settings.VECTOR_DIMENSION

        if client is not None:
            self.client = client
        elif settings.QDRANT_URL:
            logger.info("Connecting to remote Qdrant at %s", settings.QDRANT_URL)
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )
        else:
            storage_path = str(settings.QDRANT_STORAGE_DIR)
            logger.info("Using local embedded on-disk Qdrant at %s", storage_path)
            self.client = QdrantClient(path=storage_path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                logger.info(
                    "Creating Qdrant collection '%s' (dim=%d, distance=Cosine)...",
                    self.collection_name,
                    self.dimension,
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.error("Failed ensuring Qdrant collection: %s", str(e))

    def upsert_chunks(
        self, chunks: List[ChunkRecord], embeddings: List[List[float]]
    ) -> int:
        """Upsert chunk records with embeddings and payloads into Qdrant."""
        if not chunks or not embeddings:
            return 0

        points = []
        for chunk, emb in zip(chunks, embeddings):
            point_id = generate_point_id(chunk.chunk_id)
            payload = {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "title": chunk.title,
                "section": chunk.section,
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "source": chunk.source,
                "token_count": chunk.token_count,
                "text": chunk.text,
                **chunk.metadata,
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload,
                )
            )

        # Batch upsert
        batch_size = 64
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_upserted += len(batch)
            if total_upserted % 128 == 0 or total_upserted == len(points):
                logger.info(
                    "Upserted %d / %d points to Qdrant collection '%s' (%.1f%%)...",
                    total_upserted,
                    len(points),
                    self.collection_name,
                    (total_upserted / len(points)) * 100,
                )

        logger.info("Upserted %d points to Qdrant collection '%s'", total_upserted, self.collection_name)
        return total_upserted

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search Qdrant for nearest neighbors by cosine similarity."""
        query_filter = None
        if filter_section:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="section",
                        match=qmodels.MatchValue(value=filter_section),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        hits = []
        for r in results:
            payload = r.payload or {}
            hits.append({
                "chunk_id": payload.get("chunk_id", str(r.id)),
                "paper_id": payload.get("paper_id", ""),
                "title": payload.get("title", ""),
                "section": payload.get("section", ""),
                "page": payload.get("page", 1),
                "chunk_index": payload.get("chunk_index", 0),
                "score": float(r.score),
                "text": payload.get("text", ""),
                "source": payload.get("source", "arxiv"),
            })
        return hits

    def count(self) -> int:
        """Count total vectors indexed in collection."""
        try:
            res = self.client.count(collection_name=self.collection_name)
            return res.count
        except Exception:
            return 0

    def close(self) -> None:
        """Close client connection cleanly."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass
