import logging
from typing import Any, Dict, List, Optional
from src.config.settings import settings
from src.embeddings.embedder import Embedder
from src.retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Semantic search engine querying Qdrant using dense embeddings."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        qdrant_store: Optional[QdrantStore] = None,
    ):
        self.embedder = embedder or Embedder()
        dim = self.embedder.dimension
        self.qdrant_store = qdrant_store or QdrantStore(dimension=dim)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Encode query and perform nearest neighbor search in Qdrant."""
        logger.info("Executing semantic search for query: '%s' (top_k=%d)", query, top_k)
        query_vector = self.embedder.embed_query(query)

        hits = self.qdrant_store.search_vectors(
            query_vector=query_vector,
            top_k=top_k,
            filter_section=filter_section,
        )
        return hits
