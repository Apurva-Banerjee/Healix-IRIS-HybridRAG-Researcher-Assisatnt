import logging
from typing import List, Optional, Union
from src.chunking.chunk_schema import ChunkRecord
from src.config.settings import settings
from src.embeddings.model import (
    BaseEmbeddingModel,
    FastDenseEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)

logger = logging.getLogger(__name__)


class Embedder:
    """Embedder service that converts chunks and query texts into dense vector representations."""

    def __init__(
        self,
        model: Optional[BaseEmbeddingModel] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        name = model_name or settings.EMBEDDING_MODEL_NAME
        dev = device or settings.EMBEDDING_DEVICE
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        if model is not None:
            self.model = model
        else:
            if name == "fast-dense":
                self.model = FastDenseEmbeddingModel(dimension=settings.VECTOR_DIMENSION)
            else:
                self.model = SentenceTransformerEmbeddingModel(model_name=name, device=dev)

    @property
    def dimension(self) -> int:
        return self.model.dimension

    def embed_chunks(self, chunks: List[ChunkRecord]) -> List[List[float]]:
        """Extract text from chunks and produce dense vector embeddings."""
        if not chunks:
            return []

        # Optional prefixing with title and section for richer semantic context
        texts = [
            f"Title: {c.title}\nSection: {c.section}\nContent: {c.text}"
            for c in chunks
        ]

        logger.info("Generating embeddings for %d chunks...", len(texts))
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            emb_batch = self.model.embed_documents(batch)
            all_embeddings.extend(emb_batch)
            if len(all_embeddings) % 128 == 0 or len(all_embeddings) == len(texts):
                logger.info(
                    "Embedded %d / %d chunks (%.1f%%)...",
                    len(all_embeddings),
                    len(texts),
                    (len(all_embeddings) / len(texts)) * 100,
                )

        logger.info("Successfully generated %d embeddings", len(all_embeddings))
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single query."""
        return self.model.embed_query(query)
