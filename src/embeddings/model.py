import logging
import math
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbeddingModel:
    """Base interface for embedding models."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> List[float]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class FastDenseEmbeddingModel(BaseEmbeddingModel):
    """
    Lightweight, fast CPU embedding model based on character/subword n-grams and hashing.
    Used for instant fallback, testing, or low-resource CPU environments without network downloads.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_embed(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        for w in words:
            # Word hash
            h = hash(w) % self._dim
            vec[h] += 1.0
            # Subword 3-grams
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    sub = w[i:i+3]
                    h_sub = hash(sub) % self._dim
                    vec[h_sub] += 0.5

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._hash_embed(query)


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Production embedding model using HuggingFace / BAAI models (e.g., BAAI/bge-m3 or bge-small-en-v1.5).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = 384  # Default estimate

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model: %s on %s...", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
                sample = self._model.encode("test", convert_to_numpy=True)
                self._dimension = len(sample)
                logger.info("Successfully loaded %s with vector dimension %d", self.model_name, self._dimension)
            except Exception as e:
                logger.warning("Could not load SentenceTransformer (%s): %s. Falling back to FastDenseEmbedder.", self.model_name, str(e))
                self._model = FastDenseEmbeddingModel(dimension=384)

    @property
    def dimension(self) -> int:
        if self._model is None:
            self._load_model()
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            return self._model.get_sentence_embedding_dimension()
        return getattr(self._model, "dimension", self._dimension)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        if isinstance(self._model, FastDenseEmbeddingModel):
            return self._model.embed_documents(texts)

        embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        self._load_model()
        if isinstance(self._model, FastDenseEmbeddingModel):
            return self._model.embed_query(query)

        embedding = self._model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
