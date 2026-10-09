import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi
from src.chunking.chunk_schema import ChunkRecord
from src.config.settings import settings

logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25 keyword scoring."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    return tokens


class BM25SearchEngine:
    """Keyword search engine using BM25Okapi over indexed chunks."""

    def __init__(self, chunks: Optional[List[ChunkRecord]] = None):
        self.chunks: List[ChunkRecord] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if chunks:
            self.index_chunks(chunks)
        else:
            self.load_from_chunks_dir()

    def index_chunks(self, chunks: List[ChunkRecord]) -> None:
        """Build BM25 index over provided chunk records."""
        self.chunks = chunks
        self.corpus_tokens = [
            tokenize_for_bm25(f"{c.title} {c.section} {c.text}")
            for c in chunks
        ]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            logger.info("Indexed %d chunks in BM25 index", len(chunks))

    def load_from_chunks_dir(self, chunks_dir: Optional[Path] = None) -> None:
        """Load all chunks from the processed chunks directory and build BM25 index."""
        target_dir = chunks_dir or settings.PROCESSED_CHUNKS_DIR
        loaded_chunks: List[ChunkRecord] = []

        if target_dir.exists():
            for json_file in target_dir.glob("*_chunks.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    for item in data:
                        loaded_chunks.append(ChunkRecord(**item))
                except Exception as e:
                    logger.warning("Failed loading chunks from %s: %s", json_file, str(e))

        if loaded_chunks:
            self.index_chunks(loaded_chunks)
        else:
            logger.info("No pre-existing chunk files found for BM25 initialization")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search BM25 index and return scored chunk results."""
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 search requested but index is empty")
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        top_results = []
        for idx, score in indexed_scores[:top_k]:
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            top_results.append({
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "title": chunk.title,
                "section": chunk.section,
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "score": score,
                "text": chunk.text,
                "source": chunk.source,
            })

        return top_results
