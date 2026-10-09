import logging
from typing import Any, Dict, List, Optional
from src.config.settings import settings
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


def normalize_scores(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """Min-max normalize scores of retrieved items to [0, 1]."""
    if not items:
        return {}
    scores = [item["score"] for item in items]
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return {item["chunk_id"]: 1.0 for item in items}
    return {
        item["chunk_id"]: (item["score"] - min_s) / (max_s - min_s)
        for item in items
    }


class HybridSearchEngine:
    """
    Hybrid retriever combining Dense Semantic Search (Qdrant) with Sparse Keyword Search (BM25).
    Applies linear score combination and Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        semantic_search: Optional[SemanticSearchEngine] = None,
        keyword_search: Optional[BM25SearchEngine] = None,
        alpha: Optional[float] = None,
    ):
        self.semantic_search = semantic_search or SemanticSearchEngine()
        self.keyword_search = keyword_search or BM25SearchEngine()
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_section: Optional[str] = None,
        method: str = "linear",  # "linear" or "rrf"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining vector similarity with keyword matching.
        """
        logger.info(
            "Hybrid search query='%s' (top_k=%d, alpha=%.2f, method=%s)",
            query,
            top_k,
            self.alpha,
            method,
        )

        candidate_k = max(top_k * 2, 10)

        # 1. Retrieve candidates from both engines
        semantic_results = self.semantic_search.search(
            query=query, top_k=candidate_k, filter_section=filter_section
        )
        bm25_results = self.keyword_search.search(query=query, top_k=candidate_k)

        # Collect unique chunks by chunk_id
        chunk_dict: Dict[str, Dict[str, Any]] = {}
        for r in semantic_results:
            chunk_dict[r["chunk_id"]] = {**r}
        for r in bm25_results:
            if r["chunk_id"] not in chunk_dict:
                chunk_dict[r["chunk_id"]] = {**r}

        if not chunk_dict:
            return []

        # 2. Score Fusion
        final_scores: Dict[str, float] = {}

        if method == "rrf":
            # Reciprocal Rank Fusion (k=60)
            rrf_k = 60
            for rank, r in enumerate(semantic_results):
                cid = r["chunk_id"]
                final_scores[cid] = final_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            for rank, r in enumerate(bm25_results):
                cid = r["chunk_id"]
                final_scores[cid] = final_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
        else:
            # Linear score combination
            norm_sem = normalize_scores(semantic_results)
            norm_bm = normalize_scores(bm25_results)

            for cid in chunk_dict:
                s_score = norm_sem.get(cid, 0.0)
                b_score = norm_bm.get(cid, 0.0)
                combined = (self.alpha * s_score) + ((1.0 - self.alpha) * b_score)
                final_scores[cid] = combined
                chunk_dict[cid]["semantic_score"] = s_score
                chunk_dict[cid]["bm25_score"] = b_score

        # 3. Sort and rank
        ranked_chunks = []
        sorted_ids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        for cid, score in sorted_ids[:top_k]:
            item = chunk_dict[cid]
            item["score"] = round(score, 4)
            ranked_chunks.append(item)

        return ranked_chunks
