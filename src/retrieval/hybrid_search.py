import logging
from typing import Any, Dict, List, Optional
from src.config.settings import settings
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


def normalize_scores(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Min-max normalize raw scores to [0, 1].
    If all scores are equal or item count is 1, returns 1.0 for each item.
    """
    if not items:
        return {}
    scores = [float(item.get("score", 0.0)) for item in items]
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return {item["chunk_id"]: 1.0 for item in items}
    return {
        item["chunk_id"]: (float(item.get("score", 0.0)) - min_s) / (max_s - min_s)
        for item in items
    }


class HybridSearchEngine:
    """
    Unified multi-mode research retrieval coordinator supporting:
      1. 'bm25'           : Sparse lexical term matching (Okapi BM25)
      2. 'dense'          : Dense semantic vector similarity (BGE-small / Qdrant)
      3. 'hybrid_rrf'     : Reciprocal Rank Fusion (RRF, k=60, 1-based ranks)
      4. 'hybrid_weighted': Min-Max normalized linear score fusion (alpha * dense + (1-alpha) * bm25)
    """

    def __init__(
        self,
        semantic_search: Optional[SemanticSearchEngine] = None,
        keyword_search: Optional[BM25SearchEngine] = None,
        alpha: Optional[float] = None,
        rrf_k: int = 60,
    ):
        self.semantic_search = semantic_search or SemanticSearchEngine()
        self.keyword_search = keyword_search or BM25SearchEngine()
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        self.rrf_k = rrf_k

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        filter_section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search only."""
        results = self.semantic_search.search(
            query=query, top_k=top_k, filter_section=filter_section
        )
        for r in results:
            r["retrieval_lane"] = "dense"
            r["semantic_score"] = r.get("score", 0.0)
            r["bm25_score"] = 0.0
        return results

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform BM25 sparse keyword search only."""
        results = self.keyword_search.search(query=query, top_k=top_k)
        for r in results:
            r["retrieval_lane"] = "bm25"
            r["bm25_score"] = r.get("score", 0.0)
            r["semantic_score"] = 0.0
        return results

    def search_rrf(
        self,
        query: str,
        top_k: int = 5,
        filter_section: Optional[str] = None,
        rrf_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform Reciprocal Rank Fusion (RRF) between Dense and BM25 search.
        Formula: RRF(d) = sum_{r in {dense, bm25}} (1.0 / (k + rank_r(d)))
        Using 1-based ranks and k=60.
        """
        k = rrf_k or self.rrf_k
        candidate_k = max(top_k * 3, 15)

        semantic_results = self.semantic_search.search(
            query=query, top_k=candidate_k, filter_section=filter_section
        )
        bm25_results = self.keyword_search.search(query=query, top_k=candidate_k)

        chunk_dict: Dict[str, Dict[str, Any]] = {}
        for r in semantic_results:
            chunk_dict[r["chunk_id"]] = {**r}
        for r in bm25_results:
            if r["chunk_id"] not in chunk_dict:
                chunk_dict[r["chunk_id"]] = {**r}

        if not chunk_dict:
            return []

        rrf_scores: Dict[str, float] = {}
        sem_ranks: Dict[str, int] = {}
        bm_ranks: Dict[str, int] = {}

        for rank_0, r in enumerate(semantic_results):
            cid = r["chunk_id"]
            rank_1 = rank_0 + 1
            sem_ranks[cid] = rank_1
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank_1))

        for rank_0, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            rank_1 = rank_0 + 1
            bm_ranks[cid] = rank_1
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank_1))

        # Deterministic sorting: highest score, tie-break by chunk_id
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )

        ranked_results: List[Dict[str, Any]] = []
        for cid, score in sorted_items[:top_k]:
            item = dict(chunk_dict[cid])
            item["score"] = round(score, 6)
            item["retrieval_lane"] = "hybrid_rrf"
            item["dense_rank"] = sem_ranks.get(cid, None)
            item["bm25_rank"] = bm_ranks.get(cid, None)
            ranked_results.append(item)

        return ranked_results

    def search_weighted(
        self,
        query: str,
        top_k: int = 5,
        filter_section: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform min-max normalized linear score fusion:
        S(d) = alpha * Norm(S_dense(d)) + (1 - alpha) * Norm(S_bm25(d))
        """
        effective_alpha = alpha if alpha is not None else self.alpha
        candidate_k = max(top_k * 3, 15)

        semantic_results = self.semantic_search.search(
            query=query, top_k=candidate_k, filter_section=filter_section
        )
        bm25_results = self.keyword_search.search(query=query, top_k=candidate_k)

        chunk_dict: Dict[str, Dict[str, Any]] = {}
        for r in semantic_results:
            chunk_dict[r["chunk_id"]] = {**r}
        for r in bm25_results:
            if r["chunk_id"] not in chunk_dict:
                chunk_dict[r["chunk_id"]] = {**r}

        if not chunk_dict:
            return []

        norm_sem = normalize_scores(semantic_results)
        norm_bm = normalize_scores(bm25_results)

        final_scores: Dict[str, float] = {}
        for cid in chunk_dict:
            s_score = norm_sem.get(cid, 0.0)
            b_score = norm_bm.get(cid, 0.0)
            combined = (effective_alpha * s_score) + ((1.0 - effective_alpha) * b_score)
            final_scores[cid] = combined
            chunk_dict[cid]["semantic_score"] = round(s_score, 4)
            chunk_dict[cid]["bm25_score"] = round(b_score, 4)

        # Deterministic sorting
        sorted_items = sorted(
            final_scores.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )

        ranked_results: List[Dict[str, Any]] = []
        for cid, score in sorted_items[:top_k]:
            item = dict(chunk_dict[cid])
            item["score"] = round(score, 4)
            item["retrieval_lane"] = "hybrid_weighted"
            ranked_results.append(item)

        return ranked_results

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_section: Optional[str] = None,
        method: str = "hybrid_rrf",
    ) -> List[Dict[str, Any]]:
        """
        Unified dispatch search method.
        Modes: 'bm25', 'dense', 'hybrid_rrf' (or 'rrf'), 'hybrid_weighted' (or 'linear', 'hybrid')
        """
        m = (method or "hybrid_rrf").lower()
        if m == "dense":
            return self.search_dense(query=query, top_k=top_k, filter_section=filter_section)
        elif m in ("bm25", "sparse", "keyword"):
            return self.search_bm25(query=query, top_k=top_k)
        elif m in ("hybrid_weighted", "weighted", "linear", "hybrid"):
            return self.search_weighted(query=query, top_k=top_k, filter_section=filter_section)
        elif m in ("hybrid_rrf", "rrf"):
            return self.search_rrf(query=query, top_k=top_k, filter_section=filter_section)
        else:
            logger.warning("Unrecognized retrieval method '%s'. Defaulting to 'hybrid_rrf'.", method)
            return self.search_rrf(query=query, top_k=top_k, filter_section=filter_section)
