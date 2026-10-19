import math
from typing import Any, Callable, Dict, List, Optional, Set, Union


def precision_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Calculate Precision@K: proportion of top-K retrieved items that are relevant.
    P@K = |Retrieved_{1:K} ∩ Relevant| / K
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in ground_truth_ids)
    return hits / float(k)


def recall_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Calculate Recall@K: proportion of judged relevant items retrieved in top-K.
    R@K = |Retrieved_{1:K} ∩ Relevant| / |Relevant|
    Denominator is strictly the total number of relevant documents for this query.
    """
    if not ground_truth_ids or k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in ground_truth_ids)
    return hits / float(len(ground_truth_ids))


def reciprocal_rank(
    retrieved_ids: List[str],
    ground_truth_ids: Set[str],
) -> float:
    """
    Calculate Reciprocal Rank (RR): 1 / rank of first relevant item (1-based).
    Returns 0.0 if no relevant items retrieved.
    """
    for rank, item in enumerate(retrieved_ids, 1):
        if item in ground_truth_ids:
            return 1.0 / float(rank)
    return 0.0


def dcg_at_k(
    retrieved_ids: List[str],
    relevance_data: Union[Set[str], Dict[str, float]],
    k: int = 5,
) -> float:
    """
    Discounted Cumulative Gain at rank K.
    Standard formula: DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    rel_map: Dict[str, float] = (
        relevance_data if isinstance(relevance_data, dict)
        else {item: 1.0 for item in relevance_data}
    )

    dcg = 0.0
    for rank, item in enumerate(retrieved_ids[:k], 1):
        rel = rel_map.get(item, 0.0)
        if rel > 0.0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(
    retrieved_ids: List[str],
    relevance_data: Union[Set[str], Dict[str, float]],
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain at rank K.
    nDCG@K = DCG@K / IDCG@K
    IDCG is calculated from the top-K highest relevance values available in ground truth.
    """
    if k <= 0 or not relevance_data:
        return 0.0

    rel_map: Dict[str, float] = (
        relevance_data if isinstance(relevance_data, dict)
        else {item: 1.0 for item in relevance_data}
    )

    actual_dcg = dcg_at_k(retrieved_ids, rel_map, k)
    if actual_dcg == 0.0:
        return 0.0

    # Ideal ranking places highest relevance scores first
    sorted_rels = sorted(rel_map.values(), reverse=True)[:k]
    idcg = sum((math.pow(2.0, rel) - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(sorted_rels, 1))

    return actual_dcg / idcg if idcg > 0.0 else 0.0


def canonicalize_id(raw_id: str) -> str:
    """Normalize identifier for robust exact matching (handles prefixes like arxiv:, openalex:, acl:)."""
    if not raw_id:
        return ""
    s = raw_id.strip().lower()
    for prefix in ["arxiv:", "openalex:", "acl:", "file:"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()


def evaluate_retrieval_benchmark(
    eval_dataset: List[Dict[str, Any]],
    retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
    k: int = 5,
    evaluation_unit: str = "paper",  # "paper" or "passage"
) -> Dict[str, Any]:
    """
    Run mathematically rigorous retrieval evaluation across a benchmark dataset.
    Supports paper-level and passage-level evaluation.
    Returns both aggregate summary metrics and per-query auditable records.
    """
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []
    per_query_records: List[Dict[str, Any]] = []

    for sample in eval_dataset:
        qid = sample.get("query_id", "unknown_query")
        query = sample["query"]

        # 1. Build authoritative ground truth set & relevance map
        ground_truth_raw = set(sample.get("relevant_paper_ids", []))
        canonical_gt = {canonicalize_id(gt) for gt in ground_truth_raw if gt}
        relevance_map = {canonicalize_id(gt): 1.0 for gt in canonical_gt}

        # 2. Retrieve candidates
        retrieved_chunks = retriever_fn(query, k)

        # 3. Deduplicate and extract retrieved IDs based on evaluation unit
        retrieved_ids: List[str] = []
        retrieved_details: List[Dict[str, Any]] = []

        if evaluation_unit == "paper":
            seen_papers: Set[str] = set()
            for rank_idx, chunk in enumerate(retrieved_chunks, 1):
                raw_pid = chunk.get("paper_id") or chunk.get("title") or ""
                can_pid = canonicalize_id(raw_pid)
                if can_pid and can_pid not in seen_papers:
                    seen_papers.add(can_pid)
                    retrieved_ids.append(can_pid)
                    is_rel = can_pid in canonical_gt
                    retrieved_details.append({
                        "rank": len(retrieved_ids),
                        "original_rank": rank_idx,
                        "paper_id": raw_pid,
                        "canonical_id": can_pid,
                        "chunk_id": chunk.get("chunk_id"),
                        "score": chunk.get("score"),
                        "is_relevant": is_rel,
                    })
        else:
            # Passage level
            for rank_idx, chunk in enumerate(retrieved_chunks[:k], 1):
                cid = chunk.get("chunk_id", "")
                can_cid = canonicalize_id(cid)
                retrieved_ids.append(can_cid)
                is_rel = can_cid in canonical_gt
                retrieved_details.append({
                    "rank": rank_idx,
                    "chunk_id": cid,
                    "canonical_id": can_cid,
                    "paper_id": chunk.get("paper_id"),
                    "score": chunk.get("score"),
                    "is_relevant": is_rel,
                })

        # 4. Compute metrics using corrected denominators
        p = precision_at_k(retrieved_ids, canonical_gt, k=k)
        r = recall_at_k(retrieved_ids, canonical_gt, k=k)
        rr = reciprocal_rank(retrieved_ids, canonical_gt)
        ndcg = ndcg_at_k(retrieved_ids, relevance_map, k=k)

        precisions.append(p)
        recalls.append(r)
        mrrs.append(rr)
        ndcgs.append(ndcg)

        per_query_records.append({
            "query_id": qid,
            "query": query,
            "category": sample.get("category", "general"),
            "ground_truth_count": len(canonical_gt),
            "retrieved_count": len(retrieved_ids),
            f"Precision@{k}": round(p, 4),
            f"Recall@{k}": round(r, 4),
            "ReciprocalRank": round(rr, 4),
            f"nDCG@{k}": round(ndcg, 4),
            "retrieved_items": retrieved_details[:k],
        })

    n = max(len(eval_dataset), 1)
    return {
        f"Precision@{k}": round(sum(precisions) / n, 4),
        f"Recall@{k}": round(sum(recalls) / n, 4),
        "MRR": round(sum(mrrs) / n, 4),
        f"nDCG@{k}": round(sum(ndcgs) / n, 4),
        "total_queries_evaluated": len(eval_dataset),
        "evaluation_unit": evaluation_unit,
        "per_query_results": per_query_records,
    }
