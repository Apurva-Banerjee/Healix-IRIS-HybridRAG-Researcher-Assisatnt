import math
from typing import Any, Callable, Dict, List, Set


def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Calculate Precision@K: proportion of retrieved items in top-k that are relevant."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in ground_truth_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Calculate Recall@K: proportion of relevant items successfully retrieved in top-k."""
    if not ground_truth_ids or not retrieved_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in ground_truth_ids)
    return hits / len(ground_truth_ids)


def reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Calculate Reciprocal Rank (RR): 1 / rank of first relevant item."""
    for rank, item in enumerate(retrieved_ids, 1):
        if item in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for rank, item in enumerate(retrieved_ids[:k], 1):
        rel = 1.0 if item in ground_truth_ids else 0.0
        dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    actual_dcg = dcg_at_k(retrieved_ids, ground_truth_ids, k)
    # Ideal DCG assumes all min(len(ground_truth), k) top spots are relevant
    ideal_hits = min(len(ground_truth_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return actual_dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_benchmark(
    eval_dataset: List[Dict[str, Any]],
    retriever_fn: Callable[[str, int], List[Dict[str, Any]]],
    k: int = 5,
) -> Dict[str, float]:
    """
    Run evaluation across an evaluation dataset.
    eval_dataset items format: {"query": str, "relevant_paper_ids": List[str], "expected_topics": List[str]}
    """
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []

    for sample in eval_dataset:
        query = sample["query"]
        ground_truth = set(sample.get("relevant_paper_ids", []))
        topics = sample.get("expected_topics", [])

        retrieved_items = retriever_fn(query, k)

        # Identify which retrieved items are relevant
        relevant_indices = set()
        for idx, item in enumerate(retrieved_items):
            pid = item.get("paper_id", "").lower()
            title = item.get("title", "").lower()
            text = item.get("text", "").lower()

            hit = False
            for gt in ground_truth:
                gt_clean = gt.lower().replace("arxiv:", "").replace("openalex:", "")
                if gt_clean in pid or gt_clean in title:
                    hit = True
                    break

            if not hit and topics:
                for top in topics:
                    if top.lower() in text or top.lower() in title:
                        hit = True
                        break

            if hit:
                relevant_indices.add(str(idx))

        pseudo_retrieved = [str(i) for i in range(len(retrieved_items))]
        # Ground truth count is at least the hits found or 1
        gt_size = max(len(relevant_indices), 1)

        p = len(relevant_indices) / max(k, 1)
        r = len(relevant_indices) / gt_size
        rr = 0.0
        for rank, item_key in enumerate(pseudo_retrieved, 1):
            if item_key in relevant_indices:
                rr = 1.0 / rank
                break

        dcg = sum(1.0 / math.log2(rank + 1) for rank, item_key in enumerate(pseudo_retrieved[:k], 1) if item_key in relevant_indices)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_indices), k) + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        mrrs.append(rr)
        ndcgs.append(ndcg)

    n = max(len(eval_dataset), 1)
    return {
        f"Precision@{k}": round(sum(precisions) / n, 4),
        f"Recall@{k}": round(sum(recalls) / n, 4),
        "MRR": round(sum(mrrs) / n, 4),
        f"nDCG@{k}": round(sum(ndcgs) / n, 4),
        "total_queries_evaluated": len(eval_dataset),
    }
