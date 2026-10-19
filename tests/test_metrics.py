import math
import pytest
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    canonicalize_id,
    evaluate_retrieval_benchmark,
)


def test_hand_calculated_binary_metrics():
    # 2 total relevant documents in ground truth: d1 and d3
    ground_truth = {"d1", "d3"}
    retrieved = ["d1", "d2", "d3", "d4", "d5"]

    # At K = 2: top-2 is ["d1", "d2"]
    # Hits = 1 (d1)
    assert precision_at_k(retrieved, ground_truth, k=2) == 0.5
    # Recall denominator MUST be len(ground_truth) = 2, so 1 / 2 = 0.5
    assert recall_at_k(retrieved, ground_truth, k=2) == 0.5
    # First relevant item is at rank 1 -> RR = 1.0
    assert reciprocal_rank(retrieved, ground_truth) == 1.0

    # DCG@2: d1 is rank 1 -> 1 / log2(2) = 1.0; d2 is rank 2 (not relevant) -> 0.
    # IDCG@2: ideal top-2 would be d1 and d3 -> 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
    # nDCG@2 = 1.0 / 1.63092975 = 0.613147...
    expected_ndcg_2 = 1.0 / (1.0 + (1.0 / math.log2(3)))
    assert math.isclose(ndcg_at_k(retrieved, ground_truth, k=2), expected_ndcg_2, rel_tol=1e-5)

    # At K = 3: top-3 is ["d1", "d2", "d3"]
    # Hits = 2 (d1, d3)
    assert math.isclose(precision_at_k(retrieved, ground_truth, k=3), 2.0 / 3.0, rel_tol=1e-5)
    # Recall@3: 2 hits / 2 relevant = 1.0
    assert recall_at_k(retrieved, ground_truth, k=3) == 1.0


def test_edge_cases():
    ground_truth = {"d1", "d2"}

    # Zero hits
    retrieved_none = ["d9", "d8", "d7"]
    assert precision_at_k(retrieved_none, ground_truth, k=2) == 0.0
    assert recall_at_k(retrieved_none, ground_truth, k=2) == 0.0
    assert reciprocal_rank(retrieved_none, ground_truth) == 0.0
    assert ndcg_at_k(retrieved_none, ground_truth, k=2) == 0.0

    # Empty retrieved list
    assert precision_at_k([], ground_truth, k=5) == 0.0
    assert recall_at_k([], ground_truth, k=5) == 0.0
    assert reciprocal_rank([], ground_truth) == 0.0
    assert ndcg_at_k([], ground_truth, k=5) == 0.0

    # Empty ground truth
    assert recall_at_k(["d1"], set(), k=5) == 0.0
    assert ndcg_at_k(["d1"], set(), k=5) == 0.0

    # K = 0 or negative
    assert precision_at_k(["d1"], ground_truth, k=0) == 0.0
    assert recall_at_k(["d1"], ground_truth, k=-1) == 0.0


def test_graded_ndcg():
    # Graded relevance: d1 has grade 3, d2 has grade 1, d3 has grade 0
    rel_map = {"d1": 3.0, "d2": 1.0}
    retrieved_ideal = ["d1", "d2"]
    # Ideal ranking -> nDCG must be 1.0
    assert math.isclose(ndcg_at_k(retrieved_ideal, rel_map, k=2), 1.0, rel_tol=1e-5)

    # Sub-optimal ranking: d2 at rank 1, d1 at rank 2
    retrieved_sub = ["d2", "d1"]
    # DCG = (2^1 - 1)/log2(2) + (2^3 - 1)/log2(3) = 1/1 + 7/1.5849625 = 1 + 4.4165 = 5.4165
    # IDCG = (2^3 - 1)/log2(2) + (2^1 - 1)/log2(3) = 7/1 + 1/1.5849625 = 7 + 0.6309 = 7.6309
    expected_sub_ndcg = (1.0 + 7.0 / math.log2(3)) / (7.0 + 1.0 / math.log2(3))
    assert math.isclose(ndcg_at_k(retrieved_sub, rel_map, k=2), expected_sub_ndcg, rel_tol=1e-5)


def test_canonicalize_id():
    assert canonicalize_id("arxiv:2004.12832") == "2004.12832"
    assert canonicalize_id("openalex:W4406069943") == "w4406069943"
    assert canonicalize_id("acl:2024.acl-long.236") == "2024.acl-long.236"
    assert canonicalize_id("  2310.11511  ") == "2310.11511"
