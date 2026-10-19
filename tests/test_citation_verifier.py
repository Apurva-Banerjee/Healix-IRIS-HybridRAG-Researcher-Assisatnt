import pytest
from src.generation.citation_verifier import CitationVerifier


@pytest.fixture
def sample_evidence():
    return [
        {
            "chunk_id": "c1",
            "paper_id": "paper_1",
            "title": "ColBERT: Efficient Passage Search",
            "section": "Introduction",
            "page": 1,
            "text": "ColBERT introduces contextualized late interaction over BERT, significantly improving passage retrieval latency.",
        },
        {
            "chunk_id": "c2",
            "paper_id": "paper_2",
            "title": "Do We Still Need GraphRAG?",
            "section": "Experiments",
            "page": 5,
            "text": "Empirical evaluations reveal no significant accuracy improvement over hybrid RAG while incurring high computational overhead.",
        },
    ]


def test_extract_cited_indices():
    verifier = CitationVerifier()
    text = "According to [Evidence 1] and [Evidence 2], retrieval is effective. Prior work [E1] confirmed this."
    indices = verifier.extract_cited_indices(text)
    assert indices == [1, 2]


def test_verify_citations_valid(sample_evidence):
    verifier = CitationVerifier()
    text = "ColBERT improves query latency substantially [Evidence 1]."
    report = verifier.verify_citations(text, sample_evidence)

    assert report["total_citations_found"] == 1
    assert report["valid_citations_count"] == 1
    assert report["dangling_citations_count"] == 0
    assert report["citation_support_rate"] == 1.0
    assert report["is_fully_grounded"] is True


def test_verify_citations_dangling(sample_evidence):
    verifier = CitationVerifier()
    text = "Unsupported claim citing non-existent evidence [Evidence 99]."
    report = verifier.verify_citations(text, sample_evidence)

    assert report["total_citations_found"] == 1
    assert report["valid_citations_count"] == 0
    assert report["dangling_citations_count"] == 1
    assert report["dangling_indices"] == [99]
    assert report["citation_support_rate"] == 0.0
    assert report["is_fully_grounded"] is False


def test_detect_conflicting_evidence(sample_evidence):
    verifier = CitationVerifier()
    conflicts = verifier.detect_conflicting_evidence(sample_evidence)
    assert len(conflicts) >= 1
    assert conflicts[0]["type"] == "empirical_divergence"


def test_verify_claim_support_categories():
    verifier = CitationVerifier()

    # Supported
    c1 = "ColBERT utilizes contextualized late interaction over BERT to perform efficient passage retrieval."
    e1 = "ColBERT introduces contextualized late interaction over BERT, significantly improving passage retrieval latency through token-level vector representations."
    res1 = verifier.verify_claim_support(c1, e1)
    assert res1["status"] == "supported"
    assert res1["confidence"] >= 0.5

    # Unsupported: Numerical contradiction
    c2 = "The hybrid method achieved a ROUGE score of 98.5% on the defect QA benchmark."
    e2 = "Our results show that the hybrid LLM-based QA method achieved the highest ROUGE score of 81.6%."
    res2 = verifier.verify_claim_support(c2, e2)
    assert res2["status"] == "unsupported"
    assert "Numerical contradiction" in res2["reason"]

    # Unsupported: Absolute contradiction
    c3 = "GraphRAG completely replaces vector embeddings and requires zero dense indexes."
    e3 = "GraphRAG constructs knowledge graphs from text chunks, which are typically used in conjunction with dense vector similarity search."
    res3 = verifier.verify_claim_support(c3, e3)
    assert res3["status"] == "unsupported"
    assert "Contradiction" in res3["reason"]

    # Insufficient evidence
    c4 = "The indexed corpus provides insufficient evidence to determine the exact hardware specifications."
    e4 = "No hardware configuration details provided in this section."
    res4 = verifier.verify_claim_support(c4, e4)
    assert res4["status"] == "insufficient_evidence"


def test_evaluate_claim_support_gold_dataset():
    import json
    from pathlib import Path

    eval_file = Path("data/evaluation/citation_support_eval.json")
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        verifier = CitationVerifier()
        eval_report = verifier.evaluate_claim_support_dataset(pairs)

        assert eval_report["total_pairs"] == 6
        assert eval_report["correct_predictions"] == 6
        assert eval_report["accuracy"] == 1.0

