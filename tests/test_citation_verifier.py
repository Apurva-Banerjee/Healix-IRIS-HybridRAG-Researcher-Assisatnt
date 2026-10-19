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
