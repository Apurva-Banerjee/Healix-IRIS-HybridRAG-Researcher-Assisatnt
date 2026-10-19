from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["framework"] == "ScholarRAG"


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_vectors" in data
    assert "papers_in_db" in data


def test_api_audit():
    response = client.get("/audit")
    assert response.status_code == 200
    data = response.json()
    assert "db_papers_count" in data
    assert "total_chunk_units" in data
    assert data["total_chunk_units"] == 929


def test_api_search_all_modes():
    for mode in ["bm25", "dense", "hybrid_rrf", "hybrid_weighted"]:
        response = client.post(
            "/search",
            json={"query": "GraphRAG limitations", "top_k": 2, "method": mode},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data
        assert len(data["results"]) <= 2


def test_api_ask():
    response = client.post(
        "/ask",
        json={"query": "What are reflection tokens in Self-RAG?", "top_k": 2, "method": "hybrid_rrf"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "evidence_count" in data
    assert "verification_report" in data


def test_api_paper_draft():
    response = client.post(
        "/paper/draft",
        json={"topic": "Evaluation of Hybrid RAG Architectures"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_content" in data
    assert "word_count" in data
    assert data["word_count"] > 100


def test_api_papers_list():
    response = client.get("/papers?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
