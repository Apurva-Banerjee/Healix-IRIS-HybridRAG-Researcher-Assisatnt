from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_vectors" in data
    assert "papers_in_db" in data


def test_api_search():
    response = client.post(
        "/search",
        json={"query": "GraphRAG limitations", "top_k": 2, "method": "hybrid"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "count" in data


def test_api_papers_list():
    response = client.get("/papers?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
