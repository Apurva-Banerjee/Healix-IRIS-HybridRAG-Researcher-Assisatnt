import pytest
from qdrant_client import QdrantClient
from src.chunking.chunk_schema import ChunkRecord
from src.embeddings.embedder import Embedder
from src.embeddings.model import FastDenseEmbeddingModel
from src.retrieval.hybrid_search import HybridSearchEngine, normalize_scores
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.semantic_search import SemanticSearchEngine


@pytest.fixture
def test_corpus():
    c1 = ChunkRecord(
        chunk_id="c1",
        paper_id="paper_1",
        title="Graph Retrieval Paper",
        section="Methodology",
        text="Graph construction and community detection using Leiden algorithm for global search.",
    )
    c2 = ChunkRecord(
        chunk_id="c2",
        paper_id="paper_2",
        title="Dense Embedding Paper",
        section="Experiments",
        text="Dense vectors achieve high recall on semantic similarity and passage retrieval.",
    )
    c3 = ChunkRecord(
        chunk_id="c3",
        paper_id="paper_3",
        title="Lexical Search Paper",
        section="Abstract",
        text="BM25 and inverted index term frequency matching for exact keyword queries.",
    )
    return [c1, c2, c3]


@pytest.fixture
def hybrid_retriever(test_corpus):
    q_client = QdrantClient(":memory:")
    dim = 64
    model = FastDenseEmbeddingModel(dimension=dim)
    embedder = Embedder(model=model)

    q_store = QdrantStore(collection_name="test_modes", dimension=dim, client=q_client)
    embeddings = embedder.embed_chunks(test_corpus)
    q_store.upsert_chunks(test_corpus, embeddings)

    sem_engine = SemanticSearchEngine(embedder=embedder, qdrant_store=q_store)
    bm25_engine = BM25SearchEngine(chunks=test_corpus)

    return HybridSearchEngine(
        semantic_search=sem_engine,
        keyword_search=bm25_engine,
        alpha=0.7,
        rrf_k=60,
    )


def test_normalize_scores():
    items = [
        {"chunk_id": "c1", "score": 10.0},
        {"chunk_id": "c2", "score": 20.0},
        {"chunk_id": "c3", "score": 30.0},
    ]
    norm = normalize_scores(items)
    assert norm["c1"] == 0.0
    assert norm["c2"] == 0.5
    assert norm["c3"] == 1.0

    # Test single item or all equal
    single = [{"chunk_id": "c1", "score": 5.0}]
    assert normalize_scores(single)["c1"] == 1.0


def test_dense_mode(hybrid_retriever):
    results = hybrid_retriever.search("dense semantic similarity", top_k=2, method="dense")
    assert len(results) > 0
    for r in results:
        assert r["retrieval_lane"] == "dense"
        assert "score" in r
        assert "text" in r


def test_bm25_mode(hybrid_retriever):
    results = hybrid_retriever.search("Leiden algorithm", top_k=2, method="bm25")
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"
    for r in results:
        assert r["retrieval_lane"] == "bm25"
        assert r["bm25_score"] > 0


def test_hybrid_rrf_mode(hybrid_retriever):
    results = hybrid_retriever.search("exact keyword BM25", top_k=3, method="hybrid_rrf")
    assert len(results) > 0
    top = results[0]
    assert top["retrieval_lane"] == "hybrid_rrf"
    # RRF score should be <= 2 * (1 / (60 + 1)) = 0.03278
    assert top["score"] > 0.0
    assert top["score"] <= 0.04


def test_hybrid_weighted_mode(hybrid_retriever):
    results = hybrid_retriever.search("passage retrieval", top_k=2, method="hybrid_weighted")
    assert len(results) > 0
    for r in results:
        assert r["retrieval_lane"] == "hybrid_weighted"
        assert 0.0 <= r["score"] <= 1.0
