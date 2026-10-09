from qdrant_client import QdrantClient
from src.chunking.chunk_schema import ChunkRecord
from src.embeddings.embedder import Embedder
from src.embeddings.model import FastDenseEmbeddingModel
from src.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.semantic_search import SemanticSearchEngine


def test_retrieval_and_metrics():
    # Setup test chunks
    c1 = ChunkRecord(
        chunk_id="c1",
        paper_id="paper_graphrag",
        title="GraphRAG Paper",
        section="Limitations",
        text="GraphRAG suffers from high computational overhead during graph construction.",
    )
    c2 = ChunkRecord(
        chunk_id="c2",
        paper_id="paper_rag",
        title="Traditional RAG",
        section="Introduction",
        text="Standard RAG uses top-k vector chunks without graph structures.",
    )
    c3 = ChunkRecord(
        chunk_id="c3",
        paper_id="paper_vision",
        title="Vision Transformers",
        section="Experiments",
        text="Image classification accuracy on ImageNet with multi-head attention.",
    )

    chunks = [c1, c2, c3]

    # Setup in-memory Qdrant
    q_client = QdrantClient(":memory:")
    dim = 64
    model = FastDenseEmbeddingModel(dimension=dim)
    embedder = Embedder(model=model)

    q_store = QdrantStore(collection_name="test_col", dimension=dim, client=q_client)
    embeddings = embedder.embed_chunks(chunks)
    q_store.upsert_chunks(chunks, embeddings)

    assert q_store.count() == 3

    # Semantic Search
    sem_engine = SemanticSearchEngine(embedder=embedder, qdrant_store=q_store)
    sem_res = sem_engine.search("computational overhead graph", top_k=2)
    assert len(sem_res) > 0

    # BM25 Search
    bm25_engine = BM25SearchEngine(chunks=chunks)
    bm25_res = bm25_engine.search("overhead graph", top_k=2)
    assert len(bm25_res) > 0
    assert bm25_res[0]["chunk_id"] == "c1"

    # Hybrid Search
    hyb_engine = HybridSearchEngine(semantic_search=sem_engine, keyword_search=bm25_engine)
    hyb_res = hyb_engine.search("computational overhead graph", top_k=2)
    assert len(hyb_res) > 0

    # Metric tests
    retrieved_ids = ["paper_graphrag", "paper_rag"]
    ground_truth = {"paper_graphrag"}

    p = precision_at_k(retrieved_ids, ground_truth, k=1)
    r = recall_at_k(retrieved_ids, ground_truth, k=1)
    rr = reciprocal_rank(retrieved_ids, ground_truth)
    ndcg = ndcg_at_k(retrieved_ids, ground_truth, k=2)

    assert p == 1.0
    assert r == 1.0
    assert rr == 1.0
    assert ndcg == 1.0
