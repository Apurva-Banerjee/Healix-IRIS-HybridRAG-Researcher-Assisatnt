from src.chunking.chunk_schema import ChunkRecord
from src.embeddings.embedder import Embedder
from src.embeddings.model import FastDenseEmbeddingModel


def test_fast_dense_embedding_model():
    model = FastDenseEmbeddingModel(dimension=64)
    assert model.dimension == 64

    docs = ["This is a test document", "Another document about retrieval"]
    embeddings = model.embed_documents(docs)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 64

    q_emb = model.embed_query("retrieval test")
    assert len(q_emb) == 64


def test_embedder_chunk_embedding():
    model = FastDenseEmbeddingModel(dimension=128)
    embedder = Embedder(model=model)

    chunk = ChunkRecord(
        chunk_id="chunk_1",
        paper_id="paper_1",
        title="Sample Paper",
        section="Methods",
        text="Sample text content for embedding.",
    )

    embeddings = embedder.embed_chunks([chunk])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 128
