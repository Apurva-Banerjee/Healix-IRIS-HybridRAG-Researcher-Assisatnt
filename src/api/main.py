import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from src.config.settings import settings
from src.database.models import PaperRecord
from src.database.repository import PaperRepository
from src.generation.answer_generator import ResearchAnswerGenerator
from src.generation.ollama_client import OllamaClient
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.semantic_search import SemanticSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="AI Research Scientist Assistant API",
    description="Evidence-grounded scientific research intelligence and RAG API",
    version=settings.VERSION,
)

# Service Singletons
repository = PaperRepository()
qdrant_store = QdrantStore()
semantic_search = SemanticSearchEngine(qdrant_store=qdrant_store)
bm25_search = BM25SearchEngine()
hybrid_search = HybridSearchEngine(semantic_search=semantic_search, keyword_search=bm25_search)
ollama_client = OllamaClient()
answer_generator = ResearchAnswerGenerator(ollama_client=ollama_client)
ingestion_pipeline = IngestionPipeline(repository=repository)


# Request & Response Schemas
class SearchRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "limitations of GraphRAG"})
    top_k: int = Field(default=5, ge=1, le=50)
    method: str = Field(default="hybrid", pattern="^(semantic|keyword|hybrid)$")
    filter_section: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    method: str
    count: int
    results: List[Dict[str, Any]]


class AskRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What are the key limitations of GraphRAG?"})
    top_k: int = Field(default=5, ge=1, le=20)
    method: str = Field(default="hybrid", pattern="^(semantic|keyword|hybrid)$")


class AskResponse(BaseModel):
    query: str
    answer: str
    evidence_count: int
    sources: List[Dict[str, Any]]
    evidence_chunks: List[Dict[str, Any]]
    generation_time_sec: float
    is_ollama_live: bool


class IngestRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "GraphRAG"})
    target_papers: int = Field(default=20, ge=1, le=100)
    download_pdfs: bool = Field(default=True)
    max_pdf_downloads: int = Field(default=5, ge=0, le=25)


@app.get("/")
def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    indexed_vectors = qdrant_store.count()
    papers_in_db = repository.count_papers()
    ollama_live = ollama_client.is_available()

    return {
        "status": "healthy",
        "version": settings.VERSION,
        "indexed_vectors": indexed_vectors,
        "papers_in_db": papers_in_db,
        "ollama_online": ollama_live,
        "ollama_model": ollama_client.model,
    }


@app.post("/search", response_model=SearchResponse)
def search_evidence(req: SearchRequest):
    """Retrieve top-K scientific evidence passages matching the query."""
    if req.method == "semantic":
        results = semantic_search.search(req.query, top_k=req.top_k, filter_section=req.filter_section)
    elif req.method == "keyword":
        results = bm25_search.search(req.query, top_k=req.top_k)
    else:
        results = hybrid_search.search(req.query, top_k=req.top_k, filter_section=req.filter_section)

    return SearchResponse(
        query=req.query,
        method=req.method,
        count=len(results),
        results=results,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """Grounded RAG synthesis: retrieves relevant evidence and generates an attributed answer."""
    if req.method == "semantic":
        retrieved_chunks = semantic_search.search(req.query, top_k=req.top_k)
    elif req.method == "keyword":
        retrieved_chunks = bm25_search.search(req.query, top_k=req.top_k)
    else:
        retrieved_chunks = hybrid_search.search(req.query, top_k=req.top_k)

    result = answer_generator.generate_answer(query=req.query, retrieved_chunks=retrieved_chunks)
    return AskResponse(**result)


@app.post("/ingest")
def trigger_ingestion(req: IngestRequest):
    """Trigger paper discovery from arXiv and OpenAlex, deduplicate, store metadata, and download PDFs."""
    try:
        summary = ingestion_pipeline.run(
            query=req.query,
            target_papers=req.target_papers,
            download_pdfs=req.download_pdfs,
            max_pdf_downloads=req.max_pdf_downloads,
        )
        return {
            "status": "success",
            "query": summary["query"],
            "raw_retrieved": summary["raw_retrieved"],
            "deduplicated": summary["deduplicated"],
            "stored_in_db": summary["stored_in_db"],
            "pdfs_downloaded": summary["pdfs_downloaded"],
        }
    except Exception as e:
        logger.error("Ingestion failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/papers", response_model=List[PaperRecord])
def list_papers(limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)):
    """List cataloged papers with metadata and PDF availability."""
    return repository.list_papers(limit=limit, offset=offset)
