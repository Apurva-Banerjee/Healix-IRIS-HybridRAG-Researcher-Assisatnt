import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.config.settings import settings
from src.database.models import PaperRecord
from src.database.repository import PaperRepository
from src.generation.answer_generator import ResearchAnswerGenerator
from src.generation.citation_verifier import CitationVerifier
from src.generation.ollama_client import OllamaClient
from src.generation.paper_generator import AcademicPaperGenerator
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.semantic_search import SemanticSearchEngine
from scripts.audit_corpus import CorpusAuditor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="AI Research Scientist Assistant (ScholarRAG) API",
    description="Evidence-grounded scientific research intelligence, hybrid retrieval, and paper synthesis API",
    version=settings.VERSION,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Sanitized HTTP exception response handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"HTTP_{exc.status_code}", "message": exc.detail}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Global exception handler avoiding raw stack trace leaks to API clients."""
    logger.error("Unhandled API error processing %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An internal server error occurred while processing the request."}},
    )


# Service Singletons
repository = PaperRepository()
qdrant_store = QdrantStore()
semantic_search = SemanticSearchEngine(qdrant_store=qdrant_store)
bm25_search = BM25SearchEngine()
hybrid_search = HybridSearchEngine(semantic_search=semantic_search, keyword_search=bm25_search)
ollama_client = OllamaClient()
citation_verifier = CitationVerifier()
answer_generator = ResearchAnswerGenerator(
    ollama_client=ollama_client,
    citation_verifier=citation_verifier,
)
paper_generator = AcademicPaperGenerator(
    retriever=hybrid_search,
    citation_verifier=citation_verifier,
)
ingestion_pipeline = IngestionPipeline(repository=repository)
corpus_auditor = CorpusAuditor()


# Request & Response Schemas
class SearchRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "limitations of GraphRAG"})
    top_k: int = Field(default=5, ge=1, le=50)
    method: str = Field(default="hybrid_rrf", description="Retrieval mode: hybrid_rrf, hybrid_weighted, dense, bm25")
    filter_section: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    method: str
    count: int
    results: List[Dict[str, Any]]


class AskRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What are the key limitations of GraphRAG?"})
    top_k: int = Field(default=5, ge=1, le=20)
    method: str = Field(default="hybrid_rrf", description="Retrieval mode: hybrid_rrf, hybrid_weighted, dense, bm25")


class AskResponse(BaseModel):
    query: str
    answer: str
    evidence_count: int
    sources: List[Dict[str, Any]]
    evidence_chunks: List[Dict[str, Any]]
    generation_time_sec: float
    is_ollama_live: bool
    verification_report: Optional[Dict[str, Any]] = None
    conflicting_evidence: Optional[List[Dict[str, Any]]] = None


class PaperDraftRequest(BaseModel):
    topic: str = Field(default="Comparative Evaluation of GraphRAG and Hybrid RAG in Scientific Literature Synthesis")


class PaperDraftResponse(BaseModel):
    topic: str
    document_path: str
    word_count: int
    evidence_chunks_consulted: int
    cited_papers_count: int
    elapsed_seconds: float
    document_content: str


class IngestRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "GraphRAG"})
    target_papers: int = Field(default=20, ge=1, le=100)
    download_pdfs: bool = Field(default=True)
    max_pdf_downloads: int = Field(default=5, ge=0, le=25)


@app.get("/")
def root():
    return {
        "name": settings.PROJECT_NAME,
        "framework": "ScholarRAG",
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


@app.get("/audit")
def audit_corpus():
    """Return comprehensive integrity metrics across database, PDFs, chunks, and Qdrant."""
    return corpus_auditor.audit()


@app.post("/search", response_model=SearchResponse)
def search_evidence(req: SearchRequest):
    """Retrieve top-K scientific evidence passages matching the query."""
    results = hybrid_search.search(
        query=req.query,
        top_k=req.top_k,
        filter_section=req.filter_section,
        method=req.method,
    )

    return SearchResponse(
        query=req.query,
        method=req.method,
        count=len(results),
        results=results,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """Grounded RAG synthesis: retrieves relevant evidence and generates an attributed answer."""
    retrieved_chunks = hybrid_search.search(
        query=req.query,
        top_k=req.top_k,
        method=req.method,
    )

    result = answer_generator.generate_answer(query=req.query, retrieved_chunks=retrieved_chunks)
    return AskResponse(**result)


@app.post("/paper/draft", response_model=PaperDraftResponse)
def generate_paper_draft(req: PaperDraftRequest):
    """Generate a publication-grade, humanized, plagiarism-free research paper draft."""
    try:
        res = paper_generator.generate_full_paper_draft(topic=req.topic)
        return PaperDraftResponse(**res)
    except Exception as e:
        logger.error("Paper drafting failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Paper draft generation failed: {str(e)}")


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
