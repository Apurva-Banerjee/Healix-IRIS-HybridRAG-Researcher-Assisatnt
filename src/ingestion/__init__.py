from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.openalex_client import OpenAlexClient
from src.ingestion.semantic_scholar_client import SemanticScholarClient
from src.ingestion.normalizer import normalize_text, normalize_title, normalize_doi, normalize_arxiv_id
from src.ingestion.deduplicator import PaperDeduplicator
from src.ingestion.ingestion_pipeline import IngestionPipeline

__all__ = [
    "ArxivClient",
    "OpenAlexClient",
    "SemanticScholarClient",
    "normalize_text",
    "normalize_title",
    "normalize_doi",
    "normalize_arxiv_id",
    "PaperDeduplicator",
    "IngestionPipeline",
]
