import logging
from typing import Dict, List, Optional
from src.database.models import IngestionLog, PaperRecord
from src.database.repository import PaperRepository
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.deduplicator import PaperDeduplicator
from src.ingestion.openalex_client import OpenAlexClient
from src.ingestion.semantic_scholar_client import SemanticScholarClient
from src.pdf_processing.downloader import PDFDownloader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        repository: Optional[PaperRepository] = None,
        arxiv_client: Optional[ArxivClient] = None,
        openalex_client: Optional[OpenAlexClient] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        deduplicator: Optional[PaperDeduplicator] = None,
        pdf_downloader: Optional[PDFDownloader] = None,
    ):
        self.repository = repository or PaperRepository()
        self.arxiv_client = arxiv_client or ArxivClient()
        self.openalex_client = openalex_client or OpenAlexClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()
        self.deduplicator = deduplicator or PaperDeduplicator()
        self.pdf_downloader = pdf_downloader or PDFDownloader()

    def run(
        self,
        query: str,
        target_papers: int = 50,
        download_pdfs: bool = True,
        max_pdf_downloads: int = 15,
        include_semantic_scholar: bool = False,
    ) -> Dict[str, any]:
        """
        Execute full Phase 1 Ingestion Pipeline:
        1. Query arXiv
        2. Query OpenAlex
        3. (Optional) Query Semantic Scholar
        4. Normalize & Deduplicate
        5. Store in SQLite DB
        6. Download PDFs for top records
        7. Log metrics
        """
        logger.info("Starting ingestion pipeline for query: '%s' (target: %d)", query, target_papers)

        per_source_limit = max(15, target_papers // 2 + 5)
        raw_papers: List[PaperRecord] = []

        # 1. arXiv Ingestion
        arxiv_results = self.arxiv_client.search(query, max_results=per_source_limit)
        raw_papers.extend(arxiv_results)

        # 2. OpenAlex Ingestion
        openalex_results = self.openalex_client.search(query, max_results=per_source_limit)
        raw_papers.extend(openalex_results)

        # 3. Optional Semantic Scholar Ingestion
        if include_semantic_scholar:
            s2_results = self.semantic_scholar_client.search(query, max_results=per_source_limit)
            raw_papers.extend(s2_results)

        total_raw = len(raw_papers)
        logger.info("Retrieved %d raw candidate papers across sources", total_raw)

        # 4. Deduplication
        deduped_papers = self.deduplicator.deduplicate(raw_papers)
        total_deduped = len(deduped_papers)

        # 5. Database Storage
        self.repository.save_papers_batch(deduped_papers)

        # 6. PDF Downloading
        downloaded_count = 0
        if download_pdfs:
            logger.info("Initiating PDF downloads for top %d papers...", max_pdf_downloads)
            downloaded_paths = self.pdf_downloader.download_batch(
                papers=deduped_papers,
                repository=self.repository,
                max_count=max_pdf_downloads,
            )
            downloaded_count = len(downloaded_paths)

        # 7. Ingestion Logging
        log_entry = IngestionLog(
            query=query,
            source="hybrid(arxiv+openalex)",
            retrieved_count=total_raw,
            deduplicated_count=total_deduped,
            downloaded_count=downloaded_count,
            status="success",
            details=f"Target: {target_papers}, ArXiv: {len(arxiv_results)}, OpenAlex: {len(openalex_results)}",
        )
        self.repository.log_ingestion(log_entry)

        summary = {
            "query": query,
            "raw_retrieved": total_raw,
            "deduplicated": total_deduped,
            "stored_in_db": total_deduped,
            "pdfs_downloaded": downloaded_count,
            "papers": deduped_papers,
        }
        logger.info(
            "Ingestion pipeline finished: %d raw -> %d unique -> %d PDFs",
            total_raw,
            total_deduped,
            downloaded_count,
        )
        return summary
