import logging
from typing import Any, Dict, List, Optional
import httpx
from src.config.settings import settings
from src.database.models import PaperRecord
from src.ingestion.normalizer import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_text,
)

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self.api_key = api_key or settings.SEMANTIC_SCHOLAR_API_KEY
        self.timeout = timeout

    def search(self, query: str, max_results: int = 20) -> List[PaperRecord]:
        """Search Semantic Scholar API for papers."""
        headers = {"User-Agent": "AI-Research-Assistant/0.1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "query": query,
            "limit": min(max_results, 50),
            "fields": "paperId,title,abstract,authors,year,publicationDate,venue,externalIds,openAccessPdf,citationCount",
        }

        records: List[PaperRecord] = []
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(self.BASE_URL, params=params, headers=headers)
                if resp.status_code == 429:
                    logger.warning("Semantic Scholar rate limit reached, skipping.")
                    return records
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("data", []):
                record = self._parse_item(item)
                if record:
                    records.append(record)

            logger.info("Retrieved %d papers from Semantic Scholar", len(records))
        except Exception as e:
            logger.warning("Optional Semantic Scholar search encountered: %s", str(e))

        return records

    def _parse_item(self, item: Dict[str, Any]) -> Optional[PaperRecord]:
        paper_id_raw = item.get("paperId")
        if not paper_id_raw:
            return None

        title = normalize_text(item.get("title", ""))
        abstract = normalize_text(item.get("abstract", ""))

        authors = [normalize_text(a.get("name", "")) for a in item.get("authors", []) if a.get("name")]

        external_ids = item.get("externalIds", {}) or {}
        doi = normalize_doi(external_ids.get("DOI"))
        arxiv_id = normalize_arxiv_id(external_ids.get("ArXiv"))

        pub_date = item.get("publicationDate")
        if not pub_date and item.get("year"):
            pub_date = f"{item['year']}-01-01"

        venue = item.get("venue") or "Semantic Scholar"

        pdf_info = item.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url")

        return PaperRecord(
            paper_id=f"s2:{paper_id_raw}",
            title=title,
            abstract=abstract,
            authors=authors,
            doi=doi,
            arxiv_id=arxiv_id,
            publication_date=pub_date,
            venue=venue,
            source="semantic_scholar",
            source_paper_id=paper_id_raw,
            pdf_url=pdf_url,
            local_pdf_path=None,
            citation_count=item.get("citationCount") or 0,
        )
