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


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct plain-text abstract from OpenAlex inverted index dictionary."""
    if not inverted_index:
        return ""
    pos_word = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_word.append((pos, word))
    pos_word.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pos_word)


class OpenAlexClient:
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 30.0):
        self.email = email or settings.OPENALEX_EMAIL
        self.api_key = api_key or settings.OPENALEX_API_KEY
        self.timeout = timeout

    def search(self, query: str, max_results: int = 25) -> List[PaperRecord]:
        """Search OpenAlex works by topic/keyword."""
        headers = {"User-Agent": f"AI-Research-Assistant/0.1.0 (mailto:{self.email})"}
        params: Dict[str, Any] = {
            "search": query,
            "per-page": min(max_results, 50),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["mailto"] = self.email

        records: List[PaperRecord] = []
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(self.BASE_URL, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            for item in results:
                record = self._parse_item(item)
                if record:
                    records.append(record)

            logger.info("Retrieved %d papers from OpenAlex for query: %s", len(records), query)
        except Exception as e:
            logger.error("Error querying OpenAlex API: %s", str(e), exc_info=True)

        return records

    def _parse_item(self, item: Dict[str, Any]) -> Optional[PaperRecord]:
        raw_id = item.get("id", "")
        # OpenAlex IDs look like https://openalex.org/W123456789
        openalex_id = raw_id.split("/")[-1] if raw_id else ""
        if not openalex_id:
            return None

        title = normalize_text(item.get("title", ""))
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))

        # Authors
        authors = []
        for authorship in item.get("authorships", []):
            author_obj = authorship.get("author", {})
            name = author_obj.get("display_name")
            if name:
                authors.append(normalize_text(name))

        # DOI & arXiv ID
        doi = normalize_doi(item.get("doi"))
        arxiv_id = None
        ids_dict = item.get("ids", {})
        if "arxiv" in ids_dict:
            arxiv_id = normalize_arxiv_id(ids_dict["arxiv"])

        # Publication date / year
        pub_date = item.get("publication_date")
        if not pub_date and item.get("publication_year"):
            pub_date = f"{item['publication_year']}-01-01"

        # Venue
        primary_loc = item.get("primary_location") or {}
        source_obj = primary_loc.get("source") or {}
        venue = source_obj.get("display_name") or "OpenAlex"

        # Open Access PDF URL
        pdf_url = None
        oa_obj = item.get("open_access", {})
        if oa_obj.get("oa_url"):
            pdf_url = oa_obj.get("oa_url")
        elif primary_loc.get("pdf_url"):
            pdf_url = primary_loc.get("pdf_url")

        citation_count = item.get("cited_by_count", 0)

        paper_id = f"openalex:{openalex_id}"
        return PaperRecord(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            doi=doi,
            arxiv_id=arxiv_id,
            publication_date=pub_date,
            venue=venue,
            source="openalex",
            source_paper_id=openalex_id,
            pdf_url=pdf_url,
            local_pdf_path=None,
            citation_count=citation_count,
        )
