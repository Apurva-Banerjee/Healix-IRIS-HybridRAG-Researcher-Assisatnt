import logging
import time
import xml.etree.ElementTree as ET
from typing import List, Optional
import httpx
from src.database.models import PaperRecord
from src.ingestion.normalizer import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_text,
)

logger = logging.getLogger(__name__)

# Atom XML namespaces for arXiv API
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivClient:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: float = 30.0, rate_limit_delay: float = 3.0):
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

    def _respect_rate_limit(self) -> None:
        """arXiv requests should be spaced by ~3 seconds according to arXiv API guidance."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 25,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> List[PaperRecord]:
        """Search arXiv and return normalized PaperRecord models."""
        self._respect_rate_limit()

        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        papers: List[PaperRecord] = []
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                headers = {"User-Agent": "AI-Research-Scientist-Assistant/0.1.0"}
                resp = client.get(self.BASE_URL, params=params, headers=headers)
                resp.raise_for_status()

            papers = self._parse_feed(resp.text)
            logger.info("Retrieved %d papers from arXiv for query: %s", len(papers), query)
        except Exception as e:
            logger.error("Error querying arXiv API: %s", str(e), exc_info=True)

        return papers

    def _parse_feed(self, xml_text: str) -> List[PaperRecord]:
        root = ET.fromstring(xml_text)
        records: List[PaperRecord] = []

        for entry in root.findall("atom:entry", ATOM_NS):
            id_elem = entry.find("atom:id", ATOM_NS)
            if id_elem is None or not id_elem.text:
                continue

            raw_id = id_elem.text.strip()
            arxiv_id = normalize_arxiv_id(raw_id)
            if not arxiv_id:
                continue

            title_elem = entry.find("atom:title", ATOM_NS)
            title = normalize_text(title_elem.text) if title_elem is not None else ""

            summary_elem = entry.find("atom:summary", ATOM_NS)
            abstract = normalize_text(summary_elem.text) if summary_elem is not None else ""

            authors = []
            for author in entry.findall("atom:author", ATOM_NS):
                name_elem = author.find("atom:name", ATOM_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(normalize_text(name_elem.text))

            published_elem = entry.find("atom:published", ATOM_NS)
            pub_date = published_elem.text.strip()[:10] if (published_elem is not None and published_elem.text) else None

            # DOI if present
            doi_elem = entry.find("arxiv:doi", ATOM_NS)
            doi = normalize_doi(doi_elem.text) if (doi_elem is not None and doi_elem.text) else None

            # Primary Category / Venue
            primary_cat_elem = entry.find("arxiv:primary_category", ATOM_NS)
            venue = primary_cat_elem.get("term") if primary_cat_elem is not None else "arXiv"

            # PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            for link in entry.findall("atom:link", ATOM_NS):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", pdf_url)
                    break

            paper_id = f"arxiv:{arxiv_id}"
            records.append(
                PaperRecord(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    publication_date=pub_date,
                    venue=venue,
                    source="arxiv",
                    source_paper_id=arxiv_id,
                    pdf_url=pdf_url,
                    local_pdf_path=None,
                    citation_count=0,
                )
            )

        return records
