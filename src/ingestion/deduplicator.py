import logging
from typing import Dict, List, Optional
from src.database.models import PaperRecord
from src.ingestion.normalizer import normalize_title

logger = logging.getLogger(__name__)


def build_title_key(title: str, pub_date: Optional[str], authors: List[str]) -> str:
    """Build a composite key: normalized_title + year + first_author_last_name."""
    norm_t = normalize_title(title)
    year = pub_date[:4] if pub_date and len(pub_date) >= 4 else "unknown"
    first_author_last = ""
    if authors:
        first_author = authors[0].strip()
        parts = first_author.split()
        if parts:
            first_author_last = parts[-1].lower()

    return f"{norm_t}::{year}::{first_author_last}"


def merge_paper_records(primary: PaperRecord, secondary: PaperRecord) -> PaperRecord:
    """Merge secondary record data into primary record to create the most complete record."""
    # Preferred ID: preserve arXiv or DOI priority
    doi = primary.doi or secondary.doi
    arxiv_id = primary.arxiv_id or secondary.arxiv_id

    # Title: prefer longer/more descriptive if normalized match
    title = primary.title if len(primary.title) >= len(secondary.title) else secondary.title
    abstract = primary.abstract if len(primary.abstract) >= len(secondary.abstract) else secondary.abstract
    authors = primary.authors if len(primary.authors) >= len(secondary.authors) else secondary.authors

    pdf_url = primary.pdf_url or secondary.pdf_url
    local_pdf_path = primary.local_pdf_path or secondary.local_pdf_path
    citation_count = max(primary.citation_count, secondary.citation_count)
    pub_date = primary.publication_date or secondary.publication_date
    venue = primary.venue if primary.venue and primary.venue != "arXiv" else (secondary.venue or primary.venue)

    return PaperRecord(
        paper_id=primary.paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        doi=doi,
        arxiv_id=arxiv_id,
        publication_date=pub_date,
        venue=venue,
        source=primary.source,
        source_paper_id=primary.source_paper_id,
        pdf_url=pdf_url,
        local_pdf_path=local_pdf_path,
        citation_count=citation_count,
        created_at=primary.created_at,
    )


class PaperDeduplicator:
    """
    Deduplicates paper records with strict priority:
    1. DOI
    2. arXiv ID
    3. normalized title + year + first author last name
    """

    def deduplicate(self, papers: List[PaperRecord]) -> List[PaperRecord]:
        doi_map: Dict[str, PaperRecord] = {}
        arxiv_map: Dict[str, PaperRecord] = {}
        title_map: Dict[str, PaperRecord] = {}

        unique_papers: List[PaperRecord] = []

        for paper in papers:
            matched_record: Optional[PaperRecord] = None

            # 1. Match by DOI
            if paper.doi and paper.doi in doi_map:
                matched_record = doi_map[paper.doi]

            # 2. Match by arXiv ID
            elif paper.arxiv_id and paper.arxiv_id in arxiv_map:
                matched_record = arxiv_map[paper.arxiv_id]

            # 3. Match by Title + Year + Author
            else:
                title_key = build_title_key(paper.title, paper.publication_date, paper.authors)
                if title_key in title_map:
                    matched_record = title_map[title_key]

            if matched_record:
                # Merge records
                merged = merge_paper_records(matched_record, paper)
                # Update maps pointing to merged
                if merged.doi:
                    doi_map[merged.doi] = merged
                if merged.arxiv_id:
                    arxiv_map[merged.arxiv_id] = merged
                t_key = build_title_key(merged.title, merged.publication_date, merged.authors)
                title_map[t_key] = merged
                # Replace in unique list
                for idx, p in enumerate(unique_papers):
                    if p.paper_id == matched_record.paper_id:
                        unique_papers[idx] = merged
                        break
            else:
                unique_papers.append(paper)
                if paper.doi:
                    doi_map[paper.doi] = paper
                if paper.arxiv_id:
                    arxiv_map[paper.arxiv_id] = paper
                t_key = build_title_key(paper.title, paper.publication_date, paper.authors)
                title_map[t_key] = paper

        logger.info(
            "Deduplication complete: %d input records -> %d unique records",
            len(papers),
            len(unique_papers),
        )
        return unique_papers
