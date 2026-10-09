import json
from typing import List, Optional
from src.database.connection import get_db_connection
from src.database.models import PaperRecord, IngestionLog


class PaperRepository:
    def __init__(self):
        self.init_db()

    def init_db(self) -> None:
        """Initialize database tables and indexes."""
        with get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    authors TEXT, -- JSON array of strings
                    doi TEXT,
                    arxiv_id TEXT,
                    publication_date TEXT,
                    venue TEXT,
                    source TEXT NOT NULL,
                    source_paper_id TEXT,
                    pdf_url TEXT,
                    local_pdf_path TEXT,
                    citation_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers (doi);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers (arxiv_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_source ON papers (source);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    source TEXT NOT NULL,
                    retrieved_count INTEGER DEFAULT 0,
                    deduplicated_count INTEGER DEFAULT 0,
                    downloaded_count INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT
                );
            """)

    def save_paper(self, paper: PaperRecord) -> None:
        """Insert or replace a paper record."""
        with get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO papers (
                    paper_id, title, abstract, authors, doi, arxiv_id,
                    publication_date, venue, source, source_paper_id,
                    pdf_url, local_pdf_path, citation_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                paper.paper_id,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                paper.doi,
                paper.arxiv_id,
                paper.publication_date,
                paper.venue,
                paper.source,
                paper.source_paper_id,
                paper.pdf_url,
                paper.local_pdf_path,
                paper.citation_count,
                paper.created_at,
            ))

    def save_papers_batch(self, papers: List[PaperRecord]) -> int:
        """Batch insert papers, returning count of inserted items."""
        count = 0
        for paper in papers:
            self.save_paper(paper)
            count += 1
        return count

    def get_paper(self, paper_id: str) -> Optional[PaperRecord]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM papers WHERE paper_id = ?;", (paper_id,)).fetchone()
            if not row:
                return None
            return self._row_to_paper(row)

    def find_by_doi(self, doi: str) -> Optional[PaperRecord]:
        if not doi:
            return None
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM papers WHERE lower(doi) = lower(?);", (doi.strip(),)).fetchone()
            if not row:
                return None
            return self._row_to_paper(row)

    def find_by_arxiv_id(self, arxiv_id: str) -> Optional[PaperRecord]:
        if not arxiv_id:
            return None
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM papers WHERE lower(arxiv_id) = lower(?);", (arxiv_id.strip(),)).fetchone()
            if not row:
                return None
            return self._row_to_paper(row)

    def update_pdf_path(self, paper_id: str, local_pdf_path: str) -> bool:
        with get_db_connection() as conn:
            cur = conn.execute(
                "UPDATE papers SET local_pdf_path = ? WHERE paper_id = ?;",
                (local_pdf_path, paper_id)
            )
            return cur.rowcount > 0

    def list_papers(self, limit: int = 100, offset: int = 0) -> List[PaperRecord]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM papers ORDER BY created_at DESC LIMIT ? OFFSET ?;",
                (limit, offset)
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]

    def count_papers(self) -> int:
        with get_db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM papers;").fetchone()
            return row["total"] if row else 0

    def log_ingestion(self, log: IngestionLog) -> None:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO ingestion_logs (
                    query, source, retrieved_count, deduplicated_count,
                    downloaded_count, timestamp, status, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                log.query,
                log.source,
                log.retrieved_count,
                log.deduplicated_count,
                log.downloaded_count,
                log.timestamp,
                log.status,
                log.details,
            ))

    def _row_to_paper(self, row) -> PaperRecord:
        authors = []
        if row["authors"]:
            try:
                authors = json.loads(row["authors"])
            except Exception:
                authors = [row["authors"]]

        return PaperRecord(
            paper_id=row["paper_id"],
            title=row["title"],
            abstract=row["abstract"] or "",
            authors=authors,
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            publication_date=row["publication_date"],
            venue=row["venue"],
            source=row["source"],
            source_paper_id=row["source_paper_id"] or "",
            pdf_url=row["pdf_url"],
            local_pdf_path=row["local_pdf_path"],
            citation_count=row["citation_count"] or 0,
            created_at=row["created_at"],
        )
