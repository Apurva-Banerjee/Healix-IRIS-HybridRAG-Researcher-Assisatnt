from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    authors: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    publication_date: Optional[str] = None
    venue: Optional[str] = None
    source: str = "arxiv"
    source_paper_id: str = ""
    pdf_url: Optional[str] = None
    local_pdf_path: Optional[str] = None
    citation_count: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class IngestionLog(BaseModel):
    query: str
    source: str
    retrieved_count: int = 0
    deduplicated_count: int = 0
    downloaded_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "success"
    details: Optional[str] = None
