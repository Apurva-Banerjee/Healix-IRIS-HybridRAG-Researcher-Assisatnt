from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    section: str = "General"
    page_number: int = 1
    chunk_index: int = 0
    text: str
    source: str = "arxiv"
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
