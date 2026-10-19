import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.chunking.chunk_schema import ChunkRecord
from src.config.settings import settings
from src.pdf_processing.downloader import safe_filename

logger = logging.getLogger(__name__)


def approximate_tokens(text: str) -> int:
    """Approximate token count based on whitespace and punctuation splits (~1.3 tokens per word)."""
    words = len(text.split())
    return int(words * 1.3) + 1


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences handling standard abbreviations."""
    # Pattern for sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


class SectionAwareChunker:
    """
    Section-aware document chunker that divides academic papers into retrieval units
    while preserving section semantics, page numbers, and structural metadata.
    """

    def __init__(
        self,
        chunk_size_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ):
        self.chunk_size = chunk_size_tokens or settings.CHUNK_SIZE_TOKENS
        self.chunk_overlap = chunk_overlap_tokens or settings.CHUNK_OVERLAP_TOKENS
        self.output_dir = output_dir or settings.PROCESSED_CHUNKS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def chunk_paper_document(self, doc_data: Dict[str, Any], save: bool = True) -> List[ChunkRecord]:
        """
        Takes parsed paper JSON and returns a list of section-aware ChunkRecords.
        """
        paper_id = doc_data.get("paper_id", "unknown_paper")
        title = doc_data.get("title", "")
        sections = doc_data.get("sections", [])

        # Fallback if no sections were parsed
        if not sections and "text" in doc_data:
            sections = [{"name": "Body", "page_start": 1, "text": doc_data["text"]}]

        all_chunks: List[ChunkRecord] = []
        global_chunk_idx = 0

        for sec in sections:
            sec_name = sec.get("name", "General")
            page_start = sec.get("page_start", 1)
            sec_text = sec.get("text", "").strip()

            if not sec_text:
                continue

            # Don't chunk references or bibliography into retrieval index if desired,
            # but keep them if small. Usually references add noise to retrieval.
            if sec_name.lower() in ["references", "bibliography"]:
                logger.debug("Skipping indexing for section: %s", sec_name)
                continue

            sec_chunks = self._chunk_section_text(
                text=sec_text,
                paper_id=paper_id,
                title=title,
                section_name=sec_name,
                page_start=page_start,
                start_index=global_chunk_idx,
            )

            all_chunks.extend(sec_chunks)
            global_chunk_idx += len(sec_chunks)

        # Save chunks to disk only for non-test papers
        if save and not paper_id.startswith("test_"):
            self._save_chunks(paper_id, all_chunks)
            logger.info(
                "Created %d section-aware chunks for paper: %s", len(all_chunks), paper_id
            )
        else:
            logger.debug(
                "Created %d chunks in-memory for %s (disk save skipped)", len(all_chunks), paper_id
            )
        return all_chunks

    def _chunk_section_text(
        self,
        text: str,
        paper_id: str,
        title: str,
        section_name: str,
        page_start: int,
        start_index: int,
    ) -> List[ChunkRecord]:
        """Chunk a single section using paragraph and sliding-window sentence buffering."""
        paragraphs = text.split("\n\n")
        units: List[str] = []

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            if approximate_tokens(p_clean) > self.chunk_size:
                # If paragraph itself is larger than chunk size, split by sentences
                units.extend(split_into_sentences(p_clean))
            else:
                units.append(p_clean)

        chunks: List[ChunkRecord] = []
        current_buffer: List[str] = []
        current_token_count = 0
        chunk_idx = start_index

        for unit in units:
            unit_tokens = approximate_tokens(unit)

            if current_token_count + unit_tokens > self.chunk_size and current_buffer:
                # Create chunk from buffer
                chunk_text = "\n\n".join(current_buffer)
                safe_sec = safe_filename(section_name.lower())
                safe_pid = safe_filename(paper_id)
                chunk_id = f"{safe_pid}__{safe_sec}__c{chunk_idx}"

                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        title=title,
                        section=section_name,
                        page_number=page_start,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        token_count=current_token_count,
                        metadata={
                            "section": section_name,
                            "page": page_start,
                            "paper_id": paper_id,
                        },
                    )
                )
                chunk_idx += 1

                # Sliding window overlap: keep trailing units that fit in chunk_overlap
                overlap_buffer: List[str] = []
                overlap_tokens = 0
                for item in reversed(current_buffer):
                    item_tok = approximate_tokens(item)
                    if overlap_tokens + item_tok <= self.chunk_overlap:
                        overlap_buffer.insert(0, item)
                        overlap_tokens += item_tok
                    else:
                        break

                current_buffer = overlap_buffer
                current_token_count = overlap_tokens

            current_buffer.append(unit)
            current_token_count += unit_tokens

        # Flush final buffer
        if current_buffer:
            chunk_text = "\n\n".join(current_buffer)
            safe_sec = safe_filename(section_name.lower())
            safe_pid = safe_filename(paper_id)
            chunk_id = f"{safe_pid}__{safe_sec}__c{chunk_idx}"

            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    title=title,
                    section=section_name,
                    page_number=page_start,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    token_count=current_token_count,
                    metadata={
                        "section": section_name,
                        "page": page_start,
                        "paper_id": paper_id,
                    },
                )
            )

        return chunks

    def _save_chunks(self, paper_id: str, chunks: List[ChunkRecord]) -> None:
        """Save chunk list to JSON file in processed/chunks directory."""
        clean_name = safe_filename(paper_id)
        out_path = self.output_dir / f"{clean_name}_chunks.json"
        data = [chunk.model_dump() for chunk in chunks]
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class FixedWindowChunker:
    """
    Conventional fixed-window document chunker used as an experimental ablation baseline.
    Divides text into uniform token-sized sliding windows regardless of academic section boundaries.
    """

    def __init__(
        self,
        chunk_size_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ):
        self.chunk_size = chunk_size_tokens or settings.CHUNK_SIZE_TOKENS
        self.chunk_overlap = chunk_overlap_tokens or settings.CHUNK_OVERLAP_TOKENS
        self.output_dir = output_dir or (settings.DATA_DIR / "processed" / "ablation_fixed_chunks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def chunk_text(self, text: str, paper_id: str, title: str) -> List[ChunkRecord]:
        paragraphs = text.split("\n\n")
        units: List[str] = []
        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            if approximate_tokens(p_clean) > self.chunk_size:
                units.extend(split_into_sentences(p_clean))
            else:
                units.append(p_clean)

        chunks: List[ChunkRecord] = []
        current_buffer: List[str] = []
        current_token_count = 0
        chunk_idx = 0

        for unit in units:
            unit_tokens = approximate_tokens(unit)
            if current_token_count + unit_tokens > self.chunk_size and current_buffer:
                chunk_text = "\n\n".join(current_buffer)
                safe_pid = safe_filename(paper_id)
                chunk_id = f"{safe_pid}__fixed__c{chunk_idx}"

                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        title=title,
                        section="FixedWindow",
                        page_number=1,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        token_count=current_token_count,
                        metadata={
                            "section": "FixedWindow",
                            "page": 1,
                            "paper_id": paper_id,
                            "chunker_type": "fixed_window",
                        },
                    )
                )
                chunk_idx += 1

                overlap_buffer: List[str] = []
                overlap_tokens = 0
                for item in reversed(current_buffer):
                    item_tok = approximate_tokens(item)
                    if overlap_tokens + item_tok <= self.chunk_overlap:
                        overlap_buffer.insert(0, item)
                        overlap_tokens += item_tok
                    else:
                        break
                current_buffer = overlap_buffer
                current_token_count = overlap_tokens

            current_buffer.append(unit)
            current_token_count += unit_tokens

        if current_buffer:
            chunk_text = "\n\n".join(current_buffer)
            safe_pid = safe_filename(paper_id)
            chunk_id = f"{safe_pid}__fixed__c{chunk_idx}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    title=title,
                    section="FixedWindow",
                    page_number=1,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    token_count=current_token_count,
                    metadata={
                        "section": "FixedWindow",
                        "page": 1,
                        "paper_id": paper_id,
                        "chunker_type": "fixed_window",
                    },
                )
            )

        return chunks

    def chunk_paper_document(self, doc_data: Dict[str, Any], save: bool = False) -> List[ChunkRecord]:
        paper_id = doc_data.get("paper_id", "unknown_paper")
        title = doc_data.get("title", "")
        full_text = doc_data.get("text", "")
        if not full_text and "sections" in doc_data:
            full_text = "\n\n".join(s.get("text", "") for s in doc_data.get("sections", []))

        chunks = self.chunk_text(full_text, paper_id=paper_id, title=title)
        if save and not paper_id.startswith("test_"):
            clean_name = safe_filename(paper_id)
            out_path = self.output_dir / f"{clean_name}_fixed_chunks.json"
            data = [chunk.model_dump() for chunk in chunks]
            out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return chunks


