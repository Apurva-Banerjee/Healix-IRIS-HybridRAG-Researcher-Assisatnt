import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pymupdf
from src.config.settings import settings
from src.pdf_processing.cleaner import clean_pdf_text
from src.pdf_processing.downloader import safe_filename
from src.pdf_processing.section_extractor import SectionExtractor

logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(
        self,
        text_dir: Optional[Path] = None,
        json_dir: Optional[Path] = None,
        section_extractor: Optional[SectionExtractor] = None,
    ):
        self.text_dir = text_dir or settings.PROCESSED_TEXT_DIR
        self.json_dir = json_dir or settings.PROCESSED_JSON_DIR
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.section_extractor = section_extractor or SectionExtractor()

    def parse_pdf(
        self,
        pdf_path: str,
        paper_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse a local PDF file into clean text and structured sections."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        paper_id = paper_id or path.stem
        title = title or path.stem

        pages_text: List[Tuple[int, str]] = []
        total_pages = 0

        try:
            doc = pymupdf.open(str(path))
            total_pages = len(doc)

            for page_index in range(total_pages):
                page = doc[page_index]
                page_num = page_index + 1
                raw_text = page.get_text("text")
                cleaned = clean_pdf_text(raw_text)
                if cleaned:
                    pages_text.append((page_num, cleaned))

            doc.close()
        except Exception as e:
            logger.error("Failed to parse PDF %s with PyMuPDF: %s", pdf_path, str(e))
            raise

        # Extract structured sections
        sections = self.section_extractor.extract_sections(pages_text)

        document_data = {
            "paper_id": paper_id,
            "title": title,
            "pdf_path": str(path),
            "total_pages": total_pages,
            "sections": sections,
        }

        # Save processed text
        clean_name = safe_filename(paper_id)
        text_file = self.text_dir / f"{clean_name}.txt"
        full_text = "\n\n".join(
            f"--- PAGE {page_num} ---\n{text}" for page_num, text in pages_text
        )
        text_file.write_text(full_text, encoding="utf-8")

        # Save structured JSON
        json_file = self.json_dir / f"{clean_name}.json"
        json_file.write_text(json.dumps(document_data, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(
            "Successfully parsed PDF: %s (%d pages, %d sections)",
            paper_id,
            total_pages,
            len(sections),
        )
        return document_data
