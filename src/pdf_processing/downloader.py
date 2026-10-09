import logging
import re
from pathlib import Path
from typing import List, Optional
import httpx
from src.config.settings import settings
from src.database.models import PaperRecord
from src.database.repository import PaperRepository

logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    """Convert arbitrary paper identifier into a safe filesystem filename."""
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    return clean[:120]


class PDFDownloader:
    def __init__(self, download_dir: Optional[Path] = None, timeout: float = 45.0):
        self.download_dir = download_dir or settings.RAW_PDF_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def download_paper_pdf(
        self, paper: PaperRecord, repository: Optional[PaperRepository] = None
    ) -> Optional[str]:
        """Download paper PDF and return local file path, or None if unavailable/failed."""
        if not paper.pdf_url:
            logger.debug("No PDF URL for paper: %s", paper.paper_id)
            return None

        # If already downloaded and exists on disk
        dest_filename = f"{safe_filename(paper.paper_id)}.pdf"
        dest_path = self.download_dir / dest_filename
        if dest_path.exists() and dest_path.stat().st_size > 1024:
            logger.info("PDF already exists locally for %s: %s", paper.paper_id, dest_path)
            if repository:
                repository.update_pdf_path(paper.paper_id, str(dest_path))
            return str(dest_path)

        url = paper.pdf_url
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 AI-Research-Assistant/0.1"
        }

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code != 200:
                        logger.warning("Failed downloading PDF for %s: HTTP %s from %s", paper.paper_id, resp.status_code, url)
                        return None

                    # Download to temporary file first
                    temp_path = dest_path.with_suffix(".tmp")
                    with open(temp_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=16384):
                            f.write(chunk)

            # Validate PDF signature
            with open(temp_path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF-"):
                    logger.warning("Downloaded content for %s is not a valid PDF (header=%s). Removing.", paper.paper_id, header)
                    temp_path.unlink(missing_ok=True)
                    return None

            # Move temp file to destination
            if temp_path.exists():
                temp_path.replace(dest_path)

            logger.info("Successfully downloaded PDF for %s -> %s", paper.paper_id, dest_path)
            local_path_str = str(dest_path)
            if repository:
                repository.update_pdf_path(paper.paper_id, local_path_str)
            return local_path_str

        except Exception as e:
            logger.error("Exception downloading PDF for %s (%s): %s", paper.paper_id, url, str(e))
            return None

    def download_batch(
        self, papers: List[PaperRecord], repository: Optional[PaperRepository] = None, max_count: Optional[int] = None
    ) -> List[str]:
        """Download PDFs for a list of papers up to max_count."""
        downloaded = []
        count = 0
        for paper in papers:
            if max_count and count >= max_count:
                break
            path = self.download_paper_pdf(paper, repository=repository)
            if path:
                downloaded.append(path)
                count += 1
        return downloaded
