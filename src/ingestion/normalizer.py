import re
import unicodedata
from typing import Optional


def normalize_text(text: Optional[str]) -> str:
    """Normalize general text by standardizing unicode, whitespace and newlines."""
    if not text:
        return ""
    # Normalize unicode (NFKD)
    normalized = unicodedata.normalize("NFKD", text)
    # Replace multiple whitespace/newlines with single space
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_title(title: Optional[str]) -> str:
    """Normalize title for fuzzy matching and deduplication."""
    if not title:
        return ""
    # Lowercase, strip accents
    t = normalize_text(title).lower()
    # Remove punctuation, leaving only alphanumeric and spaces
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI to standard format (e.g., 10.1145/3616855.3635706)."""
    if not doi:
        return None
    cleaned = doi.strip().lower()
    # Strip URL prefixes
    cleaned = re.sub(r"^https?://(dx\.)?doi\.org/", "", cleaned)
    # Strip leading 'doi:'
    cleaned = re.sub(r"^doi:\s*", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned if cleaned.startswith("10.") else (cleaned or None)


def normalize_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
    """Normalize arXiv identifier to base ID (stripping versions like v1, v2)."""
    if not arxiv_id:
        return None
    cleaned = arxiv_id.strip()
    # Extract ID from URLs like http://arxiv.org/abs/2305.12345v2
    match = re.search(r"(\d{4}\.\d{4,5}|[a-zA-Z\-]+(?:\.[a-zA-Z]+)?/\d{7})", cleaned)
    if match:
        base_id = match.group(1)
        return base_id
    # Strip version suffix if present e.g. 2401.12345v1 -> 2401.12345
    cleaned = re.sub(r"v\d+$", "", cleaned)
    return cleaned.strip() or None
