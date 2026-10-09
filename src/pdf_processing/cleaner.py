import re
from typing import List


def fix_hyphenation(text: str) -> str:
    """Join words broken across line breaks with a hyphen (e.g., 'transfor-\\nmer' -> 'transformer')."""
    return re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)


def remove_running_headers_and_footers(lines: List[str]) -> List[str]:
    """Filter out common academic preprint headers, footers, and isolated page numbers."""
    cleaned_lines = []
    header_footer_patterns = [
        re.compile(r"^arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[[a-zA-Z\.\-]+\]", re.IGNORECASE),
        re.compile(r"^under\s+review\s+as\s+a\s+conference\s+paper", re.IGNORECASE),
        re.compile(r"^preprint\.\s*under\s+review", re.IGNORECASE),
        re.compile(r"^proceedings\s+of\s+the\s+", re.IGNORECASE),
        re.compile(r"^\d+\s*$", re.IGNORECASE),  # Solitary page number
        re.compile(r"^page\s+\d+\s*(of\s+\d+)?$", re.IGNORECASE),
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_header_footer = any(pattern.search(stripped) for pattern in header_footer_patterns)
        if not is_header_footer:
            cleaned_lines.append(line)

    return cleaned_lines


def clean_pdf_text(raw_text: str) -> str:
    """Clean and normalize raw extracted PDF text."""
    if not raw_text:
        return ""

    # Fix ligatures
    text = raw_text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")

    # Fix line-broken hyphenated words
    text = fix_hyphenation(text)

    # Process line by line to remove header/footer noise
    lines = text.splitlines()
    filtered_lines = remove_running_headers_and_footers(lines)

    text = "\n".join(filtered_lines)

    # Collapse multiple consecutive newlines to at most 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces or tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
