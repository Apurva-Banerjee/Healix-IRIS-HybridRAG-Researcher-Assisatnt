import re
from typing import Dict, List, Optional, Tuple


# Common academic headings and patterns
STANDARD_SECTIONS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "method",
    "approach",
    "architecture",
    "system design",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "analysis",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "bibliography",
    "appendix",
]

# Regex to detect section headings (e.g., "1. Introduction", "IV. METHOD", "3.2 Experiments", "Abstract")
HEADING_REGEX = re.compile(
    r"^(?:"
    r"(?:\d+\.?(?:\d+)*\s+|[IVXLCDM]+\.\s+)?"  # Optional numbering (1., 1.1, II.)
    r"(abstract|introduction|related work|background|methodology|methods|method|system architecture|model|approach|experiments|experimental setup|results|discussion|analysis|limitations|conclusion|conclusions|future work|references|bibliography|acknowledgments|appendix)"
    r")(?:\s*:|\s*[-–—]|\s*$)",
    re.IGNORECASE,
)


class SectionExtractor:
    """Extracts and segments academic paper text into named canonical sections."""

    def extract_sections(self, pages_text: List[Tuple[int, str]]) -> List[Dict[str, any]]:
        """
        Segment text across pages into sections.
        pages_text: list of (page_number, text)
        returns list of dicts: {"name": str, "page_start": int, "text": str}
        """
        sections: List[Dict[str, any]] = []
        current_section = "front_matter"
        current_page = 1
        current_buffer: List[str] = []

        for page_num, page_content in pages_text:
            lines = page_content.splitlines()
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # Check if this line looks like a major section heading
                # Headings are typically short (< 80 chars)
                if len(stripped) <= 80:
                    match = HEADING_REGEX.match(stripped)
                    if match:
                        heading_keyword = match.group(1).lower().strip()
                        # Canonicalize heading name
                        canonical_name = self._canonicalize_heading(heading_keyword)

                        # Flush current buffer to current section
                        text_block = "\n".join(current_buffer).strip()
                        if text_block:
                            sections.append({
                                "name": current_section,
                                "page_start": current_page,
                                "text": text_block,
                            })

                        # Start new section
                        current_section = canonical_name
                        current_page = page_num
                        current_buffer = []
                        continue

                current_buffer.append(stripped)

        # Flush remaining buffer
        final_block = "\n".join(current_buffer).strip()
        if final_block:
            sections.append({
                "name": current_section,
                "page_start": current_page,
                "text": final_block,
            })

        # If only front_matter was found or no sections detected, return full paper as main section
        if not sections or (len(sections) == 1 and sections[0]["name"] == "front_matter"):
            full_text = "\n\n".join(text for _, text in pages_text).strip()
            return [{"name": "full_text", "page_start": 1, "text": full_text}]

        return sections

    def _canonicalize_heading(self, heading: str) -> str:
        h = heading.lower()
        if "abstract" in h:
            return "Abstract"
        elif "intro" in h:
            return "Introduction"
        elif "related" in h or "background" in h:
            return "Related Work"
        elif any(k in h for k in ["method", "approach", "model", "architecture", "system design"]):
            return "Methodology"
        elif "experiment" in h or "setup" in h:
            return "Experiments"
        elif "result" in h:
            return "Results"
        elif "discuss" in h or "analysis" in h:
            return "Discussion"
        elif "limitation" in h:
            return "Limitations"
        elif "conclusion" in h or "future work" in h:
            return "Conclusion"
        elif "reference" in h or "bibliography" in h:
            return "References"
        else:
            return heading.title()
