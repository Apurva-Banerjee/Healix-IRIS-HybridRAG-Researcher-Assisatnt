from src.pdf_processing.cleaner import clean_pdf_text, fix_hyphenation
from src.pdf_processing.section_extractor import SectionExtractor


def test_fix_hyphenation():
    raw = "This is a transfor-\nmer based graph archi-\ntecture."
    cleaned = fix_hyphenation(raw)
    assert "transformer" in cleaned
    assert "architecture" in cleaned


def test_clean_pdf_text():
    raw = "arXiv:2401.12345v1 [cs.AI] 20 Jan 2024\n12\nActual scientific text with ligatures ﬁ and ﬂ."
    cleaned = clean_pdf_text(raw)
    assert "arXiv:2401" not in cleaned
    assert "fi and fl" in cleaned


def test_section_extractor():
    pages = [
        (1, "Title of the Paper\n\nAbstract\nThis paper proposes a new method.\n\n1. Introduction\nRecent advancements have shown..."),
        (2, "3. Methodology\nWe construct a graph representation.\n\n4. Results\nThe accuracy is 94%."),
        (3, "5. Limitations\nThe method is computationally expensive.\n\nReferences\n[1] Smith et al."),
    ]

    extractor = SectionExtractor()
    sections = extractor.extract_sections(pages)

    section_names = [s["name"] for s in sections]
    assert "Abstract" in section_names
    assert "Introduction" in section_names
    assert "Methodology" in section_names
    assert "Results" in section_names
    assert "Limitations" in section_names
