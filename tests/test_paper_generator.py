import pytest
from pathlib import Path
from src.generation.paper_generator import AcademicPaperGenerator


def test_academic_paper_generator(tmp_path: Path):
    target_file = tmp_path / "test_paper_draft.md"
    generator = AcademicPaperGenerator()

    result = generator.generate_full_paper_draft(
        topic="Evaluating Section-Aware Hybrid Retrieval for Scientific Question Answering",
        output_path=target_file,
    )

    assert result["document_path"] == str(target_file)
    assert target_file.exists()
    assert result["word_count"] > 300
    assert result["cited_papers_count"] > 0

    content = target_file.read_text(encoding="utf-8")
    assert "## Abstract" in content
    assert "## 1. Introduction" in content
    assert "## 2. Architectural Comparison" in content
    assert "## 3. Empirical Synthesis" in content
    assert "## 4. Critical Bottlenecks & Identified Research Gaps" in content
    assert "## 5. Proposed Experimental Protocol" in content
    assert "## References" in content
