import tempfile
from pathlib import Path
import pytest
from scripts.audit_corpus import compute_sha256, is_valid_pdf, CorpusAuditor


def test_compute_sha256(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("ScholarRAG integrity verification text", encoding="utf-8")
    hash_val = compute_sha256(sample_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64


def test_is_valid_pdf(tmp_path: Path):
    valid_pdf = tmp_path / "valid.pdf"
    valid_pdf.write_bytes(b"%PDF-1.4\n%test content with valid header\n" + b"0" * 150)
    assert is_valid_pdf(valid_pdf) is True

    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_bytes(b"NOT_A_PDF_CONTENT")
    assert is_valid_pdf(invalid_pdf) is False

    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    assert is_valid_pdf(empty_pdf) is False


def test_corpus_auditor_read():
    auditor = CorpusAuditor()
    report = auditor.audit()
    assert report["db_papers_count"] >= 38
    assert report["disk_pdfs_count"] >= 26
    assert report["disk_corrupt_pdfs"] == 0
    assert report["total_chunk_units"] == 929
