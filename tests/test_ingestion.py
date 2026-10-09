import pytest
from src.database.models import PaperRecord
from src.ingestion.deduplicator import PaperDeduplicator
from src.ingestion.normalizer import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_text,
    normalize_title,
)
from src.ingestion.openalex_client import reconstruct_abstract


def test_normalizer_text():
    raw = "  This   is a \n test \t string  "
    assert normalize_text(raw) == "This is a test string"


def test_normalizer_title():
    raw = "GraphRAG: Unlocking LLM Discovery on Narrative Networks!"
    norm = normalize_title(raw)
    assert norm == "graphrag unlocking llm discovery on narrative networks"


def test_normalizer_doi():
    doi1 = "https://doi.org/10.1145/3616855.3635706"
    doi2 = "10.1145/3616855.3635706"
    doi3 = "http://dx.doi.org/10.1145/3616855.3635706"
    assert normalize_doi(doi1) == "10.1145/3616855.3635706"
    assert normalize_doi(doi2) == "10.1145/3616855.3635706"
    assert normalize_doi(doi3) == "10.1145/3616855.3635706"


def test_normalizer_arxiv_id():
    id1 = "http://arxiv.org/abs/2404.16130v1"
    id2 = "2404.16130v2"
    id3 = "2404.16130"
    assert normalize_arxiv_id(id1) == "2404.16130"
    assert normalize_arxiv_id(id2) == "2404.16130"
    assert normalize_arxiv_id(id3) == "2404.16130"


def test_reconstruct_abstract():
    inverted = {
        "GraphRAG": [0],
        "combines": [1],
        "knowledge": [2],
        "graphs": [3],
        "with": [4],
        "LLMs.": [5],
    }
    expected = "GraphRAG combines knowledge graphs with LLMs."
    assert reconstruct_abstract(inverted) == expected


def test_deduplicator_priority_doi():
    p1 = PaperRecord(
        paper_id="arxiv:2404.16130",
        title="GraphRAG Paper",
        doi="10.1234/test.doi",
        arxiv_id="2404.16130",
        authors=["Edge, Darren"],
        source="arxiv",
        citation_count=10,
    )
    p2 = PaperRecord(
        paper_id="openalex:W99999",
        title="GraphRAG Paper Full Version",
        doi="10.1234/test.doi",
        authors=["Edge, Darren"],
        source="openalex",
        citation_count=45,
    )

    dedup = PaperDeduplicator()
    unique = dedup.deduplicate([p1, p2])

    assert len(unique) == 1
    # Merged record should retain max citations and combined IDs
    assert unique[0].citation_count == 45
    assert unique[0].arxiv_id == "2404.16130"
    assert unique[0].doi == "10.1234/test.doi"
