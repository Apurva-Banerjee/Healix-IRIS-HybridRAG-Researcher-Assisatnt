from src.chunking.chunker import SectionAwareChunker, approximate_tokens


def test_approximate_tokens():
    text = "This is a simple sentence with seven words."
    tokens = approximate_tokens(text)
    assert tokens > 5
    assert tokens < 20


def test_section_aware_chunking():
    doc_data = {
        "paper_id": "test_paper_01",
        "title": "Test Paper on GraphRAG",
        "sections": [
            {
                "name": "Abstract",
                "page_start": 1,
                "text": "GraphRAG creates hierarchical graph structures to assist question answering over complex corporate corpora."
            },
            {
                "name": "Methodology",
                "page_start": 2,
                "text": "First, the system extracts entities and relationships from text chunks.\n\n" * 10
            },
            {
                "name": "Limitations",
                "page_start": 5,
                "text": "The primary limitation is the high token cost incurred during extensive graph generation and LLM extraction passes."
            }
        ]
    }

    chunker = SectionAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=20)
    chunks = chunker.chunk_paper_document(doc_data)

    assert len(chunks) >= 3
    # Check that sections are properly tagged on each chunk
    sections = {c.section for c in chunks}
    assert "Abstract" in sections
    assert "Methodology" in sections
    assert "Limitations" in sections

    for c in chunks:
        assert c.paper_id == "test_paper_01"
        assert c.title == "Test Paper on GraphRAG"
        assert c.token_count > 0
        assert c.chunk_id.startswith("test_paper_01__")
