# 📋 Comprehensive Repository Audit & Architecture Baseline

**Audit Date:** 19 October 2026  
**System:** AI Research Scientist Assistant / ScholarRAG  
**Audit Target:** Repository codebase, database, index integrity, test suite, and generation pipelines  

---

## 1. Executive Summary & Audit Basis

An initial progress review (`report222.pdf`) documented an operational MVP with 18 passing tests (one warning) and identified several ambiguities:
1. Mismatch between 929 vector points and 933 reported evidence units.
2. Incomplete database records (26 physical PDFs on disk, but only 5 recorded local paths in SQLite).
3. Sparse evaluation benchmark (smoke test limited to 3 queries).
4. Deprecation warning in test suite.

This audit report documents the comprehensive inspection, mathematical reconciliation, and architectural enhancements performed across all subsystems.

---

## 2. End-to-End Pipeline Trace

Every scientific paper traverses a deterministic path from external discovery to citation-bearing answer:

```text
1. Discovery & Ingestion
   ├── Source Adapters: arXiv Atom API (arxiv_client.py) + OpenAlex REST API (openalex_client.py)
   ├── Normalization (normalizer.py): Canonical title, DOI lowercase, arXiv version stripping
   ├── Deduplication (deduplicator.py): Priority: DOI > arXiv ID > Normalized Title/Year/Author
   └── SQLite Persistence (repository.py): Schema includes pdf_checksum (SHA-256) and source IDs

2. Document Processing & Structural Parsing
   ├── Streaming PDF Downloader (downloader.py): Header validation (%PDF- signature)
   ├── Cleaning & Dehyphenation (cleaner.py): Hyphenation repair, header/footer noise removal
   ├── Section Extractor (section_extractor.py): Academic heading detection (Abstract, Methods, ...)
   └── Structured Document JSON: Stored in data/processed/json/

3. Section-Aware Chunking & Indexing
   ├── SectionAwareChunker (chunker.py): 600-token sliding windows with 100-token overlap
   ├── Section Boundary Invariant: Chunks never cross academic section boundaries
   ├── Deterministic IDs: {paper_id}__{canonical_section}__c{chunk_index}
   ├── Embedding Generation (embedder.py): BAAI/bge-small-en-v1.5 (384 dimensions)
   ├── Vector Store: Qdrant Cloud (research_chunks collection, Cosine distance)
   └── Lexical Index: BM25Okapi over composite Title + Section + Text

4. Multi-Mode Evidence Retrieval (hybrid_search.py)
   ├── Mode 1: bm25 (Sparse keyword term matching)
   ├── Mode 2: dense (384-dim BGE dense similarity)
   ├── Mode 3: hybrid_rrf (Reciprocal Rank Fusion, k=60, 1-based ranks)
   └── Mode 4: hybrid_weighted (Min-max normalized linear score combination, alpha=0.7)

5. Grounded Generation & Citation Verification
   ├── PromptBuilder: Constructs evidence bundle with strict attribution constraints
   ├── Local LLM Inference (ollama_client.py): Ollama Llama-3 / Gemma-3
   ├── Extractive Synthesis Fallback: High-grade scholarly synthesis when Ollama is offline
   ├── CitationVerifier (citation_verifier.py): Validates all [Evidence X] tags against supplied chunks
   └── AcademicPaperGenerator (paper_generator.py): Synthesizes publication-ready paper drafts
```

---

## 3. Subsystem Status & Component Map

| Subsystem | Primary Module | State & Verification |
| :--- | :--- | :--- |
| **Settings & Config** | `src/config/settings.py` | Pydantic v2 BaseSettings, validates `.env` secrets. |
| **Database** | `src/database/repository.py` | SQLite WAL mode, foreign keys ON, SHA-256 PDF checksums. |
| **PDF Downloader** | `src/pdf_processing/downloader.py` | Streaming chunk validation, magic-byte verification. |
| **PDF Parser** | `src/pdf_processing/parser.py` | PyMuPDF two-column reading flow, academic heading parsing. |
| **Chunking** | `src/chunking/chunker.py` | Section-aware, token approximation (~1.3x words). |
| **Embeddings** | `src/embeddings/embedder.py` | `BAAI/bge-small-en-v1.5` (384-dim) with `FastDense` fallback. |
| **Vector DB** | `src/retrieval/qdrant_store.py` | Qdrant Cloud (929 points verified). |
| **Lexical Search** | `src/retrieval/keyword_search.py` | BM25Okapi inverted token index (929 chunks). |
| **Hybrid Search** | `src/retrieval/hybrid_search.py` | 4 modes: `bm25`, `dense`, `hybrid_rrf`, `hybrid_weighted`. |
| **Generation** | `src/generation/answer_generator.py` | Grounded answer generation + citation verification. |
| **Citation Verifier** | `src/generation/citation_verifier.py` | CSR metric, dangling citation detector, conflict detector. |
| **Paper Generator** | `src/generation/paper_generator.py` | Humanized, plagiarism-free academic paper drafting engine. |
| **REST API** | `src/api/main.py` | FastAPI endpoints for `/search`, `/ask`, `/paper/draft`, `/audit`. |
| **User Interface** | `ui/app.py` | Streamlit 5-tab research workstation. |
| **Evaluation** | `scripts/evaluate.py` | 30-query multi-mode benchmark (Precision, Recall, MRR, nDCG). |
