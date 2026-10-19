# 🔬 AI Research Scientist Assistant (ScholarRAG)

An evidence-grounded scientific research intelligence platform that bridges the gap between complex research questions and verifiable literature answers. The system systematically ingests scientific papers from scholarly repositories (arXiv, OpenAlex, Semantic Scholar), validates PDF signatures and SHA-256 checksums, parses documents into academic sections, constructs section-aware retrieval units, performs multi-mode hybrid retrieval (Dense Semantic BGE + Sparse BM25 + Reciprocal Rank Fusion), and generates publication-grade, humanized, plagiarism-free research paper drafts and answers with verified citation grounding.

---

## 🏛️ System Architecture

```text
                                USER / RESEARCHER
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
               REST API (FastAPI)             Interactive UI (Streamlit)
           POST /search, /ask, /paper/draft    5-Tab Visual Research Workstation
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                              [ Research Question ]
                                        │
                                        ▼
                      Multi-Mode Hybrid Retrieval Coordinator
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
          Dense Vector Lane                         Sparse Lexical Lane
        BAAI/bge-small-en-v1.5                           BM25Okapi
         Qdrant Vector Store                       Inverted Token Corpus
          (384 dimensions)                           (Sub-3ms Latency)
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 Reciprocal Rank                 Min-Max Scaled
                  Fusion (RRF)                   Linear Fusion
                     (k=60)                       (alpha=0.7)
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                      Top-K Ranked Scientific Evidence
                    [Paper, Canonical Section, Page, Lane]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
               Grounded Q&A Engine           Academic Paper Generator
              (Attributed Synthesis)        (Humanized & Plagiarism-Free)
                         │                             │
                         ▼                             ▼
                Citation Verification         Publication Draft (.md)
             (Support Rate, Conflicts)     (Abstract, Review, Gaps, Roadmap)
```

---

## 📥 Offline Ingestion & Corpus Integrity

```text
  arXiv Atom API ────────┐
                         ├─► Normalizer ─► Deduplicator ─► SQLite DB (metadata.db)
  OpenAlex REST API ─────┘      │                 │            │
                                ▼                 ▼            ▼
                         Fingerprinting     DOI > arXiv >    SHA-256 Checksums
                                          Title+Year+Author  & Source Logs
                                                  │
                                                  ▼
                                            PDF Downloader (%PDF- validation)
                                                  │
                                                  ▼
                                            PyMuPDF Parser (Section Extraction)
                                                  │
                                                  ▼
                                            Section-Aware Chunker (600/100 tokens)
                                                  │
                                                  ▼
                                            Qdrant Vector DB (929 points)
                                            + BM25 Inverted Corpus (929 chunks)
```

---

## 📊 Controlled Retrieval Benchmark Results

Evaluated over **30 domain-specific scientific research queries** across 6 research categories on the reconciled corpus (59 papers, 929 section-aware chunks):

| Retrieval Method | Mode Code | Precision@5 | Recall@5 | MRR | nDCG@5 | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Sparse Lexical)** | `bm25` | 0.4667 | 0.7333 | 0.5733 | 0.6007 | **2.45 ms** |
| **Dense (BGE-small-en-v1.5)** | `dense` | 0.6467 | **0.8000** | 0.7250 | **0.7386** | 227.87 ms |
| **Hybrid Weighted ($\alpha=0.7$)** | `hybrid_weighted` | **0.6533** | 0.7667 | **0.7333** | 0.7341 | 239.84 ms |
| **Hybrid RRF ($k=60$)** | `hybrid_rrf` | 0.5933 | **0.8000** | 0.6944 | 0.7132 | 230.92 ms |

Artifacts saved to:
- `artifacts/evaluation/results.csv`
- `artifacts/evaluation/results.json`
- `artifacts/evaluation/error_analysis.md`
- `data/evaluation/queries.jsonl`
- `data/evaluation/qrels.jsonl`

---

## 📁 Repository Structure

```text
├── artifacts/
│   ├── evaluation/              # Benchmark results (CSV, JSON, Error Analysis)
│   └── research_paper_draft.md  # Generated publication-ready paper draft
├── data/
│   ├── raw/
│   │   ├── metadata/            # Raw JSON feeds from scientific APIs
│   │   └── pdf/                 # Downloaded academic PDFs (verified SHA-256)
│   ├── processed/
│   │   ├── text/                # Clean, dehyphenated plain text per paper
│   │   ├── json/                # Structured documents with extracted sections
│   │   └── chunks/              # 929 section-aware retrieval units
│   └── evaluation/              # Benchmark queries & ground truth qrels
├── docs/
│   ├── PROJECT_AUDIT.md         # Full repository audit & component map
│   ├── DATA_INTEGRITY.md        # Corpus reconciliation report
│   ├── PDF_PARSING_AUDIT.md     # PDF parsing & chunk quality audit
│   ├── RETRIEVAL_DESIGN.md      # Retrieval modes, RRF equations, schemas
│   ├── EVALUATION_PROTOCOL.md   # Evaluation metrics & query taxonomy
│   ├── EXPERIMENT_RESULTS.md    # Multi-mode benchmark analysis
│   ├── GENERATION_AND_CITATIONS.md # Grounded generation & citation verification
│   ├── KNOWN_ISSUES.md          # Known boundaries & engineering limits
│   └── REPRODUCIBILITY.md       # Setup & replication guide
├── storage/
│   ├── qdrant/                  # Local embedded Qdrant vector store
│   └── metadata.db              # SQLite database (59 cataloged papers)
├── src/
│   ├── api/                     # FastAPI endpoints (/search, /ask, /paper/draft, /audit)
│   ├── chunking/                # SectionAwareChunker (600/100 tokens)
│   ├── config/                  # Settings and path configuration
│   ├── database/                # SQLite connection manager, models, repository
│   ├── embeddings/              # BAAI/bge-small-en-v1.5 & FastDense fallback
│   ├── evaluation/              # Precision, Recall, MRR, nDCG, 30-query dataset
│   ├── generation/              # AnswerGenerator, CitationVerifier, PaperGenerator
│   ├── ingestion/               # arXiv, OpenAlex, Semantic Scholar adapters
│   ├── pdf_processing/          # PyMuPDF parser, cleaner, section extractor
│   └── retrieval/               # HybridSearchEngine (dense, bm25, rrf, weighted)
├── ui/
│   └── app.py                   # Streamlit 5-tab research assistant workstation
├── scripts/
│   ├── audit_corpus.py          # Read-only audit & safe repair tool
│   ├── build_index.py           # Index chunks into Qdrant & BM25
│   ├── create_chunks.py         # Generate section-aware chunks
│   ├── evaluate.py              # Run multi-mode retrieval benchmark
│   ├── ingest.py                # CLI paper ingestion
│   ├── process_pdfs.py          # CLI PDF parsing
│   └── run_pipeline.py          # Complete pipeline runner
├── tests/                       # Complete pytest unit & regression test suite
├── requirements.txt
├── .env.example
├── pytest.ini
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/Apurva-Banerjee/Healix-IRIS-HybridRAG-Researcher-Assisatnt.git
cd Healix-IRIS-HybridRAG-Researcher-Assisatnt

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2. Verify Corpus & Run Tests
```bash
# Verify 100% parity across DB, PDFs, chunks, and Qdrant
python scripts/audit_corpus.py

# Run test suite
python -m pytest
```

### 3. Launch Applications
```bash
# Streamlit Interactive Research Assistant
streamlit run ui/app.py

# FastAPI REST API Server
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

---

## 📜 Citation & License

This project is licensed under the Apache 2.0 License.
If utilizing this framework for academic research, please cite:
```bibtex
@software{scholar_rag_2026,
  author = {Apurva Banerjee},
  title = {ScholarRAG: Section-Aware Hybrid Retrieval-Augmented Generation for Scientific Literature Synthesis},
  year = {2026},
  url = {https://github.com/Apurva-Banerjee/Healix-IRIS-HybridRAG-Researcher-Assisatnt}
}
```
