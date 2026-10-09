# 🔬 AI Research Scientist Assistant (MVP)

An evidence-grounded scientific research intelligence platform that bridges the gap between research questions and verifiable literature answers. The system systematically searches scientific repositories (arXiv, OpenAlex), normalizes and deduplicates metadata, parses PDFs into academic sections, creates semantic section-aware chunks, indexes vectors into Qdrant, performs hybrid retrieval (Dense Semantic + BM25 Keyword), and generates grounded answers citing specific scientific papers using local LLMs (Ollama).

---

## 🏛️ System Architecture

```text
                           USER / RESEARCHER
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
          REST API (FastAPI)             Interactive UI (Streamlit)
          POST /search, /ask             Visual Q&A & Corpus Browser
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                         [ Research Query ]
                                   │
                                   ▼
                     Hybrid Retrieval Coordinator
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
     Dense Vector Search                      Sparse Keyword Search
   BAAI/bge-small-en-v1.5 / BGE-M3                  BM25Okapi
         Qdrant Vector DB                        Term Matching
              │                                         │
              └────────────────────┬────────────────────┘
                                   ▼
                     Score Fusion & Deduplication
                     (Linear α=0.7 or RRF Rank Fusion)
                                   │
                                   ▼
                    Top-K Ranked Scientific Evidence
                    [Paper, Canonical Section, Page, Score]
                                   │
                                   ▼
                       Research Prompt Builder
                     (Strict Evidence Attribution)
                                   │
                                   ▼
                        Local LLM Inference
                         Ollama (Llama-3 / Gemma-3)
                                   │
                                   ▼
                    Evidence-Grounded Research Answer
                       + Supporting Paper Citations
```

---

## 📥 Offline Ingestion & Indexing Pipeline

```text
  arXiv Atom API ────────┐
                         ├─► Normalizer ─► Deduplicator ─► SQLite DB (metadata.db)
  OpenAlex REST API ─────┘      │                 │
                                ▼                 ▼
                         Title/DOI/arXiv    DOI > arXiv >
                          Fingerprinting     Title+Year+Author
                                                  │
                                                  ▼
                                            PDF Downloader (data/raw/pdf/)
                                                  │
                                                  ▼
                                            PyMuPDF Parser (data/processed/text/)
                                                  │
                                                  ▼
                                            Section Extractor (Abstract, Methods, ...)
                                                  │
                                                  ▼
                                            SectionAwareChunker (data/processed/chunks/)
                                                  │
                                                  ▼
                                            Embedder (SentenceTransformers / BGE)
                                                  │
                                                  ▼
                                            Qdrant Vector DB (Cosine Similarity)
```

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── metadata/            # Raw JSON feeds from scientific APIs
│   │   └── pdf/                 # Downloaded open-access academic PDFs
│   ├── processed/
│   │   ├── text/                # Clean, dehyphenated plain text per paper
│   │   ├── json/                # Structured documents with extracted sections
│   │   └── chunks/              # Section-aware retrieval units
│   └── evaluation/              # Benchmark ground-truth datasets
├── storage/
│   ├── qdrant/                  # Local embedded Qdrant vector database
│   └── metadata.db              # SQLite relational paper metadata & ingestion logs
├── src/
│   ├── config/
│   │   └── settings.py          # Pydantic v2 application configuration & paths
│   ├── ingestion/
│   │   ├── arxiv_client.py      # arXiv Atom API client with rate limiting
│   │   ├── openalex_client.py   # OpenAlex REST client with abstract reconstruction
│   │   ├── semantic_scholar_client.py # Optional Semantic Scholar enrichment
│   │   ├── normalizer.py        # Text, title, DOI, and arXiv ID canonicalization
│   │   ├── deduplicator.py      # DOI > arXiv > Title/Year/Author priority deduplication
│   │   └── ingestion_pipeline.py # End-to-end ingestion pipeline
│   ├── database/
│   │   ├── models.py            # Pydantic PaperRecord and IngestionLog models
│   │   ├── connection.py        # SQLite WAL-mode connection manager
│   │   └── repository.py        # Relational CRUD operations and logging
│   ├── pdf_processing/
│   │   ├── downloader.py        # Streaming downloader with PDF signature validation
│   │   ├── cleaner.py           # Dehyphenation and header/footer noise removal
│   │   ├── section_extractor.py # Academic heading detection (Abstract, Methods, etc.)
│   │   └── parser.py            # PyMuPDF document parser
│   ├── chunking/
│   │   ├── chunk_schema.py      # ChunkRecord schema with section metadata
│   │   └── chunker.py           # Paragraph/sentence sliding window chunker
│   ├── embeddings/
│   │   ├── model.py             # SentenceTransformers BGE model & FastDense fallback
│   │   └── embedder.py          # Batch embedding generation service
│   ├── retrieval/
│   │   ├── qdrant_store.py      # Qdrant client (embedded disk or remote server)
│   │   ├── semantic_search.py   # Dense vector similarity retriever
│   │   ├── keyword_search.py    # BM25Okapi keyword retriever
│   │   └── hybrid_search.py     # Linear score fusion and Reciprocal Rank Fusion (RRF)
│   ├── generation/
│   │   ├── ollama_client.py     # Local Ollama LLM client with auto-model resolution
│   │   ├── prompt_builder.py    # Scientific prompt construction with citation rules
│   │   └── answer_generator.py  # Grounded synthesis and provenance extraction
│   ├── evaluation/
│   │   ├── retrieval_metrics.py # Precision@K, Recall@K, MRR, nDCG@K
│   │   └── evaluation_dataset.py# Benchmark research query dataset
│   └── api/
│       └── main.py              # FastAPI REST endpoints
├── ui/
│   └── app.py                   # Interactive Streamlit research assistant UI
├── scripts/
│   ├── ingest.py                # CLI: Ingest papers for a topic
│   ├── process_pdfs.py          # CLI: Parse and clean downloaded PDFs
│   ├── create_chunks.py         # CLI: Generate section-aware chunks
│   ├── generate_embeddings.py   # CLI: Compute embeddings and index into Qdrant
│   ├── run_pipeline.py          # CLI: Run complete end-to-end vertical slice
│   └── evaluate.py              # CLI: Run retrieval quality benchmark
├── tests/                       # Complete pytest unit and integration test suite
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository and navigate into project directory
cd "Project Final Year (AI Research Copilot)"

# Initialize virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy `.env.example` to `.env`. Defaults work out-of-the-box:
```env
OPENALEX_EMAIL=researcher@example.com
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest
QDRANT_COLLECTION=research_chunks
```
*Note: If `QDRANT_URL` is left empty, the system automatically uses embedded on-disk storage at `./storage/qdrant` with zero Docker dependencies.*

### 3. Run the End-to-End Vertical Slice
Execute the complete pipeline (Discovery ➔ Parsing ➔ Chunking ➔ Indexing ➔ Querying):
```bash
python scripts/run_pipeline.py --query "GraphRAG" --target 10 --max-pdfs 3
```

### 4. Launch the Interactive Web UI (Streamlit)
```bash
streamlit run ui/app.py
```
Open your browser at `http://localhost:8501` to search evidence, browse cataloged papers, and synthesize answers with citations.

### 5. Launch the REST API (FastAPI)
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | `GET` | Health check, vector count, paper count, and Ollama status |
| `POST /search` | `POST` | Retrieve top-K evidence chunks (methods: `hybrid`, `semantic`, `keyword`) |
| `POST /ask` | `POST` | Evidence-grounded synthesis using retrieved passages |
| `POST /ingest` | `POST` | Trigger discovery from arXiv/OpenAlex and download PDFs |
| `GET /papers` | `GET` | Browse cataloged papers with metadata and PDF status |

---

## 🧪 Running Tests & Evaluation

### Run Test Suite
```bash
pytest -v tests
```
*All 18 unit and integration tests validate ingestion normalization, deduplication, PDF cleaning, section extraction, sliding-window chunking, dense vector indexing, BM25 retrieval, and API endpoints.*

### Run Retrieval Benchmark Evaluation
```bash
python scripts/evaluate.py --k 3
```
Measures ranking and retrieval quality across `Precision@K`, `Recall@K`, `MRR` (Mean Reciprocal Rank), and `nDCG@K`.
