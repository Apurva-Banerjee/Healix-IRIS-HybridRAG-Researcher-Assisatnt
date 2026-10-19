# 🔄 ScholarRAG Reproducibility Guide

**Document ID:** `docs/REPRODUCIBILITY.md`  
**Purpose:** Ensure independent replication of ScholarRAG data integrity, indexing, retrieval benchmark, and UI execution.  

---

## 1. System Requirements

- **Operating System:** Windows 10/11, Ubuntu 22.04+, or macOS Sonoma+
- **Python Version:** Python 3.11.x (tested on Python 3.11.5)
- **Disk Storage:** Minimum 10 GB free space for scientific PDFs and vector storage
- **Memory (RAM):** 8 GB minimum (16 GB recommended)
- **Optional Dependencies:** Ollama daemon (`ollama run llama3:latest`), CUDA-compatible GPU

---

## 2. Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/Apurva-Banerjee/Healix-IRIS-HybridRAG-Researcher-Assisatnt.git
cd Healix-IRIS-HybridRAG-Researcher-Assisatnt

# 2. Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
```

---

## 3. End-to-End Pipeline Execution

Follow these sequenced steps to reproduce all stages of the corpus:

```bash
# Step 1: Run Corpus Integrity Audit & Reconciliation
python scripts/audit_corpus.py --repair

# Step 2: (Optional) Ingest new papers from arXiv and OpenAlex
python scripts/ingest.py --query "GraphRAG" --target-papers 20

# Step 3: Parse downloaded PDFs into structured section JSON
python scripts/process_pdfs.py

# Step 4: Generate section-aware chunks
python scripts/create_chunks.py

# Step 5: Index vectors into Qdrant Cloud or local store
python scripts/build_index.py --collection research_chunks

# Step 6: Execute multi-mode controlled retrieval evaluation
python scripts/evaluate.py --k 5

# Step 7: Run complete automated test suite
python -m pytest

# Step 8: Launch Interactive Research Assistant UI
streamlit run ui/app.py

# Step 9: Launch FastAPI REST Server
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

## 4. Benchmark Artifacts Location

- Empirical Evaluation CSV: `artifacts/evaluation/results.csv`
- Machine-Readable JSON: `artifacts/evaluation/results.json`
- Qualitative Error Analysis: `artifacts/evaluation/error_analysis.md`
- Generated Research Paper Draft: `artifacts/research_paper_draft.md`
- Benchmark Queries: `data/evaluation/queries.jsonl`
- Ground Truth Relevance Labels: `data/evaluation/qrels.jsonl`
