# 🔍 ScholarRAG Retrieval Engine Architecture & Design

**Document ID:** `docs/RETRIEVAL_DESIGN.md`  
**System Identity:** ScholarRAG / AI Research Scientist Assistant  
**Core Framework:** Multi-Mode Dense + Lexical Hybrid Information Retrieval  

---

## 1. Overview & Rationale

Scientific literature retrieval presents unique challenges compared to open-domain web search:
1. **Dense Semantic Matching:** Necessary to understand complex conceptual paraphrasing, high-level research questions, and methodology variations where exact terminology may vary.
2. **Sparse Lexical Precision:** Essential for exact keyword matching of specialized terminology, model names (e.g. "ColBERT", "Leiden", "Self-RAG"), mathematical symbols, benchmark names, and author citations.
3. **Score Fusion:** Combining raw cosine similarity (bounded $[ -1, 1 ]$ or $[ 0, 1 ]$) directly with unbounded BM25 term frequency scores (e.g., $15.4$) is mathematically invalid without rigorous score normalization or rank-based fusion.

ScholarRAG addresses this with four independently callable and benchmarkable retrieval modes adhering to a unified result schema.

---

## 2. Independent Retrieval Modes

```text
                                [ User Query ]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
         Dense Vector Lane                         Sparse Lexical Lane
      (BAAI/bge-small-en-v1.5)                         (BM25Okapi)
        Qdrant Vector Store                       Inverted Token Corpus
                 │                                         │
                 │                                         │
        Top-K Candidates                           Top-K Candidates
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  Reciprocal Rank             Min-Max Normalized
                   Fusion (RRF)                 Linear Fusion
                   (Mode: rrf)             (Mode: hybrid_weighted)
```

### 2.1 Mode 1: Sparse Lexical Retrieval (`bm25`)
- **Algorithm:** Okapi BM25 (`rank-bm25`)
- **Tokenization:** Case-insensitive alphanumeric regex filter `\b[a-zA-Z0-9_\-\.]{2,}\b` matching technical terms, hyphenated concepts, and version tags.
- **Corpus:** Indexed over composite text `Title + Section + Passage Content`.
- **Primary Use-Case:** Exact model identification, benchmark names, author citations, and acronym lookups.

### 2.2 Mode 2: Dense Semantic Retrieval (`dense`)
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (frozen revision, 384 dimensions).
- **Index:** Qdrant Cloud / Local Vector Store with Cosine distance metric.
- **Symmetry Invariant:** Query encoding and chunk passage encoding use identical normalization (`normalize_embeddings=True`) and dimension constraints (384).
- **Primary Use-Case:** Abstract conceptual queries, thematic exploration, methodology explanations.

### 2.3 Mode 3: Reciprocal Rank Fusion (`hybrid_rrf`) — *Default Primary*
Combines ranked candidate lists without depending on calibration of raw numerical scores:

$$\operatorname{RRF}(d) = \sum_{r \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \operatorname{rank}_r(d)}$$

- **Smoothing Constant:** $k = 60$
- **Rank Indexing:** 1-based ranks ($\operatorname{rank}_1 = 1, \operatorname{rank}_2 = 2, \dots$)
- **Candidate Pool:** $K_{\text{candidate}} = \max(3 \cdot K, 15)$ retrieved from each lane before fusion.
- **Advantage:** Robust against outliers; treats lexical and semantic rankings fairly regardless of raw score scale.

### 2.4 Mode 4: Min-Max Normalized Weighted Fusion (`hybrid_weighted`)
Provides calibrated linear score combination:

$$S_{\text{weighted}}(d) = \alpha \cdot \hat{S}_{\text{dense}}(d) + (1 - \alpha) \cdot \hat{S}_{\text{BM25}}(d)$$

Where raw scores are min-max normalized over candidate sets:

$$\hat{S}(d) = \frac{S(d) - \min(S)}{\max(S) - \min(S)}$$

- **Default Mixing Parameter:** $\alpha = 0.70$ (70% semantic dense weight, 30% lexical BM25 weight).
- **Degeneracy Rule:** If $\max(S) - \min(S) < 10^{-9}$, all candidates receive normalized score $1.0$.

---

## 3. Common Evidence Result Schema

Every retrieval lane outputs evidence items conforming to the standardized contract:

```json
{
  "chunk_id": "arxiv_2004.12832__Introduction__c1",
  "paper_id": "arxiv:2004.12832",
  "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
  "section": "Introduction",
  "page": 1,
  "chunk_index": 1,
  "text": "Contextualized late interaction allows scaling BERT retrieval to massive corpora...",
  "score": 0.032258,
  "retrieval_lane": "hybrid_rrf",
  "source": "arxiv",
  "dense_rank": 1,
  "bm25_rank": 2
}
```

### Deterministic Tie-Breaking
When candidates achieve identical fusion scores, ties are resolved deterministically by lexicographical ordering of `chunk_id`.

---

## 4. Safe Index Rebuild Procedure

To rebuild both the BM25 index and Qdrant vector collection from canonical parsed chunks:

```bash
# 1. Audit corpus integrity
python scripts/audit_corpus.py --repair

# 2. Re-generate embeddings and index into Qdrant
python scripts/build_index.py --collection research_chunks
```
