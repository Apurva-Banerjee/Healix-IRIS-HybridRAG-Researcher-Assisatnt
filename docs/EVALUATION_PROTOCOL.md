# 🧪 ScholarRAG Evaluation Protocol & Benchmark Methodology

**Document ID:** `docs/EVALUATION_PROTOCOL.md`  
**Purpose:** Establish reproducible scientific evaluation metrics and benchmark query dataset standards for ScholarRAG.  

---

## 1. Query Dataset Taxonomy

The evaluation dataset (`data/evaluation/queries.jsonl` and `data/evaluation/qrels.jsonl`) comprises 30 domain-specific scientific queries spanning six research categories:

1. **Conceptual Foundations (6 queries):** Foundational definitions of RAG, GraphRAG, Self-RAG, ColBERT, PaperQA2.
2. **Architecture & Methodology (6 queries):** Leiden clustering, reflection token training, CRAG confidence thresholds, HYBGRAG.
3. **Empirical Benchmarks (5 queries):** HotpotQA, MultiHop-RAG, accuracy, ROUGE scores, latency gains.
4. **Critical Limitations & Trade-Offs (4 queries):** Indexing cost, scalability bottlenecks, lexical vocabulary mismatch, query latency.
5. **Cross-Paper Comparisons (4 queries):** Self-RAG vs CRAG, BM25 vs Dense, PaperQA vs OpenScholar, RRF vs Linear Fusion.
6. **Literature Synthesis & Roadmaps (5 queries):** Automated literature review, citation graph QA, adaptive routing.

---

## 2. Metric Formulations

### 2.1 Precision@K

$$\operatorname{Precision}@K = \frac{|\operatorname{Retrieved}_{1:K} \cap \operatorname{Relevant}|}{K}$$

Measures the proportion of retrieved evidence units in the top-$K$ that are relevant to the query.

### 2.2 Recall@K

$$\operatorname{Recall}@K = \frac{|\operatorname{Retrieved}_{1:K} \cap \operatorname{Relevant}|}{|\operatorname{Relevant}|}$$

Measures the fraction of annotated relevant evidence captured in the top-$K$.

### 2.3 Mean Reciprocal Rank (MRR)

$$\operatorname{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\operatorname{rank}_q^*}$$

Where $\operatorname{rank}_q^*$ is the rank position of the first relevant evidence chunk for query $q$.

### 2.4 Normalized Discounted Cumulative Gain (nDCG@K)

$$\operatorname{DCG}@K = \sum_{i=1}^{K} \frac{r_i}{\log_2(i + 1)}, \quad \operatorname{nDCG}@K = \frac{\operatorname{DCG}@K}{\operatorname{IDCG}@K}$$

Where $r_i \in \{0, 1\}$ denotes binary relevance (or graded relevance), and $\operatorname{IDCG}@K$ is the ideal discounted gain achieved by placing all relevant documents at the top.

---

## 3. Benchmark Execution Command

```bash
# Execute evaluation over all 4 retrieval modes at cutoff K=5
python scripts/evaluate.py --k 5
```
Output artifacts are saved automatically to:
- `artifacts/evaluation/results.csv`
- `artifacts/evaluation/results.json`
- `artifacts/evaluation/error_analysis.md`
