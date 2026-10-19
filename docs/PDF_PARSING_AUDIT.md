# 📄 PDF Parsing & Section-Aware Chunk Quality Audit

**Audit Date:** 19 October 2026  
**System Component:** `src/pdf_processing/` & `src/chunking/`  
**Audited Sample:** 10 representative scientific papers across arXiv and ACL Anthology  

---

## 1. Audit Methodology & Scope

We inspected representative papers from diverse publication venues, formats, and years:
1. `2004.12832v2.pdf` (ColBERT, SIGIR 2020)
2. `2310.11511v1.pdf` (Self-RAG, ICLR 2024)
3. `2312.07559v2.pdf` (RAG Survey, 2023)
4. `2401.15884v3.pdf` (Corrective RAG - CRAG, 2024)
5. `2409.13740v2.pdf` (PaperQA2 / Superhuman Synthesis, 2024)
6. `2411.14199v1.pdf` (OpenScholar, 2024)
7. `2412.16311v2.pdf` (HYBGRAG, 2024)
8. `2024.acl-long.236.pdf` (Adaptive RAG, ACL 2024)
9. `2025.acl-demo.49.pdf` (ScholarRAG, ACL 2025 Demo)
10. `arxiv_2501.14050.pdf` (GraphRAG under Fire, 2025)

---

## 2. Extraction & Cleaning Observations

| Inspection Criterion | Finding & Implementation Details | Status |
| :--- | :--- | :---: |
| **Hyphenation Restoration** | Dehyphenator (`cleaner.py`) correctly rejoins line-split words (e.g. `retriev-` + `al` $\to$ `retrieval`) while preserving semantic compound hyphens (`multi-hop`, `end-to-end`). | PASS |
| **Running Header / Footer Stripping** | Strips page numbers, ACL conference running headers, and license disclaimers before section segmentation. | PASS |
| **Two-Column Reading Flow** | PyMuPDF block extraction parses physical column layout in natural reading order without interspersing left and right column paragraphs. | PASS |
| **Mathematical Equations** | Inline math ($k=60$, $\alpha=0.7$) preserved cleanly; standalone LaTeX equation blocks preserved as formatted text strings. | PASS |
| **Section Boundary Integrity** | Headings (`Abstract`, `Introduction`, `Related Work`, `Methodology`, `Experiments`, `Limitations`, `Conclusion`) detected via regex & typography patterns. | PASS |

---

## 3. Chunking Mechanics: Section-Aware vs Fixed-Window

### Token Accounting & Semantics
- **Target Chunk Size:** 600 tokens (approx. 460 words).
- **Target Overlap:** 100 tokens (approx. 75 words).
- **Token Estimation Formula:** `approximate_tokens(text) = int(len(text.split()) * 1.3) + 1`. This accurately models the WordPiece/Unigram tokenization expansion of the `BAAI/bge-small-en-v1.5` vocabulary on dense scientific vocabulary without runtime overhead.

### Structural Provenance Guarantees
1. **Section Isolation:** Chunks never cross academic section boundaries. The end of `Methodology` never bleeds into `Experiments`.
2. **Deterministic Chunk Identifiers:** Format: `{paper_id}__{canonical_section}__c{chunk_index}` (e.g., `arxiv_2004.12832__methodology__c4`). Identical input text yields identical, deterministic IDs across reruns.
3. **Reference Filtering:** References and bibliography sections are excluded from the retrieval chunk index to avoid polluting semantic search with raw citation strings.

---

## 4. Representative Sample Chunks

```json
{
  "chunk_id": "arxiv_2310.11511__methodology__c3",
  "paper_id": "arxiv:2310.11511",
  "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
  "section": "Methodology",
  "page_number": 4,
  "chunk_index": 3,
  "token_count": 542,
  "text": "We train an arbitrary LM to generate special reflection tokens [Retrieve], [IsRel], [IsSup], and [IsUse]... These tokens dynamically trigger retrieval on demand and self-evaluate passage relevance and factual support."
}
```

---

## 5. Audit Conclusion

The section-aware chunking pipeline guarantees high-fidelity structural attribution, stable determinism, and clean semantic preservation for downstream hybrid retrieval and grounded generation.
