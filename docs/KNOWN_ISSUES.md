# ⚠️ Known Issues, Limitations & Engineering Boundaries

**Document ID:** `docs/KNOWN_ISSUES.md`  
**System Identity:** AI Research Scientist Assistant / ScholarRAG  
**Status Date:** 19 October 2026  

---

## 1. Document Extraction & Full-Text Availability

1. **Paywalled Literature Coverage:**
   - Open-access preprints (arXiv, ACL Anthology) are downloaded and parsed into full-text section chunks.
   - For closed-access commercial publisher papers (e.g., certain Springer or Elsevier publications), ScholarRAG indexes the validated abstract and metadata into dedicated abstract-level evidence chunks.
   
2. **Complex Embedded Tables and Multi-Panel Vector Graphics:**
   - PyMuPDF extracts raw text, captions, and structural headings with high fidelity.
   - Embedded raster images, circuit diagrams, and multi-panel chemical plots are represented via their text captions rather than direct vision-language parsing.

---

## 2. Computational Latency & Hardware Boundaries

1. **CPU vs. GPU Inference Latency:**
   - In CPU-only environments, dense embedding generation (`BAAI/bge-small-en-v1.5`) takes ~230 ms per query.
   - BM25 lexical retrieval operates in sub-3 ms (< 2.5 ms).
   - Enabling a CUDA-capable GPU reduces dense query latency to under 15 ms.

2. **Local LLM Daemon Dependency:**
   - Ollama generation requires the local daemon to be running (`ollama serve`). If offline, ScholarRAG automatically activates its structured extractive synthesis engine to prevent pipeline crashes.

---

## 3. Scientific Verification Boundaries

1. **Automated Citation Grounding vs. Human Scientific Truth:**
   - The `CitationVerifier` programmatically validates that claims map to supplied evidence IDs and flags contradictory empirical claims.
   - It does not replace peer-reviewed scientific refereeing or experimental laboratory replication.
