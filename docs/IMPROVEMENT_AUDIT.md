# 🔍 ScholarRAG / Helix IRIS — Active Repository Improvement Audit (IMP-00)

**Audit Date:** 20 October 2026  
**Auditor:** Senior Engineering Agent  
**Branch:** `main` (clean working tree)  
**Target Repository:** `https://github.com/Apurva-Banerjee/Healix-IRIS-HybridRAG-Researcher-Assisatnt`  

---

## 1. Executive Audit Findings & Evidence

We traced all execution paths across evaluation, dataset management, chunking, citation verification, paper generation, and API services on the active branch:

| ID | Workstream | Active Source Location | Observed Implementation Defect | Confirmation Status | Severity |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **IMP-01** | Metric Correctness | `src/evaluation/retrieval_metrics.py:96-109` | Recall denominator `gt_size = max(len(relevant_indices), 1)` uses retrieved hits instead of full ground truth `len(ground_truth)`. IDCG calculation similarly constrained to retrieved hits rather than true relevant set. | **CONFIRMED** | **Critical (P0)** |
| **IMP-02** | Qrels Immutability | `scripts/evaluate.py:31` | `export_evaluation_dataset(eval_dir)` overwrites `queries.jsonl` and `qrels.jsonl` on every benchmark execution instead of treating them as immutable authoritative inputs. | **CONFIRMED** | **Critical (P0)** |
| **IMP-03** | Auditable Outputs | `scripts/evaluate.py:84` | Only aggregate results (`results.csv`, `results.json`) saved; query-level rankings, scores, and per-query breakdown missing. | **CONFIRMED** | **High (P0)** |
| **IMP-04** | Section-Aware Ablation | `src/chunking/chunker.py` | No fixed-window baseline chunker implemented to isolate the contribution of section-aware chunking against standard flat chunking. | **CONFIRMED** | **High (P0)** |
| **IMP-05** | Statistical Uncertainty | `scripts/evaluate.py` | No confidence intervals or paired difference calculations across queries; potential for overstating descriptive differences. | **CONFIRMED** | **High (P0)** |
| **IMP-06** | Claim-Level Support | `src/generation/citation_verifier.py` | Checks citation ID validity (1..K) and conflict regex, but does not parse atomic claims or perform fine-grained claim-to-evidence support classification. | **CONFIRMED** | **High (P0)** |
| **IMP-07** | Evidence-Constrained Answer | `src/generation/answer_generator.py:112` | Fallback template asserts papers "demonstrate" findings without fine-grained claim extraction. | **CONFIRMED** | **Medium (P0)** |
| **IMP-08** | Paper Generator Traceability | `src/generation/paper_generator.py:165` | `_draft_empirical_synthesis` contains static illustrative tables rather than reading live from machine-readable benchmark artifacts (`results.json`). | **CONFIRMED** | **High (P0)** |
| **IMP-09** | Bibliography Metadata | `src/generation/paper_generator.py:228` | Reference formatter uses internal `paper_id` instead of full bibliographic string (Authors, Year, Title, Venue, DOI). | **CONFIRMED** | **Medium (P0)** |
| **IMP-10** | Corpus Manifest | `data/` | Full-text vs abstract-only access status not cataloged in a structured manifest file (`corpus_manifest.json`). | **CONFIRMED** | **Medium (P1)** |
| **IMP-11** | API Sanitization | `src/api/main.py:165` | Endpoint errors can return raw exception strings. Structured error format required. | **CONFIRMED** | **Medium (P1)** |

---

## 2. Prioritized Implementation Sequence

1. **IMP-01:** Correct metric equations (Recall, Precision, MRR, nDCG) with hand-calculated unit test verification.
2. **IMP-02:** Decouple dataset generation from evaluation; make `queries.jsonl` and `qrels.jsonl` immutable inputs with CLI validation.
3. **IMP-03:** Save query-level rankings (`query_level_rankings.jsonl`) and generate a cryptographic `run_manifest.json`.
4. **IMP-04:** Implement `FixedWindowChunker`, build isolated ablation index, and run controlled ablation comparing Fixed-Window vs Section-Aware chunking.
5. **IMP-05:** Compute paired per-query differences and 95% bootstrap confidence intervals; align claims across all docs.
6. **IMP-06 & IMP-07:** Upgrade `CitationVerifier` to claim-level support classification (`supported`, `partially_supported`, `unsupported`, `insufficient_evidence`) with gold test set.
7. **IMP-08 & IMP-09:** Update `AcademicPaperGenerator` to dynamically populate empirical tables from verified benchmark artifacts and format complete IEEE bibliographic entries.
8. **IMP-10 & IMP-11:** Build `data/corpus_manifest.json`, sanitize FastAPI exception handling, and update UI.
9. **Verification & Git Deployment:** Run all tests with 0 warnings, execute benchmarks, and push to GitHub.
