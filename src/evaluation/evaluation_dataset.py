import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Canonical benchmark query dataset (30 scientific research queries across 6 categories)
DEFAULT_EVALUATION_DATASET: List[Dict[str, Any]] = [
    # Category 1: Conceptual Foundations
    {
        "query_id": "q01",
        "category": "conceptual",
        "query": "What is the foundational definition of Retrieval-Augmented Generation in NLP?",
        "expected_topics": ["parametric memory", "non-parametric memory", "pre-trained seq2seq", "dense passage retrieval"],
        "relevant_paper_ids": ["arxiv:2004.12832", "arxiv:2312.07559"],
    },
    {
        "query_id": "q02",
        "category": "conceptual",
        "query": "How does GraphRAG conceptually differ from standard flat-chunk vector RAG?",
        "expected_topics": ["knowledge graph", "entities and relationships", "global sensemaking", "community detection", "hierarchical summarization"],
        "relevant_paper_ids": ["arxiv:2409.13740", "arxiv:2501.00309", "arxiv:2506.05690"],
    },
    {
        "query_id": "q03",
        "category": "conceptual",
        "query": "What is Self-RAG and what are reflection tokens?",
        "expected_topics": ["self-reflection", "reflection tokens", "[Retrieve]", "[IsRel]", "[IsSup]", "[IsUse]", "adaptive retrieval"],
        "relevant_paper_ids": ["arxiv:2310.11511"],
    },
    {
        "query_id": "q04",
        "category": "conceptual",
        "query": "What is Corrective Retrieval Augmented Generation (CRAG)?",
        "expected_topics": ["retrieval evaluator", "corrective actions", "web search fallback", "decompose and recombine"],
        "relevant_paper_ids": ["arxiv:2401.15884"],
    },
    {
        "query_id": "q05",
        "category": "conceptual",
        "query": "What does ColBERT contribute to dense passage retrieval?",
        "expected_topics": ["contextualized late interaction", "token-level embeddings", "MaxSim", "efficient BERT search"],
        "relevant_paper_ids": ["arxiv:2004.12832"],
    },
    {
        "query_id": "q06",
        "category": "conceptual",
        "query": "What is PaperQA2 and what does superhuman scientific synthesis refer to?",
        "expected_topics": ["scientific literature review", "citation precision", "evidence validation", "agentic paper question answering"],
        "relevant_paper_ids": ["arxiv:2409.13740"],
    },

    # Category 2: Architecture & Methodology
    {
        "query_id": "q07",
        "category": "architecture",
        "query": "How does community detection using Leiden algorithm operate in GraphRAG?",
        "expected_topics": ["Leiden algorithm", "hierarchical clustering", "graph partition", "community summaries", "global query answering"],
        "relevant_paper_ids": ["arxiv:2409.13740", "arxiv:2501.00309"],
    },
    {
        "query_id": "q08",
        "category": "architecture",
        "query": "How are reflection tokens trained in Self-RAG?",
        "expected_topics": ["critic model", "generator model", "offline preference data", "supervised fine-tuning"],
        "relevant_paper_ids": ["arxiv:2310.11511"],
    },
    {
        "query_id": "q09",
        "category": "architecture",
        "query": "How does CRAG evaluate the quality of retrieved documents?",
        "expected_topics": ["confidence score", "correct", "incorrect", "ambiguous", "knowledge refinement"],
        "relevant_paper_ids": ["arxiv:2401.15884"],
    },
    {
        "query_id": "q10",
        "category": "architecture",
        "query": "How does OpenScholar retrieve and cite scientific literature?",
        "expected_topics": ["scholarly document search", "citation precision", "open access corpus", "retrieval augmented language model"],
        "relevant_paper_ids": ["arxiv:2411.14199"],
    },
    {
        "query_id": "q11",
        "category": "architecture",
        "query": "What is HYBGRAG and how does it integrate textual and relational graph structures?",
        "expected_topics": ["relational knowledge", "textual corpus", "hybrid retrieval", "knowledge graph embeddings"],
        "relevant_paper_ids": ["arxiv:2412.16311"],
    },
    {
        "query_id": "q12",
        "category": "architecture",
        "query": "How do knowledge filters empower GraphRAG against noisy entity connections?",
        "expected_topics": ["knowledge filter", "noisy edges", "pruning irrelevant entities", "graph compression"],
        "relevant_paper_ids": ["arxiv:2503.13804"],
    },

    # Category 3: Limitations & Failure Modes
    {
        "query_id": "q13",
        "category": "limitations",
        "query": "What are the primary limitations and computational overheads of GraphRAG?",
        "expected_topics": ["graph construction cost", "LLM extraction tokens", "index latency", "noisy entity relations", "scalability limits"],
        "relevant_paper_ids": ["arxiv:2501.14050", "arxiv:2604.09666", "arxiv:2506.05690"],
    },
    {
        "query_id": "q14",
        "category": "limitations",
        "query": "Under what conditions does GraphRAG fail or underperform flat vector RAG?",
        "expected_topics": ["specific factoid queries", "local single-document questions", "high query latency", "sparse graphs"],
        "relevant_paper_ids": ["arxiv:2501.14050", "arxiv:2604.09666"],
    },
    {
        "query_id": "q15",
        "category": "limitations",
        "query": "What limitations exist in standard dense vector retrieval for scientific terminology?",
        "expected_topics": ["out-of-vocabulary terms", "exact acronym matching", "lexical gap", "rare technical entities"],
        "relevant_paper_ids": ["arxiv:2004.12832", "acl:2023.eacl-main.72"],
    },
    {
        "query_id": "q16",
        "category": "limitations",
        "query": "What are the latency bottlenecks in iterative retrieval-augmented generation?",
        "expected_topics": ["multi-turn inference", "repeated LLM calls", "retrieval overhead", "token generation latency"],
        "relevant_paper_ids": ["acl:2023.acl-long.99", "arxiv:2310.11511"],
    },

    # Category 4: Empirical & Benchmark Performance
    {
        "query_id": "q17",
        "category": "empirical",
        "query": "What benchmarks are used to evaluate multi-hop scientific reasoning in RAG?",
        "expected_topics": ["HotpotQA", "QASA", "PubMedQA", "BioASQ", "GraphRAG-Bench", "MultiHop-RAG"],
        "relevant_paper_ids": ["arxiv:2506.02404", "acl:2024.acl-long.642", "arxiv:2604.09666"],
    },
    {
        "query_id": "q18",
        "category": "empirical",
        "query": "How does Self-RAG perform on open-domain question answering compared to vanilla RAG baselines?",
        "expected_topics": ["factual accuracy", "ROUGE", "accuracy improvement", "fewer hallucinations"],
        "relevant_paper_ids": ["arxiv:2310.11511"],
    },
    {
        "query_id": "q19",
        "category": "empirical",
        "query": "What accuracy and citation precision does PaperQA2 achieve on scientific benchmark evaluations?",
        "expected_topics": ["citation accuracy", "human expert comparison", "precision", "recall", "hallucination rate"],
        "relevant_paper_ids": ["arxiv:2409.13740"],
    },
    {
        "query_id": "q20",
        "category": "empirical",
        "query": "What empirical gains does ColBERT late interaction offer in query latency and throughput?",
        "expected_topics": ["computation speedup", "vector quantization", "millisecond latency", "FLOP reduction"],
        "relevant_paper_ids": ["arxiv:2004.12832"],
    },
    {
        "query_id": "q21",
        "category": "empirical",
        "query": "How significant are the real-world performance differences between RAG and GraphRAG?",
        "expected_topics": ["benchmark comparison", "empirical evaluation", "marginal gains", "cost-benefit ratio"],
        "relevant_paper_ids": ["arxiv:2506.06331", "arxiv:2502.11371"],
    },

    # Category 5: Cross-Paper Comparisons
    {
        "query_id": "q22",
        "category": "comparative",
        "query": "Compare Self-RAG versus Corrective RAG (CRAG) mechanisms for handling poor retrieval.",
        "expected_topics": ["reflection tokens", "external retrieval evaluator", "internal critic vs external action", "web search fallback"],
        "relevant_paper_ids": ["arxiv:2310.11511", "arxiv:2401.15884"],
    },
    {
        "query_id": "q23",
        "category": "comparative",
        "query": "Compare dense semantic retrieval with sparse BM25 retrieval across academic documents.",
        "expected_topics": ["semantic similarity", "keyword matching", "hybrid fusion", "vocabulary mismatch"],
        "relevant_paper_ids": ["acl:2023.eacl-main.72", "acl:2026.eacl-long.303", "arxiv:2004.12832"],
    },
    {
        "query_id": "q24",
        "category": "comparative",
        "query": "What trade-offs exist between OpenScholar and PaperQA for scientific literature synthesis?",
        "expected_topics": ["agentic workflows", "iterative query refinement", "citation precision", "open source model accessibility"],
        "relevant_paper_ids": ["arxiv:2409.13740", "arxiv:2411.14199"],
    },
    {
        "query_id": "q25",
        "category": "comparative",
        "query": "Compare linear score fusion with Reciprocal Rank Fusion (RRF) for hybrid RAG retrieval.",
        "expected_topics": ["score normalization", "min-max scaling", "rank order", "smoothing constant k", "score calibration"],
        "relevant_paper_ids": ["acl:2026.eacl-long.303"],
    },

    # Category 6: Scientific Literature Review & Synthesis
    {
        "query_id": "q26",
        "category": "synthesis",
        "query": "How is automated multi-document scientific literature review generated using retrieval?",
        "expected_topics": ["multi-document summarization", "citation chains", "related work generation", "evidence clustering"],
        "relevant_paper_ids": ["acl:2022.naacl-main.193", "arxiv:2508.05660"],
    },
    {
        "query_id": "q27",
        "category": "synthesis",
        "query": "How do citation graphs assist in scientific question answering?",
        "expected_topics": ["citation graph", "CG-RAG", "co-citation networks", "reference propagation"],
        "relevant_paper_ids": ["arxiv:2501.15067"],
    },
    {
        "query_id": "q28",
        "category": "synthesis",
        "query": "What is Adaptive Retrieval-Augmented Generation (Adaptive-RAG)?",
        "expected_topics": ["query complexity classifier", "adaptive routing", "iterative retrieval", "efficiency"],
        "relevant_paper_ids": ["acl:2024.acl-long.236"],
    },
    {
        "query_id": "q29",
        "category": "synthesis",
        "query": "What are the core capabilities of the ScholarRAG platform for researchers?",
        "expected_topics": ["section-aware chunking", "hybrid retrieval", "grounded synthesis", "citation traceability", "interactive UI"],
        "relevant_paper_ids": ["acl:2025.acl-demo.49"],
    },
    {
        "query_id": "q30",
        "category": "synthesis",
        "query": "How can researchers determine when to use graphs versus flat vector indices in RAG?",
        "expected_topics": ["relationship density", "cross-document themes", "query type", "global vs local questions"],
        "relevant_paper_ids": ["arxiv:2506.05690"],
    },
]


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_evaluation_dataset(output_dir: Path, overwrite: bool = False) -> Tuple[Path, Path]:
    """
    Export benchmark queries.jsonl and qrels.jsonl for reproducibility.
    Does NOT overwrite existing files unless overwrite=True is explicitly passed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_file = output_dir / "queries.jsonl"
    qrels_file = output_dir / "qrels.jsonl"

    if queries_file.exists() and qrels_file.exists() and not overwrite:
        logger.info("Evaluation files already exist at %s. Skipping export (immutable mode).", output_dir)
        return queries_file, qrels_file

    with open(queries_file, "w", encoding="utf-8") as f_q, open(qrels_file, "w", encoding="utf-8") as f_r:
        for item in DEFAULT_EVALUATION_DATASET:
            f_q.write(json.dumps({
                "query_id": item["query_id"],
                "query": item["query"],
                "category": item.get("category", "general"),
                "expected_topics": item.get("expected_topics", []),
            }) + "\n")

            for pid in item.get("relevant_paper_ids", []):
                f_r.write(json.dumps({
                    "query_id": item["query_id"],
                    "paper_id": pid,
                    "relevance": 3,
                }) + "\n")

    logger.info("Exported %d benchmark queries to %s", len(DEFAULT_EVALUATION_DATASET), output_dir)
    return queries_file, qrels_file


def load_authoritative_dataset(queries_path: Path, qrels_path: Path) -> List[Dict[str, Any]]:
    """
    Load authoritative benchmark queries and qrels from disk.
    Strictly read-only; maps qrels paper IDs into each corresponding query object.
    """
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found at {queries_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found at {qrels_path}")

    # 1. Load queries
    queries_by_id: Dict[str, Dict[str, Any]] = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = item["query_id"]
                item["relevant_paper_ids"] = []
                item["relevance_grades"] = {}
                queries_by_id[qid] = item

    # 2. Load qrels and map to queries
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qrel = json.loads(line)
                qid = qrel["query_id"]
                pid = qrel["paper_id"]
                rel = qrel.get("relevance", 1)

                if qid in queries_by_id:
                    queries_by_id[qid]["relevant_paper_ids"].append(pid)
                    queries_by_id[qid]["relevance_grades"][pid] = rel
                else:
                    logger.warning("Qrel record references unknown query_id '%s'", qid)

    dataset = list(queries_by_id.values())
    logger.info("Loaded %d authoritative queries from %s (with qrels from %s)", len(dataset), queries_path, qrels_path)
    return dataset


def validate_benchmark_inputs(queries_path: Path, qrels_path: Path) -> Dict[str, Any]:
    """Validate consistency, counts, and SHA-256 hashes of queries and qrels files."""
    if not queries_path.exists():
        return {"valid": False, "error": f"Missing {queries_path}"}
    if not qrels_path.exists():
        return {"valid": False, "error": f"Missing {qrels_path}"}

    queries_hash = file_sha256(queries_path)
    qrels_hash = file_sha256(qrels_path)

    dataset = load_authoritative_dataset(queries_path, qrels_path)
    unjudged_queries = [q["query_id"] for q in dataset if not q["relevant_paper_ids"]]

    total_qrels = sum(len(q["relevant_paper_ids"]) for q in dataset)

    return {
        "valid": True,
        "queries_count": len(dataset),
        "total_judgments_count": total_qrels,
        "unjudged_queries_count": len(unjudged_queries),
        "unjudged_query_ids": unjudged_queries,
        "queries_file": str(queries_path),
        "queries_sha256": queries_hash,
        "qrels_file": str(qrels_path),
        "qrels_sha256": qrels_hash,
    }
