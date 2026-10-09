from typing import Any, Dict, List

# Benchmark evaluation questions for scientific RAG evaluation
DEFAULT_EVALUATION_DATASET: List[Dict[str, Any]] = [
    {
        "query": "What are the limitations and challenges of GraphRAG?",
        "expected_topics": ["graph construction overhead", "scalability", "noise in graph", "computational cost"],
        "relevant_paper_ids": ["arxiv:2404.16130", "arxiv:2408.08921"],
    },
    {
        "query": "How does community detection improve retrieval in GraphRAG?",
        "expected_topics": ["hierarchical summarization", "Leiden algorithm", "global search", "high-level queries"],
        "relevant_paper_ids": ["arxiv:2404.16130"],
    },
    {
        "query": "Compare traditional vector RAG versus Graph-based RAG architectures.",
        "expected_topics": ["flat vector chunks", "multi-hop relationships", "global sensemaking", "index latency"],
        "relevant_paper_ids": ["arxiv:2404.16130", "openalex:W4391252874"],
    },
]
