import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CitationVerifier:
    """
    Verifies claim-to-evidence grounding and citation validity in generated research answers.
    Detects hallucinated citations, missing evidence, and conflicting sources.
    """

    def __init__(self):
        # Regex patterns matching [Evidence X], [E X], [Evidence 1, 2], [1], [2]
        self.evidence_pattern = re.compile(r"\[(?:Evidence\s*|E\s*)?(\d+)(?:\s*,\s*(\d+))*\]", re.IGNORECASE)
        self.single_evidence_pattern = re.compile(r"\[(?:Evidence\s*|E\s*)?(\d+)\]", re.IGNORECASE)

    def extract_cited_indices(self, text: str) -> List[int]:
        """Extract all cited evidence numbers (1-based) from generated answer."""
        indices = set()
        for match in re.finditer(r"\[(?:Evidence\s*|E\s*)?(\d+)\]", text, re.IGNORECASE):
            try:
                indices.add(int(match.group(1)))
            except ValueError:
                pass

        # Also search for multi-citations like [Evidence 1, Evidence 2]
        for match in re.finditer(r"\[(?:Evidence\s*)?(\d+)(?:\s*,\s*(?:Evidence\s*)?(\d+))+\]", text, re.IGNORECASE):
            for group in match.groups():
                if group:
                    try:
                        indices.add(int(group))
                    except ValueError:
                        pass

        return sorted(list(indices))

    def verify_citations(
        self,
        answer_text: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify every citation marker against supplied evidence chunks.
        Returns detailed grounding report.
        """
        num_evidence = len(retrieved_chunks)
        cited_indices = self.extract_cited_indices(answer_text)

        valid_citations = []
        dangling_citations = []
        cited_chunks = []

        for idx in cited_indices:
            if 1 <= idx <= num_evidence:
                chunk = retrieved_chunks[idx - 1]
                valid_citations.append(idx)
                cited_chunks.append({
                    "evidence_id": f"Evidence {idx}",
                    "chunk_id": chunk.get("chunk_id"),
                    "paper_id": chunk.get("paper_id"),
                    "title": chunk.get("title"),
                    "section": chunk.get("section"),
                    "page": chunk.get("page"),
                })
            else:
                dangling_citations.append(idx)

        total_cited = len(cited_indices)
        support_rate = round(len(valid_citations) / total_cited, 4) if total_cited > 0 else 1.0

        # Check for insufficient evidence declaration
        has_insufficient_notice = bool(
            re.search(r"insufficient evidence|corpus provides insufficient|not found in indexed|cannot be determined", answer_text, re.IGNORECASE)
        )

        return {
            "total_citations_found": total_cited,
            "valid_citations_count": len(valid_citations),
            "valid_indices": valid_citations,
            "dangling_citations_count": len(dangling_citations),
            "dangling_indices": dangling_citations,
            "citation_support_rate": support_rate,
            "is_fully_grounded": len(dangling_citations) == 0,
            "insufficient_evidence_declared": has_insufficient_notice,
            "cited_evidence_units": cited_chunks,
        }

    def detect_conflicting_evidence(
        self,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Identify potential empirical contradictions or divergent claims across retrieved papers.
        Flags opposing sentiments on performance gains, computational cost, or necessity.
        """
        conflicts = []
        if len(retrieved_chunks) < 2:
            return conflicts

        opposing_pairs = [
            (r"significantly improv(?:es?|ing|ed)|superior performance|outperforms?|substantial gain",
             r"no significant|marginal gain|fails to outperform|underperforms?|does not improve"),
            (r"low overhead|efficient|scalable|speedup",
             r"high computational overhead|expensive|scalability bottleneck|prohibitive cost|overhead"),
            (r"essential|always needed|indispensable",
             r"do we still need|redundant|unnecessary|not needed"),
        ]

        for i in range(len(retrieved_chunks)):
            for j in range(i + 1, len(retrieved_chunks)):
                t1 = f"{retrieved_chunks[i].get('title', '')} {retrieved_chunks[i].get('text', '')}"
                t2 = f"{retrieved_chunks[j].get('title', '')} {retrieved_chunks[j].get('text', '')}"
                p1 = retrieved_chunks[i].get("title", f"Paper {i+1}")
                p2 = retrieved_chunks[j].get("title", f"Paper {j+1}")

                for pos_pattern, neg_pattern in opposing_pairs:
                    m1_pos = re.search(pos_pattern, t1, re.IGNORECASE)
                    m2_neg = re.search(neg_pattern, t2, re.IGNORECASE)
                    m1_neg = re.search(neg_pattern, t1, re.IGNORECASE)
                    m2_pos = re.search(pos_pattern, t2, re.IGNORECASE)

                    if (m1_pos and m2_neg) or (m1_neg and m2_pos):
                        conflicts.append({
                            "type": "empirical_divergence",
                            "paper_a": p1,
                            "paper_b": p2,
                            "section_a": retrieved_chunks[i].get("section"),
                            "section_b": retrieved_chunks[j].get("section"),
                            "description": f"Divergent perspectives identified between '{p1}' and '{p2}' regarding performance or overhead.",
                        })
                        break

        return conflicts
