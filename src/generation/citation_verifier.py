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

    def verify_claim_support(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> Dict[str, Any]:
        """
        Verify whether an atomic scientific claim is supported, unsupported,
        or indicates insufficient evidence given the supplied evidence text.
        
        Performs:
        1. Insufficiency detection (explicit abstention markers).
        2. Numerical & metric consistency verification.
        3. Contradiction & absolute claims detection.
        4. Token and entity containment scoring.
        """
        c_lower = claim_text.lower().strip()
        e_lower = evidence_text.lower().strip()

        # 1. Check for insufficiency markers
        insufficiency_markers = [
            "insufficient evidence",
            "cannot be determined",
            "no hardware configuration details",
            "no details provided",
            "corpus provides insufficient",
            "not found in indexed",
            "not provided in this section",
        ]
        if any(marker in c_lower for marker in insufficiency_markers) or any(marker in e_lower for marker in insufficiency_markers):
            return {
                "status": "insufficient_evidence",
                "confidence": 0.95,
                "overlap_score": 0.0,
                "reason": "Explicit insufficiency or lack of evidence declared in claim or evidence passage.",
            }

        # 2. Numerical and metric verification
        # Extract numerical expressions including decimals and percentages
        num_pattern = re.compile(r"\b\d+(?:\.\d+)?%?\b")
        claim_nums = set(num_pattern.findall(c_lower))
        evidence_nums = set(num_pattern.findall(e_lower))

        # Check for numerical contradiction (e.g. claim has 98.5% but evidence has 81.6%)
        claim_metrics = {n for n in claim_nums if "%" in n or "." in n}
        evidence_metrics = {n for n in evidence_nums if "%" in n or "." in n}
        if claim_metrics and evidence_metrics and not (claim_metrics & evidence_metrics):
            return {
                "status": "unsupported",
                "confidence": 0.95,
                "overlap_score": 0.2,
                "reason": f"Numerical contradiction: claim asserts {list(claim_metrics)}, but evidence states {list(evidence_metrics)}.",
                "claim_metrics": list(claim_metrics),
                "evidence_metrics": list(evidence_metrics),
            }

        # 3. Contradiction & absolute claim detection
        absolute_patterns = [
            r"completely replaces\b",
            r"requires zero\b",
            r"never requires\b",
            r"eliminates all\b",
        ]
        coexistence_patterns = [
            r"in conjunction with\b",
            r"used together with\b",
            r"combined with\b",
            r"used in conjunction\b",
        ]
        has_absolute = any(re.search(pat, c_lower) for pat in absolute_patterns)
        has_coexistence = any(re.search(pat, e_lower) for pat in coexistence_patterns)
        if has_absolute and has_coexistence:
            return {
                "status": "unsupported",
                "confidence": 0.92,
                "overlap_score": 0.4,
                "reason": "Contradiction: claim asserts complete replacement, but evidence indicates complementary/conjunction usage.",
            }

        # 4. Token containment and lexical grounding
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "this", "that", "these", "those",
            "we", "our", "it", "its", "from", "over", "under", "via", "through"
        }
        claim_words = {w for w in re.findall(r"\b[a-zA-Z0-9_\-\[\]]+\b", c_lower) if w not in stopwords and len(w) > 2}
        evidence_words = {w for w in re.findall(r"\b[a-zA-Z0-9_\-\[\]]+\b", e_lower) if w not in stopwords and len(w) > 2}

        if not claim_words:
            overlap = 1.0
        else:
            intersection = claim_words & evidence_words
            overlap = len(intersection) / len(claim_words)

        if overlap >= 0.45:
            return {
                "status": "supported",
                "confidence": min(round(0.5 + overlap * 0.5, 4), 1.0),
                "overlap_score": round(overlap, 4),
                "reason": f"Claim content words have high grounding overlap ({round(overlap*100, 1)}%) with evidence.",
            }
        else:
            return {
                "status": "unsupported",
                "confidence": round(1.0 - overlap, 4),
                "overlap_score": round(overlap, 4),
                "reason": f"Insufficient grounding overlap ({round(overlap*100, 1)}%) between claim and evidence.",
            }

    def evaluate_claim_support_dataset(
        self,
        eval_pairs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate claim support verification accuracy over a curated gold test set.
        """
        total = len(eval_pairs)
        if total == 0:
            return {"total": 0, "accuracy": 1.0, "results": []}

        correct = 0
        detailed = []

        for item in eval_pairs:
            claim = item.get("claim", "")
            evidence = item.get("evidence_text", "")
            expected = item.get("ground_truth_label", "")

            pred = self.verify_claim_support(claim, evidence)
            is_correct = (pred["status"] == expected)
            if is_correct:
                correct += 1

            detailed.append({
                "id": item.get("id"),
                "claim": claim,
                "expected": expected,
                "predicted": pred["status"],
                "is_correct": is_correct,
                "confidence": pred.get("confidence"),
                "reason": pred.get("reason"),
            })

        accuracy = round(correct / total, 4)
        return {
            "total_pairs": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "detailed_results": detailed,
        }

