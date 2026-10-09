from typing import Any, Dict, List


class PromptBuilder:
    """Constructs evidence-grounded research prompts for scientific synthesis."""

    SYSTEM_PROMPT = (
        "You are an expert scientific AI Research Assistant. Your goal is to provide rigorous, "
        "evidence-grounded answers to research questions based EXCLUSIVELY on the provided scientific literature passages.\n\n"
        "STRICT RESEARCH RULES:\n"
        "1. Ground every claim directly in the provided evidence. Cite sources using [Evidence X] or (Author/Title, Section).\n"
        "2. Do NOT hallucinate, extrapolate beyond given facts, or invent external citations.\n"
        "3. If the provided evidence does not contain sufficient details to answer part or all of the question, "
        "explicitly declare: 'The retrieved corpus provides insufficient evidence for this aspect.'\n"
        "4. Organize your response into: Key Findings, Technical Details/Methodology, Comparative Insights/Limitations, and Cited Sources.\n"
        "5. Maintain objective, academic tone."
    )

    def build_evidence_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into numbered, attributed evidence blocks."""
        if not chunks:
            return "No relevant evidence chunks retrieved."

        blocks = []
        for i, c in enumerate(chunks, 1):
            title = c.get("title", "Unknown Title")
            section = c.get("section", "General")
            page = c.get("page", 1)
            text = c.get("text", "").strip()

            block = (
                f"[Evidence {i}]\n"
                f"Paper: {title}\n"
                f"Section: {section} (Page {page})\n"
                f"Passage:\n{text}\n"
            )
            blocks.append(block)

        return "\n----------------------------------------\n".join(blocks)

    def build_user_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Construct user prompt integrating question and evidence blocks."""
        evidence_str = self.build_evidence_context(chunks)

        return (
            f"RESEARCH QUESTION:\n{query}\n\n"
            f"RETRIEVED SCIENTIFIC EVIDENCE:\n"
            f"{evidence_str}\n\n"
            "TASK:\n"
            "Synthesize a rigorous, structured scientific answer to the research question strictly using the evidence passages above. "
            "Attribute your claims with bracketed citations (e.g. [Evidence 1], [Evidence 2]). Conclude with a summary of supporting papers."
        )
