import logging
import time
from typing import Any, Dict, List, Optional
from src.generation.ollama_client import OllamaClient
from src.generation.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ResearchAnswerGenerator:
    """
    Synthesizes evidence-grounded research answers from retrieved scientific paper chunks.
    Ensures strict citation provenance and trace back to source papers.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.ollama_client = ollama_client or OllamaClient()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce a grounded answer with cited papers and evidence attribution.
        """
        start_time = time.time()

        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "No relevant literature or evidence was found in the indexed corpus for this research query.",
                "evidence_count": 0,
                "sources": [],
                "evidence_chunks": [],
                "generation_time_sec": round(time.time() - start_time, 2),
                "is_ollama_live": self.ollama_client.is_available(),
            }

        # Build prompt
        user_prompt = self.prompt_builder.build_user_prompt(query, retrieved_chunks)
        system_prompt = self.prompt_builder.SYSTEM_PROMPT

        # Call LLM
        raw_answer = self.ollama_client.generate(prompt=user_prompt, system=system_prompt)

        # Extract unique sources
        seen_papers = set()
        sources = []
        for c in retrieved_chunks:
            pid = c.get("paper_id")
            title = c.get("title")
            sec = c.get("section")
            page = c.get("page")
            score = c.get("score")

            key = (pid, sec)
            if key not in seen_papers:
                seen_papers.add(key)
                sources.append({
                    "paper_id": pid,
                    "title": title,
                    "section": sec,
                    "page": page,
                    "relevance_score": score,
                })

        elapsed = round(time.time() - start_time, 2)

        return {
            "query": query,
            "answer": raw_answer,
            "evidence_count": len(retrieved_chunks),
            "sources": sources,
            "evidence_chunks": retrieved_chunks,
            "generation_time_sec": elapsed,
            "is_ollama_live": self.ollama_client.is_available(),
        }
