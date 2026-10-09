import logging
from typing import List, Optional
import httpx
from src.config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local LLM inference using Ollama API with dynamic model resolution."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT

    def get_available_models(self) -> List[str]:
        """Fetch list of model tags currently pulled in Ollama."""
        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                    return models
        except Exception:
            pass
        return []

    def is_available(self) -> bool:
        """Check if local Ollama daemon is active and responding."""
        models = self.get_available_models()
        return len(models) > 0

    def _resolve_model(self) -> str:
        """Resolve model name, picking the best match among installed models."""
        available = self.get_available_models()
        if not available:
            return self.model

        if self.model in available:
            return self.model

        # Fuzzy match (e.g. llama3 matches llama3:latest)
        base_req = self.model.split(":")[0].lower()
        for m in available:
            if base_req in m.lower():
                logger.info("Resolved requested model '%s' to available '%s'", self.model, m)
                return m

        # Filter out embedding-only models if possible
        generative = [m for m in available if "bge" not in m.lower() and "embed" not in m.lower()]
        if generative:
            logger.info("Using available generative model '%s'", generative[0])
            return generative[0]

        return available[0]

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Send prompt to Ollama generation endpoint."""
        target_model = self._resolve_model()
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("response", "").strip()
        except httpx.ConnectError:
            logger.warning(
                "Ollama daemon not reachable at %s. Falling back to structured synthesis generator.",
                self.base_url,
            )
            return self._fallback_synthesis(prompt)
        except Exception as e:
            logger.error("Error calling Ollama API: %s", str(e))
            return self._fallback_synthesis(prompt)

    def _fallback_synthesis(self, prompt: str) -> str:
        """Fallback synthesis when local Ollama is offline."""
        return (
            "### [Notice: Live Ollama LLM was unavailable for synthesis]\n\n"
            "#### Direct Evidence Summary:\n"
            "The system retrieved high-relevance evidence passages addressing your query. "
            "Please review the extracted passages below for exact empirical findings and citations."
        )
