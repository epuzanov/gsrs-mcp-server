"""
GSRS MCP Server - OpenAI Embeddings Service

Simple OpenAI-compatible embeddings service.
"""
from typing import Any, List

import httpx


class EmbeddingService:
    """
    OpenAI-compatible embeddings service.

    Works with:
    - OpenAI API (api.openai.com)
    - Azure OpenAI
    - Any OpenAI-compatible API (vLLM, Ollama, etc.)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        url: str = "https://api.openai.com/v1/embeddings",
        dimension: int = 1536,
        verify_ssl: bool = True,
    ):
        """
        Initialize embeddings service.

        Args:
            api_key: API key for authentication
            model: Model name (default: text-embedding-3-small)
            url: Full embeddings endpoint URL
            dimension: Embedding dimension (default: 1536)
            verify_ssl: Whether to verify TLS certificates (default: True)
        """
        self.api_key = api_key
        self.model = model
        self.url = url.rstrip("/")
        self.dimension = dimension
        self.verify_ssl = verify_ssl
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=60.0, verify=self.verify_ssl)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, input_data: str | List[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "dimensions": self.dimension,
            "input": input_data,
        }
        if self.url.endswith("/embeddings"):
            payload["encoding_format"] = "float"
        return payload

    def _parse_embeddings(self, payload: dict[str, Any]) -> List[List[float]]:
        if "data" in payload:
            sorted_data = sorted(payload["data"], key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in sorted_data]

        if "embeddings" in payload:
            embeddings = payload["embeddings"]
            if embeddings and isinstance(embeddings[0], (int, float)):
                return [embeddings]
            return embeddings

        raise ValueError("Embedding response did not contain 'data' or 'embeddings'")

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.post(
            self.url,
            headers=self._headers(),
            json=self._build_payload(text),
        )
        response.raise_for_status()
        return self._parse_embeddings(response.json())[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.post(
                self.url,
                headers=self._headers(),
                json=self._build_payload(batch),
            )
            response.raise_for_status()
            embeddings.extend(self._parse_embeddings(response.json()))

        return embeddings

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "dimension": self.dimension,
            "url": self.url,
            "verify_ssl": self.verify_ssl,
        }

    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
