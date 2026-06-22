"""
GSRS MCP Server - OpenAI Embeddings Service

Simple OpenAI-compatible embeddings service.
"""
import time
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
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff_ms: int = 250,
        send_dimensions: bool = False,
        send_encoding_format: bool = False,
    ):
        """
        Initialize embeddings service.

        Args:
            api_key: API key for authentication
            model: Model name (default: text-embedding-3-small)
            url: Full embeddings endpoint URL
            dimension: Embedding dimension (default: 1536). Used for client-side
                validation of the response length; only sent in the request payload
                when ``send_dimensions`` is true (OpenAI v3 supports the
                ``dimensions`` truncation param; most other providers do not).
            verify_ssl: Whether to verify TLS certificates (default: True)
            send_dimensions: Whether to include ``dimensions`` in the request
                payload (OpenAI-only; off by default for portability).
            send_encoding_format: Whether to include ``encoding_format: float``
                in the request payload (OpenAI-only; off by default for
                portability — LiteLLM-in-front-of-Ollama rejects it).
        """
        self.api_key = api_key
        self.model = model
        self.url = url.rstrip("/")
        self.dimension = dimension
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        self.send_dimensions = send_dimensions
        self.send_encoding_format = send_encoding_format
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, verify=self.verify_ssl)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, input_data: str | List[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
        }
        # ``dimensions`` is only honored by OpenAI v3 models. Most
        # OpenAI-compatible providers (Ollama, vLLM, LiteLLM routing to
        # non-OpenAI backends) reject it with a 400. Only include it
        # when the operator has explicitly opted in.
        if self.send_dimensions and self.dimension > 0:
            payload["dimensions"] = self.dimension
        # ``encoding_format: float`` is OpenAI-only. LiteLLM rejects it
        # for Ollama-served models with a 400 (UnsupportedParamsError).
        # Only include it when the operator has explicitly opted in.
        if self.send_encoding_format and self.url.endswith("/embeddings"):
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
        response = self._post_with_retry(self._build_payload(text))
        return self._parse_embeddings(response.json())[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._post_with_retry(self._build_payload(batch))
            embeddings.extend(self._parse_embeddings(response.json()))

        return embeddings

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(
                    self.url,
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                return response
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_ms / 1000)

        raise RuntimeError(
            f"Embedding request failed after {self.max_retries + 1} attempt(s) "
            f"against {self.url}: {last_error}"
        ) from last_error

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "dimension": self.dimension,
            "url": self.url,
            "verify_ssl": self.verify_ssl,
            "timeout": self.timeout,
            "send_dimensions": self.send_dimensions,
            "send_encoding_format": self.send_encoding_format,
        }

    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
