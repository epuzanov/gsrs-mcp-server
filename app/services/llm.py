"""
GSRS MCP Server - LLM Service
Provides OpenAI-compatible LLM calls for query rewrite, answering, etc.
"""
import json
import time
from typing import Any, Dict

import httpx

from app.config import settings


class LLMService:
    """OpenAI-compatible LLM client for structured and text completion."""

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        model: str | None = None,
        verify_ssl: bool | None = None,
        timeout: int | None = None,
        max_retries: int = 1,
        retry_backoff_ms: int = 250,
    ):
        self.api_key = api_key or settings.llm_api_key
        self.url = (url or settings.llm_url).rstrip("/")
        self.model = model or settings.llm_model
        self.verify_ssl = verify_ssl if verify_ssl is not None else settings.llm_verify_ssl
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(verify=self.verify_ssl, timeout=self.timeout)
        return self._client

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request expecting structured JSON output.
        If `schema` is provided, it will be included as json_schema for structured output.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                },
            }

        resp = self._post_with_retry(payload)
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return json.loads(content)

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat completion request expecting plain text output."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        resp = self._post_with_retry(payload)
        data = resp.json()

        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _post_with_retry(self, payload: Dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.post(
                    self.url,
                    headers=self._build_headers(),
                    json=payload,
                )
                resp.raise_for_status()
                return resp
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_ms / 1000)

        raise RuntimeError(
            f"LLM request failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def get_model_info(self) -> Dict[str, Any]:
        """Return non-sensitive provider information."""
        return {
            "model": self.model,
            "url": self.url,
            "verify_ssl": self.verify_ssl,
            "timeout": self.timeout,
        }

    def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
