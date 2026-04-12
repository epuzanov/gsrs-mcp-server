"""Tests for EmbeddingService behavior."""
import unittest
from unittest.mock import patch

from app.services.embedding import EmbeddingService


class TestEmbeddingService(unittest.TestCase):
    def test_embedding_service_verifies_ssl_by_default(self):
        seen = {}

        class FakeClient:
            def __init__(self, *, timeout, verify):
                seen["timeout"] = timeout
                seen["verify"] = verify

        with patch("app.services.embedding.httpx.Client", FakeClient):
            service = EmbeddingService(api_key="test-key")
            client = service.client

        self.assertIsNotNone(client)
        self.assertEqual(seen["timeout"], 60.0)
        self.assertTrue(seen["verify"])

    def test_embedding_service_can_disable_ssl_verification(self):
        seen = {}

        class FakeClient:
            def __init__(self, *, timeout, verify):
                seen["timeout"] = timeout
                seen["verify"] = verify

        with patch("app.services.embedding.httpx.Client", FakeClient):
            service = EmbeddingService(api_key="test-key", verify_ssl=False)
            client = service.client

        self.assertIsNotNone(client)
        self.assertEqual(seen["timeout"], 60.0)
        self.assertFalse(seen["verify"])

    def test_embed_uses_full_openai_endpoint_and_parses_data(self):
        seen = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

        class FakeClient:
            def __init__(self, *, timeout, verify):
                pass

            def post(self, url, headers, json):
                seen["url"] = url
                seen["headers"] = headers
                seen["json"] = json
                return FakeResponse()

        with patch("app.services.embedding.httpx.Client", FakeClient):
            service = EmbeddingService(
                api_key="test-key",
                url="https://api.openai.com/v1/embeddings",
            )
            result = service.embed("hello")

        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(seen["url"], "https://api.openai.com/v1/embeddings")
        self.assertEqual(seen["json"]["input"], "hello")
        self.assertEqual(seen["json"]["encoding_format"], "float")

    def test_embed_batch_supports_ollama_embeddings_response(self):
        seen = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

        class FakeClient:
            def __init__(self, *, timeout, verify):
                pass

            def post(self, url, headers, json):
                seen["url"] = url
                seen["headers"] = headers
                seen["json"] = json
                return FakeResponse()

        with patch("app.services.embedding.httpx.Client", FakeClient):
            service = EmbeddingService(
                api_key="ollama",
                url="http://localhost:11434/api/embed",
            )
            result = service.embed_batch(["hello", "world"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(seen["url"], "http://localhost:11434/api/embed")
        self.assertEqual(
            seen["json"],
            {"model": "text-embedding-3-small", "dimensions": 1536, "input": ["hello", "world"]},
        )

    def test_embed_supports_flat_ollama_embedding_response(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embeddings": [0.5, 0.6]}

        class FakeClient:
            def __init__(self, *, timeout, verify):
                pass

            def post(self, url, headers, json):
                return FakeResponse()

        with patch("app.services.embedding.httpx.Client", FakeClient):
            service = EmbeddingService(api_key="ollama", url="http://localhost:11434/api/embed")
            result = service.embed("hello")

        self.assertEqual(result, [0.5, 0.6])


if __name__ == "__main__":
    unittest.main()
