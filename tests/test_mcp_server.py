"""Tests for MCP server wiring, readiness semantics, and MCP smoke paths."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.config import settings
from app.models.api import AskResponse


class FakeMetrics:
    def increment(self, name, value=1):
        return None

    def observe_latency(self, name, latency_ms):
        return None

    def snapshot(self):
        return {"counters": {}, "latencies": {}}


class FakeVectorDb:
    def get_statistics(self):
        return {"total_chunks": 0, "total_substances": 0}

    def delete_documents_by_substance(self, substance_uuid):
        return 1


class FakeQueryPipeline:
    def __init__(self):
        self.identifier_router = None

    def ask(self, request):
        return AskResponse(
            query=request.query,
            rewritten_queries=[request.query],
            applied_filters={},
            answer="A grounded answer.",
            citations=[],
            evidence_chunks=[],
            confidence=0.8,
            abstained=False,
            degraded=True,
            degraded_reason="Answer generation provider unavailable; returned retrieval-grounded fallback answer.",
            debug={"retrieval_mode": "hybrid"} if request.debug else None,
        )


class FakeRuntime:
    def __init__(self, *, ready=True, retrieval_ready=True):
        self.backend_name = "chroma"
        self.ready = ready
        self.degraded = not retrieval_ready
        self.metrics = FakeMetrics()
        self.vector_db = FakeVectorDb()
        self.query_pipeline = FakeQueryPipeline()
        self.chunker = object()
        self.llm_service = None
        self.gsrs_api = None
        self._retrieval_ready = retrieval_ready
        self.initialized = False
        self.stopped = False
        self.components = {
            "vector_db": SimpleNamespace(
                ready=True,
                required=True,
                error=None,
                details={"statistics": {"total_chunks": 0, "total_substances": 0}},
            ),
            "embedding": SimpleNamespace(
                ready=retrieval_ready,
                required=True,
                error=None if retrieval_ready else "Embedding provider is not ready.",
                details={},
            ),
            "answer_generation": SimpleNamespace(
                ready=False,
                required=False,
                error="Answer generation provider unavailable.",
                details={},
            ),
        }

    def initialize(self):
        self.initialized = True

    def shutdown(self):
        self.stopped = True

    def get_status_payload(self):
        return {
            "status": "ready" if self.ready else "starting_or_degraded",
            "ready": self.ready,
            "degraded": self.degraded,
            "backend": self.backend_name,
            "statistics": {"total_chunks": 0, "total_substances": 0},
            "components": {
                "vector_db": {
                    "required": True,
                    "ready": True,
                    "error": None,
                    "details": {"statistics": {"total_chunks": 0, "total_substances": 0}},
                },
            },
            "metrics": self.metrics.snapshot(),
        }

    def retrieval_available(self):
        return self._retrieval_ready

    def retrieval_unavailable_reason(self):
        return "Embedding provider is not ready."


class TestMCPConfig(unittest.TestCase):
    def test_default_config(self):
        self.assertEqual(settings.api_username, "admin")
        self.assertEqual(settings.default_top_k, 5)

    def test_token_verifier_accepts_api_password(self):
        from app.main import SimpleTokenVerifier

        verifier = SimpleTokenVerifier()
        token = asyncio.run(verifier.verify_token(settings.api_password))
        self.assertIsNotNone(token)

    def test_token_verifier_rejects_other_values(self):
        from app.main import SimpleTokenVerifier

        verifier = SimpleTokenVerifier()
        token = asyncio.run(verifier.verify_token("wrong-token"))
        self.assertIsNone(token)


class TestHealthRoutes(unittest.TestCase):
    def test_readyz_reports_empty_database_as_ready(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=True, retrieval_ready=True)):
            response = asyncio.run(main.readiness_check(None))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["statistics"]["total_chunks"], 0)
        self.assertEqual(payload["statistics"]["total_substances"], 0)

    def test_health_includes_liveness_and_components(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=False, retrieval_ready=False)):
            response = asyncio.run(main.health_check(None))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertTrue(payload["live"])
        self.assertIn("components", payload)
        self.assertFalse(payload["ready"])


class TestToolBehavior(unittest.TestCase):
    def test_gsrs_ask_reports_runtime_unavailable(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=False, retrieval_ready=False)):
            output = asyncio.run(main.gsrs_ask("What is aspirin?"))

        self.assertIn("Retrieval is currently unavailable", output)
        self.assertIn("Embedding provider is not ready", output)


class TestMCPTransportSmoke(unittest.IsolatedAsyncioTestCase):
    async def test_streamable_http_smoke_path(self):
        from app import main

        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        with patch.object(main, "runtime", fake_runtime):
            app = main.mcp.streamable_http_app()
            async with main.mcp.session_manager.run():
                transport = httpx.ASGITransport(app=app)
                headers = {"Authorization": f"Bearer {settings.api_password}"}

                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers=headers,
                    timeout=30.0,
                ) as client:
                    async with streamable_http_client(
                        "http://testserver/mcp",
                        http_client=client,
                        terminate_on_close=False,
                    ) as (read_stream, write_stream, _):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            tool_names = {tool.name for tool in tools.tools}
                            self.assertIn("gsrs_health", tool_names)

                            result = await session.call_tool("gsrs_health", {})
                            text_blocks = [block.text for block in result.content if hasattr(block, "text")]
                            combined = "\n".join(text_blocks)
                            self.assertIn('"backend"', combined)
