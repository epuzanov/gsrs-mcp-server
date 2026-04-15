"""Tests for MCP server wiring, readiness semantics, and MCP smoke paths."""
import asyncio
import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.config import settings
from app.models.api import AskResponse, Citation, QueryResult
from app.models.db import VectorDocument


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

    def search_by_example(self, example, top_k=20, mode="match"):
        return []

    def similarity_search(self, embedding, top_k=10, filters=None):
        return []


class FakeQueryPipeline:
    def __init__(self, response=None, diagnostics=None):
        self._response = response or self._default_response()
        self._diagnostics = diagnostics or {
            "retrieval_mode": "hybrid",
            "answer_generation": {
                "mode": "template_fallback",
                "error_type": None,
            },
            "stages": [],
        }
        self.identifier_router = SimpleNamespace(route=lambda query, top_k=10: None)

    def _default_response(self):
        document = VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="codes",
            text="CAS code for aspirin is 50-78-2.",
            embedding=[0.0] * settings.embedding_dimension,
            metadata_json={"canonical_name": "Aspirin", "codes": [{"codeSystem": "CAS", "code": "50-78-2"}]},
        )
        return AskResponse(
            query="What is the CAS code for aspirin?",
            rewritten_queries=["What is the CAS code for aspirin?"],
            applied_filters={"sections": ["codes"]},
            answer="The CAS code for aspirin is 50-78-2.",
            citations=[
                Citation(
                    chunk_id=document.chunk_id,
                    document_id=str(document.document_id),
                    section=document.section,
                    quote="CAS code for aspirin is 50-78-2.",
                )
            ],
            evidence_chunks=[QueryResult(chunk=document, score=0.92)],
            confidence=0.92,
            abstained=False,
            degraded=True,
            degraded_reason="Answer generation provider unavailable; returned retrieval-grounded fallback answer.",
            debug=None,
        )

    def ask(self, request):
        response, _ = self.ask_with_diagnostics(request)
        return response

    def ask_with_diagnostics(self, request):
        response = self._response.model_copy(deep=True)
        response.query = request.query
        if request.debug:
            response.debug = {
                "retrieval_mode": self._diagnostics.get("retrieval_mode", "hybrid"),
            }
        return response, dict(self._diagnostics)


def _make_abstained_response() -> AskResponse:
    return AskResponse(
        query="unknown identifier",
        rewritten_queries=["unknown identifier"],
        applied_filters={},
        answer=None,
        citations=[],
        evidence_chunks=[],
        confidence=0.0,
        abstained=True,
        abstain_reason="No exact metadata match found for the identifier-first lookup.",
        degraded=True,
        degraded_reason="Answer generation provider unavailable; returning retrieval-only response.",
        debug=None,
    )


def _make_identifier_diagnostics() -> dict:
    return {
        "retrieval_mode": "identifier-first:code",
        "answer_generation": {
            "mode": "template_fallback",
            "error_type": None,
        },
        "stages": [],
    }


class FakeRuntime:
    def __init__(
        self,
        *,
        ready=True,
        retrieval_ready=True,
        vector_ready=True,
        chunker_ready=True,
        gsrs_api_ready=True,
    ):
        self.backend_name = "chroma"
        self.ready = ready
        self.degraded = not retrieval_ready or not gsrs_api_ready
        if not chunker_ready:
            self.degraded = True
        self.degraded_summary = None if not self.degraded else (
            "Chunker initialization failed: gsrs model is unavailable."
            if not chunker_ready
            else ("GSRS upstream validation failed: timed out." if not gsrs_api_ready else "Optional dependency degraded.")
        )
        self.metrics = FakeMetrics()
        self.vector_db = FakeVectorDb()
        self.embedding_service = SimpleNamespace(embed=lambda query: [0.0] * settings.embedding_dimension)
        self.query_pipeline = FakeQueryPipeline()
        self.chunker = object() if chunker_ready else None
        self.llm_service = None
        self.gsrs_api = object()
        self._retrieval_ready = retrieval_ready
        self._vector_ready = vector_ready
        self._chunker_ready = chunker_ready
        self._gsrs_api_ready = gsrs_api_ready
        self.initialized = False
        self.stopped = False
        self.components = {
            "vector_db": SimpleNamespace(
                ready=vector_ready,
                required=True,
                error=None if vector_ready else "Vector backend initialization failed: database is offline.",
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
            "chunker": SimpleNamespace(
                ready=chunker_ready,
                required=False,
                error=None if chunker_ready else "Chunker initialization failed: gsrs model is unavailable.",
                details={},
            ),
            "gsrs_api": SimpleNamespace(
                ready=gsrs_api_ready,
                required=False,
                error=None if gsrs_api_ready else "GSRS upstream validation failed: timed out.",
                details={"base_url": "https://gsrs.example.test/api/v1"},
            ),
        }

    def initialize(self):
        self.initialized = True

    def shutdown(self):
        self.stopped = True

    def get_status_payload(self):
        return {
            "status": "ready_degraded" if self.ready and self.degraded else ("ready" if self.ready else "not_ready"),
            "ready": self.ready,
            "degraded": self.degraded,
            "backend": self.backend_name,
            "statistics": {"total_chunks": 0, "total_substances": 0},
            "components": {
                "vector_db": {
                    "required": True,
                    "ready": self._vector_ready,
                    "error": None if self._vector_ready else "Vector backend initialization failed: database is offline.",
                    "details": {"statistics": {"total_chunks": 0, "total_substances": 0}},
                },
                "chunker": {
                    "required": False,
                    "ready": self._chunker_ready,
                    "error": None if self._chunker_ready else "Chunker initialization failed: gsrs model is unavailable.",
                    "details": {},
                },
            },
            "readiness_summary": "Core retrieval dependencies are ready." if self.ready else self.retrieval_unavailable_reason(),
            "degraded_summary": None if not self.degraded else (
                "Chunker initialization failed: gsrs model is unavailable."
                if not self._chunker_ready
                else "GSRS upstream validation failed: timed out."
            ),
            "required_component_errors": {} if self.ready else {"embedding": self.retrieval_unavailable_reason()},
            "optional_component_errors": (
                {"chunker": "Chunker initialization failed: gsrs model is unavailable."}
                if not self._chunker_ready
                else ({"gsrs_api": "GSRS upstream validation failed: timed out."} if not self._gsrs_api_ready else {})
            ),
            "metrics": self.metrics.snapshot(),
        }

    def vector_backend_available(self):
        return self._vector_ready

    def vector_backend_unavailable_reason(self):
        return "Vector backend initialization failed: database is offline."

    def metadata_lookup_available(self):
        return self._vector_ready

    def metadata_lookup_unavailable_reason(self):
        return self.vector_backend_unavailable_reason()

    def retrieval_available(self):
        return self._vector_ready and self._retrieval_ready

    def retrieval_unavailable_reason(self):
        if not self._vector_ready:
            return self.vector_backend_unavailable_reason()
        return "Embedding provider is not ready."

    def ingestion_available(self):
        return self._vector_ready and self._retrieval_ready and self._chunker_ready

    def ingestion_unavailable_reason(self):
        if not self._vector_ready:
            return self.vector_backend_unavailable_reason()
        if not self._retrieval_ready:
            return "Embedding provider is not ready."
        return "Chunker initialization failed: gsrs model is unavailable."

    def gsrs_api_available(self):
        return self._gsrs_api_ready

    def gsrs_api_unavailable_reason(self):
        return "GSRS upstream validation failed: timed out."


class TestMCPConfig(unittest.TestCase):
    def test_default_config(self):
        self.assertEqual(settings.mcp_username, "admin")
        self.assertEqual(settings.default_top_k, 5)

    def test_token_verifier_accepts_mcp_password(self):
        from app.main import SimpleTokenVerifier

        verifier = SimpleTokenVerifier()
        token = asyncio.run(verifier.verify_token(settings.mcp_password))
        self.assertIsNotNone(token)

    def test_token_verifier_rejects_other_values(self):
        from app.main import SimpleTokenVerifier

        verifier = SimpleTokenVerifier()
        token = asyncio.run(verifier.verify_token("wrong-token"))
        self.assertIsNone(token)

    def test_auth_settings_enable_bearer_auth_without_username(self):
        from app.main import _build_auth_settings

        auth, verifier = _build_auth_settings(
            settings.model_copy(update={"mcp_username": "", "mcp_password": "secret-token"})
        )

        self.assertIsNotNone(auth)
        self.assertIsNotNone(verifier)


class TestHealthRoutes(unittest.TestCase):
    def test_health_initializes_runtime_when_first_request_arrives(self):
        from app import main

        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        fake_runtime.initialized = False

        with patch.object(main, "runtime", fake_runtime):
            response = asyncio.run(main.health_check(None))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(fake_runtime.initialized)

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

    def test_health_reports_ready_degraded_status_when_optional_dependency_is_down(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=True, retrieval_ready=True, gsrs_api_ready=False)):
            response = asyncio.run(main.health_check(None))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "ready_degraded")
        self.assertTrue(payload["ready"])
        self.assertTrue(payload["degraded"])

    def test_health_can_be_ready_while_chunker_is_degraded(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=True, retrieval_ready=True, chunker_ready=False)):
            response = asyncio.run(main.health_check(None))

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "ready_degraded")
        self.assertTrue(payload["ready"])
        self.assertIn("chunker", payload["optional_component_errors"])


class TestLegacyERIRoutes(unittest.IsolatedAsyncioTestCase):
    async def test_eri_query_allows_unauthenticated_compatibility_requests(self):
        from app import main as main_module

        main = importlib.reload(main_module)
        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        fake_runtime.query_pipeline.identifier_router = SimpleNamespace(route=lambda query, top_k=10: None)

        with patch.object(main, "runtime", fake_runtime):
            app = main.mcp.streamable_http_app()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                timeout=30.0,
            ) as client:
                response = await client.post("/eri/query", json={"query": "aspirin"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json())

    async def test_eri_query_returns_legacy_result_shape_for_open_webui_tool(self):
        from app import main as main_module

        main = importlib.reload(main_module)

        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        document = VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="codes",
            source_url="https://example.test/substances/aspirin",
            text="CAS code for aspirin is 50-78-2.",
            embedding=[0.0] * settings.embedding_dimension,
            metadata_json={"canonical_name": "Aspirin"},
        )
        result = SimpleNamespace(document=document, score=0.99)
        route_result = SimpleNamespace(
            route="code",
            results=[result],
            matched_value="50-78-2",
            example={"reliable_codes": {"CAS": "50-78-2"}},
        )
        fake_runtime.query_pipeline.identifier_router = SimpleNamespace(
            route=lambda query, top_k=10: route_result
        )

        with patch.object(main, "runtime", fake_runtime):
            app = main.mcp.streamable_http_app()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                timeout=30.0,
            ) as client:
                response = await client.post(
                    "/eri/query",
                    json={"query": "CAS 50-78-2", "top_k": 3},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-gsrs-retrieval-mode"), "identifier-first:code")
        payload = response.json()
        self.assertIn("results", payload)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["text"], "CAS code for aspirin is 50-78-2.")
        self.assertEqual(payload["results"][0]["score"], 0.99)
        self.assertEqual(payload["results"][0]["metadata"]["section"], "codes")
        self.assertEqual(payload["results"][0]["metadata"]["source_url"], "https://example.test/substances/aspirin")
        self.assertEqual(payload["results"][0]["metadata"]["document_id"], str(document.document_id))

    async def test_eri_query_reports_runtime_unavailable(self):
        from app import main as main_module

        main = importlib.reload(main_module)
        fake_runtime = FakeRuntime(ready=False, retrieval_ready=False)

        with patch.object(main, "runtime", fake_runtime):
            app = main.mcp.streamable_http_app()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                timeout=30.0,
            ) as client:
                response = await client.post("/eri/query", json={"query": "aspirin"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("Retrieval is currently unavailable", response.json()["detail"])


class TestToolBehavior(unittest.TestCase):
    def test_gsrs_ask_reports_runtime_unavailable(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=False, retrieval_ready=False)):
            output = asyncio.run(main.gsrs_ask("What is aspirin?"))

        self.assertIn("Retrieval is currently unavailable", output)
        self.assertIn("Embedding provider is not ready", output)

    def test_gsrs_ingest_reports_chunker_specific_failure(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=False, retrieval_ready=True, chunker_ready=False)):
            output = asyncio.run(main.gsrs_ingest("{}"))

        self.assertIn("Ingestion is currently unavailable", output)
        self.assertIn("Chunker initialization failed", output)

    def test_gsrs_api_search_reports_upstream_unavailable(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=True, retrieval_ready=True, gsrs_api_ready=False)):
            output = asyncio.run(main.gsrs_api_search("aspirin"))

        self.assertIn("GSRS API search is currently unavailable", output)
        self.assertIn("GSRS upstream validation failed", output)

    def test_similarity_search_only_requires_vector_backend(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=False, retrieval_ready=False, vector_ready=True)):
            output = asyncio.run(main.gsrs_similarity_search('{"uuid":"12345678-1234-1234-1234-123456789abc"}'))

        self.assertIn("No similar substances found", output)

    def test_gsrs_ask_redirects_gsrs_json_to_similarity_search(self):
        from app import main

        payload = json.dumps(
            {
                "uuid": "12345678-1234-1234-1234-123456789abc",
                "names": [{"name": "Aspirin"}],
                "codes": [],
            }
        )

        with patch.object(main, "runtime", FakeRuntime(ready=False, retrieval_ready=False)):
            output = asyncio.run(main.gsrs_ask(payload))

        self.assertIn("gsrs_ask only accepts natural-language questions", output)
        self.assertIn("gsrs_similarity_search", output)

    def test_gsrs_ask_degraded_answer_includes_citations(self):
        from app import main

        with patch.object(main, "runtime", FakeRuntime(ready=True, retrieval_ready=True)):
            output = asyncio.run(main.gsrs_ask("What is the CAS code for aspirin?"))

        self.assertIn("Direct answer:", output)
        self.assertIn("Mode:", output)
        self.assertIn("Supporting evidence:", output)
        self.assertIn("Citations:", output)
        self.assertIn("50-78-2", output)

    def test_gsrs_ask_identifier_debug_output_includes_query_type(self):
        from app import main

        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        fake_runtime.query_pipeline = FakeQueryPipeline(
            diagnostics=_make_identifier_diagnostics()
        )

        with patch.object(main, "runtime", fake_runtime):
            output = asyncio.run(main.gsrs_ask("CAS 50-78-2", debug=True))

        self.assertIn('"query_type": "code"', output)
        self.assertIn('"retrieval_mode": "identifier-first:code"', output)

    def test_gsrs_ask_abstains_when_no_grounded_answer_is_available(self):
        from app import main

        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        fake_runtime.query_pipeline = FakeQueryPipeline(
            response=_make_abstained_response(),
            diagnostics=_make_identifier_diagnostics(),
        )

        with patch.object(main, "runtime", fake_runtime):
            output = asyncio.run(main.gsrs_ask("CAS DOES-NOT-EXIST"))

        self.assertIn("Insufficient evidence to answer confidently", output)
        self.assertIn("Uncertainty:", output)
        self.assertNotIn("Citations:", output)

    def test_gsrs_retrieve_uses_hybrid_pipeline_for_natural_language_queries(self):
        from app import main

        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        document = VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="codes",
            text="CAS code for aspirin is 50-78-2.",
            embedding=[0.0] * settings.embedding_dimension,
            metadata_json={"canonical_name": "Aspirin"},
        )
        candidate = SimpleNamespace(document=document, score=0.93)
        fake_runtime.query_pipeline = SimpleNamespace(
            rewrite_service=SimpleNamespace(
                rewrite=lambda query: SimpleNamespace(
                    canonical_query="what is the cas code for aspirin?",
                    rewrites=["CAS aspirin", "aspirin code"],
                    filters={"sections": ["codes"]},
                    intent="identifier_lookup",
                )
            ),
            filter_builder=SimpleNamespace(
                build=lambda request_filters=None, substance_classes=None, sections=None, inferred_filters=None: {
                    "sections": ["codes"]
                }
            ),
            identifier_router=SimpleNamespace(route=lambda query, top_k=10: None),
            hybrid_retriever=SimpleNamespace(retrieve=lambda queries, filters=None: [candidate]),
            reranker=SimpleNamespace(
                rerank=lambda candidates, query, rewritten_queries=None, filters=None: candidates
            ),
        )

        with patch.object(main, "runtime", fake_runtime):
            output = asyncio.run(main.gsrs_retrieve("What is the CAS code for aspirin?", debug=True))

        self.assertIn("Found 1 result(s)", output)
        self.assertIn("50-78-2", output)
        self.assertIn('"retrieval_mode": "hybrid"', output)
        self.assertIn('"canonical_query": "what is the cas code for aspirin?"', output)
        self.assertIn('"applied_filters"', output)


class TestMCPTransportSmoke(unittest.IsolatedAsyncioTestCase):
    async def test_streamable_http_requires_bearer_token(self):
        from app import main

        app = main.mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=30.0,
        ) as client:
            response = await client.post("/mcp", json={})

        self.assertEqual(response.status_code, 401)
        payload = response.json()
        self.assertEqual(payload["error"], "invalid_token")

    async def test_streamable_http_smoke_path(self):
        from app import main as main_module

        main = importlib.reload(main_module)

        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        with patch.object(main, "runtime", fake_runtime):
            app = main.mcp.streamable_http_app()
            async with main.mcp.session_manager.run():
                transport = httpx.ASGITransport(app=app)
                headers = {"Authorization": f"Bearer {settings.mcp_password}"}

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

    async def test_streamable_http_query_smoke_path_returns_grounded_answer(self):
        from app import main as main_module

        main = importlib.reload(main_module)

        fake_runtime = FakeRuntime(ready=True, retrieval_ready=True)
        with patch.object(main, "runtime", fake_runtime):
            app = main.mcp.streamable_http_app()
            async with main.mcp.session_manager.run():
                transport = httpx.ASGITransport(app=app)
                headers = {"Authorization": f"Bearer {settings.mcp_password}"}

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
                            result = await session.call_tool(
                                "gsrs_ask",
                                {"query": "What is the CAS code for aspirin?"},
                            )
                            text_blocks = [block.text for block in result.content if hasattr(block, "text")]
                            combined = "\n".join(text_blocks)
                            self.assertIn("Direct answer:", combined)
                            self.assertTrue("Citations:" in combined or "Insufficient evidence" in combined)
