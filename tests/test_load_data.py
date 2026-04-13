"""Tests for the load_data.py script functionality."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from scripts.load_data import (
    MCPConnectionSettings,
    MCPToolClient,
    fetch_all_substance_uuids,
    fetch_substance_by_uuid,
    ingest_batch_via_mcp,
    load_from_file,
    load_substances_from_api,
    parse_gsrs_file,
)


class TestParseGsrsFile(unittest.TestCase):
    """Tests for .gsrs file parsing."""

    def test_parse_gsrs_file_basic(self):
        """Test parsing basic .gsrs file."""
        substances = [
            {"uuid": "uuid-1", "substanceClass": "chemical"},
            {"uuid": "uuid-2", "substanceClass": "protein"},
        ]

        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                for sub in substances:
                    gz.write("\t\t" + json.dumps(sub) + "\n")

        try:
            parsed = list(parse_gsrs_file(temp_path))
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["uuid"], "uuid-1")
            self.assertEqual(parsed[1]["uuid"], "uuid-2")
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_with_empty_lines(self):
        """Test parsing .gsrs file with empty lines."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                gz.write("\t\t{\"uuid\": \"uuid-1\"}\n")
                gz.write("\t\t\n")
                gz.write("\n")
                gz.write("\t\t{\"uuid\": \"uuid-2\"}\n")

        try:
            parsed = list(parse_gsrs_file(temp_path))
            self.assertEqual(len(parsed), 2)
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_with_invalid_json(self):
        """Test parsing .gsrs file with invalid JSON lines."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                gz.write("\t\t{\"uuid\": \"uuid-1\"}\n")
                gz.write("\t\t{invalid json}\n")
                gz.write("\t\t{\"uuid\": \"uuid-2\"}\n")

        try:
            parsed = list(parse_gsrs_file(temp_path))
            self.assertEqual(len(parsed), 2)
        finally:
            os.unlink(temp_path)

    def test_parse_gsrs_file_single_tab(self):
        """Test parsing .gsrs file with single leading tab."""
        with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
            temp_path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                gz.write("\t{\"uuid\": \"uuid-1\"}\n")

        try:
            parsed = list(parse_gsrs_file(temp_path))
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["uuid"], "uuid-1")
        finally:
            os.unlink(temp_path)


class TestIngestBatchViaMCP(unittest.TestCase):
    """Tests for MCP client-based batch ingestion."""

    MCP_URL = "http://localhost:8000/mcp"

    def test_ingest_batch_via_mcp_success(self):
        """Test ingestion via a single MCP client session with successful results."""

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ensure_ingest_available(self):
                return None

            async def ingest_substance(self, substance):
                return f"Ingested **{substance['uuid']}** - 3 chunks."

        with patch("scripts.load_data.build_mcp_client", return_value=FakeClient()):
            result = ingest_batch_via_mcp(
                self.MCP_URL,
                True,
                [{"uuid": "test-1"}, {"uuid": "test-2"}],
            )

        self.assertEqual(result["successful"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["total_chunks"], 6)
        self.assertEqual(len(result["errors"]), 0)

    def test_ingest_batch_via_mcp_error(self):
        """Test ingestion via MCP that returns an unparseable result."""

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ensure_ingest_available(self):
                return None

            async def ingest_substance(self, substance):
                return "No chunks generated from substance."

        with patch("scripts.load_data.build_mcp_client", return_value=FakeClient()):
            result = ingest_batch_via_mcp(
                self.MCP_URL,
                True,
                [{"uuid": "bad"}],
            )

        self.assertEqual(result["successful"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["total_chunks"], 0)

    def test_ingest_batch_via_mcp_exception(self):
        """Test ingestion via MCP when a client call raises."""

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ensure_ingest_available(self):
                return None

            async def ingest_substance(self, substance):
                raise RuntimeError("MCP server error")

        with patch("scripts.load_data.build_mcp_client", return_value=FakeClient()):
            result = ingest_batch_via_mcp(
                self.MCP_URL,
                True,
                [{"uuid": "bad"}],
            )

        self.assertEqual(result["successful"], 0)
        self.assertEqual(result["failed"], 1)

    def test_ingest_batch_via_mcp_retries_taskgroup_http_session_error(self):
        """HTTP ingest retries once with a fresh session after a client-side TaskGroup error."""
        seen = {"retry_calls": 0}

        class PrimaryClient:
            connection = MCPConnectionSettings(
                transport="http",
                mcp_url=TestIngestBatchViaMCP.MCP_URL,
                command="gsrs-mcp-server",
                verify_ssl=True,
                bearer_token="token",
            )

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ensure_ingest_available(self):
                return None

            async def ingest_substance(self, substance):
                raise RuntimeError("unhandled errors in a TaskGroup (1 sub-exception)")

        class RetryClient:
            connection = PrimaryClient.connection

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ingest_substance(self, substance):
                seen["retry_calls"] += 1
                return f"Ingested **{substance['uuid']}** - 4 chunks."

        with patch("scripts.load_data.build_mcp_client", side_effect=[PrimaryClient(), RetryClient()]):
            result = ingest_batch_via_mcp(
                self.MCP_URL,
                True,
                [{"uuid": "retry-me"}],
                bearer_token="token",
            )

        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["total_chunks"], 4)
        self.assertEqual(seen["retry_calls"], 1)

    def test_load_from_file_with_mcp(self):
        """Test file loading reuses one MCP client."""
        seen = {"ingest_calls": []}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ensure_ingest_available(self):
                return None

            async def get_statistics(self):
                return {"total_chunks": 0, "total_substances": 0}

            async def ingest_substance(self, substance):
                seen["ingest_calls"].append(substance["uuid"])
                return f"Ingested **{substance['uuid']}** - 2 chunks."

        with patch("scripts.load_data.build_mcp_client", return_value=FakeClient()):
            with tempfile.NamedTemporaryFile(suffix=".gsrs", delete=False) as f:
                temp_path = f.name
                with gzip.open(f, "wt", encoding="utf-8") as gz:
                    gz.write("\t\t{\"uuid\": \"uuid-1\"}\n")
                    gz.write("\t\t{\"uuid\": \"uuid-2\"}\n")

            try:
                result = load_from_file(
                    self.MCP_URL, temp_path,
                    batch_size=1, dry_run=False, verify_ssl=False,
                )
            finally:
                os.unlink(temp_path)

        self.assertEqual(result["successful"], 2)
        self.assertEqual(result["total_chunks"], 4)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(seen["ingest_calls"], ["uuid-1", "uuid-2"])


class TestMCPToolClient(unittest.IsolatedAsyncioTestCase):
    async def test_call_tool_text_unwraps_structured_result(self):
        client = MCPToolClient(MCPConnectionSettings())

        class FakeResult:
            structuredContent = {"result": "Ingested **abc** - 2 chunks."}
            content = []

        class FakeSession:
            async def call_tool(self, tool_name, arguments):
                return FakeResult()

        client._session = FakeSession()
        text = await client.call_tool_text("gsrs_ingest", {"substance_json": "{}"})

        self.assertEqual(text, "Ingested **abc** - 2 chunks.")

    async def test_get_health_payload_parses_structured_json_result(self):
        client = MCPToolClient(MCPConnectionSettings())
        client._tool_names = {"gsrs_health"}

        class FakeResult:
            structuredContent = {
                "result": json.dumps(
                    {
                        "components": {
                            "vector_db": {"ready": True},
                            "embedding": {"ready": True},
                            "chunker": {"ready": True},
                        }
                    }
                )
            }
            content = []

        class FakeSession:
            async def call_tool(self, tool_name, arguments):
                return FakeResult()

        client._session = FakeSession()
        payload = await client.get_health_payload()

        self.assertTrue(payload["components"]["vector_db"]["ready"])

    async def test_ensure_ingest_available_raises_component_error(self):
        client = MCPToolClient(MCPConnectionSettings())
        client._tool_names = {"gsrs_ingest", "gsrs_health"}
        client.get_health_payload = lambda: asyncio.sleep(0, result={
            "components": {
                "vector_db": {"ready": True},
                "embedding": {"ready": False, "error": "Embedding provider is down"},
                "chunker": {"ready": True},
            }
        })

        with self.assertRaisesRegex(RuntimeError, "Embedding provider is down"):
            await client.ensure_ingest_available()

    async def test_stdio_client_preserves_parent_environment(self):
        seen = {}

        class DummyStack:
            async def __aenter__(self):
                return ("read-stream", "write-stream")

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class DummySession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def initialize(self):
                return None

            async def list_tools(self):
                return type("ToolList", (), {"tools": []})()

        def fake_stdio_client(server):
            seen["env"] = server.env
            return DummyStack()

        with patch.dict(os.environ, {"EMBEDDING_API_KEY": "test-key", "DATABASE_URL": "postgresql://example"}, clear=False), \
             patch("scripts.load_data.stdio_client", side_effect=fake_stdio_client), \
             patch("scripts.load_data.ClientSession", return_value=DummySession()):
            client = MCPToolClient(MCPConnectionSettings(transport="stdio"))
            await client.__aenter__()
            await client.__aexit__(None, None, None)

        self.assertEqual(seen["env"]["MCP_TRANSPORT"], "stdio")
        self.assertEqual(seen["env"]["EMBEDDING_API_KEY"], "test-key")
        self.assertEqual(seen["env"]["DATABASE_URL"], "postgresql://example")


class TestFetchSubstanceByUuid(unittest.IsolatedAsyncioTestCase):
    """Tests for fetching substances from GSRS API."""

    async def test_fetch_valid_substance(self):
        """Test fetching a valid substance from GSRS API."""
        test_uuid = "0103a288-6eb6-4ced-b13a-849cd7edf028"

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"uuid": test_uuid, "substanceClass": "chemical"}

        class FakeSession:
            async def get(self, url, timeout=30.0):
                self.last_url = url
                return FakeResponse()

        session = FakeSession()
        substance = await fetch_substance_by_uuid(test_uuid, session)

        self.assertIsNotNone(substance)
        if substance is not None:
            self.assertEqual(substance["uuid"], test_uuid)
            self.assertEqual(substance["substanceClass"], "chemical")

    async def test_fetch_invalid_substance(self):
        """Test fetching an invalid substance UUID."""
        invalid_uuid = "00000000-0000-0000-0000-000000000000"

        class FakeResponse:
            status_code = 404

            @staticmethod
            def json():
                return {}

        class FakeSession:
            async def get(self, url, timeout=30.0):
                return FakeResponse()

        session = FakeSession()
        substance = await fetch_substance_by_uuid(invalid_uuid, session)

        self.assertIsNone(substance)

    async def test_fetch_multiple_substances_parallel(self):
        """Test fetching multiple substances in parallel."""
        import asyncio

        test_uuids = [
            "0103a288-6eb6-4ced-b13a-849cd7edf028",
            "80edf0eb-b6c5-4a9a-adde-28c7254046d9",
        ]

        class FakeResponse:
            def __init__(self, payload, status_code=200):
                self._payload = payload
                self.status_code = status_code

            def json(self):
                return self._payload

        class FakeSession:
            async def get(self, url, timeout=30.0):
                substance_uuid = url.split("(")[-1].split(")")[0]
                return FakeResponse({"uuid": substance_uuid})

        session = FakeSession()
        tasks = [fetch_substance_by_uuid(uuid, session) for uuid in test_uuids]
        results = await asyncio.gather(*tasks)

        valid_results = [r for r in results if r is not None]
        self.assertEqual(len(valid_results), 2)


class TestFetchAllSubstanceUuids(unittest.IsolatedAsyncioTestCase):
    """Tests for fetching all substance UUIDs."""

    async def test_fetch_uuids_limited(self):
        """Test fetching limited number of UUIDs."""

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "results": [
                        {"uuid": "00000000-0000-0000-0000-000000000001"},
                        {"uuid": "00000000-0000-0000-0000-000000000002"},
                        {"uuid": "00000000-0000-0000-0000-000000000003"},
                    ]
                }

        class FakeSession:
            async def get(self, url, params=None, timeout=30.0):
                return FakeResponse()

        uuids = await fetch_all_substance_uuids(FakeSession(), max_results=10)

        self.assertLessEqual(len(uuids), 10)
        self.assertTrue(all(isinstance(u, str) for u in uuids))

    async def test_fetch_uuids_format(self):
        """Test that fetched UUIDs have correct format."""
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "results": [
                        {"uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028"},
                        {"uuid": "80edf0eb-b6c5-4a9a-adde-28c7254046d9"},
                    ]
                }

        class FakeSession:
            async def get(self, url, params=None, timeout=30.0):
                return FakeResponse()

        uuids = await fetch_all_substance_uuids(FakeSession(), max_results=5)

        for uuid in uuids:
            self.assertRegex(uuid, uuid_pattern)

    async def test_load_substances_from_api_uses_mcp(self):
        """Test API loading uses one MCP client for ingestion."""
        seen = {"async_verify": [], "mcp_calls": []}

        class FakeAsyncResponse:
            def __init__(self, payload, status_code=200):
                self._payload = payload
                self.status_code = status_code

            def json(self):
                return self._payload

        class FakeAsyncClient:
            def __init__(self, *, timeout, verify):
                seen["async_verify"].append(verify)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url, timeout=30.0, params=None):
                substance_uuid = url.split("(")[-1].split(")")[0]
                return FakeAsyncResponse({"uuid": substance_uuid})

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ensure_ingest_available(self):
                return None

            async def get_statistics(self):
                return {"total_chunks": 0, "total_substances": 0}

            async def ingest_substance(self, substance):
                seen["mcp_calls"].append(substance.get("uuid"))
                return f"Ingested **{substance['uuid']}** - 1 chunks."

        with patch("scripts.load_data.httpx.AsyncClient", FakeAsyncClient), \
             patch("scripts.load_data.build_mcp_client", return_value=FakeClient()):
            result = await load_substances_from_api(
                mcp_url="http://localhost:8000/mcp",
                uuids=["uuid-1", "uuid-2"],
                batch_size=2,
                dry_run=False,
                verify_ssl=False,
            )

        self.assertEqual(result["downloaded"], 2)
        self.assertEqual(result["successful"], 2)
        self.assertEqual(seen["async_verify"], [False])
        self.assertEqual(len(seen["mcp_calls"]), 2)

    async def test_load_substances_from_api_stdio_prefers_uuid_ingest_without_preflight(self):
        """stdio API loading should skip preflight chatter and use compact UUID ingest when available."""
        seen = {"uuid_calls": [], "ensure_called": 0, "stats_called": 0}

        class FakeAsyncResponse:
            def __init__(self, payload, status_code=200):
                self._payload = payload
                self.status_code = status_code

            def json(self):
                return self._payload

        class FakeAsyncClient:
            def __init__(self, *, timeout, verify):
                self.verify = verify

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url, timeout=30.0, params=None):
                substance_uuid = url.split("(")[-1].split(")")[0]
                return FakeAsyncResponse({"uuid": substance_uuid})

        class FakeClient:
            connection = MCPConnectionSettings(
                transport="stdio",
                command="gsrs-mcp-server",
            )
            _tool_names = {"gsrs_ingest", "gsrs_ingest_from_uuid", "gsrs_health", "gsrs_statistics"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ensure_ingest_available(self):
                seen["ensure_called"] += 1
                raise AssertionError("stdio preflight should be skipped")

            async def get_statistics(self):
                seen["stats_called"] += 1
                raise AssertionError("stdio statistics preflight should be skipped")

            def supports_uuid_ingest(self):
                return True

            async def ingest_uuid(self, substance_uuid):
                seen["uuid_calls"].append(substance_uuid)
                return f"Ingested **{substance_uuid}** - 2 chunks."

        with patch("scripts.load_data.httpx.AsyncClient", FakeAsyncClient), \
             patch("scripts.load_data.build_mcp_client", return_value=FakeClient()):
            result = await load_substances_from_api(
                mcp_url="http://localhost:8000/mcp",
                uuids=["uuid-1"],
                batch_size=1,
                dry_run=False,
                verify_ssl=False,
                transport="stdio",
                command="gsrs-mcp-server",
            )

        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["total_chunks"], 2)
        self.assertEqual(seen["uuid_calls"], ["uuid-1"])
        self.assertEqual(seen["ensure_called"], 0)
        self.assertEqual(seen["stats_called"], 0)

    async def test_load_substances_from_api_skips_statistics_after_taskgroup_error(self):
        """A failing gsrs_statistics call should not prevent later ingestion attempts."""
        seen = {"mcp_calls": []}

        class FakeAsyncResponse:
            def __init__(self, payload, status_code=200):
                self._payload = payload
                self.status_code = status_code

            def json(self):
                return self._payload

        class FakeAsyncClient:
            def __init__(self, *, timeout, verify):
                self.verify = verify

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url, timeout=30.0, params=None):
                substance_uuid = url.split("(")[-1].split(")")[0]
                return FakeAsyncResponse({"uuid": substance_uuid})

        class FakeClient:
            connection = MCPConnectionSettings(
                transport="http",
                mcp_url="http://localhost:8000/mcp",
                command="gsrs-mcp-server",
                verify_ssl=False,
                bearer_token="token",
            )
            _tool_names = {"gsrs_ingest", "gsrs_health", "gsrs_statistics"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def ensure_ingest_available(self):
                return None

            async def get_statistics(self):
                raise RuntimeError("unhandled errors in a TaskGroup (1 sub-exception)")

            async def ingest_substance(self, substance):
                seen["mcp_calls"].append(substance.get("uuid"))
                return f"Ingested **{substance['uuid']}** - 1 chunks."

        with patch("scripts.load_data.httpx.AsyncClient", FakeAsyncClient), \
             patch("scripts.load_data.build_mcp_client", return_value=FakeClient()):
            result = await load_substances_from_api(
                mcp_url="http://localhost:8000/mcp",
                uuids=["uuid-1"],
                batch_size=1,
                dry_run=False,
                verify_ssl=False,
                bearer_token="token",
            )

        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(seen["mcp_calls"], ["uuid-1"])


if __name__ == "__main__":
    unittest.main()
