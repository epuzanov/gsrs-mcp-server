"""Tests for runtime startup, readiness, and dependency failure behavior."""
import unittest
from unittest.mock import MagicMock, patch

from app.config import settings
from app.runtime import ServerRuntime


def _test_settings(**overrides):
    return settings.model_copy(
        update={
            "database_url": "chroma://./.test-runtime/chunks",
            "embedding_api_key": "test-key",
            "embedding_url": "https://example.test/embeddings",
            "embedding_model": "test-embedding-model",
            "embedding_dimension": 8,
            "startup_validate_external": False,
            "llm_api_key": "",
            **overrides,
        }
    )


class TestServerRuntime(unittest.TestCase):
    def _mark_chunker_ready(self, runtime: ServerRuntime):
        runtime.chunker = object()
        runtime._set_component("chunker", required=True, ready=True)

    def test_runtime_starts_in_starting_state(self):
        runtime = ServerRuntime(_test_settings())

        self.assertEqual(runtime.runtime_status, "starting")
        self.assertFalse(runtime.initialized)
        self.assertFalse(runtime.ready)

    def test_initialize_smoke_path_builds_ready_runtime(self):
        runtime = ServerRuntime(_test_settings())

        with patch.object(runtime.vector_db, "connect", MagicMock()), \
             patch.object(runtime.vector_db, "initialize", MagicMock()), \
             patch.object(runtime.vector_db, "get_statistics", return_value={"total_chunks": 0, "total_substances": 0}), \
             patch.object(runtime.embedding_service, "get_model_info", return_value={"model": "test-embedding-model"}), \
             patch.object(runtime.gsrs_api, "get_status", return_value={"base_url": "https://gsrs.example.test/api/v1"}), \
             patch.object(runtime, "_initialize_chunker", side_effect=lambda: self._mark_chunker_ready(runtime)):
            runtime.initialize()

        self.assertTrue(runtime.initialized)
        self.assertEqual(runtime.runtime_status, "ready_degraded")
        self.assertTrue(runtime.ready)
        self.assertTrue(runtime.degraded)
        self.assertIsNotNone(runtime.query_pipeline)
        self.assertEqual(runtime.get_status_payload()["statistics"]["total_chunks"], 0)

    def test_initialize_reports_vector_backend_failure(self):
        runtime = ServerRuntime(_test_settings())

        with patch.object(runtime.vector_db, "connect", side_effect=RuntimeError("database unavailable")), \
             patch.object(runtime.embedding_service, "get_model_info", return_value={"model": "test-embedding-model"}), \
             patch.object(runtime.gsrs_api, "get_status", return_value={"base_url": "https://gsrs.example.test/api/v1"}), \
             patch.object(runtime, "_initialize_chunker", side_effect=lambda: self._mark_chunker_ready(runtime)):
            runtime.initialize()

        self.assertTrue(runtime.initialized)
        self.assertEqual(runtime.runtime_status, "not_ready")
        self.assertFalse(runtime.ready)
        self.assertFalse(runtime.vector_backend_available())
        self.assertIn("database unavailable", runtime.vector_backend_unavailable_reason())
        self.assertIsNone(runtime.query_pipeline)
        payload = runtime.get_status_payload()
        self.assertIn("vector_db", payload["required_component_errors"])
        self.assertIn("database unavailable", payload["readiness_summary"])

    def test_chunker_failure_degrades_but_keeps_runtime_ready_for_retrieval(self):
        runtime = ServerRuntime(_test_settings())

        def mark_chunker_failed():
            runtime.chunker = None
            runtime._set_component(
                "chunker",
                required=False,
                ready=False,
                error="Chunker initialization failed: gsrs model is unavailable.",
            )

        with patch.object(runtime.vector_db, "connect", MagicMock()), \
             patch.object(runtime.vector_db, "initialize", MagicMock()), \
             patch.object(runtime.vector_db, "get_statistics", return_value={"total_chunks": 0, "total_substances": 0}), \
             patch.object(runtime.embedding_service, "get_model_info", return_value={"model": "test-embedding-model"}), \
             patch.object(runtime.gsrs_api, "get_status", return_value={"base_url": "https://gsrs.example.test/api/v1"}), \
             patch.object(runtime, "_initialize_chunker", side_effect=mark_chunker_failed):
            runtime.initialize()

        self.assertTrue(runtime.initialized)
        self.assertTrue(runtime.ready)
        self.assertTrue(runtime.degraded)
        self.assertEqual(runtime.runtime_status, "ready_degraded")
        self.assertIsNotNone(runtime.query_pipeline)
        self.assertFalse(runtime.ingestion_available())
        payload = runtime.get_status_payload()
        self.assertIn("chunker", payload["optional_component_errors"])
        self.assertIn("Chunker initialization failed", payload["degraded_summary"])

    def test_shutdown_resets_initialized_flag(self):
        runtime = ServerRuntime(_test_settings())

        with patch.object(runtime.vector_db, "disconnect", MagicMock()), \
             patch.object(runtime.embedding_service, "close", MagicMock()):
            runtime.initialized = True
            runtime.shutdown()

        self.assertFalse(runtime.initialized)
