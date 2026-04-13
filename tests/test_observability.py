"""Tests for structured logging helpers."""
import json
import logging
import unittest

from app.observability import InMemoryMetrics, JsonLogFormatter, ToolTelemetry


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestObservability(unittest.TestCase):
    def test_json_formatter_includes_structured_fields(self):
        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="tool_finished",
            args=(),
            exc_info=None,
        )
        record.request_id = "req-1"
        record.tool_name = "gsrs_ask"
        record.latency_ms = 12.5

        payload = json.loads(formatter.format(record))
        self.assertEqual(payload["request_id"], "req-1")
        self.assertEqual(payload["tool_name"], "gsrs_ask")
        self.assertEqual(payload["latency_ms"], 12.5)

    def test_tool_telemetry_records_metrics(self):
        logger = logging.getLogger("test.telemetry")
        metrics = InMemoryMetrics()
        telemetry = ToolTelemetry.start(logger, metrics, "gsrs_health", "chroma")
        telemetry.stage("runtime_check", outcome="success", component="vector_db")
        telemetry.finish("success", result_count=0, citation_count=0)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"]["tool_calls.gsrs_health"], 1)
        self.assertEqual(snapshot["counters"]["tool_stage.gsrs_health.runtime_check.success"], 1)
        self.assertEqual(snapshot["counters"]["tool_outcomes.gsrs_health.success"], 1)

    def test_tool_telemetry_preserves_bound_query_type(self):
        logger = logging.getLogger("test.telemetry.query_type")
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = _CaptureHandler()
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)

        metrics = InMemoryMetrics()
        telemetry = ToolTelemetry.start(
            logger,
            metrics,
            "gsrs_ask",
            "chroma",
            query_type="question",
        )
        telemetry.bind(query_type="code")
        telemetry.stage("retrieval", outcome="success", retrieval_mode="identifier-first:code")
        telemetry.finish("success", result_count=1, citation_count=1)

        self.assertEqual(handler.records[0].query_type, "question")
        self.assertEqual(handler.records[1].query_type, "code")
        self.assertEqual(handler.records[2].query_type, "code")

    def test_json_formatter_redacts_sensitive_fields(self):
        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="tool_finished",
            args=(),
            exc_info=None,
        )
        record.mcp_password = "super-secret"
        record.headers = {"Authorization": "Bearer top-secret"}
        record.debug = {"token": "abc123", "safe": "value"}

        payload = json.loads(formatter.format(record))
        self.assertEqual(payload["mcp_password"], "[REDACTED]")
        self.assertEqual(payload["headers"]["Authorization"], "[REDACTED]")
        self.assertEqual(payload["debug"]["token"], "[REDACTED]")
        self.assertEqual(payload["debug"]["safe"], "value")
