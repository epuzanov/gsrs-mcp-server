"""Structured logging and light metrics for MCP tool execution."""
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


_STANDARD_LOG_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}

_SENSITIVE_FIELD_MARKERS = (
    "api_key",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)


def _sanitize_for_logging(value: Any, field_name: str | None = None) -> Any:
    """Recursively redact sensitive-looking values before serializing logs."""
    normalized_name = (field_name or "").lower()
    if any(marker in normalized_name for marker in _SENSITIVE_FIELD_MARKERS):
        return "[REDACTED]"

    if isinstance(value, dict):
        return {
            key: _sanitize_for_logging(item, key)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_for_logging(item, field_name) for item in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_for_logging(item, field_name) for item in value)

    return value


class JsonLogFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_FIELDS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(_sanitize_for_logging(payload), default=str)


def configure_logging(debug: bool = False) -> None:
    """Configure root logging once for structured JSON output."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_gsrs_logging_configured", False):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger._gsrs_logging_configured = True  # type: ignore[attr-defined]


class InMemoryMetrics:
    """Tiny metrics abstraction suitable for tests and local diagnostics."""

    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()
        self._latencies_ms: dict[str, list[float]] = {}
        self._lock = Lock()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe_latency(self, name: str, latency_ms: float) -> None:
        with self._lock:
            self._latencies_ms.setdefault(name, []).append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latency_summary = {
                name: {
                    "count": len(values),
                    "avg_ms": round(sum(values) / len(values), 2) if values else 0.0,
                    "max_ms": round(max(values), 2) if values else 0.0,
                }
                for name, values in self._latencies_ms.items()
            }
            return {
                "counters": dict(self._counters),
                "latencies": latency_summary,
            }


@dataclass
class ToolTelemetry:
    """Lifecycle helper for structured tool logs."""

    logger: logging.Logger
    metrics: InMemoryMetrics
    tool_name: str
    backend: str
    request_id: str
    started_at: float

    @classmethod
    def start(
        cls,
        logger: logging.Logger,
        metrics: InMemoryMetrics,
        tool_name: str,
        backend: str,
    ) -> "ToolTelemetry":
        request_id = str(uuid4())
        logger.info(
            "tool_started",
            extra={
                "request_id": request_id,
                "tool_name": tool_name,
                "backend": backend,
                "outcome": "started",
            },
        )
        metrics.increment(f"tool_calls.{tool_name}")
        return cls(
            logger=logger,
            metrics=metrics,
            tool_name=tool_name,
            backend=backend,
            request_id=request_id,
            started_at=time.perf_counter(),
        )

    def finish(self, outcome: str, **fields: Any) -> None:
        latency_ms = round((time.perf_counter() - self.started_at) * 1000, 2)
        payload = {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "backend": self.backend,
            "latency_ms": latency_ms,
            "outcome": outcome,
            **fields,
        }
        self.logger.info("tool_finished", extra=payload)
        self.metrics.observe_latency(f"tool_latency.{self.tool_name}", latency_ms)
        self.metrics.increment(f"tool_outcomes.{self.tool_name}.{outcome}")

    def stage(self, stage_name: str, outcome: str = "success", **fields: Any) -> None:
        """Emit a structured intermediate stage event for long-running tools."""
        payload = {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "backend": self.backend,
            "stage": stage_name,
            "outcome": outcome,
            **fields,
        }
        self.logger.info("tool_stage", extra=payload)
        self.metrics.increment(f"tool_stage.{self.tool_name}.{stage_name}.{outcome}")

    def fail(self, error: Exception, **fields: Any) -> None:
        payload = {
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            **fields,
        }
        self.finish("error", **payload)
