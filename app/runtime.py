"""Runtime state and startup validation for the GSRS MCP server."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import Settings, settings
from app.observability import InMemoryMetrics
from app.services import EmbeddingService, VectorDatabaseService
from app.services.gsrs_api import GsrsApiService
from app.services.llm import LLMService
from app.services.query_pipeline import QueryPipelineService


@dataclass
class ComponentStatus:
    """Health/readiness state for a runtime dependency."""

    name: str
    required: bool
    ready: bool
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ServerRuntime:
    """Owns long-lived service instances and runtime readiness state."""

    def __init__(self, app_settings: Settings = settings) -> None:
        self.settings = app_settings
        self.metrics = InMemoryMetrics()
        self.vector_db = VectorDatabaseService(database_url=app_settings.database_url)
        self.embedding_service = EmbeddingService(
            api_key=app_settings.embedding_api_key,
            model=app_settings.embedding_model,
            url=app_settings.embedding_url,
            dimension=app_settings.embedding_dimension,
            verify_ssl=app_settings.embedding_verify_ssl,
            timeout=app_settings.embedding_timeout,
            max_retries=app_settings.embedding_max_retries,
            retry_backoff_ms=app_settings.embedding_retry_backoff_ms,
        )
        self.llm_service = (
            LLMService(
                api_key=app_settings.llm_api_key,
                url=app_settings.llm_url,
                model=app_settings.llm_model,
                verify_ssl=app_settings.llm_verify_ssl,
                timeout=app_settings.llm_timeout,
                max_retries=app_settings.llm_max_retries,
                retry_backoff_ms=app_settings.llm_retry_backoff_ms,
            )
            if app_settings.llm_url
            else None
        )
        self.gsrs_api = GsrsApiService(
            base_url=app_settings.gsrs_api_url,
            timeout=app_settings.gsrs_api_timeout,
            verify_ssl=app_settings.gsrs_api_verify_ssl,
            public_only=app_settings.gsrs_api_public_only,
            max_retries=app_settings.gsrs_api_max_retries,
            retry_backoff_ms=app_settings.gsrs_api_retry_backoff_ms,
        )
        self.chunker = None
        self.query_pipeline: QueryPipelineService | None = None
        self.components: dict[str, ComponentStatus] = {}
        self.started_at: datetime | None = None
        self.initialized = False

    @property
    def backend_name(self) -> str:
        return self.vector_db.backend_name

    @property
    def ready(self) -> bool:
        required_components = [status for status in self.components.values() if status.required]
        return bool(required_components) and all(status.ready for status in required_components)

    @property
    def degraded(self) -> bool:
        return any(not status.ready for status in self.components.values() if not status.required)

    def initialize(self) -> None:
        """Initialize runtime services and capture readiness state."""
        self.components = {}
        self.started_at = datetime.now(timezone.utc)
        self.initialized = False

        self._initialize_vector_db()
        self._validate_embedding_provider()
        self._initialize_chunker()
        self._initialize_query_pipeline()
        self._validate_llm_provider()
        self._validate_gsrs_api()
        self.initialized = True

    def shutdown(self) -> None:
        """Close long-lived clients and database connections."""
        self.vector_db.disconnect()
        self.embedding_service.close()
        if self.llm_service is not None:
            self.llm_service.close()
        self.initialized = False

    def get_component(self, name: str) -> ComponentStatus | None:
        return self.components.get(name)

    def required_component_errors(self) -> dict[str, str]:
        """Return required-component failures keyed by component name."""
        return {
            name: status.error or "Component is not ready."
            for name, status in self.components.items()
            if status.required and not status.ready
        }

    def optional_component_errors(self) -> dict[str, str]:
        """Return degraded optional-component failures keyed by component name."""
        return {
            name: status.error or "Component is not ready."
            for name, status in self.components.items()
            if not status.required and not status.ready
        }

    @property
    def readiness_summary(self) -> str:
        """Summarize required-component readiness for operators and tests."""
        if self.ready:
            return "Core retrieval dependencies are ready."
        if not self.initialized and not self.components:
            return "Runtime startup is still in progress."

        failures = self.required_component_errors()
        if not failures:
            return "Required runtime components are not ready."
        return "; ".join(f"{name}: {error}" for name, error in failures.items())

    @property
    def degraded_summary(self) -> str | None:
        """Summarize optional-component degradation when present."""
        failures = self.optional_component_errors()
        if not failures:
            return None
        return "; ".join(f"{name}: {error}" for name, error in failures.items())

    def get_status_payload(self) -> dict[str, Any]:
        """Return a structured health/readiness payload."""
        payload = {
            "status": self.runtime_status,
            "ready": self.ready,
            "degraded": self.degraded,
            "readiness_summary": self.readiness_summary,
            "degraded_summary": self.degraded_summary,
            "backend": self.backend_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "required_component_errors": self.required_component_errors(),
            "optional_component_errors": self.optional_component_errors(),
            "components": {
                name: {
                    "required": status.required,
                    "ready": status.ready,
                    "error": status.error,
                    "details": status.details,
                }
                for name, status in self.components.items()
            },
            "metrics": self.metrics.snapshot(),
        }
        vector_status = self.components.get("vector_db")
        if vector_status:
            payload["statistics"] = vector_status.details.get("statistics", {})
        return payload

    @property
    def runtime_status(self) -> str:
        """Return a deterministic runtime state label for health endpoints."""
        if not self.initialized and not self.components:
            return "starting"
        if self.ready and self.degraded:
            return "ready_degraded"
        if self.ready:
            return "ready"
        return "not_ready"

    def _component_ready(self, name: str, *, default: bool = False) -> bool:
        component = self.components.get(name)
        if component is None:
            return default
        return component.ready

    def _component_error(self, name: str, default: str) -> str:
        component = self.components.get(name)
        if component and component.error:
            return component.error
        return default

    def vector_backend_available(self) -> bool:
        return self._component_ready("vector_db")

    def vector_backend_unavailable_reason(self) -> str:
        return self._component_error("vector_db", "Vector database is not ready.")

    def metadata_lookup_available(self) -> bool:
        return self.vector_backend_available()

    def metadata_lookup_unavailable_reason(self) -> str:
        return self.vector_backend_unavailable_reason()

    def retrieval_available(self) -> bool:
        return (
            self.vector_backend_available()
            and self._component_ready("embedding")
            and self.query_pipeline is not None
        )

    def answer_generation_available(self) -> bool:
        llm_status = self.components.get("answer_generation")
        return bool(llm_status and llm_status.ready and self.query_pipeline is not None)

    def retrieval_unavailable_reason(self) -> str:
        if not self.vector_backend_available():
            return self.vector_backend_unavailable_reason()
        if not self._component_ready("embedding"):
            return self._component_error("embedding", "Embedding provider is not ready.")
        return "Retrieval pipeline is not ready."

    def ingestion_available(self) -> bool:
        return (
            self.vector_backend_available()
            and self._component_ready("embedding")
            and self._component_ready("chunker")
            and self.chunker is not None
        )

    def ingestion_unavailable_reason(self) -> str:
        if not self.vector_backend_available():
            return self.vector_backend_unavailable_reason()
        if not self._component_ready("embedding"):
            return self._component_error("embedding", "Embedding provider is not ready.")
        if not self._component_ready("chunker") or self.chunker is None:
            return self._component_error("chunker", "Chunker is not ready.")
        return "Ingestion pipeline is not ready."

    def gsrs_api_available(self) -> bool:
        return self._component_ready("gsrs_api")

    def gsrs_api_unavailable_reason(self) -> str:
        return self._component_error(
            "gsrs_api",
            "GSRS upstream service is not ready.",
        )

    def _set_component(
        self,
        name: str,
        *,
        required: bool,
        ready: bool,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.components[name] = ComponentStatus(
            name=name,
            required=required,
            ready=ready,
            error=error,
            details=details or {},
        )

    def _initialize_vector_db(self) -> None:
        try:
            self.vector_db.connect()
            self.vector_db.initialize(dimension=self.settings.embedding_dimension)
            stats = self.vector_db.get_statistics()
            self._set_component(
                "vector_db",
                required=True,
                ready=True,
                details={
                    "backend": self.backend_name,
                    "statistics": stats,
                },
            )
        except Exception as exc:
            self._set_component(
                "vector_db",
                required=True,
                ready=False,
                error=f"Vector backend initialization failed: {exc}",
                details={"backend": self.backend_name},
            )

    def _validate_embedding_provider(self) -> None:
        config_error: str | None = None
        if not self.settings.embedding_api_key:
            config_error = "EMBEDDING_API_KEY is not configured."
        elif not self.settings.embedding_url:
            config_error = "EMBEDDING_URL is not configured."
        elif self.settings.embedding_dimension <= 0:
            config_error = "EMBEDDING_DIMENSION must be greater than zero."

        if config_error:
            self._set_component("embedding", required=True, ready=False, error=config_error)
            return

        details = self.embedding_service.get_model_info()
        if self.settings.startup_validate_external:
            try:
                self.embedding_service.embed("gsrs readiness probe")
                details["validated_via"] = "embed_probe"
            except Exception as exc:
                self._set_component(
                    "embedding",
                    required=True,
                    ready=False,
                    error=f"Embedding provider validation failed: {exc}",
                    details=details,
                )
                return
        else:
            details["validated_via"] = "configuration"

        self._set_component("embedding", required=True, ready=True, details=details)

    def _initialize_chunker(self) -> None:
        try:
            from gsrs.model import Substance
            from gsrs.services.ai import ChunkerConfig, SubstanceChunker
            from app.models import VectorDocument

            self.chunker = SubstanceChunker(
                class_=VectorDocument,
                config=ChunkerConfig(
                    name_batch_size=self.settings.chunker_name_batch_size,
                    emit_atomic_name_chunks=self.settings.chunker_emit_atomic_name_chunks,
                    emit_sequence_segments=self.settings.chunker_emit_sequence_segments,
                    max_sequence_segment_len=self.settings.chunker_max_sequence_segment_len,
                    emit_full_sequence_in_text=self.settings.chunker_emit_full_sequence_in_text,
                    include_admin_validation_notes=self.settings.chunker_include_admin_validation_notes,
                    include_reference_index_chunk=self.settings.chunker_include_reference_index_chunk,
                    include_classification_chunk=self.settings.chunker_include_classification_chunk,
                    include_grouped_relationship_summaries=self.settings.chunker_include_grouped_relationship_summaries,
                ),
            )
            _ = Substance
            self._set_component("chunker", required=False, ready=True)
        except Exception as exc:
            self.chunker = None
            self._set_component(
                "chunker",
                required=False,
                ready=False,
                error=f"Chunker initialization failed: {exc}",
            )

    def _initialize_query_pipeline(self) -> None:
        if not self.retrieval_available_for_initialization:
            self.query_pipeline = None
            self._set_component(
                "query_pipeline",
                required=True,
                ready=False,
                error="Query pipeline is unavailable because required retrieval dependencies are not ready.",
            )
            return

        try:
            self.query_pipeline = QueryPipelineService(
                vector_db=self.vector_db,
                embedding_service=self.embedding_service,
                llm_service=self.llm_service,
                max_evidence=self.settings.max_answer_evidence,
                min_confidence=self.settings.answer_confidence_threshold,
                use_llm=self.llm_service is not None,
            )
            self._set_component("query_pipeline", required=True, ready=True)
        except Exception as exc:
            self.query_pipeline = None
            self._set_component(
                "query_pipeline",
                required=True,
                ready=False,
                error=f"Query pipeline initialization failed: {exc}",
            )

    @property
    def retrieval_available_for_initialization(self) -> bool:
        return (
            self.vector_backend_available()
            and self._component_ready("embedding")
        )

    def _validate_llm_provider(self) -> None:
        if self.llm_service is None:
            self._set_component(
                "answer_generation",
                required=False,
                ready=False,
                error="LLM provider is not configured; gsrs_ask will return retrieval-grounded fallback answers.",
            )
            return

        details = self.llm_service.get_model_info()
        if self.settings.startup_validate_external:
            try:
                self.llm_service.complete_text(
                    system_prompt="Reply with the single word ok.",
                    user_prompt="ok",
                    temperature=0.0,
                )
                details["validated_via"] = "completion_probe"
            except Exception as exc:
                self._set_component(
                    "answer_generation",
                    required=False,
                    ready=False,
                    error=f"Answer generation provider validation failed: {exc}",
                    details=details,
                )
                return
        else:
            details["validated_via"] = "configuration"

        self._set_component("answer_generation", required=False, ready=True, details=details)

        if self.query_pipeline is not None:
            self.query_pipeline.set_answer_generation_enabled(True, self.llm_service)

    def _validate_gsrs_api(self) -> None:
        details = self.gsrs_api.get_status()
        if self.settings.startup_validate_external:
            try:
                self.gsrs_api.ping()
                details["validated_via"] = "http_probe"
                self._set_component("gsrs_api", required=False, ready=True, details=details)
                return
            except Exception as exc:
                self._set_component(
                    "gsrs_api",
                    required=False,
                    ready=False,
                    error=f"GSRS upstream validation failed: {exc}",
                    details=details,
                )
                return

        details["validated_via"] = "configuration"
        self._set_component("gsrs_api", required=False, ready=True, details=details)
