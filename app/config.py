"""
GSRS MCP Server Configuration
"""
import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a sensible default.

    An unset variable (key not in os.environ) AND an empty value both fall
    back to ``default``. This matches the behaviour of "${VAR:-default}"
    in shell, so an empty value rendered into a Kubernetes manifest by
    ``envsubst`` (e.g. from a missing .env key) behaves the same as the
    key being absent.
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    """Like :func:`_get_bool_env` but for integer values.

    An unset variable or an empty value falls back to ``default``. Other
    non-integer strings raise ``ValueError`` at startup — that's an
    operator error and should be loud.
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float_env(name: str, default: float) -> float:
    """Like :func:`_get_int_env` but for floating-point values."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore unused environment variables
    )

    # Database URL
    # PostgreSQL: postgresql://user:pass@host:port/dbname
    # ChromaDB: chroma://./chroma_data/chunks
    # The `or` form makes an empty-string env value (e.g. produced by
    # envsubst from a missing .env key) fall back to the in-code default
    # rather than be passed through as an empty string.
    database_url: str = (
        os.getenv("DATABASE_URL")
        or "postgresql://postgres:postgres@localhost:5432/gsrs_mcp"
    )

    # Embedding API Configuration
    # EMBEDDING_API_KEY intentionally uses the strict form: an empty key
    # is a legitimate value (e.g. local Ollama without authentication)
    # and must not be silently replaced with a default.
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_url: str = os.getenv("EMBEDDING_URL") or "https://api.openai.com/v1/embeddings"
    embedding_model: str = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
    embedding_dimension: int = _get_int_env("EMBEDDING_DIMENSION", 1536)
    embedding_verify_ssl: bool = _get_bool_env("EMBEDDING_VERIFY_SSL", True)
    embedding_timeout: float = _get_float_env("EMBEDDING_TIMEOUT", 30.0)
    embedding_max_retries: int = _get_int_env("EMBEDDING_MAX_RETRIES", 2)
    embedding_retry_backoff_ms: int = _get_int_env("EMBEDDING_RETRY_BACKOFF_MS", 250)
    # OpenAI-only request-payload toggles. Both default to False so the
    # server works against any OpenAI-compatible provider (Ollama, vLLM,
    # LiteLLM routing to non-OpenAI backends) out of the box. Set these
    # to True only when talking to OpenAI's own embeddings API.
    embedding_send_dimensions: bool = _get_bool_env("EMBEDDING_SEND_DIMENSIONS", False)
    embedding_send_encoding_format: bool = _get_bool_env("EMBEDDING_SEND_ENCODING_FORMAT", False)

    # SubstanceChunker Configuration (ChunkerConfig fine-tuning)
    # See app.services.chunker.ChunkerConfig
    chunker_emit_atomic_name_chunks: bool = _get_bool_env("CHUNKER_EMIT_ATOMIC_NAME_CHUNKS", True)
    chunker_emit_sequence_segments: bool = _get_bool_env("CHUNKER_EMIT_SEQUENCE_SEGMENTS", False)
    chunker_emit_full_sequence_in_text: bool = _get_bool_env("CHUNKER_EMIT_FULL_SEQUENCE_IN_TEXT", True)
    chunker_include_admin_validation_notes: bool = _get_bool_env("CHUNKER_INCLUDE_ADMIN_VALIDATION_NOTES", True)
    chunker_include_classification_chunk: bool = _get_bool_env("CHUNKER_INCLUDE_CLASSIFICATION_CHUNK", True)

    # MCP endpoint
    mcp_transport: Literal["stdio", "sse", "streamable-http"] = (
        os.getenv("MCP_TRANSPORT") or "streamable-http"
    ).strip().lower()
    mcp_api: str = os.getenv("MCP_API") or "0.0.0.0"
    mcp_port: int = _get_int_env("MCP_PORT", 8000)

    # Authentication
    mcp_password: str = os.getenv("MCP_PASSWORD") or "admin123"

    # Vector Search
    default_top_k: int = _get_int_env("DEFAULT_TOP_K", 5)

    # GSRS Official API Configuration
    gsrs_api_url: str = os.getenv("GSRS_API_URL") or "https://gsrs.ncats.nih.gov/api/v1"
    gsrs_api_timeout: int = _get_int_env("GSRS_API_TIMEOUT", 30)
    gsrs_api_verify_ssl: bool = _get_bool_env("GSRS_API_VERIFY_SSL", True)
    gsrs_api_public_only: bool = _get_bool_env("GSRS_API_PUBLIC_ONLY", False)
    gsrs_api_max_retries: int = _get_int_env("GSRS_API_MAX_RETRIES", 1)
    gsrs_api_retry_backoff_ms: int = _get_int_env("GSRS_API_RETRY_BACKOFF_MS", 250)

    # Runtime/observability
    debug_mode: bool = _get_bool_env("DEBUG_MODE", False)
    startup_validate_external: bool = _get_bool_env("STARTUP_VALIDATE_EXTERNAL", False)


settings = Settings()
