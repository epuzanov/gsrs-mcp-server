"""
GSRS MCP Server Configuration
"""
import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a sensible default."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore unused environment variables
    )

    # Database URL
    # PostgreSQL: postgresql://user:pass@host:port/dbname
    # ChromaDB: chroma://./chroma_data/chunks
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gsrs_mcp")

    # Embedding API Configuration
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_url: str = os.getenv("EMBEDDING_URL", "https://api.openai.com/v1/embeddings")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    embedding_verify_ssl: bool = _get_bool_env("EMBEDDING_VERIFY_SSL", True)
    embedding_timeout: float = float(os.getenv("EMBEDDING_TIMEOUT", "30"))
    embedding_max_retries: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "2"))
    embedding_retry_backoff_ms: int = int(os.getenv("EMBEDDING_RETRY_BACKOFF_MS", "250"))

    # SubstanceChunker Configuration (ChunkerConfig fine-tuning)
    # See app.services.chunker.ChunkerConfig
    chunker_emit_atomic_name_chunks: bool = _get_bool_env("CHUNKER_EMIT_ATOMIC_NAME_CHUNKS", True)
    chunker_emit_sequence_segments: bool = _get_bool_env("CHUNKER_EMIT_SEQUENCE_SEGMENTS", False)
    chunker_emit_full_sequence_in_text: bool = _get_bool_env("CHUNKER_EMIT_FULL_SEQUENCE_IN_TEXT", True)
    chunker_include_admin_validation_notes: bool = _get_bool_env("CHUNKER_INCLUDE_ADMIN_VALIDATION_NOTES", True)
    chunker_include_classification_chunk: bool = _get_bool_env("CHUNKER_INCLUDE_CLASSIFICATION_CHUNK", True)

    # MCP endpoint
    mcp_transport: Literal["stdio", "sse", "streamable-http"] = os.getenv("MCP_TRANSPORT", "streamable-http").strip().lower()
    mcp_api: str = os.getenv("MCP_API", "0.0.0.0")
    mcp_port: int = int(os.getenv("MCP_PORT", "8000"))

    # Authentication
    mcp_password: str = os.getenv("MCP_PASSWORD", "admin123")

    # Vector Search
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    # GSRS Official API Configuration
    gsrs_api_url: str = os.getenv("GSRS_API_URL", "https://gsrs.ncats.nih.gov/api/v1")
    gsrs_api_timeout: int = int(os.getenv("GSRS_API_TIMEOUT", "30"))
    gsrs_api_verify_ssl: bool = _get_bool_env("GSRS_API_VERIFY_SSL", True)
    gsrs_api_public_only: bool = _get_bool_env("GSRS_API_PUBLIC_ONLY", False)
    gsrs_api_max_retries: int = int(os.getenv("GSRS_API_MAX_RETRIES", "1"))
    gsrs_api_retry_backoff_ms: int = int(os.getenv("GSRS_API_RETRY_BACKOFF_MS", "250"))

    # Runtime/observability
    debug_mode: bool = _get_bool_env("DEBUG_MODE", False)
    startup_validate_external: bool = _get_bool_env("STARTUP_VALIDATE_EXTERNAL", False)


settings = Settings()
