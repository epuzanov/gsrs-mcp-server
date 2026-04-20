"""
GSRS MCP Server Configuration
"""
import json
import os
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a sensible default."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_list_env(name: str, default: list[str]) -> list[str]:
    """Parse a list environment variable from JSON array or comma-separated values."""
    value = os.getenv(name)
    if value is None:
        return list(default)

    return _get_list_env_value(value)


def _get_list_env_value(value: str) -> list[str]:
    """Parse a list from a string value (JSON array or comma-separated)."""
    raw_value = value.strip()
    if not raw_value:
        return []

    if raw_value.startswith("["):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [item for item in (str(entry).strip() for entry in parsed) if item]

    return [item for item in (part.strip() for part in raw_value.split(",")) if item]


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
    # See gsrs.services.ai.ChunkerConfig
    chunker_name_batch_size: int = int(os.getenv("CHUNKER_NAME_BATCH_SIZE", "30"))
    chunker_emit_atomic_name_chunks: bool = _get_bool_env("CHUNKER_EMIT_ATOMIC_NAME_CHUNKS", False)
    chunker_emit_sequence_segments: bool = _get_bool_env("CHUNKER_EMIT_SEQUENCE_SEGMENTS", False)
    chunker_max_sequence_segment_len: int = int(os.getenv("CHUNKER_MAX_SEQUENCE_SEGMENT_LEN", "300"))
    chunker_emit_full_sequence_in_text: bool = _get_bool_env("CHUNKER_EMIT_FULL_SEQUENCE_IN_TEXT", False)
    chunker_include_admin_validation_notes: bool = _get_bool_env("CHUNKER_INCLUDE_ADMIN_VALIDATION_NOTES", False)
    chunker_include_reference_index_chunk: bool = _get_bool_env("CHUNKER_INCLUDE_REFERENCE_INDEX_CHUNK", True)
    chunker_include_classification_chunk: bool = _get_bool_env("CHUNKER_INCLUDE_CLASSIFICATION_CHUNK", True)
    chunker_include_grouped_relationship_summaries: bool = _get_bool_env("CHUNKER_INCLUDE_GROUPED_RELATIONSHIP_SUMMARIES", True)

    # MCP endpoint
    mcp_transport: Literal["stdio", "sse", "streamable-http"] = os.getenv(
        "MCP_TRANSPORT",
        "streamable-http",
    ).lower()
    mcp_api: str = os.getenv("MCP_API", "0.0.0.0")
    mcp_port: int = int(os.getenv("MCP_PORT", "8000"))

    # Authentication
    mcp_username: str = os.getenv("MCP_USERNAME", "admin")
    mcp_password: str = os.getenv("MCP_PASSWORD", "admin123")

    # Vector Search
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    # LLM API Configuration (for query rewrite, answering, etc.)
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_url: str = os.getenv("LLM_URL", "https://api.openai.com/v1/chat/completions")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_verify_ssl: bool = _get_bool_env("LLM_VERIFY_SSL", True)
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "30"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))
    llm_retry_backoff_ms: int = int(os.getenv("LLM_RETRY_BACKOFF_MS", "250"))

    # Similar Substance Search - Reliable identifier codes (prioritized)
    identifier_code_systems: list[str] | None = None
    similarity_reliable_codes: list[str] | None = None

    @field_validator("identifier_code_systems", mode="before")
    @classmethod
    def parse_identifier_code_systems(cls, v):
        if v is None:
            return _get_list_env(
                "IDENTIFIER_CODE_SYSTEMS",
                [
                    "CAS",
                    "UNII",
                    "FDA UNII",
                    "PubChem",
                    "DrugBank",
                    "ChEMBL",
                    "RXCUI",
                    "SMS_ID",
                    "SMSID",
                    "EVMPD",
                    "xEVMPD",
                    "ASK",
                    "ASKP",
                ],
            )
        if isinstance(v, str):
            return _get_list_env_value(v)
        return v

    @field_validator("similarity_reliable_codes", mode="before")
    @classmethod
    def parse_similarity_reliable_codes(cls, v):
        if v is None:
            return _get_list_env(
                "SIMILARITY_RELIABLE_CODES",
                ["FDA UNII", "UNII", "SMS_ID", "SMSID", "xEVMPD", "EVMPD", "ASK", "ASKP"],
            )
        if isinstance(v, str):
            return _get_list_env_value(v)
        return v

    @field_validator("mcp_transport", mode="before")
    @classmethod
    def parse_mcp_transport(cls, v):
        if v is None:
            return "streamable-http"
        return str(v).strip().lower()

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
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "30"))
    answer_confidence_threshold: float = float(os.getenv("ANSWER_CONFIDENCE_THRESHOLD", "0.35"))
    max_answer_evidence: int = int(os.getenv("MAX_ANSWER_EVIDENCE", "5"))


settings = Settings()
