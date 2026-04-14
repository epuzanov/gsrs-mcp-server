"""
GSRS MCP Server - Vector Database Abstract Interface
"""
from functools import total_ordering
from typing import Any, List, Dict, Optional
from uuid import UUID, uuid4
from datetime import datetime, date

from sqlalchemy import String, Text, DateTime, JSON, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func as sql_func
from pgvector.sqlalchemy import Vector, HALFVEC as HalfVec

from app.config import settings


class Base(DeclarativeBase):
    pass


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert datetime/date objects to ISO strings for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


class VectorDocument(Base):
    """
    Represents a chunk of a GSRS Substance document.
    Each chunk corresponds to a specific section in the substance JSON.

    Compatible with gsrs.services.ai SubstanceChunker(class_=VectorDocument).chunks(substance)
    output format.
    """
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))

    # Unique chunk identifier from gsrs.model (e.g., "root_uuid:12345678-...")
    chunk_id: Mapped[str] = mapped_column(String(512), nullable=False, unique=True, index=True)

    # Document ID (substance UUID)
    document_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)

    # Section name (e.g., "root", "names", "codes", "structure", "references")
    section: Mapped[str] = mapped_column(String(256), nullable=False, index=True)

    # Source URL/name (system-generated, from gsrs.model)
    source_url: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)

    # The actual text content of the chunk
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Vector embedding (uses HalfVec for dimensions > 2000)
    embedding: Mapped[List[float]] = mapped_column(
        HalfVec(settings.embedding_dimension) if settings.embedding_dimension > 2000 else Vector(settings.embedding_dimension),
        nullable=False,
    )

    # Metadata containing all element attributes from gsrs.model:
    # - canonical_name: preferred substance name
    # - chunk_type: type of chunk (overview, name, code, etc.)
    # - hierarchy: parent context information
    # - additional gsrs.model metadata fields
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Denormalized search text for lexical/keyword retrieval
    # Contains chunk text + important metadata terms for full-text search
    search_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # PostgreSQL tsvector for full-text search (generated column)
    search_tsv: Mapped[Optional[Any]] = mapped_column(TSVECTOR, nullable=True)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=sql_func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=sql_func.now(), onupdate=sql_func.now())

    __table_args__ = (
        Index('idx_document_id', 'document_id'),
        Index('idx_section', 'section'),
        Index('idx_source_url', 'source_url'),
        Index('idx_search_tsv', 'search_tsv', postgresql_using='gin'),
        Index(
            'idx_embedding_hnsw',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': "halfvec_cosine_ops" if settings.embedding_dimension > 2000 else "vector_cosine_ops"},
        ),
    )

    def __init__(self, *args: Any, **kwargs: Any):
        """Accept `metadata` constructor input and store it in `metadata_json`."""
        if "metadata" in kwargs:
            kwargs["metadata_json"] = kwargs.pop("metadata")
        super().__init__(*args, **kwargs)

    def values(self):
        return {
            "document_id": self.document_id,
            "section": self.section,
            "source_url": self.source_url,
            "text": self.text,
            "embedding": self.embedding,
            "metadata_json": _sanitize_for_json(self.metadata_json),
            "search_text": self.search_text,
            "search_tsv": sql_func.to_tsvector(
                "english", " ".join([
                    self.search_text if self.search_text else "",
                    self.metadata_json["canonical_name"] if self.metadata_json.get("canonical_name") else "",
                    self.metadata_json["chunk_type"] if self.metadata_json.get("chunk_type") else "",
                    self.section if self.section else "",
                ])
            )
        }

    def set_embedding(self, embedding: List[float]) -> None:
        """Set the embedding vector."""
        self.embedding = embedding

    def __repr__(self):
        return f"<VectorDocument(chunk_id={self.chunk_id}, section={self.section})>"


@total_ordering
class DBQueryResult:
    """Represents a query result with similarity score.

    Comparison operators order results by score descending, so that
    ``sorted(results)`` and ``list.sort()`` yield highest scores first
    without needing a custom key function.
    """

    __slots__ = ("document", "score")

    document: VectorDocument  # VectorDocument from app.models
    score: float

    def __init__(self, document: VectorDocument, score: float):
        self.document = document
        self.score = score

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, DBQueryResult):
            return NotImplemented
        return self.score > other.score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DBQueryResult):
            return NotImplemented
        return self.score == other.score

    def __repr__(self):
        return f"<DBQueryResult(score={self.score:.4f}, section={self.document.section!r})>"

