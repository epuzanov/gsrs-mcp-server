"""
GSRS MCP Server - Vector Database Service
Unified service layer for vector database operations.
"""
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from uuid import UUID

from app.db.base import VectorDatabase
from app.db.factory import create_vector_database
from app.models import DBQueryResult, VectorDocument


class VectorDatabaseService(VectorDatabase):
    """
    Service layer for vector database operations.

    Provides a unified interface regardless of the underlying backend.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the vector database service.

        Args:
            database_url: Database URL (scheme determines backend)
            **kwargs: Backend-specific arguments
        """
        self._database_url = database_url
        self._db: Optional[VectorDatabase] = None
        self._kwargs = kwargs

    @property
    def backend_name(self) -> str:
        """Return the configured backend name."""
        if not self._database_url:
            return "unknown"
        scheme = urlparse(self._database_url).scheme.lower()
        if scheme == "postgresql":
            return "pgvector"
        if scheme == "chroma":
            return "chroma"
        return scheme or "unknown"

    def _ensure_db(self) -> VectorDatabase:
        """Lazy-load the database connection."""
        if self._db is None:
            self._db = create_vector_database(self._database_url, **self._kwargs)
            self._db.connect()
        return self._db

    def connect(self) -> None:
        """Establish connection to the database."""
        self._ensure_db()

    def disconnect(self) -> None:
        """Close connection to the database."""
        self._ensure_db().disconnect()
        self._db = None

    def initialize(self, dimension: int = 384) -> None:
        """
        Initialize the database (create tables/collections).

        Args:
            dimension: Embedding dimension
        """
        self._ensure_db().initialize(dimension=dimension)

    def upsert_documents(self, documents: List[VectorDocument]) -> int:
        """
        Insert or update documents.

        Args:
            documents: List of VectorDocument objects

        Returns:
            Number of documents inserted/updated
        """
        return self._ensure_db().upsert_documents(documents)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DBQueryResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Additional filters

        Returns:
            List of (document, score) tuples
        """
        return self._ensure_db().similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters or {}
        )

    def lexical_search(
        self,
        query: str,
        top_k: int = 40,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DBQueryResult]:
        """
        Search for documents using lexical/keyword search.

        Args:
            query: Search query text
            top_k: Number of results
            filters: Additional filters

        Returns:
            List of (document, score) tuples
        """
        return self._ensure_db().lexical_search(
            query=query,
            top_k=top_k,
            filters=filters or {}
        )

    def search_by_example(
        self,
        example: Dict[str, Any],
        top_k: int = 20,
        mode: str = "match",
    ) -> List[DBQueryResult]:
        """
        Search for documents matching example metadata.

        Args:
            example: Metadata dict to match
            top_k: Maximum results
            mode: 'match', 'contains', or 'nested'

        Returns:
            List of (document, score) tuples
        """
        return self._ensure_db().search_by_example(
            example=example,
            top_k=top_k,
            mode=mode,
        )

    def get_documents(
        self,
        doc_id: Optional[str] = None,
        substance_uuid: Optional[UUID] = None,
        sections: Optional[List[str]] = None,
        root_sections: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[VectorDocument]:
        """
        Get documents with flexible filtering.

        Args:
            doc_id: Optional document ID to retrieve a specific document
            substance_uuid: Optional substance UUID to retrieve all documents for a substance
            sections: Optional list of section names to filter results.
                     Results are sorted in the order provided. Use OR logic if multiple sections.
            root_sections: Optional list of root section names to filter results.
            limit: Optional limit on number of results

        Returns:
            List of matching documents
        """
        return self._ensure_db().get_documents(
            doc_id=doc_id,
            substance_uuid=substance_uuid,
            sections=sections,
            root_sections=root_sections,
            limit=limit
        )

    def get_root_sections(
        self,
        substance_uuid: Optional[UUID] = None,
    ) -> List[str]:
        """Get distinct root_sections, optionally scoped to a substance."""
        return self._ensure_db().get_root_sections(substance_uuid)

    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a field."""
        return self._ensure_db().get_unique_values(field)

    def delete(self, substance_uuid: Optional[UUID] = None) -> int:
        """
        Delete all chunks for a substance.

        Args:
            substance_uuid: Substance UUID

        Returns:
            Number of deleted documents
        """
        return self._ensure_db().delete(substance_uuid)

    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        return self._ensure_db().get_statistics()

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.disconnect()
            self._db = None
