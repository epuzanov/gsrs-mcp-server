"""
GSRS MCP Server - Vector Database Abstract Interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from uuid import UUID
from app.models import DBQueryResult


class VectorDatabase(ABC):
    """
    Abstract base class for vector database backends.

    Implement this class to add support for different vector databases.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the database."""
        pass

    @abstractmethod
    def initialize(self, dimension: int) -> None:
        """Create a new collection/table for embeddings."""
        pass

    @abstractmethod
    def upsert_documents(self, documents: List[Any]) -> int:
        """
        Insert or update documents.

        Returns:
            Number of documents inserted/updated
        """
        pass

    @abstractmethod
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
            filters: Optional filters

        Returns:
            List of query results with scores
        """
        pass

    @abstractmethod
    def lexical_search(
        self,
        query: str,
        top_k: int = 40,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQueryResult]:
        """
        Search for documents using lexical/keyword search.
        """
        pass

    @abstractmethod
    def search_by_example(
        self,
        example: Dict[str, Any],
        top_k: int = 20,
        mode: str = "match",
    ) -> List[DBQueryResult]:
        """
        Search for documents matching example metadata.
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Any]:
        """Get a document by ID."""
        pass

    @abstractmethod
    def get_documents_by_substance(
        self,
        substance_uuid: UUID,
        limit: Optional[int] = None
    ) -> List[Any]:
        """Get all documents for a substance."""
        pass

    @abstractmethod
    def delete_documents_by_substance(self, substance_uuid: UUID) -> int:
        """
        Delete all documents for a substance.

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all documents from the database."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        pass

    @abstractmethod
    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a field."""
        pass
