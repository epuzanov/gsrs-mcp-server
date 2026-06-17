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
    def get_documents(
        self,
        doc_id: Optional[str] = None,
        substance_uuid: Optional[UUID] = None,
        sections: Optional[List[str]] = None,
        root_sections: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Any]:
        """
        Get documents with flexible filtering.

        Query by document ID, substance UUID, or substance with specific sections/root_sections.

        Args:
            doc_id: Optional chunk ID to retrieve document by chunk_id
            substance_uuid: Optional substance UUID to retrieve all documents for substance
            sections: Optional list of section names to filter results (requires substance_uuid).
                     Results are sorted in the order provided. Use OR logic if multiple sections.
            root_sections: Optional list of root section names to filter results.
                          May be used with or without substance_uuid depending on backend.
            limit: Optional limit on number of results

        Returns:
            List of documents matching the criteria
        """
        pass

    @abstractmethod
    def get_root_sections(
        self,
        substance_uuid: Optional[UUID] = None,
    ) -> List[str]:
        """
        Get distinct root_sections, optionally scoped to a substance.

        Args:
            substance_uuid: Optional substance UUID to scope the aggregation.

        Returns:
            Sorted list of distinct root_section values.
        """
        pass

    @abstractmethod
    def delete(self, substance_uuid: Optional[UUID] = None) -> int:
        """
        Delete all documents for a substance.

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        pass

    @abstractmethod
    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a field."""
        pass