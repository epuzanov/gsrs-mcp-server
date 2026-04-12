"""
GSRS MCP Server - Vector Database Factory
Creates the appropriate vector database backend based on configuration.
"""
from typing import Optional
from urllib.parse import urlparse
from app.db.base import VectorDatabase
from app.config import settings


def detect_backend(database_url: str) -> str:
    """Detect vector backend from database URL scheme."""
    parsed = urlparse(database_url)
    scheme = parsed.scheme.lower()
    
    if scheme == "postgresql":
        return "pgvector"
    elif scheme == "chroma":
        return "chroma"
    else:
        raise ValueError(
            f"Unknown database scheme: {scheme}. "
            f"Supported schemes: 'postgresql', 'chroma'"
        )


def create_vector_database(
    database_url: Optional[str] = None,
    **kwargs
) -> VectorDatabase:
    """
    Factory function to create a vector database instance.

    Args:
        database_url: Database URL (scheme determines backend)
        **kwargs: Additional backend-specific arguments

    Returns:
        VectorDatabase instance

    Raises:
        ValueError: If backend is not supported
        ImportError: If required dependencies are not installed
    """
    database_url = database_url or settings.database_url
    backend = detect_backend(database_url)

    if backend == "pgvector":
        from app.db.backends.pgvector import PGVectorDatabase
        return PGVectorDatabase(database_url=database_url)

    elif backend == "chroma":
        from app.db.backends.chroma import ChromaDatabase
        return ChromaDatabase(database_url=database_url)

    else:
        raise ValueError(
            f"Unsupported vector backend: {backend}. "
            f"Supported backends: 'pgvector', 'chroma'"
        )


def get_available_backends() -> list:
    """Get list of available backends based on installed dependencies."""
    backends = []
    
    # pgvector is always available (in requirements)
    backends.append("pgvector")
    
    # Check if chromadb is installed
    try:
        import chromadb
        backends.append("chroma")
    except ImportError:
        pass
    
    return backends
