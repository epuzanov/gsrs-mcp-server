"""
GSRS MCP Server - Vector Database Backends

Backends are imported lazily to avoid requiring all dependencies.
"""

__all__ = [
    "PGVectorDatabase",
    "ChromaDatabase",
]


def __getattr__(name):
    """Lazy import backends."""
    if name == "PGVectorDatabase":
        from app.db.backends.pgvector import PGVectorDatabase
        return PGVectorDatabase
    elif name == "ChromaDatabase":
        from app.db.backends.chroma import ChromaDatabase
        return ChromaDatabase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
