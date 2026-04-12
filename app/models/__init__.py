"""
GSRS MCP Server - Data Models and API Schemas
"""
from app.models.api import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, QueryResult,
    BatchIngestRequest, BatchIngestResponse,
    ModelInfo, HealthResponse,
    DeleteResponse, AvailableModelsResponse,
    ERIQueryRequest, ERIQueryResponse, ERIResult,
    AskRequest, AskResponse, Citation,
    SimilarSubstanceRequest, SimilarSubstanceResponse, SimilarSubstanceResult,
)
from app.models.db import (
    Base, VectorDocument, DBQueryResult
)

__all__ = [
    "Base",
    "VectorDocument",
    "DBQueryResult",
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "QueryResult",
    "BatchIngestRequest",
    "BatchIngestResponse",
    "ModelInfo",
    "HealthResponse",
    "DeleteResponse",
    "AvailableModelsResponse",
    "ERIQueryRequest",
    "ERIQueryResponse",
    "ERIResult",
    "AskRequest",
    "AskResponse",
    "Citation",
    "SimilarSubstanceRequest",
    "SimilarSubstanceResponse",
    "SimilarSubstanceResult",
]
