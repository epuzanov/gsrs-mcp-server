"""
GSRS MCP Server - API Schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

from app.models.db import VectorDocument


class IngestRequest(BaseModel):
    """Request to ingest a substance document."""
    substance: Dict[str, Any] = Field(..., description="GSRS Substance JSON document")


class IngestResponse(BaseModel):
    """Response after ingesting a substance."""
    substance_uuid: str
    chunks_created: int
    element_paths: List[str]


class QueryRequest(BaseModel):
    """Request for semantic search."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")


class QueryResult(BaseModel):
    """A single query result."""
    element_path: str
    substance_uuid: str
    text: str
    similarity_score: float
    metadata: Dict[str, Any]

    def __init__(self, chunk: VectorDocument, score: float = 0.0):
        """Accept `chunk` and `score` constructor input."""
        super().__init__(
            element_path=str(chunk.section),
            substance_uuid=str(chunk.document_id),
            text=str(chunk.text),
            similarity_score=score,
            metadata=chunk.metadata_json
        )


class QueryResponse(BaseModel):
    """Response for semantic search."""
    query: str
    results: List[QueryResult]
    total_results: int


class BatchIngestRequest(BaseModel):
    """Request to ingest multiple substances."""
    substances: List[Dict[str, Any]] = Field(..., description="List of GSRS Substance JSON documents")


class BatchIngestResponse(BaseModel):
    """Response after batch ingestion."""
    total_substances: int
    total_chunks: int
    successful: int
    failed: int
    errors: List[str] = []


class ModelInfo(BaseModel):
    """Information about the embedding model."""
    name: str
    path: str
    dimension: int
    description: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database_connected: bool
    model_loaded: bool
    statistics: Dict[str, int]


# ERI (External Retrieval Interface) Schemas
class ERIQueryRequest(BaseModel):
    """ERI query request schema."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")


class ERIResult(BaseModel):
    """A single ERI result."""
    id: str = Field(..., description="Unique result identifier")
    text: str = Field(..., description="Result text content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")

    def __init__(self, chunk: VectorDocument, score: float = 0.0):
        """Accept `metadata` constructor input and store it in `metadata_json`."""
        metadata = chunk.metadata_json or {}
        metadata["section"] = str(chunk.section)
        metadata["document_id"] = str(chunk.document_id)
        metadata["source_url"] = str(chunk.source_url)
        super().__init__(
            id=str(chunk.chunk_id),
            text=str(chunk.text),
            score=score,
            metadata=metadata
        )


class ERIQueryResponse(BaseModel):
    """ERI query response schema."""
    results: List[ERIResult] = Field(default_factory=list, description="List of results")


class DeleteResponse(BaseModel):
    """Response after deletion."""
    substance_uuid: str
    chunks_deleted: int


class AvailableModelsResponse(BaseModel):
    """Response with available embedding models."""
    models: Dict[str, Dict[str, str]]
    current_model: str


# Ask endpoint schemas
class AskRequest(BaseModel):
    """Request for the full answering pipeline."""
    query: str = Field(..., description="Question text or GSRS JSON")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata filters")
    substance_classes: Optional[List[str]] = Field(default=None, description="Substance class filters")
    sections: Optional[List[str]] = Field(default=None, description="Section filters")
    answer_style: Literal["concise", "standard", "detailed"] = Field(default="standard", description="Answer style")
    return_evidence: bool = Field(default=True, description="Whether to return evidence chunks")
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    substance_json: Optional[Dict[str, Any]] = Field(default=None, description="GSRS substance JSON for similarity search")
    debug: bool = Field(default=False, description="Include internal routing and ranking details")


class Citation(BaseModel):
    """A citation to a retrieved chunk."""
    chunk_id: str
    document_id: str
    section: str
    source_url: Optional[str] = None
    quote: Optional[str] = None


class AskResponse(BaseModel):
    """Response from the answering pipeline."""
    query: str
    rewritten_queries: List[str]
    applied_filters: Dict[str, Any]
    answer: Optional[str]
    citations: List[Citation]
    evidence_chunks: List[QueryResult]
    confidence: float
    abstained: bool
    abstain_reason: Optional[str] = None
    degraded: bool = False
    degraded_reason: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


# Similar substance search schemas
class SimilarSubstanceRequest(BaseModel):
    """Request to find similar substances based on a GSRS JSON file."""
    substance: Dict[str, Any] = Field(..., description="GSRS Substance JSON document to match against")
    top_k: int = Field(default=10, ge=1, le=50, description="Maximum number of matching substances")
    match_mode: Literal["match", "contains"] = Field(default="contains", description="Matching strategy")
    exclude_self: bool = Field(default=True, description="Exclude exact matches (same substance UUID)")


class SimilarSubstanceResult(BaseModel):
    """A matching substance with similarity score."""
    substance_uuid: str
    canonical_name: Optional[str] = None
    match_score: float
    matched_fields: List[str]
    chunks: List[QueryResult]


class SimilarSubstanceResponse(BaseModel):
    """Response with similar substances."""
    query_substance_name: Optional[str] = None
    results: List[SimilarSubstanceResult]
    total_substances: int
    total_chunks: int
