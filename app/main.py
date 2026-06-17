"""Compact GSRS MCP server."""
import atexit
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.observability import ToolTelemetry, configure_logging
from app.runtime import ServerRuntime
from app.services.summary import substance_to_markdown

configure_logging(
    settings.debug_mode,
    use_stderr=settings.mcp_transport == "stdio",
)
logger = logging.getLogger(__name__)


class SimpleTokenVerifier(TokenVerifier):
    """Validates HTTP Bearer tokens against the configured MCP password."""

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        if token == settings.mcp_password:
            return AccessToken(
                token=token,
                client_id="mcp-client",
                scopes=["mcp:tools"],
            )
        return None


runtime = ServerRuntime(settings)


@asynccontextmanager
async def server_lifespan(server):
    """Initialise shared runtime services on startup."""
    _ensure_runtime_initialized()
    yield


def _log_runtime_status() -> None:
    """Emit a consistent startup log snapshot after runtime initialization."""
    payload = runtime.get_status_payload()
    logger.info("runtime_initialized", extra=payload)
    if not runtime.ready:
        logger.warning(
            "runtime_not_ready",
            extra={
                "backend": runtime.backend_name,
                "readiness_summary": payload.get("readiness_summary"),
                "required_component_errors": payload.get("required_component_errors", {}),
            },
        )
    elif runtime.degraded:
        logger.warning(
            "runtime_degraded",
            extra={
                "backend": runtime.backend_name,
                "degraded_summary": payload.get("degraded_summary"),
                "optional_component_errors": payload.get("optional_component_errors", {}),
            },
        )


def _ensure_runtime_initialized() -> None:
    """Initialize the shared runtime once."""
    if getattr(runtime, "initialized", False):
        return
    runtime.initialize()
    _log_runtime_status()


def _shutdown_runtime() -> None:
    """Best-effort process shutdown for long-lived runtime clients."""
    if getattr(runtime, "initialized", False):
        runtime.shutdown()


atexit.register(_shutdown_runtime)


def _build_auth_settings(app_settings) -> tuple[AuthSettings | None, TokenVerifier | None]:
    """Build MCP HTTP auth settings from the current runtime configuration."""
    if not app_settings.mcp_password:
        return None, None

    auth_settings = AuthSettings(
        issuer_url=AnyHttpUrl("http://localhost"),
        resource_server_url=AnyHttpUrl(f"http://localhost:{app_settings.mcp_port}"),
        required_scopes=["mcp:tools"],
    )
    return auth_settings, SimpleTokenVerifier()


auth, token_verifier = _build_auth_settings(settings)

mcp = FastMCP(
    "GSRS MCP Server",
    instructions=(
        "Compact GSRS MCP server. Use rag_query for local retrieval evidence with parent context, "
        "rag_query_chunks for raw chunk retrieval, rag_ingest to load GSRS substance JSON, "
        "gsrs_get_substance/summary for upstream records, and the GSRS API search tools for live API lookup. "
        "Use get_parent_context to explore parent context for specific chunks."
    ),
    token_verifier=token_verifier,
    auth=auth,
    host=settings.mcp_api,
    port=settings.mcp_port,
    streamable_http_path="/mcp",
    lifespan=server_lifespan,
)


@mcp.custom_route("/livez", methods=["GET"], include_in_schema=True)
async def live_check(request: Request) -> JSONResponse:
    """Liveness probe: process is up if this route responds."""
    return JSONResponse({"status": "alive"})


@mcp.custom_route("/readyz", methods=["GET"], include_in_schema=True)
async def readiness_check(request: Request) -> JSONResponse:
    """Readiness probe for local retrieval dependencies."""
    _ensure_runtime_initialized()
    payload = runtime.get_status_payload()
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)


@mcp.custom_route("/health", methods=["GET"], include_in_schema=True)
async def health_check(request: Request) -> JSONResponse:
    """Combined health endpoint with liveness, readiness, and dependency state."""
    _ensure_runtime_initialized()
    payload = runtime.get_status_payload()
    payload["live"] = True
    return JSONResponse(payload)


def _tool_call(tool_name: str, *, query_type: Optional[str] = None) -> ToolTelemetry:
    """Create a telemetry context for a tool call."""
    return ToolTelemetry.start(
        logger=logger,
        metrics=runtime.metrics,
        tool_name=tool_name,
        backend=runtime.backend_name,
        query_type=query_type,
    )


def _parse_json_object(value: str | None, *, field_name: str) -> dict[str, Any]:
    """Parse an optional JSON object argument."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return parsed


def _parse_list(value: str | None) -> list[str]:
    """Parse comma-separated or JSON-array tool arguments."""
    if not value:
        return []
    raw = value.strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array.")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _display_name(payload: dict[str, Any], default: str = "?") -> str:
    """Return the preferred display label for a GSRS substance-like payload."""
    if payload.get("_name"):
        return str(payload["_name"])
    names = payload.get("names")
    if isinstance(names, list):
        for entry in names:
            if not isinstance(entry, dict):
                continue
            display = entry.get("displayName", entry.get("display_name", False))
            if isinstance(display, str):
                display = display.strip().lower() == "true"
            if display and entry.get("name"):
                return str(entry["name"])
    for key in ("display_name", "substance_name", "entity_name", "name"):
        if payload.get(key):
            return str(payload[key])
    return default


def _format_api_results(payload: dict[str, Any], *, label: str) -> str:
    """Format a GSRS API search response as compact markdown."""
    results = payload.get("content") or payload.get("results") or []
    total = payload.get("total", len(results))
    if not results:
        return f"No results found for **{label}**."

    lines = [f"Found **{total}** result(s) for **{label}**:\n"]
    for index, item in enumerate(results, 1):
        if not isinstance(item, dict):
            continue
        uuid = item.get("uuid", "?")
        name = _display_name(item)
        substance_class = item.get("substanceClass", "")
        approval_id = item.get("_approvalIDDisplay") or item.get("approvalID")
        lines.append(f"{index}. **{name}** ({substance_class})")
        lines.append(f"   UUID: `{uuid}`")
        if approval_id:
            lines.append(f"   Approval ID: `{approval_id}`")
    return "\n".join(lines)


def _ingest_substance_payload(substance: dict[str, Any]) -> tuple[str, int]:
    """Validate, chunk, embed, and upsert a GSRS substance payload."""
    if runtime.chunker is None:
        raise RuntimeError("Chunker is not initialized.")

    chunks = runtime.chunker.chunk(substance)
    if not chunks:
        return str(substance.get("uuid", "unknown")), 0

    display_name = _display_name(substance, default="")
    texts = [str(chunk.text) for chunk in chunks]
    embeddings = runtime.embedding_service.embed_batch(texts)

    documents = []
    for chunk, embedding in zip(chunks, embeddings):
        if display_name:
            metadata = chunk.metadata_json or {}
            metadata["display_name"] = display_name
            metadata["substance_name"] = display_name
            chunk.metadata_json = metadata
        chunk.set_embedding(embedding)
        documents.append(chunk)

    count = runtime.vector_db.upsert_documents(documents)
    return str(substance.get("uuid", "unknown")), count


@mcp.tool()
async def rag_query_chunks(
    query: str,
    top_k: int = 8,
    filters: str = "",
) -> str:
    """Search local ingested GSRS chunks and return retrieval evidence only."""
    _ensure_runtime_initialized()
    tool = _tool_call("rag_query_chunks", query_type="rag")
    try:
        if not runtime.retrieval_available():
            reason = runtime.retrieval_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"RAG query is currently unavailable: {reason}"

        parsed_filters = _parse_json_object(filters, field_name="filters")
        embedding = runtime.embedding_service.embed(query)
        results = runtime.vector_db.similarity_search(
            embedding,
            top_k=max(1, min(top_k, 50)),
            filters=parsed_filters,
        )
        tool.finish("success" if results else "abstained", result_count=len(results), citation_count=0)
        if not results:
            return f"No local RAG results found for **{query}**."

        lines = [f"Found **{len(results)}** local RAG result(s) for **{query}**:\n"]
        for index, result in enumerate(results, 1):
            document = result.document
            text = document.text.strip()
            if len(text) > 700:
                text = text[:700].rstrip() + "..."
            lines.append(f"{index}. Score: `{result.score:.4f}`")
            lines.append(f"   Substance UUID: `{document.document_id}`")
            lines.append(f"   Section: `{document.section}`")
            lines.append(f"   Chunk: `{document.chunk_id}`")
            lines.append(f"   Text: {text}")
        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"RAG query error: {exc}"


@mcp.tool()
async def rag_ingest(substance_json: str) -> str:
    """Ingest one GSRS substance JSON document into the local RAG store."""
    _ensure_runtime_initialized()
    tool = _tool_call("rag_ingest", query_type="substance_json")
    try:
        if not runtime.ingestion_available():
            reason = runtime.ingestion_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"RAG ingest is currently unavailable: {reason}"

        substance = _parse_json_object(substance_json, field_name="substance_json")
        uid, count = _ingest_substance_payload(substance)
        tool.finish("success" if count else "abstained", result_count=count, citation_count=0)
        if count == 0:
            return f"No chunks generated from substance **{uid}**."
        return f"Ingested **{uid}** into local RAG store: **{count}** chunk(s)."
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"RAG ingest error: {exc}"


@mcp.tool()
async def rag_query(
    query: str,
    top_k: int = 8,
    include_parent_text: bool = True,
    parent_text_limit: int = 1000,
    filters: str = "",
) -> str:
    """Search local ingested GSRS chunks with parent context reconstruction.

    Performs RAG query and enriches results with parent context reconstructed
    from chunks sharing the same (document_id, root_section). This provides
    broader document context without requiring dedicated parent storage.
    """
    _ensure_runtime_initialized()
    tool = _tool_call("rag_query", query_type="rag_with_parent")
        if not runtime.retrieval_available():
            reason = runtime.retrieval_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"RAG query is currently unavailable: {reason}"

        parsed_filters = _parse_json_object(filters, field_name="filters")
        embedding = runtime.embedding_service.embed(query)
        results = runtime.vector_db.similarity_search(
            embedding,
            top_k=max(1, min(top_k, 50)),
            filters=parsed_filters,
        )

        if not results:
            tool.finish("abstained", result_count=0, citation_count=0)
            return f"No local RAG results found for **{query}**."

        # Enrich results with parent context
        enriched_results = runtime.parent_enricher.enrich_search_results(
            results,
            include_parent_text=include_parent_text,
            parent_text_limit=parent_text_limit,
        )

        tool.finish("success", result_count=len(enriched_results), citation_count=0)

        lines = [f"Found **{len(enriched_results)}** local RAG result(s) with parent context for **{query}**:\n"]
        for index, enriched in enumerate(enriched_results, 1):
            chunk = enriched["chunk"]
            score = enriched["score"]
            text = chunk["text"].strip()
            if len(text) > 500:
                text = text[:500].rstrip() + "..."

            lines.append(f"{index}. Score: `{score:.4f}` | Section: `{chunk['section']}`")
            lines.append(f"   Substance UUID: `{chunk['document_id']}`")
            lines.append(f"   Chunk: `{chunk['chunk_id']}`")
            lines.append(f"   Text: {text}")

            # Include parent context summary if available
            if "parent_context" in enriched:
                parent_ctx = enriched["parent_context"]
                lines.append(f"   **Parent Context**: {parent_ctx['num_chunks']} chunks in {', '.join(parent_ctx['sections_included'])}")
                if "parent_text_summary" in enriched:
                    parent_text = enriched["parent_text_summary"]
                    lines.append(f"   Parent Summary: {parent_text[:300]}..." if len(parent_text) > 300 else f"   Parent Summary: {parent_text}")

        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"RAG query with parent context error: {exc}"


@mcp.tool()
async def get_parent_context(chunk_id: str) -> str:
    """Retrieve parent context for a specific chunk by chunk_id.
    
    Returns the reconstructed parent context containing all chunks from the
    same document and root section as the specified chunk.
    """
    _ensure_runtime_initialized()
    tool = _tool_call("get_parent_context", query_type="parent_context")
    try:
        if not runtime.retrieval_available():
            reason = runtime.retrieval_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"Retrieval is currently unavailable: {reason}"

        # Get the chunk
        docs = runtime.vector_db.get_documents(doc_id=chunk_id)
        chunk = docs[0] if docs else None
        if chunk is None:
            tool.finish("abstained", result_count=0, citation_count=0)
            return f"Chunk **{chunk_id}** not found."

        # Get parent context
        enricher = runtime.parent_enricher
        parent_identity = enricher.get_parent_identity(chunk)
        parent_context = enricher.reconstruct_parent_context(parent_identity)

        if not parent_context:
            tool.finish("abstained", result_count=0, citation_count=0)
            return f"No parent context found for chunk **{chunk_id}** (Document: {parent_identity.document_id}, Section: {parent_identity.root_section})"

        tool.finish("success", result_count=1, citation_count=0)

        lines = [
            f"**Parent Context for Chunk**: `{chunk_id}`\n",
            f"**Document**: `{parent_identity.document_id}`",
            f"**Root Section**: `{parent_identity.root_section}`",
            f"**Num Chunks in Parent**: {parent_context['num_chunks']}",
            f"**Sections Included**: {', '.join(parent_context['sections_included'])}\n",
            "**Text Parts**:",
        ]

        for i, part in enumerate(parent_context.get("text_parts", [])[:10], 1):
            section = part.get("section", "unknown")
            text = part.get("text", "")[:400]
            lines.append(f"{i}. [{section}] {text}...")

        if parent_context.get("metadata_unified"):
            lines.append("\n**Unified Metadata**:")
            for key, value in list(parent_context["metadata_unified"].items())[:5]:
                value_str = str(value)[:100]
                lines.append(f"- {key}: {value_str}")

        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Get parent context error: {exc}"


@mcp.tool()
async def gsrs_get_substance(identifier: str) -> str:
    """Fetch a complete GSRS substance JSON document by UUID or approval identifier."""
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_substance", query_type="identifier")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        substance = runtime.gsrs_api.get_substance(identifier)
        tool.finish("success" if substance else "abstained", result_count=1 if substance else 0, citation_count=0)
        if substance is None:
            return f"Substance **{identifier}** not found in GSRS API."
        return json.dumps(substance, indent=2, default=str)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS get_substance error: {exc}"


@mcp.tool()
async def gsrs_get_summary(identifier: str) -> str:
    """Fetch a GSRS substance and return a markdown summary."""
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_summary", query_type="identifier")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        substance = runtime.gsrs_api.get_substance(identifier)
        tool.finish("success" if substance else "abstained", result_count=1 if substance else 0, citation_count=0)
        if substance is None:
            return f"Substance **{identifier}** not found in GSRS API."
        return substance_to_markdown(substance)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS summary error: {exc}"


@mcp.tool()
async def gsrs_parametric_search(
    query: str = "",
    filters: str = "",
    facets: str = "",
    page: int = 1,
    size: int = 20,
    fields: str = "",
) -> str:
    """Search GSRS API with free text, fielded filters, and optional facets."""
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_parametric_search", query_type="parametric")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        parsed_filters = _parse_json_object(filters, field_name="filters")
        parsed_facets = _parse_list(facets)
        payload = runtime.gsrs_api.parametric_search(
            query=query,
            filters=parsed_filters,
            facets=parsed_facets,
            page=max(1, page),
            size=max(1, min(size, 100)),
            fields=fields or None,
        )
        results = payload.get("content") or payload.get("results") or []
        tool.finish("success" if results else "abstained", result_count=len(results), citation_count=0)
        label = query or json.dumps(parsed_filters, sort_keys=True) or "parametric search"
        return _format_api_results(payload, label=label)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS parametric search error: {exc}"


@mcp.tool()
async def gsrs_structure_search(
    structure: str,
    search_type: Literal["exact", "exactplus", "sim", "substructure", "flex", "flexplus"] = "exact",
    cutoff: float = 0.8,
    size: int = 20,
) -> str:
    """Search substances by chemical structure via the GSRS API."""
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_structure_search", query_type="structure")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"
        payload = runtime.gsrs_api.structure_search(
            structure=structure,
            search_type=search_type,
            cutoff=cutoff,
            size=max(1, min(size, 100)),
        )
        results = payload.get("content") or payload.get("results") or []
        tool.finish("success" if results else "abstained", result_count=len(results), citation_count=0)
        return _format_api_results(payload, label=f"{search_type} structure search")
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS structure search error: {exc}"


@mcp.tool()
async def gsrs_sequence_search(
    sequence: str,
    search_type: Literal["GLOBAL", "SUB"] = "GLOBAL",
    sequence_type: Literal["protein", "nucleicAcid"] = "protein",
    cutoff: float = 0.95,
    size: int = 20,
) -> str:
    """Search substances by protein or nucleic-acid sequence via the GSRS API."""
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_sequence_search", query_type="sequence")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"
        payload = runtime.gsrs_api.sequence_search(
            sequence=sequence,
            search_type=search_type,
            sequence_type=sequence_type,
            cutoff=cutoff,
            size=max(1, min(size, 100)),
        )
        results = payload.get("content") or payload.get("results") or []
        tool.finish("success" if results else "abstained", result_count=len(results), citation_count=0)
        return _format_api_results(payload, label=f"{search_type} {sequence_type} sequence search")
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS sequence search error: {exc}"


@mcp.tool()
async def health() -> str:
    """Return structured runtime health and readiness information."""
    _ensure_runtime_initialized()
    tool = _tool_call("health", query_type="runtime")
    payload = runtime.get_status_payload()
    payload["live"] = True
    tool.finish("success", result_count=0, citation_count=0)
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
async def statistics() -> str:
    """Return local vector store statistics."""
    _ensure_runtime_initialized()
    tool = _tool_call("statistics", query_type="statistics")
    try:
        if not runtime.vector_backend_available():
            reason = runtime.vector_backend_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"Statistics are currently unavailable: {reason}"
        payload = runtime.vector_db.get_statistics()
        tool.finish("success", result_count=0, citation_count=0)
        return json.dumps(payload, indent=2, default=str)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Statistics error: {exc}"


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport=settings.mcp_transport, mount_path="/")


if __name__ == "__main__":
    main()
