"""Compact GSRS MCP server."""
import atexit
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import Prompt, PromptMessage, TextContent, ToolAnnotations

from app.config import settings
from app.observability import ToolTelemetry, configure_logging
from app.runtime import ServerRuntime
from app.services.details import FilterError, validate_filter
from app.services.gsrs_schema import (
    GsrsSchemaError,
    SUPPORTED_MODELS as GSRS_SCHEMA_MODELS,
    get_model_schema,
    get_model_schema_text,
)
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
        "and the GSRS API search tools for live API lookup. "
        "Resources: gsrs://substances/{identifier} for raw JSON, gsrs://substances/{identifier}/summary for markdown summary, "
        "gsrs://substances/{identifier}/parents/{root_section}/summary for a chunk-driven parent summary, "
        "gsrs://substances/{identifier}/details/{filter} for the GSRS details section, "
        "gsrs://cv/domains for controlled vocabulary domain list, gsrs://cv/{domain}/terms for controlled vocabulary lookup, "
        "gsrs://schema/{model} for the Pydantic JSON Schema of a GSRS substance model (use as a reference for field names when calling gsrs_parametric_search or gsrs_get_substance_details), "
        "and server://health / server://statistics for runtime state. "
        "Use get_parent_context to explore parent context for specific chunks and get_parent_summary to render a parent as markdown. "
        "Prompts: fetch_substance and substance_summary for GSRS record lookup; "
        "resolve_cv_terms to decode CV codes; rag_reasoning to answer from the local RAG store.\n\n"
        "GSRS vocabulary (apply when interpreting tool output and when calling tools):\n"
        "- Display Name: the substance's preferred name — the entry in ``substance.names`` whose ``displayName`` is true. The local RAG chunker writes the same value on every chunk's ``metadata_json.display_name`` and as the H1 header of ``get_parent_summary`` output. Substance UUID and Substance Name are surfaced side by side in ``rag_query`` / ``rag_query_chunks`` results.\n"
        "- Name types: ``of`` = Official Name, ``sys`` = Systematic Name, ``cn`` = Common Name, ``bn`` = Brand Name, ``sci`` = Scientific Name, ``syn`` = Synonym, ``cd`` = Code. The raw code is preserved on each name chunk's ``name_type`` metadata field.\n"
        "- Naming Organizations (``nameOrgs``) such as INN, USAN, INCI are surfaced on Official Name chunks.\n"
        "- Access: ``access_status`` is ``Public`` (absent / empty list) or ``Protected`` (non-empty list, e.g. ``[\"admin\"]``). Per-row access on a name, code, or classification overrides the substance-level access.\n"
        "- Definition types: ``PRIMARY`` (the substance is its own root) and ``ALTERNATIVE`` (re-parented under the primary substance via a ``SUB_ALTERNATE->SUBSTANCE`` relationship for parent-child retrieval).\n"
        "- Substance classes: ``chemical``, ``protein``, ``nucleicAcid``, ``polymer``, ``mixture``, ``structurallyDiverse``, ``concept``, ``specifiedSubstanceG1..G4`` (all G-variants collapse to the ``specifiedsubstance`` chunk section). Per-class payload (structure, moieties, sequence, monomers, source material, constituents) lives under the ``definitions`` root parent in the local RAG store.\n"
        "- Codes, identifiers, and classifications: a single ``substance.codes`` array holds both identifiers (regular codes such as CAS, UNII, WHO-ATC) and classifications (controlled-vocabulary category paths). The local RAG chunker splits each entry into its own chunk and routes it to one of two sub-sections under the ``codes`` root: ``identifiers`` (the default) and ``classifications`` (when ``_isClassification`` is truthy, or when ``comments`` contains a ``|`` character, a pragmatic marker for classification-style rows in payloads that lack the explicit flag). Identifier chunks lead with the label ``Identifier: <CODE_SYSTEM>:<CODE>`` and surface the ``code_system``, ``code_type``, ``code_url``, and per-row ``access`` on the chunk metadata. Classification chunks lead with ``Classification: <CODE_SYSTEM>:<CODE>`` and surface ``code_system``, ``code_url``, and ``comments`` on the metadata. Both sub-sections share the ``codes`` root, so a single parent query on ``codes`` (or on any substance, via the relationship between the two) returns both. The ``codes`` field in the GSRS payload has no top-level equivalent — there is no separate ``classifications`` array. Pass ``root_section=codes`` to ``get_parent_summary`` to render both at once.\n"
        "- Relationship types and the typed sub-buckets they route to: ``ACTIVE MOIETY`` and ``SUBSTANCE PART`` → Active Moieties; ``METABOLITE INACTIVE->PARENT`` → Metabolites; ``IMPURITY->PARENT`` → Impurities; types containing ``CONSTITUENT`` → Constituents; ``SALT/SOLVATE->PARENT`` → Salts or Solvates; anything else → Other Relationships. A relationship of type ``SUB_ALTERNATE->SUBSTANCE`` is the ALTERNATIVE-substance pointer used internally to re-parent chunks; it is not surfaced as a user-visible relationship.\n"
        "- Chunk sections: top-level roots are ``overview``, ``names``, ``codes`` (with sub-sections ``identifiers`` and ``classifications``), ``definitions`` (with sub-sections ``chemical``, ``moieties``, ``protein``, ``nucleicacid``, ``polymer``, ``structurallydiverse``, ``mixture``, ``specifiedsubstance``, ``modifications``), ``relationships`` (with the typed sub-buckets above), ``properties``, ``references``, ``tags``, ``notes``. A parent-child retrieval on any root returns the chunks for every sub-section.\n"
        "- Parent identity: ``(document_id, root_section)``. Use ``get_all_parents_in_document`` to enumerate them; ``get_parent_summary`` renders a single parent as markdown without calling the GSRS upstream."
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


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
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
            # Display name lives on the chunk's metadata_json; the
            # chunker always writes it. Older indexes may not have
            # the field — fall back to the document UUID so the
            # label is never empty.
            display_name = (
                (document.metadata_json or {}).get("display_name")
                or str(document.document_id)
            )
            lines.append(f"{index}. Score: `{result.score:.4f}`")
            lines.append(f"   Substance UUID: `{document.document_id}`")
            lines.append(f"   Substance Name: `{display_name}`")
            lines.append(f"   Section: `{document.section}`")
            lines.append(f"   Chunk: `{document.chunk_id}`")
            lines.append(f"   Text: {text}")
        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"RAG query error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )
)
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


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
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

            # Display name is surfaced on the chunk dict by
            # ``enrich_chunk_with_parent`` (top-level + metadata).
            # Prefer the top-level field; fall back to the nested
            # metadata, then to the document UUID so the label is
            # never empty.
            chunk_metadata = chunk.get("metadata") or {}
            display_name = (
                chunk.get("display_name")
                or chunk_metadata.get("display_name")
                or str(chunk["document_id"])
            )
            lines.append(f"{index}. Score: `{score:.4f}` | Section: `{chunk['section']}`")
            lines.append(f"   Substance UUID: `{chunk['document_id']}`")
            lines.append(f"   Substance Name: `{display_name}`")
            lines.append(f"   Chunk: `{chunk['chunk_id']}`")
            lines.append(f"   Text: {text}")

            # Include parent context summary if available
            if "parent_context" in enriched:
                parent_ctx = enriched["parent_context"]
                lines.append(f"   **Parent Context**: {parent_ctx['num_chunks']} chunks in {', '.join(parent_ctx['sections_included'])}")
                if "parent_text_summary" in enriched:
                    parent_text = enriched["parent_text_summary"]
                    lines.append(f"   Parent Summary: {parent_text[:300]}..." if len(parent_text) > 300 else f"   Parent Summary: {parent_text}")
                lines.append(f"   parent_text_truncated: {enriched.get('parent_text_truncated', False)}")
                if enriched.get("parent_grouping_fallback_used"):
                    lines.append("   parent_grouping_fallback_used: true")

        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"RAG query with parent context error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
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
        parent_identity, fallback_used = enricher.get_parent_identity(chunk)
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


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def get_parent_summary(
    substance_uuid: str,
    root_section: str,
    max_chars: int = 4000,
    include_text_parts: bool = True,
    exclude_chunk_id: str = "",
) -> str:
    """Render a parent context as a chunk-driven markdown summary.

    The summary is reconstructed from chunks in the local RAG store —
    it does not call the GSRS upstream. It is a **parent-scoped**
    view: it covers the chunks that share ``(substance_uuid,
    root_section)``, not the whole substance.

    Compare to ``gsrs_get_summary`` (which renders the full GSRS
    substance JSON): the chunk-driven summary covers whatever the
    chunker indexed, with a footer that flags the gap
    (approval ID, status, substance-level access).

    Args:
        substance_uuid: Substance UUID. Must be a valid UUID string.
        root_section: Root section identifier (e.g. ``"overview"``,
            ``"names"``, ``"codes"``, ``"definitions"``,
            ``"relationships"``, ``"properties"``,
            ``"references"``). These match the values returned by
            :func:`get_parent_context`.
        max_chars: Soft cap on the rendered markdown size. Default
            4000. When the output exceeds this, it is truncated and
            the ``parent_summary.truncated`` metric is incremented.
        include_text_parts: When True (default), include the raw
            chunk text for each rendered row. When False, only the
            structured metadata fields are rendered.
        exclude_chunk_id: Optional chunk ID to exclude from the
            summary (e.g. the child chunk the caller is showing
            alongside the summary). Empty string means no exclusion.

    Returns:
        A markdown string. Empty when the parent has no chunks.
    """
    _ensure_runtime_initialized()
    tool = _tool_call("get_parent_summary", query_type="parent_summary")
    try:
        if not runtime.retrieval_available():
            reason = runtime.retrieval_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"Retrieval is currently unavailable: {reason}"

        body, error = _render_parent_summary(
            substance_uuid=substance_uuid,
            root_section=root_section,
            max_chars=max_chars,
            include_text_parts=include_text_parts,
            exclude_chunk_id=exclude_chunk_id,
        )
        if error:
            tool.finish("abstained", result_count=0, citation_count=0)
            return error
        if not body:
            tool.finish("abstained", result_count=0, citation_count=0)
            return (
                f"No parent context found for substance "
                f"**{substance_uuid}** (root_section: `{root_section}`)."
            )
        tool.finish("success", result_count=1, citation_count=0)
        return body
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Get parent summary error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_get_substance(identifier: str) -> str:
    """Fetch a complete GSRS substance JSON document by UUID or approval identifier."""
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_substance", query_type="identifier")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        # Delegate to the canonical resource implementation (single source of truth).
        result = await gsrs_substance_resource(identifier)
        try:
            parsed = json.loads(result)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error"):
            tool.finish("abstained", result_count=0, citation_count=0)
        else:
            tool.finish("success", result_count=1, citation_count=0)
        return result
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS get_substance error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_get_summary(identifier: str) -> str:
    """Fetch a GSRS substance and return a markdown summary."""
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_summary", query_type="identifier")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        # Delegate to the canonical resource implementation (single source of truth).
        result = await gsrs_substance_summary_resource(identifier)
        if result.startswith("GSRS API is currently unavailable") or "not found in GSRS API" in result:
            tool.finish("abstained", result_count=0, citation_count=0)
        else:
            tool.finish("success", result_count=1, citation_count=0)
        return result
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS summary error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_get_substance_details(
    substance_identifier: str,
    filter_steps: str = "",
) -> str:
    """Fetch a filtered view of a GSRS substance record.

    Implements the GSRS "details" API section
    (https://gsrs.ncats.nih.gov/api-documentation). The substance is
    looked up by UUID, approval ID, or display name, then a list of
    filter steps is joined with ``!`` and appended to the path
    ``/api/v1/substances({uuid})/{joined}`` to narrow the response.

    **Reference for element paths:** before building a multi-step
    filter, call ``gsrs_get_schema(model="Substance")`` (or the
    matching subclass, e.g. ``ChemicalSubstance``) to fetch the
    Pydantic JSON Schema. The schema's top-level ``properties``
    keys are the legal element paths (e.g. ``names``,
    ``codes``, ``relationships``, ``protein``); nested model
    definitions under ``$defs`` are addressable as
    ``parent/child`` paths. The schema is also exposed as the
    resource ``gsrs://schema/{model}``.

    Args:
        substance_identifier: GSRS substance UUID, approval ID, or
            display name. Names are resolved to a UUID via a name
            search.
        filter_steps: Optional list of filter steps. Accepted as a
            JSON array (e.g. ``["names(type:of)", "(name)"]``) or a
            comma-separated string. Empty or omitted means "no
            filter" (the full substance record is returned). The
            steps are joined with ``!`` in the order given.

    Filter syntax (each step):
        The joined string is a sequence of ``!``-delimited "steps"
        applied in order. Read it left-to-right: each step takes the
        previous result and transforms it.

        The first step is an **elements path** (the only step that
        does not begin with ``!``); every subsequent step is a
        ``!``-prefixed transformation.

        1. **Elements path** — one or more ``/``-separated path
           segments that name the JSON section to walk into (e.g.
           ``names``, ``codes``, ``relationships/relatedSubstance``).
           May be followed by zero or more parenthesized
           **locators** that filter the elements:

           - ``($0)`` selects the first element of a collection by
             index.
           - ``(field:value)`` keeps only elements where
             ``field == value``.
           - ``(relatedSubstance/approvalID:WK2XYI10QM)`` uses ``/``
             to walk a sub-path before matching.

           Multiple locators may be combined; they are AND-ed
           together.
        2. **Subsequent steps** — each step picks one of these
           operations (every step is prefixed by the ``!`` delimiter
           already introduced above):

           - **`(fieldName)`** — field projection: returns a JSON
             array of values for that field on each remaining
             element.
           - **`sort(field)`** / **`revsort(field)`** — sort
             ascending or descending by field.
           - **`skip(N)`** — skip the first N elements.
           - **`limit(N)`** — keep the first N elements.
           - **`distinct(field)`** — return distinct values of
             ``field``.
           - **`count()`** — return the count as a number.
           - **`group(field)`** — return a JSON object grouped by
             ``field``.

           Steps are evaluated in the order they appear; e.g.
           ``sort(x)!limit(1)`` (after the initial elements step)
           returns the first element after sorting.

    Examples:
        # Names of type "of" projected to the "name" field
        gsrs_get_substance_details(
            "Ibuprofen",
            '["names(type:of)", "(name)"]',
        )

        # First common name
        gsrs_get_substance_details(
            "Ibuprofen",
            '["names(type:cn)", "(name)", "limit(1)"]',
        )

        # Count of non-deprecated names
        gsrs_get_substance_details(
            "Ibuprofen",
            '["names(deprecated:false)", "count()"]',
        )

        # Display names of relationships pointing to a specific substance
        gsrs_get_substance_details(
            "Ibuprofen",
            '["relationships(type:IMPURITY->PARENT)'
            '(relatedSubstance/approvalID:600T0F0PX0)",'
            ' "(relatedSubstance/refPname)"]',
        )

        # Codes sorted by code system, then projected
        gsrs_get_substance_details(
            "Ibuprofen",
            '["codes", "sort(codeSystem)", "(codeSystem)"]',
        )

        # Comma-separated form is also accepted
        gsrs_get_substance_details("Ibuprofen", "names(type:of), (name)")
    """
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_substance_details", query_type="details")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        steps = _parse_list(filter_steps)
        joined_filter = runtime.gsrs_api.join_filter_steps(steps)
        # Delegate to the canonical resource implementation (single
        # source of truth). The resource accepts the joined string
        # so it remains a thin wrapper over the GSRS URL contract.
        result = await gsrs_substance_details_resource(
            substance_identifier, joined_filter
        )
        payload = json.loads(result)
        if isinstance(payload, dict) and payload.get("error"):
            error = str(payload["error"])
            # Distinguish "abstained" (e.g. 404 / not found) from "degraded".
            if "not found" in error.lower() or "could not resolve" in error.lower():
                tool.finish("abstained", result_count=0, citation_count=0)
            else:
                tool.finish("degraded", result_count=0, citation_count=0, error_message=error)
        else:
            tool.finish("success", result_count=1, citation_count=0)
        return result
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS get_substance_details error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_parametric_search(
    query: str = "",
    filters: str = "",
    facets: str = "",
    page: int = 1,
    size: int = 20,
    fields: str = "",
) -> str:
    """Search the GSRS API with free text, fielded filters, and optional facets.

    GSRS indexes substance records with Lucene. Use this tool for general
    substance lookup and for narrowing results by indexed fields.

    **Reference for indexed field names:** the ``filters`` argument
    uses Lucene-indexed ``root_<field>`` paths whose names mirror
    the Pydantic attributes of the GSRS substance model. Before
    building a complex ``filters`` object, call
    ``gsrs_get_schema(model="Substance")`` (or the matching
    subclass) to fetch the canonical JSON Schema; the top-level
    ``properties`` keys are the legal base field names. The
    schema is also exposed as the resource ``gsrs://schema/{model}``.

    Args:
        query: Free-text search (names, codes, identifiers, etc.). GSRS promotes
            exact matches in configured fields such as names and codes.
        filters: JSON object of field:value terms, e.g.
            {"root_substanceClass": "chemical", "root_status": "approved"}.
            Multiple values for the same field can be passed as a list; they are
            combined with OR. Different fields are combined with AND. Field names
            use the indexed path, e.g. root_names_name, root_codes_code,
            root_substanceClass.
        facets: Comma-separated or JSON-array of facet filters. Each facet has
            the form "Facet Name/Value". Example:
            "Substance Class/chemical,Protein Type/ENZYME". Facets narrow the
            result set to records that belong to the given category buckets
            (combined with AND across facet groups). Use `gsrs_get_facets` to
            discover which facet names and values are available for a query.
        page: Page number (1-based).
        size: Results per page (max 100).
        fields: Comma-separated field list to return (server-dependent).

    Examples:
        query="ASPIRIN"
        filters={"root_substanceClass": "chemical"}
        facets="Substance Class/chemical"
    """
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


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_get_facets(
    query: str = "*",
    filters: str = "",
    page: int = 1,
    size: int = 20,
) -> str:
    """Discover available GSRS facet names and values for a search context.

    GSRS search responses include Lucene facet buckets that categorize the
    result set (e.g. "Substance Class", "Protein Type"). This tool returns
    those buckets so you can build accurate `facets` arguments for
    `gsrs_parametric_search`.

    Args:
        query: Free-text search context. Defaults to "*" so that all available
            system facets are returned when no context is given. Provide a
            query or filters to narrow the returned facet values to a specific
            search context.
        filters: JSON object of field:value filters (same format as
            `gsrs_parametric_search`).
        page: Page number (1-based).
        size: Page size; keep small because facet metadata is requested.

    Example:
        query="*"
        filters={"root_substanceClass": "protein"}

    Returns a markdown list of facet groups and their top values/counts.
    """
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_facets", query_type="facets")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        parsed_filters = _parse_json_object(filters, field_name="filters")
        payload = runtime.gsrs_api.get_facets(
            query=query,
            filters=parsed_filters,
            page=max(1, page),
            size=max(1, min(size, 100)),
        )
        facets = payload.get("facets") or []
        tool.finish("success", result_count=len(facets), citation_count=0)

        if not facets:
            return f"No facets available for this query context."

        lines = [f"Found **{len(facets)}** facet group(s) for this query context:\n"]
        for facet in facets:
            name = facet.get("name", "Unknown")
            values = facet.get("values") or []
            lines.append(f"### {name}")
            for value in values[:10]:
                label = value.get("label") or value.get("value", "?")
                count = value.get("count", 0)
                lines.append(f"- `{label}` ({count})")
            if len(values) > 10:
                lines.append(f"- ... and {len(values) - 10} more")
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS get facets error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_get_cv_domains(size: int = 200) -> str:
    """List available GSRS controlled vocabulary (CV) domains.

    Each domain represents a controlled vocabulary that can be queried
    with `gsrs_get_cv_terms`. Common domains include NAME_TYPE (for
    `names.type`), CODE_TYPE (for `codes.type`), and SUBSTANCE_CLASS.

    Args:
        size: Maximum number of domains to return (default 200).

    Returns:
        Markdown list of CV domain names. Use a domain name as the
        `domain` argument to `gsrs_get_cv_terms`.
    """
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_cv_domains", query_type="cv")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        # Delegate to the canonical resource implementation (single source of truth)
        # and render its JSON payload as the tool's markdown surface.
        result = await gsrs_cv_domains_resource()
        payload = json.loads(result)
        if payload.get("error"):
            tool.finish("degraded", result_count=0, citation_count=0, error_message=str(payload["error"]))
            return f"GSRS API is currently unavailable: {payload['error']}"
        domains = payload.get("domains", [])
        tool.finish("success", result_count=len(domains), citation_count=0)

        if not domains:
            return "No controlled vocabulary domains available from GSRS API."

        lines = [f"Found **{len(domains)}** controlled vocabulary domain(s):\n"]
        for entry in domains:
            domain = entry.get("domain") or entry.get("value") or "?"
            display = entry.get("display") or domain
            lines.append(f"- `{domain}` — {display}")
        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS get CV domains error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_get_cv_terms(domain: str) -> str:
    """Return the terms for a single GSRS controlled vocabulary domain.

    Use this to resolve short codes such as `of` (Official Name), `sys`
    (Systematic Name), `cn` (Common Name), or `cd` (Code) from the
    `NAME_TYPE` domain, and analogous codes from other domains.

    Args:
        domain: CV domain name, e.g. `NAME_TYPE`, `CODE_TYPE`,
            `SUBSTANCE_CLASS`, `POLYMER_CLASS`.

    Example:
        domain="NAME_TYPE"

    Returns:
        Markdown table mapping each term `value` to its display label.
    """
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_cv_terms", query_type="cv")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"GSRS API is currently unavailable: {reason}"

        # Delegate to the canonical resource implementation (single source of truth)
        # and render its JSON payload as the tool's markdown surface.
        result = await gsrs_cv_domain_terms_resource(domain)
        payload = json.loads(result)
        if payload.get("error"):
            tool.finish("degraded", result_count=0, citation_count=0, error_message=str(payload["error"]))
            return f"GSRS API is currently unavailable: {payload['error']}"
        terms = payload.get("terms", [])
        tool.finish("success", result_count=len(terms), citation_count=0)

        if not terms:
            return f"No terms found for CV domain **{domain}**."

        lines = [f"Terms for CV domain **{domain}** (`{payload.get('domain', domain)}`):\n"]
        lines.append("| Value | Display |")
        lines.append("|---|---|")
        for term in terms:
            value = term.get("value") or ""
            display = term.get("display") or value
            lines.append(f"| `{value}` | {display} |")
        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS get CV terms error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def gsrs_get_schema(model: str = "Substance") -> str:
    """Return the Pydantic JSON Schema for a GSRS substance model.

    This tool exposes the JSON Schema of the vendored
    ``gsrs.model`` Pydantic classes (``Substance``,
    ``ChemicalSubstance``, ``ProteinSubstance``, etc.) so the model
    can use them as a reference for:

    - **Indexed field names** — GSRS uses Lucene-indexed
      ``root_<field>`` paths whose names mirror the Pydantic
      attributes; consult the schema before building
      ``gsrs_parametric_search`` ``filters`` arguments.
    - **Element paths** — ``gsrs_get_substance_details`` walks the
      substance JSON via ``/``-separated element paths. Knowing
      the model attributes (and their nested ``$defs``) lets the
      model build accurate paths like ``names``,
      ``relationships/relatedSubstance``, or
      ``protein/subunits``.

    Args:
        model: One of the supported GSRS Pydantic model class
            names. Default ``"Substance"`` (the polymorphic base).
            Other valid options: ``"ChemicalSubstance"``,
            ``"ProteinSubstance"``, ``"NucleicAcidSubstance"``,
            ``"MixtureSubstance"``, ``"PolymerSubstance"``,
            ``"StructurallyDiverseSubstance"``.

    Returns:
        A JSON document string with the model's JSON Schema. The
        top-level ``properties`` keys are the substance fields;
        nested models appear under ``$defs``. The document is the
        exact output of ``Pydantic.model_json_schema()``.

    Example:
        gsrs_get_schema("ChemicalSubstance")
    """
    _ensure_runtime_initialized()
    tool = _tool_call("gsrs_get_schema", query_type="schema")
    try:
        # Delegate to the canonical resource implementation (single
        # source of truth) so the tool output exactly matches the
        # ``gsrs://schema/{model}`` resource payload.
        result = await gsrs_schema_resource(model)
        try:
            parsed = json.loads(result)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error"):
            error = str(parsed["error"])
            # Distinguish user errors (unknown model) from server
            # errors (gsrs-model not installed) so the tool finish
            # telemetry is accurate.
            if "Unknown GSRS model" in error or "model_not_found" in error:
                tool.finish("abstained", result_count=0, citation_count=0)
            else:
                tool.finish("degraded", result_count=0, citation_count=0, error_message=error)
        else:
            tool.finish("success", result_count=1, citation_count=0)
        return result
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS get_schema error: {exc}"


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_structure_search(
    structure: str,
    search_type: Literal["exact", "exactplus", "sim", "substructure", "flex", "flexplus"] = "exact",
    cutoff: float = 0.8,
    size: int = 20,
) -> str:
    """Search substances by chemical structure via the GSRS API.

    Provide a chemical structure as SMILES or InChI and choose a search type.

    Args:
        structure: Chemical structure string (SMILES or InChI).
        search_type:
            - exact: identical structure, tautomer included.
            - exactplus: exact match plus related salts/solvates/tautomers.
            - sim: global fingerprint similarity; uses `cutoff` (recommended 0.8).
            - substructure: query is contained within the target structure.
            - flex: ignores stereochemistry, isotope number, salts/solvates,
              hydrates, tautomers, and mixtures.
            - flexplus: searches by moiety only, ignoring stereochemistry.
        cutoff: Tanimoto-Jaccard similarity cutoff for sim searches (0.0-1.0).
            Ignored for other search types.
        size: Max results (max 100).

    Example:
        structure="CC(=O)Oc1ccccc1C(=O)O"
        search_type="sim"
        cutoff=0.8
    """
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


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def gsrs_sequence_search(
    sequence: str,
    search_type: Literal["GLOBAL", "SUB"] = "GLOBAL",
    sequence_type: Literal["protein", "nucleicAcid"] = "protein",
    cutoff: float = 0.95,
    size: int = 20,
) -> str:
    """Search substances by protein or nucleic-acid sequence via the GSRS API.

    Provide an amino-acid or nucleotide sequence and choose the alignment mode.

    Args:
        sequence: Amino-acid or nucleotide sequence. For proteins use one-letter
            codes. Spaces, dashes, and numbers are cleaned automatically by GSRS.
        search_type:
            - GLOBAL: global alignment match; finds sequences similar to the
              complete query (useful for full proteins/peptides/oligonucleotides).
            - SUB: contains alignment match; finds the query motif within larger
              sequences.
        sequence_type: protein or nucleicAcid.
        cutoff: Similarity/identity cutoff (0.0-1.0). A higher value requires a
            closer match. Recommended: 0.98 for proteins, lower for short motifs.
        size: Max results (max 100).

    Example:
        sequence="ACDEFGHIKLMNPQRSTVWY"
        search_type="GLOBAL"
        sequence_type="protein"
        cutoff=0.98
    """
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


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def health() -> str:
    """Return structured runtime health and readiness information."""
    _ensure_runtime_initialized()
    tool = _tool_call("health", query_type="runtime")
    # Delegate to the canonical resource implementation (single source of truth).
    result = await server_health_resource()
    tool.finish("success", result_count=0, citation_count=0)
    return result


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def statistics() -> str:
    """Return local vector store statistics."""
    _ensure_runtime_initialized()
    tool = _tool_call("statistics", query_type="statistics")
    try:
        if not runtime.vector_backend_available():
            reason = runtime.vector_backend_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_message=reason)
            return f"Statistics are currently unavailable: {reason}"
        # Delegate to the canonical resource implementation (single source of truth).
        result = await server_statistics_resource()
        payload = json.loads(result)
        if isinstance(payload, dict) and payload.get("error"):
            tool.finish("degraded", result_count=0, citation_count=0, error_message=str(payload["error"]))
        else:
            tool.finish("success", result_count=0, citation_count=0)
        return result
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Statistics error: {exc}"


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport=settings.mcp_transport, mount_path="/")


# ------------------------------------------------------------------
# MCP Resources: stable, idempotent, identifier-based lookups
# ------------------------------------------------------------------

@mcp.resource("gsrs://substances/{identifier}", mime_type="application/json")
async def gsrs_substance_resource(identifier: str) -> str:
    """Return a complete GSRS substance JSON document by UUID or approval identifier."""
    _ensure_runtime_initialized()
    if not runtime.gsrs_api_available():
        return json.dumps({"error": runtime.gsrs_api_unavailable_reason()})
    try:
        substance = runtime.gsrs_api.get_substance(identifier)
        if substance is None:
            return json.dumps({"error": f"Substance {identifier} not found."})
        return json.dumps(substance, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("gsrs://substances/{identifier}/summary", mime_type="text/markdown")
async def gsrs_substance_summary_resource(identifier: str) -> str:
    """Return a markdown summary for a GSRS substance by UUID or approval identifier."""
    _ensure_runtime_initialized()
    if not runtime.gsrs_api_available():
        return f"GSRS API is currently unavailable: {runtime.gsrs_api_unavailable_reason()}"
    try:
        substance = runtime.gsrs_api.get_substance(identifier)
        if substance is None:
            return f"Substance **{identifier}** not found in GSRS API."
        return substance_to_markdown(substance)
    except Exception as exc:
        return f"GSRS summary error: {exc}"


def _render_parent_summary(
    *,
    substance_uuid: str,
    root_section: str,
    max_chars: int,
    include_text_parts: bool,
    exclude_chunk_id: str,
) -> tuple[str, str]:
    """Canonical implementation shared by ``get_parent_summary`` and the
    ``gsrs://substances/{identifier}/parents/{root_section}/summary`` resource.

    Returns a ``(body, error)`` tuple. ``body`` is the rendered
    markdown (empty string when the parent has no chunks).
    ``error`` is a non-empty user-facing error message on
    validation failures; on success it is the empty string.

    Both the tool and the resource delegate to this function so
    they share a single code path — the "canonical resource"
    pattern used elsewhere in this module (e.g.
    ``gsrs_get_substance`` → ``gsrs_substance_resource``).
    """
    uuid_text = (substance_uuid or "").strip()
    if not uuid_text:
        return "", "A substance_uuid is required."
    try:
        document_id = UUID(uuid_text)
    except (TypeError, ValueError):
        return "", f"Invalid substance_uuid: {substance_uuid!r} is not a valid UUID."

    section_text = (root_section or "").strip()
    if not section_text:
        return "", "A root_section is required."

    exclude_ids: set[str] = set()
    if exclude_chunk_id:
        exclude_ids.add(str(exclude_chunk_id).strip())

    try:
        from app.services.parent_child_retrieval import ParentIdentity

        identity = ParentIdentity(document_id=document_id, root_section=section_text)
        body = runtime.parent_enricher.summarize_parent(
            identity,
            max_chars=max(1, int(max_chars)),
            include_text_parts=bool(include_text_parts),
            exclude_chunk_ids=exclude_ids,
        )
    except Exception as exc:
        return "", f"Parent summary error: {exc}"
    return body or "", ""


@mcp.resource("gsrs://substances/{identifier}/parents/{root_section}/summary", mime_type="text/markdown")
async def gsrs_parent_summary_resource(identifier: str, root_section: str) -> str:
    """Return a chunk-driven markdown summary for one parent.

    Unlike :func:`gsrs_substance_summary_resource`, this surface
    does **not** call the GSRS upstream. It is reconstructed from
    the chunks in the local RAG store and is a **parent-scoped**
    view: it covers the chunks sharing ``(identifier, root_section)``.

    Use this when the GSRS API is unreachable but the local RAG
    store has been pre-ingested. For a full substance view (which
    needs approval ID, status, and substance-level access) use
    ``gsrs://substances/{identifier}/summary`` instead.

    Args:
        identifier: Substance UUID. Must be a valid UUID string —
            the resource does not go through GSRS, so approval IDs
            and display names are not resolved here.
        root_section: Root section identifier (e.g. ``"overview"``,
            ``"names"``, ``"codes"``, ``"definitions"``,
            ``"relationships"``, ``"properties"``,
            ``"references"``). These match the values returned by
            the ``get_parent_context`` tool.

    Returns:
        Markdown string. Empty when the parent has no chunks.
        Validation errors are returned as inline markdown starting
        with a ``Parent summary error:`` line.
    """
    _ensure_runtime_initialized()
    body, error = _render_parent_summary(
        substance_uuid=identifier,
        root_section=root_section,
        max_chars=4000,
        include_text_parts=True,
        exclude_chunk_id="",
    )
    if error:
        return error
    if not body:
        return (
            f"No parent context found for substance **{identifier}** "
            f"(root_section: `{root_section}`)."
        )
    return body


@mcp.resource("gsrs://substances/{identifier}/details/{filter}", mime_type="application/json")
async def gsrs_substance_details_resource(identifier: str, filter: str = "") -> str:
    """Return a filtered view of a GSRS substance record.

    Implements the GSRS "details" API section
    (https://gsrs.ncats.nih.gov/api-documentation). The substance
    is resolved by UUID, approval ID, or display name, then a
    list of filter steps is joined with ``!`` and appended to the
    path ``/api/v1/substances({uuid})/{joined}`` to narrow the
    response.

    The filter contract is a **list of steps** (matching
    :func:`gsrs_get_substance_details`). The URI template's
    ``{filter}`` segment is a single pre-joined string for
    transport; this function splits it back into steps, joins
    them again, and validates the result so URI callers and
    in-process callers share the same code path. In-process
    callers (the :func:`gsrs_get_substance_details` tool) pass
    the raw ``filter`` string the same way and get the same
    treatment — there is no separate "joined" form.

    Args:
        identifier: GSRS substance UUID, approval ID, or display
            name. Names are resolved to a UUID via a name search.
        filter: Pre-joined details-filter expression
            (``!``-delimited). May be empty. See
            :func:`gsrs_get_substance_details` for the grammar.
    """
    _ensure_runtime_initialized()
    if not runtime.gsrs_api_available():
        return json.dumps({"error": runtime.gsrs_api_unavailable_reason()})

    identifier_text = identifier.strip() if identifier else ""
    if not identifier_text:
        return json.dumps({"error": "A substance_identifier is required."})

    # The contract is a list of steps. The URI template's
    # ``{filter}`` is a pre-joined string; split it back into a
    # list so joining/validation happens exactly once, here.
    filter_steps = _split_filter_text(filter)

    try:
        joined_filter = runtime.gsrs_api.join_filter_steps(filter_steps)
        if joined_filter:
            validate_filter(joined_filter)
    except FilterError as exc:
        return json.dumps({"error": f"Invalid filter: {exc}"})

    try:
        payload = runtime.gsrs_api.get_substance_details_by_identifier(
            identifier_text, joined_filter or None
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    if payload is None:
        return json.dumps(
            {
                "error": (
                    f"Could not resolve substance identifier {identifier_text!r}: "
                    "no matching record found in the GSRS API."
                )
            }
        )

    resolved = payload.get("resolved") or {}
    result = payload.get("result")
    return json.dumps(
        {
            "resolved": resolved,
            "filter": joined_filter or None,
            "filter_steps": filter_steps or None,
            "result": result,
        },
        indent=2,
        default=str,
    )


def _split_filter_text(filter_text: str | None) -> list[str]:
    """Split a pre-joined details filter string into its ``!``-delimited steps.

    The resource's URI template transports the filter as a single
    pre-joined string (e.g. ``"names(type:of)!(name)!limit(1)"``).
    The canonical contract is a list of steps, so we split the
    string back into its component steps here. ``!`` is a
    delimiter between steps and does not appear inside any valid
    step body, so a simple split is sufficient (and round-trips
    with :meth:`GsrsApiService.join_filter_steps`).

    An empty/None input returns an empty list — equivalent to
    "no filter, return the full substance record".
    """
    if not filter_text:
        return []
    text = filter_text.strip()
    if not text:
        return []
    return [step.strip() for step in text.split("!") if step.strip()]


@mcp.resource("gsrs://schema/{model}", mime_type="application/json")
async def gsrs_schema_resource(model: str) -> str:
    """Return the Pydantic JSON Schema for a GSRS substance model.

    Implements the JSON-Schema side of the contract exposed by
    :func:`gsrs_get_schema`. The schema is sourced from the
    vendored ``gsrs.model`` Pydantic classes and is intended as a
    *reference* for the LLM-driven tools:

    - **Indexed field names** — GSRS uses Lucene-indexed
      ``root_<field>`` paths whose names mirror the Pydantic
      attributes; consult the schema before building
      ``gsrs_parametric_search`` ``filters`` arguments.
    - **Element paths** — ``gsrs_get_substance_details`` walks the
      substance JSON via ``/``-separated element paths. Knowing
      the model attributes (and their nested ``$defs``) lets the
      model build accurate paths like ``names``,
      ``relationships/relatedSubstance``, or
      ``protein/subunits``.

    Args:
        model: One of the supported GSRS Pydantic model class
            names (``Substance``, ``ChemicalSubstance``,
            ``ProteinSubstance``, ``NucleicAcidSubstance``,
            ``MixtureSubstance``, ``PolymerSubstance``,
            ``StructurallyDiverseSubstance``).

    Returns:
        A JSON document string with the model's JSON Schema. The
        top-level ``properties`` keys are the substance fields;
        nested models appear under ``$defs``. The document is the
        exact output of ``Pydantic.model_json_schema()``.

    On error, returns a JSON document of the form
    ``{"error": "...", "code": "model_not_found|dependency_missing|model_unavailable"}``
    so callers can distinguish user errors (unknown model) from
    server errors (gsrs-model not installed).
    """
    model_name = (model or "").strip() or "Substance"
    try:
        schema = get_model_schema(model_name)
    except GsrsSchemaError as exc:
        return json.dumps({"error": exc.message, "code": exc.code}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc), "code": "internal_error"}, indent=2)
    # Always include the requested model name so URI callers can
    # tell the response apart from a generic blob.
    return json.dumps(
        {"model": model_name, "schema": schema},
        indent=2,
        default=str,
    )


@mcp.resource("gsrs://cv/domains", mime_type="application/json")
async def gsrs_cv_domains_resource() -> str:
    """List available GSRS controlled vocabulary (CV) domains as JSON."""
    _ensure_runtime_initialized()
    if not runtime.gsrs_api_available():
        return json.dumps({"error": runtime.gsrs_api_unavailable_reason()})
    try:
        payload = runtime.gsrs_api.get_cv_domains(size=200)
        domains = payload.get("content") or payload.get("results") or []
        return json.dumps({
            "total": payload.get("total", len(domains)),
            "count": len(domains),
            "domains": [
                {
                    "domain": entry.get("domain") or entry.get("value"),
                    "display": entry.get("display") or entry.get("domain") or entry.get("value"),
                }
                for entry in domains
                if isinstance(entry, dict)
            ],
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("gsrs://cv/{domain}/terms", mime_type="application/json")
async def gsrs_cv_domain_terms_resource(domain: str) -> str:
    """Return the terms for a GSRS controlled vocabulary domain as JSON."""
    _ensure_runtime_initialized()
    if not runtime.gsrs_api_available():
        return json.dumps({"error": runtime.gsrs_api_unavailable_reason()})
    try:
        payload = runtime.gsrs_api.get_cv_terms(domain)
        terms = payload.get("terms") or []
        return json.dumps({
            "domain": payload.get("domain", domain),
            "count": len(terms),
            "terms": [
                {
                    "value": term.get("value"),
                    "display": term.get("display") or term.get("value"),
                    "deprecated": term.get("deprecated", False),
                    "selected": term.get("selected", False),
                }
                for term in terms
                if isinstance(term, dict)
            ],
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("server://health", mime_type="application/json")
async def server_health_resource() -> str:
    """Return structured runtime health and readiness information."""
    _ensure_runtime_initialized()
    payload = runtime.get_status_payload()
    payload["live"] = True
    return json.dumps(payload, indent=2, default=str)


@mcp.resource("server://statistics", mime_type="application/json")
async def server_statistics_resource() -> str:
    """Return local vector store statistics."""
    _ensure_runtime_initialized()
    try:
        if not runtime.vector_backend_available():
            return json.dumps({"error": runtime.vector_backend_unavailable_reason()})
        payload = runtime.vector_db.get_statistics()
        return json.dumps(payload, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------
# MCP Prompts: reusable guidance for common GSRS workflows
# ------------------------------------------------------------------

@mcp.prompt(name="fetch_substance", title="Fetch a GSRS substance record", description="Prompt template for fetching the complete raw GSRS substance JSON document by UUID or approval identifier.")
async def fetch_substance_prompt(identifier: str) -> Prompt:
    """Return a prompt that asks the LLM to fetch a raw GSRS substance record."""
    messages = [
        {
            "role": "user",
            "content": TextContent(
                type="text",
                text=(
                    f"Fetch the complete GSRS substance record for identifier `{identifier}` "
                    "using the `gsrs_get_substance` tool (or the resource "
                    "`gsrs://substances/{identifier}`). Return the raw JSON document "
                    "verbatim and identify the substance's UUID, approval ID, substance "
                    "class, and status. Do not paraphrase or omit fields — the full "
                    "payload is required for downstream processing."
                ),
            ),
        }
    ]
    return Prompt(name="fetch_substance", title="Fetch a GSRS substance record", messages=messages)


@mcp.prompt(name="substance_summary", title="Summarize a GSRS substance", description="Prompt template for fetching and summarizing a single GSRS substance record.")
async def substance_summary_prompt(identifier: str) -> Prompt:
    """Return a prompt that asks the LLM to summarize a GSRS substance."""
    messages = [
        {
            "role": "user",
            "content": TextContent(
                type="text",
                text=(
                    f"Fetch the GSRS substance record for identifier `{identifier}` "
                    "using the `gsrs_get_summary` tool (or the resource "
                    "`gsrs://substances/{identifier}/summary`), and produce a concise "
                    "human-readable summary. Include the preferred display name, "
                    "approval ID, substance class, status, key identifiers, names, "
                    "and any structural or biological details."
                ),
            ),
        }
    ]
    return Prompt(name="substance_summary", title="Summarize a GSRS substance", messages=messages)


@mcp.prompt(name="resolve_cv_terms", title="Resolve GSRS controlled vocabulary terms", description="Prompt template for decoding a GSRS controlled-vocabulary code using the CV resources.")
async def resolve_cv_terms_prompt(domain: str, code: str) -> Prompt:
    """Return a prompt that resolves a single CV code to its display label."""
    messages = [
        {
            "role": "user",
            "content": TextContent(
                type="text",
                text=(
                    f"Look up the controlled vocabulary domain `{domain}` and resolve "
                    f"the code `{code}` to its human-readable display value. Use the "
                    f"resource `gsrs://cv/{domain}/terms` (or the tool "
                    f"`gsrs_get_cv_terms` with domain `{domain}`). Explain what the "
                    f"code means in the context of a GSRS substance record."
                ),
            ),
        }
    ]
    return Prompt(name="resolve_cv_terms", title="Resolve GSRS controlled vocabulary terms", messages=messages)


@mcp.prompt(name="rag_reasoning", title="Reason over local GSRS RAG results", description="Prompt template for grounded question answering using the local RAG store.")
async def rag_reasoning_prompt(question: str) -> Prompt:
    """Return a prompt that guides the LLM to answer from local RAG evidence."""
    messages = [
        {
            "role": "user",
            "content": TextContent(
                type="text",
                text=(
                    f"Answer the following question using evidence from the local GSRS "
                    f"RAG store: {question}\n\n"
                    "Instructions:\n"
                    "1. Use `rag_query` to retrieve the most relevant chunks with parent "
                    "context.\n"
                    "2. If results are sparse, also try `rag_query_chunks`.\n"
                    "3. Cite the substance UUID and section for each piece of evidence.\n"
                    "4. If the local store does not contain relevant data, say so and "
                    "do not invent facts.\n"
                    "5. Keep the answer concise and grounded in the retrieved text."
                ),
            ),
        }
    ]
    return Prompt(name="rag_reasoning", title="Reason over local GSRS RAG results", messages=messages)
