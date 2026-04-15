"""
GSRS MCP Server — MCP Server
Exposes GSRS substance search, Q&A, similarity search, and management
via the Model Context Protocol (MCP) using streamable-http transport.
"""
import atexit
import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import AnyHttpUrl, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.models.api import (
    AskRequest,
    ERIQueryRequest,
    ERIQueryResponse,
    ERIResult,
    QueryResult,
    SimilarSubstanceResult,
)
from app.observability import ToolTelemetry, configure_logging
from app.runtime import ServerRuntime
from app.services.aggregation import AggregationService
from app.services.metadata_filters import MetadataFilterBuilder
from app.services.query_rewrite import QueryRewriteService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
configure_logging(
    settings.debug_mode,
    use_stderr=os.getenv("MCP_TRANSPORT", "streamable-http").lower() == "stdio",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bearer Token Authentication
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core services
# ---------------------------------------------------------------------------
runtime = ServerRuntime(settings)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def server_lifespan(server):
    """Initialise shared runtime services on startup."""
    _ensure_runtime_initialized()
    yield


def _log_runtime_status() -> None:
    """Emit a consistent startup log snapshot after runtime initialization."""
    status_payload = runtime.get_status_payload()
    readiness_summary = getattr(runtime, "readiness_summary", status_payload.get("readiness_summary"))
    required_component_errors = (
        runtime.required_component_errors()
        if hasattr(runtime, "required_component_errors")
        else status_payload.get("required_component_errors", {})
    )
    degraded_summary = getattr(runtime, "degraded_summary", status_payload.get("degraded_summary"))
    optional_component_errors = (
        runtime.optional_component_errors()
        if hasattr(runtime, "optional_component_errors")
        else status_payload.get("optional_component_errors", {})
    )
    logger.info(
        "runtime_initialized",
        extra=status_payload,
    )
    if not runtime.ready:
        logger.warning(
            "runtime_not_ready",
            extra={
                "backend": runtime.backend_name,
                "readiness_summary": readiness_summary,
                "required_component_errors": required_component_errors,
            },
        )
    elif runtime.degraded:
        logger.warning(
            "runtime_degraded",
            extra={
                "backend": runtime.backend_name,
                "degraded_summary": degraded_summary,
                "optional_component_errors": optional_component_errors,
            },
        )


def _ensure_runtime_initialized() -> None:
    """Initialize the shared runtime once, even when health routes are hit before MCP sessions."""
    if getattr(runtime, "initialized", False):
        return
    runtime.initialize()
    _log_runtime_status()


def _shutdown_runtime() -> None:
    """Best-effort process shutdown for long-lived runtime clients."""
    if getattr(runtime, "initialized", False):
        runtime.shutdown()


atexit.register(_shutdown_runtime)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
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
        "GSRS (Global Substance Registration System) MCP server. "
        "The PRIMARY tool for all substance queries is **gsrs_ask** — use it FIRST for any "
        "substance-related question. It answers natural-language queries with AI-synthesized "
        "results and citations. Other tools (gsrs_substance_search, gsrs_similarity_search, etc.) "
        "are for specific technical needs only."
    ),
    token_verifier=token_verifier,
    auth=auth,
    host=settings.mcp_api,
    port=settings.mcp_port,
    streamable_http_path="/mcp",
    lifespan=server_lifespan,
)


# ---------------------------------------------------------------------------
# Health checks (custom routes, no auth required)
# ---------------------------------------------------------------------------

@mcp.custom_route("/livez", methods=["GET"], include_in_schema=True)
async def live_check(request: Request) -> JSONResponse:
    """Liveness probe: process is up if this route responds."""
    return JSONResponse({"status": "alive"})


@mcp.custom_route("/readyz", methods=["GET"], include_in_schema=True)
async def readiness_check(request: Request) -> JSONResponse:
    """Readiness probe: dependencies and runtime state are ready for retrieval."""
    _ensure_runtime_initialized()
    payload = runtime.get_status_payload()
    status_code = 200 if payload["ready"] else 503
    return JSONResponse(payload, status_code=status_code)


@mcp.custom_route("/health", methods=["GET"], include_in_schema=True)
async def health_check(request: Request) -> JSONResponse:
    """Combined health endpoint with liveness, readiness, and dependency state."""
    _ensure_runtime_initialized()
    payload = runtime.get_status_payload()
    payload["live"] = True
    return JSONResponse(payload)


@mcp.custom_route("/eri/query", methods=["POST"], include_in_schema=True)
async def eri_query(request: Request) -> JSONResponse:
    """Legacy ERI retrieval route kept for older Open WebUI tools."""
    _ensure_runtime_initialized()
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"detail": "Invalid JSON request body."}, status_code=400)

    try:
        eri_request = ERIQueryRequest.model_validate(payload)
    except ValidationError as exc:
        return JSONResponse(
            {
                "detail": "Invalid ERI query payload.",
                "errors": exc.errors(),
            },
            status_code=422,
        )

    try:
        results, retrieval_mode, _ = _retrieve_query_results(
            query=eri_request.query,
            top_k=eri_request.top_k,
            filters=eri_request.filters,
        )
    except RuntimeError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)
    except Exception as exc:
        logger.exception("eri_query_failed")
        return JSONResponse({"detail": f"Retrieval error: {exc}"}, status_code=500)

    response = ERIQueryResponse(
        results=[ERIResult(chunk=result.document, score=result.score) for result in results]
    )
    return JSONResponse(
        response.model_dump(),
        headers={"X-GSRS-Retrieval-Mode": retrieval_mode},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_search_criteria(substance: Dict[str, Any]) -> Dict[str, Any]:
    """Extract prioritised searchable metadata from a GSRS JSON document."""
    criteria: Dict[str, Any] = {}
    if "uuid" in substance:
        criteria["uuid"] = str(substance["uuid"])
    if "approvalID" in substance:
        criteria["approvalID"] = str(substance["approvalID"])
    if "codes" in substance and isinstance(substance["codes"], list):
        reliable_systems = set(settings.similarity_reliable_codes)
        reliable_codes: Dict[str, str] = {}
        all_codes: Dict[str, str] = {}
        for entry in substance["codes"]:
            if isinstance(entry, dict):
                sys_ = entry.get("codeSystem", entry.get("type", ""))
                val = entry.get("code", "")
                if sys_ and val:
                    all_codes[sys_] = val
                    if sys_ in reliable_systems:
                        reliable_codes[sys_] = val
        if reliable_codes:
            criteria["reliable_codes"] = reliable_codes
        if all_codes:
            criteria["all_codes"] = all_codes
    if "structure" in substance and isinstance(substance["structure"], dict):
        sd: Dict[str, str] = {}
        for k in ["stereochemistry", "molecularFormula", "molecularWeight",
                   "smiles", "inchi", "inchikey", "sequence"]:
            v = substance["structure"].get(k)
            if v:
                sd[k] = str(v)
        if sd:
            criteria["structure"] = sd
    if "names" in substance and isinstance(substance["names"], list):
        sys_names, off_names, oth = [], [], []
        for n in substance["names"]:
            val = n.get("name", "") if isinstance(n, dict) else n
            if not val:
                continue
            t = n.get("type", "").lower() if isinstance(n, dict) else ""
            if "systematic" in t:
                sys_names.append(val)
            elif any(x in t for x in ["official", "common", "preferred"]):
                off_names.append(val)
            else:
                oth.append(val)
        all_n = sys_names + off_names + oth
        if sys_names:
            criteria["systematic_names"] = sys_names
        if off_names:
            criteria["official_names"] = off_names
        if oth:
            criteria["other_names"] = oth
        if all_n:
            criteria["canonical_name"] = all_n[0]
    if "classifications" in substance and isinstance(substance["classifications"], list):
        cls_list = []
        for c in substance["classifications"]:
            name = c.get("name", c.get("className", "")) if isinstance(c, dict) else c
            if name:
                cls_list.append(name)
        if cls_list:
            criteria["classifications"] = cls_list
    return criteria


def _group_by_substance(results, exclude_self: bool = True,
                        self_uuid: Optional[str] = None):
    """Group search results by substance UUID, return SimilarSubstanceResult list."""
    groups: Dict[str, List] = defaultdict(list)
    for item in results:
        if hasattr(item, "document") and hasattr(item, "score"):
            doc, score = item.document, item.score
        else:
            doc, score = item
        uid = str(doc.document_id)
        if exclude_self and self_uuid and uid == self_uuid:
            continue
        groups[uid].append((doc, score))

    out = []
    for uid, chunks in groups.items():
        best = max(s for _, s in chunks)
        matched: set = set()
        cname = None
        for d, _ in chunks:
            m = d.metadata_json or {}
            if "canonical_name" in m and not cname:
                cname = m["canonical_name"]
            matched.update(m.keys())
        cr = [QueryResult(chunk=d, score=s) for d, s in chunks]
        out.append(SimilarSubstanceResult(
            substance_uuid=uid, canonical_name=cname,
            match_score=best, matched_fields=sorted(matched), chunks=cr,
        ))
    out.sort(key=lambda r: r.match_score, reverse=True)
    return out


def _is_gsrs_substance(data: Dict[str, Any]) -> bool:
    keys = {"uuid", "names", "codes", "relationships",
            "properties", "references", "notes"}
    return len(keys.intersection(data.keys())) >= 2


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse text as JSON. Returns dict if successful, None otherwise."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _tool_call(tool_name: str, *, query_type: Optional[str] = None) -> ToolTelemetry:
    """Create a telemetry context for a tool call."""
    return ToolTelemetry.start(
        logger=logger,
        metrics=runtime.metrics,
        tool_name=tool_name,
        backend=runtime.backend_name,
        query_type=query_type,
    )


def _query_type_from_retrieval_mode(
    retrieval_mode: Optional[str],
    *,
    default: str = "question",
) -> str:
    """Map retrieval routing decisions to a stable telemetry/debug query type."""
    if retrieval_mode and retrieval_mode.startswith("identifier-first:"):
        return retrieval_mode.split(":", 1)[1]
    return default


def _runtime_debug_state() -> Dict[str, Any]:
    """Expose lightweight degraded-state context for local diagnostics."""
    return {
        "ready": runtime.ready,
        "degraded": runtime.degraded,
        "backend": runtime.backend_name,
        "degraded_summary": runtime.degraded_summary,
    }


def _retrieve_query_results(
    query: str,
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
):
    """Run the raw retrieval path shared by MCP and legacy ERI interfaces."""
    if not runtime.retrieval_available():
        raise RuntimeError(f"Retrieval is currently unavailable: {runtime.retrieval_unavailable_reason()}")

    diagnostics: Dict[str, Any] = {
        "canonical_query": query.lower().strip(),
        "rewrites": [query],
        "intent": "semantic",
        "applied_filters": filters,
        "candidate_count": None,
        "reranked_count": None,
    }

    pipeline = runtime.query_pipeline
    required_pipeline_parts = (
        "rewrite_service",
        "filter_builder",
        "identifier_router",
        "hybrid_retriever",
        "reranker",
    )
    if pipeline and all(hasattr(pipeline, attr) for attr in required_pipeline_parts):
        rewrite_result = pipeline.rewrite_service.rewrite(query)
        applied_filters = pipeline.filter_builder.build(
            request_filters=filters,
            inferred_filters=rewrite_result.filters,
        )
        diagnostics.update(
            {
                "canonical_query": rewrite_result.canonical_query,
                "rewrites": [query] + rewrite_result.rewrites,
                "intent": rewrite_result.intent,
                "applied_filters": applied_filters,
            }
        )

        route_result = pipeline.identifier_router.route(query, top_k=top_k)
        if route_result is not None and route_result.results:
            diagnostics["candidate_count"] = len(route_result.results)
            diagnostics["reranked_count"] = len(route_result.results)
            return route_result.results[:top_k], f"identifier-first:{route_result.route}", diagnostics

        candidates = pipeline.hybrid_retriever.retrieve(
            queries=diagnostics["rewrites"],
            filters=applied_filters,
        )
        reranked = pipeline.reranker.rerank(
            candidates=candidates,
            query=query,
            rewritten_queries=rewrite_result.rewrites,
            filters=applied_filters,
        )
        diagnostics["candidate_count"] = len(candidates)
        diagnostics["reranked_count"] = len(reranked)
        return reranked[:top_k], "hybrid", diagnostics

    route_result = pipeline.identifier_router.route(query, top_k=top_k) if pipeline else None
    if route_result is not None:
        diagnostics["candidate_count"] = len(route_result.results)
        diagnostics["reranked_count"] = len(route_result.results)
        return route_result.results[:top_k], f"identifier-first:{route_result.route}", diagnostics

    embedding = runtime.embedding_service.embed(query)
    results = runtime.vector_db.similarity_search(embedding, top_k=top_k, filters=filters)
    diagnostics["candidate_count"] = len(results)
    diagnostics["reranked_count"] = len(results)
    return results, "semantic", diagnostics


def _format_ask_response(response) -> str:
    """Format AskResponse as a grounded, operator-friendly MCP string."""
    sections: List[str] = []

    if response.abstained:
        sections.append("Direct answer:\nInsufficient evidence to answer confidently.")
    else:
        sections.append(f"Direct answer:\n{response.answer or 'No answer available.'}")

    if response.degraded_reason:
        sections.append(f"Mode:\n{response.degraded_reason}")

    sections.append(f"Confidence:\n{response.confidence:.2f}")

    if response.abstain_reason:
        sections.append(f"Uncertainty:\n{response.abstain_reason}")

    if response.evidence_chunks:
        evidence_lines = []
        for i, chunk in enumerate(response.evidence_chunks[:5], 1):
            evidence_lines.append(
                f"[{i}] ({chunk.element_path}, {chunk.similarity_score:.2f}) {chunk.text[:180]}"
            )
        sections.append("Supporting evidence:\n" + "\n".join(evidence_lines))

    if response.citations:
        citation_lines = []
        for i, citation in enumerate(response.citations[:5], 1):
            suffix = f" - {citation.source_url}" if citation.source_url else ""
            quote = f' "{citation.quote}"' if citation.quote else ""
            citation_lines.append(f"[{i}] {citation.section} ({citation.chunk_id}){suffix}{quote}")
        sections.append("Citations:\n" + "\n".join(citation_lines))

    if response.debug:
        sections.append("Debug:\n" + json.dumps(response.debug, indent=2))

    return "\n\n".join(sections)


def _emit_pipeline_stages(tool: ToolTelemetry, diagnostics: Dict[str, Any]) -> None:
    """Emit structured stage logs for ask/retrieval style tools."""
    for stage in diagnostics.get("stages", []):
        tool.stage(
            stage_name=stage.get("stage", "unknown"),
            outcome=stage.get("outcome", "success"),
            **{key: value for key, value in stage.items() if key not in {"stage", "outcome"}},
        )


def _ingest_substance_payload(substance: Dict[str, Any]) -> tuple[str, int]:
    """Validate, chunk, embed, and upsert a GSRS substance payload."""
    from gsrs.model import Substance

    parsed_substance = Substance.model_validate(substance)
    if runtime.chunker is None:
        raise RuntimeError("Chunker is not initialized.")

    chunks = runtime.chunker.chunk(parsed_substance)
    if not chunks:
        return str(getattr(parsed_substance, "uuid", "unknown")), 0

    texts = [str(chunk.text) for chunk in chunks]
    embeddings = runtime.embedding_service.embed_batch(texts)
    documents = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk.set_embedding(embedding)
        documents.append(chunk)
    count = runtime.vector_db.upsert_documents(documents)
    uid = str(getattr(parsed_substance, "uuid", "unknown"))
    return uid, count


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def gsrs_ask(
    query: str,
    top_k: int = 10,
    answer_style: Literal["concise", "standard", "detailed"] = "standard",
    return_evidence: bool = True,
    min_confidence: float = 0.0,
    debug: bool = False,
) -> str:
    """PRIMARY tool for GSRS substance queries. Use this FIRST for any substance-related question.

    Full AI answering pipeline with citations and evidence. Answers natural-language
    questions about substances (names, codes, structures, classifications, relationships)
    by searching the GSRS database and synthesizing results.

    ⚠️ PRIORITY: Always prefer this tool over gsrs_substance_search or gsrs_similarity_search
    when the user asks a question about a substance. Only use the other tools for specific
    technical needs (e.g., raw search results without AI synthesis).

    Args:
        query: Natural-language question about a substance.
        top_k: Results to retrieve.
        answer_style: concise | standard | detailed.
        return_evidence: Include evidence chunks.
        min_confidence: Minimum confidence threshold.

    Returns:
        Answer with citations.
    """
    tool = _tool_call("gsrs_ask", query_type="question")
    try:
        parsed = _try_parse_json(query)
        if parsed and _is_gsrs_substance(parsed):
            tool.bind(query_type="substance_json")
            tool.finish("abstained", result_count=0, citation_count=0)
            return (
                "gsrs_ask only accepts natural-language questions. "
                "For GSRS substance JSON, call gsrs_similarity_search with the payload as "
                "`substance_json`."
            )

        if not runtime.retrieval_available():
            reason = runtime.retrieval_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"Retrieval is currently unavailable: {reason}"

        if runtime.query_pipeline is None:
            raise RuntimeError("Query pipeline is not initialized.")

        request = AskRequest(
            query=query,
            top_k=top_k,
            answer_style=answer_style,
            return_evidence=return_evidence,
            min_confidence=min_confidence,
            debug=debug or settings.debug_mode,
        )
        response, diagnostics = runtime.query_pipeline.ask_with_diagnostics(request)
        tool.bind(
            query_type=_query_type_from_retrieval_mode(
                diagnostics.get("retrieval_mode"),
                default="question",
            )
        )
        _emit_pipeline_stages(tool, diagnostics)
        if response.debug is not None:
            response.debug["request_id"] = tool.request_id
            response.debug["query_type"] = tool.context.get("query_type", "question")
            response.debug["runtime_state"] = _runtime_debug_state()
        tool.finish(
            "success" if not response.abstained else "abstained",
            result_count=len(response.evidence_chunks),
            citation_count=len(response.citations),
            retrieval_mode=diagnostics.get("retrieval_mode"),
            confidence=round(response.confidence, 4),
            degraded=response.degraded,
            answer_mode=(diagnostics.get("answer_generation") or {}).get("mode"),
            answer_error_type=(diagnostics.get("answer_generation") or {}).get("error_type"),
        )
        return _format_ask_response(response)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Error answering query: {exc}"


@mcp.tool()
async def gsrs_similarity_search(
    substance_json: str,
    top_k: int = 10,
    match_mode: str = "contains",
    exclude_self: bool = True,
) -> str:
    """Find substances similar to a GSRS JSON document.

    Extracts identifiers (UUID, codes, names, structure) and matches
    against stored metadata using priority-based scoring.

    Args:
        substance_json: GSRS JSON string.
        top_k: Maximum matches.
        match_mode: contains | match.
        exclude_self: Exclude exact UUID match.

    Returns:
        Ranked similar substances with match scores.
    """
    tool = _tool_call("gsrs_similarity_search", query_type="substance_json")
    try:
        if not runtime.metadata_lookup_available():
            reason = runtime.metadata_lookup_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"Similarity search is currently unavailable: {reason}"

        try:
            substance = json.loads(substance_json)
        except (json.JSONDecodeError, ValueError) as exc:
            return f"Error: invalid JSON - {exc}"

        example = _extract_search_criteria(substance)
        if not example:
            return "No searchable metadata found. Provide uuid, names, codes, or classifications."

        results = runtime.vector_db.search_by_example(
            example=example, top_k=top_k, mode=match_mode,
        )

        groups = _group_by_substance(results, exclude_self, example.get("uuid"))
        tool.finish("success", result_count=len(groups), citation_count=0)
        if not groups:
            return "No similar substances found."

        name = substance.get("names", [{}])[0].get("name", "substance")
        lines = [f"Found {len(groups)} substance(s) similar to **{name}**:\n"]
        for i, result in enumerate(groups, 1):
            substance_name = result.canonical_name or result.substance_uuid
            lines.append(f"{i}. **{substance_name}** (score {result.match_score:.2f})")
            lines.append(f"   UUID: {result.substance_uuid}")
            if result.matched_fields:
                lines.append(f"   Matched: {', '.join(result.matched_fields[:5])}")
            if result.chunks:
                lines.append(f"   Chunks: {len(result.chunks)} - {result.chunks[0].text[:120]}...")
        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Similarity search error: {exc}"


@mcp.tool()
async def gsrs_retrieve(
    query: str,
    top_k: int = 10,
    filters: Optional[str] = None,
    debug: bool = False,
) -> str:
    """Semantic chunk retrieval — raw results, no AI answer.

    Args:
        query: Search text.
        top_k: Number of chunks.
        filters: Optional JSON metadata filter string.

    Returns:
        Ranked text chunks with scores.
    """
    tool = _tool_call("gsrs_retrieve", query_type="semantic")
    try:
        parsed_filters = json.loads(filters) if filters else None
        try:
            results, retrieval_mode, diagnostics = _retrieve_query_results(
                query=query,
                top_k=top_k,
                filters=parsed_filters,
            )
        except RuntimeError as exc:
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=str(exc),
            )
            return str(exc)
        tool.bind(
            query_type=_query_type_from_retrieval_mode(
                retrieval_mode,
                default=str(diagnostics.get("intent") or "semantic"),
            )
        )

        tool.stage(
            "retrieval",
            retrieval_mode=retrieval_mode,
            result_count=len(results),
            candidate_count=diagnostics.get("candidate_count"),
            reranked_count=diagnostics.get("reranked_count"),
        )
        tool.finish("success", result_count=len(results), citation_count=0, retrieval_mode=retrieval_mode)
        if not results:
            return f"No results for '{query}'."

        lines = [f"Found {len(results)} result(s) for '{query}':\n"]
        for i, result in enumerate(results, 1):
            text = result.document.text[:250] + ("..." if len(result.document.text) > 250 else "")
            lines.append(f"{i}. {text}\n   Score: {result.score:.2f}  Section: {result.document.section}")
        if debug or settings.debug_mode:
            lines.append(
                "\nDebug: "
                + json.dumps(
                    {
                        "request_id": tool.request_id,
                        "query_type": tool.context.get("query_type", "semantic"),
                        "canonical_query": diagnostics.get("canonical_query"),
                        "rewrites": diagnostics.get("rewrites"),
                        "intent": diagnostics.get("intent"),
                        "retrieval_mode": retrieval_mode,
                        "filters": parsed_filters,
                        "applied_filters": diagnostics.get("applied_filters"),
                        "candidate_count": diagnostics.get("candidate_count"),
                        "reranked_count": diagnostics.get("reranked_count"),
                        "runtime_state": _runtime_debug_state(),
                        "results": [
                            {
                                "chunk_id": result.document.chunk_id,
                                "document_id": str(result.document.document_id),
                                "section": result.document.section,
                                "score": round(result.score, 4),
                            }
                            for result in results[:10]
                        ],
                    },
                    indent=2,
                )
            )
        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Retrieval error: {exc}"


@mcp.tool()
async def gsrs_ingest(substance_json: str) -> str:
    """Ingest a GSRS substance JSON into the database.

    Chunks the substance, generates embeddings, stores in vector DB.

    Args:
        substance_json: GSRS JSON string.

    Returns:
        Ingestion result.
    """
    tool = _tool_call("gsrs_ingest", query_type="substance_json")
    try:
        if not runtime.ingestion_available():
            reason = runtime.ingestion_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"Ingestion is currently unavailable: {reason}"

        try:
            substance = json.loads(substance_json)
        except (json.JSONDecodeError, ValueError) as exc:
            return f"Error: invalid JSON - {exc}"
        uid, count = _ingest_substance_payload(substance)
        if count == 0:
            return "No chunks generated from substance."
        tool.finish("success", result_count=count, citation_count=0)
        return f"Ingested **{uid}** - {count} chunks."
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Ingestion error: {exc}"


@mcp.tool()
async def gsrs_ingest_from_uuid(substance_uuid: str) -> str:
    """Fetch a GSRS substance by UUID from the upstream API and ingest it locally.

    Args:
        substance_uuid: GSRS substance UUID.

    Returns:
        Ingestion result.
    """
    tool = _tool_call("gsrs_ingest_from_uuid", query_type="uuid")
    try:
        if not runtime.ingestion_available():
            reason = runtime.ingestion_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"Ingestion is currently unavailable: {reason}"
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"GSRS upstream is currently unavailable: {reason}"

        substance = runtime.gsrs_api.get_substance_by_uuid(substance_uuid)
        if substance is None:
            tool.finish("abstained", result_count=0, citation_count=0)
            return f"Substance **{substance_uuid}** not found in GSRS API."

        uid, count = _ingest_substance_payload(substance)
        if count == 0:
            return f"No chunks generated from substance **{uid}**."
        tool.finish("success", result_count=count, citation_count=0)
        return f"Ingested **{uid}** - {count} chunks."
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Ingestion error: {exc}"


@mcp.tool()
async def gsrs_delete(substance_uuid: str) -> str:
    """Delete all chunks for a substance.

    Args:
        substance_uuid: Substance UUID.

    Returns:
        Deletion confirmation.
    """
    tool = _tool_call("gsrs_delete", query_type="uuid")
    try:
        if not runtime.vector_backend_available():
            reason = runtime.vector_backend_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"Deletion is currently unavailable: {reason}"
        count = runtime.vector_db.delete_documents_by_substance(UUID(substance_uuid))
        tool.finish("success", result_count=count, citation_count=0)
        return f"Deleted **{substance_uuid}** - {count} chunks removed."
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Deletion error: {exc}"


@mcp.tool()
async def gsrs_health() -> str:
    """Return structured runtime health and readiness information."""
    tool = _tool_call("gsrs_health", query_type="runtime")
    payload = runtime.get_status_payload()
    tool.finish("success", result_count=0, citation_count=0)
    return json.dumps(payload, indent=2)


@mcp.tool()
async def gsrs_statistics() -> str:
    """Return database statistics (chunk count, substance count)."""
    tool = _tool_call("gsrs_statistics", query_type="statistics")
    try:
        if not runtime.vector_backend_available():
            reason = runtime.vector_backend_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"Statistics are currently unavailable: {reason}"
        stats = runtime.vector_db.get_statistics()
        tool.finish("success", result_count=0, citation_count=0)
        return json.dumps(stats, indent=2)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Statistics error: {exc}"


# ---------------------------------------------------------------------------
# MCP Tools — GSRS API
# ---------------------------------------------------------------------------

@mcp.tool()
async def gsrs_aggregation(
    query: str,
    aggregation_type: Literal["count", "names", "identifiers", "relationships", "general"] = "count",
    top_k: int = 50,
) -> str:
    """Aggregated substance queries — counts, lists, summaries.

    Answers questions like "How many identifiers has Aspirin?",
    "List all names of Ibuprofen", "Count the relationships of X".

    Args:
        query: User question describing the aggregation.
        aggregation_type: Type of aggregation — count | names | identifiers | relationships | general.
        top_k: Number of candidate chunks to retrieve.

    Returns:
        Formatted aggregation result (counts, lists, summaries).
    """
    tool = _tool_call("gsrs_aggregation", query_type=aggregation_type)
    try:
        if not runtime.retrieval_available():
            reason = runtime.retrieval_unavailable_reason()
            tool.finish("degraded", result_count=0, citation_count=0, error_type="RuntimeUnavailable", error_message=reason)
            return f"Aggregation is currently unavailable: {reason}"

        rewriter = QueryRewriteService()
        aggregator = AggregationService()
        filter_builder = MetadataFilterBuilder()

        rewrite_result = rewriter.rewrite(query)
        tool.bind(query_type=rewrite_result.intent)
        queries = [rewrite_result.canonical_query] + rewrite_result.rewrites
        applied_filters = filter_builder.build(inferred_filters=rewrite_result.filters)

        all_candidates: Dict[str, Any] = {}
        for candidate_query in queries:
            embedding = runtime.embedding_service.embed(candidate_query)
            results = runtime.vector_db.similarity_search(embedding, top_k=top_k, filters=applied_filters)
            for result in results:
                chunk_id = result.document.chunk_id
                if chunk_id not in all_candidates:
                    all_candidates[chunk_id] = (result.document, result.score)

        if not all_candidates:
            tool.finish("abstained", result_count=0, citation_count=0)
            return f"No data found for aggregation query: **{query}**."

        sorted_candidates = sorted(all_candidates.values(), key=lambda item: item[1], reverse=True)[:top_k]
        result = aggregator.aggregate(sorted_candidates, query, intent=rewrite_result.intent)
        tool.finish("success", result_count=len(sorted_candidates), citation_count=0)

        if aggregation_type == "count":
            return f"**{result.substance_name}** has **{result.total_count}** {result.aggregation_type}."

        return result.raw_text_summary
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Aggregation error: {exc}"


@mcp.tool()
async def gsrs_query_optimizer(
    query: str,
    mode: Literal["rewrite", "translate", "optimize"] = "optimize",
    target_language: str = "en",
) -> str:
    """Rewrite, translate, or optimise a user query for better retrieval.

    Generates multiple query variants, optionally translates, and enriches
    with GSRS-specific keywords.

    Args:
        query: Original user question.
        mode: rewrite | translate | optimize.
        target_language: Target language code for translation (default: en).

    Returns:
        Optimised query variants.
    """
    tool = _tool_call("gsrs_query_optimizer", query_type=mode)
    try:
        rewriter = QueryRewriteService()
        rewrite_result = rewriter.rewrite(query)

        lines = [f"**Original:** {query}", f"**Intent:** {rewrite_result.intent}", f"**Canonical:** {rewrite_result.canonical_query}"]

        if rewrite_result.filters:
            lines.append(f"**Inferred filters:** {json.dumps(rewrite_result.filters, indent=2)}")

        if mode == "translate" and target_language != "en":
            if runtime.llm_service:
                translated = runtime.llm_service.complete_text(
                    system_prompt=f"Translate the following query to {target_language}. Return only the translation.",
                    user_prompt=query,
                    temperature=0.0,
                )
                lines.append(f"\n**Translated ({target_language}):** {translated}")
            else:
                lines.append("\n**Translation unavailable** (no LLM configured).")
        else:
            lines.append("\n**Query variants:**")
            for i, variant in enumerate(rewrite_result.rewrites, 1):
                lines.append(f"  {i}. {variant}")

        tool.finish("success", result_count=len(rewrite_result.rewrites), citation_count=0)
        return "\n".join(lines)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Query optimizer error: {exc}"


@mcp.tool()
async def gsrs_get_document(substance_uuid: str) -> str:
    """Fetch a complete GSRS substance JSON document by UUID from the official GSRS API.

    Args:
        substance_uuid: Substance UUID (e.g. "0103a288-6eb6-4ced-b13a-849cd7edf028").

    Returns:
        Full substance JSON or an error message.
    """
    tool = _tool_call("gsrs_get_document", query_type="uuid")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"GSRS upstream is currently unavailable: {reason}"
        doc = runtime.gsrs_api.get_substance_by_uuid(substance_uuid)
        tool.finish("success" if doc else "abstained", result_count=1 if doc else 0, citation_count=0)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Error fetching substance **{substance_uuid}**: {exc}"

    if doc is None:
        return f"Substance **{substance_uuid}** not found in GSRS API."

    return json.dumps(doc, indent=2)


@mcp.tool()
async def gsrs_api_substance_schema() -> str:
    """Return the JSON Schema for GSRS substance documents (from Substance.model_json_schema()).

    Use this to understand the structure of GSRS API search results and substance payloads
    for gsrs_ingest or gsrs_similarity_search.
    """
    tool = _tool_call("gsrs_api_substance_schema")
    try:
        from gsrs.model import Substance

        schema = Substance.model_json_schema()
        tool.finish("success", result_count=1, citation_count=0)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"Error generating substance schema: {exc}"

    return json.dumps(schema, indent=2)


@mcp.tool()
async def gsrs_api_search(
    query: str,
    page: int = 1,
    size: int = 20,
    fields: str = "",
) -> str:
    """Search substances via the official GSRS API text search endpoint.

    Searches names, codes, classifications, and other fields in the
    public GSRS database. See https://gsrs.ncats.nih.gov/api-documentation
    for "text search" examples.

    Args:
        query: Search text.
        page: Page number (1-based).
        size: Results per page.
        fields: Comma-separated field list to return (empty = all).

    Returns:
        Search results as formatted text with UUID, name, and substance class.
    """
    tool = _tool_call("gsrs_api_search", query_type="text")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"GSRS API search is currently unavailable: {reason}"
        resp = runtime.gsrs_api.text_search(query, page=page, size=size, fields=fields or None)
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS API search error: {exc}"

    results = resp.get("content", [])
    total = resp.get("total", 0)

    tool.finish("success" if results else "abstained", result_count=len(results), citation_count=0)
    if not results:
        return f"No results found for **{query}**."

    lines = [f"Found **{total}** result(s) for **{query}** (page {page}):\n"]
    for i, sub in enumerate(results, 1):
        uuid = sub.get("uuid", "?")
        names = sub.get("names", [])
        name = names[0].get("name", "?") if names else "?"
        sclass = sub.get("substanceClass", "")
        lines.append(f"{i}. **{name}** ({sclass})")
        lines.append(f"   UUID: `{uuid}`")

    return "\n".join(lines)


@mcp.tool()
async def gsrs_api_structure_search(
    smiles: str = "",
    inchi: str = "",
    search_type: Literal["EXACT", "SIMILAR", "SUBSTRUCTURE", "SUPERSTRUCTURE"] = "EXACT",
    size: int = 20,
) -> str:
    """Search substances by chemical structure via the official GSRS API.

    Supports exact, similar, substructure, and superstructure matching.
    See https://gsrs.ncats.nih.gov/api-documentation for "chemical search" examples.

    Args:
        smiles: SMILES string (e.g. "CC(=O)OC1=CC=CC=C1C(=O)O").
        inchi: InChI string (alternative to SMILES).
        search_type: EXACT | SIMILAR | SUBSTRUCTURE | SUPERSTRUCTURE.
        size: Maximum number of results.

    Returns:
        Matching substances with names, UUIDs, and similarity info.
    """
    if not smiles and not inchi:
        return "Error: provide either **smiles** or **inchi**."

    tool = _tool_call("gsrs_api_structure_search", query_type="structure")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"GSRS structure search is currently unavailable: {reason}"
        resp = runtime.gsrs_api.structure_search(
            smiles=smiles or None,
            inchi=inchi or None,
            search_type=search_type,
            size=size,
        )
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS structure search error: {exc}"

    results = resp.get("content", [])
    tool.finish("success" if results else "abstained", result_count=len(results), citation_count=0)
    if not results:
        return f"No structures found for the given {smiles or inchi}."

    lines = [f"Found **{len(results)}** structure match(es) ({search_type}):\n"]
    for i, sub in enumerate(results, 1):
        uuid = sub.get("uuid", "?")
        names = sub.get("names", [])
        name = names[0].get("name", "?") if names else "?"
        sclass = sub.get("substanceClass", "")
        lines.append(f"{i}. **{name}** ({sclass})")
        lines.append(f"   UUID: `{uuid}`")

    return "\n".join(lines)


@mcp.tool()
async def gsrs_api_sequence_search(
    sequence: str,
    search_type: Literal["EXACT", "CONTAINS", "SIMILAR"] = "EXACT",
    sequence_type: Literal["PROTEIN", "NUCLEIC_ACID"] = "PROTEIN",
    size: int = 20,
) -> str:
    """Search substances by biological sequence via the official GSRS API.

    Searches protein or nucleic acid sequences.
    See https://gsrs.ncats.nih.gov/ginas/app/ui/sequence-search for examples.

    Args:
        sequence: Amino acid or nucleotide sequence (e.g. "MVLSPADKTNVKAAWGKVGA").
        search_type: EXACT | CONTAINS | SIMILAR.
        sequence_type: PROTEIN | NUCLEIC_ACID.
        size: Maximum number of results.

    Returns:
        Matching substances with names, UUIDs, and sequence info.
    """
    if not sequence or len(sequence) < 3:
        return "Error: sequence must be at least 3 characters."

    tool = _tool_call("gsrs_api_sequence_search", query_type="sequence")
    try:
        if not runtime.gsrs_api_available():
            reason = runtime.gsrs_api_unavailable_reason()
            tool.finish(
                "degraded",
                result_count=0,
                citation_count=0,
                error_type="RuntimeUnavailable",
                error_message=reason,
            )
            return f"GSRS sequence search is currently unavailable: {reason}"
        resp = runtime.gsrs_api.sequence_search(
            sequence=sequence,
            search_type=search_type,
            sequence_type=sequence_type,
            size=size,
        )
    except Exception as exc:
        tool.fail(exc, result_count=0, citation_count=0)
        return f"GSRS sequence search error: {exc}"

    results = resp.get("content", [])
    tool.finish("success" if results else "abstained", result_count=len(results), citation_count=0)
    if not results:
        return f"No sequence matches found for the given {sequence_type.lower()} sequence."

    lines = [f"Found **{len(results)}** sequence match(es) ({search_type}, {sequence_type}):\n"]
    for i, sub in enumerate(results, 1):
        uuid = sub.get("uuid", "?")
        names = sub.get("names", [])
        name = names[0].get("name", "?") if names else "?"
        sclass = sub.get("substanceClass", "")
        lines.append(f"{i}. **{name}** ({sclass})")
        lines.append(f"   UUID: `{uuid}`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server."""
    transport = os.getenv("MCP_TRANSPORT", "streamable-http").lower()
    if transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    else:
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
