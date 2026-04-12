"""
GSRS MCP Server — MCP Server
Exposes GSRS substance search, Q&A, similarity search, and management
via the Model Context Protocol (MCP) using streamable-http transport.
"""
import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.models import VectorDocument
from app.models.api import (
    AskRequest,
    Citation,
    QueryResult,
    SimilarSubstanceResult,
)
from app.services import VectorDatabaseService, EmbeddingService
from app.services.aggregation import AggregationService
from app.services.gsrs_api import GsrsApiService
from app.services.llm import LLMService
from app.services.metadata_filters import MetadataFilterBuilder
from app.services.query_pipeline import QueryPipelineService
from app.services.query_rewrite import QueryRewriteService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bearer Token Authentication
# ---------------------------------------------------------------------------

class SimpleTokenVerifier(TokenVerifier):
    """Validates Bearer tokens against configured API credentials."""

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        if token == settings.api_password:
            return AccessToken(
                token=token,
                client_id="mcp-client",
                scopes=["mcp:tools"],
            )
        return None


# ---------------------------------------------------------------------------
# Core services
# ---------------------------------------------------------------------------
vector_db = VectorDatabaseService()
embedding_service = EmbeddingService(
    api_key=settings.embedding_api_key,
    model=settings.embedding_model,
    url=settings.embedding_url,
    dimension=settings.embedding_dimension,
    verify_ssl=settings.embedding_verify_ssl,
)
chunker = None  # lazily initialized in lifespan
llm_service = (
    LLMService(
        api_key=settings.llm_api_key,
        url=settings.llm_url,
        model=settings.llm_model,
        verify_ssl=settings.llm_verify_ssl,
        timeout=settings.llm_timeout,
    )
    if settings.llm_api_key
    else None
)
query_pipeline = None  # lazily initialized in lifespan
gsrs_api = GsrsApiService(
    base_url=settings.gsrs_api_url,
    timeout=settings.gsrs_api_timeout,
    verify_ssl=settings.gsrs_api_verify_ssl,
    public_only=settings.gsrs_api_public_only,
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def server_lifespan(server):
    """Initialise vector database and services on startup."""
    global chunker, query_pipeline

    logger.info("Initializing vector database...")
    vector_db.connect()
    vector_db.initialize(dimension=settings.embedding_dimension)
    logger.info("Loaded embedding model: %s", embedding_service.get_model_info())

    from gsrs.model import Substance
    from gsrs.services.ai import SubstanceChunker, ChunkerConfig
    from app.models import VectorDocument

    chunker = SubstanceChunker(
        class_=VectorDocument,
        config=ChunkerConfig(
            name_batch_size=settings.chunker_name_batch_size,
            emit_atomic_name_chunks=settings.chunker_emit_atomic_name_chunks,
            emit_sequence_segments=settings.chunker_emit_sequence_segments,
            max_sequence_segment_len=settings.chunker_max_sequence_segment_len,
            emit_full_sequence_in_text=settings.chunker_emit_full_sequence_in_text,
            include_admin_validation_notes=settings.chunker_include_admin_validation_notes,
            include_reference_index_chunk=settings.chunker_include_reference_index_chunk,
            include_classification_chunk=settings.chunker_include_classification_chunk,
            include_grouped_relationship_summaries=settings.chunker_include_grouped_relationship_summaries,
        ),
    )
    query_pipeline = QueryPipelineService(
        vector_db=vector_db,
        embedding_service=embedding_service,
        llm_service=llm_service,
        use_llm=llm_service is not None,
    )

    yield

    vector_db.disconnect()
    embedding_service.close()


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
auth = None
token_verifier = None
if settings.api_username and settings.api_password:
    auth = AuthSettings(
        issuer_url=AnyHttpUrl("http://localhost"),
        resource_server_url=AnyHttpUrl(f"http://localhost:{settings.api_port}"),
        required_scopes=["mcp:tools"],
    )
    token_verifier = SimpleTokenVerifier()

mcp = FastMCP(
    "GSRS MCP Server",
    instructions=(
        "GSRS (Global Substance Registration System) MCP server. "
        "Provides tools for querying substance data, finding similar substances "
        "by JSON document, and managing the substance database."
    ),
    token_verifier=token_verifier,
    auth=auth,
    host=settings.api_host,
    port=settings.api_port,
    streamable_http_path="/mcp",
    lifespan=server_lifespan,
)


# ---------------------------------------------------------------------------
# Health check (custom route, no auth required)
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"], include_in_schema=True)
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    try:
        stats = vector_db.get_statistics()
    except Exception:
        stats = {"total_chunks": 0, "total_substances": 0}
    return JSONResponse({
        "status": "healthy",
        "database_connected": bool(stats.get("total_chunks") or stats.get("total_substances")),
        "statistics": stats,
    })


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
    keys = {"uuid", "names", "codes", "classifications", "relationships",
            "structure", "properties", "references", "notes"}
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
) -> str:
    """Full AI answering pipeline with citations and evidence.

    Automatically detects if *query* contains a GSRS JSON document and
    runs similarity search instead.

    Args:
        query: Question or GSRS JSON string.
        top_k: Results to retrieve.
        answer_style: concise | standard | detailed.
        return_evidence: Include evidence chunks.
        min_confidence: Minimum confidence threshold.

    Returns:
        Answer with citations, or similar-substance report.
    """
    parsed = _try_parse_json(query)
    substance = parsed if parsed and _is_gsrs_substance(parsed) else None

    if substance:
        example = _extract_search_criteria(substance)
        results = vector_db.search_by_example(
            example=example, top_k=top_k, mode="contains",
        )
        groups = _group_by_substance(results, True, example.get("uuid"))
        name = substance.get("names", [{}])[0].get("name", "substance")
        if not groups:
            return f"No similar substances found for **{name}**."
        lines = [f"Found {len(groups)} substance(s) similar to "
                 f"**{name or 'the provided substance'}**:\n"]
        for i, r in enumerate(groups, 1):
            n = r.canonical_name or r.substance_uuid
            lines.append(f"{i}. **{n}** (score {r.match_score:.2f}, "
                         f"{len(r.chunks)} chunks)")
            if r.matched_fields:
                lines.append(f"   Matched: {', '.join(r.matched_fields[:5])}")
            if r.chunks:
                lines.append(f"   Preview: {r.chunks[0].text[:150]}...")
        return "\n".join(lines)

    req = AskRequest(
        query=query, top_k=top_k, answer_style=answer_style,
        return_evidence=return_evidence, min_confidence=min_confidence,
    )
    resp = query_pipeline.ask(req)
    parts: List[str] = []
    if resp.abstained:
        parts.append(f"⚠️ Could not answer: {resp.abstain_reason or 'Insufficient evidence.'}")
    elif resp.answer:
        parts.append(resp.answer)
    else:
        parts.append("No answer available.")
    parts.append(f"\n📊 Confidence: {resp.confidence:.2f}")
    if resp.citations:
        parts.append("\n📎 Citations:")
        for i, c in enumerate(resp.citations[:5], 1):
            parts.append(f"  [{i}] {c.section}" + (f" — {c.source_url}" if c.source_url else ""))
    if resp.evidence_chunks and return_evidence:
        parts.append("\n📄 Evidence:")
        for i, ch in enumerate(resp.evidence_chunks[:5], 1):
            parts.append(f"  [{i}] ({ch.element_path}, {ch.similarity_score:.2f}) "
                         f"{ch.text[:180]}...")
    return "\n".join(parts)


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
    try:
        substance = json.loads(substance_json)
    except (json.JSONDecodeError, ValueError) as e:
        return f"Error: invalid JSON — {e}"

    example = _extract_search_criteria(substance)
    if not example:
        return ("No searchable metadata found. Provide uuid, names, codes, "
                "or classifications.")

    results = vector_db.search_by_example(
        example=example, top_k=top_k, mode=match_mode,
    )

    groups = _group_by_substance(results, exclude_self, example.get("uuid"))
    if not groups:
        return "No similar substances found."

    name = substance.get("names", [{}])[0].get("name", "substance")
    lines = [f"Found {len(groups)} substance(s) similar to **{name}**:\n"]
    for i, r in enumerate(groups, 1):
        n = r.canonical_name or r.substance_uuid
        lines.append(f"{i}. **{n}** (score {r.match_score:.2f})")
        lines.append(f"   UUID: {r.substance_uuid}")
        if r.matched_fields:
            lines.append(f"   Matched: {', '.join(r.matched_fields[:5])}")
        if r.chunks:
            lines.append(f"   Chunks: {len(r.chunks)} — {r.chunks[0].text[:120]}...")
    return "\n".join(lines)


@mcp.tool()
async def gsrs_retrieve(
    query: str,
    top_k: int = 10,
    filters: Optional[str] = None,
) -> str:
    """Semantic chunk retrieval — raw results, no AI answer.

    Args:
        query: Search text.
        top_k: Number of chunks.
        filters: Optional JSON metadata filter string.

    Returns:
        Ranked text chunks with scores.
    """
    qe = embedding_service.embed(query)
    flt = json.loads(filters) if filters else None
    results = vector_db.similarity_search(qe, top_k=top_k, filters=flt)
    if not results:
        return f"No results for '{query}'."
    lines = [f"Found {len(results)} result(s) for '{query}':\n"]
    for i, result in enumerate(results, 1):
        t = result.document.text[:250] + ("..." if len(result.document.text) > 250 else "")
        lines.append(f"{i}. {t}\n   Score: {result.score:.2f}  Section: {result.document.section}")
    return "\n".join(lines)


@mcp.tool()
async def gsrs_ingest(substance_json: str) -> str:
    """Ingest a GSRS substance JSON into the database.

    Chunks the substance, generates embeddings, stores in vector DB.

    Args:
        substance_json: GSRS JSON string.

    Returns:
        Ingestion result.
    """
    try:
        substance = json.loads(substance_json)
    except (json.JSONDecodeError, ValueError) as e:
        return f"Error: invalid JSON — {e}"
    from gsrs.model import Substance
    sm = Substance.model_validate(substance)
    chunks = chunker.chunk(sm)
    if not chunks:
        return "No chunks generated from substance."
    texts = [str(c.text) for c in chunks]
    embeddings = embedding_service.embed_batch(texts)
    documents = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk.set_embedding(embedding)
        documents.append(chunk)
    count = vector_db.upsert_documents(documents)
    uid = str(getattr(sm, "uuid", "unknown"))
    logger.info("Ingested %s: %d chunks", uid, count)
    return f"Ingested **{uid}** — {count} chunks."


@mcp.tool()
async def gsrs_delete(substance_uuid: str) -> str:
    """Delete all chunks for a substance.

    Args:
        substance_uuid: Substance UUID.

    Returns:
        Deletion confirmation.
    """
    count = vector_db.delete_documents_by_substance(UUID(substance_uuid))
    logger.info("Deleted %s: %d chunks", substance_uuid, count)
    return f"Deleted **{substance_uuid}** — {count} chunks removed."


@mcp.tool()
async def gsrs_health() -> str:
    """Gateway health, model info, and database statistics."""
    try:
        stats = vector_db.get_statistics()
    except Exception:
        stats = {"total_chunks": 0, "total_substances": 0}
    try:
        mi = embedding_service.get_model_info()
    except Exception:
        mi = {}
    lines = [
        f"🟢 Database: {stats.get('total_chunks', 0)} chunks, "
        f"{stats.get('total_substances', 0)} substances",
        f"🤖 Model: {mi.get('model', 'N/A')} ({mi.get('dimension', '?')} dim)",
    ]
    return "\n".join(lines)


@mcp.tool()
async def gsrs_statistics() -> str:
    """Return database statistics (chunk count, substance count)."""
    return json.dumps(vector_db.get_statistics(), indent=2)


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
    rewriter = QueryRewriteService()
    agg = AggregationService()
    filter_builder = MetadataFilterBuilder()

    rw = rewriter.rewrite(query)
    queries = [rw.canonical_query] + rw.rewrites
    applied_filters = filter_builder.build(inferred_filters=rw.filters)

    all_candidates: Dict[str, Any] = {}
    for q in queries:
        emb = embedding_service.embed(q)
        results = vector_db.similarity_search(emb, top_k=top_k, filters=applied_filters)
        for r in results:
            cid = r.document.chunk_id
            if cid not in all_candidates:
                all_candidates[cid] = (r.document, r.score)

    if not all_candidates:
        return f"No data found for aggregation query: **{query}**."

    sorted_candidates = sorted(all_candidates.values(), key=lambda x: x[1], reverse=True)[:top_k]
    result = agg.aggregate(sorted_candidates, query, intent=rw.intent)

    if aggregation_type == "count":
        return f"**{result.substance_name}** has **{result.total_count}** {result.aggregation_type}."

    return result.raw_text_summary


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
    rewriter = QueryRewriteService()
    rw = rewriter.rewrite(query)

    lines = [f"**Original:** {query}", f"**Intent:** {rw.intent}", f"**Canonical:** {rw.canonical_query}"]

    if rw.filters:
        lines.append(f"**Inferred filters:** {json.dumps(rw.filters, indent=2)}")

    if mode == "translate" and target_language != "en":
        if llm_service:
            translated = llm_service.complete_text(
                system_prompt=f"Translate the following query to {target_language}. Return only the translation.",
                user_prompt=query,
                temperature=0.0,
            )
            lines.append(f"\n**Translated ({target_language}):** {translated}")
        else:
            lines.append(f"\n**Translation unavailable** (no LLM configured).")
    else:
        lines.append("\n**Query variants:**")
        for i, v in enumerate(rw.rewrites, 1):
            lines.append(f"  {i}. {v}")

    return "\n".join(lines)


@mcp.tool()
async def gsrs_get_document(substance_uuid: str) -> str:
    """Fetch a complete GSRS substance JSON document by UUID from the official GSRS API.

    Args:
        substance_uuid: Substance UUID (e.g. "0103a288-6eb6-4ced-b13a-849cd7edf028").

    Returns:
        Full substance JSON or an error message.
    """
    try:
        doc = gsrs_api.get_substance_by_uuid(substance_uuid)
    except Exception as e:
        return f"Error fetching substance **{substance_uuid}**: {e}"

    if doc is None:
        return f"Substance **{substance_uuid}** not found in GSRS API."

    return json.dumps(doc, indent=2)


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
    try:
        resp = gsrs_api.text_search(query, page=page, size=size, fields=fields or None)
    except Exception as e:
        return f"GSRS API search error: {e}"

    results = resp.get("results", [])
    total = resp.get("total", 0)

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

    try:
        resp = gsrs_api.structure_search(
            smiles=smiles or None,
            inchi=inchi or None,
            search_type=search_type,
            size=size,
        )
    except Exception as e:
        return f"GSRS structure search error: {e}"

    results = resp.get("results", [])
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

    try:
        resp = gsrs_api.sequence_search(
            sequence=sequence,
            search_type=search_type,
            sequence_type=sequence_type,
            size=size,
        )
    except Exception as e:
        return f"GSRS sequence search error: {e}"

    results = resp.get("results", [])
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
