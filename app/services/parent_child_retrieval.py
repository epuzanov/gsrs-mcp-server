"""
Parent-Child Retrieval using Virtual Parent Reconstruction

Implements parent-child retrieval through virtual reconstruction of parent context
from existing chunk records. Parent identity is defined as (document_id, root_section),
and parent context is reconstructed by loading all chunks sharing the same parent identity.

This approach minimizes schema changes, migration effort, and reindexing while
delivering most of the benefits of parent-child retrieval.
"""
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID

from app.models import VectorDocument, DBQueryResult
from app.observability import InMemoryMetrics
from app.services.chunker import sections_in_root

logger = logging.getLogger(__name__)


# Default cap on per-chunk text inside the parent context. Mirrors the
# value previously hard-coded in ``reconstruct_parent_context``.
_DEFAULT_PARENT_CHUNK_TEXT_LIMIT = 500

# Default cap on aggregated parent text passed to the model.
_DEFAULT_PARENT_TEXT_LIMIT = 1000

# Default cap on number of text parts aggregated from the parent context.
_DEFAULT_PARENT_TEXT_PARTS_LIMIT = 5

# Cap on chunks pulled from the backend when reconstructing a parent.
# Keeps a runaway large section from blowing up the response. The
# backends all support ``limit`` pushdown.
_DEFAULT_PARENT_BACKEND_LIMIT = 200

# Default cap on the size of the markdown rendered by ``summarize_parent``.
# Like the per-context text limit, the cap keeps a runaway parent from
# blowing up the response when retrieval triggers a wide rebuild.
_DEFAULT_PARENT_SUMMARY_MAX_CHARS = 4000

# Cap on number of chunks rendered per ``chunk_type`` bucket. Each
# bucket's chunks are rendered in their original (deterministic) order;
# chunks beyond the cap are dropped from the markdown to keep the
# output readable, but are still counted in the per-section header.
_DEFAULT_PARENT_SUMMARY_BUCKET_LIMIT = 50

# Map of ``chunk_type`` → human-readable section title. Driven by
# what the chunker actually emits (see ``app/services/chunker.py``)
# so the title list stays in sync with the chunker.
#
# The relationship sub-sections (activemoiety / metabolites / …) are
# intentionally not listed here — those are rebucketed under
# ``Relationships`` by ``summarize_parent`` using the
# ``_RELATIONSHIP_CHUNK_TO_BUCKET`` table below. They surface under
# the typed sub-bucket rather than as their own top-level section so
# the summary mirrors ``app/services/summary.py``.
_CHUNK_TYPE_TITLES: dict[str, str] = {
    "overview": "Overview",
    "name": "Names",
    "identifier": "Identifiers",
    "classification": "Classifications",
    "structure": "Chemical Structure",
    "moiety": "Moieties",
    "protein": "Protein Details",
    "protein_sequence": "Protein Sequence",
    "protein_sequence_segment": "Protein Sequence",
    "protein_sequence_summary": "Protein Sequence",
    "nucleic_acid": "Nucleic Acid Details",
    "nucleic_acid_sequence": "Nucleic Acid Sequence",
    "nucleic_acid_sequence_segment": "Nucleic Acid Sequence",
    "nucleic_acid_sequence_summary": "Nucleic Acid Sequence",
    "polymer": "Polymer Details",
    "polymer_display_structure": "Polymer Display Structure",
    "polymer_idealized_structure": "Polymer Idealized Structure",
    "structurally_diverse": "Source Material",
    "mixture": "Mixture",
    "mixture_component": "Mixture Components",
    "specified_substance_constituents": "Specified Substance Constituents",
    "tag": "Tags",
    "property": "Properties",
    "reference": "References",
    "note": "Notes",
    "modification": "Modifications",
}

# Map of relationship section (``section``) → typed bucket. Mirrors the
# routing in ``summary.py._RELATIONSHIP_SECTION_TITLES`` and the
# chunker's ``SubstanceChunker._section_for_relationship_type``. The
# section value comes from the chunker (``activemoiety``,
# ``metabolites``, ``impurities``, ``constituents``, ``salts``,
# ``relationships``) — note that the chunker writes the literal
# ``salts`` whereas ``summary.py`` displays ``Salts or Solvates`` and
# uses bucket key ``salts_or_solvates``. We map chunk section →
# summary bucket key here so the two stay aligned.
_RELATIONSHIP_CHUNK_TO_BUCKET: dict[str, str] = {
    "activemoiety": "active_moieties",
    "metabolites": "metabolites",
    "impurities": "impurities",
    "constituents": "constituents",
    "salts": "salts_or_solvates",
    "relationships": "other",
}

# Section title for each relationship bucket. Kept in lock-step with
# ``app.services.summary._RELATIONSHIP_SECTION_TITLES``. If a new
# bucket is added there, add it here too.
_RELATIONSHIP_BUCKET_TITLES: dict[str, str] = {
    "active_moieties": "Active Moieties",
    "metabolites": "Metabolites",
    "impurities": "Impurities",
    "constituents": "Constituents",
    "salts_or_solvates": "Salts or Solvates",
    "other": "Other Relationships",
}

# Top-level section rendering order. ``definitions`` is special — it
# collects every per-class sub-section under a single H2 — so the
# list intentionally doesn't enumerate the per-class chunk_types.
_PARENT_SUMMARY_SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("overview", "Overview"),
    ("names", "Names"),
    ("identifiers", "Identifiers"),
    ("classifications", "Classifications"),
    ("definitions", "Definitions"),
    ("relationships", "Relationships"),
    ("properties", "Properties"),
    ("references", "References"),
    ("tags", "Tags"),
    ("notes", "Notes"),
)


@dataclass
class ParentIdentity:
    """Virtual parent identity: (document_id, root_section)."""

    document_id: UUID
    root_section: str

    def __hash__(self):
        return hash((self.document_id, self.root_section))

    def __eq__(self, other):
        if not isinstance(other, ParentIdentity):
            return False
        return self.document_id == other.document_id and self.root_section == other.root_section

    def __repr__(self):
        return f"ParentIdentity(doc_id={self.document_id}, section={self.root_section})"


class ParentContextEnricher:
    """
    Reconstructs parent context from chunks and enriches child chunks with parent information.
    
    This class handles virtual parent reconstruction, allowing child chunks to be
    augmented with context from all chunks belonging to the same parent identity.
    """

    def __init__(
        self,
        vector_db,
        metrics: Optional[InMemoryMetrics] = None,
    ):
        """
        Initialize the parent context enricher.

        Args:
            vector_db: VectorDatabase instance for retrieving chunks
            metrics: Optional ``InMemoryMetrics`` used to record rebuild
                latency and truncation events. When ``None`` a fresh
                instance is created — keeps the enricher observable in
                tests and isolated callers without changing the API.
        """
        self.vector_db = vector_db
        self.metrics = metrics if metrics is not None else InMemoryMetrics()


    def get_parent_identity(self, chunk: VectorDocument) -> Tuple[ParentIdentity, bool]:
        """
        Get the virtual parent identity for a chunk.

        Returns:
            A tuple of (ParentIdentity, fallback_used) where ``fallback_used``
            is True when the chunk did not have an explicit ``root_section``
            or hierarchy in its metadata. The chunker now writes the
            ``root_section`` as a dedicated column, but metadata is still
            checked for backwards compatibility with older indexes.
        """
        metadata = chunk.metadata_json or {}
        explicit_root = "root_section" in metadata
        explicit_hierarchy = (
            isinstance(metadata.get("hierarchy"), dict)
            and metadata["hierarchy"].get("parent_section")
        )
        root_section = self.extract_root_section(chunk)
        fallback_used = not (explicit_root or explicit_hierarchy)
        identity = ParentIdentity(
            document_id=chunk.document_id,
            root_section=root_section,
        )
        return identity, fallback_used

    def reconstruct_parent_context(
        self,
        parent_identity: ParentIdentity,
        exclude_sections: Optional[Set[str]] = None,
        exclude_chunk_ids: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reconstruct parent context from chunks with the given parent identity.

        Loads all chunks sharing the same parent identity and builds a unified
        parent context document containing aggregated information. The query
        is pushed down to the backend (``sections=[...]``, ``limit=...``) so
        we don't fetch the whole document and filter in Python.

        Args:
            parent_identity: The parent identity to reconstruct context for
            exclude_sections: Sections to exclude from the parent context.
                Legacy parameter; prefer ``exclude_chunk_ids`` so siblings
                in the same section are not silently dropped.
            exclude_chunk_ids: Chunk IDs to exclude (typically the child
                chunk's own ID). Preferred over ``exclude_sections``.

        Returns:
            Dictionary containing parent context, or None if no chunks found
        """
        if exclude_sections is None:
            exclude_sections = set()
        if exclude_chunk_ids is None:
            exclude_chunk_ids = set()

        started_at = time.perf_counter()
        try:
            context = self._reconstruct_parent_context_inner(
                parent_identity, exclude_sections, exclude_chunk_ids
            )
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self.metrics.increment("parent_rebuild.count")
            self.metrics.observe_latency("parent_rebuild.latency_ms", elapsed_ms)

        return context

    def _reconstruct_parent_context_inner(
        self,
        parent_identity: ParentIdentity,
        exclude_sections: Set[str],
        exclude_chunk_ids: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Inner reconstruction logic, isolated for clean latency wrapping.

        Pushes the filter down to the vector store so PGVector / Chroma can
        use indexed column filters. We request chunks by `root_section`
        when the backend supports it; otherwise we fall back to the concrete
        `sections` list (the previous behaviour). The in-Python filter is
        still applied as a safety net for legacy data and for
        `exclude_chunk_ids`.
        """
        # Try the indexed root_section filter first.
        try:
            parent_chunks = self.vector_db.get_documents(
                substance_uuid=parent_identity.document_id,
                root_sections=[parent_identity.root_section],
                limit=_DEFAULT_PARENT_BACKEND_LIMIT,
            )
        except TypeError:
            # Backends that don't yet accept root_sections fall back to
            # the concrete section list.
            parent_sections = sections_in_root(parent_identity.root_section)
            try:
                parent_chunks = self.vector_db.get_documents(
                    substance_uuid=parent_identity.document_id,
                    sections=parent_sections,
                    limit=_DEFAULT_PARENT_BACKEND_LIMIT,
                )
            except TypeError:
                # Backends that don't accept sections/limit either fall back
                # to the whole-document fetch.
                parent_chunks = self.vector_db.get_documents(
                    substance_uuid=parent_identity.document_id
                )

        if not parent_chunks:
            return None

        # Defensive in-Python filter for legacy data and the
        # ``exclude_chunk_ids`` case (the backend only knows about
        # ``section`` / ``root_section``; chunk_id exclusion has to happen
        # here).
        filtered_chunks = []
        for chunk in parent_chunks:
            if self.extract_root_section(chunk) != parent_identity.root_section:
                continue
            if chunk.chunk_id in exclude_chunk_ids:
                continue
            if chunk.section in exclude_sections:
                continue
            filtered_chunks.append(chunk)
        parent_chunks = filtered_chunks

        if not parent_chunks:
            return None

        # Build parent context from chunks, preserving the order in which
        # sections first appear across the returned chunks. Deterministic
        # output avoids noisy tests/snapshots and inconsistent MCP text.
        seen_sections: List[str] = []
        for chunk in parent_chunks:
            if chunk.section and chunk.section not in seen_sections:
                seen_sections.append(chunk.section)

        parent_context = {
            "parent_identity": {
                "document_id": str(parent_identity.document_id),
                "root_section": parent_identity.root_section,
            },
            "num_chunks": len(parent_chunks),
            "sections_included": seen_sections,
            "text_parts": [],
            "metadata_unified": {},
        }

        # Aggregate text from all parent chunks
        for chunk in parent_chunks:
            if chunk.text:
                original_len = len(chunk.text)
                truncated_text = (
                    chunk.text[:_DEFAULT_PARENT_CHUNK_TEXT_LIMIT]
                    if original_len > _DEFAULT_PARENT_CHUNK_TEXT_LIMIT
                    else chunk.text
                )
                if (
                    original_len > _DEFAULT_PARENT_CHUNK_TEXT_LIMIT
                    and chunk.section not in exclude_sections
                ):
                    # Per-chunk truncation is rare but possible; surface it on
                    # the context so callers can decide whether to widen the
                    # limit.
                    self.metrics.increment("parent_text.truncated")
                parent_context["text_parts"].append({
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "text": truncated_text,
                    "source_url": chunk.source_url,
                    "truncated": original_len > _DEFAULT_PARENT_CHUNK_TEXT_LIMIT,
                    # Carried on the part so the markdown renderer
                    # (``summarize_parent``) can group rows by
                    # ``chunk_type`` and pull structured fields
                    # (e.g. ``code_system``, ``smiles``) without
                    # re-walking the chunk list. Empty/missing on
                    # legacy chunks — the renderer treats both as
                    # the generic ``other`` bucket.
                    "chunk_type": (chunk.metadata_json or {}).get("chunk_type", ""),
                    "metadata": dict(chunk.metadata_json or {}),
                })

            # Merge metadata
            if chunk.metadata_json:
                for key, value in chunk.metadata_json.items():
                    if key not in parent_context["metadata_unified"]:
                        parent_context["metadata_unified"][key] = value

        return parent_context

    def enrich_chunk_with_parent(
        self,
        chunk: VectorDocument,
        parent_context: Optional[Dict[str, Any]] = None,
        include_parent_text: bool = True,
        parent_text_limit: int = _DEFAULT_PARENT_TEXT_LIMIT,
    ) -> Dict[str, Any]:
        """
        Enrich a chunk with parent context information.

        Args:
            chunk: The chunk to enrich
            parent_context: Pre-computed parent context (if None, will be reconstructed)
            include_parent_text: Whether to include aggregated parent text
            parent_text_limit: Maximum character limit for parent text content

        Returns:
            Dictionary with chunk data and parent context. When
            `include_parent_text` is true and the aggregated text was cut
            down to fit `parent_text_limit`, the result includes
            `parent_text_truncated: true` so callers can detect lossy
            summaries.
        """
        enriched: Dict[str, Any] = {
            "chunk": {
                "document_id": str(chunk.document_id),
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_url": chunk.source_url,
                "metadata": chunk.metadata_json or {},
            },
        }

        # Determine whether this chunk relies on the legacy fallback
        # before we possibly reconstruct the parent context.
        _, fallback_used = self.get_parent_identity(chunk)

        # Get or reconstruct parent context. Exclude the child by chunk_id
        # (not by section) so siblings in the same section are preserved.
        if parent_context is None:
            parent_identity, _ = self.get_parent_identity(chunk)
            parent_context = self.reconstruct_parent_context(
                parent_identity,
                exclude_chunk_ids={chunk.chunk_id},
            )

        if parent_context:
            enriched["parent_context"] = parent_context
            if fallback_used:
                enriched["parent_grouping_fallback_used"] = True
                enriched["parent_context"]["parent_grouping_fallback_used"] = True

            if include_parent_text and parent_context.get("text_parts"):
                # Aggregate parent text, respecting the parts and char limits.
                aggregated_parts = parent_context["text_parts"][
                    :_DEFAULT_PARENT_TEXT_PARTS_LIMIT
                ]
                parent_text = "\n\n".join(
                    f"[{part['section']}] {part['text']}"
                    for part in aggregated_parts
                )
                truncated = False
                if len(parent_text) > parent_text_limit:
                    parent_text = parent_text[:parent_text_limit].rstrip() + "..."
                    truncated = True
                    self.metrics.increment("parent_text.truncated")
                enriched["parent_text_summary"] = parent_text
                enriched["parent_text_truncated"] = truncated
                if truncated:
                    enriched["parent_text_truncated_chars"] = parent_text_limit

        return enriched

    def enrich_search_results(
        self,
        results: List[DBQueryResult],
        include_parent_text: bool = True,
        parent_text_limit: int = _DEFAULT_PARENT_TEXT_LIMIT,
    ) -> List[Dict[str, Any]]:
        """
        Enrich multiple search results with parent context.

        Reconstructs parent context for results, deduping on parent
        identity so multiple results that share a parent do not
        duplicate the backend fetch. Per-child exclusion
        (``exclude_chunk_ids``) is applied after the lookup, so a
        child never echoes itself in its own parent summary even when
        several results in the same call share a parent.

        Args:
            results: List of DBQueryResult from vector search
            include_parent_text: Whether to include parent text summaries
            parent_text_limit: Maximum character limit for parent text

        Returns:
            List of enriched result dictionaries
        """
        # Dedup the *raw* parent context (no exclusions applied) by parent
        # identity. Each child applies its own ``exclude_chunk_ids`` after
        # the lookup so siblings remain visible in the parent. The dict
        # is keyed by identity (membership test is key-based), so a
        # ``None`` value still counts as "already looked up" and avoids
        # repeating the backend fetch.
        seen_parents: Dict[ParentIdentity, Optional[Dict[str, Any]]] = {}

        def _get_or_build_parent(
            parent_identity: ParentIdentity,
        ) -> Optional[Dict[str, Any]]:
            if parent_identity not in seen_parents:
                seen_parents[parent_identity] = self.reconstruct_parent_context(
                    parent_identity
                )
            return seen_parents[parent_identity]

        enriched_results = []
        for result in results:
            chunk = result.document
            parent_identity, fallback_used = self.get_parent_identity(chunk)
            raw_parent = _get_or_build_parent(parent_identity)

            # Strip the child chunk's text part out of the cached parent so
            # the child never appears inside its own parent summary. This
            # walks the cached ``text_parts`` once and is cheap.
            parent_context = self._exclude_chunk_from_parent(
                raw_parent, chunk.chunk_id
            )

            # Enrich the chunk
            enriched = self.enrich_chunk_with_parent(
                chunk=chunk,
                parent_context=parent_context,
                include_parent_text=include_parent_text,
                parent_text_limit=parent_text_limit,
            )
            enriched["score"] = result.score
            if fallback_used:
                enriched["parent_grouping_fallback_used"] = True
                if enriched.get("parent_context"):
                    enriched["parent_context"]["parent_grouping_fallback_used"] = True

            enriched_results.append(enriched)

        return enriched_results

    @staticmethod
    def _exclude_chunk_from_parent(
        parent_context: Optional[Dict[str, Any]],
        chunk_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return a shallow copy of ``parent_context`` with ``chunk_id`` removed.

        Operates on the cached text_parts and chunks list (when present)
        so multiple children of the same parent each see an honest
        summary. If the parent context is ``None`` or has no parts, the
        input is returned unchanged.
        """
        if not parent_context or not chunk_id:
            return parent_context

        text_parts = parent_context.get("text_parts") or []
        filtered_parts = [
            part for part in text_parts if part.get("chunk_id") != chunk_id
        ]
        if len(filtered_parts) == len(text_parts):
            # No parts carried a chunk_id, but legacy entries may still
            # represent the child. The chunker only writes ``chunk_id``
            # for sub-section children, so this branch is a no-op for
            # most flows. Return the original context untouched.
            return parent_context

        # Build a shallow copy so the cached parent is not mutated for
        # other children that share the cache.
        new_context = dict(parent_context)
        new_context["text_parts"] = filtered_parts
        new_context["num_chunks"] = len(filtered_parts)
        return new_context

    def get_all_parents_in_document(
        self,
        document_id: UUID,
    ) -> List[ParentIdentity]:
        """
        Get all unique parent identities in a document.

        Args:
            document_id: The document (substance) UUID

        Returns:
            List of unique ParentIdentity objects
        """
        # Use the backend's native aggregation when available; otherwise
        # fall back to fetching all chunks and deriving parents in Python.
        try:
            root_sections = self.vector_db.get_root_sections(substance_uuid=document_id)
        except (AttributeError, TypeError):
            root_sections = []

        if root_sections:
            return sorted(
                [ParentIdentity(document_id=document_id, root_section=rs) for rs in root_sections],
                key=lambda p: p.root_section,
            )

        all_chunks = self.vector_db.get_documents(substance_uuid=document_id)
        if not all_chunks:
            return []

        parent_identities: Set[ParentIdentity] = set()
        for chunk in all_chunks:
            parent_identity, _ = self.get_parent_identity(chunk)
            parent_identities.add(parent_identity)

        return sorted(parent_identities, key=lambda p: p.root_section)

    @staticmethod
    def extract_root_section(chunk: VectorDocument) -> str:
        """
        Extract the root section identifier from a chunk.

        The root section represents the top-level section of a document and
        drives parent-child grouping. Newly chunked data writes
        ``root_section`` as a dedicated column on ``VectorDocument`` and
        also keeps it in ``metadata_json`` for backwards compatibility. This
        method prefers the column value, then metadata, then hierarchy, and
        finally falls back to the chunk's own ``section`` for legacy data.

        Args:
            chunk: VectorDocument to extract root section from

        Returns:
            Root section identifier (e.g. "overview", "codes", "names")
        """
        # 1. Dedicated column -- preferred for newly indexed data.
        if getattr(chunk, "root_section", None):
            return str(chunk.root_section)

        # 2. Explicit root_section in metadata -- backwards compatibility.
        metadata = chunk.metadata_json or {}
        if "root_section" in metadata:
            return str(metadata["root_section"])

        # 3. Hierarchy information (parent_section) from older chunkers.
        hierarchy = metadata.get("hierarchy")
        if isinstance(hierarchy, dict):
            parent_section = hierarchy.get("parent_section")
            if parent_section:
                return str(parent_section)

        # 4. Legacy / missing root_section: fall back to the chunk's own
        #    section. This is safe but may group sub-optimally; callers can
        #    detect the fallback via ``parent_grouping_fallback_used``.
        if chunk.section:
            return chunk.section

        return "overview"

    # ------------------------------------------------------------------
    # Markdown rendering of a reconstructed parent
    # ------------------------------------------------------------------

    def summarize_parent(
        self,
        parent_identity: ParentIdentity,
        *,
        max_chars: int = _DEFAULT_PARENT_SUMMARY_MAX_CHARS,
        exclude_chunk_ids: Optional[Set[str]] = None,
        include_text_parts: bool = True,
    ) -> str:
        """Render a parent context as a chunk-driven markdown summary.

        The summary is reconstructed from chunks in the vector store —
        it does not call the GSRS upstream. It is a **parent-scoped**
        view: it covers the chunks that share ``parent_identity`` (i.e.
        the same document and root section), not the whole substance.

        Compare to :func:`app.services.summary.substance_to_markdown`,
        which renders the full GSRS substance JSON. Fields that the
        chunker does not emit (top-level ``approvalID``, ``status``,
        substance-level ``access``) are omitted here, and a footer
        line is added to flag the gap.

        Args:
            parent_identity: The parent identity to summarize. Typically
                obtained via :meth:`get_parent_identity` on a child
                chunk, or via :meth:`get_all_parents_in_document`.
            max_chars: Soft cap on the rendered markdown size. If the
                output exceeds this, it is truncated and the
                ``parent_summary.truncated`` metric is incremented.
            exclude_chunk_ids: Chunk IDs to exclude (e.g. the child
                chunk the caller is showing alongside the summary).
            include_text_parts: When True, include the raw chunk text
                for each rendered row. When False, only the structured
                metadata fields are rendered.

        Returns:
            A markdown string. Empty string when the parent has no
            chunks. The output never raises for missing data — empty
            sub-sections are simply omitted.
        """
        if exclude_chunk_ids is None:
            exclude_chunk_ids = set()

        context = self.reconstruct_parent_context(
            parent_identity,
            exclude_chunk_ids=exclude_chunk_ids,
        )
        if not context:
            return ""

        text_parts = context.get("text_parts") or []
        sections_included = context.get("sections_included") or []
        metadata_unified = context.get("metadata_unified") or {}

        # Group chunks by chunk_type while preserving the order in
        # which each chunk_type first appears in the input. This keeps
        # the rendered output stable across calls (and matches the
        # deterministic order in ``sections_included``).
        ordered_chunk_types: List[str] = []
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for part in text_parts:
            chunk_type = part.get("chunk_type") or "other"
            if chunk_type not in buckets:
                ordered_chunk_types.append(chunk_type)
                buckets[chunk_type] = []
            buckets[chunk_type].append(part)

        # The relationship typed sub-sections (``activemoiety``,
        # ``metabolites``, …) are emitted by the chunker with
        # ``chunk_type == "relationship"`` and a ``section`` set to the
        # sub-section. Re-bucket them by summary bucket key so they
        # render under a single ``Relationships`` H2 with typed
        # sub-sections, mirroring ``app.services.summary``.
        ordered_relationship_buckets: List[str] = []
        relationship_buckets: Dict[str, List[Dict[str, Any]]] = {}

        def _reassign_to_relationship_bucket(part: Dict[str, Any]) -> None:
            section = part.get("section") or ""
            bucket_key = _RELATIONSHIP_CHUNK_TO_BUCKET.get(section, "other")
            if bucket_key not in relationship_buckets:
                ordered_relationship_buckets.append(bucket_key)
                relationship_buckets[bucket_key] = []
            relationship_buckets[bucket_key].append(part)

        for chunk_type in list(ordered_chunk_types):
            if chunk_type != "relationship":
                continue
            for part in buckets[chunk_type]:
                _reassign_to_relationship_bucket(part)
            # Drop the flat ``relationship`` bucket — its rows now
            # live under the typed sub-buckets.
            del buckets[chunk_type]
            ordered_chunk_types.remove(chunk_type)

        # Group per-class sub-section chunks (e.g. ``structure``,
        # ``moiety``, ``protein``, ``polymer``, …) under a single
        # ``Definitions`` H2 instead of rendering each as its own
        # top-level section.
        definition_chunk_types = {
            "structure",
            "moiety",
            "protein",
            "protein_sequence",
            "protein_sequence_segment",
            "protein_sequence_summary",
            "nucleic_acid",
            "nucleic_acid_sequence",
            "nucleic_acid_sequence_segment",
            "nucleic_acid_sequence_summary",
            "polymer",
            "polymer_display_structure",
            "polymer_idealized_structure",
            "structurally_diverse",
            "mixture",
            "mixture_component",
            "specified_substance_constituents",
            "modification",
        }
        ordered_definition_types: List[str] = []
        for chunk_type in list(ordered_chunk_types):
            if chunk_type in definition_chunk_types:
                ordered_definition_types.append(chunk_type)
                ordered_chunk_types.remove(chunk_type)

        # Pull the display name out of the merged metadata. Chunks
        # produced by the chunker always carry ``display_name``, but
        # older indexes may not — fall back to the parent identity
        # UUID in that case so the header is never empty.
        display_name = metadata_unified.get("display_name") or str(
            parent_identity.document_id
        )

        # --------------------------------------------------------------
        # Build markdown
        # --------------------------------------------------------------
        lines: List[str] = []
        lines.append(f"# Parent: {display_name} — {parent_identity.root_section}")
        lines.append("")

        header_facts: List[str] = []
        definition_type = metadata_unified.get("substance_definition_type")
        if definition_type:
            header_facts.append(f"**Definition Type:** {definition_type}")
        if sections_included:
            header_facts.append(
                "**Sections:** " + ", ".join(sections_included)
            )
        header_facts.append(
            f"**Chunks:** {context.get('num_chunks', len(text_parts))}"
        )
        for fact in header_facts:
            lines.append(fact)
        lines.append("")

        def _emit_section(chunk_type: str) -> None:
            rows = buckets.get(chunk_type) or []
            if not rows:
                return
            title = _CHUNK_TYPE_TITLES.get(chunk_type, chunk_type.title())
            lines.append(f"## {title}")
            lines.append("")
            rendered = _render_rows_as_table(
                rows,
                include_text_parts=include_text_parts,
                limit=_DEFAULT_PARENT_SUMMARY_BUCKET_LIMIT,
            )
            if rendered:
                lines.extend(rendered)
                lines.append("")

        # Render the top-level sections in the order the chunker
        # actually emitted them, falling back to a deterministic
        # default order for the well-known roots so output is stable
        # when a section has no chunks of its own (those are skipped
        # by ``_emit_section`` anyway, so the loop is just a no-op).
        rendered_roots: set = set()
        for chunk_type in ordered_chunk_types:
            if chunk_type in definition_chunk_types:
                continue  # handled under ``Definitions`` below
            if chunk_type in rendered_roots:
                continue
            rendered_roots.add(chunk_type)
            _emit_section(chunk_type)

        # Definitions (per-class data) — always rendered as a single
        # block. Inside, each ``chunk_type`` is its own H3.
        if ordered_definition_types:
            lines.append("## Definitions")
            lines.append("")
            for chunk_type in ordered_definition_types:
                rows = buckets.get(chunk_type) or []
                if not rows:
                    continue
                title = _CHUNK_TYPE_TITLES.get(chunk_type, chunk_type.title())
                lines.append(f"### {title}")
                lines.append("")
                rendered = _render_rows_as_table(
                    rows,
                    include_text_parts=include_text_parts,
                    limit=_DEFAULT_PARENT_SUMMARY_BUCKET_LIMIT,
                )
                if rendered:
                    lines.extend(rendered)
                    lines.append("")

        # Relationships — typed sub-sections, mirroring
        # ``summary._format_relationships``.
        if relationship_buckets:
            lines.append("## Relationships")
            lines.append("")
            for bucket_key in ordered_relationship_buckets:
                rows = relationship_buckets.get(bucket_key) or []
                if not rows:
                    continue
                title = _RELATIONSHIP_BUCKET_TITLES.get(bucket_key, bucket_key)
                lines.append(f"### {title}")
                lines.append("")
                rendered = _render_rows_as_table(
                    rows,
                    include_text_parts=include_text_parts,
                    limit=_DEFAULT_PARENT_SUMMARY_BUCKET_LIMIT,
                )
                if rendered:
                    lines.extend(rendered)
                    lines.append("")

        # Footer — flag the gap between chunk-driven and
        # substance-driven summaries. Only emit on the root sections
        # where the missing fields actually matter (definitions /
        # overview / names); otherwise this is just noise.
        if parent_identity.root_section in {
            "overview",
            "definitions",
            "names",
        }:
            lines.append("> **Note:** substance-level metadata not available from chunks; approval ID, status, and substance-level access require `gsrs_get_summary`.")
            lines.append("")

        markdown = "\n".join(lines).rstrip() + "\n"
        if len(markdown) > max_chars:
            # Reserve 4 characters for the ``...\n`` truncation
            # marker so the final output is at most ``max_chars``
            # characters (the marker itself counts toward the cap).
            head_room = max(max_chars - 4, 1)
            head = markdown[:head_room].rstrip()
            # ``...`` is 3 chars. The slice above may have left up
            # to 3 trailing characters of headroom; re-slice to
            # guarantee the final string is <= max_chars regardless
            # of the original body content (whitespace stripping,
            # mid-X run, etc.).
            if len(head) + 4 > max_chars:
                head = head[: max_chars - 4].rstrip()
            markdown = head + "...\n"
            self.metrics.increment("parent_summary.truncated")
        return markdown


# ------------------------------------------------------------------
# Module-level helpers for ``summarize_parent``
# ------------------------------------------------------------------

# Column ordering per ``chunk_type`` (and per relationship bucket).
# Columns not present on a row are rendered as an empty cell. The
# ``text`` column is appended at the end when ``include_text_parts``
# is True; otherwise the structured columns stand on their own.
_TABLE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "overview": ("definition_type",),
    "name": ("name", "name_type", "name_orgs", "access"),
    "identifier": ("code", "code_system", "code_type", "code_url", "access"),
    "classification": ("code", "code_system", "code_url", "access"),
    "structure": ("smiles", "molecular_formula", "inchi_key", "access"),
    "moiety": ("moiety_index", "smiles", "molecular_formula", "inchi_key", "count", "access"),
    "protein": ("substance_class", "access"),
    "protein_sequence": ("subunit_index",),
    "protein_sequence_segment": ("subunit_index", "segment_index"),
    "protein_sequence_summary": ("subunit_index",),
    "nucleic_acid": ("substance_class",),
    "nucleic_acid_sequence": ("subunit_index",),
    "nucleic_acid_sequence_segment": ("subunit_index", "segment_index"),
    "nucleic_acid_sequence_summary": ("subunit_index",),
    "polymer": ("substance_class",),
    "polymer_display_structure": ("smiles", "molecular_formula"),
    "polymer_idealized_structure": ("smiles", "molecular_formula"),
    "structurally_diverse": ("substance_class",),
    "mixture": ("substance_class",),
    "mixture_component": ("component_type", "component_uuid"),
    "specified_substance_constituents": ("constituent_count", "substance_class"),
    "tag": ("tag",),
    "property": ("property_name", "property_type", "value"),
    "reference": ("doc_type", "reference_id", "reference_url"),
    "note": (),
    "modification": ("modification_type",),
    "relationship": ("relationship_type", "related_substance_name", "qualification"),
    # Generic catch-all for any chunk_type not in the table.
    "other": (),
}


def _escape_cell(value: Any) -> str:
    """Return a markdown-table-safe scalar string."""
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _metadata_lookup(part: Dict[str, Any]) -> Dict[str, Any]:
    """Return the chunk metadata merged from the parent context part.

    ``reconstruct_parent_context`` only stores the chunk's per-chunk
    metadata into ``text_parts`` via the parent context's
    ``metadata_unified`` aggregate. To get per-row metadata, we stash
    a ``metadata`` key on each part at reconstruction time. Older
    callers that pre-date that field get an empty dict.
    """
    meta = part.get("metadata")
    if isinstance(meta, dict):
        return meta
    return {}


def _row_value(part: Dict[str, Any], column: str) -> Any:
    """Resolve a column name to a value for one chunk's text-part.

    The chunker stashes most structured fields directly on the chunk's
    ``metadata_json``; ``reconstruct_parent_context`` exposes them
    through the per-part ``metadata`` dict (when present). For a few
    columns (e.g. ``value`` for properties) we fall back to parsing
    the rendered chunk text.
    """
    meta = _metadata_lookup(part)
    if column in meta:
        return meta[column]
    # Heuristic fallbacks for the most common chunks.
    if column == "value" and meta.get("chunk_type") == "property":
        text = part.get("text") or ""
        # The chunker writes ``Value: <formatted>`` as one of the lines.
        for line in text.splitlines():
            if line.startswith("Value:"):
                return line.split(":", 1)[1].strip()
    if column == "modification_type":
        text = part.get("text") or ""
        for line in text.splitlines():
            if line.startswith("Modification Type:"):
                return line.split(":", 1)[1].strip()
    return ""


def _render_rows_as_table(
    rows: List[Dict[str, Any]],
    *,
    include_text_parts: bool,
    limit: int,
) -> List[str]:
    """Render a list of parent-context text-parts as a markdown table.

    The columns are derived from the rows' ``chunk_type`` (see
    :data:`_TABLE_COLUMNS`). Rows beyond ``limit`` are dropped; the
    caller is expected to surface the count elsewhere if it matters.
    Empty inputs return an empty list — callers should skip emitting
    a section header in that case.
    """
    if not rows:
        return []
    rows = rows[:limit]
    chunk_type = rows[0].get("chunk_type") or "other"
    columns = list(_TABLE_COLUMNS.get(chunk_type, ()))
    if include_text_parts:
        columns.append("text")

    md: List[str] = []
    headers = [col.replace("_", " ").title() for col in columns]
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = []
        for col in columns:
            if col == "text":
                value = row.get("text") or ""
            else:
                value = _row_value(row, col)
            cells.append(_escape_cell(value))
        md.append("| " + " | ".join(cells) + " |")
    return md
