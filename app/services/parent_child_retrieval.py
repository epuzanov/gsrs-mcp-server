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
