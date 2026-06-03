"""
Parent-Child Retrieval using Virtual Parent Reconstruction

Implements parent-child retrieval through virtual reconstruction of parent context
from existing chunk records. Parent identity is defined as (document_id, root_section),
and parent context is reconstructed by loading all chunks sharing the same parent identity.

This approach minimizes schema changes, migration effort, and reindexing while
delivering most of the benefits of parent-child retrieval.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import UUID

from app.models import VectorDocument, DBQueryResult


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

    def __init__(self, vector_db):
        """
        Initialize the parent context enricher.

        Args:
            vector_db: VectorDatabase instance for retrieving chunks
        """
        self.vector_db = vector_db

    @staticmethod
    def extract_root_section(chunk: VectorDocument) -> str:
        """
        Extract the root section identifier from a chunk.

        The root section represents the top-level section of a document.
        For most chunks, this is determined by their section field or metadata.

        Args:
            chunk: VectorDocument to extract root section from

        Returns:
            Root section identifier (e.g., "root", "compound", "protein")
        """
        # Priority order for determining root section:
        # 1. Explicit root_section in metadata
        metadata = chunk.metadata_json or {}
        if "root_section" in metadata:
            return str(metadata["root_section"])

        # 2. Hierarchy information (parent_section)
        if "hierarchy" in metadata and isinstance(metadata["hierarchy"], dict):
            parent_section = metadata["hierarchy"].get("parent_section")
            if parent_section:
                return str(parent_section)

        # 3. Use section if it's a top-level section (root, compound, etc.)
        if chunk.section:
            # Simple heuristic: if section is "overview", it's the root section itself
            if chunk.section.lower() == "overview":
                return "overview"
            # Otherwise, look at metadata for the top-level parent
            # For now, treat the section as the root unless it's a subsection
            return chunk.section

        # Fallback to "overview"
        return "overview"

    def get_parent_identity(self, chunk: VectorDocument) -> ParentIdentity:
        """
        Get the virtual parent identity for a chunk.

        Args:
            chunk: VectorDocument to identify parent for

        Returns:
            ParentIdentity identifying the parent
        """
        root_section = self.extract_root_section(chunk)
        return ParentIdentity(
            document_id=chunk.document_id,
            root_section=root_section,
        )

    def reconstruct_parent_context(
        self,
        parent_identity: ParentIdentity,
        exclude_sections: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reconstruct parent context from chunks with the given parent identity.

        Loads all chunks sharing the same parent identity and builds a unified
        parent context document containing aggregated information.

        Args:
            parent_identity: The parent identity to reconstruct context for
            exclude_sections: Sections to exclude from parent context

        Returns:
            Dictionary containing parent context, or None if no chunks found
        """
        if exclude_sections is None:
            exclude_sections = set()

        # Retrieve all chunks for this document
        all_chunks = self.vector_db.get_documents(
            substance_uuid=parent_identity.document_id
        )

        if not all_chunks:
            return None

        # Filter chunks by root section
        parent_chunks = [
            chunk for chunk in all_chunks
            if self.extract_root_section(chunk) == parent_identity.root_section
            and chunk.section not in exclude_sections
        ]

        if not parent_chunks:
            return None

        # Build parent context from chunks
        parent_context = {
            "parent_identity": {
                "document_id": str(parent_identity.document_id),
                "root_section": parent_identity.root_section,
            },
            "num_chunks": len(parent_chunks),
            "sections_included": list(set(c.section for c in parent_chunks if c.section)),
            "text_parts": [],
            "metadata_unified": {},
        }

        # Aggregate text from all parent chunks
        for chunk in parent_chunks:
            if chunk.text:
                parent_context["text_parts"].append({
                    "section": chunk.section,
                    "text": chunk.text[:500],  # Limit to first 500 chars per chunk
                    "source_url": chunk.source_url,
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
        parent_text_limit: int = 1000,
    ) -> Dict[str, Any]:
        """
        Enrich a chunk with parent context information.

        Args:
            chunk: The chunk to enrich
            parent_context: Pre-computed parent context (if None, will be reconstructed)
            include_parent_text: Whether to include aggregated parent text
            parent_text_limit: Maximum character limit for parent text content

        Returns:
            Dictionary with chunk data and parent context
        """
        enriched = {
            "chunk": {
                "document_id": str(chunk.document_id),
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_url": chunk.source_url,
                "metadata": chunk.metadata_json or {},
            },
        }

        # Get or reconstruct parent context
        if parent_context is None:
            parent_identity = self.get_parent_identity(chunk)
            parent_context = self.reconstruct_parent_context(
                parent_identity,
                exclude_sections={chunk.section}  # Don't include the chunk's own section
            )

        if parent_context:
            enriched["parent_context"] = parent_context
            if include_parent_text and parent_context.get("text_parts"):
                # Aggregate parent text, respecting limit
                parent_text = "\n\n".join(
                    f"[{part['section']}] {part['text']}"
                    for part in parent_context["text_parts"][:5]  # Limit to first 5 parts
                )
                if len(parent_text) > parent_text_limit:
                    parent_text = parent_text[:parent_text_limit].rstrip() + "..."
                enriched["parent_text_summary"] = parent_text

        return enriched

    def enrich_search_results(
        self,
        results: List[DBQueryResult],
        include_parent_text: bool = True,
        parent_text_limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Enrich multiple search results with parent context.

        Reconstructs parent context for results, caching parent contexts to avoid
        redundant lookups when multiple results share the same parent.

        Args:
            results: List of DBQueryResult from vector search
            include_parent_text: Whether to include parent text summaries
            parent_text_limit: Maximum character limit for parent text

        Returns:
            List of enriched result dictionaries
        """
        # Cache parent contexts by parent identity to avoid redundant lookups
        parent_cache: Dict[ParentIdentity, Optional[Dict[str, Any]]] = {}

        enriched_results = []
        for result in results:
            chunk = result.document
            parent_identity = self.get_parent_identity(chunk)

            # Use cache or reconstruct
            if parent_identity not in parent_cache:
                parent_cache[parent_identity] = self.reconstruct_parent_context(
                    parent_identity,
                    exclude_sections={chunk.section}
                )

            parent_context = parent_cache[parent_identity]

            # Enrich the chunk
            enriched = self.enrich_chunk_with_parent(
                chunk=chunk,
                parent_context=parent_context,
                include_parent_text=include_parent_text,
                parent_text_limit=parent_text_limit,
            )
            enriched["score"] = result.score

            enriched_results.append(enriched)

        return enriched_results

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
        all_chunks = self.vector_db.get_documents(substance_uuid=document_id)
        if not all_chunks:
            return []

        parent_identities: Set[ParentIdentity] = set()
        for chunk in all_chunks:
            parent_identities.add(self.get_parent_identity(chunk))

        return sorted(parent_identities, key=lambda p: p.root_section)
