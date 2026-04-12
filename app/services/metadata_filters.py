"""
GSRS MCP Server - Metadata Filter Builder
Merges request-provided and inferred filters, normalizes to storage schema.
"""
from typing import Any, Dict, List, Optional


class MetadataFilterBuilder:
    """
    Builds metadata filters for retrieval by merging:
    - Request-provided filters (AskRequest.filters)
    - Request substance_classes
    - Request sections
    - Rewrite-derived filters
    """

    def build(
        self,
        request_filters: Optional[Dict[str, Any]] = None,
        substance_classes: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        inferred_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Merge all filter sources into a unified filter dict.
        Compatible with current VectorDatabaseService.similarity_search filters.
        """
        merged: Dict[str, Any] = {}

        # Start with request-provided filters
        if request_filters:
            merged.update(request_filters)

        # Merge substance_classes
        all_classes = self._merge_list(
            merged.get("substance_classes"),
            substance_classes,
            (inferred_filters or {}).get("substance_classes"),
        )
        if all_classes:
            merged["substance_classes"] = all_classes

        # Merge sections
        all_sections = self._merge_list(
            merged.get("sections"),
            sections,
            (inferred_filters or {}).get("sections"),
        )
        if all_sections:
            merged["sections"] = all_sections

        # Merge document_id filters (exact match)
        doc_ids = self._merge_list(
            merged.get("document_id"),
            (inferred_filters or {}).get("document_id"),
        )
        if doc_ids:
            merged["document_id"] = doc_ids

        # Clean up: remove empty values
        return {k: v for k, v in merged.items() if v}

    def _merge_list(self, *sources: Any) -> List[str]:
        """Merge list sources, deduplicating while preserving order."""
        seen = set()
        result = []
        for source in sources:
            if source is None:
                continue
            if isinstance(source, str):
                source = [source]
            for item in source:
                item_str = str(item)
                if item_str not in seen:
                    seen.add(item_str)
                    result.append(item_str)
        return result
