"""
GSRS MCP Server - Aggregation Service
Handles counting/collecting queries like "How many identifiers has Ibuprofen?"
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.config import Settings, settings
from app.models.db import DBQueryResult, VectorDocument
from app.services.code_systems import get_identifier_field_names


@dataclass
class AggregationResult:
    """Structured result from an aggregation query."""
    substance_name: str
    aggregation_type: str  # "identifiers", "names", "relationships", "general"
    items: List[Dict[str, Any]]
    total_count: int
    raw_text_summary: str = ""


class AggregationService:
    """
    Extracts structured aggregations from retrieved documents.
    - Counts identifiers (CAS, UNII, etc.)
    - Collects names/synonyms
    - Gathers relationships
    - Builds a summary
    """

    def __init__(self, app_settings: Settings = settings):
        self.identifier_field_names = get_identifier_field_names(app_settings)

    def aggregate(
        self,
        candidates: List[DBQueryResult],
        query: str,
        intent: str,
    ) -> AggregationResult:
        """
        Perform aggregation over retrieved candidates.

        Args:
            candidates: Ranked DBQueryResult objects
            query: Original query
            intent: e.g., "aggregation_identifiers"

        Returns:
            AggregationResult with collected items
        """
        # Determine aggregation type from intent
        if "identifier" in intent:
            agg_type = "identifiers"
        elif "name" in intent:
            agg_type = "names"
        elif "relationship" in intent:
            agg_type = "relationships"
        else:
            agg_type = "general"

        # Collect items from all candidates
        items = []
        substance_name = ""
        seen_codes: Set[str] = set()
        seen_names: Set[str] = set()
        authoritative_total_count: Optional[int] = None

        for r in candidates:
            doc = r.document
            metadata = doc.metadata_json or {}
            if not substance_name:
                substance_name = self._extract_substance_name(metadata)

            if agg_type == "identifiers":
                codes = self._extract_codes(metadata)
                for code in codes:
                    code_key = f"{code.get('type', '')}:{code.get('code', '')}"
                    if code_key not in seen_codes:
                        seen_codes.add(code_key)
                        items.append(code)

            elif agg_type == "names":
                names = self._extract_names(metadata)
                for name in names:
                    name_key = self._normalize_name_key(name)
                    if name_key not in seen_names:
                        seen_names.add(name_key)
                        items.append({"name": name})

            elif agg_type == "relationships":
                rels = self._extract_relationships(metadata, doc.text)
                for rel in rels:
                    items.append(rel)

            else:
                # General: collect key facts from text
                items.append({
                    "section": doc.section,
                    "text": doc.text[:300],
                })

        if agg_type == "names":
            authoritative_total_count = self._extract_authoritative_name_count(candidates)

        total_count = authoritative_total_count if authoritative_total_count is not None else len(items)

        # Build summary
        summary = self._build_summary(substance_name, agg_type, items, total_count=total_count)

        return AggregationResult(
            substance_name=substance_name or "Unknown",
            aggregation_type=agg_type,
            items=items,
            total_count=total_count,
            raw_text_summary=summary,
        )

    def _extract_substance_name(self, metadata: Dict) -> str:
        """Extract the best available substance display name from chunk metadata."""
        display_name = self._extract_display_name(metadata.get("names", []))
        if display_name:
            return display_name

        for key in ["canonical_name", "entity_name", "substance_name", "name"]:
            value = metadata.get(key)
            if value:
                return str(value)
        return ""

    def _extract_display_name(self, names: Any) -> str:
        """Return the GSRS display name from a names collection when present."""
        if not isinstance(names, list):
            return ""

        for entry in names:
            if not isinstance(entry, dict):
                continue
            is_display = entry.get("displayName", entry.get("display_name", False))
            if isinstance(is_display, str):
                is_display = is_display.strip().lower() == "true"
            if is_display and entry.get("name"):
                return str(entry["name"])
        return ""

    def _extract_codes(self, metadata: Dict) -> List[Dict[str, str]]:
        """Extract all identifier codes from metadata."""
        codes = []
        codes_raw = metadata.get("codes", [])
        if isinstance(codes_raw, list):
            for code in codes_raw:
                if isinstance(code, dict):
                    codes.append({
                        "type": code.get("codeSystem", code.get("type", "")),
                        "code": code.get("code", ""),
                        "url": code.get("url", ""),
                    })
                elif isinstance(code, str):
                    codes.append({"type": "unknown", "code": code})

        direct_code = metadata.get("code")
        if direct_code:
            code_type = (
                metadata.get("code_system")
                or metadata.get("codeSystem")
                or metadata.get("type")
                or metadata.get("code_type")
                or "unknown"
            )
            codes.append({
                "type": str(code_type),
                "code": str(direct_code),
                "url": str(metadata.get("url", "")),
            })

        # Also check direct code fields
        for key in self.identifier_field_names:
            val = metadata.get(key)
            if val:
                codes.append({"type": key.upper(), "code": str(val)})

        for key in ["reliable_codes", "all_codes"]:
            bucket = metadata.get(key, {})
            if isinstance(bucket, dict):
                for code_system, code in bucket.items():
                    if code:
                        codes.append({"type": str(code_system), "code": str(code)})

        return codes

    def _extract_names(self, metadata: Dict) -> List[str]:
        """Extract all names/synonyms from metadata."""
        names = []
        section_hint = str(metadata.get("section", "")).lower()
        group_type = str(metadata.get("group_type", "")).lower()
        entity_type = str(metadata.get("entity_type", "")).lower()
        hierarchy = " ".join(str(item).lower() for item in metadata.get("hierarchy", []))

        names_raw = metadata.get("names", [])
        if isinstance(names_raw, list):
            canonical = self._extract_substance_name(metadata)
            if canonical:
                names.append(canonical)
            for name in names_raw:
                if isinstance(name, dict):
                    name_text = name.get("name", "")
                    if name_text:
                        names.append(str(name_text))
                elif isinstance(name, str):
                    names.append(name)

        if entity_type == "name" or "name" in section_hint or "names" in hierarchy:
            # Core name chunks are summaries. Their exact_match_terms are useful
            # for search, but can contain aliases/variants and should not be
            # double-counted with name_batch chunks.
            if section_hint == "core_names" or group_type == "core_names":
                return names

            exact_terms = metadata.get("exact_match_terms", [])
            if isinstance(exact_terms, list):
                names.extend(str(term) for term in exact_terms if term)

        if not names and not any([section_hint, group_type, entity_type]):
            canonical = self._extract_substance_name(metadata)
            if canonical:
                names.append(canonical)

        return names

    def _extract_authoritative_name_count(self, candidates: List[DBQueryResult]) -> Optional[int]:
        """Use chunker-provided name counts when present instead of search terms."""
        core_counts: List[int] = []
        batch_counts: List[int] = []

        for result in candidates:
            metadata = result.document.metadata_json or {}
            raw_count = metadata.get("name_count")
            if not isinstance(raw_count, int) or raw_count < 0:
                continue

            section_hint = str(metadata.get("section", "")).lower()
            group_type = str(metadata.get("group_type", "")).lower()
            if section_hint == "core_names" or group_type == "core_names":
                core_counts.append(raw_count)
            elif "name" in section_hint or "name" in group_type:
                batch_counts.append(raw_count)

        if core_counts:
            return max(core_counts)
        if batch_counts:
            return sum(batch_counts)
        return None

    def _normalize_name_key(self, name: str) -> str:
        """Normalize display names just enough to deduplicate casing/spacing."""
        return " ".join(str(name).split()).casefold()

    def _extract_relationships(self, metadata: Dict, text: str) -> List[Dict[str, str]]:
        """Extract relationship information from metadata and text."""
        relationships = []

        # Check metadata for relationship fields
        for key in ["metabolites", "impurities", "binders", "transporters", "targets"]:
            rel_data = metadata.get(key, [])
            if isinstance(rel_data, list) and rel_data:
                for item in rel_data:
                    if isinstance(item, dict):
                        relationships.append({
                            "type": key,
                            "data": str(item),
                        })

        return relationships

    def _build_summary(
        self,
        substance_name: str,
        agg_type: str,
        items: List,
        total_count: Optional[int] = None,
    ) -> str:
        """Build a human-readable summary of the aggregation."""
        count = total_count if total_count is not None else len(items)

        if agg_type == "identifiers":
            if count == 0:
                return f"No identifier codes found for {substance_name}."
            summary = f"{substance_name} has {count} identifier code(s):\n"
            for item in items:
                code_type = item.get("type", "unknown")
                code_val = item.get("code", "N/A")
                summary += f"  - {code_type}: {code_val}\n"
            return summary

        elif agg_type == "names":
            if count == 0:
                return f"No names found for {substance_name}."
            summary = f"{substance_name} has {count} name(s)/synonym(s):\n"
            for item in items:
                summary += f"  - {item.get('name', 'N/A')}\n"
            return summary

        elif agg_type == "relationships":
            if count == 0:
                return f"No relationships found for {substance_name}."
            summary = f"{substance_name} has {count} relationship(s):\n"
            for item in items:
                summary += f"  - {item.get('type', 'N/A')}: {item.get('data', 'N/A')[:100]}\n"
            return summary

        else:
            if count == 0:
                return f"No information found for {substance_name}."
            return f"Found {count} information item(s) for {substance_name}."
