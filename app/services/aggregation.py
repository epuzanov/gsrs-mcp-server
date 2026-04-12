"""
GSRS MCP Server - Aggregation Service
Handles counting/collecting queries like "How many identifiers has Ibuprofen?"
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.models.db import VectorDocument


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

    def aggregate(
        self,
        candidates: List[Tuple[VectorDocument, float]],
        query: str,
        intent: str,
    ) -> AggregationResult:
        """
        Perform aggregation over retrieved candidates.

        Args:
            candidates: Ranked (document, score) tuples
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

        for doc, score in candidates:
            metadata = doc.metadata_json or {}
            if not substance_name:
                substance_name = metadata.get("canonical_name", "")

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
                    if name not in seen_names:
                        seen_names.add(name)
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

        # Build summary
        summary = self._build_summary(substance_name, agg_type, items)

        return AggregationResult(
            substance_name=substance_name or "Unknown",
            aggregation_type=agg_type,
            items=items,
            total_count=len(items),
            raw_text_summary=summary,
        )

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

        # Also check direct code fields
        for key in ["cas", "unii", "pubchem", "drugbank", "chembl", "rxcui"]:
            val = metadata.get(key)
            if val:
                codes.append({"type": key.upper(), "code": str(val)})

        return codes

    def _extract_names(self, metadata: Dict) -> List[str]:
        """Extract all names/synonyms from metadata."""
        names = []
        canonical = metadata.get("canonical_name")
        if canonical:
            names.append(str(canonical))

        names_raw = metadata.get("names", [])
        if isinstance(names_raw, list):
            for name in names_raw:
                if isinstance(name, dict):
                    name_text = name.get("name", "")
                    if name_text:
                        names.append(str(name_text))
                elif isinstance(name, str):
                    names.append(name)

        return names

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

    def _build_summary(self, substance_name: str, agg_type: str, items: List) -> str:
        """Build a human-readable summary of the aggregation."""
        count = len(items)

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
