"""
GSRS MCP Server - Query Rewrite Service
Normalizes questions, detects intent, and generates multiple rewrites for hybrid retrieval.
"""
import re
from dataclasses import dataclass
from typing import Dict, List

from app.config import Settings, settings
from app.services.code_systems import (
    get_identifier_mention_patterns,
)

# Relationship keywords
_RELATIONSHIP_KEYWORDS = [
    "metabolite", "impurity", "binder", "transporter", "target",
    "salt", "formulation", "derivative", "precursor", "analog",
    "conjugate", "complex", "interaction",
]

# Section-indicating keywords
_SECTION_KEYWORDS = [
    "protein", "nucleic acid", "mixture", "polymer", "strain",
    "component", "constituent", "part", "substance",
    "name", "structure", "property", "activity",
    "indication", "pharmacology", "mechanism",
]

# Known GSRS substance class keywords
_SUBSTANCE_CLASS_KEYWORDS = [
    "chemical", "protein", "nucleic acid", "polymer", "mixture",
    "structurally diverse", "cell", "gene therapy", "oligonucleotide",
    "vaccine", "allergen", "toxin",
]


# Aggregation keywords that signal counting/collecting intent
_AGGREGATION_KEYWORDS = [
    "how many", "count", "list", "all", "total", "number of",
    "enumerate", "collect", "gather", "summary", "overview",
]


@dataclass
class RewriteResult:
    canonical_query: str
    rewrites: List[str]
    filters: Dict
    intent: str


class QueryRewriteService:
    """
    Rewrites user queries for better retrieval.
    - Normalizes the query
    - Detects intent (identifier lookup, relationship, property, etc.)
    - Generates multiple rewrites optimized for GSRS content
    - Infers likely metadata filters
    """

    def __init__(self, app_settings: Settings = settings):
        self.app_settings = app_settings
        self.identifier_mention_patterns = get_identifier_mention_patterns(app_settings)

    def rewrite(self, query: str) -> RewriteResult:
        query_lower = query.lower().strip()
        canonical = self._normalize(query)
        intent = self._detect_intent(query, query_lower)
        filters = self._infer_filters(query, query_lower)
        rewrites = self._generate_rewrites(query, query_lower, intent)

        return RewriteResult(
            canonical_query=canonical,
            rewrites=rewrites,
            filters=filters,
            intent=intent,
        )

    def _normalize(self, query: str) -> str:
        """Normalize query text: strip, lowercase, collapse whitespace."""
        return re.sub(r"\s+", " ", query.strip().lower())

    def _detect_intent(self, query: str, query_lower: str) -> str:
        """Detect the query intent."""
        has_identifier_reference = self._contains_identifier_reference(query)
        # Check for aggregation/count queries first (higher priority)
        if any(kw in query_lower for kw in _AGGREGATION_KEYWORDS):
            # e.g., "How many identifiers has Ibuprofen?"
            if has_identifier_reference or any(kw in query_lower for kw in ["code", "identifier", "id"]):
                return "aggregation_identifiers"
            if any(kw in query_lower for kw in ["name", "names", "called", "synonym"]):
                return "aggregation_names"
            if any(kw in query_lower for kw in _RELATIONSHIP_KEYWORDS):
                return "aggregation_relationships"
            return "aggregation_general"

        # Check for identifier lookup
        if has_identifier_reference or any(kw in query_lower for kw in ["code", "identifier", "id"]):
            if any(w in query_lower for w in ["what is", "what are", "find", "lookup", "get"]):
                return "identifier_lookup"
            return "identifier_query"

        # Check for relationship queries
        if any(kw in query_lower for kw in _RELATIONSHIP_KEYWORDS):
            return "relationship_query"

        # Check for section-specific queries
        if any(kw in query_lower for kw in _SECTION_KEYWORDS):
            return "section_query"

        # Check for substance class queries
        if any(kw in query_lower for kw in _SUBSTANCE_CLASS_KEYWORDS):
            return "substance_class_query"

        # Default
        return "general"

    def _infer_filters(self, query: str, query_lower: str) -> Dict:
        """Infer metadata filters from the query."""
        filters: Dict = {}

        # Detect substance class filters
        substance_classes = []
        for kw in _SUBSTANCE_CLASS_KEYWORDS:
            if kw in query_lower:
                # Map common aliases
                if kw == "structurally diverse":
                    substance_classes.append("Structurally Diverse")
                elif kw == "gene therapy":
                    substance_classes.append("Gene Therapy")
                elif kw == "nucleic acid":
                    substance_classes.append("Nucleic Acid")
                else:
                    substance_classes.append(kw.title())

        if substance_classes:
            filters["substance_classes"] = substance_classes

        # Detect section filters
        sections = []
        if any(w in query_lower for w in ["name", "called", "nomenclature"]):
            sections.append("names")
        if self._contains_identifier_reference(query) or any(w in query_lower for w in ["code", "identifier"]):
            sections.append("codes")
        if any(w in query_lower for w in ["structure", "molecular", "formula", "smiles", "inchi"]):
            sections.append("structure")
        if any(w in query_lower for w in ["property", "physical", "molecular weight", "melting", "boiling"]):
            sections.append("properties")
        if any(w in query_lower for w in ["activity", "pharmacology", "mechanism", "indication", "therapeutic"]):
            sections.append("activity")
        if any(w in query_lower for w in ["reference", "source", "citation"]):
            sections.append("references")

        if sections:
            filters["sections"] = sections

        return filters

    def _generate_rewrites(self, query: str, query_lower: str, intent: str) -> List[str]:
        """Generate multiple query rewrites for better retrieval coverage."""
        rewrites = [query]  # Always include original

        if intent.startswith("aggregation_"):
            # Aggregation queries need all relevant chunks from the substance
            substance = self._extract_substance_name(query_lower)
            if "identifier" in intent:
                rewrites.append(f"codes {substance}")
                rewrites.append(f"{substance} all codes identifiers")
                rewrites.append(f"{substance} identifiers list")
                rewrites.append(f"all codes for {substance}")
            elif "name" in intent:
                rewrites.append(f"names {substance}")
                rewrites.append(f"{substance} all names synonyms")
                rewrites.append(f"{substance} names list")
            elif "relationship" in intent:
                rewrites.append(f"relationships {substance}")
                rewrites.append(f"{substance} all relationships")
            else:
                rewrites.append(f"{substance} all information")
                rewrites.append(f"{substance} complete record")

        elif intent == "identifier_lookup":
            # Extract substance name (rough heuristic: words after preposition)
            substance = self._extract_substance_name(query_lower)
            identifiers = self._detect_identifiers(query)

            for ident in identifiers:
                rewrites.append(f"{ident} {substance}")
                rewrites.append(f"{substance} {ident}")
                rewrites.append(f"identifier {ident} {substance}")
                rewrites.append(f"{ident} code {substance}")

            rewrites.append(f"code {substance}")
            rewrites.append(f"{substance} code")

        elif intent == "identifier_query":
            substance = self._extract_substance_name(query_lower)
            identifiers = self._detect_identifiers(query)

            for ident in identifiers:
                rewrites.append(f"{ident} {substance}")
                rewrites.append(f"{substance} {ident}")

        elif intent == "relationship_query":
            substance = self._extract_substance_name(query_lower)
            relationships = self._detect_relationships(query_lower)

            for rel in relationships:
                rewrites.append(f"{substance} {rel}")
                rewrites.append(f"{rel} of {substance}")
                rewrites.append(f"{substance} {rel} relationship")

        elif intent == "section_query":
            substance = self._extract_substance_name(query_lower)
            sections = self._detect_sections(query_lower)

            for sec in sections:
                rewrites.append(f"{substance} {sec}")
                rewrites.append(f"{sec} {substance}")

        elif intent == "substance_class_query":
            substance = self._extract_substance_name(query_lower)
            classes = self._detect_substance_classes(query_lower)

            for cls in classes:
                rewrites.append(f"{cls} {substance}")
                rewrites.append(f"{substance} {cls}")

        else:
            # General: add keyword-focused variants
            substance = self._extract_substance_name(query_lower)
            if substance and substance.lower() not in query_lower[:20]:
                rewrites.append(substance)

        # Deduplicate while preserving order
        seen = set()
        unique_rewrites = []
        for r in rewrites:
            rl = r.lower().strip()
            if rl and rl not in seen:
                seen.add(rl)
                unique_rewrites.append(r)

        return unique_rewrites[:10]

    def _extract_substance_name(self, query_lower: str) -> str:
        """Extract the likely substance name from the query."""
        # Remove common question patterns
        patterns = [
            r"what\s+is\s+(?:the\s+)?",
            r"what\s+are\s+(?:the\s+)?",
            r"tell\s+me\s+about\s+",
            r"describe\s+",
            r"find\s+(?:the\s+)?",
            r"get\s+(?:the\s+)?",
            r"lookup\s+",
            r"search\s+for\s+",
        ]
        name = query_lower
        for pat in patterns:
            name = re.sub(pat, "", name).strip()

        # Remove trailing question marks and whitespace
        name = name.rstrip("?").strip()

        # Return the longest meaningful phrase
        return name if name else query_lower

    def _detect_identifiers(self, query_text: str) -> List[str]:
        """Detect identifier types mentioned in the query."""
        found = []
        for label, pattern in self.identifier_mention_patterns.items():
            if pattern.search(query_text):
                found.append(label)
        return found if found else ["code"]

    def _contains_identifier_reference(self, query: str) -> bool:
        query_lower = query.lower()
        if any(pattern.search(query) for pattern in self.identifier_mention_patterns.values()):
            return True
        return any(keyword in query_lower for keyword in ["code", "identifier"])

    def _detect_relationships(self, query_lower: str) -> List[str]:
        """Detect relationship types mentioned in the query."""
        found = []
        for kw in _RELATIONSHIP_KEYWORDS:
            if kw in query_lower:
                found.append(kw)
        return found if found else []

    def _detect_sections(self, query_lower: str) -> List[str]:
        """Detect section references in the query."""
        found = []
        section_map = {
            "name": "names",
            "names": "names",
            "nomenclature": "names",
            "code": "codes",
            "codes": "codes",
            "structure": "structure",
            "molecular": "structure",
            "formula": "structure",
            "property": "properties",
            "properties": "properties",
            "activity": "activity",
            "pharmacology": "activity",
            "mechanism": "activity",
            "indication": "activity",
            "reference": "references",
            "references": "references",
        }
        for kw, sec in section_map.items():
            if kw in query_lower and sec not in found:
                found.append(sec)
        return found

    def _detect_substance_classes(self, query_lower: str) -> List[str]:
        """Detect substance class references in the query."""
        found = []
        for kw in _SUBSTANCE_CLASS_KEYWORDS:
            if kw in query_lower:
                if kw == "structurally diverse":
                    found.append("Structurally Diverse")
                elif kw == "gene therapy":
                    found.append("Gene Therapy")
                elif kw == "nucleic acid":
                    found.append("Nucleic Acid")
                else:
                    found.append(kw.title())
        return found
