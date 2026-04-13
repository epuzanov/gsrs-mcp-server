"""
GSRS MCP Server - Reranker Service
Heuristic-based reranking with optional LLM rerank mode.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings, settings
from app.models.db import VectorDocument
from app.services.code_systems import get_identifier_field_names, get_identifier_mention_patterns


class RerankerService:
    """
    Reranks retrieval candidates using heuristics:
    - Exact identifier match boost
    - Exact alias/name match boost
    - Section match boost
    - Substance class match boost
    - Lexical score component
    - Semantic score component
    - Term overlap with rewritten queries
    """

    def __init__(
        self,
        semantic_weight: float = 0.4,
        lexical_weight: float = 0.3,
        metadata_weight: float = 0.3,
        app_settings: Settings = settings,
    ):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.metadata_weight = metadata_weight
        self.identifier_field_names = get_identifier_field_names(app_settings)
        self.identifier_mention_patterns = get_identifier_mention_patterns(app_settings)

    def rerank(
        self,
        candidates: List[Tuple[VectorDocument, float]],
        query: str,
        rewritten_queries: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """
        Rerank candidates and return sorted by normalized rerank score.

        Args:
            candidates: List of (document, hybrid_score) tuples
            query: Original query
            rewritten_queries: List of rewritten queries
            filters: Applied filters

        Returns:
            Reranked list of (document, normalized_score) tuples
        """
        if not candidates:
            return []

        all_queries = [query] + (rewritten_queries or [])
        query_terms = self._tokenize(" ".join(all_queries))
        query_signals = self._extract_query_signals(query, rewritten_queries, filters)

        scored = []
        for doc, hybrid_score in candidates:
            score = self._score_document(doc, query_terms, hybrid_score, filters, query_signals)
            scored.append((doc, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Normalize scores to [0, 1]
        if scored:
            max_score = max(s for _, s in scored)
            if max_score > 0:
                scored = [(doc, s / max_score) for doc, s in scored]

        return scored

    def _score_document(
        self,
        doc: VectorDocument,
        query_terms: List[str],
        hybrid_score: float,
        filters: Optional[Dict[str, Any]] = None,
        query_signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute rerank score for a document."""
        metadata = doc.metadata_json or {}

        # Semantic component (from hybrid retrieval)
        semantic_score = hybrid_score

        # Lexical component: term overlap
        lexical_score = self._compute_lexical_score(doc, query_terms)

        # Metadata component
        metadata_score = self._compute_metadata_score(doc, query_terms, filters, query_signals)

        # Combined score
        total = (
            self.semantic_weight * semantic_score
            + self.lexical_weight * lexical_score
            + self.metadata_weight * metadata_score
        )

        return total

    def _compute_lexical_score(self, doc: VectorDocument, query_terms: List[str]) -> float:
        """Compute lexical overlap score."""
        if not query_terms:
            return 0.0

        metadata = doc.metadata_json or {}
        doc_text = " ".join(
            [
                doc.text.lower(),
                str(metadata.get("canonical_name", "")).lower(),
                " ".join(name.lower() for name in self._extract_names(metadata)),
                " ".join(identifier.lower() for identifier in self._extract_identifiers(metadata)),
                str(doc.section).lower(),
            ]
        )
        matched = 0

        for term in query_terms:
            if term in doc_text:
                matched += 1

        return matched / len(query_terms)

    def _compute_metadata_score(
        self,
        doc: VectorDocument,
        query_terms: List[str],
        filters: Optional[Dict[str, Any]] = None,
        query_signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute metadata-based score with boosts."""
        metadata = doc.metadata_json or {}
        score = 0.0
        boosts = 0

        # Exact identifier match boost
        identifier_boost = self._check_identifier_match(metadata, query_terms, doc, query_signals)
        if identifier_boost > 0:
            score += identifier_boost
            boosts += 1

        # Exact alias/name match boost
        name_boost = self._check_name_match(metadata, query_terms, doc, query_signals)
        if name_boost > 0:
            score += name_boost
            boosts += 1

        # Section match boost
        section_boost = self._check_section_match(doc, filters, query_signals)
        if section_boost > 0:
            score += section_boost
            boosts += 1

        # Substance class match boost
        class_boost = self._check_substance_class_match(metadata, filters)
        if class_boost > 0:
            score += class_boost
            boosts += 1

        # Average if there were boosts
        if boosts > 0:
            score = min(score / (boosts * 2.0), 1.0)

        return score

    def _check_identifier_match(
        self,
        metadata: Dict,
        query_terms: List[str],
        doc: VectorDocument,
        query_signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Check for exact identifier matches (CAS, UNII, etc.)."""
        identifiers = self._extract_identifiers(metadata)
        exact_identifiers = set((query_signals or {}).get("exact_identifiers", []))
        if exact_identifiers:
            for ident in identifiers:
                ident_lower = ident.lower()
                if ident_lower in exact_identifiers:
                    return 4.0 if str(doc.section) == "codes" else 3.5

        for ident in identifiers:
            ident_lower = ident.lower()
            for term in query_terms:
                if term == ident_lower:
                    return 3.5 if str(doc.section) == "codes" else 3.0
                if len(term) >= 4 and (term in ident_lower or ident_lower in term):
                    return 2.0
        return 0.0

    def _check_name_match(
        self,
        metadata: Dict,
        query_terms: List[str],
        doc: VectorDocument,
        query_signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Check for exact name/alias matches."""
        names = self._extract_names(metadata)
        exact_name = (query_signals or {}).get("exact_name")
        if exact_name:
            exact_name_lower = exact_name.lower()
            for name in names:
                name_lower = name.lower()
                if name_lower == exact_name_lower:
                    return 3.0 if str(doc.section) in {"names", "root"} else 2.5

        for name in names:
            name_lower = name.lower()
            for term in query_terms:
                if term == name_lower or term in name_lower or name_lower in term:
                    return 1.5  # Boost for name match
        return 0.0

    def _check_section_match(
        self,
        doc: VectorDocument,
        filters: Optional[Dict],
        query_signals: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Check if document section matches filter."""
        if filters and "sections" in filters:
            sections = filters["sections"]
            if isinstance(sections, list):
                if doc.section in sections:
                    return 1.0
            elif doc.section == sections:
                return 1.0

        preferred_sections = (query_signals or {}).get("preferred_sections", [])
        if str(doc.section) in preferred_sections:
            return 0.8
        return 0.0

    def _check_substance_class_match(self, metadata: Dict, filters: Optional[Dict]) -> float:
        """Check if substance class matches filter."""
        if filters and "substance_classes" in filters:
            classes = filters["substance_classes"]
            if isinstance(classes, list):
                doc_classes = metadata.get("substance_classes", [])
                if isinstance(doc_classes, list):
                    for cls in classes:
                        if cls in doc_classes:
                            return 1.0
        return 0.0

    def _extract_identifiers(self, metadata: Dict) -> List[str]:
        """Extract identifiers from metadata."""
        identifiers = []
        # Check codes in metadata
        codes = metadata.get("codes", [])
        if isinstance(codes, list):
            for code in codes:
                if isinstance(code, dict):
                    code_text = code.get("code", "")
                    if code_text:
                        identifiers.append(str(code_text))
                elif isinstance(code, str):
                    identifiers.append(code)

        # Check metadata_json for code-related fields
        for key in [
            "uuid",
            "approvalID",
        ]:
            val = metadata.get(key)
            if val:
                identifiers.append(str(val))

        for key in self.identifier_field_names:
            val = metadata.get(key)
            if val:
                identifiers.append(str(val))

        structure = metadata.get("structure", {})
        if isinstance(structure, dict):
            for key in ["inchikey", "inchi", "smiles"]:
                val = structure.get(key)
                if val:
                    identifiers.append(str(val))

        for key in ["reliable_codes", "all_codes"]:
            values = metadata.get(key, {})
            if isinstance(values, dict):
                identifiers.extend(str(value) for value in values.values() if value)

        return identifiers

    def _extract_names(self, metadata: Dict) -> List[str]:
        """Extract names/aliases from metadata."""
        names = []

        # Canonical name
        canonical = metadata.get("canonical_name")
        if canonical:
            names.append(str(canonical))

        # Names list
        names_list = metadata.get("names", [])
        if isinstance(names_list, list):
            for name in names_list:
                if isinstance(name, dict):
                    name_text = name.get("name", "")
                    if name_text:
                        names.append(str(name_text))
                elif isinstance(name, str):
                    names.append(name)

        return names

    def _extract_query_signals(
        self,
        query: str,
        rewritten_queries: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        quoted_match = re.search(r'"([^"]+)"|\'([^\']+)\'', query)
        exact_name = None
        if quoted_match:
            exact_name = next(group for group in quoted_match.groups() if group)
        elif 0 < len(query.split()) <= 4 and not any(pattern.search(query) for pattern in self.identifier_mention_patterns.values()) and not any(
            token in query_lower for token in ["code", "identifier", "approval"]
        ):
            exact_name = query.strip().rstrip("?")

        exact_identifiers = set(
            token.lower()
            for token in re.findall(
                r"[a-z0-9]{2,}(?:[-_][a-z0-9]+)+|[0-9]{2,}-[0-9]{2,}-[0-9]",
                query_lower,
            )
        )

        preferred_sections: List[str] = []
        if any(pattern.search(query) for pattern in self.identifier_mention_patterns.values()) or any(
            keyword in query_lower for keyword in ["approval", "identifier", "code", "inchi", "smiles", "inchikey"]
        ):
            preferred_sections.append("codes")
        if exact_name is not None or any(keyword in query_lower for keyword in ["name", "called", "synonym"]):
            preferred_sections.append("names")
        if any(keyword in query_lower for keyword in ["structure", "smiles", "inchi", "inchikey", "formula"]):
            preferred_sections.append("structure")
        if filters and "sections" in filters:
            sections = filters["sections"]
            if isinstance(sections, list):
                preferred_sections.extend(str(section) for section in sections)
            elif sections:
                preferred_sections.append(str(sections))

        return {
            "exact_name": exact_name,
            "exact_identifiers": sorted(exact_identifiers),
            "preferred_sections": list(dict.fromkeys(preferred_sections)),
            "rewritten_queries": rewritten_queries or [],
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        text = text.lower().strip()
        tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text)
        return [t for t in tokens if len(t) >= 2]
