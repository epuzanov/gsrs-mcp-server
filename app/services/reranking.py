"""
GSRS MCP Server - Reranker Service
Heuristic-based reranking with optional LLM rerank mode.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from app.models.db import VectorDocument


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
    ):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.metadata_weight = metadata_weight

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

        scored = []
        for doc, hybrid_score in candidates:
            score = self._score_document(doc, query_terms, hybrid_score, filters)
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
    ) -> float:
        """Compute rerank score for a document."""
        metadata = doc.metadata_json or {}

        # Semantic component (from hybrid retrieval)
        semantic_score = hybrid_score

        # Lexical component: term overlap
        lexical_score = self._compute_lexical_score(doc, query_terms)

        # Metadata component
        metadata_score = self._compute_metadata_score(doc, query_terms, filters)

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

        doc_text = doc.text.lower()
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
    ) -> float:
        """Compute metadata-based score with boosts."""
        metadata = doc.metadata_json or {}
        score = 0.0
        boosts = 0

        # Exact identifier match boost
        identifier_boost = self._check_identifier_match(metadata, query_terms)
        if identifier_boost > 0:
            score += identifier_boost
            boosts += 1

        # Exact alias/name match boost
        name_boost = self._check_name_match(metadata, query_terms)
        if name_boost > 0:
            score += name_boost
            boosts += 1

        # Section match boost
        section_boost = self._check_section_match(doc, filters)
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

    def _check_identifier_match(self, metadata: Dict, query_terms: List[str]) -> float:
        """Check for exact identifier matches (CAS, UNII, etc.)."""
        identifiers = self._extract_identifiers(metadata)
        for ident in identifiers:
            ident_lower = ident.lower()
            for term in query_terms:
                if term == ident_lower or term in ident_lower or ident_lower in term:
                    return 2.0  # Strong boost for exact identifier match
        return 0.0

    def _check_name_match(self, metadata: Dict, query_terms: List[str]) -> float:
        """Check for exact name/alias matches."""
        names = self._extract_names(metadata)
        for name in names:
            name_lower = name.lower()
            for term in query_terms:
                if term == name_lower or term in name_lower or name_lower in term:
                    return 1.5  # Boost for name match
        return 0.0

    def _check_section_match(self, doc: VectorDocument, filters: Optional[Dict]) -> float:
        """Check if document section matches filter."""
        if filters and "sections" in filters:
            sections = filters["sections"]
            if isinstance(sections, list):
                if doc.section in sections:
                    return 1.0
            elif doc.section == sections:
                return 1.0
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
        for key in ["cas", "unii", "pubchem", "drugbank", "chembl", "rxcui"]:
            val = metadata.get(key)
            if val:
                identifiers.append(str(val))

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

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        text = text.lower().strip()
        tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text)
        return [t for t in tokens if len(t) >= 2]
