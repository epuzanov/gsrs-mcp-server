"""
GSRS MCP Server - Lexical Retriever
Keyword/exact-match retrieval for identifiers, aliases, and section-specific terminology.
"""
import re
from typing import Any, Dict, List, Optional

from app.models.db import DBQueryResult, VectorDocument


class LexicalRetriever:
    """
    Lightweight lexical retrieval fallback.
    Supports exact term matching, token overlap scoring,
    and metadata-prefiltered substring matching.
    """

    def __init__(self, top_k: int = 40):
        self.top_k = top_k

    def search(
        self,
        query: str,
        candidates: List[DBQueryResult],
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQueryResult]:
        """
        Perform lexical retrieval over candidates.
        Returns scored candidates sorted by lexical score (descending).

        If `candidates` is provided, scores them lexically.
        If candidates is empty, returns empty list (lexical retrieval
        is meant to complement semantic retrieval, not replace it).
        """
        if not candidates:
            return []

        terms = self._tokenize(query)
        if not terms:
            return []

        scored = []
        for r in candidates:
            score = self._score_document(r.document, terms, filters)
            if score > 0:
                scored.append(DBQueryResult(r.document, score))

        # Sort by score descending using DBQueryResult comparison
        scored.sort(reverse=True)
        return scored[: self.top_k]

    def score_candidates(
        self,
        query: str,
        candidates: List[DBQueryResult],
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQueryResult]:
        """Score candidates with lexical relevance."""
        return self.search(query, candidates, filters)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase terms, removing punctuation."""
        text = text.lower().strip()
        # Split on whitespace and punctuation
        tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text)
        # Filter very short tokens
        return [t for t in tokens if len(t) >= 2]

    def _score_document(
        self,
        doc: VectorDocument,
        terms: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Score a document based on term overlap."""
        text = doc.text.lower()
        metadata = doc.metadata_json or {}

        # Build search text: chunk text + metadata fields
        search_text = self._build_search_text(doc, metadata)

        score = 0.0
        matched_terms = 0

        for term in terms:
            term_score = 0.0

            # Exact match in search text
            if term in search_text:
                term_score = 1.0
            # Substring match
            elif any(term in token for token in search_text.split()):
                term_score = 0.5

            if term_score > 0:
                matched_terms += 1
                score += term_score

        # Normalize by number of terms
        if terms:
            score = score / len(terms)

        # Boost for exact phrase match
        phrase = " ".join(terms)
        if phrase.lower() in search_text.lower():
            score += 0.5

        return score

    def _build_search_text(self, doc: VectorDocument, metadata: Dict) -> str:
        """Build comprehensive search text from document and metadata."""
        parts = [doc.text]

        # Add metadata fields that are useful for lexical matching
        for key in ["canonical_name", "chunk_type", "section", "source_url"]:
            val = metadata.get(key)
            if val:
                parts.append(str(val))

        # Add any names from metadata
        names = metadata.get("names", [])
        if isinstance(names, list):
            parts.extend(str(n) for n in names)

        # Add codes from metadata
        codes = metadata.get("codes", [])
        if isinstance(codes, list):
            parts.extend(str(c) for c in codes)

        return " ".join(str(p) for p in parts if p)
