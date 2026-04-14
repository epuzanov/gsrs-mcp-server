"""
GSRS MCP Server - Evidence Extractor
Selects the most useful chunks for answering, preserving citations.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.models.db import DBQueryResult, VectorDocument
from app.models.api import Citation


@dataclass
class EvidenceResult:
    """Extracted evidence with citations."""
    document: VectorDocument
    score: float
    citation: Citation
    snippet: Optional[str] = None


class EvidenceExtractor:
    """
    Selects the smallest useful set of chunks for answering.
    - Preserves citations
    - Optionally trims text spans/snippets
    - Passes weak evidence signal to abstention
    """

    def __init__(
        self,
        max_evidence_count: int = 10,
        max_snippet_length: int = 1000,
        max_chunks_per_section: int = 2,
        additional_evidence_score_ratio: float = 0.75,
        minimum_additional_score: float = 0.2,
    ):
        self.max_evidence_count = max_evidence_count
        self.max_snippet_length = max_snippet_length
        self.max_chunks_per_section = max_chunks_per_section
        self.additional_evidence_score_ratio = additional_evidence_score_ratio
        self.minimum_additional_score = minimum_additional_score

    def extract(
        self,
        candidates: List[DBQueryResult],
        query: str,
        intent: str = "general",
    ) -> List[EvidenceResult]:
        """
        Extract evidence from ranked candidates.

        Args:
            candidates: Ranked DBQueryResult objects from reranker
            query: Original query
            intent: Query intent

        Returns:
            List of EvidenceResult objects
        """
        if not candidates:
            return []

        results = []
        seen_chunks = set()
        section_counts: dict[str, int] = {}
        top_score = candidates[0].score

        for index, r in enumerate(candidates):
            if len(results) >= self.max_evidence_count:
                break

            if r.document.chunk_id in seen_chunks:
                continue
            seen_chunks.add(r.document.chunk_id)

            if index > 0:
                minimum_score = max(
                    self.minimum_additional_score,
                    top_score * self.additional_evidence_score_ratio,
                )
                if r.score < minimum_score:
                    continue

            # Prefer fewer, higher-confidence chunks per section to reduce answer drift.
            section = str(r.document.section)
            if section_counts.get(section, 0) >= self.max_chunks_per_section:
                continue
            section_counts[section] = section_counts.get(section, 0) + 1

            citation = self._build_citation(r.document)
            snippet = self._extract_snippet(r.document, query)
            citation.quote = snippet[:240] if snippet else None

            results.append(EvidenceResult(
                document=r.document,
                score=r.score,
                citation=citation,
                snippet=snippet,
            ))

        return results

    def _build_citation(self, doc: VectorDocument) -> Citation:
        """Build a citation object from a document."""
        metadata = doc.metadata_json or {}
        return Citation(
            chunk_id=str(doc.chunk_id),
            document_id=str(doc.document_id),
            section=str(doc.section),
            source_url=str(doc.source_url) if doc.source_url else None,
            quote=None,  # Can be populated if needed
        )

    def _extract_snippet(self, doc: VectorDocument, query: str) -> str:
        """Extract a relevant snippet from the document."""
        text = doc.text

        # Truncate if too long
        if len(text) > self.max_snippet_length:
            # Try to find query terms and extract around them
            query_terms = query.lower().split()
            best_pos = 0
            best_count = 0

            for term in query_terms:
                pos = text.lower().find(term.lower())
                if pos >= 0:
                    # Count how many terms appear near this position
                    window = text[max(0, pos-100):pos+100].lower()
                    count = sum(1 for t in query_terms if t in window)
                    if count > best_count:
                        best_count = count
                        best_pos = max(0, pos - 200)

            # Extract snippet around best position
            start = best_pos
            end = min(start + self.max_snippet_length, len(text))
            text = text[start:end]

            # Add ellipsis if truncated
            if start > 0:
                text = "..." + text
            if end < len(doc.text):
                text = text + "..."

        return text
