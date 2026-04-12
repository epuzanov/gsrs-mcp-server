"""
GSRS MCP Server - Abstention Policy
Decides whether to answer or abstain based on evidence quality.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.models.db import VectorDocument
from app.services.evidence import EvidenceResult


@dataclass
class AbstentionDecision:
    abstained: bool
    confidence: float
    abstain_reason: Optional[str] = None


class AbstentionPolicy:
    """
    Decides whether the system should answer or abstain.

    Abstain if:
    - No relevant evidence after rerank
    - Top rerank score below threshold
    - Identifier lookup has no exact-supporting chunk
    - Evidence is conflicting without resolution
    - Evidence is only tangentially related
    """

    def __init__(
        self,
        min_score_threshold: float = 0.3,
        min_confidence: float = 0.0,
        min_evidence_count: int = 1,
    ):
        self.min_score_threshold = min_score_threshold
        self.min_confidence = min_confidence
        self.min_evidence_count = min_evidence_count

    def evaluate(
        self,
        evidence: List[EvidenceResult],
        query: str,
        intent: str = "general",
        applied_filters: Optional[Dict[str, Any]] = None,
    ) -> AbstentionDecision:
        """
        Evaluate whether to answer or abstain.

        Args:
            evidence: List of evidence results
            query: Original query
            intent: Query intent
            applied_filters: Filters that were applied during retrieval

        Returns:
            AbstentionDecision with abstain/answer decision
        """
        # Check: no evidence
        if not evidence:
            return AbstentionDecision(
                abstained=True,
                confidence=0.0,
                abstain_reason="No relevant evidence found in GSRS database.",
            )

        # Check: insufficient evidence count
        if len(evidence) < self.min_evidence_count:
            return AbstentionDecision(
                abstained=True,
                confidence=0.0,
                abstain_reason="Insufficient evidence to provide an answer.",
            )

        # Check: top scores below threshold
        top_scores = [e.score for e in evidence[:5]]
        if not top_scores:
            return AbstentionDecision(
                abstained=True,
                confidence=0.0,
                abstain_reason="No evidence with sufficient relevance score.",
            )

        avg_score = sum(top_scores) / len(top_scores)
        max_score = max(top_scores)

        if max_score < self.min_score_threshold:
            return AbstentionDecision(
                abstained=True,
                confidence=max_score,
                abstain_reason=f"Evidence relevance too low (max score: {max_score:.2f}, threshold: {self.min_score_threshold}).",
            )

        # Check: identifier lookup without exact match
        if intent in ("identifier_lookup", "identifier_query"):
            has_exact_match = self._has_identifier_support(evidence, query)
            if not has_exact_match:
                return AbstentionDecision(
                    abstained=True,
                    confidence=max_score * 0.5,
                    abstain_reason="Identifier lookup requested but no exact identifier match found in evidence.",
                )

        # Check: conflicting evidence
        has_conflict = self._detect_conflicts(evidence)
        if has_conflict:
            return AbstentionDecision(
                abstained=True,
                confidence=max_score * 0.6,
                abstain_reason="Evidence appears conflicting; cannot provide a definitive answer.",
            )

        # Check: tangential evidence
        is_tangential = self._is_evidence_tangential(evidence, query)
        if is_tangential:
            return AbstentionDecision(
                abstained=True,
                confidence=max_score * 0.7,
                abstain_reason="Evidence is only tangentially related to the query.",
            )

        # No abstention triggers - allow answering
        return AbstentionDecision(
            abstained=False,
            confidence=max_score,
            abstain_reason=None,
        )

    def _has_identifier_support(self, evidence: List[EvidenceResult], query: str) -> bool:
        """Check if any evidence contains exact identifier matches."""
        query_lower = query.lower()
        # Look for common identifier patterns
        identifier_keywords = ["cas", "unii", "pubchem", "drugbank", "chembl", "rxcui"]

        for e in evidence:
            text_lower = e.document.text.lower()
            metadata = e.document.metadata_json or {}

            # Check for identifier codes in metadata
            codes = metadata.get("codes", [])
            for code in codes:
                code_str = str(code).lower() if isinstance(code, str) else str(code.get("code", "")).lower()
                for kw in identifier_keywords:
                    if kw in query_lower and kw in code_str:
                        return True

            # Check for identifier in text
            for kw in identifier_keywords:
                if kw in query_lower and kw in text_lower:
                    return True

        return False

    def _detect_conflicts(self, evidence: List[EvidenceResult]) -> bool:
        """Detect if evidence contains conflicting information."""
        if len(evidence) < 2:
            return False

        # Simple conflict detection: check for contradictory values
        # This is a heuristic - look for numeric values that differ significantly
        values_by_section: Dict[str, List[str]] = {}

        for e in evidence:
            section = e.citation.section
            text = e.document.text.lower()

            if section not in values_by_section:
                values_by_section[section] = []
            values_by_section[section].append(text)

        # Check for sections with very different content
        for section, texts in values_by_section.items():
            if len(texts) >= 2:
                # Simple heuristic: if texts share less than 20% common words
                words_sets = [set(t.split()) for t in texts]
                if len(words_sets) >= 2:
                    intersection = words_sets[0].intersection(*words_sets[1:])
                    union = words_sets[0].union(*words_sets[1:])
                    if union and len(intersection) / len(union) < 0.1:
                        return True

        return False

    def _is_evidence_tangential(self, evidence: List[EvidenceResult], query: str) -> bool:
        """Check if evidence is only tangentially related."""
        query_terms = set(query.lower().split())
        query_terms = {t for t in query_terms if len(t) >= 3}

        if not query_terms:
            return False

        # Check overlap between query terms and evidence
        total_overlap = 0
        total_terms = 0

        for e in evidence[:3]:  # Check top 3
            text_terms = set(e.document.text.lower().split())
            overlap = len(query_terms.intersection(text_terms))
            total_overlap += overlap
            total_terms += len(query_terms)

        if total_terms == 0:
            return True

        overlap_ratio = total_overlap / total_terms
        return overlap_ratio < 0.1  # Less than 10% term overlap
