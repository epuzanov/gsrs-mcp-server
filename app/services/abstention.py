"""
GSRS MCP Server - Abstention Policy
Decides whether to answer or abstain based on evidence quality.
"""
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from app.config import Settings, settings
from app.models.db import VectorDocument
from app.services.code_systems import get_identifier_field_names, get_identifier_value_patterns
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
        app_settings: Settings = settings,
    ):
        self.min_score_threshold = min_score_threshold
        self.min_confidence = min_confidence
        self.min_evidence_count = min_evidence_count
        self.identifier_field_names = get_identifier_field_names(app_settings)
        self.identifier_value_patterns = get_identifier_value_patterns(app_settings)

    def evaluate(
        self,
        evidence: List[EvidenceResult],
        query: str,
        intent: str = "general",
        applied_filters: Optional[Dict[str, Any]] = None,
        retrieval_mode: str = "hybrid",
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
            if retrieval_mode.startswith("identifier-first"):
                return AbstentionDecision(
                    abstained=True,
                    confidence=0.0,
                    abstain_reason="No exact metadata match found for the identifier-first lookup.",
                )
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
        required_confidence = max(self.min_score_threshold, self.min_confidence)

        if max_score < required_confidence:
            return AbstentionDecision(
                abstained=True,
                confidence=max_score,
                abstain_reason=f"Evidence relevance too low (max score: {max_score:.2f}, threshold: {required_confidence:.2f}).",
            )

        if avg_score < max(required_confidence * 0.85, 0.2) and len(evidence) == 1:
            return AbstentionDecision(
                abstained=True,
                confidence=max_score,
                abstain_reason="Only weak single-chunk evidence was retrieved; insufficient confidence to answer.",
            )

        # Check: identifier-first lookup without exact match
        if retrieval_mode.startswith("identifier-first"):
            has_exact_match = self._has_identifier_support(evidence, query)
            if not has_exact_match:
                return AbstentionDecision(
                    abstained=True,
                    confidence=max_score * 0.5,
                    abstain_reason="Identifier-first lookup requested but no exact identifier or exact-name match was found in evidence.",
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
        query_literals = self._extract_identifier_literals(query)
        query_name = self._extract_candidate_name(query)

        for e in evidence:
            text_lower = e.document.text.lower()
            metadata = e.document.metadata_json or {}

            if query_literals:
                candidate_values = self._extract_metadata_literals(metadata)
                if any(literal in candidate_values for literal in query_literals):
                    return True
                if any(literal in text_lower for literal in query_literals):
                    return True

            if query_name:
                names = [name.lower() for name in self._extract_names(metadata)]
                if query_name.lower() in names:
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

    def _extract_identifier_literals(self, query_text: str) -> List[str]:
        matches: List[str] = []
        for pattern in self.identifier_value_patterns.values():
            match = pattern.search(query_text)
            if match:
                matches.append(match.group(1).lower())

        query_lower = query_text.lower()
        patterns = [
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            r"\b[0-9]{2,}-[0-9]{2,}-[0-9]\b",
            r"\b[a-z0-9]{4,}(?:-[a-z0-9]{2,})+\b",
            r"\b[a-z]{14}-[a-z]{10}-[a-z]\b",
        ]
        for pattern in patterns:
            matches.extend(match.lower() for match in re.findall(pattern, query_lower, re.IGNORECASE))
        return list(dict.fromkeys(matches))

    def _extract_metadata_literals(self, metadata: Dict[str, Any]) -> set[str]:
        values: set[str] = set()
        for key in ["uuid", "approvalID", "cas", "unii", "pubchem", "drugbank", "chembl", "rxcui"]:
            value = metadata.get(key)
            if value:
                values.add(str(value).lower())

        for key in self.identifier_field_names:
            value = metadata.get(key)
            if value:
                values.add(str(value).lower())

        codes = metadata.get("codes", [])
        if isinstance(codes, list):
            for code in codes:
                if isinstance(code, dict) and code.get("code"):
                    values.add(str(code["code"]).lower())
                elif isinstance(code, str):
                    values.add(code.lower())

        for key in ["reliable_codes", "all_codes"]:
            bucket = metadata.get(key, {})
            if isinstance(bucket, dict):
                values.update(str(value).lower() for value in bucket.values() if value)

        structure = metadata.get("structure", {})
        if isinstance(structure, dict):
            for key in ["inchikey", "inchi", "smiles"]:
                value = structure.get(key)
                if value:
                    values.add(str(value).lower())

        return values

    def _extract_candidate_name(self, query: str) -> Optional[str]:
        quoted_match = re.search(r'"([^"]+)"|\'([^\']+)\'', query)
        if quoted_match:
            return next(group for group in quoted_match.groups() if group)

        stripped = query.strip().rstrip("?")
        tokens = stripped.split()
        if 0 < len(tokens) <= 4 and not any(pattern.search(query) for pattern in self.identifier_value_patterns.values()) and not any(
            token.lower() in {"approval", "identifier", "code"} for token in tokens
        ):
            return stripped
        return None

    def _extract_names(self, metadata: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        canonical = metadata.get("canonical_name")
        if canonical:
            names.append(str(canonical))

        names_list = metadata.get("names", [])
        if isinstance(names_list, list):
            for name in names_list:
                if isinstance(name, dict) and name.get("name"):
                    names.append(str(name["name"]))
                elif isinstance(name, str):
                    names.append(name)
        return names
