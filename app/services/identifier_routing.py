"""Deterministic identifier-first routing for high-signal GSRS lookups."""
import re
from dataclasses import dataclass
from typing import Any

from app.config import Settings, settings
from app.models import DBQueryResult
from app.services.code_systems import get_identifier_value_patterns
from app.services.vector_database import VectorDatabaseService


_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_INCHIKEY_RE = re.compile(r"\b[A-Z]{14}-[A-Z]{10}-[A-Z]\b")
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')

_QUESTION_PREFIXES = (
    "what is",
    "what are",
    "tell me about",
    "describe",
    "find",
    "get",
    "lookup",
    "search for",
    "show me",
)

_APPROVAL_ID_RE = re.compile(r"\bapproval(?:\s+id)?[:\s-]*([A-Z0-9-]{3,})", re.IGNORECASE)


@dataclass
class IdentifierRouteResult:
    """Deterministic routing result for a query, if any."""

    route: str
    example: dict[str, Any]
    results: list[DBQueryResult]
    matched_value: str


class IdentifierRouter:
    """Prefer exact metadata lookup when a query clearly contains an identifier."""

    def __init__(self, vector_db: VectorDatabaseService, app_settings: Settings = settings):
        self.vector_db = vector_db
        self.identifier_keyword_patterns = get_identifier_value_patterns(app_settings)

    def route(self, query: str, top_k: int) -> IdentifierRouteResult | None:
        example, route, matched_value = self._build_example(query)
        if not example:
            return None

        results = self.vector_db.search_by_example(example=example, top_k=top_k, mode="match")
        if not results:
            return IdentifierRouteResult(route=route, example=example, results=[], matched_value=matched_value)

        return IdentifierRouteResult(
            route=route,
            example=example,
            results=results,
            matched_value=matched_value,
        )

    def _build_example(self, query: str) -> tuple[dict[str, Any], str, str]:
        query_text = query.strip()

        uuid_match = _UUID_RE.search(query_text)
        if uuid_match:
            matched = uuid_match.group(0)
            return {"uuid": matched}, "uuid", matched

        inchikey_match = _INCHIKEY_RE.search(query_text.upper())
        if inchikey_match:
            matched = inchikey_match.group(0)
            return {"structure": {"inchikey": matched}}, "inchikey", matched

        approval_match = _APPROVAL_ID_RE.search(query_text)
        if approval_match:
            matched = approval_match.group(1)
            return {"approvalID": matched}, "approval_id", matched

        for label, pattern in self.identifier_keyword_patterns.items():
            match = pattern.search(query_text)
            if match:
                matched = match.group(1)
                return {
                    "reliable_codes": {label: matched},
                    "all_codes": {label: matched},
                }, "code", matched

        quoted = self._extract_exact_name(query_text)
        if quoted:
            example = {
                "canonical_name": quoted,
                "systematic_names": [quoted],
                "official_names": [quoted],
                "other_names": [quoted],
            }
            return example, "exact_name", quoted

        return {}, "none", ""

    def _extract_exact_name(self, query: str) -> str | None:
        phrase_match = _QUOTED_PHRASE_RE.search(query)
        if phrase_match:
            return next(group for group in phrase_match.groups() if group)

        lowered = query.lower().strip().rstrip("?")
        if any(lowered.startswith(prefix) for prefix in _QUESTION_PREFIXES):
            return None

        tokens = [token for token in re.split(r"\s+", query.strip()) if token]
        if 0 < len(tokens) <= 5:
            return query.strip()

        return None
