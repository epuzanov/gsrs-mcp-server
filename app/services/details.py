"""Parser and validator for the GSRS "details" filter syntax.

The details endpoint at
``/api/v1/substances({uuid})/{filter}`` accepts a compact filter
expression described in the GSRS API documentation:

    GET /api/v1/substances({id})/names(type:of)!(name)
    GET /api/v1/substances({id})/names!(name)!limit(1)
    GET /api/v1/substances({id})/relationships(type:IMPURITY->PARENT)

A filter is a concatenation of segments. Each segment starts with a
field path (the "elements" section) and may be followed by zero or
more parenthesized "locators" and bang-prefixed operations
(projections, sorts, aggregations).

The parser implemented here is intentionally conservative: it does
not attempt to reproduce the upstream GSRS server-side evaluation,
it only validates the local syntax and exposes the components for
introspection. The actual filter string is forwarded to the server
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# Recognized aggregations (the bang-prefixed functions). The upstream
# GSRS server may support additional registered functions, so this is
# a "well-known" list used for local validation only. Unknown
# aggregations are passed through.
KNOWN_AGGREGATIONS = frozenset(
    {
        "sort",
        "revsort",
        "skip",
        "limit",
        "distinct",
        "count",
        "group",
        "map",
        "flatmap",
        "filter",
    }
)


# Bang-prefixed functions that take a long argument instead of a
# field path.
LONG_AGGREGATIONS = frozenset({"skip", "limit"})

# Bang-prefixed functions that may be called without an argument.
NO_ARG_AGGREGATIONS = frozenset({"count"})


class FilterError(ValueError):
    """Raised when a details filter string is malformed."""


@dataclass(frozen=True)
class FilterSegment:
    """A single elements-section segment of a filter.

    A segment begins with a field path (e.g. ``names``,
    ``relationships/relatedSubstance``) and is followed by zero or
    more locators or bang-operations.
    """

    field_path: str
    locators: List[str] = field(default_factory=list)
    projections: List[str] = field(default_factory=list)
    aggregations: List[tuple[str, str]] = field(default_factory=list)

    @property
    def is_index_locator(self) -> bool:
        """True if the first locator is an index like ``($0)``."""
        if not self.locators:
            return False
        first = self.locators[0].strip()
        if not (first.startswith("($") and first.endswith(")")):
            return False
        return first[2:-1].isdigit()


@dataclass(frozen=True)
class ParsedFilter:
    """A parsed filter string.

    ``raw`` is the original (stripped) filter. ``segments`` is the
    list of parsed elements-section segments. ``empty`` is True when
    the input is empty or whitespace.
    """

    raw: str
    segments: List[FilterSegment] = field(default_factory=list)
    empty: bool = False


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Split a filter string into raw tokens.

    Tokens are the smallest atomic pieces we surface to consumers:
    a plain field segment, a parenthesized locator, a bang-field
    projection, or a bang-aggregation like ``!limit(1)``. Whitespace
    is significant only as a separator between two adjacent field
    paths; inside parentheses or after a bang it is part of the
    token.
    """
    tokens: List[str] = []
    i = 0
    n = len(text)
    current_field: List[str] = []

    def flush_field() -> None:
        if current_field:
            tokens.append("".join(current_field))
            current_field.clear()

    while i < n:
        ch = text[i]
        if ch == "(":
            flush_field()
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                cj = text[j]
                if cj == "(":
                    depth += 1
                elif cj == ")":
                    depth -= 1
                j += 1
            if depth != 0:
                raise FilterError(
                    f"Unbalanced parentheses in filter starting at position {i}: "
                    f"missing ')' for '('"
                )
            tokens.append(text[i:j])
            i = j
            continue
        if ch == "!":
            flush_field()
            j = i + 1
            # Read the name (letters/digits/underscore).
            name_start = j
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            name = text[name_start:j]
            # Optional parenthesized argument.
            if j < n and text[j] == "(":
                depth = 1
                k = j + 1
                while k < n and depth > 0:
                    ck = text[k]
                    if ck == "(":
                        depth += 1
                    elif ck == ")":
                        depth -= 1
                    k += 1
                if depth != 0:
                    raise FilterError(
                        f"Unbalanced parentheses in '!{name}' starting at position {i}"
                    )
                tokens.append(text[i:k])
                i = k
                continue
            if not name:
                # The token is just "!" with nothing meaningful after it.
                raise FilterError(
                    f"Expected an aggregation or field name after '!' at position {i}"
                )
            tokens.append(text[i:j])
            i = j
            continue
        if ch == "/":
            # Field-path separator within a single elements section.
            current_field.append(ch)
            i += 1
            continue
        if ch.isspace():
            flush_field()
            i += 1
            continue
        current_field.append(ch)
        i += 1

    flush_field()
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _classify_token(token: str) -> str:
    """Return one of: "locator", "projection", "aggregation", "field"."""
    if token.startswith("("):
        return "locator"
    if token.startswith("!"):
        if "(" in token:
            return "aggregation"
        return "projection"
    return "field"


def _parse_aggregation(token: str) -> tuple[str, str]:
    """Split a ``!name(arg)`` or ``!(arg)`` token into ``(name, arg)``.

    Tokens of the form ``!(arg)`` (empty name) are treated as
    projections whose argument is the field path inside the parens.
    The caller is expected to handle these as projections rather
    than aggregations.
    """
    # token starts with '!'.
    inner = token[1:]
    paren = inner.find("(")
    if paren == -1:
        # Should not happen because _classify_token distinguished
        # projections from aggregations.
        raise FilterError(f"Invalid aggregation token: {token!r}")
    name = inner[:paren]
    if not inner.endswith(")"):
        raise FilterError(
            f"Aggregation '{name}' is missing closing parenthesis: {token!r}"
        )
    arg = inner[paren + 1 : -1]
    return name, arg


def _validate_long_arg(name: str, arg: str) -> None:
    if not arg.isdigit():
        raise FilterError(
            f"Aggregation '{name}' requires a non-negative integer argument, got {arg!r}"
        )


def _validate_field_arg(name: str, arg: str) -> None:
    if not arg:
        raise FilterError(f"Aggregation '{name}' requires a field-path argument")


def _validate_aggregation(name: str, arg: str) -> None:
    if name in LONG_AGGREGATIONS:
        _validate_long_arg(name, arg)
        return
    if name in NO_ARG_AGGREGATIONS:
        # ``!count()`` and ``!count(field)`` are both valid per the
        # GSRS parser.
        return
    if name in KNOWN_AGGREGATIONS:
        _validate_field_arg(name, arg)
        return
    # Unknown aggregations (e.g. custom registered functions on the
    # server) are accepted without further validation.
    if not name:
        # ``!(arg)`` form — only meaningful when there's a field name
        # to project to.
        if not arg:
            raise FilterError("Empty '!()' token: expected a field projection")


def parse_filter(filter_text: Optional[str]) -> ParsedFilter:
    """Parse a filter string into a structured form.

    Args:
        filter_text: Raw filter expression. ``None`` and empty
            strings are accepted and produce an empty parsed filter.

    Returns:
        A :class:`ParsedFilter` instance.

    Raises:
        FilterError: If the filter is syntactically invalid.
    """
    if filter_text is None:
        return ParsedFilter(raw="", empty=True)

    text = filter_text.strip()
    if not text:
        return ParsedFilter(raw="", empty=True)

    tokens = _tokenize(text)
    if not tokens:
        return ParsedFilter(raw=text, empty=True)

    segments: List[FilterSegment] = []
    current: Optional[FilterSegment] = None

    for token in tokens:
        kind = _classify_token(token)
        if kind == "field":
            if current is not None:
                segments.append(current)
            if not token:
                raise FilterError("Empty field path in filter")
            current = FilterSegment(field_path=token)
            continue
        # All non-field tokens must be attached to a current segment.
        if current is None:
            raise FilterError(
                f"Filter expression must start with a field path; got {token!r}"
            )
        if kind == "locator":
            current.locators.append(token)
            continue
        if kind == "projection":
            current.projections.append(token[1:])
            continue
        if kind == "aggregation":
            name, arg = _parse_aggregation(token)
            if not name:
                # ``!(arg)`` is a projection-style token. The arg is
                # the field name to project to.
                if not arg:
                    _validate_aggregation(name, arg)
                current.projections.append(arg)
                continue
            _validate_aggregation(name, arg)
            current.aggregations.append((name, arg))
            continue
        # Defensive: should never happen.
        raise FilterError(f"Unrecognized token: {token!r}")

    if current is not None:
        segments.append(current)

    return ParsedFilter(raw=text, segments=segments, empty=False)


def validate_filter(filter_text: Optional[str]) -> None:
    """Raise :class:`FilterError` if the filter is malformed.

    Convenience wrapper around :func:`parse_filter` for callers that
    only need validation.
    """
    parse_filter(filter_text)


__all__ = [
    "KNOWN_AGGREGATIONS",
    "LONG_AGGREGATIONS",
    "FilterError",
    "FilterSegment",
    "ParsedFilter",
    "parse_filter",
    "validate_filter",
]
