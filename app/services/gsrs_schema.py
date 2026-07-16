"""Pydantic JSON-Schema lookup for the vendored ``gsrs.model`` package.

The server vendors the GSRS Pydantic models from
``https://github.com/epuzanov/gsrs.model.git`` (see
``pyproject.toml`` / ``requirements.txt``). These models describe the
shape of a GSRS substance record (``Substance`` is the polymorphic
base; ``ChemicalSubstance``, ``ProteinSubstance``, etc. are
subclasses selected by ``substanceClass``).

This module is the **single source of truth** for fetching the
JSON Schema of any of those models. It is used by the
``gsrs_get_schema`` tool and the ``gsrs://schema/{model}`` resource,
and is intended as a *reference* for the LLM-driven tools:

- ``gsrs_parametric_search`` uses indexed Lucene fields
  (``root_*``) whose names mirror the Pydantic model attributes.
  A model that knows the field names can build more precise
  ``filters`` arguments.
- ``gsrs_get_substance_details`` walks the substance JSON via
  ``/``-separated element paths. Knowing the model attributes
  (and their nested structure under ``$defs``) lets the model
  construct accurate paths like ``names``, ``codes``,
  ``relationships/relatedSubstance``.

The module degrades gracefully if ``gsrs-model`` is not installed
(returns a clear error rather than crashing the server).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


# Public, allow-listed set of GSRS Pydantic models. The MCP tool
# and resource will only accept names from this set — anything else
# returns a 404 / ``UnknownModel`` error. This is intentionally a
# hand-curated list (not a ``getattr`` over ``gsrs.model``) so we
# do not accidentally expose unrelated symbols from the package.
SUPPORTED_MODELS = frozenset(
    {
        "Substance",
        "ChemicalSubstance",
        "ProteinSubstance",
        "NucleicAcidSubstance",
        "MixtureSubstance",
        "PolymerSubstance",
        "StructurallyDiverseSubstance",
    }
)


class GsrsSchemaError(Exception):
    """Raised when a GSRS model schema cannot be returned.

    ``code`` is a short machine-readable token so callers can map
    errors to MCP resource / tool responses without parsing prose.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _load_model(model_name: str):
    """Resolve a public GSRS model name to a Pydantic class.

    Raises:
        GsrsSchemaError: with ``code="model_not_found"`` if
            ``model_name`` is not in :data:`SUPPORTED_MODELS`, or
            ``code="dependency_missing"`` if ``gsrs.model`` cannot
            be imported, or ``code="model_unavailable"`` if the
            package is installed but the model symbol is missing
            (a version-skew situation).
    """
    if model_name not in SUPPORTED_MODELS:
        raise GsrsSchemaError(
            "model_not_found",
            f"Unknown GSRS model {model_name!r}. "
            f"Supported models: {', '.join(sorted(SUPPORTED_MODELS))}.",
        )

    try:
        import gsrs.model  # type: ignore[import-untyped]
    except ImportError as exc:
        raise GsrsSchemaError(
            "dependency_missing",
            "The 'gsrs-model' package is not installed; install it via "
            "`pip install -r requirements.txt` to enable schema lookup.",
        ) from exc

    try:
        cls = getattr(gsrs.model, model_name)
    except AttributeError as exc:
        raise GsrsSchemaError(
            "model_unavailable",
            f"gsrs.model is installed but does not expose {model_name!r}.",
        ) from exc

    if not hasattr(cls, "model_json_schema"):
        raise GsrsSchemaError(
            "model_unavailable",
            f"gsrs.model.{model_name} does not expose model_json_schema(); "
            "is it a Pydantic model?",
        )

    return cls


def get_model_schema(
    model_name: str = "Substance",
    *,
    indent: int = 2,
) -> Dict[str, Any]:
    """Return the Pydantic JSON Schema for a GSRS model.

    Args:
        model_name: One of the names in :data:`SUPPORTED_MODELS`.
            Defaults to ``"Substance"`` (the polymorphic base).
        indent: Pretty-print indent level for the embedded JSON
            string. ``None`` produces compact output. This is the
            indent passed to ``json.dumps`` when callers ask for
            the string form via :func:`get_model_schema_text`.

    Returns:
        The parsed schema dict (as returned by
        :meth:`pydantic.BaseModel.model_json_schema`).

    Raises:
        GsrsSchemaError: if the model name is unknown or the
            ``gsrs-model`` package is unavailable.
    """
    cls = _load_model(model_name)
    return cls.model_json_schema()


def get_model_schema_text(
    model_name: str = "Substance",
    *,
    indent: Optional[int] = 2,
) -> str:
    """Return the JSON Schema for a GSRS model, serialized as text.

    The text is what the MCP tool returns to the LLM — a compact
    JSON document the model can scan for field names and
    ``$defs`` references.
    """
    schema = get_model_schema(model_name)
    return json.dumps(schema, indent=indent, default=str)


__all__ = [
    "GsrsSchemaError",
    "SUPPORTED_MODELS",
    "get_model_schema",
    "get_model_schema_text",
]
