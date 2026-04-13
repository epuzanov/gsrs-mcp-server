"""Shared identifier code-system configuration and matching helpers."""
import re
from dataclasses import dataclass

from app.config import Settings, settings


@dataclass(frozen=True)
class CodeSystemDefinition:
    """Configuration for a supported identifier code system."""

    canonical_name: str
    aliases: tuple[str, ...]
    require_qualified_mention: bool = False


_DEFAULT_CODE_SYSTEM_DEFINITIONS = {
    "CAS": CodeSystemDefinition("CAS", ("cas",)),
    "UNII": CodeSystemDefinition("UNII", ("unii",)),
    "FDA UNII": CodeSystemDefinition("FDA UNII", ("fda unii", "fdaunii")),
    "PubChem": CodeSystemDefinition("PubChem", ("pubchem",)),
    "DrugBank": CodeSystemDefinition("DrugBank", ("drugbank",)),
    "ChEMBL": CodeSystemDefinition("ChEMBL", ("chembl",)),
    "RXCUI": CodeSystemDefinition("RXCUI", ("rxcui",)),
    "SMS_ID": CodeSystemDefinition("SMS_ID", ("sms_id", "sms id")),
    "SMSID": CodeSystemDefinition("SMSID", ("smsid",)),
    "EVMPD": CodeSystemDefinition("EVMPD", ("evmpd",)),
    "xEVMPD": CodeSystemDefinition("xEVMPD", ("xevmpd",)),
    "ASK": CodeSystemDefinition("ASK", ("ask",), require_qualified_mention=True),
    "ASKP": CodeSystemDefinition("ASKP", ("askp",), require_qualified_mention=True),
}


def _normalize_code_system_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _alias_to_regex(alias: str) -> str:
    parts = [re.escape(part) for part in re.split(r"[\s_-]+", alias) if part]
    return r"[\s_-]*".join(parts)


def get_configured_code_systems(app_settings: Settings = settings) -> list[CodeSystemDefinition]:
    """Return configured code systems with canonical names and aliases."""
    configured: list[CodeSystemDefinition] = []
    seen: set[str] = set()
    normalized_registry = {
        _normalize_code_system_name(definition.canonical_name): definition
        for definition in _DEFAULT_CODE_SYSTEM_DEFINITIONS.values()
    }
    for value in app_settings.identifier_code_systems:
        normalized = _normalize_code_system_name(value)
        definition = normalized_registry.get(normalized)
        if definition is None:
            definition = CodeSystemDefinition(
                canonical_name=value.strip(),
                aliases=(value.strip().lower(),),
            )
        if definition.canonical_name not in seen:
            configured.append(definition)
            seen.add(definition.canonical_name)
    return configured


def get_identifier_value_patterns(app_settings: Settings = settings) -> dict[str, re.Pattern[str]]:
    """Build query patterns that extract identifier values by configured code system."""
    patterns: dict[str, re.Pattern[str]] = {}
    for definition in get_configured_code_systems(app_settings):
        alias_pattern = "|".join(_alias_to_regex(alias) for alias in definition.aliases)
        if definition.require_qualified_mention:
            canonical = re.escape(definition.canonical_name)
            patterns[definition.canonical_name] = re.compile(
                rf"(?:\b{canonical}\b|\b(?:{alias_pattern})\b\s+(?:code|id|identifier))[:\s-]*([A-Z0-9-]{{3,}})\b"
            )
        else:
            qualifier = "cid" if definition.canonical_name == "PubChem" else "identifier"
            patterns[definition.canonical_name] = re.compile(
                rf"\b(?:{alias_pattern})(?:\s+(?:code|id|{qualifier}))?[:\s-]*([A-Z0-9-]{{3,}})\b",
                re.IGNORECASE,
            )
    return patterns


def get_identifier_mention_patterns(app_settings: Settings = settings) -> dict[str, re.Pattern[str]]:
    """Build query patterns that detect code-system mentions even without a value."""
    patterns: dict[str, re.Pattern[str]] = {}
    for definition in get_configured_code_systems(app_settings):
        alias_pattern = "|".join(_alias_to_regex(alias) for alias in definition.aliases)
        if definition.require_qualified_mention:
            canonical = re.escape(definition.canonical_name)
            patterns[definition.canonical_name] = re.compile(
                rf"(?:\b{canonical}\b|\b(?:{alias_pattern})\b\s+(?:code|codes|id|identifier|identifiers)\b)",
            )
        else:
            patterns[definition.canonical_name] = re.compile(
                rf"\b(?:{alias_pattern})\b(?:\s+(?:code|codes|id|identifier|identifiers|cid))?\b",
                re.IGNORECASE,
            )
    return patterns


def get_identifier_field_names(app_settings: Settings = settings) -> list[str]:
    """Return normalized metadata field names derived from configured code systems."""
    field_names: list[str] = []
    for definition in get_configured_code_systems(app_settings):
        for candidate in {
            definition.canonical_name.lower(),
            definition.canonical_name.lower().replace(" ", "_"),
            definition.canonical_name.lower().replace(" ", ""),
            definition.canonical_name.lower().replace("-", "_"),
        }:
            if candidate not in field_names:
                field_names.append(candidate)
    return field_names


def get_identifier_keyword_labels(app_settings: Settings = settings) -> list[str]:
    """Return canonical labels for rewrite expansion and docs/debug output."""
    return [definition.canonical_name for definition in get_configured_code_systems(app_settings)]
