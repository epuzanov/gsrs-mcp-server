"""Markdown summaries for GSRS substance JSON payloads."""
from typing import Any


def _escape_table(value: Any) -> str:
    """Return a markdown-table-safe scalar string."""
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _join_values(value: Any) -> str:
    """Join GSRS list-like fields into a compact display string."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _display_name(data: dict[str, Any]) -> str:
    """Extract the preferred GSRS display name."""
    if data.get("_name"):
        return str(data["_name"])
    for entry in data.get("names") or []:
        if not isinstance(entry, dict):
            continue
        display = entry.get("displayName", entry.get("display_name", False))
        if isinstance(display, str):
            display = display.lower() == "true"
        if display and entry.get("name"):
            return str(entry["name"])
    if data.get("name"):
        return str(data["name"])
    return "Unknown"


def substance_to_markdown(data: dict[str, Any]) -> str:
    """Generate a compact markdown summary for one GSRS substance document."""
    md: list[str] = []
    md.append(f"# Substance: {_display_name(data)}")
    md.append("")
    md.append(f"**Approval ID:** {data.get('_approvalIDDisplay') or data.get('approvalID') or 'N/A'}")
    md.append(f"**Class:** {data.get('substanceClass', 'N/A')} | **Status:** {data.get('status', 'N/A')}")
    md.append("")

    structure = data.get("structure")
    if isinstance(structure, dict):
        md.append("## Structure")
        md.append("")
        md.append(f"- **Formula:** {structure.get('formula') or structure.get('molecularFormula') or 'N/A'}")
        md.append(f"- **SMILES:** `{structure.get('smiles', 'N/A')}`")
        md.append(f"- **Molecular Weight:** {structure.get('mwt') or structure.get('molecularWeight') or 'N/A'}")
        md.append(f"- **Stereochemistry:** {structure.get('stereochemistry', 'N/A')}")
        md.append("")

    names = data.get("names") or []
    if isinstance(names, list) and names:
        md.append("## Names")
        md.append("")
        md.append("| Name | Type | Domains | Languages |")
        md.append("|---|---|---|---|")
        for entry in names:
            if isinstance(entry, dict):
                name = _escape_table(entry.get("name"))
                name_type = _escape_table(entry.get("type"))
                domains = _escape_table(_join_values(entry.get("domains")))
                languages = _escape_table(_join_values(entry.get("languages")))
            else:
                name = _escape_table(entry)
                name_type = domains = languages = ""
            md.append(f"| {name} | {name_type} | {domains} | {languages} |")
        md.append("")

    codes = data.get("codes") or []
    if isinstance(codes, list) and codes:
        md.append("## Codes")
        md.append("")
        md.append("| Code System | Code | Comments |")
        md.append("|---|---|---|")
        for entry in codes:
            if not isinstance(entry, dict):
                continue
            system = _escape_table(entry.get("codeSystem"))
            code = _escape_table(entry.get("code"))
            comments = _escape_table(entry.get("comments"))
            md.append(f"| {system} | {code} | {comments} |")
        md.append("")

    relationships = data.get("relationships") or []
    if isinstance(relationships, list) and relationships:
        md.append("## Relationships")
        md.append("")
        md.append("| Type | Related Substance | Qualification |")
        md.append("|---|---|---|")
        for entry in relationships[:50]:
            if not isinstance(entry, dict):
                continue
            related = entry.get("relatedSubstance") or {}
            if isinstance(related, dict):
                related_name = related.get("refPname") or related.get("name") or related.get("uuid") or ""
            else:
                related_name = str(related)
            md.append(
                "| "
                f"{_escape_table(entry.get('type'))} | "
                f"{_escape_table(related_name)} | "
                f"{_escape_table(entry.get('qualification'))} |"
            )
        md.append("")

    return "\n".join(md).rstrip() + "\n"
