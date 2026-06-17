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


def _format_access(access: Any) -> str:
    """Return 'Public' when access is empty, otherwise 'Protected'."""
    if access and (not isinstance(access, list) or any(access)):
        return "Protected"
    return "Public"


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


def _section_header(title: str) -> list[str]:
    """Return a markdown section header with trailing blank line."""
    return [f"## {title}", ""]


def _format_structure(structure: dict[str, Any]) -> list[str]:
    """Render a structure block for chemical/polymer summaries."""
    md = _section_header("Structure")
    md.append(f"- **Formula:** {structure.get('formula') or structure.get('molecularFormula') or 'N/A'}")
    md.append(f"- **SMILES:** `{structure.get('smiles', 'N/A')}`")
    md.append(
        f"- **Molecular Weight:** {structure.get('mwt') or structure.get('molecularWeight') or 'N/A'}"
    )
    md.append(f"- **Stereochemistry:** {structure.get('stereochemistry', 'N/A')}")
    return md


def _format_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    """Render a markdown table from a list of row dicts and ordered keys."""
    if not rows:
        return []
    md: list[str] = []
    headers = [col.replace("_", " ").title() for col in columns]
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = [_escape_table(row.get(col, "")) for col in columns]
        md.append("| " + " | ".join(cells) + " |")
    return md


def _resolve_substance_name(substance: Any) -> str:
    """Return a readable name/approval ID from a related substance reference."""
    if isinstance(substance, dict):
        return (
            substance.get("refPname")
            or substance.get("name")
            or substance.get("approvalID")
            or substance.get("uuid")
            or ""
        )
    return str(substance or "")


def _format_names(names: list[Any]) -> list[str]:
    """Render the names table."""
    rows: list[dict[str, Any]] = []
    for entry in names:
        if isinstance(entry, dict):
            name_orgs = entry.get("nameOrgs") or []
            orgs = []
            for org in name_orgs:
                if isinstance(org, dict) and org.get("nameOrg"):
                    orgs.append(str(org["nameOrg"]))
            display = entry.get("displayName",  False)
            if isinstance(display, str):
                display = display.lower() == "true"
            preferred = entry.get("preferred", False)
            if isinstance(preferred, str):
                preferred = preferred.lower() == "true"
            rows.append(
                {
                    "name": entry.get("name"),
                    "type": entry.get("type"),
                    "display_name": "yes" if display else "",
                    "preferred": "yes" if preferred else "",
                    "name_orgs": ", ".join(orgs),
                    "domains": _join_values(entry.get("domains")),
                    "languages": _join_values(entry.get("languages")),
                }
            )
        else:
            rows.append(
                {
                    "name": entry,
                    "type": "",
                    "display_name": "",
                    "preferred": "",
                    "name_orgs": "",
                    "domains": "",
                    "languages": "",
                }
            )
    return _section_header("Names") + _format_table(
        rows, ["name", "type", "display_name", "preferred", "name_orgs", "domains", "languages"]
    )


def _format_codes(codes: list[Any]) -> list[str]:
    """Render identifiers and classifications tables from GSRS codes.

    Codes with a truthy `_isClassification` flag or a `comments` value that
    looks like a classification path go into the Classifications table;
    all other codes go into Identifiers.
    """
    identifiers: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    for entry in codes:
        if not isinstance(entry, dict):
            continue
        is_classification = entry.get("_isClassification") or "|" in str(
            entry.get("comments") or ""
        )
        row = {
            "code_system": entry.get("codeSystem"),
            "code": entry.get("code"),
            "type": entry.get("type"),
            "comments": entry.get("comments"),
        }
        if is_classification:
            classifications.append(row)
        else:
            identifiers.append(row)

    md: list[str] = []
    if identifiers:
        md += _section_header("Identifiers")
        md += _format_table(identifiers, ["code_system", "code", "type", "comments"])
        md.append("")
    if classifications:
        md += _section_header("Classifications")
        md += _format_table(
            classifications, ["code_system", "code", "type", "comments"]
        )
        md.append("")
    return md


def _format_relationships(relationships: list[Any]) -> list[str]:
    """Render the relationships table."""
    rows: list[dict[str, Any]] = []
    for entry in relationships[:50]:
        if not isinstance(entry, dict):
            continue
        related = entry.get("relatedSubstance") or {}
        rows.append(
            {
                "type": entry.get("type"),
                "related_substance": _resolve_substance_name(related),
                "qualification": entry.get("qualification"),
            }
        )
    return _section_header("Relationships") + _format_table(
        rows, ["type", "related_substance", "qualification"]
    )


def _format_properties(properties: list[Any]) -> list[str]:
    """Render a compact properties table."""
    rows: list[dict[str, Any]] = []
    for entry in properties:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value") or {}
        amount = ""
        if isinstance(value, dict):
            parts = []
            for key in ("average", "high", "low", "nonNumericValue"):
                if value.get(key) not in (None, ""):
                    parts.append(f"{key}={value[key]}")
            if value.get("units"):
                parts.append(f"units={value['units']}")
            amount = ", ".join(parts)
        rows.append(
            {
                "name": entry.get("name"),
                "type": entry.get("propertyType"),
                "value": amount,
                "defining": entry.get("defining"),
            }
        )
    return _section_header("Properties") + _format_table(
        rows, ["name", "type", "value", "defining"]
    )


def _format_protein(protein: dict[str, Any]) -> list[str]:
    """Render protein-specific details."""
    if not isinstance(protein, dict):
        return []
    md = _section_header("Protein Details")
    md.append(f"- **Type:** {protein.get('proteinType', 'N/A')}")
    md.append(f"- **Subtype:** {protein.get('proteinSubType', 'N/A')}")
    md.append(f"- **Sequence Origin:** {protein.get('sequenceOrigin', 'N/A')}")
    md.append(f"- **Sequence Type:** {protein.get('sequenceType', 'N/A')}")
    md.append("")

    subunits = protein.get("subunits") or []
    if subunits:
        md.append("### Subunits")
        md.append("")
        for entry in subunits[:20]:
            if not isinstance(entry, dict):
                continue
            seq = entry.get("sequence") or ""
            length = entry.get("length", len(seq))
            md.append(
                f"- Subunit {entry.get('subunitIndex', '?')} — length {length}: "
                f"`{seq[:60]}{'...' if len(seq) > 60 else ''}`"
            )
        md.append("")

    glycosylation = protein.get("glycosylation")
    if isinstance(glycosylation, dict):
        md.append("### Glycosylation")
        md.append("")
        md.append(
            f"- **Type:** {glycosylation.get('glycosylationType', 'N/A')}"
        )
        for key in ("NGlycosylationSites", "OGlycosylationSites", "CGlycosylationSites"):
            sites = glycosylation.get(key) or []
            if sites:
                md.append(f"- **{key}:** {len(sites)} site(s)")
        md.append("")

    links = protein.get("disulfideLinks") or []
    if links:
        md.append("### Disulfide Links")
        md.append("")
        for entry in links[:20]:
            if not isinstance(entry, dict):
                continue
            md.append(f"- {entry.get('sitesShorthand') or entry.get('sites', 'N/A')}")
        md.append("")
    return md


def _format_nucleic_acid(nucleic_acid: dict[str, Any]) -> list[str]:
    """Render nucleic-acid-specific details."""
    if not isinstance(nucleic_acid, dict):
        return []
    md = _section_header("Nucleic Acid Details")
    md.append(f"- **Type:** {nucleic_acid.get('nucleicAcidType', 'N/A')}")
    md.append(f"- **Sequence Type:** {nucleic_acid.get('sequenceType', 'N/A')}")
    sub_type = nucleic_acid.get("nucleicAcidSubType")
    if sub_type:
        md.append(f"- **Subtype:** {_join_values(sub_type)}")
    md.append("")

    subunits = nucleic_acid.get("subunits") or []
    if subunits:
        md.append("### Subunits")
        md.append("")
        for entry in subunits[:20]:
            if not isinstance(entry, dict):
                continue
            seq = entry.get("sequence") or ""
            length = entry.get("length", len(seq))
            md.append(
                f"- Subunit {entry.get('subunitIndex', '?')} — length {length}: "
                f"`{seq[:60]}{'...' if len(seq) > 60 else ''}`"
            )
        md.append("")
    return md


def _format_polymer(polymer: dict[str, Any]) -> list[str]:
    """Render polymer-specific details."""
    if not isinstance(polymer, dict):
        return []
    md: list[str] = []
    classification = polymer.get("classification")
    if isinstance(classification, dict):
        md += _section_header("Polymer Classification")
        md.append(f"- **Class:** {classification.get('polymerClass', 'N/A')}")
        md.append(f"- **Geometry:** {classification.get('polymerGeometry', 'N/A')}")
        md.append(
            f"- **Subclass:** {_join_values(classification.get('polymerSubclass'))}"
        )
        md.append(f"- **Source Type:** {classification.get('sourceType', 'N/A')}")
        md.append("")

    display = polymer.get("displayStructure") or polymer.get("idealizedStructure")
    if isinstance(display, dict):
        md += _section_header("Polymer Structure")
        md += _format_structure(display)
        md.append("")

    monomers = polymer.get("monomers") or []
    if monomers:
        md += _section_header("Monomers")
        rows: list[dict[str, Any]] = []
        for entry in monomers:
            if not isinstance(entry, dict):
                continue
            sub = entry.get("monomerSubstance") or {}
            rows.append(
                {
                    "monomer": _resolve_substance_name(sub),
                    "type": entry.get("type"),
                    "defining": entry.get("defining"),
                }
            )
        md += _format_table(rows, ["monomer", "type", "defining"])
        md.append("")
    return md


def _format_mixture(mixture: dict[str, Any]) -> list[str]:
    """Render mixture-specific details."""
    if not isinstance(mixture, dict):
        return []
    md = _section_header("Mixture Components")
    components = mixture.get("components") or []
    rows: list[dict[str, Any]] = []
    for entry in components:
        if not isinstance(entry, dict):
            continue
        sub = entry.get("substance") or {}
        rows.append(
            {
                "substance": _resolve_substance_name(sub),
                "type": entry.get("type"),
            }
        )
    md += _format_table(rows, ["substance", "type"])
    md.append("")
    return md


def _format_structurally_diverse(sd: dict[str, Any]) -> list[str]:
    """Render structurally-diverse (organism/whole-material) details."""
    if not isinstance(sd, dict):
        return []
    md = _section_header("Source Material")
    md.append(f"- **Class:** {sd.get('sourceMaterialClass', 'N/A')}")
    md.append(f"- **Type:** {sd.get('sourceMaterialType', 'N/A')}")
    md.append(f"- **State:** {sd.get('sourceMaterialState', 'N/A')}")
    md.append(
        f"- **Organism:** {sd.get('organismFamily', '')} > {sd.get('organismGenus', '')} > {sd.get('organismSpecies', '')}".strip(
            " >"
        )
    )
    md.append(f"- **Part:** {_join_values(sd.get('part'))}")
    if sd.get("infraSpecificType"):
        md.append(f"- **Infra-specific Type:** {sd.get('infraSpecificType')}")
    if sd.get("infraSpecificName"):
        md.append(f"- **Infra-specific Name:** {sd.get('infraSpecificName')}")
    md.append("")
    return md


def _format_specified_substance(specified: dict[str, Any]) -> list[str]:
    """Render specified-substance-G1 constituent details."""
    if not isinstance(specified, dict):
        return []
    md = _section_header("Specified Substance Constituents")
    constituents = specified.get("constituents") or []
    rows: list[dict[str, Any]] = []
    for entry in constituents:
        if not isinstance(entry, dict):
            continue
        sub = entry.get("substance") or {}
        rows.append(
            {
                "constituent": _resolve_substance_name(sub),
                "role": entry.get("role"),
            }
        )
    md += _format_table(rows, ["constituent", "role"])
    md.append("")
    return md


def _format_moieties(moieties: list[Any]) -> list[str]:
    """Render a moieties table for chemical substances."""
    rows: list[dict[str, Any]] = []
    if not moieties:
        return []
    for entry in moieties:
        if not isinstance(entry, dict):
            continue
        count_amount = entry.get("countAmount") or {}
        amount = ""
        if isinstance(count_amount, dict):
            parts = []
            for key in ("average", "high", "low"):
                if count_amount.get(key) not in (None, ""):
                    parts.append(f"{key}={count_amount[key]}")
            if count_amount.get("units"):
                parts.append(f"units={count_amount['units']}")
            amount = ", ".join(parts)
        rows.append(
            {
                "formula": entry.get("formula"),
                "smiles": entry.get("smiles"),
                "molecular_weight": entry.get("mwt"),
                "count": entry.get("count"),
                "amount": amount,
                "stereochemistry": entry.get("stereochemistry"),
            }
        )
    return _section_header("Moieties") + _format_table(
        rows,
        ["formula", "smiles", "molecular_weight", "count", "amount", "stereochemistry"],
    ) + [""]


def substance_to_markdown(data: dict[str, Any]) -> str:
    """Generate a compact markdown summary for one GSRS substance document."""
    md: list[str] = []
    md.append(f"# Substance: {_display_name(data)}")
    md.append("")
    md.append(
        f"**Approval ID:** {data.get('_approvalIDDisplay') or data.get('approvalID') or 'N/A'}"
    )
    md.append(f"**Class:** {data.get('substanceClass', 'N/A')}")
    md.append(f"**Status:** {data.get('status', 'N/A')}")
    md.append(f"**Access:** {_format_access(data.get('access'))}")
    md.append("")

    substance_class = data.get("substanceClass", "")

    if substance_class != "concept":
        md += _section_header("Definitional Information")
        if substance_class == "chemical":
            md += _format_structure(data.get("structure"))
            md += _format_moieties(data.get("moieties"))
        elif substance_class == "protein":
            md += _format_protein(data.get("protein"))
        elif substance_class == "nucleicAcid":
            md += _format_nucleic_acid(data.get("nucleicAcid"))
        elif substance_class == "polymer":
            md += _format_polymer(data.get("polymer"))
        elif substance_class == "mixture":
            md += _format_mixture(data.get("mixture"))
        elif substance_class == "structurallyDiverse":
            md += _format_structurally_diverse(data.get("structurallyDiverse"))
        elif substance_class == "specifiedSubstanceG1":
            md += _format_specified_substance(data.get("specifiedSubstance"))
        md.append("")

    names = data.get("names") or []
    if isinstance(names, list) and names:
        md += _format_names(names)
        md.append("")

    codes = data.get("codes") or []
    if isinstance(codes, list) and codes:
        md += _format_codes(codes)
        md.append("")

    relationships = data.get("relationships") or []
    if isinstance(relationships, list) and relationships:
        md += _format_relationships(relationships)
        md.append("")

    properties = data.get("properties") or []
    if isinstance(properties, list) and properties:
        md += _format_properties(properties)
        md.append("")

    return "\n".join(md).rstrip() + "\n"
