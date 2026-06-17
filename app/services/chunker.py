"""
GSRS MCP Server - Native Chunker Service

Pure-Python substance chunker compatible with parent-child retrieval.
Does not depend on gsrs.services.ai or gsrs.model.

Special handling for ALTERNATIVE substances:
- When ``definitionType`` is ``ALTERNATIVE``, the root document is the
  ``relatedSubstance`` referenced by a ``SUB_ALTERNATE->SUBSTANCE``
  relationship.  The chunk's ``document_id`` is set to that substance's
  ``refuuid`` so all ALTERNATIVE chunks group under the primary substance
  in parent-child retrieval.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.models import VectorDocument


# Section-group mapping for parent-child retrieval.
#
# Top-level sections map to themselves (``root_section == section``).
# Sub-section entries map to a top-level parent; the chunker records that
# relationship in ``metadata.hierarchy`` so the enricher can choose the
# correct parent without having to duplicate this table.
_SECTION_TO_ROOT: Dict[str, str] = {
    # Top-level (root_section = self)
    "overview": "overview",
    "names": "names",
    "codes": "codes",
    "definitions": "definitions",
    "tags": "tags",
    "references": "references",
    "relationships": "relationships",
    "properties": "properties",
    # Sub-sections of ``definitions`` (root_section = "definitions").
    # Each per-class chunk section is the lower-cased substanceClass, so
    # the mapping is open-ended: any substanceClass value becomes a
    # sub-section grouped under ``definitions``.
    "chemical": "definitions",
    "moieties": "definitions",
    "protein": "definitions",
    "nucleicacid": "definitions",
    "polymer": "definitions",
    "structurallydiverse": "definitions",
    "mixture": "definitions",
    "specifiedsubstance": "definitions",
    "modifications": "definitions",
    # Sub-sections of ``codes`` (root_section = "codes").
    # ``identifiers`` carries non-classification codes; ``classifications``
    # carries classification rows. Both live under the same root so a
    # single parent context reconstructs all of the substance's code
    # universe in one query.
    "identifiers": "codes",
    "classifications": "codes",
}

_TOP_LEVEL_SECTIONS = frozenset(
    {
        "overview",
        "names",
        "codes",
        "definitions",
        "tags",
        "references",
        "relationships",
        "properties",
        "notes",
    }
)


def root_section_for(section: str) -> str:
    """Return the top-level root section for a given chunk ``section``.

    Unknown sections fall back to the section's own name so legacy data
    and new section names continue to work without explicit registration.
    """
    if not section:
        return "overview"
    return _SECTION_TO_ROOT.get(section, section)


def is_top_level_section(section: str) -> bool:
    """Return True if ``section`` is itself a root section."""
    return section in _TOP_LEVEL_SECTIONS


def section_for_substance_class(substance_class: str) -> str:
    """Return the chunk ``section`` name for a given ``substanceClass``.

    The substance class becomes its own chunk section (lower-cased), and
    is mapped to the ``definitions`` root so the per-class data lives
    under a single parent context. Unknown classes fall back to the
    lower-cased name so the mapping is open-ended.
    """
    if not substance_class:
        return "definitions"
    return str(substance_class).strip().lower()


# ---------------------------------------------------------------------------
# Access control
#

def access_status_for(value: Any) -> str:
    """Map a GSRS ``access`` field to a ``Public`` / ``Protected`` string.

    Public: field absent, ``None``, ``False``, ``0``, ``""``, or an
    empty list.
    Protected: non-empty list (e.g. ``["protected"]``) or any other
    truthy non-list value (e.g. a future GSRS string/dict access
    payload that hasn't been observed yet).

    The function is intentionally permissive so that unexpected
    payload shapes degrade safely to the more restrictive status.
    """
    if value is None:
        return "Public"
    if isinstance(value, list):
        return "Protected" if len(value) > 0 else "Public"
    return "Protected" if value else "Public"


def sections_in_root(root_section: str) -> List[str]:
    """Return the set of chunk ``section`` values that map to ``root_section``.

    Used by the parent-child enricher to push the section filter down to
    the vector store (PGVector/Chroma), avoiding a full-document fetch
    followed by an in-Python filter. The returned list is the smallest set
    the backend needs to load to reconstruct a parent.
    """
    if not root_section:
        return ["overview"]
    # Always include the root section itself plus any sub-sections whose
    # parent is this root.
    members: List[str] = [root_section]
    for section, parent in _SECTION_TO_ROOT.items():
        if parent == root_section and section not in members:
            members.append(section)
    return members


# ---------------------------------------------------------------------------
# Substance name type resolution
#
# GSRS names carry a short ``type`` code (``of``, ``sys``, ``bn``, ``cn``,
# ``sci``, ``syn``, ``cd``). These are surfaced to retrieval users as a
# human-readable label so the chunk text reads naturally and so the label
# can be embedded as a discrete metadata field for filtering.
NAME_TYPE_LABELS: Dict[str, str] = {
    "of": "Official Name",
    "sys": "Systematic Name",
    "bn": "Brand Name",
    "cn": "Common Name",
    "sci": "Scientific Name",
    "syn": "Synonym",
    "cd": "Code",
}


def name_type_label(name_type: str) -> str:
    """Return a human-readable label for a GSRS ``name.type`` code.

    Unknown codes are returned verbatim so future payload shapes
    surface in retrieval results rather than silently disappearing.
    An empty / missing code returns an empty string.
    """
    if not name_type:
        return ""
    key = str(name_type).strip().lower()
    return NAME_TYPE_LABELS.get(key, str(name_type))


@dataclass
class ChunkerConfig:
    """Configuration for substance chunking.

    The chunker emits one chunk per list item — no batch summary
    chunks and no string/list shrinking — so every payload field
    is independently queryable. The remaining flags control what
    is emitted at all, not how much is collapsed.
    """

    emit_atomic_name_chunks: bool = True
    # Sequences are emitted in full by default — no segmentation,
    # no truncation. Set ``emit_sequence_segments=True`` to chunk
    # long sequences into fixed-length segments instead.
    emit_sequence_segments: bool = False
    emit_full_sequence_in_text: bool = True
    include_classification_chunk: bool = True
    # Curator annotations whose ``note`` text starts with the
    # ``[Validation]`` prefix are produced by the GSRS admin validator
    # (duplicate detection, controlled-vocabulary enforcement, etc.).
    # When this flag is False, those notes are filtered out of the
    # emitted note chunks so they don't dominate the curator's other
    # annotations during retrieval.
    include_admin_validation_notes: bool = False


class SubstanceChunker:
    """
    Chunk a GSRS substance JSON payload into VectorDocument records.

    Produces chunks compatible with parent-child retrieval by embedding
    ``root_section`` inside each chunk's ``metadata_json``.

    ALTERNATIVE substances are re-parented under their primary substance
    (the target of a ``SUB_ALTERNATE->SUBSTANCE`` relationship) so that
    parent-child retrieval treats them as children of the root document.
    """

    def __init__(self, class_: type = VectorDocument, config: Optional[ChunkerConfig] = None):
        self.class_ = class_
        self.config = config or ChunkerConfig()

    @staticmethod
    def _display_name(substance: Dict[str, Any]) -> str:
        """Return the preferred display name for a substance."""
        names = substance.get("names")
        if isinstance(names, list):
            for entry in names:
                if not isinstance(entry, dict):
                    continue
                display = entry.get("displayName", entry.get("display_name", False))
                if isinstance(display, str):
                    display = display.strip().lower() == "true"
                if display and entry.get("name"):
                    return str(entry["name"])
        if substance.get("_name"):
            return str(substance["_name"])
        if substance.get("name"):
            return str(substance["name"])
        return "Unknown"

    @staticmethod
    def _resolve_root_document_id(substance: Dict[str, Any]) -> str:
        """
        Determine the effective document (root) UUID for a substance.

        For PRIMARY substances this is simply ``substance["uuid"]``.
        For ALTERNATIVE substances we locate the ``SUB_ALTERNATE->SUBSTANCE``
        relationship and return the referenced substance's ``refuuid``.  That
        makes every ALTERNATIVE chunk belong to the same parent document as
        the primary substance, which is exactly what parent-child retrieval
        expects.

        Returns:
            The UUID string to use as ``document_id`` for all chunks.
        """
        definition_type = substance.get("definitionType", "")
        if str(definition_type).upper() != "ALTERNATIVE":
            return str(substance.get("uuid", ""))

        relationships = substance.get("relationships") or []
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            if str(rel.get("type", "")).upper() == "SUB_ALTERNATE->SUBSTANCE":
                related = rel.get("relatedSubstance") or {}
                if isinstance(related, dict):
                    refuuid = related.get("refuuid", "")
                    if refuuid:
                        return str(refuuid)

        # Fallback – if no SUB_ALTERNATE->SUBSTANCE relationship exists,
        # treat the substance as its own root document.
        return str(substance.get("uuid", ""))

    @staticmethod
    def _chunk_id(substance_uuid: str, section: str, suffix: str = "") -> str:
        base = f"{section}_{substance_uuid}"
        if suffix:
            return f"{base}_{suffix}"
        return base

    def _make_chunk(
        self,
        substance: Dict[str, Any],
        section: str,
        text: str,
        chunk_id_suffix: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        source_url: Optional[str] = None,
        access_status: Optional[str] = None,
    ) -> VectorDocument:
        """Create a VectorDocument chunk with parent-child compatible metadata."""
        substance_uuid = str(substance.get("uuid", ""))
        chunk_id = self._chunk_id(substance_uuid, section, chunk_id_suffix)

        # ALTERNATIVE substances attach to their primary substance's UUID
        document_id = self._resolve_root_document_id(substance)

        root_section = root_section_for(section)
        # Default to the substance's top-level access. Per-row callers
        # (name/code/per-class) override this with the row's own access.
        if access_status is None:
            access_status = access_status_for(substance.get("access"))
        meta: Dict[str, Any] = {
            "root_section": root_section,
            "chunk_type": section,
            "display_name": self._display_name(substance),
            "substance_definition_type": substance.get("definitionType", "PRIMARY"),
            "substance_uuid": substance_uuid,
            "access_status": access_status,
            **(metadata or {}),
        }

        # Record hierarchy for sub-section chunks so the enricher can pick
        # the right parent without re-deriving the mapping.
        if root_section != section:
            meta["hierarchy"] = {
                "parent_section": root_section,
                "level": 1,
            }

        return self.class_(
            chunk_id=chunk_id,
            document_id=UUID(document_id) if document_id else UUID(int=0),
            section=section,
            root_section=root_section,
            text=text,
            embedding=[],
            metadata_json=meta,
            source_url=source_url,
            search_text=text,
        )

    def _build_overview(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build the root/overview chunk for a substance.

        The overview carries only the high-level identity of the
        substance (display name, class, definition type, status, IDs,
        version). Detailed data — names, codes/identifiers,
        structure/definitions, moieties — is emitted in its own chunk
        section and surfaced via parent-child retrieval.
        """
        parts: List[str] = []
        parts.append(f"Substance: {self._display_name(substance)}")

        substance_class = substance.get("substanceClass", "N/A")
        parts.append(f"Class: {substance_class}")

        definition_type = substance.get("definitionType", "PRIMARY")
        parts.append(f"Definition Type: {definition_type}")

        status = substance.get("status", "N/A")
        parts.append(f"Status: {status}")

        approval_id = substance.get("_approvalIDDisplay") or substance.get("approvalID")
        if approval_id:
            parts.append(f"Approval ID: {approval_id}")

        version = substance.get("version", "")
        if version:
            parts.append(f"Version: {version}")

        text = "\n".join(parts)
        access_list = substance.get("access") or []
        text += f"\nAccess: {access_status_for(access_list)}"
        chunk = self._make_chunk(
            substance,
            section="overview",
            text=text,
            metadata={
                "chunk_type": "overview",
                "access": list(access_list) if isinstance(access_list, list) else [],
            },
            access_status=access_status_for(access_list),
        )
        return [chunk]

    def _build_name_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build name chunks from substance names.

        Emits one ``name`` chunk per substance name — no batch summary
        chunk and no list shrinking — so every name is independently
        queryable.
        """
        chunks: List[VectorDocument] = []
        names = substance.get("names") or []
        if not names:
            return chunks

        substance_uuid = str(substance.get("uuid", ""))

        for idx, entry in enumerate(names):
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            name_text = str(entry["name"])
            name_type = entry.get("type", "")
            name_type_human = name_type_label(name_type)
            languages = entry.get("languages", [])
            lang_str = ", ".join(str(l) for l in languages) if languages else ""
            entry_access = entry.get("access") or []
            entry_access_status = access_status_for(entry_access)

            # Naming organizations (e.g. INN, USAN, INCI) are typically
            # carried on Official Names (``type="of"``). We surface them
            # whenever the field is present so retrieval can answer
            # "which regulatory body assigned this name" without
            # having to walk the raw payload.
            name_orgs_raw = entry.get("nameOrgs") or []
            name_orgs: List[str] = []
            if isinstance(name_orgs_raw, list):
                for org in name_orgs_raw:
                    if isinstance(org, dict):
                        org_name = org.get("nameOrg")
                        if org_name:
                            name_orgs.append(str(org_name))
                    elif org:
                        name_orgs.append(str(org))

            text_parts = [f"Name: {name_text}"]
            if name_type_human:
                text_parts.append(f"Type: {name_type_human}")
            elif name_type:
                # Unknown type code — surface verbatim so retrieval can
                # still resolve the row.
                text_parts.append(f"Type: {name_type}")
            if lang_str:
                text_parts.append(f"Languages: {lang_str}")
            if name_orgs:
                text_parts.append("Naming Organizations: " + ", ".join(name_orgs))
            text_parts.append(f"Access: {entry_access_status}")

            chunks.append(
                self._make_chunk(
                    substance,
                    section="names",
                    text="\n".join(text_parts),
                    chunk_id_suffix=f"atomic_{idx}",
                    metadata={
                        "chunk_type": "name",
                        "name_type": name_type,
                        "name_type_label": name_type_human,
                        "name": name_text,
                        "name_orgs": name_orgs,
                        "access": list(entry_access) if isinstance(entry_access, list) else [],
                    },
                    access_status=entry_access_status,
                )
            )

        return chunks

    def _build_code_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build code/identifier/classification chunks from substance codes.

        Each entry in ``substance.codes`` is emitted as its own chunk
        — no batch summary chunk and no list shrinking — routed to
        one of two sub-sections under the ``codes`` root:

        * ``identifiers`` — regular codes (default).
        * ``classifications`` — rows where ``_isClassification`` is
          truthy, or whose ``comments`` field contains a ``|``
          character. The ``|`` heuristic is a pragmatic marker for
          classification-style rows in payloads that don't carry the
          ``_isClassification`` field.
        """
        chunks: List[VectorDocument] = []
        codes = substance.get("codes") or []
        if not codes:
            return chunks

        ident_idx = 0
        class_idx = 0
        for entry in codes:
            if not isinstance(entry, dict):
                continue
            code = entry.get("code", "")
            if not code:
                continue
            code_system = entry.get("codeSystem", "")
            code_type = entry.get("type", "")
            comments = entry.get("comments") or ""

            if self._is_classification_code(entry):
                if not self.config.include_classification_chunk:
                    continue
                head = f"{code_system}:{code}" if code_system else code
                entry_access = entry.get("access") or []
                entry_access_status = access_status_for(entry_access)
                code_url = entry.get("url") or ""
                text_parts = [f"Classification: {head}"]
                if code_system:
                    text_parts.append(f"Code System: {code_system}")
                if code_url:
                    text_parts.append(f"URL: {code_url}")
                if comments:
                    text_parts.append(f"Comments: {comments}")
                text_parts.append(f"Access: {entry_access_status}")
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="classifications",
                        text="\n".join(text_parts),
                        chunk_id_suffix=f"item_{class_idx}",
                        metadata={
                            "chunk_type": "classification",
                            "code": code,
                            "code_system": code_system,
                            "code_url": code_url,
                            "comments": comments,
                            "access": list(entry_access) if isinstance(entry_access, list) else [],
                        },
                        access_status=entry_access_status,
                    )
                )
                class_idx += 1
            else:
                head = f"{code_system}:{code}" if code_system else code
                if code_type:
                    head += f" ({code_type})"
                entry_access = entry.get("access") or []
                entry_access_status = access_status_for(entry_access)
                code_url = entry.get("url") or ""
                text_parts = [f"Identifier: {head}"]
                if code_system:
                    text_parts.append(f"Code System: {code_system}")
                if code_url:
                    text_parts.append(f"URL: {code_url}")
                if comments:
                    text_parts.append(f"Comments: {comments}")
                text_parts.append(f"Access: {entry_access_status}")
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="identifiers",
                        text="\n".join(text_parts),
                        chunk_id_suffix=f"item_{ident_idx}",
                        metadata={
                            "chunk_type": "identifier",
                            "code": code,
                            "code_system": code_system,
                            "code_type": code_type,
                            "code_url": code_url,
                            "access": list(entry_access) if isinstance(entry_access, list) else [],
                        },
                        access_status=entry_access_status,
                    )
                )
                ident_idx += 1

        return chunks

    @staticmethod
    def _is_classification_code(entry: Dict[str, Any]) -> bool:
        """Return True if a code entry should be treated as a classification.

        A code is a classification when either:

        * the entry carries a truthy ``_isClassification`` field, or
        * its ``comments`` field contains a ``|`` character (a pragmatic
          marker for classification-style rows in payloads that don't
          carry the explicit field).
        """
        flag = entry.get("_isClassification")
        if isinstance(flag, str):
            if flag.strip().lower() in ("true", "1", "yes", "y"):
                return True
        elif flag:
            return True
        comments = entry.get("comments") or ""
        return "|" in str(comments)

    def _build_class_specific_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Dispatch to class-specific chunk builders based on substanceClass.

        Each per-class builder hardcodes its own ``section`` (e.g.
        ``chemical``, ``protein``, ``mixture``); the section is mapped
        to the ``definitions`` root via ``_SECTION_TO_ROOT`` so the
        per-class data lives under a single parent context.
        """
        substance_class = str(substance.get("substanceClass", "")).lower()
        if not substance_class:
            return []

        if substance_class == "chemical":
            return self._build_chemical_structure_chunks(substance)
        if substance_class == "protein":
            return self._build_protein_chunks(substance)
        if substance_class == "nucleicacid":
            return self._build_nucleic_acid_chunks(substance)
        if substance_class == "polymer":
            return self._build_polymer_chunks(substance)
        if substance_class == "structurallydiverse":
            return self._build_structurally_diverse_chunks(substance)
        if substance_class == "mixture":
            return self._build_mixture_chunks(substance)
        if substance_class == "concept":
            return []
        if substance_class.startswith("specifiedsubstance"):
            return self._build_specified_substance_chunks(substance)

        # Unknown class – no class-specific chunks
        return []

    def _build_chemical_structure_chunks(
        self, substance: Dict[str, Any]
    ) -> List[VectorDocument]:
        """Build structure + moieties chunks for chemical substances.

        The chunk ``section`` is hardcoded to ``"chemical"``, which is
        mapped to the ``definitions`` root.

        Two chunk shapes are emitted (both rooted at the ``chemical``
        section):

        1. The molecule's top-level ``structure`` — one summary chunk.
        2. One chunk per ``moieties`` entry — each moiety is a structural
           fragment that the substance ionises or dissociates into (e.g.
           a salt, a hydrate, a counter-ion). Each moiety is emitted as
           a standalone chunk with its own smiles/formula/etc, and a
           batch chunk summarises the full moiety list.

        Sequence data is NOT handled here — that's covered by the
        protein / nucleic-acid class-specific builders via
        ``subunits[*].sequence``.
        """
        section = "chemical"
        chunks: List[VectorDocument] = []
        structure = substance.get("structure")
        structure_access = (
            structure.get("access") if isinstance(structure, dict) else None
        )
        structure_access_status = access_status_for(structure_access)
        structure_access_list = (
            list(structure_access)
            if isinstance(structure_access, list)
            else []
        )

        # ---- 1. Top-level structure summary ---------------------------
        if isinstance(structure, dict):
            parts = ["Chemical Structure"]
            smiles = structure.get("smiles", "")
            formula = structure.get("formula") or structure.get("molecularFormula", "")
            mwt = structure.get("mwt") or structure.get("molecularWeight", "")
            stereo = structure.get("stereochemistry", "")
            optical = structure.get("opticalActivity", "")
            atrop = structure.get("atropisomerism", "")
            inchi = structure.get("_inchi") or ""
            inchi_key = structure.get("_inchiKey") or ""

            if smiles:
                parts.append(f"SMILES: {smiles}")
            if formula:
                parts.append(f"Formula: {formula}")
            if mwt:
                parts.append(f"Molecular Weight: {mwt}")
            if stereo:
                parts.append(f"Stereochemistry: {stereo}")
            if optical:
                parts.append(f"Optical Activity: {optical}")
            if atrop:
                parts.append(f"Atropisomerism: {atrop}")
            if inchi:
                parts.append(f"InChI: {inchi}")
            if inchi_key:
                parts.append(f"InChI Key: {inchi_key}")
            parts.append(f"Access: {structure_access_status}")

            if len(parts) > 2:  # header + at least one fact + Access
                text = "\n".join(parts)
                chunks.append(
                    self._make_chunk(
                        substance,
                        section=section,
                        text=text,
                        metadata={
                            "chunk_type": "structure",
                            "smiles": smiles,
                            "molecular_formula": formula,
                            "inchi": inchi,
                            "inchi_key": inchi_key,
                            "access": structure_access_list,
                        },
                        access_status=structure_access_status,
                    )
                )

        # ---- 2. Moieties ----------------------------------------------
        # A moiety is a structural fragment that the substance ionises
        # or dissociates into. In GSRS payloads, ``moieties`` is an
        # array of objects with the same shape as ``structure``:
        # ``{smiles, formula, mwt, stereochemistry, opticalActivity,
        # atropisomerism, count, ...}``. We emit one chunk per moiety
        # — no batch summary chunk and no list shrinking — so every
        # moiety is independently queryable.
        moieties = substance.get("moieties") or []
        if isinstance(moieties, list) and moieties:
            moiety_entries = [m for m in moieties if isinstance(m, dict)]

            for idx, moiety in enumerate(moiety_entries):
                m_smiles = moiety.get("smiles", "")
                m_formula = (
                    moiety.get("formula") or moiety.get("molecularFormula", "")
                )
                m_mwt = moiety.get("mwt") or moiety.get("molecularWeight", "")
                m_stereo = moiety.get("stereochemistry", "")
                m_optical = moiety.get("opticalActivity", "")
                m_atrop = moiety.get("atropisomerism", "")
                m_count = moiety.get("count", "")
                m_inchi = moiety.get("_inchi") or ""
                m_inchi_key = moiety.get("_inchiKey") or ""

                m_parts = [f"Moiety {idx + 1}"]
                if m_smiles:
                    m_parts.append(f"SMILES: {m_smiles}")
                if m_formula:
                    m_parts.append(f"Formula: {m_formula}")
                if m_mwt:
                    m_parts.append(f"Molecular Weight: {m_mwt}")
                if m_stereo:
                    m_parts.append(f"Stereochemistry: {m_stereo}")
                if m_optical:
                    m_parts.append(f"Optical Activity: {m_optical}")
                if m_atrop:
                    m_parts.append(f"Atropisomerism: {m_atrop}")
                if m_inchi:
                    m_parts.append(f"InChI: {m_inchi}")
                if m_inchi_key:
                    m_parts.append(f"InChI Key: {m_inchi_key}")
                if m_count != "" and m_count is not None:
                    m_parts.append(f"Count: {m_count}")
                m_parts.append(f"Access: {structure_access_status}")

                # Don't emit an empty moiety chunk.
                if len(m_parts) <= 2:
                    continue

                chunks.append(
                    self._make_chunk(
                        substance,
                        section=section,
                        text="\n".join(m_parts),
                        chunk_id_suffix=f"moiety_{idx}",
                        metadata={
                            "chunk_type": "moiety",
                            "moiety_index": idx,
                            "smiles": m_smiles,
                            "molecular_formula": m_formula,
                            "inchi": m_inchi,
                            "inchi_key": m_inchi_key,
                            "count": m_count,
                            "access": structure_access_list,
                        },
                        access_status=structure_access_status,
                    )
                )

        return chunks

    def _build_protein_chunks(
        self, substance: Dict[str, Any]
    ) -> List[VectorDocument]:
        """Build structure/composition chunks for protein substances.

        The chunk ``section`` is hardcoded to ``"protein"``, which is
        mapped to the ``definitions`` root.
        """
        section = "protein"
        chunks: List[VectorDocument] = []
        protein = substance.get("protein")
        if not isinstance(protein, dict):
            return chunks

        protein_access = protein.get("access")
        protein_access_status = access_status_for(protein_access)
        protein_access_list = (
            list(protein_access)
            if isinstance(protein_access, list)
            else []
        )

        # Overview protein chunk
        parts: List[str] = ["Type: Protein"]

        organism = protein.get("organism")
        if organism:
            parts.append(f"Organism: {organism}")

        disulfide = protein.get("disulfideLinks")
        if disulfide:
            parts.append(f"Disulfide Links: {len(disulfide)}")

        glyco = protein.get("glycosylation")
        if isinstance(glyco, dict):
            glyco_parts = []
            for k in ("N-linked", "C-linked", "O-linked"):
                v = glyco.get(k)
                if v:
                    glyco_parts.append(f"{k}: {v}")
            if glyco_parts:
                parts.append("Glycosylation: " + "; ".join(glyco_parts))

        subunits = protein.get("subunits") or []
        if subunits:
            parts.append(f"Subunits: {len(subunits)}")
            for su in subunits:
                if isinstance(su, dict):
                    seq = su.get("sequence", "")
                    sub_idx = su.get("subunitIndex", "")
                    if seq and sub_idx:
                        # One full sequence chunk per subunit — no
                        # truncation, no segmentation. The sequence
                        # lives under the per-class sub-section
                        # (``protein``) so it groups with the rest
                        # of the protein payload rather than as its
                        # own top-level section.
                        if self.config.emit_full_sequence_in_text:
                            chunks.append(
                                self._make_chunk(
                                    substance,
                                    section=section,
                                    text=f"Subunit {sub_idx}: {seq}\nAccess: {protein_access_status}",
                                    chunk_id_suffix=f"subunit_{sub_idx}_full",
                                    metadata={
                                        "chunk_type": "protein_sequence",
                                        "subunit_index": sub_idx,
                                        "access": protein_access_list,
                                    },
                                    access_status=protein_access_status,
                                )
                            )
                        # Segmented sequence — chunks the subunit
                        # into fixed-length segments so each segment
                        # is independently queryable. The full
                        # sequence is preserved across segments.
                        elif self.config.emit_sequence_segments:
                            for seg_idx, start in enumerate(range(0, len(seq), 300)):
                                segment = seq[start : start + 300]
                                chunks.append(
                                    self._make_chunk(
                                        substance,
                                        section=section,
                                        text=f"Subunit {sub_idx} segment {seg_idx + 1}: {segment}\nAccess: {protein_access_status}",
                                        chunk_id_suffix=f"subunit_{sub_idx}_seg_{seg_idx}",
                                        metadata={
                                            "chunk_type": "protein_sequence_segment",
                                            "subunit_index": sub_idx,
                                            "segment_index": seg_idx + 1,
                                            "access": protein_access_list,
                                        },
                                        access_status=protein_access_status,
                                    )
                                )
                        else:
                            chunks.append(
                                self._make_chunk(
                                    substance,
                                    section=section,
                                    text=f"Subunit {sub_idx} Sequence: {seq}\nAccess: {protein_access_status}",
                                    chunk_id_suffix=f"subunit_{sub_idx}_summary",
                                    metadata={
                                        "chunk_type": "protein_sequence_summary",
                                        "subunit_index": sub_idx,
                                        "access": protein_access_list,
                                    },
                                    access_status=protein_access_status,
                                )
                            )

        modifications = protein.get("modifications")
        if isinstance(modifications, dict):
            for mod_type in ("physicalModifications", "agentModifications", "structuralModifications"):
                mod_list = modifications.get(mod_type)
                if mod_list:
                    parts.append(f"{mod_type}: {len(mod_list)} modification(s)")

        if len(parts) > 1:
            parts.append(f"Access: {protein_access_status}")
            chunks.insert(
                0,
                self._make_chunk(
                    substance,
                    section=section,
                    text="\n".join(parts),
                    metadata={
                        "chunk_type": "protein",
                        "substance_class": "protein",
                        "access": protein_access_list,
                    },
                    access_status=protein_access_status,
                ),
            )

        return chunks

    def _build_nucleic_acid_chunks(
        self, substance: Dict[str, Any]
    ) -> List[VectorDocument]:
        """Build structure/composition chunks for nucleic acid substances.

        The chunk ``section`` is hardcoded to ``"nucleicacid"``, which
        is mapped to the ``definitions`` root.
        """
        section = "nucleicacid"
        chunks: List[VectorDocument] = []
        na = substance.get("nucleicAcid")
        if not isinstance(na, dict):
            return chunks

        na_access = na.get("access")
        na_access_status = access_status_for(na_access)
        na_access_list = (
            list(na_access) if isinstance(na_access, list) else []
        )

        parts: List[str] = ["Type: Nucleic Acid"]

        sequence_origin = na.get("sequenceOrigin")
        if sequence_origin:
            parts.append(f"Sequence Origin: {sequence_origin}")

        sequence_type = na.get("sequenceType")
        if sequence_type:
            parts.append(f"Sequence Type: {sequence_type}")

        subunits = na.get("subunits") or []
        if subunits:
            parts.append(f"Subunits: {len(subunits)}")

        linkages = na.get("linkages") or []
        if linkages:
            parts.append(f"Linkages: {len(linkages)}")

        sugars = na.get("sugars") or []
        if sugars:
            sugar_names = [str(s.get("sugar", "")) for s in sugars if isinstance(s, dict) and s.get("sugar")]
            if sugar_names:
                parts.append(f"Sugars: {', '.join(sugar_names)}")

        if len(parts) > 1:
            parts.append(f"Access: {na_access_status}")
            chunks.append(
                self._make_chunk(
                    substance,
                    section=section,
                    text="\n".join(parts),
                    metadata={
                        "chunk_type": "nucleic_acid",
                        "substance_class": "nucleicAcid",
                        "access": na_access_list,
                    },
                    access_status=na_access_status,
                )
            )

        # Subunit sequences — same sub-section as the parent
        # nucleic-acid root so the sequence data lives under
        # ``nucleicacid`` rather than as its own top-level section.
        if subunits:
            for su in subunits:
                if not isinstance(su, dict):
                    continue
                seq = su.get("sequence", "")
                sub_idx = su.get("subunitIndex", "")
                if not seq:
                    continue
                if self.config.emit_full_sequence_in_text:
                    chunks.append(
                        self._make_chunk(
                            substance,
                            section=section,
                            text=f"Subunit {sub_idx}: {seq}\nAccess: {na_access_status}",
                            chunk_id_suffix=f"subunit_{sub_idx}_full",
                            metadata={
                                "chunk_type": "nucleic_acid_sequence",
                                "subunit_index": sub_idx,
                                "access": na_access_list,
                            },
                            access_status=na_access_status,
                        )
                    )
                elif self.config.emit_sequence_segments:
                    for seg_idx, start in enumerate(range(0, len(seq), 300)):
                        segment = seq[start : start + 300]
                        chunks.append(
                            self._make_chunk(
                                substance,
                                section=section,
                                text=f"Subunit {sub_idx} segment {seg_idx + 1}: {segment}\nAccess: {na_access_status}",
                                chunk_id_suffix=f"subunit_{sub_idx}_seg_{seg_idx}",
                                metadata={
                                    "chunk_type": "nucleic_acid_sequence_segment",
                                    "subunit_index": sub_idx,
                                    "segment_index": seg_idx + 1,
                                    "access": na_access_list,
                                },
                                access_status=na_access_status,
                            )
                        )
                else:
                    chunks.append(
                        self._make_chunk(
                            substance,
                            section=section,
                            text=f"Subunit {sub_idx} Sequence: {seq}\nAccess: {na_access_status}",
                            chunk_id_suffix=f"subunit_{sub_idx}_summary",
                            metadata={
                                "chunk_type": "nucleic_acid_sequence_summary",
                                "subunit_index": sub_idx,
                                "access": na_access_list,
                            },
                            access_status=na_access_status,
                        )
                    )

        return chunks

    def _build_polymer_chunks(
        self, substance: Dict[str, Any]
    ) -> List[VectorDocument]:
        """Build structure/composition chunks for polymer substances.

        The chunk ``section`` is hardcoded to ``"polymer"``, which is
        mapped to the ``definitions`` root.
        """
        section = "polymer"
        chunks: List[VectorDocument] = []
        polymer = substance.get("polymer")
        if not isinstance(polymer, dict):
            return chunks

        polymer_access = polymer.get("access")
        polymer_access_status = access_status_for(polymer_access)
        polymer_access_list = (
            list(polymer_access) if isinstance(polymer_access, list) else []
        )

        parts: List[str] = ["Type: Polymer"]

        classification = polymer.get("classification")
        if isinstance(classification, dict):
            polymer_class = classification.get("polymerClass", "")
            geometry = classification.get("polymerGeometry", "")
            subclass = classification.get("polymerSubclass") or []
            if polymer_class:
                parts.append(f"Class: {polymer_class}")
            if geometry:
                parts.append(f"Geometry: {geometry}")
            if subclass:
                parts.append(f"Subclass: {', '.join(str(s) for s in subclass)}")

        monomers = polymer.get("monomers") or []
        if monomers:
            monomer_texts = []
            for m in monomers:
                if isinstance(m, dict):
                    ms = m.get("monomerSubstance") or {}
                    name = ms.get("name") or ms.get("refPname", "")
                    if name:
                        monomer_texts.append(str(name))
            if monomer_texts:
                parts.append(f"Monomers: {', '.join(monomer_texts)}")

        structural_units = polymer.get("structuralUnits") or []
        if structural_units:
            parts.append(f"Structural Units: {len(structural_units)}")

        if len(parts) > 1:
            parts.append(f"Access: {polymer_access_status}")
            chunks.append(
                self._make_chunk(
                    substance,
                    section=section,
                    text="\n".join(parts),
                    metadata={
                        "chunk_type": "polymer",
                        "substance_class": "polymer",
                        "access": polymer_access_list,
                    },
                    access_status=polymer_access_status,
                )
            )

        # Display/idealized structure if available
        display_structure = polymer.get("displayStructure")
        idealized = polymer.get("idealizedStructure")
        for key, struct in (("display", display_structure), ("idealized", idealized)):
            if isinstance(struct, dict):
                smi = struct.get("smiles", "")
                formula = struct.get("formula", "")
                if smi or formula:
                    text_parts = [f"Polymer {key.capitalize()} Structure"]
                    if smi:
                        text_parts.append(f"SMILES: {smi}")
                    if formula:
                        text_parts.append(f"Formula: {formula}")
                    text_parts.append(f"Access: {polymer_access_status}")
                    chunks.append(
                        self._make_chunk(
                            substance,
                            section=section,
                            text="\n".join(text_parts),
                            chunk_id_suffix=f"polymer_{key}",
                            metadata={
                                "chunk_type": f"polymer_{key}_structure",
                                "smiles": smi,
                                "molecular_formula": formula,
                                "access": polymer_access_list,
                            },
                            access_status=polymer_access_status,
                        )
                    )

        return chunks

    def _build_structurally_diverse_chunks(
        self, substance: Dict[str, Any]
    ) -> List[VectorDocument]:
        """Build structure/source-material chunks for structurally diverse substances.

        The chunk ``section`` is hardcoded to ``"structurallydiverse"``,
        which is mapped to the ``definitions`` root.
        """
        section = "structurallydiverse"
        chunks: List[VectorDocument] = []
        sd = substance.get("structurallyDiverse")
        if not isinstance(sd, dict):
            return chunks

        sd_access = sd.get("access")
        sd_access_status = access_status_for(sd_access)
        sd_access_list = (
            list(sd_access) if isinstance(sd_access, list) else []
        )

        parts: List[str] = ["Type: Structurally Diverse"]

        source_class = sd.get("sourceMaterialClass", "")
        source_type = sd.get("sourceMaterialType", "")
        source_state = sd.get("sourceMaterialState", "")
        organism_family = sd.get("organismFamily", "")
        organism_genus = sd.get("organismGenus", "")
        organism_species = sd.get("organismSpecies", "")
        part = sd.get("part")
        infra_type = sd.get("infraSpecificType", "")
        infra_name = sd.get("infraSpecificName", "")

        if source_class:
            parts.append(f"Source Material Class: {source_class}")
        if source_type:
            parts.append(f"Source Material Type: {source_type}")
        if source_state:
            parts.append(f"Source Material State: {source_state}")
        if organism_family:
            parts.append(f"Organism Family: {organism_family}")
        if organism_genus:
            parts.append(f"Organism Genus: {organism_genus}")
        if organism_species:
            parts.append(f"Organism Species: {organism_species}")
        if part:
            if isinstance(part, list):
                parts.append(f"Part: {', '.join(str(p) for p in part)}")
            else:
                parts.append(f"Part: {part}")
        if infra_type:
            parts.append(f"Infra Specific Type: {infra_type}")
        if infra_name:
            parts.append(f"Infra Specific Name: {infra_name}")

        if len(parts) > 1:
            parts.append(f"Access: {sd_access_status}")
            chunks.append(
                self._make_chunk(
                    substance,
                    section=section,
                    text="\n".join(parts),
                    metadata={
                        "chunk_type": "structurally_diverse",
                        "substance_class": "structurallyDiverse",
                        "access": sd_access_list,
                    },
                    access_status=sd_access_status,
                )
            )

        return chunks

    def _build_mixture_chunks(
        self, substance: Dict[str, Any]
    ) -> List[VectorDocument]:
        """Build composition chunks for mixture substances.

        The chunk ``section`` is hardcoded to ``"mixture"``, which is
        mapped to the ``definitions`` root.
        """
        section = "mixture"
        chunks: List[VectorDocument] = []
        mixture = substance.get("mixture")
        if not isinstance(mixture, dict):
            return chunks

        mixture_access = mixture.get("access")
        mixture_access_status = access_status_for(mixture_access)
        mixture_access_list = (
            list(mixture_access) if isinstance(mixture_access, list) else []
        )

        parts: List[str] = ["Type: Mixture"]

        components = mixture.get("components") or []
        if components:
            parts.append(f"Components: {len(components)}")
            comp_texts = []
            for comp in components:
                if isinstance(comp, dict):
                    ctype = comp.get("type", "")
                    csub = comp.get("substance") or {}
                    cname = csub.get("name") or csub.get("refPname", "")
                    if cname:
                        comp_texts.append(f"{cname} ({ctype})")
            if comp_texts:
                parts.append("Composition: " + ", ".join(comp_texts))

        if len(parts) > 1:
            parts.append(f"Access: {mixture_access_status}")
            chunks.append(
                self._make_chunk(
                    substance,
                    section=section,
                    text="\n".join(parts),
                    metadata={
                        "chunk_type": "mixture",
                        "substance_class": "mixture",
                        "access": mixture_access_list,
                    },
                    access_status=mixture_access_status,
                )
            )

        # Individual component chunks for detailed lookup — emitted
        # under the ``mixture`` sub-section rather than a separate
        # ``composition`` section, so the components are surfaced
        # alongside the rest of the mixture payload.
        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                continue
            ctype = comp.get("type", "")
            csub = comp.get("substance") or {}
            cname = csub.get("name") or csub.get("refPname", "")
            cuuid = csub.get("refuuid", "")
            if cname:
                text_parts = [f"Component: {cname}", f"Type: {ctype}"]
                if cuuid:
                    text_parts.append(f"UUID: {cuuid}")
                text_parts.append(f"Access: {mixture_access_status}")
                chunks.append(
                    self._make_chunk(
                        substance,
                        section=section,
                        text="\n".join(text_parts),
                        chunk_id_suffix=f"component_{idx}",
                        metadata={
                            "chunk_type": "mixture_component",
                            "component_type": ctype,
                            "component_uuid": cuuid,
                            "access": mixture_access_list,
                        },
                        access_status=mixture_access_status,
                    )
                )

        return chunks

    def _build_tag_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build tag chunks (applicable to any substance class).

        Emits one ``tag`` chunk per tag — no batch summary chunk and
        no list shrinking.
        """
        chunks: List[VectorDocument] = []
        tags = substance.get("tags") or []
        if not tags:
            return chunks
        for idx, tag in enumerate(tags):
            tag_text = str(tag)
            if not tag_text:
                continue
            chunks.append(
                self._make_chunk(
                    substance,
                    section="tags",
                    text=f"Tag: {tag_text}",
                    chunk_id_suffix=f"item_{idx}",
                    metadata={"chunk_type": "tag", "tag": tag_text},
                )
            )
        return chunks

    def _build_specified_substance_chunks(
        self, substance: Dict[str, Any], section: str = "specifiedsubstance"
    ) -> List[VectorDocument]:
        """Build chunks for specified-substance (G1..G4) substances.

        All G-variants (``specifiedSubstanceG1`` … ``specifiedSubstanceG4``,
        ``specifiedSubstanceGroup1`` …) emit under the registered
        ``specifiedsubstance`` sub-section, which is mapped to the
        ``definitions`` root via ``_SECTION_TO_ROOT``. The per-variant
        substance class is preserved in ``metadata.substance_class`` so
        callers can still distinguish G1 from G2 etc.

        The builder emits a single constituents summary chunk when
        ``specifiedSubstance.constituents`` is a non-empty list — the
        typical G1 case, where the substance is built from a list of
        referenced sub-substances. ``grade`` is not part of the GSRS
        ``specifiedSubstance`` payload and is intentionally ignored.
        """
        # Normalise: any G-variant collapses to the registered
        # ``specifiedsubstance`` sub-section. Don't trust the
        # caller's ``section`` arg or the raw substanceClass — they
        # would otherwise leak unregistered section values like
        # ``specifiedsubstancegroup1``.
        section = "specifiedsubstance"
        substance_class = str(substance.get("substanceClass", "specifiedSubstance"))
        chunks: List[VectorDocument] = []
        ss = substance.get("specifiedSubstance")
        if not isinstance(ss, dict):
            return chunks

        ss_access = ss.get("access")
        ss_access_status = access_status_for(ss_access)
        ss_access_list = (
            list(ss_access) if isinstance(ss_access, list) else []
        )

        constituents = ss.get("constituents") or []
        if not (isinstance(constituents, list) and constituents):
            return chunks
        # Build a human-readable summary of the constituents so the
        # chunk is queryable ("G1 of X made from constituents A, B, C").
        constituent_entries = [c for c in constituents if isinstance(c, dict)]
        if not constituent_entries:
            return chunks
        parts: List[str] = [
            f"Specified Substance G1 Constituents ({len(constituent_entries)}):"
        ]
        for idx, c in enumerate(constituent_entries, start=1):
            role = c.get("role", "")
            sub_ref = c.get("substance") or {}
            ref_name = (
                sub_ref.get("name")
                or sub_ref.get("refPname")
                or sub_ref.get("approvalID")
                or ""
            )
            ref_uuid = sub_ref.get("refuuid") or sub_ref.get("uuid") or ""
            head = f"  #{idx}"
            if ref_name:
                head += f" {ref_name}"
            if role:
                head += f" (role: {role})"
            if ref_uuid:
                head += f" [uuid: {ref_uuid}]"
            parts.append(head)
        parts.append(f"Access: {ss_access_status}")
        chunks.append(
            self._make_chunk(
                substance,
                section=section,
                text="\n".join(parts),
                chunk_id_suffix="constituents",
                metadata={
                    "chunk_type": "specified_substance_constituents",
                    "constituent_count": len(constituent_entries),
                    "substance_class": substance_class,
                    "access": ss_access_list,
                },
                access_status=ss_access_status,
            )
        )
        return chunks

    def _build_reference_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build reference chunks.

        Emits one ``reference`` chunk per entry — no batch summary
        chunk and no list shrinking — so every reference is
        independently queryable.
        """
        chunks: List[VectorDocument] = []
        references = substance.get("references") or []
        if not references:
            return chunks

        for idx, entry in enumerate(references):
            if not isinstance(entry, dict):
                continue
            doc_type = entry.get("docType", "")
            ref_id = entry.get("id", "")
            citation = entry.get("citation", "")
            ref_url = entry.get("url") or ""
            tags = entry.get("tags") or []
            text_parts: List[str] = ["Reference"]
            if doc_type:
                text_parts.append(f"Type: {doc_type}")
            if ref_id:
                text_parts.append(f"ID: {ref_id}")
            if citation:
                text_parts.append(f"Citation: {citation}")
            if ref_url:
                text_parts.append(f"URL: {ref_url}")
            if tags:
                text_parts.append(
                    "Tags: " + ", ".join(str(t) for t in tags)
                )
            chunks.append(
                self._make_chunk(
                    substance,
                    section="references",
                    text="\n".join(text_parts),
                    chunk_id_suffix=f"item_{idx}",
                    metadata={
                        "chunk_type": "reference",
                        "doc_type": doc_type,
                        "reference_id": ref_id,
                        "reference_url": ref_url,
                    },
                )
            )

        return chunks

    def _build_relationship_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build relationship chunks.

        Emits one ``relationship`` chunk per entry — no batch
        summary chunk and no list shrinking — so every relationship
        is independently queryable.
        """
        chunks: List[VectorDocument] = []
        relationships = substance.get("relationships") or []
        if not relationships:
            return chunks

        for idx, entry in enumerate(relationships):
            if not isinstance(entry, dict):
                continue
            rel_type = entry.get("type", "unknown")
            related = entry.get("relatedSubstance") or {}
            if isinstance(related, dict):
                related_name = (
                    related.get("refPname")
                    or related.get("name")
                    or related.get("uuid", "unknown")
                )
            else:
                related_name = str(related)
            qualification = entry.get("qualification", "")
            interaction_type = entry.get("interactionType", "")

            text_parts = [
                f"Relationship: {rel_type}",
                f"Related Substance: {related_name}",
            ]
            if qualification:
                text_parts.append(f"Qualification: {qualification}")
            if interaction_type:
                text_parts.append(f"Interaction Type: {interaction_type}")
            chunks.append(
                self._make_chunk(
                    substance,
                    section="relationships",
                    text="\n".join(text_parts),
                    chunk_id_suffix=f"item_{idx}",
                    metadata={
                        "chunk_type": "relationship",
                        "relationship_type": rel_type,
                        "related_substance_name": str(related_name),
                    },
                )
            )

        return chunks

    @staticmethod
    def _format_property_value(value: Any) -> str:
        """Render a property's ``value`` field as a queryable string.

        Real GSRS payloads store property values as nested objects
        (``{average, low, high, units}`` or ``{nonNumericValue}``);
        the synthetic test fixture stores them as plain strings. This
        helper flattens both shapes into a single human-readable form
        so the chunk text is actually useful for retrieval.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            # Numeric range
            low = value.get("low")
            high = value.get("high")
            average = value.get("average")
            units = value.get("units", "")
            non_numeric = value.get("nonNumericValue")
            numeric_value = value.get("numericValue")
            if low is not None and high is not None:
                body = f"{low}-{high}"
            elif average is not None:
                body = str(average)
            elif non_numeric is not None:
                body = str(non_numeric)
            elif numeric_value is not None:
                body = str(numeric_value)
            else:
                return ""
            if units:
                body += f" {units}"
            return body
        return str(value)

    def _build_property_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build property chunks from ``substance.properties``.

        Real GSRS payloads store property values as nested objects
        (``{average, low, high, units}`` or ``{nonNumericValue}``);
        the previous implementation stringified the value as a
        ``{key: value}`` dict-repr and silently dropped every property
        whose value wasn't a string, which caused 100% of ibuprofen's
        pharmacokinetic properties to be lost.

        We now emit one ``property`` chunk per entry — no batch
        summary chunk and no list shrinking — using the flattened
        value so each chunk is independently queryable.
        """
        chunks: List[VectorDocument] = []
        properties = substance.get("properties") or []
        if not properties:
            return chunks

        for idx, entry in enumerate(properties):
            if not isinstance(entry, dict):
                continue
            prop_name = str(entry.get("name", "")).strip()
            if not prop_name:
                continue
            prop_type = str(entry.get("propertyType", "")).strip()
            formatted = self._format_property_value(entry.get("value"))
            if not formatted:
                continue

            text_parts = [f"Property: {prop_name}"]
            if prop_type:
                text_parts.append(f"Type: {prop_type}")
            text_parts.append(f"Value: {formatted}")
            chunks.append(
                self._make_chunk(
                    substance,
                    section="properties",
                    text="\n".join(text_parts),
                    chunk_id_suffix=f"item_{idx}",
                    metadata={
                        "chunk_type": "property",
                        "property_name": prop_name,
                        "property_type": prop_type,
                    },
                )
            )
        return chunks

    def _build_note_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build note chunks from ``substance.notes``.

        GSRS notes carry free-form text — validation warnings, duplicate
        detection messages, curation history. They were previously
        dropped entirely. We emit one ``note`` chunk per entry — no
        batch summary chunk and no list shrinking — so every note is
        independently queryable.

        Admin validation notes (those whose ``note`` text starts with
        ``[Validation]``) are produced by the GSRS admin validator and
        are extremely numerous on real payloads (e.g. ibuprofen carries
        25 such notes). They are included by default, but can be
        filtered out via ``ChunkerConfig.include_admin_validation_notes``
        when the caller wants to focus on human-curated annotations
        only.
        """
        chunks: List[VectorDocument] = []
        notes = substance.get("notes") or []
        if not notes:
            return chunks
        for idx, entry in enumerate(notes):
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("note", "")).strip()
            if not text:
                continue
            if (
                not self.config.include_admin_validation_notes
                and self._is_admin_validation_note(text)
            ):
                continue
            chunks.append(
                self._make_chunk(
                    substance,
                    section="notes",
                    text=text,
                    chunk_id_suffix=f"item_{idx}",
                    metadata={
                        "chunk_type": "note",
                    },
                )
            )
        return chunks

    @staticmethod
    def _is_admin_validation_note(text: str) -> bool:
        """Return True if ``text`` is a GSRS admin validation note.

        GSRS admin validation notes are emitted by the validator
        (duplicate detection, controlled-vocabulary enforcement, etc.)
        and are prefixed with ``[Validation]``. They are distinct
        from human-curated annotations and are extremely numerous on
        real payloads, so callers may want to filter them out of the
        emitted note chunk.
        """
        return text.startswith("[Validation]")

    def _build_modification_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build modification chunks from ``substance.modifications``.

        Modifications are stored as a flat list of
        ``{modificationType, ...}`` records in the GSRS payload.
        They live in their own sub-section of ``definitions`` (root
        ``definitions``). We emit one ``modification`` chunk per
        entry — no batch summary chunk and no list shrinking — so
        every modification is independently queryable.
        """
        chunks: List[VectorDocument] = []
        modifications = substance.get("modifications") or []
        if not modifications:
            return chunks
        for idx, entry in enumerate(modifications):
            if not isinstance(entry, dict):
                continue
            mod_type = entry.get("modificationType", "")
            if not mod_type:
                continue
            text_parts = [f"Modification: {mod_type}"]
            for opt_field in ("agent", "amount", "agentModificationRole", "physicalRole"):
                if entry.get(opt_field):
                    text_parts.append(f"{opt_field.capitalize()}: {entry[opt_field]}")
            chunks.append(
                self._make_chunk(
                    substance,
                    section="modifications",
                    text="\n".join(text_parts),
                    chunk_id_suffix=f"item_{idx}",
                    metadata={
                        "chunk_type": "modification",
                        "modification_type": str(mod_type),
                    },
                )
            )
        return chunks

    def chunk(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """
        Chunk a GSRS substance JSON payload into VectorDocument records.

        Args:
            substance: GSRS substance JSON payload (dict)

        Returns:
            List of VectorDocument chunks compatible with parent-child retrieval.
        """
        if not substance or not isinstance(substance, dict):
            return []

        chunks: List[VectorDocument] = []
        chunks.extend(self._build_overview(substance))
        chunks.extend(self._build_name_chunks(substance))
        chunks.extend(self._build_code_chunks(substance))
        chunks.extend(self._build_class_specific_chunks(substance))
        chunks.extend(self._build_reference_chunks(substance))
        chunks.extend(self._build_relationship_chunks(substance))
        chunks.extend(self._build_tag_chunks(substance))
        chunks.extend(self._build_property_chunks(substance))
        chunks.extend(self._build_note_chunks(substance))
        chunks.extend(self._build_modification_chunks(substance))

        return chunks

    def __repr__(self) -> str:
        return f"<SubstanceChunker(class_={self.class_.__name__})>"
