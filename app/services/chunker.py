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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.models import VectorDocument


@dataclass
class ChunkerConfig:
    """Configuration for substance chunking."""

    name_batch_size: int = 30
    emit_atomic_name_chunks: bool = False
    emit_sequence_segments: bool = False
    max_sequence_segment_len: int = 300
    emit_full_sequence_in_text: bool = False
    include_admin_validation_notes: bool = False
    include_reference_index_chunk: bool = True
    include_classification_chunk: bool = True
    include_grouped_relationship_summaries: bool = True


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
    ) -> VectorDocument:
        """Create a VectorDocument chunk with parent-child compatible metadata."""
        substance_uuid = str(substance.get("uuid", ""))
        chunk_id = self._chunk_id(substance_uuid, section, chunk_id_suffix)

        # ALTERNATIVE substances attach to their primary substance's UUID
        document_id = self._resolve_root_document_id(substance)

        meta: Dict[str, Any] = {
            "root_section": "overview",
            "chunk_type": section,
            "canonical_name": self._display_name(substance),
            "substance_definition_type": substance.get("definitionType", "PRIMARY"),
            "substance_uuid": substance_uuid,
            **(metadata or {}),
        }

        return self.class_(
            chunk_id=chunk_id,
            document_id=UUID(document_id) if document_id else UUID(int=0),
            section=section,
            text=text,
            embedding=[],
            metadata_json=meta,
            source_url=source_url,
            search_text=text,
        )

    def _build_overview(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build the root/overview chunk for a substance."""
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

        structure = substance.get("structure")
        if isinstance(structure, dict):
            smiles = structure.get("smiles", "")
            formula = structure.get("formula") or structure.get("molecularFormula", "")
            mwt = structure.get("mwt") or structure.get("molecularWeight", "")
            stereo = structure.get("stereochemistry", "")
            if smiles:
                parts.append(f"SMILES: {smiles}")
            if formula:
                parts.append(f"Formula: {formula}")
            if mwt:
                parts.append(f"Molecular Weight: {mwt}")
            if stereo:
                parts.append(f"Stereochemistry: {stereo}")

        # Names summary
        names = substance.get("names") or []
        if names:
            name_texts = []
            for entry in names:
                if isinstance(entry, dict) and entry.get("name"):
                    name_texts.append(str(entry["name"]))
            if name_texts:
                parts.append(f"Names: {', '.join(name_texts[:self.config.name_batch_size])}")

        # Codes summary
        codes = substance.get("codes") or []
        if codes:
            code_texts = []
            for entry in codes:
                if isinstance(entry, dict) and entry.get("code"):
                    code_texts.append(f"{entry.get('codeSystem', '')}:{entry['code']}")
            if code_texts:
                parts.append(f"Codes: {', '.join(code_texts[:20])}")

        # Moieties summary
        moieties = substance.get("moieties") or []
        if moieties:
            moiety_texts = []
            for m in moieties:
                if isinstance(m, dict) and m.get("smiles"):
                    moiety_texts.append(str(m["smiles"]))
            if moiety_texts:
                parts.append(f"Moieties: {', '.join(moiety_texts)}")

        text = "\n".join(parts)
        chunk = self._make_chunk(
            substance,
            section="overview",
            text=text,
            metadata={"chunk_type": "overview"},
        )
        return [chunk]

    def _build_name_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build name chunks from substance names."""
        chunks: List[VectorDocument] = []
        names = substance.get("names") or []
        if not names:
            return chunks

        substance_uuid = str(substance.get("uuid", ""))

        # Batch name chunk
        name_texts = []
        for entry in names:
            if isinstance(entry, dict) and entry.get("name"):
                name_texts.append(str(entry["name"]))

        if name_texts:
            batch_text = f"Names: {', '.join(name_texts[:self.config.name_batch_size])}"
            chunks.append(
                self._make_chunk(
                    substance,
                    section="names",
                    text=batch_text,
                    chunk_id_suffix="batch",
                    metadata={"chunk_type": "name_batch"},
                )
            )

        # Atomic name chunks (if enabled)
        if self.config.emit_atomic_name_chunks:
            for idx, entry in enumerate(names):
                if not isinstance(entry, dict) or not entry.get("name"):
                    continue
                name_text = str(entry["name"])
                name_type = entry.get("type", "")
                languages = entry.get("languages", [])
                lang_str = ", ".join(str(l) for l in languages) if languages else ""
                text_parts = [f"Name: {name_text}"]
                if name_type:
                    text_parts.append(f"Type: {name_type}")
                if lang_str:
                    text_parts.append(f"Languages: {lang_str}")

                chunks.append(
                    self._make_chunk(
                        substance,
                        section="names",
                        text="\n".join(text_parts),
                        chunk_id_suffix=f"atomic_{idx}",
                        metadata={
                            "chunk_type": "name",
                            "name_type": name_type,
                            "name": name_text,
                        },
                    )
                )

        return chunks

    def _build_code_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build code/identifier chunks from substance codes."""
        chunks: List[VectorDocument] = []
        codes = substance.get("codes") or []
        if not codes:
            return chunks

        code_texts = []
        for entry in codes:
            if not isinstance(entry, dict):
                continue
            code = entry.get("code", "")
            code_system = entry.get("codeSystem", "")
            code_type = entry.get("type", "")
            if code:
                code_texts.append(f"{code_system}:{code} ({code_type})")

        if code_texts:
            batch_text = f"Codes: {', '.join(code_texts[:50])}"
            chunks.append(
                self._make_chunk(
                    substance,
                    section="codes",
                    text=batch_text,
                    chunk_id_suffix="batch",
                    metadata={"chunk_type": "code_batch"},
                )
            )

        # Individual code chunks for parent-child grouping
        for idx, entry in enumerate(codes):
            if not isinstance(entry, dict):
                continue
            code = entry.get("code", "")
            if not code:
                continue
            code_system = entry.get("codeSystem", "")
            code_type = entry.get("type", "")
            comments = entry.get("comments", "")
            text_parts = [f"Code: {code}"]
            if code_system:
                text_parts.append(f"System: {code_system}")
            if code_type:
                text_parts.append(f"Type: {code_type}")
            if comments:
                text_parts.append(f"Comments: {comments}")

            chunks.append(
                self._make_chunk(
                    substance,
                    section="codes",
                    text="\n".join(text_parts),
                    chunk_id_suffix=f"item_{idx}",
                    metadata={
                        "chunk_type": "code",
                        "code": code,
                        "code_system": code_system,
                        "code_type": code_type,
                    },
                )
            )

        return chunks

    def _build_class_specific_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Dispatch to class-specific chunk builders based on substanceClass."""
        substance_class = str(substance.get("substanceClass", "")).lower()

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

    def _build_chemical_structure_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build structure-related chunks for chemical substances."""
        chunks: List[VectorDocument] = []
        structure = substance.get("structure")
        if not isinstance(structure, dict):
            return chunks

        parts = []
        smiles = structure.get("smiles", "")
        formula = structure.get("formula") or structure.get("molecularFormula", "")
        mwt = structure.get("mwt") or structure.get("molecularWeight", "")
        stereo = structure.get("stereochemistry", "")
        optical = structure.get("opticalActivity", "")
        atrop = structure.get("atropisomerism", "")

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

        if parts:
            text = "\n".join(parts)
            chunks.append(
                self._make_chunk(
                    substance,
                    section="structure",
                    text=text,
                    metadata={
                        "chunk_type": "structure",
                        "smiles": smiles,
                        "molecular_formula": formula,
                    },
                )
            )

        # Sequence segments (for protein/nucleic substances stored in structure)
        if self.config.emit_sequence_segments:
            sequences = structure.get("sequences") or []
            if not sequences and structure.get("sequence"):
                sequences = [structure["sequence"]]

            for seq_idx, seq in enumerate(sequences):
                seq_str = str(seq) if not isinstance(seq, str) else seq
                if not seq_str:
                    continue

                if self.config.emit_full_sequence_in_text:
                    chunks.append(
                        self._make_chunk(
                            substance,
                            section="sequence",
                            text=f"Sequence: {seq_str}",
                            chunk_id_suffix=f"full_{seq_idx}",
                            metadata={"chunk_type": "sequence_full"},
                        )
                    )

                segment_len = self.config.max_sequence_segment_len
                for seg_idx, start in enumerate(range(0, len(seq_str), segment_len)):
                    segment = seq_str[start : start + segment_len]
                    chunks.append(
                        self._make_chunk(
                            substance,
                            section="sequence",
                            text=f"Sequence segment {seg_idx + 1}: {segment}",
                            chunk_id_suffix=f"seg_{seq_idx}_{seg_idx}",
                            metadata={
                                "chunk_type": "sequence_segment",
                                "segment_index": seg_idx + 1,
                                "segment_start": start,
                            },
                        )
                    )

        return chunks

    def _build_protein_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build structure/composition chunks for protein substances."""
        chunks: List[VectorDocument] = []
        protein = substance.get("protein")
        if not isinstance(protein, dict):
            return chunks

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
                        # Full sequence chunk
                        if self.config.emit_full_sequence_in_text:
                            chunks.append(
                                self._make_chunk(
                                    substance,
                                    section="sequence",
                                    text=f"Subunit {sub_idx}: {seq}",
                                    chunk_id_suffix=f"subunit_{sub_idx}_full",
                                    metadata={
                                        "chunk_type": "protein_sequence",
                                        "subunit_index": sub_idx,
                                    },
                                )
                            )
                        # Segmented sequence
                        if self.config.emit_sequence_segments:
                            segment_len = self.config.max_sequence_segment_len
                            for seg_idx, start in enumerate(range(0, len(seq), segment_len)):
                                segment = seq[start : start + segment_len]
                                chunks.append(
                                    self._make_chunk(
                                        substance,
                                        section="sequence",
                                        text=f"Subunit {sub_idx} segment {seg_idx + 1}: {segment}",
                                        chunk_id_suffix=f"subunit_{sub_idx}_seg_{seg_idx}",
                                        metadata={
                                            "chunk_type": "protein_sequence_segment",
                                            "subunit_index": sub_idx,
                                            "segment_index": seg_idx + 1,
                                        },
                                    )
                                )
                        else:
                            # Truncated summary
                            display_seq = seq[:300] + ("..." if len(seq) > 300 else "")
                            chunks.append(
                                self._make_chunk(
                                    substance,
                                    section="sequence",
                                    text=f"Subunit {sub_idx} Sequence: {display_seq}",
                                    chunk_id_suffix=f"subunit_{sub_idx}_summary",
                                    metadata={
                                        "chunk_type": "protein_sequence_summary",
                                        "subunit_index": sub_idx,
                                    },
                                )
                            )

        modifications = protein.get("modifications")
        if isinstance(modifications, dict):
            for mod_type in ("physicalModifications", "agentModifications", "structuralModifications"):
                mod_list = modifications.get(mod_type)
                if mod_list:
                    parts.append(f"{mod_type}: {len(mod_list)} modification(s)")

        if len(parts) > 1:
            chunks.insert(
                0,
                self._make_chunk(
                    substance,
                    section="structure",
                    text="\n".join(parts),
                    metadata={"chunk_type": "protein", "substance_class": "protein"},
                ),
            )

        return chunks

    def _build_nucleic_acid_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build structure/composition chunks for nucleic acid substances."""
        chunks: List[VectorDocument] = []
        na = substance.get("nucleicAcid")
        if not isinstance(na, dict):
            return chunks

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
            chunks.append(
                self._make_chunk(
                    substance,
                    section="structure",
                    text="\n".join(parts),
                    metadata={"chunk_type": "nucleic_acid", "substance_class": "nucleicAcid"},
                )
            )

        # Subunit sequences
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
                            section="sequence",
                            text=f"Subunit {sub_idx}: {seq}",
                            chunk_id_suffix=f"subunit_{sub_idx}_full",
                            metadata={
                                "chunk_type": "nucleic_acid_sequence",
                                "subunit_index": sub_idx,
                            },
                        )
                    )
                if self.config.emit_sequence_segments:
                    segment_len = self.config.max_sequence_segment_len
                    for seg_idx, start in enumerate(range(0, len(seq), segment_len)):
                        segment = seq[start : start + segment_len]
                        chunks.append(
                            self._make_chunk(
                                substance,
                                section="sequence",
                                text=f"Subunit {sub_idx} segment {seg_idx + 1}: {segment}",
                                chunk_id_suffix=f"subunit_{sub_idx}_seg_{seg_idx}",
                                metadata={
                                    "chunk_type": "nucleic_acid_sequence_segment",
                                    "subunit_index": sub_idx,
                                    "segment_index": seg_idx + 1,
                                },
                            )
                        )
                else:
                    display_seq = seq[:300] + ("..." if len(seq) > 300 else "")
                    chunks.append(
                        self._make_chunk(
                            substance,
                            section="sequence",
                            text=f"Subunit {sub_idx} Sequence: {display_seq}",
                            chunk_id_suffix=f"subunit_{sub_idx}_summary",
                            metadata={
                                "chunk_type": "nucleic_acid_sequence_summary",
                                "subunit_index": sub_idx,
                            },
                        )
                    )

        return chunks

    def _build_polymer_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build structure/composition chunks for polymer substances."""
        chunks: List[VectorDocument] = []
        polymer = substance.get("polymer")
        if not isinstance(polymer, dict):
            return chunks

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
            chunks.append(
                self._make_chunk(
                    substance,
                    section="structure",
                    text="\n".join(parts),
                    metadata={"chunk_type": "polymer", "substance_class": "polymer"},
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
                    chunks.append(
                        self._make_chunk(
                            substance,
                            section="structure",
                            text="\n".join(text_parts),
                            chunk_id_suffix=f"polymer_{key}",
                            metadata={
                                "chunk_type": f"polymer_{key}_structure",
                                "smiles": smi,
                                "molecular_formula": formula,
                            },
                        )
                    )

        return chunks

    def _build_structurally_diverse_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build structure/source-material chunks for structurally diverse substances."""
        chunks: List[VectorDocument] = []
        sd = substance.get("structurallyDiverse")
        if not isinstance(sd, dict):
            return chunks

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
            chunks.append(
                self._make_chunk(
                    substance,
                    section="structure",
                    text="\n".join(parts),
                    metadata={
                        "chunk_type": "structurally_diverse",
                        "substance_class": "structurallyDiverse",
                    },
                )
            )

        return chunks

    def _build_mixture_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build composition chunks for mixture substances."""
        chunks: List[VectorDocument] = []
        mixture = substance.get("mixture")
        if not isinstance(mixture, dict):
            return chunks

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
            chunks.append(
                self._make_chunk(
                    substance,
                    section="structure",
                    text="\n".join(parts),
                    metadata={"chunk_type": "mixture", "substance_class": "mixture"},
                )
            )

        # Individual component chunks for detailed lookup
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
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="composition",
                        text="\n".join(text_parts),
                        chunk_id_suffix=f"component_{idx}",
                        metadata={
                            "chunk_type": "mixture_component",
                            "component_type": ctype,
                            "component_uuid": cuuid,
                        },
                    )
                )

        return chunks

    def _build_tag_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build tag chunks (applicable to any substance class)."""
        chunks: List[VectorDocument] = []
        tags = substance.get("tags") or []
        if tags:
            tag_texts = [str(t) for t in tags]
            chunks.append(
                self._make_chunk(
                    substance,
                    section="tags",
                    text=f"Tags: {', '.join(tag_texts)}",
                    metadata={"chunk_type": "tag_batch"},
                )
            )
        return chunks

    def _build_specified_substance_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build chunks for specified-substance (G1..G4) substances."""
        chunks: List[VectorDocument] = []
        # Specified substances often carry manufacturing/process info in properties/notes
        # Already handled by the generic property/modification chunkers
        ss = substance.get("specifiedSubstance")
        if isinstance(ss, dict):
            grade = ss.get("grade", "")
            if grade:
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="properties",
                        text=f"Specified Substance Grade: {grade}",
                        metadata={"chunk_type": "specified_substance", "substance_class": "specifiedSubstance"},
                    )
                )
        return chunks

    def _build_reference_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build reference chunks."""
        chunks: List[VectorDocument] = []
        references = substance.get("references") or []
        if not references:
            return chunks

        # Reference index chunk (if enabled)
        if self.config.include_reference_index_chunk:
            ref_summaries = []
            for entry in references:
                if not isinstance(entry, dict):
                    continue
                doc_type = entry.get("docType", "")
                ref_id = entry.get("id", "")
                citation = entry.get("citation", "")
                parts = []
                if doc_type:
                    parts.append(doc_type)
                if ref_id:
                    parts.append(f"ID:{ref_id}")
                if citation:
                    parts.append(citation)
                if parts:
                    ref_summaries.append(" ".join(parts))

            if ref_summaries:
                text = f"References ({len(references)} total): {', '.join(ref_summaries[:30])}"
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="references",
                        text=text,
                        chunk_id_suffix="index",
                        metadata={"chunk_type": "reference_index"},
                    )
                )

        # Individual reference chunks
        for idx, entry in enumerate(references):
            if not isinstance(entry, dict):
                continue
            doc_type = entry.get("docType", "")
            ref_id = entry.get("id", "")
            citation = entry.get("citation", "")
            url = entry.get("url", "")
            tags = entry.get("tags", [])

            text_parts = []
            if doc_type:
                text_parts.append(f"Document Type: {doc_type}")
            if ref_id:
                text_parts.append(f"ID: {ref_id}")
            if citation:
                text_parts.append(f"Citation: {citation}")
            if url:
                text_parts.append(f"URL: {url}")
            if tags:
                text_parts.append(f"Tags: {', '.join(str(t) for t in tags)}")

            if text_parts:
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="references",
                        text="\n".join(text_parts),
                        chunk_id_suffix=f"item_{idx}",
                        metadata={
                            "chunk_type": "reference",
                            "doc_type": doc_type,
                            "ref_id": ref_id,
                        },
                    )
                )

        return chunks

    def _build_relationship_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build relationship chunks."""
        chunks: List[VectorDocument] = []
        relationships = substance.get("relationships") or []
        if not relationships:
            return chunks

        if self.config.include_grouped_relationship_summaries:
            # Group by relationship type
            by_type: Dict[str, List[str]] = {}
            for entry in relationships:
                if not isinstance(entry, dict):
                    continue
                rel_type = entry.get("type", "unknown")
                related = entry.get("relatedSubstance") or {}
                if isinstance(related, dict):
                    related_name = related.get("refPname") or related.get("name") or related.get("uuid", "unknown")
                else:
                    related_name = str(related)
                qualification = entry.get("qualification", "")
                entry_text = f"{related_name}"
                if qualification:
                    entry_text += f" ({qualification})"
                by_type.setdefault(rel_type, []).append(entry_text)

            for rel_type, items in by_type.items():
                text = f"Relationships of type '{rel_type}': {', '.join(items[:20])}"
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="relationships",
                        text=text,
                        chunk_id_suffix=f"group_{rel_type}",
                        metadata={
                            "chunk_type": "relationship_group",
                            "relationship_type": rel_type,
                        },
                    )
                )

        # Individual relationship chunks
        for idx, entry in enumerate(relationships):
            if not isinstance(entry, dict):
                continue
            rel_type = entry.get("type", "")
            related = entry.get("relatedSubstance") or {}
            qualification = entry.get("qualification", "")
            if isinstance(related, dict):
                related_name = related.get("refPname") or related.get("name") or related.get("uuid", "")
            else:
                related_name = str(related)

            text_parts = []
            if rel_type:
                text_parts.append(f"Type: {rel_type}")
            if related_name:
                text_parts.append(f"Related Substance: {related_name}")
            if qualification:
                text_parts.append(f"Qualification: {qualification}")

            if text_parts:
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="relationships",
                        text="\n".join(text_parts),
                        chunk_id_suffix=f"item_{idx}",
                        metadata={
                            "chunk_type": "relationship",
                            "relationship_type": rel_type,
                        },
                    )
                )

        return chunks

    def _build_classification_chunks(self, substance: Dict[str, Any]) -> List[VectorDocument]:
        """Build classification/property chunks."""
        chunks: List[VectorDocument] = []

        # Agent modifications / classifications
        modifications = substance.get("modifications") or []
        if modifications:
            mod_texts = []
            for entry in modifications:
                if isinstance(entry, dict):
                    mod_type = entry.get("modificationType", "")
                    if mod_type:
                        mod_texts.append(str(mod_type))
            if mod_texts:
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="modifications",
                        text=f"Modifications: {', '.join(mod_texts)}",
                        metadata={"chunk_type": "modification_batch"},
                    )
                )

        # Properties
        properties = substance.get("properties") or []
        if properties:
            prop_texts = []
            for entry in properties:
                if isinstance(entry, dict):
                    prop_name = entry.get("name", "")
                    prop_value = entry.get("value", "")
                    if prop_name and prop_value:
                        prop_texts.append(f"{prop_name}={prop_value}")
            if prop_texts:
                chunks.append(
                    self._make_chunk(
                        substance,
                        section="properties",
                        text=f"Properties: {', '.join(prop_texts[:30])}",
                        metadata={"chunk_type": "property_batch"},
                    )
                )

        # Classifications
        if self.config.include_classification_chunk:
            classifications = substance.get("classifications") or []
            if classifications:
                class_texts = []
                for entry in classifications:
                    if isinstance(entry, dict):
                        classification = entry.get("classification", "")
                        if classification:
                            class_texts.append(str(classification))
                if class_texts:
                    chunks.append(
                        self._make_chunk(
                            substance,
                            section="classifications",
                            text=f"Classifications: {', '.join(class_texts[:30])}",
                            metadata={"chunk_type": "classification_batch"},
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
        chunks.extend(self._build_classification_chunks(substance))

        return chunks

    def __repr__(self) -> str:
        return f"<SubstanceChunker(class_={self.class_.__name__})>"
