"""
GSRS MCP Server - Chunking Service Tests

Tests for the native ChunkerService using gsrs.model library's
Substance.to_embedding_chunks() method.
"""
import unittest
from typing import Dict, List

from app.models import VectorDocument
from app.services.chunker import (
    ChunkerConfig,
    SubstanceChunker,
    access_status_for,
    is_top_level_section,
    name_type_label,
    root_section_for,
    section_for_substance_class,
    sections_in_root,
)


class TestChunkerService(unittest.TestCase):
    """Tests for the native SubstanceChunker."""

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    def test_chunker_initialization(self):
        """Test chunker can be initialized."""
        self.assertIsNotNone(self._chunker)

    def test_chemical_substance_chunking(self):
        """Test chemical substance chunking with minimal valid data."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Aspirin",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"],
                }
            ],
            "references": [{"docType": "journal article", "id": "12345"}],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No",
            },
            "moieties": [
                {
                    "smiles": "CC(=O)O",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No",
                }
            ],
        }

        chunks = self._chunker.chunk(substance)

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, VectorDocument) for c in chunks))

        root_chunks = [c for c in chunks if c.section == "overview"]
        self.assertGreater(len(root_chunks), 0)
        self.assertIn("Aspirin", root_chunks[0].text)

        # Verify document_id matches substance uuid
        self.assertEqual(
            str(root_chunks[0].document_id),
            "12345678-1234-5678-1234-567812345678",
        )

    def test_concept_substance_chunking(self):
        """Test concept substance chunking."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "concept",
            "names": [
                {
                    "name": "Test Concept",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"],
                }
            ],
            "references": [{"docType": "journal article", "id": "12345"}],
            "version": "1.0",
            "status": "Active",
        }

        chunks = self._chunker.chunk(substance)

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, VectorDocument) for c in chunks))

        root_chunks = [c for c in chunks if c.section == "overview"]
        self.assertGreater(len(root_chunks), 0)
        self.assertIn("Test Concept", root_chunks[0].text)

    def test_names_chunking(self):
        """Test that names are properly chunked."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Aspirin",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"],
                },
                {
                    "name": "Acetylsalicylic acid",
                    "type": "SYSTEMATIC",
                    "languages": ["en"],
                },
            ],
            "references": [{"docType": "journal article", "id": "12345"}],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No",
            },
            "moieties": [
                {
                    "smiles": "CC(=O)O",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No",
                }
            ],
        }

        chunks = self._chunker.chunk(substance)

        name_chunks = [c for c in chunks if "name" in c.section.lower()]
        self.assertGreater(len(name_chunks), 0)

        all_text = " ".join([c.text for c in chunks])
        self.assertIn("Aspirin", all_text)
        self.assertIn("Acetylsalicylic acid", all_text)

    def test_codes_chunking(self):
        """Test that codes are properly chunked."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Test Substance",
                    "type": "COMMON",
                    "displayName": True,
                    "preferred": True,
                    "languages": ["en"],
                }
            ],
            "codes": [{"code": "ASA-101", "codeSystem": "INTERNAL", "type": "SYSTEM"}],
            "references": [{"docType": "journal article", "id": "12345"}],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CCO",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No",
            },
            "moieties": [
                {
                    "smiles": "CCO",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No",
                }
            ],
        }

        chunks = self._chunker.chunk(substance)

        # Code info is included in overview and identifier sections
        all_text = " ".join([c.text for c in chunks])
        self.assertIn("ASA-101", all_text)


class TestAlternativeSubstanceChunking(unittest.TestCase):
    """Tests for ALTERNATIVE substance chunking with parent-child retrieval."""

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    def test_alternative_document_id_is_reparented(self):
        """
        ALTERNATIVE substances must use the primary substance UUID as document_id.
        """
        alternative_substance = {
            "uuid": "d4ee19a6-a33f-4d37-bf05-5feeaa938b83",
            "definitionType": "ALTERNATIVE",
            "definitionLevel": "COMPLETE",
            "substanceClass": "chemical",
            "status": "alternative",
            "names": [],
            "relationships": [
                {
                    "uuid": "b9a9b4c7-9865-419f-9e48-725d8035056a",
                    "type": "SUB_ALTERNATE->SUBSTANCE",
                    "relatedSubstance": {
                        "uuid": "39a8971e-17da-40c7-a532-79887c098d6e",
                        "refPname": "2-Methylcinnamic acid",
                        "refuuid": "34a6ff49-23f3-4079-8d1a-544146ac6d62",
                        "substanceClass": "reference",
                        "approvalID": "3KLP72K23V",
                        "name": "2-Methylcinnamic acid",
                        "linkingID": "3KLP72K23V",
                    },
                }
            ],
        }

        chunks = self._chunker.chunk(alternative_substance)
        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            # All chunks for ALTERNATIVE substances must belong to the primary
            # substance (the target of SUB_ALTERNATE->SUBSTANCE)
            self.assertEqual(
                str(chunk.document_id),
                "34a6ff49-23f3-4079-8d1a-544146ac6d62",
                f"Chunk {chunk.chunk_id} has wrong document_id",
            )
            # But chunk_id should retain the ALTERNATIVE substance's UUID
            self.assertIn("d4ee19a6-a33f-4d37-bf05-5feeaa938b83", chunk.chunk_id)
            # Metadata should indicate this is an ALTERNATIVE
            self.assertEqual(
                chunk.metadata_json.get("substance_definition_type"),
                "ALTERNATIVE",
            )
            self.assertEqual(
                chunk.metadata_json.get("substance_uuid"),
                "d4ee19a6-a33f-4d37-bf05-5feeaa938b83",
            )

    def test_primary_substance_document_id_unchanged(self):
        """PRIMARY substances should keep their own UUID as document_id."""
        primary_substance = {
            "uuid": "34a6ff49-23f3-4079-8d1a-544146ac6d62",
            "definitionType": "PRIMARY",
            "substanceClass": "chemical",
            "status": "approved",
            "names": [
                {"name": "2-Methylcinnamic acid", "displayName": True}
            ],
        }

        chunks = self._chunker.chunk(primary_substance)
        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            self.assertEqual(
                str(chunk.document_id),
                "34a6ff49-23f3-4079-8d1a-544146ac6d62",
            )
            self.assertEqual(
                chunk.metadata_json.get("substance_definition_type"),
                "PRIMARY",
            )

    def test_alternative_without_relationship_fallback(self):
        """
        If an ALTERNATIVE substance lacks a SUB_ALTERNATE->SUBSTANCE
        relationship, it should fall back to its own UUID.
        """
        orphan_alternative = {
            "uuid": "aaaa1111-2222-3333-4444-555566667777",
            "definitionType": "ALTERNATIVE",
            "substanceClass": "chemical",
            "status": "alternative",
            "names": [],
            "relationships": [],
        }

        chunks = self._chunker.chunk(orphan_alternative)
        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            self.assertEqual(
                str(chunk.document_id),
                "aaaa1111-2222-3333-4444-555566667777",
            )

    def test_alternative_substance_overview(self):
        """Overview chunk for ALTERNATIVE substance should include definitionType."""
        alternative_substance = {
            "uuid": "d4ee19a6-a33f-4d37-bf05-5feeaa938b83",
            "definitionType": "ALTERNATIVE",
            "definitionLevel": "COMPLETE",
            "substanceClass": "chemical",
            "status": "alternative",
            "names": [
                {"name": "Alternate Name 1", "displayName": True},
            ],
            "relationships": [
                {
                    "type": "SUB_ALTERNATE->SUBSTANCE",
                    "relatedSubstance": {
                        "refuuid": "34a6ff49-23f3-4079-8d1a-544146ac6d62",
                        "refPname": "2-Methylcinnamic acid",
                    },
                }
            ],
        }

        chunks = self._chunker.chunk(alternative_substance)
        overview_chunks = [c for c in chunks if c.section == "overview"]
        self.assertEqual(len(overview_chunks), 1)

        overview_text = overview_chunks[0].text
        self.assertIn("Definition Type: ALTERNATIVE", overview_text)
        self.assertIn("Alternate Name 1", overview_text)


class TestClassSpecificChunking(unittest.TestCase):
    """Tests for substance-class-specific chunking."""

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    def test_protein_chunking(self):
        """Test protein substance chunking with subunit sequences."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "protein",
            "names": [{"name": "Elfritatug", "displayName": True}],
            "status": "approved",
            "protein": {
                "organism": "CHO cells",
                "disulfideLinks": [{"sitesShorthand": "H22-L88"}],
                "glycosylation": {"N-linked": "G0F, G1F", "O-linked": ""},
                "subunits": [
                    {
                        "subunitIndex": "1",
                        "sequence": "QVQLVQSGGGVVQPGRSLRLSCKASGYTFTSYWMHWVRQAPGKGLEWVGFIRY",
                    },
                    {
                        "subunitIndex": "2",
                        "sequence": "DIQMTQSPSSLSASVGDRVTITCKASQDINKYIAWYQQKPGKAPKLLIY",
                    },
                ],
                "modifications": {
                    "physicalModifications": [],
                    "agentModifications": [],
                    "structuralModifications": [],
                },
            },
        }

        chunks = self._chunker.chunk(substance)
        protein_chunks = [c for c in chunks if c.section == "protein"]
        self.assertGreater(len(protein_chunks), 0)
        self.assertIn("Protein", protein_chunks[0].text)
        self.assertIn("CHO cells", protein_chunks[0].text)
        self.assertIn("Subunits: 2", protein_chunks[0].text)

        # Sequence chunks are emitted under the ``protein`` sub-section
        # (root ``definitions``). One full-sequence chunk per subunit
        # — no truncation, no segmentation by default.
        seq_chunks = [c for c in chunks if c.section == "protein" and "Subunit " in c.text]
        self.assertEqual(len(seq_chunks), 2)
        self.assertTrue(any("QVQLVQSGGGVVQPGRSLRLSCKASGYTFTSYWMHWVRQAPGKGLEWVGFIRY" in c.text for c in seq_chunks))
        self.assertTrue(any("DIQMTQSPSSLSASVGDRVTITCKASQDINKYIAWYQQKPGKAPKLLIY" in c.text for c in seq_chunks))

    def test_nucleic_acid_chunking(self):
        """Test nucleic acid substance chunking."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "nucleicAcid",
            "names": [{"name": "Test DNA", "displayName": True}],
            "status": "approved",
            "nucleicAcid": {
                "sequenceOrigin": "Synthetic",
                "sequenceType": "DNA",
                "subunits": [
                    {
                        "subunitIndex": "1",
                        "sequence": "ATCGATCGATCGATCG",
                    }
                ],
                "linkages": [{"linkage": "phosphodiester"}],
                "sugars": [{"sugar": "deoxyribose"}],
            },
        }

        chunks = self._chunker.chunk(substance)
        nucleic_acid_chunks = [c for c in chunks if c.section == "nucleicacid"]
        self.assertGreater(len(nucleic_acid_chunks), 0)
        self.assertIn("Nucleic Acid", nucleic_acid_chunks[0].text)
        self.assertIn("Synthetic", nucleic_acid_chunks[0].text)

        seq_chunks = [c for c in chunks if c.section == "nucleicacid" and "Subunit" in c.text]
        # Full sequence is emitted (no truncation).
        self.assertTrue(any("ATCGATCGATCGATCG" in c.text for c in seq_chunks))

    def test_polymer_chunking(self):
        """Test polymer substance chunking with monomers."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "polymer",
            "names": [{"name": "Test Polymer", "displayName": True}],
            "status": "approved",
            "polymer": {
                "classification": {
                    "polymerClass": "HOMOPOLYMER",
                    "polymerGeometry": "LINEAR",
                    "polymerSubclass": ["SUBSTITUTED"],
                },
                "monomers": [
                    {
                        "monomerSubstance": {
                            "refPname": "Glucose",
                            "refuuid": "glucose-uuid",
                        }
                    }
                ],
                "structuralUnits": [],
            },
        }

        chunks = self._chunker.chunk(substance)
        polymer_chunks = [c for c in chunks if c.section == "polymer"]
        self.assertGreater(len(polymer_chunks), 0)
        self.assertIn("Polymer", polymer_chunks[0].text)
        self.assertIn("HOMOPOLYMER", polymer_chunks[0].text)
        self.assertIn("LINEAR", polymer_chunks[0].text)
        self.assertIn("Glucose", polymer_chunks[0].text)

    def test_structurally_diverse_chunking(self):
        """Test structurally diverse substance chunking."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "structurallyDiverse",
            "names": [{"name": "Tovesangene Parvec", "displayName": True}],
            "status": "approved",
            "structurallyDiverse": {
                "sourceMaterialClass": "ORGANISM",
                "sourceMaterialType": "RECOMBINANT VIRUS",
                "sourceMaterialState": "LIVE NON-REPLICATING GENETICALLY MODIFIED",
                "organismFamily": "Parvoviridae",
                "organismGenus": "Dependoparvovirus",
                "organismSpecies": "adeno-associated dependoparvovirus a",
                "part": ["whole"],
                "infraSpecificType": "RECOMBINANT VECTOR",
                "infraSpecificName": "AAV9-miniSHANK3",
            },
        }

        chunks = self._chunker.chunk(substance)
        sd_chunks = [c for c in chunks if c.section == "structurallydiverse"]
        self.assertGreater(len(sd_chunks), 0)
        text = sd_chunks[0].text
        self.assertIn("Structurally Diverse", text)
        self.assertIn("RECOMBINANT VIRUS", text)
        self.assertIn("Parvoviridae", text)
        self.assertIn("AAV9-miniSHANK3", text)

    def test_mixture_chunking(self):
        """Test mixture substance chunking."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "mixture",
            "names": [{"name": "2-Methylcinnamic acid", "displayName": True}],
            "status": "approved",
            "mixture": {
                "components": [
                    {
                        "type": "MUST_BE_PRESENT",
                        "substance": {
                            "refPname": "trans-isomer",
                            "refuuid": "trans-uuid",
                        },
                    },
                    {
                        "type": "MUST_BE_PRESENT",
                        "substance": {
                            "refPname": "cis-isomer",
                            "refuuid": "cis-uuid",
                        },
                    },
                ]
            },
        }

        chunks = self._chunker.chunk(substance)
        mixture_chunks = [c for c in chunks if c.section == "mixture"]
        self.assertGreater(len(mixture_chunks), 0)
        self.assertIn("Mixture", mixture_chunks[0].text)
        self.assertIn("Components: 2", mixture_chunks[0].text)

        # Mixture component chunks are now emitted under the
        # ``mixture`` sub-section (root ``definitions``) rather than as
        # a separate ``composition`` top-level section.
        comp_chunks = [c for c in chunks if c.section == "mixture" and c.metadata_json.get("chunk_type") == "mixture_component"]
        self.assertEqual(len(comp_chunks), 2)
        self.assertIn("trans-isomer", comp_chunks[0].text)
        self.assertIn("cis-isomer", comp_chunks[1].text)

    def test_concept_chunking(self):
        """Test concept substance chunking (tags)."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "concept",
            "names": [{"name": "Test Concept", "displayName": True}],
            "status": "approved",
            "tags": ["clinical", "inactive"],
        }

        chunks = self._chunker.chunk(substance)
        tag_chunks = [c for c in chunks if c.section == "tags"]
        # Per-item tag chunks — no batch summary.
        self.assertEqual(len(tag_chunks), 2)
        self.assertIn("clinical", tag_chunks[0].text)
        self.assertIn("inactive", tag_chunks[1].text)

    def test_chemical_fallback(self):
        """Test that unknown classes produce no class-specific structure chunks."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "unknownNewClass",
            "names": [{"name": "Test Unknown", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CCO",
                "molecularFormula": "C2H6O",
                "molecularWeight": 46.07,
            },
        }

        chunks = self._chunker.chunk(substance)
        # Unknown class returns [] from _build_class_specific_chunks
        chemical_chunks = [c for c in chunks if c.section == "unknownnewclass"]
        self.assertEqual(len(chemical_chunks), 0)

        # Overview should still work
        overview_chunks = [c for c in chunks if c.section == "overview"]
        self.assertGreater(len(overview_chunks), 0)
        self.assertIn("Test Unknown", overview_chunks[0].text)


class TestRootSectionAndHierarchy(unittest.TestCase):
    """Verify that the chunker emits a meaningful root_section + hierarchy."""

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    def test_top_level_section_is_its_own_root(self):
        """Top-level chunks must have root_section equal to their section."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "chemical",
            "names": [
                {"name": "Aspirin", "type": "COMMON", "displayName": True}
            ],
            "codes": [{"code": "ASA-101", "codeSystem": "INTERNAL", "type": "SYSTEM"}],
            "references": [{"docType": "journal article", "id": "1"}],
            "version": "1.0",
            "status": "Active",
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No",
            },
            "moieties": [
                {
                    "smiles": "CC(=O)O",
                    "stereochemistry": "ACHIRAL",
                    "opticalActivity": "NONE",
                    "atropisomerism": "No",
                }
            ],
        }

        chunks = self._chunker.chunk(substance)

        # top-level sections (no `hierarchy` key in metadata)
        # Note: ``identifiers``, ``classifications``, ``chemical``, etc.
        # are sub-sections of a top-level root, not roots themselves.
        for section in ("overview", "names", "codes", "definitions", "tags", "references"):
            section_chunks = [c for c in chunks if c.section == section]
            if not section_chunks:
                continue
            for c in section_chunks:
                self.assertEqual(
                    c.root_section,
                    section,
                    f"{section} chunk should have root_section={section}, got {c.root_section}",
                )
                self.assertEqual(
                    c.metadata_json.get("root_section"),
                    section,
                    f"{section} chunk should mirror root_section in metadata",
                )
                self.assertNotIn("hierarchy", c.metadata_json)

    def test_sub_section_chunks_carry_hierarchy(self):
        """Sub-section chunks must carry a hierarchy.parent_section pointer."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "mixture",
            "names": [{"name": "Test Mixture", "displayName": True}],
            "mixture": {
                "components": [
                    {
                        "type": "MUST_BE_PRESENT",
                        "substance": {
                            "refPname": "trans-isomer",
                            "refuuid": "trans-uuid",
                        },
                    },
                ]
            },
            "references": [{"docType": "journal article", "id": "1"}],
            "version": "1.0",
            "status": "Active",
        }

        chunks = self._chunker.chunk(substance)
        # The former ``composition`` sub-section was merged into
        # ``mixture`` so components are emitted as ``mixture`` chunks
        # with ``chunk_type == "mixture_component"``.
        component_chunks = [
            c
            for c in chunks
            if c.section == "mixture"
            and c.metadata_json.get("chunk_type") == "mixture_component"
        ]
        self.assertGreater(
            len(component_chunks),
            0,
            "Expected at least one mixture component chunk to verify hierarchy.",
        )

        for c in component_chunks:
            self.assertEqual(
                c.root_section,
                "definitions",
                "mixture component chunks should map root_section column to definitions",
            )
            self.assertEqual(
                c.metadata_json.get("root_section"),
                "definitions",
                "mixture component chunks should mirror root_section in metadata",
            )
            hierarchy = c.metadata_json.get("hierarchy")
            self.assertIsInstance(hierarchy, dict)
            self.assertEqual(hierarchy.get("parent_section"), "definitions")
            self.assertEqual(hierarchy.get("level"), 1)

    def test_tags_section_is_top_level(self):
        """Tags is its own top-level root section (no longer under overview)."""
        substance = {
            "uuid": "12345678-1234-5678-1234-567812345678",
            "substanceClass": "concept",
            "names": [{"name": "Test Concept", "displayName": True}],
            "status": "approved",
            "tags": ["clinical", "inactive"],
        }

        chunks = self._chunker.chunk(substance)
        tag_chunks = [c for c in chunks if c.section == "tags"]
        # Per-item tag chunks — no batch summary.
        self.assertEqual(len(tag_chunks), 2)
        # Top-level section: root_section == section, no hierarchy key.
        for c in tag_chunks:
            self.assertEqual(c.root_section, "tags")
            self.assertEqual(c.metadata_json.get("root_section"), "tags")
            self.assertNotIn("hierarchy", c.metadata_json)


class TestSectionRename(unittest.TestCase):
    """Coverage for the section rename: structure -> definitions, codes split
    into identifiers + classifications (under codes root), tags promoted to
    top-level, and overview stripped of names/codes/structure/moieties."""

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    def test_root_section_helpers(self):
        # Per-class sub-sections map to definitions.
        for cls_name in (
            "chemical",
            "protein",
            "nucleicacid",
            "polymer",
            "structurallydiverse",
            "mixture",
            "specifiedsubstance",
        ):
            self.assertEqual(root_section_for(cls_name), "definitions")
        # codes root, with two sub-sections.
        self.assertEqual(root_section_for("identifiers"), "codes")
        self.assertEqual(root_section_for("classifications"), "codes")
        # tags is its own top-level root.
        self.assertEqual(root_section_for("tags"), "tags")
        # notes is its own top-level root (curator annotations
        # are surfaced independently of properties).
        self.assertEqual(root_section_for("notes"), "notes")
        # Sequence/composition are no longer in the section table: the
        # open-ended fallback in ``root_section_for`` returns the section
        # name itself, so they become their own top-level roots.
        self.assertEqual(root_section_for("sequence"), "sequence")
        self.assertEqual(root_section_for("composition"), "composition")
        # ``modifications`` still lives under definitions.
        self.assertEqual(root_section_for("modifications"), "definitions")
        # ``moieties`` is a registered sub-section of definitions so the
        # parent-child enricher can group moiety-only chunks separately
        # from the rest of the ``chemical`` payload if they're ever
        # emitted as their own section.
        self.assertEqual(root_section_for("moieties"), "definitions")
        # structure is no longer a recognised section name (fallback is
        # to use the section name itself — this is a deliberate choice
        # for forward-compat with legacy data).
        self.assertEqual(root_section_for("structure"), "structure")
        # Top-level membership.
        self.assertTrue(is_top_level_section("definitions"))
        self.assertTrue(is_top_level_section("tags"))
        self.assertTrue(is_top_level_section("notes"))
        self.assertFalse(is_top_level_section("identifiers"))
        self.assertFalse(is_top_level_section("classifications"))
        self.assertFalse(is_top_level_section("chemical"))

    def test_section_for_substance_class(self):
        self.assertEqual(section_for_substance_class("chemical"), "chemical")
        self.assertEqual(section_for_substance_class("Protein"), "protein")
        self.assertEqual(section_for_substance_class(""), "definitions")
        self.assertEqual(section_for_substance_class("  Mixture  "), "mixture")

    def test_sections_in_root_for_codes(self):
        members = sections_in_root("codes")
        self.assertIn("codes", members)
        self.assertIn("identifiers", members)
        self.assertIn("classifications", members)
        # No foreign members.
        self.assertNotIn("names", members)
        self.assertNotIn("definitions", members)

    def test_sections_in_root_for_definitions(self):
        members = sections_in_root("definitions")
        # The root itself.
        self.assertIn("definitions", members)
        # Every per-class sub-section.
        for sub in (
            "chemical",
            "protein",
            "nucleicacid",
            "polymer",
            "structurallydiverse",
            "mixture",
            "specifiedsubstance",
            "modifications",
            "moieties",
        ):
            self.assertIn(sub, members)
        # Sequence/composition are no longer in the table — they are
        # now their own root sections.
        self.assertNotIn("sequence", members)
        self.assertNotIn("composition", members)
        # No foreign members.
        self.assertNotIn("codes", members)
        self.assertNotIn("identifiers", members)

    def test_sections_in_root_for_properties(self):
        """``properties`` is its own root and no longer groups
        ``notes`` under it (notes was promoted to a top-level
        section so curator annotations are queryable
        independently of the substance's other properties)."""
        members = sections_in_root("properties")
        # The root itself.
        self.assertIn("properties", members)
        # ``notes`` was promoted to a top-level section, so it is
        # no longer a member of the properties root.
        self.assertNotIn("notes", members)
        # No foreign members.
        self.assertNotIn("definitions", members)
        self.assertNotIn("codes", members)

    def test_sections_in_root_for_notes(self):
        """``notes`` is its own top-level root section."""
        members = sections_in_root("notes")
        # The root itself.
        self.assertIn("notes", members)
        # No foreign members.
        self.assertNotIn("properties", members)
        self.assertNotIn("definitions", members)
        self.assertNotIn("codes", members)
        # ``is_top_level_section`` recognises notes as a top-level root.
        self.assertTrue(is_top_level_section("notes"))

    def test_chemical_uses_substance_class_as_section(self):
        substance = {
            "uuid": "11111111-1111-1111-1111-111111111111",
            "substanceClass": "chemical",
            "names": [{"name": "Aspirin", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CCO",
                "molecularFormula": "C2H6O",
            },
        }
        chunks = self._chunker.chunk(substance)
        chemical_chunks = [c for c in chunks if c.section == "chemical"]
        self.assertEqual(len(chemical_chunks), 1)
        self.assertEqual(chemical_chunks[0].metadata_json.get("root_section"), "definitions")
        # ``chemical`` is a sub-section of ``definitions``: hierarchy is set.
        hierarchy = chemical_chunks[0].metadata_json.get("hierarchy")
        self.assertIsInstance(hierarchy, dict)
        self.assertEqual(hierarchy.get("parent_section"), "definitions")
        self.assertEqual(hierarchy.get("level"), 1)

    def test_codes_emit_identifiers_not_codes(self):
        substance = {
            "uuid": "22222222-2222-2222-2222-222222222222",
            "substanceClass": "concept",
            "names": [{"name": "Test Codes", "displayName": True}],
            "status": "approved",
            "codes": [
                {"code": "ABC-001", "codeSystem": "INTERNAL", "type": "PRIMARY"},
                {"code": "ABC-002", "codeSystem": "INTERNAL", "type": "ALTERNATE"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        # No chunk should have section='codes' anymore — codes is purely a
        # root, and the per-row section is 'identifiers'.
        self.assertFalse(any(c.section == "codes" for c in chunks))
        ident_chunks = [c for c in chunks if c.section == "identifiers"]
        # Per-item identifier chunks — no batch summary.
        self.assertEqual(len(ident_chunks), 2)
        for c in ident_chunks:
            self.assertEqual(c.metadata_json.get("chunk_type"), "identifier")
            self.assertEqual(c.metadata_json.get("root_section"), "codes")
            hierarchy = c.metadata_json.get("hierarchy")
            self.assertIsInstance(hierarchy, dict)
            self.assertEqual(hierarchy.get("parent_section"), "codes")
            self.assertEqual(hierarchy.get("level"), 1)

    def test_classifications_live_under_codes_root(self):
        """Codes with ``_isClassification=True`` live under the codes root
        in a ``classifications`` sub-section."""
        substance = {
            "uuid": "33333333-3333-3333-3333-333333333333",
            "substanceClass": "concept",
            "names": [{"name": "Test Classifications", "displayName": True}],
            "status": "approved",
            "codes": [
                {"code": "HUMAN_DRUG", "_isClassification": True},
                {"code": "PRESCRIPTION", "_isClassification": True},
            ],
        }
        chunks = self._chunker.chunk(substance)
        class_chunks = [c for c in chunks if c.section == "classifications"]
        # Per-item classification chunks — no batch summary.
        self.assertEqual(len(class_chunks), 2)
        for c in class_chunks:
            self.assertEqual(c.metadata_json.get("chunk_type"), "classification")
            self.assertEqual(c.metadata_json.get("root_section"), "codes")
            # Sub-section → has hierarchy.
            hierarchy = c.metadata_json.get("hierarchy")
            self.assertIsInstance(hierarchy, dict)
            self.assertEqual(hierarchy.get("parent_section"), "codes")
            self.assertEqual(hierarchy.get("level"), 1)

    def test_classifications_detected_via_pipe_in_comments(self):
        """Codes whose ``comments`` contain a ``|`` are classifications too."""
        substance = {
            "uuid": "33333333-3333-3333-3333-333333333334",
            "substanceClass": "concept",
            "names": [{"name": "Pipe Classifications", "displayName": True}],
            "status": "approved",
            "codes": [
                {"code": "FLAVOR", "comments": "code1|code2|code3"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        class_chunks = [c for c in chunks if c.section == "classifications"]
        # Per-item classification chunks were dropped — only the batch
        # chunk is emitted per substance.
        self.assertEqual(len(class_chunks), 1)
        ident_chunks = [c for c in chunks if c.section == "identifiers"]
        self.assertEqual(len(ident_chunks), 0)

    def test_classifications_and_identifiers_share_codes_root(self):
        """A single parent query on 'codes' must return both kinds."""
        substance = {
            "uuid": "44444444-4444-4444-4444-444444444444",
            "substanceClass": "concept",
            "names": [{"name": "Mixed Codes", "displayName": True}],
            "status": "approved",
            "codes": [
                {"code": "X-1", "codeSystem": "EXT"},
                {"code": "OTC", "_isClassification": True},
            ],
        }
        chunks = self._chunker.chunk(substance)
        members = sections_in_root("codes")
        all_in_root = [
            c for c in chunks
            if c.section in members and c.metadata_json.get("root_section") == "codes"
        ]
        # Both identifiers and classifications chunks should be there.
        self.assertTrue(any(c.section == "identifiers" for c in all_in_root))
        self.assertTrue(any(c.section == "classifications" for c in all_in_root))

    def test_overview_excludes_names_codes_structure_moieties(self):
        substance = {
            "uuid": "55555555-5555-5555-5555-555555555555",
            "substanceClass": "chemical",
            "names": [
                {"name": "Aspirin", "displayName": True},
                {"name": "Acetylsalicylic acid"},
            ],
            "codes": [{"code": "ASA", "codeSystem": "INTERNAL"}],
            "structure": {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            "moieties": [{"smiles": "CC(=O)O"}],
            "status": "approved",
            "version": "1.0",
        }
        chunks = self._chunker.chunk(substance)
        overview_chunks = [c for c in chunks if c.section == "overview"]
        self.assertEqual(len(overview_chunks), 1)
        text = overview_chunks[0].text
        # Identity fields are present.
        self.assertIn("Aspirin", text)
        self.assertIn("Class: chemical", text)
        self.assertIn("Version: 1.0", text)
        # Stripped fields are NOT in the overview.
        self.assertNotIn("Names:", text)
        self.assertNotIn("Codes:", text)
        self.assertNotIn("SMILES:", text)
        self.assertNotIn("Moieties:", text)
        self.assertNotIn("Acetylsalicylic acid", text)

    def test_per_class_section_uses_substance_class(self):
        """Each class-specific chunk uses section == substanceClass."""
        cases = (
            ("chemical", "smiles"),
            ("protein", None),
            ("nucleicacid", None),
            ("polymer", None),
            ("structurallydiverse", None),
            ("mixture", None),
        )
        base = {
            "uuid": "66666666-6666-6666-6666-666666666666",
            "names": [{"name": "Poly Subst", "displayName": True}],
            "status": "approved",
        }
        for cls_name, _ in cases:
            sub = dict(base)
            sub["substanceClass"] = cls_name
            if cls_name == "chemical":
                sub["structure"] = {"smiles": "CCO"}
            elif cls_name == "protein":
                sub["protein"] = {"subunits": [{"subunitIndex": "1", "sequence": "MKT"}]}
            elif cls_name == "nucleicacid":
                sub["nucleicAcid"] = {"subunits": [{"subunitIndex": "1", "sequence": "ATCG"}]}
            elif cls_name == "polymer":
                sub["polymer"] = {
                    "classification": {"polymerClass": "HOMOPOLYMER"},
                    "monomers": [],
                }
            elif cls_name == "structurallydiverse":
                sub["structurallyDiverse"] = {
                    "sourceMaterialClass": "ORGANISM",
                }
            elif cls_name == "mixture":
                sub["mixture"] = {
                    "components": [
                        {
                            "type": "MUST_BE_PRESENT",
                            "substance": {
                                "refPname": "Component A",
                                "refuuid": "comp-a-uuid",
                            },
                        }
                    ]
                }

            chunks = self._chunker.chunk(sub)
            section_chunks = [c for c in chunks if c.section == cls_name]
            self.assertGreater(
                len(section_chunks), 0,
                f"Expected at least one chunk with section='{cls_name}'",
            )
            for c in section_chunks:
                self.assertEqual(
                    c.metadata_json.get("root_section"), "definitions",
                    f"{cls_name} chunks should map root_section=definitions",
                )


class TestPropertyAndModificationSplit(unittest.TestCase):
    """Properties and modifications each have their own builder.

    The previous monolithic ``_build_classification_chunks`` produced
    three different chunk types (properties, modifications, and
    classifications) in one method. This class exercises the split
    where:

    * properties live in ``section='properties'`` and
    * modifications live in ``section='modifications'``.
    """

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    def test_property_chunks_are_separate_section(self):
        substance = {
            "uuid": "77777777-7777-7777-7777-777777777777",
            "substanceClass": "chemical",
            "names": [{"name": "Has Properties", "displayName": True}],
            "status": "approved",
            "properties": [
                {"name": "MELTING_POINT", "value": "180 C"},
                {"name": "SOLUBILITY", "value": "10 mg/mL"},
            ],
            "structure": {"smiles": "CCO"},
        }
        chunks = self._chunker.chunk(substance)
        prop_chunks = [c for c in chunks if c.section == "properties"]
        # 2 per-item property chunks — no batch summary.
        self.assertEqual(len(prop_chunks), 2)
        all_text = " ".join(c.text for c in prop_chunks)
        self.assertIn("MELTING_POINT", all_text)
        self.assertIn("180 C", all_text)
        self.assertIn("SOLUBILITY", all_text)
        self.assertIn("10 mg/mL", all_text)
        # Top-level section under root 'properties' (its own root).
        self.assertEqual(prop_chunks[0].metadata_json.get("root_section"), "properties")
        # No modifications emitted.
        self.assertFalse(any(c.section == "modifications" for c in chunks))

    def test_modification_chunks_are_separate_section(self):
        substance = {
            "uuid": "88888888-8888-8888-8888-888888888888",
            "substanceClass": "protein",
            "names": [{"name": "Has Mods", "displayName": True}],
            "status": "approved",
            "modifications": [
                {"modificationType": "PHOSPHORYLATION"},
                {"modificationType": "ACETYLATION"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        mod_chunks = [c for c in chunks if c.section == "modifications"]
        # 2 per-item modification chunks — no batch summary.
        self.assertEqual(len(mod_chunks), 2)
        all_text = " ".join(c.text for c in mod_chunks)
        self.assertIn("PHOSPHORYLATION", all_text)
        self.assertIn("ACETYLATION", all_text)
        # ``modifications`` is a sub-section of root ``definitions``,
        # following the existing section taxonomy.
        self.assertEqual(mod_chunks[0].metadata_json.get("root_section"), "definitions")
        hierarchy = mod_chunks[0].metadata_json.get("hierarchy")
        self.assertIsInstance(hierarchy, dict)
        self.assertEqual(hierarchy.get("parent_section"), "definitions")
        self.assertEqual(hierarchy.get("level"), 1)
        # No properties emitted.
        self.assertFalse(any(c.section == "properties" for c in chunks))

    def test_property_and_modification_sections_are_independent(self):
        """When both are present, each section is emitted separately."""
        substance = {
            "uuid": "99999999-9999-9999-9999-999999999999",
            "substanceClass": "chemical",
            "names": [{"name": "Both", "displayName": True}],
            "status": "approved",
            "properties": [{"name": "DENSITY", "value": "1.2 g/mL"}],
            "modifications": [{"modificationType": "OXIDATION"}],
            "structure": {"smiles": "CCO"},
        }
        chunks = self._chunker.chunk(substance)
        sections = {c.section for c in chunks}
        self.assertIn("properties", sections)
        self.assertIn("modifications", sections)
        # Properties: 1 per-item chunk. Modifications: 1 per-item chunk.
        self.assertEqual(
            sum(1 for c in chunks if c.section == "properties"), 1
        )
        self.assertEqual(
            sum(1 for c in chunks if c.section == "modifications"), 1
        )

    def test_no_classification_chunk_builder_remains(self):
        """The legacy top-level ``classifications`` field should not produce
        a classifications chunk any more — classifications come from the
        ``codes`` array."""
        substance = {
            "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "substanceClass": "concept",
            "names": [{"name": "Legacy", "displayName": True}],
            "status": "approved",
            # Legacy top-level field — should be ignored.
            "classifications": [
                {"classification": "HUMAN_DRUG"},
                {"classification": "PRESCRIPTION"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        # No classifications chunk should be produced at all.
        self.assertFalse(
            any(c.section == "classifications" for c in chunks),
            "Top-level classifications array is no longer a source; "
            "classifications are sourced from codes entries.",
        )


class TestChemicalStructureAndMoieties(unittest.TestCase):
    """``_build_chemical_structure_chunks`` emits the molecule's structure
    and one chunk per moiety (with an optional batch summary). Sequence
    data is no longer processed here — protein/nucleic-acid sequences
    are handled in their dedicated class-specific builders."""

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    def test_structure_only_emits_one_chemical_chunk(self):
        substance = {
            "uuid": "11000000-0000-0000-0000-000000000001",
            "substanceClass": "chemical",
            "names": [{"name": "Ethanol", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CCO",
                "formula": "C2H6O",
                "mwt": 46.07,
                "stereochemistry": "ACHIRAL",
            },
        }
        chunks = self._chunker.chunk(substance)
        chem = [c for c in chunks if c.section == "chemical"]
        # 1 structure summary; no moieties, no batch.
        self.assertEqual(len(chem), 1)
        self.assertEqual(chem[0].metadata_json.get("chunk_type"), "structure")
        self.assertIn("CCO", chem[0].text)

    def test_single_moiety_emits_per_item_and_batch(self):
        substance = {
            "uuid": "11000000-0000-0000-0000-000000000002",
            "substanceClass": "chemical",
            "names": [{"name": "Ibuprofen", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                "formula": "C13H18O2",
                "mwt": 206.28,
            },
            "moieties": [
                {
                    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                    "formula": "C13H18O2",
                    "mwt": 206.28,
                    "stereochemistry": "RACEMIC",
                    "opticalActivity": "( + / - )",
                    "atropisomerism": "No",
                    "count": 1,
                }
            ],
        }
        chunks = self._chunker.chunk(substance)
        chem = [c for c in chunks if c.section == "chemical"]
        # 1 structure + 1 moiety — no batch summary.
        self.assertEqual(len(chem), 2)
        types = {c.metadata_json.get("chunk_type") for c in chem}
        self.assertEqual(types, {"structure", "moiety"})

        moiety = next(c for c in chem if c.metadata_json.get("chunk_type") == "moiety")
        self.assertEqual(moiety.metadata_json.get("moiety_index"), 0)
        self.assertEqual(moiety.metadata_json.get("count"), 1)
        self.assertIn("Moiety 1", moiety.text)
        self.assertIn("RACEMIC", moiety.text)
        self.assertIn("Count: 1", moiety.text)
        # No batch summary chunk.
        self.assertFalse(
            any(c.metadata_json.get("chunk_type") == "moiety_batch" for c in chem)
        )

    def test_multi_moiety_salt(self):
        """A salt with two moieties should produce 1 structure + 2 per-item
        moieties — all under the chemical section."""
        substance = {
            "uuid": "11000000-0000-0000-0000-000000000003",
            "substanceClass": "chemical",
            "names": [{"name": "Acetate Salt", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CC(=O)O.CC(N)CCO",
                "formula": "C5H13NO3",
            },
            "moieties": [
                {
                    "smiles": "CC(=O)O",
                    "formula": "C2H4O2",
                    "count": 1,
                },
                {
                    "smiles": "CC(N)CCO",
                    "formula": "C3H9NO",
                    "stereochemistry": "RACEMIC",
                    "count": 2,
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        chem = [c for c in chunks if c.section == "chemical"]
        # 1 structure + 2 moieties — no batch summary.
        self.assertEqual(len(chem), 3)
        per_items = [
            c for c in chem
            if c.metadata_json.get("chunk_type") == "moiety"
        ]
        self.assertEqual(len(per_items), 2)
        self.assertEqual(
            sorted(c.metadata_json.get("moiety_index") for c in per_items),
            [0, 1],
        )
        # Per-item moieties are individually searchable.
        all_text = " ".join(c.text for c in per_items)
        self.assertIn("CC(=O)O", all_text)
        self.assertIn("CC(N)CCO", all_text)
        # No batch summary chunk.
        self.assertFalse(
            any(c.metadata_json.get("chunk_type") == "moiety_batch" for c in chem)
        )

    def test_no_sequences_processing_in_chemical(self):
        """The chemical builder no longer emits sequence chunks.

        Sequences are exclusively handled by the protein and nucleic-acid
        class-specific builders via ``subunits[*].sequence``.
        """
        substance = {
            "uuid": "11000000-0000-0000-0000-000000000004",
            "substanceClass": "chemical",
            "names": [{"name": "Has Sequences", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CCO",
                # Legacy fields that the chemical builder should now ignore.
                "sequence": "MKTLLLTLVVVTIVCLDLGYTFQPQNGQFICTTAG",
                "sequences": ["MKTLLLTLVVVTIVCLDLGYTFQPQNGQFICTTAG"],
            },
        }
        chunks = self._chunker.chunk(substance)
        # No sequence-segment / sequence-full chunks from the chemical builder.
        self.assertFalse(
            any(c.metadata_json.get("chunk_type", "").startswith("sequence_") for c in chunks),
            "Chemical builder must not emit sequence_* chunks",
        )
        # And there are no 'sequence' section chunks at all for a pure chemical.
        self.assertFalse(
            any(c.section == "sequence" for c in chunks),
            "Chemical substances should not produce sequence-section chunks",
        )

    def test_moiety_chunk_root_and_section(self):
        """Moiety chunks live under the chemical section, root definitions."""
        substance = {
            "uuid": "11000000-0000-0000-0000-000000000005",
            "substanceClass": "chemical",
            "names": [{"name": "Has Moiety", "displayName": True}],
            "status": "approved",
            "structure": {"smiles": "CCO"},
            "moieties": [{"smiles": "C", "count": 1}],
        }
        chunks = self._chunker.chunk(substance)
        for c in chunks:
            if c.section != "chemical":
                continue
            self.assertEqual(c.metadata_json.get("root_section"), "definitions")
            if c.metadata_json.get("chunk_type") == "moiety":
                # Sub-section carries hierarchy pointing to definitions.
                h = c.metadata_json.get("hierarchy")
                self.assertIsInstance(h, dict)
                self.assertEqual(h.get("parent_section"), "definitions")

    def test_empty_moieties_emits_no_moiety_chunks(self):
        """An empty ``moieties`` list should not produce any moiety chunks."""
        substance = {
            "uuid": "11000000-0000-0000-0000-000000000006",
            "substanceClass": "chemical",
            "names": [{"name": "No Moieties", "displayName": True}],
            "status": "approved",
            "structure": {"smiles": "CCO"},
            "moieties": [],
        }
        chunks = self._chunker.chunk(substance)
        chem = [c for c in chunks if c.section == "chemical"]
        self.assertEqual(len(chem), 1)
        self.assertEqual(chem[0].metadata_json.get("chunk_type"), "structure")

    def test_real_ibuprofen_payload(self):
        """End-to-end check against the real ibuprofen GSRS payload
        (uuid 0103a288-6eb6-4ced-b13a-849cd7edf028) shipped in the repo."""
        import os

        payload_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "ibuprofen.json",
        )
        if not os.path.exists(payload_path):
            self.skipTest("ibuprofen.json fixture not present")
        import json as _json

        with open(payload_path) as f:
            sub = _json.load(f)

        chunks = self._chunker.chunk(sub)
        chem = [c for c in chunks if c.section == "chemical"]
        # 1 structure + 1 moiety — no batch summary.
        self.assertEqual(len(chem), 2)
        types = {c.metadata_json.get("chunk_type") for c in chem}
        self.assertEqual(types, {"structure", "moiety"})

        # The moiety item carries the count from the real payload.
        moiety = next(
            c for c in chem
            if c.metadata_json.get("chunk_type") == "moiety"
        )
        self.assertEqual(moiety.metadata_json.get("count"), 1)
        self.assertIn("Count: 1", moiety.text)

    def test_specified_substance_g1_constituents_chunk(self):
        """G1 substances emit a constituents summary under the
        ``specifiedsubstance`` sub-section, with role + uuid per
        constituent and the constituent count in metadata."""
        substance = {
            "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "substanceClass": "specifiedSubstanceG1",
            "names": [{"name": "Test G1", "displayName": True}],
            "specifiedSubstance": {
                "constituents": [
                    {
                        "role": "ACTIVE COMPONENT",
                        "substance": {
                            "refPname": "Starmarella bombicola",
                            "refuuid": "9f7ec4af-b71f-4640-863b-6576d4b764f1",
                        },
                    },
                    {
                        "role": "COMPONENT",
                        "substance": {
                            "refPname": "Anhydrous dextrose",
                            "refuuid": "62b08ba9-5032-4949-9187-e36925b5e940",
                        },
                    },
                ]
            },
        }
        chunks = self._chunker.chunk(substance)
        ss_chunks = [c for c in chunks if c.section == "specifiedsubstance"]
        self.assertEqual(len(ss_chunks), 1)
        c = ss_chunks[0]
        self.assertEqual(
            c.metadata_json.get("chunk_type"),
            "specified_substance_constituents",
        )
        self.assertEqual(c.metadata_json.get("constituent_count"), 2)
        self.assertEqual(
            c.metadata_json.get("substance_class"),
            "specifiedSubstanceG1",
        )
        # All three names and both uuids should appear in the text.
        self.assertIn("Starmarella bombicola", c.text)
        self.assertIn("Anhydrous dextrose", c.text)
        self.assertIn("ACTIVE COMPONENT", c.text)
        self.assertIn("COMPONENT", c.text)
        self.assertIn("9f7ec4af-b71f-4640-863b-6576d4b764f1", c.text)
        self.assertIn("62b08ba9-5032-4949-9187-e36925b5e940", c.text)
        # Root section is "definitions" via _SECTION_TO_ROOT.
        self.assertEqual(c.metadata_json.get("root_section"), "definitions")

    def test_specified_substance_g1_collapses_to_registered_section(self):
        """All G-variants collapse to ``specifiedsubstance``; the raw
        substanceClass never leaks as a section name (regression test
        for the turn-2 mapping cleanup)."""
        for cls in (
            "specifiedSubstanceG1",
            "specifiedSubstanceGroup1",
        ):
            substance = {
                "uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "substanceClass": cls,
                "names": [{"name": f"Test {cls}", "displayName": True}],
                # No ``grade`` — that's not part of the GSRS payload.
                # The chunker must not emit a grade chunk.
                "specifiedSubstance": {},
            }
            chunks = self._chunker.chunk(substance)
            sections = {c.section for c in chunks}
            # The raw substance class must never appear as a section.
            self.assertNotIn(cls.lower(), sections)
            # No ``specifiedsubstance`` chunk is emitted when there's
            # nothing to chunk (no constituents, no grade).
            self.assertNotIn("specifiedsubstance", sections)
            # No chunk must ever carry the legacy ``specified_substance``
            # chunk_type (the old "grade"-only chunk shape).
            for c in chunks:
                self.assertNotEqual(
                    c.metadata_json.get("chunk_type"),
                    "specified_substance",
                )


class TestNotesAndStructuredPropertyValues(unittest.TestCase):
    """Notes chunking + structured (dict-shaped) property values.

    Real GSRS payloads store property values as nested objects
    (``{average, low, high, units}`` or ``{nonNumericValue}``); the
    chunker must flatten these into a queryable string, not stringify
    the value as a dict-repr. The previous implementation
    silently dropped every property whose value wasn't a string.
    """

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    # ---- notes ----------------------------------------------------------

    def test_notes_chunk_emitted(self):
        """Curator annotations are surfaced as a single notes chunk."""
        # Opt in to admin validation notes so both [Validation] rows
        # are emitted. The default is to filter them out (see
        # ``ChunkerConfig.include_admin_validation_notes``).
        chunker = SubstanceChunker(
            class_=VectorDocument,
            config=ChunkerConfig(include_admin_validation_notes=True),
        )
        substance = {
            "uuid": "cccccccc-1111-2222-3333-444444444444",
            "substanceClass": "chemical",
            "names": [{"name": "Annotated", "displayName": True}],
            "status": "approved",
            "notes": [
                {
                    "note": "[Validation]WARNING:Record Foo is a potential duplicate",
                },
                {
                    "note": "[Validation]WARNING:Record Bar needs curator review",
                },
            ],
        }
        chunks = chunker.chunk(substance)
        note_chunks = [c for c in chunks if c.section == "notes"]
        # One chunk per note — no batch summary.
        self.assertEqual(len(note_chunks), 2)
        for c in note_chunks:
            self.assertEqual(c.metadata_json.get("chunk_type"), "note")
            # ``notes`` is its own top-level root (not a sub-section of
            # properties anymore).
            self.assertEqual(c.metadata_json.get("root_section"), "notes")
            # Top-level: no hierarchy key recorded.
            self.assertNotIn("hierarchy", c.metadata_json)
        all_text = " ".join(c.text for c in note_chunks)
        # Both notes are in the chunk text.
        self.assertIn("potential duplicate", all_text)
        self.assertIn("curator review", all_text)

    def test_notes_is_a_top_level_section(self):
        """``notes`` is a top-level section, not a sub-section.

        Curator annotations are surfaced as their own root so they
        can be queried independently of the substance's other
        properties.
        """
        substance = {
            "uuid": "cccccccc-1111-2222-3333-555555555555",
            "substanceClass": "chemical",
            "names": [{"name": "Annotated 2", "displayName": True}],
            "status": "approved",
            "notes": [{"note": "test"}],
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "notes")
        # Top-level section: root_section == section, no hierarchy.
        self.assertEqual(c.metadata_json.get("root_section"), "notes")
        self.assertNotIn("hierarchy", c.metadata_json)
        # ``is_top_level_section`` recognises it.
        self.assertTrue(is_top_level_section("notes"))

    def test_no_notes_emits_no_notes_chunk(self):
        """A substance without notes must not produce a notes chunk."""
        substance = {
            "uuid": "cccccccc-1111-2222-3333-666666666666",
            "substanceClass": "chemical",
            "names": [{"name": "No Notes", "displayName": True}],
            "status": "approved",
            "structure": {"smiles": "CCO"},
        }
        chunks = self._chunker.chunk(substance)
        self.assertFalse(any(c.section == "notes" for c in chunks))

    def test_empty_notes_emits_no_notes_chunk(self):
        """An empty notes list must not produce a notes chunk."""
        substance = {
            "uuid": "cccccccc-1111-2222-3333-777777777777",
            "substanceClass": "chemical",
            "names": [{"name": "Empty Notes", "displayName": True}],
            "status": "approved",
            "notes": [],
        }
        chunks = self._chunker.chunk(substance)
        self.assertFalse(any(c.section == "notes" for c in chunks))

    def test_admin_validation_notes_excluded_by_default(self):
        """By default, notes prefixed with ``[Validation]`` are filtered
        out (they are produced in bulk by the GSRS admin validator and
        tend to dominate retrieval). Callers that want them must opt
        in via ``ChunkerConfig(include_admin_validation_notes=True)``.
        """
        substance = {
            "uuid": "cccccccc-1111-2222-3333-aaaaaaaaaaaa",
            "substanceClass": "chemical",
            "names": [{"name": "Validated", "displayName": True}],
            "status": "approved",
            "notes": [
                {"note": "[Validation]WARNING:potential duplicate"},
                {"note": "Curator: needs review"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        note_chunks = [c for c in chunks if c.section == "notes"]
        # Only the human-curated note remains.
        self.assertEqual(len(note_chunks), 1)
        all_text = " ".join(c.text for c in note_chunks)
        self.assertNotIn("[Validation]WARNING:potential duplicate", all_text)
        self.assertIn("Curator: needs review", all_text)

    def test_admin_validation_notes_included_when_enabled(self):
        """With ``include_admin_validation_notes=True``, ``[Validation]``
        notes are emitted alongside the human-curated ones."""
        chunker = SubstanceChunker(
            class_=VectorDocument,
            config=ChunkerConfig(include_admin_validation_notes=True),
        )
        substance = {
            "uuid": "cccccccc-1111-2222-3333-aaaaaaaaaaaa",
            "substanceClass": "chemical",
            "names": [{"name": "Validated", "displayName": True}],
            "status": "approved",
            "notes": [
                {"note": "[Validation]WARNING:potential duplicate"},
                {"note": "Curator: needs review"},
            ],
        }
        chunks = chunker.chunk(substance)
        note_chunks = [c for c in chunks if c.section == "notes"]
        # One chunk per note — no batch summary.
        self.assertEqual(len(note_chunks), 2)
        all_text = " ".join(c.text for c in note_chunks)
        self.assertIn("[Validation]WARNING:potential duplicate", all_text)
        self.assertIn("Curator: needs review", all_text)

    def test_admin_validation_notes_filtered_when_disabled(self):
        """When ``include_admin_validation_notes=False``, ``[Validation]``
        notes are dropped, leaving only human-curated annotations."""
        chunker = SubstanceChunker(
            class_=VectorDocument,
            config=ChunkerConfig(include_admin_validation_notes=False),
        )
        substance = {
            "uuid": "cccccccc-1111-2222-3333-bbbbbbbbbbbb",
            "substanceClass": "chemical",
            "names": [{"name": "Mixed Notes", "displayName": True}],
            "status": "approved",
            "notes": [
                {"note": "[Validation]WARNING:potential duplicate"},
                {"note": "[Validation]WARNING:vocab mismatch"},
                {"note": "Curator: needs review"},
            ],
        }
        chunks = chunker.chunk(substance)
        note_chunks = [c for c in chunks if c.section == "notes"]
        self.assertEqual(len(note_chunks), 1)
        c = note_chunks[0]
        # Admin validation notes are filtered out.
        self.assertNotIn("potential duplicate", c.text)
        self.assertNotIn("vocab mismatch", c.text)
        # The human-curated note remains.
        self.assertIn("Curator: needs review", c.text)

    def test_admin_validation_notes_only_no_chunk_when_disabled(self):
        """If filtering leaves zero notes, no note chunk is emitted."""
        chunker = SubstanceChunker(
            class_=VectorDocument,
            config=ChunkerConfig(include_admin_validation_notes=False),
        )
        substance = {
            "uuid": "cccccccc-1111-2222-3333-cccccccccccc",
            "substanceClass": "chemical",
            "names": [{"name": "Only Admin", "displayName": True}],
            "status": "approved",
            "notes": [
                {"note": "[Validation]WARNING:foo"},
                {"note": "[Validation]WARNING:bar"},
            ],
        }
        chunks = chunker.chunk(substance)
        self.assertFalse(any(c.section == "notes" for c in chunks))

    def test_is_admin_validation_note_helper(self):
        """The helper recognises the ``[Validation]`` prefix only."""
        self.assertTrue(
            SubstanceChunker._is_admin_validation_note("[Validation]WARNING:x")
        )
        self.assertTrue(
            SubstanceChunker._is_admin_validation_note("[Validation]")
        )
        self.assertFalse(
            SubstanceChunker._is_admin_validation_note("Curator note")
        )
        self.assertFalse(
            SubstanceChunker._is_admin_validation_note(
                "prefix [Validation] suffix"
            )
        )
        self.assertFalse(SubstanceChunker._is_admin_validation_note(""))

    # ---- property value formatting --------------------------------------

    def test_property_numeric_average(self):
        """``{average, units}`` flattens to ``1600.0 mg/dose``."""
        substance = {
            "uuid": "dddddddd-1111-2222-3333-444444444444",
            "substanceClass": "chemical",
            "names": [{"name": "Has Average", "displayName": True}],
            "status": "approved",
            "properties": [
                {
                    "name": "MAXIMUM TOLERATED DOSE",
                    "propertyType": "PHARMACOKINETIC",
                    "value": {"average": 1600.0, "units": "mg/dose"},
                }
            ],
        }
        chunks = self._chunker.chunk(substance)
        prop = next(c for c in chunks if c.section == "properties" and c.metadata_json.get("chunk_type") == "property")
        self.assertIn("MAXIMUM TOLERATED DOSE", prop.text)
        self.assertIn("1600.0 mg/dose", prop.text)

    def test_property_numeric_range(self):
        """``{low, high, units}`` flattens to ``2.22-2.44 hours``."""
        substance = {
            "uuid": "dddddddd-1111-2222-3333-555555555555",
            "substanceClass": "chemical",
            "names": [{"name": "Has Range", "displayName": True}],
            "status": "approved",
            "properties": [
                {
                    "name": "Biological Half-life",
                    "value": {"low": 2.22, "high": 2.44, "units": "hours"},
                }
            ],
        }
        chunks = self._chunker.chunk(substance)
        prop = next(c for c in chunks if c.section == "properties" and c.metadata_json.get("chunk_type") == "property")
        self.assertIn("2.22-2.44 hours", prop.text)

    def test_property_non_numeric_value(self):
        """``{nonNumericValue}`` flattens to its string form."""
        substance = {
            "uuid": "dddddddd-1111-2222-3333-666666666666",
            "substanceClass": "nucleicAcid",
            "names": [{"name": "Has NonNumeric", "displayName": True}],
            "status": "approved",
            "properties": [
                {
                    "name": "CMV enhancer",
                    "value": {"nonNumericValue": "1_236-1_615"},
                }
            ],
        }
        chunks = self._chunker.chunk(substance)
        prop = next(c for c in chunks if c.section == "properties" and c.metadata_json.get("chunk_type") == "property")
        self.assertIn("1_236-1_615", prop.text)

    def test_property_string_value_still_works(self):
        """Plain-string property values continue to work (synthetic fixture)."""
        substance = {
            "uuid": "dddddddd-1111-2222-3333-777777777777",
            "substanceClass": "chemical",
            "names": [{"name": "String Val", "displayName": True}],
            "status": "approved",
            "properties": [
                {"name": "melting_point", "value": "135 °C"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        prop = next(c for c in chunks if c.section == "properties" and c.metadata_json.get("chunk_type") == "property")
        self.assertIn("135 °C", prop.text)

    def test_property_value_not_stringified_as_dict(self):
        """Property text must not be the Python repr of the value dict.

        The previous implementation built ``f"{prop_name}={prop_value}"``
        where ``prop_value`` was a dict, producing text like
        ``MAXIMUM TOLERATED DOSE={'created': 1772..., 'average': 1600.0}``.
        The current implementation extracts the meaningful value
        (``average + units``) instead.
        """
        substance = {
            "uuid": "dddddddd-1111-2222-3333-888888888888",
            "substanceClass": "chemical",
            "names": [{"name": "Dict Val", "displayName": True}],
            "status": "approved",
            "properties": [
                {
                    "name": "MAXIMUM TOLERATED DOSE",
                    "value": {
                        "created": 1234,
                        "createdBy": "ADMIN",
                        "average": 1600.0,
                        "units": "mg/dose",
                    },
                }
            ],
        }
        chunks = self._chunker.chunk(substance)
        all_text = " ".join(c.text for c in chunks)
        # The Python-repr form must not appear.
        self.assertNotIn("'created':", all_text)
        self.assertNotIn("'createdBy':", all_text)
        # The flattened form should appear.
        self.assertIn("1600.0 mg/dose", all_text)

    def test_real_ibuprofen_properties_are_emitted(self):
        """The 5 structured-property entries in the real ibuprofen
        payload must all be surfaced as chunks. Previously they were
        silently dropped because their values were dicts.
        """
        import json as _json
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "ibuprofen.json")
        if not os.path.exists(path):
            self.skipTest("ibuprofen.json fixture not present")
        with open(path) as f:
            sub = _json.load(f)

        chunks = self._chunker.chunk(sub)
        property_chunks = [
            c for c in chunks
            if c.section == "properties"
            and c.metadata_json.get("chunk_type") == "property"
        ]
        # 1 batch + 5 per-item property chunks.
        self.assertEqual(len(property_chunks), 5)
        names = {
            c.metadata_json.get("property_name")
            for c in property_chunks
        }
        self.assertEqual(
            names,
            {
                "MAXIMUM TOLERATED DOSE",
                "Biological Half-life",
                "Cmax",
                "Tmax",
                "Volume of Distribution",
            },
        )
        # Each chunk is independently queryable.
        all_text = " ".join(c.text for c in property_chunks)
        self.assertIn("1600.0 mg/dose", all_text)
        self.assertIn("2.22-2.44 hours", all_text)
        self.assertIn("50.0-100.0 microgram/mL", all_text)

    def test_real_ibuprofen_notes_are_emitted(self):
        """The real ibuprofen payload has 25 curator notes; the
        chunker must surface them as a single notes chunk.
        """
        # Ibuprofen's notes are mostly admin validation notes;
        # opt in to surface them. The default is to filter them
        # out so the curator's other annotations dominate retrieval.
        chunker = SubstanceChunker(
            class_=VectorDocument,
            config=ChunkerConfig(include_admin_validation_notes=True),
        )
        import json as _json
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "ibuprofen.json")
        if not os.path.exists(path):
            self.skipTest("ibuprofen.json fixture not present")
        with open(path) as f:
            sub = _json.load(f)

        chunks = chunker.chunk(sub)
        note_chunks = [c for c in chunks if c.section == "notes"]
        # One chunk per note — no batch summary.
        self.assertGreaterEqual(len(note_chunks), 3)
        all_text = " ".join(c.text for c in note_chunks)
        self.assertIn("Dexibuprofen", all_text)
        self.assertIn("potential duplicate", all_text)


class TestPerItemChunking(unittest.TestCase):
    """The chunker emits one chunk per payload item, with no batch
    summary chunks and no string/list shrinking.

    Earlier versions of the chunker collapsed references / codes /
    classifications / relationships into a small number of batch
    chunks to keep the total chunk count low. That optimization
    has been reverted: every payload field is now independently
    queryable, so a list of 25 references produces 25 reference
    chunks, not one ``reference_index`` chunk.
    """

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    def test_per_item_reference_chunks(self):
        substance = {
            "uuid": "eeeeeeee-1111-2222-3333-444444444444",
            "substanceClass": "chemical",
            "names": [{"name": "Has Refs", "displayName": True}],
            "status": "approved",
            "references": [
                {"docType": "journal article", "id": "r1"},
                {"docType": "book chapter", "id": "r2"},
                {"docType": "patent", "id": "r3"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        ref_chunks = [c for c in chunks if c.section == "references"]
        # 3 per-item reference chunks — no index / batch summary.
        self.assertEqual(len(ref_chunks), 3)
        self.assertTrue(
            all(
                c.metadata_json.get("chunk_type") == "reference"
                for c in ref_chunks
            )
        )
        self.assertNotIn(
            "reference_index", {c.metadata_json.get("chunk_type") for c in ref_chunks}
        )

    def test_per_item_code_chunks(self):
        substance = {
            "uuid": "eeeeeeee-1111-2222-3333-555555555555",
            "substanceClass": "chemical",
            "names": [{"name": "Has Codes", "displayName": True}],
            "status": "approved",
            "codes": [
                {"code": "A", "codeSystem": "EXT"},
                {"code": "B", "codeSystem": "EXT"},
                {"code": "C", "codeSystem": "EXT"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        ident_chunks = [c for c in chunks if c.section == "identifiers"]
        # 3 per-item identifier chunks — no batch summary.
        self.assertEqual(len(ident_chunks), 3)
        self.assertTrue(
            all(
                c.metadata_json.get("chunk_type") == "identifier"
                for c in ident_chunks
            )
        )
        self.assertNotIn(
            "code_batch", {c.metadata_json.get("chunk_type") for c in ident_chunks}
        )

    def test_per_item_classification_chunks(self):
        substance = {
            "uuid": "eeeeeeee-1111-2222-3333-666666666666",
            "substanceClass": "concept",
            "names": [{"name": "Has Classifications", "displayName": True}],
            "status": "approved",
            "codes": [
                {"code": "HUMAN_DRUG", "_isClassification": True},
                {"code": "PRESCRIPTION", "_isClassification": True},
            ],
        }
        chunks = self._chunker.chunk(substance)
        class_chunks = [c for c in chunks if c.section == "classifications"]
        # 2 per-item classification chunks — no batch summary.
        self.assertEqual(len(class_chunks), 2)
        self.assertTrue(
            all(
                c.metadata_json.get("chunk_type") == "classification"
                for c in class_chunks
            )
        )
        self.assertNotIn(
            "classification_batch",
            {c.metadata_json.get("chunk_type") for c in class_chunks},
        )

    def test_per_item_relationship_chunks(self):
        substance = {
            "uuid": "eeeeeeee-1111-2222-3333-777777777777",
            "substanceClass": "chemical",
            "names": [{"name": "Has Rels", "displayName": True}],
            "status": "approved",
            "relationships": [
                {
                    "type": "BINDER->LIGAND",
                    "relatedSubstance": {
                        "refPname": "PROTEIN A",
                        "refuuid": "p-a-uuid",
                    },
                },
                {
                    "type": "BINDER->LIGAND",
                    "relatedSubstance": {
                        "refPname": "PROTEIN B",
                        "refuuid": "p-b-uuid",
                    },
                },
                {
                    "type": "PARENT->CHILD",
                    "relatedSubstance": {
                        "refPname": "PARENT",
                        "refuuid": "parent-uuid",
                    },
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        rel_chunks = [c for c in chunks if c.section == "relationships"]
        # 3 per-item relationship chunks — no group / batch summary.
        self.assertEqual(len(rel_chunks), 3)
        self.assertTrue(
            all(
                c.metadata_json.get("chunk_type") == "relationship"
                for c in rel_chunks
            )
        )
        self.assertNotIn(
            "relationship_group",
            {c.metadata_json.get("chunk_type") for c in rel_chunks},
        )

    def test_relationship_sub_sections_by_type(self):
        """Relationships are routed to typed sub-sections under the
        ``relationships`` root, mirroring the identifiers /
        classifications split used for codes.

        Routing:

        * ``ACTIVE MOIETY`` / ``SUBSTANCE PART`` → ``activemoiety``
        * ``METABOLITE INACTIVE->PARENT`` → ``metabolites``
        * ``IMPURITY->PARENT`` → ``impurities``
        * type contains ``CONSTITUENT`` → ``constituents``
        * ``SALT/SOLVATE->PARENT`` → ``salts``
        * any other type → ``relationships`` (the root)
        """
        substance = {
            "uuid": "eeeeeeee-1111-2222-3333-aaaaaaaaaaaa",
            "substanceClass": "chemical",
            "names": [{"name": "Mixed Rels", "displayName": True}],
            "status": "approved",
            "relationships": [
                {
                    "type": "ACTIVE MOIETY",
                    "relatedSubstance": {
                        "refPname": "PARENT-1",
                        "refuuid": "p-1-uuid",
                    },
                },
                {
                    "type": "SUBSTANCE PART",
                    "relatedSubstance": {
                        "refPname": "PART-1",
                        "refuuid": "p-2-uuid",
                    },
                },
                {
                    "type": "METABOLITE INACTIVE->PARENT",
                    "relatedSubstance": {
                        "refPname": "MET-1",
                        "refuuid": "p-3-uuid",
                    },
                },
                {
                    "type": "IMPURITY->PARENT",
                    "relatedSubstance": {
                        "refPname": "IMP-1",
                        "refuuid": "p-4-uuid",
                    },
                },
                {
                    "type": "CONSTITUENT ALWAYS->SUBSTANCE",
                    "relatedSubstance": {
                        "refPname": "CON-1",
                        "refuuid": "p-5-uuid",
                    },
                },
                {
                    "type": "SALT/SOLVATE->PARENT",
                    "relatedSubstance": {
                        "refPname": "SALT-1",
                        "refuuid": "p-6-uuid",
                    },
                },
                {
                    "type": "BINDER->LIGAND",
                    "relatedSubstance": {
                        "refPname": "OTHER",
                        "refuuid": "p-7-uuid",
                    },
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        by_section: Dict[str, List[VectorDocument]] = {}
        for c in chunks:
            by_section.setdefault(c.section, []).append(c)

        # Two entries route to activemoiety (ACTIVE MOIETY + SUBSTANCE PART).
        self.assertEqual(len(by_section.get("activemoiety", [])), 2)
        self.assertEqual(len(by_section.get("metabolites", [])), 1)
        self.assertEqual(len(by_section.get("impurities", [])), 1)
        self.assertEqual(len(by_section.get("constituents", [])), 1)
        self.assertEqual(len(by_section.get("salts", [])), 1)
        # Untyped / unmatched types fall through to the root section.
        self.assertEqual(len(by_section.get("relationships", [])), 1)

        # All sub-section chunks share the same root_section parent.
        for section in (
            "activemoiety",
            "metabolites",
            "impurities",
            "constituents",
            "salts",
        ):
            for c in by_section[section]:
                self.assertEqual(c.root_section, "relationships")
                self.assertIn(
                    c.metadata_json.get("hierarchy", {}).get("parent_section"),
                    {"relationships"},
                )

    def test_relationship_sub_sections_resolve_to_root(self):
        """``sections_in_root("relationships")`` should include every
        new sub-section so the parent-child enricher can fan out the
        full relationships parent in one backend query.
        """
        members = set(sections_in_root("relationships"))
        for sub in (
            "relationships",
            "activemoiety",
            "metabolites",
            "impurities",
            "constituents",
            "salts",
        ):
            self.assertIn(sub, members)

    def test_no_batch_chunk_types_are_emitted(self):
        """No ``*_batch`` or ``reference_index`` chunk types anywhere
        in the output for a payload that exercises every list field."""
        substance = {
            "uuid": "eeeeeeee-1111-2222-3333-888888888888",
            "substanceClass": "chemical",
            "names": [
                {"name": "N1", "displayName": True},
                {"name": "N2"},
            ],
            "status": "approved",
            "codes": [{"code": "A"}, {"code": "B"}],
            "references": [{"id": "r1"}, {"id": "r2"}],
            "relationships": [
                {
                    "type": "PARENT->CHILD",
                    "relatedSubstance": {"refPname": "P", "refuuid": "p-uuid"},
                },
            ],
            "properties": [{"name": "MELTING_POINT", "value": "100 C"}],
            "modifications": [{"modificationType": "OXIDATION"}],
            "notes": [{"note": "test"}],
            "tags": [{"tag": "tag1"}],
            "structure": {"smiles": "CCO"},
        }
        chunks = self._chunker.chunk(substance)
        chunk_types = {c.metadata_json.get("chunk_type") for c in chunks}
        for forbidden in (
            "reference_index",
            "code_batch",
            "classification_batch",
            "relationship_group",
            "note_batch",
            "property_batch",
            "modification_batch",
            "tag_batch",
            "moiety_batch",
        ):
            self.assertNotIn(
                forbidden, chunk_types,
                f"Batch summary chunk type {forbidden!r} should not be emitted.",
            )

    def test_full_ibuprofen_payload_uses_per_item_chunking(self):
        """End-to-end: ibuprofen emits per-item chunks for every list
        field — references, codes, relationships, notes, etc.

        The total chunk count is much higher than the old optimized
        budget (~50) but every payload field is independently
        queryable.
        """
        # Opt in to admin validation notes — ibuprofen's notes are
        # mostly [Validation] entries, which are filtered by default.
        chunker = SubstanceChunker(
            class_=VectorDocument,
            config=ChunkerConfig(include_admin_validation_notes=True),
        )
        import json as _json
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "ibuprofen.json")
        if not os.path.exists(path):
            self.skipTest("ibuprofen.json fixture not present")
        with open(path) as f:
            sub = _json.load(f)

        chunks = chunker.chunk(sub)
        sections = {c.section for c in chunks}
        for s in (
            "overview", "names", "identifiers", "classifications",
            "chemical", "references", "relationships", "properties",
            "notes",
        ):
            self.assertIn(s, sections)
        # References, codes, and relationships are emitted per-item.
        # Ibuprofen carries 24 references / 11 codes / 11 relationships
        # (approximate; assert greater-than-zero so the test stays
        # robust against fixture edits).
        self.assertGreater(
            sum(1 for c in chunks if c.section == "references"),
            5,
        )
        self.assertGreater(
            sum(1 for c in chunks if c.section == "identifiers"),
            5,
        )
        self.assertGreater(
            sum(1 for c in chunks if c.section == "relationships"),
            5,
        )


class TestAccessStatus(unittest.TestCase):
    """Access status is surfaced as ``access_status`` on the metadata
    of every chunk in the summary, names, codes, classifications and
    definitions sections.

    The mapping is:

    * absent / ``None`` / empty list → ``"Public"``
    * non-empty list (e.g. ``["admin"]``) → ``"Protected"``

    Per-row access (a name, a code, a classification) takes priority
    over the substance's top-level access, so a substance with
    ``access=["admin"]`` at the top level can still carry public
    individual rows.
    """

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    # --- Pure helper ------------------------------------------------

    def test_access_status_for_helper(self):
        self.assertEqual(access_status_for(None), "Public")
        self.assertEqual(access_status_for([]), "Public")
        self.assertEqual(access_status_for(["admin"]), "Protected")
        self.assertEqual(access_status_for(["admin", "curator"]), "Protected")
        # Non-list truthy values degrade to Protected.
        self.assertEqual(access_status_for("admin"), "Protected")
        self.assertEqual(access_status_for(True), "Protected")
        # Non-list falsy values degrade to Public.
        self.assertEqual(access_status_for(False), "Public")
        self.assertEqual(access_status_for(0), "Public")
        self.assertEqual(access_status_for(""), "Public")

    # --- Summary (overview) ----------------------------------------

    def test_overview_defaults_to_public_when_access_absent(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000001",
            "substanceClass": "chemical",
            "names": [{"name": "X", "displayName": True}],
            "status": "approved",
        }
        chunks = self._chunker.chunk(substance)
        overview = next(c for c in chunks if c.section == "overview")
        self.assertEqual(overview.metadata_json.get("access_status"), "Public")
        self.assertIn("Access: Public", overview.text)
        self.assertEqual(overview.metadata_json.get("access"), [])

    def test_overview_is_protected_when_top_level_access_set(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000002",
            "substanceClass": "chemical",
            "names": [{"name": "X", "displayName": True}],
            "status": "approved",
            "access": ["admin"],
        }
        chunks = self._chunker.chunk(substance)
        overview = next(c for c in chunks if c.section == "overview")
        self.assertEqual(overview.metadata_json.get("access_status"), "Protected")
        self.assertIn("Access: Protected", overview.text)
        self.assertEqual(overview.metadata_json.get("access"), ["admin"])

    def test_overview_is_public_when_top_level_access_empty(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000003",
            "substanceClass": "chemical",
            "names": [{"name": "X", "displayName": True}],
            "status": "approved",
            "access": [],
        }
        chunks = self._chunker.chunk(substance)
        overview = next(c for c in chunks if c.section == "overview")
        self.assertEqual(overview.metadata_json.get("access_status"), "Public")

    # --- Names ------------------------------------------------------

    def test_name_uses_per_row_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000010",
            "substanceClass": "chemical",
            "names": [
                {"name": "Public Name", "displayName": True, "access": []},
                {"name": "Protected Name", "access": ["admin"]},
                {"name": "Untagged Name"},
            ],
            "status": "approved",
            "structure": {"smiles": "CCO"},
        }
        chunks = self._chunker.chunk(substance)
        names = [c for c in chunks if c.section == "names"]
        by_name = {c.metadata_json.get("name"): c for c in names}
        self.assertEqual(
            by_name["Public Name"].metadata_json.get("access_status"), "Public"
        )
        self.assertEqual(
            by_name["Protected Name"].metadata_json.get("access_status"), "Protected"
        )
        # Missing access field falls back to Public.
        self.assertEqual(
            by_name["Untagged Name"].metadata_json.get("access_status"), "Public"
        )
        # Each name's text exposes its own access status.
        self.assertIn("Access: Public", by_name["Public Name"].text)
        self.assertIn("Access: Protected", by_name["Protected Name"].text)
        # And the raw list is preserved on the chunk metadata.
        self.assertEqual(by_name["Protected Name"].metadata_json.get("access"), ["admin"])

    # --- Codes / Identifiers ----------------------------------------

    def test_identifier_uses_per_row_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000020",
            "substanceClass": "chemical",
            "names": [{"name": "X", "displayName": True}],
            "status": "approved",
            "codes": [
                {"code": "PUB-1", "codeSystem": "EXT", "access": []},
                {"code": "PRO-1", "codeSystem": "EXT", "access": ["admin"]},
            ],
            "structure": {"smiles": "CCO"},
        }
        chunks = self._chunker.chunk(substance)
        idents = [c for c in chunks if c.section == "identifiers"]
        by_code = {c.metadata_json.get("code"): c for c in idents}
        self.assertEqual(
            by_code["PUB-1"].metadata_json.get("access_status"), "Public"
        )
        self.assertEqual(
            by_code["PRO-1"].metadata_json.get("access_status"), "Protected"
        )
        self.assertIn("Access: Public", by_code["PUB-1"].text)
        self.assertIn("Access: Protected", by_code["PRO-1"].text)

    # --- Classifications -------------------------------------------

    def test_classification_uses_per_row_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000030",
            "substanceClass": "concept",
            "names": [{"name": "X", "displayName": True}],
            "status": "approved",
            "codes": [
                {
                    "code": "HUMAN_DRUG",
                    "_isClassification": True,
                    "access": ["admin"],
                },
                {
                    "code": "PRESCRIPTION",
                    "_isClassification": True,
                    "access": [],
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        classes = [c for c in chunks if c.section == "classifications"]
        by_code = {c.metadata_json.get("code"): c for c in classes}
        self.assertEqual(
            by_code["HUMAN_DRUG"].metadata_json.get("access_status"),
            "Protected",
        )
        self.assertEqual(
            by_code["PRESCRIPTION"].metadata_json.get("access_status"),
            "Public",
        )

    # --- Definitions: chemical (structure + moieties) -------------

    def test_chemical_definitions_use_structure_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000040",
            "substanceClass": "chemical",
            "names": [{"name": "X", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CC(=O)O",
                "formula": "C2H4O2",
                "mwt": 60.05,
                "access": ["admin"],
            },
            "moieties": [
                {
                    "smiles": "CC(=O)[O-]",
                    "formula": "C2H3O2",
                    "count": 1,
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        chem = [c for c in chunks if c.section == "chemical"]
        self.assertGreater(len(chem), 0)
        for c in chem:
            self.assertEqual(c.metadata_json.get("access_status"), "Protected")
            self.assertIn("Access: Protected", c.text)
        # Raw access list preserved on the metadata.
        self.assertEqual(
            chem[0].metadata_json.get("access"),
            ["admin"],
        )

    def test_chemical_definitions_default_to_public_when_structure_access_empty(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000041",
            "substanceClass": "chemical",
            "names": [{"name": "X", "displayName": True}],
            "status": "approved",
            "structure": {"smiles": "CCO", "access": []},
        }
        chunks = self._chunker.chunk(substance)
        chem = [c for c in chunks if c.section == "chemical"]
        for c in chem:
            self.assertEqual(c.metadata_json.get("access_status"), "Public")

    # --- Definitions: protein / nucleic acid / polymer / etc. -----

    def test_protein_definitions_use_protein_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000050",
            "substanceClass": "protein",
            "names": [{"name": "TestProt", "displayName": True}],
            "status": "approved",
            "protein": {
                "proteinType": "Antibody",
                "subunits": [
                    {"subunitIndex": "1", "sequence": "MKTLLLTLVVVTIVCLDLGY"},
                ],
                "access": ["admin"],
            },
        }
        chunks = self._chunker.chunk(substance)
        prot = [c for c in chunks if c.section == "protein"]
        self.assertGreater(len(prot), 0)
        for c in prot:
            self.assertEqual(c.metadata_json.get("access_status"), "Protected")

    def test_nucleic_acid_definitions_use_nucleic_acid_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000060",
            "substanceClass": "nucleicAcid",
            "names": [{"name": "TestDNA", "displayName": True}],
            "status": "approved",
            "nucleicAcid": {
                "sequenceType": "DNA",
                "subunits": [
                    {"subunitIndex": "1", "sequence": "ATCGATCGATCGATCG"},
                ],
                "access": ["curator"],
            },
        }
        chunks = self._chunker.chunk(substance)
        na = [c for c in chunks if c.section == "nucleicacid"]
        for c in na:
            self.assertEqual(c.metadata_json.get("access_status"), "Protected")

    def test_polymer_definitions_use_polymer_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000070",
            "substanceClass": "polymer",
            "names": [{"name": "TestPoly", "displayName": True}],
            "status": "approved",
            "polymer": {
                "classification": {"polymerClass": "HOMOPOLYMER"},
                "displayStructure": {"smiles": "CCO", "formula": "C2H6O"},
                "access": ["admin"],
            },
        }
        chunks = self._chunker.chunk(substance)
        pol = [c for c in chunks if c.section == "polymer"]
        for c in pol:
            self.assertEqual(c.metadata_json.get("access_status"), "Protected")

    def test_structurally_diverse_definitions_use_source_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000080",
            "substanceClass": "structurallyDiverse",
            "names": [{"name": "TestSD", "displayName": True}],
            "status": "approved",
            "structurallyDiverse": {
                "sourceMaterialClass": "PLANT",
                "organismFamily": "Fabaceae",
                "access": ["admin"],
            },
        }
        chunks = self._chunker.chunk(substance)
        sd = [c for c in chunks if c.section == "structurallydiverse"]
        self.assertEqual(len(sd), 1)
        self.assertEqual(sd[0].metadata_json.get("access_status"), "Protected")

    def test_mixture_definitions_use_mixture_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-000000000090",
            "substanceClass": "mixture",
            "names": [{"name": "TestMix", "displayName": True}],
            "status": "approved",
            "mixture": {
                "components": [
                    {
                        "type": "ACTIVE",
                        "substance": {"name": "ActiveA", "refuuid": "a-uuid"},
                    },
                ],
                "access": ["admin"],
            },
        }
        chunks = self._chunker.chunk(substance)
        mix = [c for c in chunks if c.section == "mixture"]
        self.assertGreater(len(mix), 0)
        for c in mix:
            self.assertEqual(c.metadata_json.get("access_status"), "Protected")

    def test_specified_substance_definitions_use_specified_substance_access(self):
        substance = {
            "uuid": "11111111-0000-0000-0000-0000000000a0",
            "substanceClass": "specifiedSubstanceG1",
            "names": [{"name": "TestG1", "displayName": True}],
            "status": "approved",
            "specifiedSubstance": {
                "constituents": [
                    {
                        "role": "ACTIVE",
                        "substance": {"name": "A", "refuuid": "a-uuid"},
                    },
                ],
                "access": ["admin"],
            },
        }
        chunks = self._chunker.chunk(substance)
        ss = [c for c in chunks if c.section == "specifiedsubstance"]
        self.assertEqual(len(ss), 1)
        self.assertEqual(ss[0].metadata_json.get("access_status"), "Protected")

    # --- End-to-end: real protected fixture -----------------------

    def test_protected_ibuprofen_like_fixture(self):
        """End-to-end: a substance with ``access=['admin']`` at the
        top level emits Protected on the overview; per-row access
        on names/codes (if any) takes priority and is surfaced
        verbatim.
        """
        import json as _json
        import os

        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "examples",
            "b7d919e4-7fef-4c9e-bb03-f8112608c050.json",
        )
        if not os.path.exists(path):
            self.skipTest("protected fixture not present")
        with open(path) as f:
            sub = _json.load(f)

        chunks = self._chunker.chunk(sub)
        overview = next(c for c in chunks if c.section == "overview")
        self.assertEqual(overview.metadata_json.get("access_status"), "Protected")
        # Every chunk in the substance carries an access_status field.
        for c in chunks:
            self.assertIn(
                c.metadata_json.get("access_status"),
                {"Public", "Protected"},
            )

    def test_public_ibuprofen_fixture(self):
        """End-to-end: a substance with no top-level access emits
        ``Public`` everywhere on the targeted sections."""
        import json as _json
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "ibuprofen.json")
        if not os.path.exists(path):
            self.skipTest("ibuprofen.json fixture not present")
        with open(path) as f:
            sub = _json.load(f)

        chunks = self._chunker.chunk(sub)
        # Targeted sections must all be Public.
        for section in ("overview", "names", "identifiers", "classifications",
                        "chemical", "references", "relationships", "properties",
                        "notes"):
            for c in chunks:
                if c.section == section:
                    self.assertEqual(
                        c.metadata_json.get("access_status"), "Public",
                        f"section={section} expected Public",
                    )


class TestNameTypeAndOrganizations(unittest.TestCase):
    """Name chunks surface a human-readable ``name_type`` label and
    the naming organizations (e.g. INN, USAN, INCI) carried on
    the name entry.

    The raw ``type`` code (``of`` / ``sys`` / ``bn`` / ``cn`` /
    ``sci`` / ``syn`` / ``cd``) is preserved on the metadata so
    callers that need the original code can still filter on it.
    """

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    # --- Pure helper ------------------------------------------------

    def test_name_type_label_known_codes(self):
        self.assertEqual(name_type_label("of"), "Official Name")
        self.assertEqual(name_type_label("sys"), "Systematic Name")
        self.assertEqual(name_type_label("bn"), "Brand Name")
        self.assertEqual(name_type_label("cn"), "Common Name")
        self.assertEqual(name_type_label("sci"), "Scientific Name")
        self.assertEqual(name_type_label("syn"), "Synonym")
        self.assertEqual(name_type_label("cd"), "Code")

    def test_name_type_label_unknown_and_empty(self):
        # Unknown codes are returned verbatim so future payload
        # shapes still surface in retrieval.
        self.assertEqual(name_type_label("xyz"), "xyz")
        # Empty / missing code returns empty string.
        self.assertEqual(name_type_label(""), "")
        self.assertEqual(name_type_label(None), "")
        # Lookup is case-insensitive.
        self.assertEqual(name_type_label("OF"), "Official Name")
        self.assertEqual(name_type_label(" Sys "), "Systematic Name")

    # --- Type label on chunk text / metadata -----------------------

    def test_official_name_uses_human_readable_label(self):
        substance = {
            "uuid": "22222222-0000-0000-0000-000000000001",
            "substanceClass": "chemical",
            "names": [
                {"name": "Aspirin", "type": "of", "displayName": True},
            ],
            "status": "approved",
            "structure": {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "names")
        self.assertIn("Type: Official Name", c.text)
        # Raw code preserved on the metadata.
        self.assertEqual(c.metadata_json.get("name_type"), "of")
        self.assertEqual(
            c.metadata_json.get("name_type_label"), "Official Name"
        )

    def test_unknown_type_code_surfaces_verbatim(self):
        substance = {
            "uuid": "22222222-0000-0000-0000-000000000002",
            "substanceClass": "chemical",
            "names": [
                {"name": "Mystery", "type": "future_code", "displayName": True},
            ],
            "status": "approved",
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "names")
        self.assertIn("Type: future_code", c.text)
        # label is the raw code when unknown.
        self.assertEqual(
            c.metadata_json.get("name_type_label"), "future_code"
        )

    def test_no_type_field_emits_no_type_line(self):
        substance = {
            "uuid": "22222222-0000-0000-0000-000000000003",
            "substanceClass": "chemical",
            "names": [
                {"name": "NoType", "displayName": True},
            ],
            "status": "approved",
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "names")
        self.assertNotIn("Type:", c.text)
        self.assertEqual(c.metadata_json.get("name_type"), "")
        self.assertEqual(c.metadata_json.get("name_type_label"), "")

    # --- nameOrgs ---------------------------------------------------

    def test_official_name_carries_name_orgs(self):
        substance = {
            "uuid": "22222222-0000-0000-0000-000000000010",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Ibuprofen",
                    "type": "of",
                    "displayName": True,
                    "nameOrgs": [
                        {"nameOrg": "INN", "uuid": "org-1"},
                        {"nameOrg": "INCI", "uuid": "org-2"},
                        {"nameOrg": "USAN", "uuid": "org-3"},
                    ],
                },
            ],
            "status": "approved",
            "structure": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"},
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "names")
        self.assertIn("Naming Organizations: INN, INCI, USAN", c.text)
        self.assertEqual(
            c.metadata_json.get("name_orgs"), ["INN", "INCI", "USAN"]
        )

    def test_non_official_name_with_name_orgs_is_still_surfaced(self):
        """``nameOrgs`` is rendered whenever present — not gated on
        the ``type`` field — so retrieval can resolve naming bodies
        for any name that carries the field."""
        substance = {
            "uuid": "22222222-0000-0000-0000-000000000011",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "BrandX",
                    "type": "bn",
                    "nameOrgs": [{"nameOrg": "INN"}],
                },
            ],
            "status": "approved",
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "names")
        self.assertIn("Type: Brand Name", c.text)
        self.assertIn("Naming Organizations: INN", c.text)

    def test_name_without_name_orgs_omits_organizations_line(self):
        substance = {
            "uuid": "22222222-0000-0000-0000-000000000012",
            "substanceClass": "chemical",
            "names": [
                {"name": "Plain", "type": "of", "displayName": True},
            ],
            "status": "approved",
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "names")
        self.assertNotIn("Naming Organizations", c.text)
        self.assertEqual(c.metadata_json.get("name_orgs"), [])

    def test_empty_name_orgs_omits_organizations_line(self):
        substance = {
            "uuid": "22222222-0000-0000-0000-000000000013",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Empty",
                    "type": "of",
                    "displayName": True,
                    "nameOrgs": [],
                },
            ],
            "status": "approved",
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "names")
        self.assertNotIn("Naming Organizations", c.text)
        self.assertEqual(c.metadata_json.get("name_orgs"), [])

    def test_name_orgs_filters_out_blank_entries(self):
        substance = {
            "uuid": "22222222-0000-0000-0000-000000000014",
            "substanceClass": "chemical",
            "names": [
                {
                    "name": "Mixed",
                    "type": "of",
                    "displayName": True,
                    "nameOrgs": [
                        {"nameOrg": "INN"},
                        {"nameOrg": ""},
                        {"uuid": "no-name-field"},
                        {"nameOrg": "USAN"},
                    ],
                },
            ],
            "status": "approved",
        }
        chunks = self._chunker.chunk(substance)
        c = next(c for c in chunks if c.section == "names")
        self.assertIn("Naming Organizations: INN, USAN", c.text)
        self.assertEqual(c.metadata_json.get("name_orgs"), ["INN", "USAN"])

    # --- End-to-end: real payload ---------------------------------

    def test_real_ibuprofen_name_orgs(self):
        """The real ibuprofen payload exposes an Official Name
        (Ibuprofen) with three naming organizations — verify they
        all reach the chunk.
        """
        import json as _json
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "ibuprofen.json")
        if not os.path.exists(path):
            self.skipTest("ibuprofen.json fixture not present")
        with open(path) as f:
            sub = _json.load(f)

        chunks = self._chunker.chunk(sub)
        official = [
            c for c in chunks
            if c.section == "names"
            and c.metadata_json.get("name_type") == "of"
        ]
        self.assertGreaterEqual(len(official), 1)
        c = official[0]
        # Human-readable label is rendered in the text.
        self.assertIn("Type: Official Name", c.text)
        # All three naming organizations are surfaced.
        orgs = c.metadata_json.get("name_orgs") or []
        for org in ("INN", "INCI", "USAN"):
            self.assertIn(org, orgs)
        self.assertIn("Naming Organizations:", c.text)


class TestEnrichmentFields(unittest.TestCase):
    """Additional enrichment fields on identifier / classification /
    structure / moiety / reference / note chunks.

    These changes were requested in the same session as the access
    status / name type changes; they were added together to keep
    the per-item chunking output maximally useful for retrieval.

    * **Identifiers / Classifications** carry a separate
      ``Code System:`` line and (when present) a ``URL:`` line.
      The identifier text now leads with ``Identifier:`` instead
      of ``Code:``.
    * **Structure** chunks lead with a ``Chemical Structure``
      header and surface ``InChI:`` and ``InChI Key:`` lines.
    * **Moiety** chunks surface ``InChI:`` and ``InChI Key:``
      lines too.
    * **Reference** chunks surface a ``URL:`` line when present.
    * **Note** chunks no longer carry ``is_admin_validation`` on
      the metadata (the validation flag is encoded in the text
      prefix ``[Validation]``).
    """

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    # ---- Identifier / Classification --------------------------------

    def test_identifier_text_uses_identifier_label(self):
        substance = {
            "uuid": "aaaaaaaa-0000-0000-0000-000000000001",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "codes": [
                {
                    "code": "m6189",
                    "codeSystem": "MERCK INDEX",
                    "type": "PRIMARY",
                    "url": "https://merckindex.rsc.org/monographs/m6189",
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        ident = next(c for c in chunks if c.section == "identifiers")
        self.assertTrue(ident.text.startswith("Identifier: MERCK INDEX:m6189 (PRIMARY)"))
        # The legacy ``Code:`` label is no longer used for identifiers.
        self.assertNotIn("\nCode:", ident.text)
        # The first line of the chunk is the Identifier label.
        self.assertEqual(ident.text.splitlines()[0],
                         "Identifier: MERCK INDEX:m6189 (PRIMARY)")

    def test_identifier_includes_code_system_line(self):
        substance = {
            "uuid": "aaaaaaaa-0000-0000-0000-000000000002",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "codes": [
                {
                    "code": "m6189",
                    "codeSystem": "MERCK INDEX",
                    "type": "PRIMARY",
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        ident = next(c for c in chunks if c.section == "identifiers")
        self.assertIn("Code System: MERCK INDEX", ident.text)
        # Code system also on metadata for filtering.
        self.assertEqual(
            ident.metadata_json.get("code_system"), "MERCK INDEX"
        )

    def test_identifier_includes_url_line_when_present(self):
        substance = {
            "uuid": "aaaaaaaa-0000-0000-0000-000000000003",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "codes": [
                {
                    "code": "N02AJ19",
                    "codeSystem": "WHO-ATC",
                    "url": "http://www.whocc.no/atc_ddd_index/?code=N02AJ19",
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        ident = next(c for c in chunks if c.section == "identifiers")
        self.assertIn(
            "URL: http://www.whocc.no/atc_ddd_index/?code=N02AJ19", ident.text
        )
        self.assertEqual(
            ident.metadata_json.get("code_url"),
            "http://www.whocc.no/atc_ddd_index/?code=N02AJ19",
        )

    def test_identifier_omits_url_line_when_absent(self):
        substance = {
            "uuid": "aaaaaaaa-0000-0000-0000-000000000004",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "codes": [
                {"code": "ABC", "codeSystem": "INTERNAL"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        ident = next(c for c in chunks if c.section == "identifiers")
        self.assertNotIn("URL:", ident.text)
        self.assertEqual(ident.metadata_json.get("code_url"), "")

    def test_classification_includes_code_system_line(self):
        substance = {
            "uuid": "aaaaaaaa-0000-0000-0000-000000000005",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "codes": [
                {
                    "code": "N02AJ19",
                    "codeSystem": "WHO-ATC",
                    "_isClassification": True,
                    "url": "http://www.whocc.no/atc_ddd_index/?code=N02AJ19",
                    "comments": "ATC|NERVOUS SYSTEM",
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        cls_chunk = next(c for c in chunks if c.section == "classifications")
        # Classification label preserved (renamed only for identifiers).
        self.assertIn("Classification: WHO-ATC:N02AJ19", cls_chunk.text)
        self.assertIn("Code System: WHO-ATC", cls_chunk.text)
        self.assertIn(
            "URL: http://www.whocc.no/atc_ddd_index/?code=N02AJ19",
            cls_chunk.text,
        )
        self.assertIn("Comments: ATC|NERVOUS SYSTEM", cls_chunk.text)
        # Metadata exposes code_system and code_url.
        self.assertEqual(cls_chunk.metadata_json.get("code_system"), "WHO-ATC")
        self.assertEqual(
            cls_chunk.metadata_json.get("code_url"),
            "http://www.whocc.no/atc_ddd_index/?code=N02AJ19",
        )

    def test_classification_omits_url_when_absent(self):
        substance = {
            "uuid": "aaaaaaaa-0000-0000-0000-000000000006",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "codes": [
                {
                    "code": "HUMAN_DRUG",
                    "_isClassification": True,
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        cls_chunk = next(c for c in chunks if c.section == "classifications")
        self.assertNotIn("URL:", cls_chunk.text)
        self.assertEqual(cls_chunk.metadata_json.get("code_url"), "")

    # ---- Structure (Chemical Structure header + InChI) -------------

    def test_structure_chunk_has_chemical_structure_header(self):
        substance = {
            "uuid": "bbbbbbbb-0000-0000-0000-000000000001",
            "substanceClass": "chemical",
            "names": [{"name": "Ethanol", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CCO",
                "formula": "C2H6O",
                "mwt": 46.07,
                "stereochemistry": "ACHIRAL",
                "opticalActivity": "NONE",
                "atropisomerism": "No",
            },
        }
        chunks = self._chunker.chunk(substance)
        struct = next(
            c for c in chunks
            if c.section == "chemical"
            and c.metadata_json.get("chunk_type") == "structure"
        )
        # First line is the header.
        self.assertEqual(struct.text.splitlines()[0], "Chemical Structure")
        self.assertIn("SMILES: CCO", struct.text)
        self.assertIn("Formula: C2H6O", struct.text)

    def test_structure_chunk_includes_inchi_and_inchi_key(self):
        substance = {
            "uuid": "bbbbbbbb-0000-0000-0000-000000000002",
            "substanceClass": "chemical",
            "names": [{"name": "Aspirin", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "formula": "C9H8O4",
                "mwt": 180.16,
                "_inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
                "_inchiKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            },
        }
        chunks = self._chunker.chunk(substance)
        struct = next(
            c for c in chunks
            if c.section == "chemical"
            and c.metadata_json.get("chunk_type") == "structure"
        )
        self.assertIn(
            "InChI: InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
            struct.text,
        )
        self.assertIn("InChI Key: BSYNRYMUTXBXSQ-UHFFFAOYSA-N", struct.text)
        # Metadata exposes inchi / inchi_key for filtering.
        self.assertEqual(
            struct.metadata_json.get("inchi"),
            "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
        )
        self.assertEqual(
            struct.metadata_json.get("inchi_key"), "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        )

    def test_structure_omits_inchi_lines_when_absent(self):
        substance = {
            "uuid": "bbbbbbbb-0000-0000-0000-000000000003",
            "substanceClass": "chemical",
            "names": [{"name": "NoInChI", "displayName": True}],
            "status": "approved",
            "structure": {
                "smiles": "CCO",
                "formula": "C2H6O",
            },
        }
        chunks = self._chunker.chunk(substance)
        struct = next(
            c for c in chunks
            if c.section == "chemical"
            and c.metadata_json.get("chunk_type") == "structure"
        )
        self.assertNotIn("InChI:", struct.text)
        self.assertNotIn("InChI Key:", struct.text)
        # Header is still present.
        self.assertTrue(struct.text.startswith("Chemical Structure\n"))

    # ---- Moiety -----------------------------------------------------

    def test_moiety_chunk_includes_inchi_and_inchi_key(self):
        substance = {
            "uuid": "cccccccc-0000-0000-0000-000000000001",
            "substanceClass": "chemical",
            "names": [{"name": "Aspirin", "displayName": True}],
            "status": "approved",
            "structure": {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            "moieties": [
                {
                    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "formula": "C9H8O4",
                    "mwt": 180.16,
                    "count": 1,
                    "_inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
                    "_inchiKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        moiety = next(
            c for c in chunks
            if c.section == "chemical"
            and c.metadata_json.get("chunk_type") == "moiety"
        )
        self.assertIn(
            "InChI: InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
            moiety.text,
        )
        self.assertIn("InChI Key: BSYNRYMUTXBXSQ-UHFFFAOYSA-N", moiety.text)
        # Metadata exposes inchi / inchi_key.
        self.assertEqual(
            moiety.metadata_json.get("inchi_key"), "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        )

    def test_moiety_omits_inchi_lines_when_absent(self):
        substance = {
            "uuid": "cccccccc-0000-0000-0000-000000000002",
            "substanceClass": "chemical",
            "names": [{"name": "NoInChIMoiety", "displayName": True}],
            "status": "approved",
            "structure": {"smiles": "CCO"},
            "moieties": [
                {"smiles": "C", "count": 1},
            ],
        }
        chunks = self._chunker.chunk(substance)
        moiety = next(
            c for c in chunks
            if c.section == "chemical"
            and c.metadata_json.get("chunk_type") == "moiety"
        )
        self.assertNotIn("InChI:", moiety.text)
        self.assertNotIn("InChI Key:", moiety.text)

    # ---- Reference --------------------------------------------------

    def test_reference_includes_url_line_when_present(self):
        substance = {
            "uuid": "dddddddd-0000-0000-0000-000000000001",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "references": [
                {
                    "docType": "PATENT",
                    "citation": "Patent",
                    "url": "https://example.com/patent.pdf",
                },
            ],
        }
        chunks = self._chunker.chunk(substance)
        ref = next(c for c in chunks if c.section == "references")
        self.assertIn("URL: https://example.com/patent.pdf", ref.text)
        self.assertEqual(
            ref.metadata_json.get("reference_url"),
            "https://example.com/patent.pdf",
        )

    def test_reference_omits_url_line_when_absent(self):
        substance = {
            "uuid": "dddddddd-0000-0000-0000-000000000002",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "references": [
                {"docType": "journal article", "id": "r1"},
            ],
        }
        chunks = self._chunker.chunk(substance)
        ref = next(c for c in chunks if c.section == "references")
        self.assertNotIn("URL:", ref.text)
        self.assertEqual(ref.metadata_json.get("reference_url"), "")

    # ---- Note metadata ---------------------------------------------

    def test_note_metadata_omits_is_admin_validation(self):
        """``is_admin_validation`` is encoded in the text prefix
        ``[Validation]``; the metadata field is no longer emitted
        so retrieval filters that key off metadata continue to
        match every note (or none, depending on intent)."""
        # Opt in so both [Validation] and curator notes are emitted
        # as their own chunks — the test wants to inspect every
        # emitted note's metadata.
        chunker = SubstanceChunker(
            class_=VectorDocument,
            config=ChunkerConfig(include_admin_validation_notes=True),
        )
        substance = {
            "uuid": "eeeeeeee-0000-0000-0000-000000000001",
            "substanceClass": "chemical",
            "names": [{"name": "Test", "displayName": True}],
            "status": "approved",
            "notes": [
                {"note": "[Validation]WARNING:potential duplicate"},
                {"note": "Curator: needs review"},
            ],
        }
        chunks = chunker.chunk(substance)
        note_chunks = [c for c in chunks if c.section == "notes"]
        self.assertEqual(len(note_chunks), 2)
        for c in note_chunks:
            self.assertNotIn("is_admin_validation", c.metadata_json)

    # ---- End-to-end: real ibuprofen payload ------------------------

    def test_real_ibuprofen_enrichment(self):
        """End-to-end: the real ibuprofen payload exposes InChI on
        the structure and moiety chunks, ``Code System:`` and
        ``URL:`` on the classification chunk, and a ``Identifier:``
        label on the identifier chunk."""
        import json as _json
        import os

        path = os.path.join(
            os.path.dirname(__file__), "..", "ibuprofen.json"
        )
        if not os.path.exists(path):
            self.skipTest("ibuprofen.json fixture not present")
        with open(path) as f:
            sub = _json.load(f)

        chunks = self._chunker.chunk(sub)

        # Structure chunk: header + InChI.
        struct = next(
            c for c in chunks
            if c.section == "chemical"
            and c.metadata_json.get("chunk_type") == "structure"
        )
        self.assertEqual(
            struct.text.splitlines()[0], "Chemical Structure"
        )
        self.assertIn("InChI:", struct.text)
        self.assertIn("InChI Key: HEFNNWSXXWATRW-UHFFFAOYSA-N", struct.text)

        # Moieties chunk: InChI / InChI Key lines.
        moiety = next(
            c for c in chunks
            if c.section == "chemical"
            and c.metadata_json.get("chunk_type") == "moiety"
        )
        self.assertIn("InChI Key: HEFNNWSXXWATRW-UHFFFAOYSA-N", moiety.text)

        # Classification chunk: Code System + URL.
        cls_chunk = next(
            c for c in chunks
            if c.section == "classifications"
        )
        self.assertIn("Code System: WHO-ATC", cls_chunk.text)
        self.assertIn(
            "URL: http://www.whocc.no/atc_ddd_index/?code=N02AJ19&showdescription=yes",
            cls_chunk.text,
        )

        # Identifier chunk: Identifier label + Code System + URL.
        ident = next(
            c for c in chunks
            if c.section == "identifiers"
        )
        self.assertTrue(ident.text.startswith("Identifier:"))
        self.assertIn("Code System:", ident.text)
        self.assertIn("URL:", ident.text)


if __name__ == "__main__":
    unittest.main()
