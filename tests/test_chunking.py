"""
GSRS MCP Server - Chunking Service Tests

Tests for the native ChunkerService using gsrs.model library's
Substance.to_embedding_chunks() method.
"""
import unittest

from app.models import VectorDocument
from app.services.chunker import ChunkerConfig, SubstanceChunker


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
        structure_chunks = [c for c in chunks if c.section == "structure"]
        self.assertGreater(len(structure_chunks), 0)
        self.assertIn("Protein", structure_chunks[0].text)
        self.assertIn("CHO cells", structure_chunks[0].text)
        self.assertIn("Subunits: 2", structure_chunks[0].text)

        seq_chunks = [c for c in chunks if c.section == "sequence"]
        self.assertGreater(len(seq_chunks), 0)
        # Two subunit summary chunks (sequence segmentation disabled by default)
        self.assertTrue(any("Subunit 1 Sequence:" in c.text for c in seq_chunks))
        self.assertTrue(any("Subunit 2 Sequence:" in c.text for c in seq_chunks))

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
        structure_chunks = [c for c in chunks if c.section == "structure"]
        self.assertGreater(len(structure_chunks), 0)
        self.assertIn("Nucleic Acid", structure_chunks[0].text)
        self.assertIn("Synthetic", structure_chunks[0].text)

        seq_chunks = [c for c in chunks if c.section == "sequence"]
        self.assertTrue(any("Subunit 1 Sequence:" in c.text for c in seq_chunks))

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
        structure_chunks = [c for c in chunks if c.section == "structure"]
        self.assertGreater(len(structure_chunks), 0)
        self.assertIn("Polymer", structure_chunks[0].text)
        self.assertIn("HOMOPOLYMER", structure_chunks[0].text)
        self.assertIn("LINEAR", structure_chunks[0].text)
        self.assertIn("Glucose", structure_chunks[0].text)

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
        structure_chunks = [c for c in chunks if c.section == "structure"]
        self.assertGreater(len(structure_chunks), 0)
        text = structure_chunks[0].text
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
        structure_chunks = [c for c in chunks if c.section == "structure"]
        self.assertGreater(len(structure_chunks), 0)
        self.assertIn("Mixture", structure_chunks[0].text)
        self.assertIn("Components: 2", structure_chunks[0].text)

        comp_chunks = [c for c in chunks if c.section == "composition"]
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
        self.assertEqual(len(tag_chunks), 1)
        self.assertIn("clinical", tag_chunks[0].text)
        self.assertIn("inactive", tag_chunks[0].text)

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
        structure_chunks = [c for c in chunks if c.section == "structure"]
        self.assertEqual(len(structure_chunks), 0)

        # Overview should still work
        overview_chunks = [c for c in chunks if c.section == "overview"]
        self.assertGreater(len(overview_chunks), 0)
        self.assertIn("Test Unknown", overview_chunks[0].text)


if __name__ == "__main__":
    unittest.main()
