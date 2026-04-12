"""
GSRS MCP Server - Chunking Service Tests

Tests for the ChunkerService using gsrs.model library's
Substance.to_embedding_chunks() method.
"""
import unittest

from gsrs.model import Substance
from gsrs.services.ai import SubstanceChunker

from app.models import VectorDocument


class TestChunkerService(unittest.TestCase):
    """Tests for ChunkerService using gsrs.model."""

    @classmethod
    def setUpClass(cls):
        cls._chunker = SubstanceChunker(class_=VectorDocument)

    @staticmethod
    def _to_substance(payload):
        return Substance.model_validate(payload)

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

        chunks = self._chunker.chunk(self._to_substance(substance))

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, VectorDocument) for c in chunks))

        root_chunks = [c for c in chunks if c.section == "overview"]
        self.assertGreater(len(root_chunks), 0)
        self.assertIn("Aspirin", root_chunks[0].text)

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

        chunks = self._chunker.chunk(self._to_substance(substance))

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

        chunks = self._chunker.chunk(self._to_substance(substance))

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

        chunks = self._chunker.chunk(self._to_substance(substance))

        # Code info is included in overview and identifier sections
        all_text = " ".join([c.text for c in chunks])
        self.assertIn("ASA-101", all_text)


if __name__ == "__main__":
    unittest.main()


class TestEmbeddingColumnType(unittest.TestCase):
    """Tests for dynamic embedding column type (Vector vs HalfVec)."""

    def test_embedding_column_exists_on_model(self):
        """Test that embedding column is defined on VectorDocument."""
        columns = {c.key for c in VectorDocument.__table__.columns}
        self.assertIn("embedding", columns)

    def test_embedding_column_uses_correct_type_for_current_dimension(self):
        """Test embedding column type matches current EMBEDDING_DIMENSION setting."""
        from pgvector.sqlalchemy import Vector, HALFVEC
        from app.config import settings

        col = VectorDocument.__table__.columns.embedding
        if settings.embedding_dimension > 2000:
            self.assertIsInstance(col.type, HALFVEC)
            self.assertEqual(col.type.dim, settings.embedding_dimension)
        else:
            self.assertIsInstance(col.type, Vector)
            self.assertEqual(col.type.dim, settings.embedding_dimension)

    def test_hnsw_index_exists_in_table_args(self):
        """Test HNSW index is defined in __table_args__."""
        indexes = VectorDocument.__table__.indexes
        hnsw_index = None
        for idx in indexes:
            if idx.name == "idx_embedding_hnsw":
                hnsw_index = idx
                break

        self.assertIsNotNone(hnsw_index, "idx_embedding_hnsw should be in __table_args__")

        # Check the index covers the embedding column
        col_names = [c.name for c in hnsw_index.columns]
        self.assertIn("embedding", col_names)

    def test_hnsw_index_uses_correct_operator_class(self):
        """Test HNSW index dialect kwargs have correct postgresql_ops."""
        from app.config import settings

        indexes = {idx.name: idx for idx in VectorDocument.__table__.indexes}
        hnsw_index = indexes.get("idx_embedding_hnsw")
        self.assertIsNotNone(hnsw_index)

        # Check postgresql dialect options
        dialect_kwargs = hnsw_index.dialect_kwargs
        self.assertEqual(dialect_kwargs.get("postgresql_using"), "hnsw")
        self.assertIn("postgresql_with", dialect_kwargs)
        self.assertIn("postgresql_ops", dialect_kwargs)

        expected_op_class = (
            "halfvec_cosine_ops"
            if settings.embedding_dimension > 2000
            else "vector_cosine_ops"
        )
        self.assertEqual(
            dialect_kwargs["postgresql_ops"]["embedding"],
            expected_op_class,
        )

    def test_hnsw_index_hnsw_parameters(self):
        """Test HNSW index has correct m and ef_construction values."""
        indexes = {idx.name: idx for idx in VectorDocument.__table__.indexes}
        hnsw_index = indexes.get("idx_embedding_hnsw")
        self.assertIsNotNone(hnsw_index)

        dialect_kwargs = hnsw_index.dialect_kwargs
        postgresql_with = dialect_kwargs.get("postgresql_with", {})
        self.assertEqual(postgresql_with.get("m"), 16)
        self.assertEqual(postgresql_with.get("ef_construction"), 64)
