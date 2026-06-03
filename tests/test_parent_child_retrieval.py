"""
Tests for Parent-Child Retrieval using Virtual Parent Reconstruction
"""
import unittest
from uuid import uuid4
from app.models import VectorDocument
from app.services.parent_child_retrieval import (
    ParentContextEnricher,
    ParentIdentity,
)
from unittest.mock import MagicMock, Mock


class TestParentIdentity(unittest.TestCase):
    """Test ParentIdentity class."""

    def test_parent_identity_creation(self):
        """Test creating parent identity."""
        doc_id = uuid4()
        identity = ParentIdentity(document_id=doc_id, root_section="root")
        self.assertEqual(identity.document_id, doc_id)
        self.assertEqual(identity.root_section, "root")

    def test_parent_identity_equality(self):
        """Test parent identity equality."""
        doc_id = uuid4()
        identity1 = ParentIdentity(document_id=doc_id, root_section="root")
        identity2 = ParentIdentity(document_id=doc_id, root_section="root")
        self.assertEqual(identity1, identity2)

    def test_parent_identity_hash(self):
        """Test parent identity hashing."""
        doc_id = uuid4()
        identity1 = ParentIdentity(document_id=doc_id, root_section="root")
        identity2 = ParentIdentity(document_id=doc_id, root_section="root")
        # Can be used in sets
        identity_set = {identity1, identity2}
        self.assertEqual(len(identity_set), 1)

    def test_parent_identity_repr(self):
        """Test parent identity string representation."""
        doc_id = uuid4()
        identity = ParentIdentity(document_id=doc_id, root_section="root")
        repr_str = repr(identity)
        self.assertIn("ParentIdentity", repr_str)
        self.assertIn("root", repr_str)


class TestParentContextEnricher(unittest.TestCase):
    """Test ParentContextEnricher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db = MagicMock()
        self.enricher = ParentContextEnricher(self.mock_db)
        self.doc_id = uuid4()

    def test_extract_root_section_from_metadata(self):
        """Test extracting root section from metadata."""
        chunk = VectorDocument(
            chunk_id="test_chunk",
            document_id=self.doc_id,
            section="someSection",
            text="test text",
            embedding=[0.1] * 384,
            metadata_json={"root_section": "compound"},
        )
        root_section = self.enricher.extract_root_section(chunk)
        self.assertEqual(root_section, "compound")

    def test_extract_root_section_from_hierarchy(self):
        """Test extracting root section from hierarchy metadata."""
        chunk = VectorDocument(
            chunk_id="test_chunk",
            document_id=self.doc_id,
            section="names",
            text="test text",
            embedding=[0.1] * 384,
            metadata_json={"hierarchy": {"parent_section": "compound"}},
        )
        root_section = self.enricher.extract_root_section(chunk)
        self.assertEqual(root_section, "compound")

    def test_extract_root_section_from_section(self):
        """Test extracting root section from section field."""
        chunk = VectorDocument(
            chunk_id="test_chunk",
            document_id=self.doc_id,
            section="root",
            text="test text",
            embedding=[0.1] * 384,
            metadata_json={},
        )
        root_section = self.enricher.extract_root_section(chunk)
        self.assertEqual(root_section, "root")

    def test_extract_root_section_fallback(self):
        """Test fallback to root when no section info available."""
        chunk = VectorDocument(
            chunk_id="test_chunk",
            document_id=self.doc_id,
            section="",
            text="test text",
            embedding=[0.1] * 384,
            metadata_json={},
        )
        root_section = self.enricher.extract_root_section(chunk)
        self.assertEqual(root_section, "root")

    def test_get_parent_identity(self):
        """Test getting parent identity from chunk."""
        chunk = VectorDocument(
            chunk_id="test_chunk",
            document_id=self.doc_id,
            section="names",
            text="test text",
            embedding=[0.1] * 384,
            metadata_json={"root_section": "compound"},
        )
        parent_identity = self.enricher.get_parent_identity(chunk)
        self.assertEqual(parent_identity.document_id, self.doc_id)
        self.assertEqual(parent_identity.root_section, "compound")

    def test_reconstruct_parent_context_success(self):
        """Test reconstructing parent context from chunks."""
        # Create mock chunks
        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=f"section_{i}",
                text=f"Text content for chunk {i}",
                embedding=[float(i)] * 384,
                metadata_json={"key": f"value_{i}"},
                source_url=f"source_{i}",
            )
            for i in range(3)
        ]

        self.mock_db.get_documents.return_value = chunks

        parent_identity = ParentIdentity(
            document_id=self.doc_id, root_section="root"
        )
        context = self.enricher.reconstruct_parent_context(parent_identity)

        self.assertIsNotNone(context)
        self.assertEqual(context["num_chunks"], 3)
        self.assertEqual(len(context["text_parts"]), 3)
        self.assertIn("document_id", context["parent_identity"])

    def test_reconstruct_parent_context_exclude_sections(self):
        """Test excluding sections from parent context."""
        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=f"section_{i}",
                text=f"Text {i}",
                embedding=[float(i)] * 384,
                metadata_json={},
            )
            for i in range(3)
        ]

        self.mock_db.get_documents.return_value = chunks

        parent_identity = ParentIdentity(
            document_id=self.doc_id, root_section="root"
        )
        context = self.enricher.reconstruct_parent_context(
            parent_identity, exclude_sections={"section_0", "section_1"}
        )

        self.assertIsNotNone(context)
        self.assertEqual(context["num_chunks"], 1)

    def test_reconstruct_parent_context_no_chunks(self):
        """Test handling when no chunks found."""
        self.mock_db.get_documents.return_value = []

        parent_identity = ParentIdentity(
            document_id=self.doc_id, root_section="root"
        )
        context = self.enricher.reconstruct_parent_context(parent_identity)

        self.assertIsNone(context)

    def test_enrich_chunk_with_parent(self):
        """Test enriching a chunk with parent context."""
        chunk = VectorDocument(
            chunk_id="test_chunk",
            document_id=self.doc_id,
            section="names",
            text="Chunk text content",
            embedding=[0.1] * 384,
            metadata_json={"key": "value"},
            source_url="test_source",
        )

        parent_context = {
            "parent_identity": {"document_id": str(self.doc_id), "root_section": "root"},
            "num_chunks": 5,
            "sections_included": ["root", "names", "codes"],
            "text_parts": [
                {"section": "root", "text": "Root text", "source_url": "src"}
            ],
        }

        enriched = self.enricher.enrich_chunk_with_parent(
            chunk, parent_context=parent_context, include_parent_text=True
        )

        self.assertIn("chunk", enriched)
        self.assertIn("parent_context", enriched)
        self.assertEqual(enriched["chunk"]["chunk_id"], "test_chunk")
        self.assertEqual(enriched["parent_context"]["num_chunks"], 5)
        self.assertIn("parent_text_summary", enriched)

    def test_enrich_search_results(self):
        """Test enriching multiple search results with parent context."""
        from app.models import DBQueryResult

        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=f"section_{i}",
                text=f"Text {i}",
                embedding=[float(i)] * 384,
                metadata_json={},
            )
            for i in range(2)
        ]

        results = [
            DBQueryResult(document=chunks[0], score=0.95),
            DBQueryResult(document=chunks[1], score=0.87),
        ]

        # Mock parent context reconstruction
        parent_context = {
            "parent_identity": {
                "document_id": str(self.doc_id),
                "root_section": "root",
            },
            "num_chunks": 2,
            "sections_included": ["section_0", "section_1"],
            "text_parts": [
                {"section": "section_0", "text": "Text 0", "source_url": "src"}
            ],
        }

        self.enricher.reconstruct_parent_context = Mock(
            return_value=parent_context
        )

        enriched_results = self.enricher.enrich_search_results(
            results, include_parent_text=True
        )

        self.assertEqual(len(enriched_results), 2)
        for enriched in enriched_results:
            self.assertIn("chunk", enriched)
            self.assertIn("score", enriched)
            if enriched["parent_context"]:
                self.assertIn("parent_context", enriched)

    def test_get_all_parents_in_document(self):
        """Test getting all parent identities in a document."""
        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=f"section_{i}",
                text=f"Text {i}",
                embedding=[float(i)] * 384,
                metadata_json={"root_section": "root" if i < 2 else "compound"},
            )
            for i in range(4)
        ]

        self.mock_db.get_documents.return_value = chunks

        parents = self.enricher.get_all_parents_in_document(self.doc_id)

        self.assertEqual(len(parents), 2)
        parent_sections = [p.root_section for p in parents]
        self.assertIn("root", parent_sections)
        self.assertIn("compound", parent_sections)

    def test_get_all_parents_empty_document(self):
        """Test getting parents for document with no chunks."""
        self.mock_db.get_documents.return_value = []

        parents = self.enricher.get_all_parents_in_document(self.doc_id)

        self.assertEqual(len(parents), 0)


class TestParentChildIntegration(unittest.TestCase):
    """Integration tests for parent-child retrieval."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db = MagicMock()
        self.enricher = ParentContextEnricher(self.mock_db)
        self.doc_id = uuid4()

    def test_full_workflow(self):
        """Test full workflow: get chunk -> extract parent -> enrich."""
        # Create a set of chunks representing a document
        chunks = [
            VectorDocument(
                chunk_id="root_overview",
                document_id=self.doc_id,
                section="root",
                text="Main substance overview and definition",
                embedding=[0.1] * 384,
                metadata_json={
                    "root_section": "root",
                    "canonical_name": "Acetylsalicylic Acid",
                },
                source_url="source_root",
            ),
            VectorDocument(
                chunk_id="names_aspirin",
                document_id=self.doc_id,
                section="names",
                text="Aspirin is a commonly used name",
                embedding=[0.2] * 384,
                metadata_json={"root_section": "root", "chunk_type": "name"},
                source_url="source_names",
            ),
            VectorDocument(
                chunk_id="codes_unii",
                document_id=self.doc_id,
                section="codes",
                text="UNII: R16CO5Y76E",
                embedding=[0.3] * 384,
                metadata_json={
                    "root_section": "root",
                    "code_system": "UNII",
                },
                source_url="source_codes",
            ),
        ]

        self.mock_db.get_documents.return_value = chunks

        # Get the "names" chunk
        query_chunk = chunks[1]

        # Get parent identity
        parent_identity = self.enricher.get_parent_identity(query_chunk)
        self.assertEqual(parent_identity.root_section, "root")

        # Reconstruct parent context
        parent_context = self.enricher.reconstruct_parent_context(
            parent_identity, exclude_sections={"names"}
        )
        self.assertIsNotNone(parent_context)
        self.assertEqual(parent_context["num_chunks"], 2)  # root + codes, not names

        # Enrich the chunk
        enriched = self.enricher.enrich_chunk_with_parent(
            query_chunk, parent_context=parent_context, include_parent_text=True
        )

        self.assertIn("chunk", enriched)
        self.assertIn("parent_context", enriched)
        self.assertEqual(enriched["parent_context"]["num_chunks"], 2)
        self.assertIn("parent_text_summary", enriched)


if __name__ == "__main__":
    unittest.main()
