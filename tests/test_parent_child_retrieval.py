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
        """Test fallback to overview when no section info available."""
        chunk = VectorDocument(
            chunk_id="test_chunk",
            document_id=self.doc_id,
            section="",
            text="test text",
            embedding=[0.1] * 384,
            metadata_json={},
        )
        root_section = self.enricher.extract_root_section(chunk)
        self.assertEqual(root_section, "overview")

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
        parent_identity, fallback = self.enricher.get_parent_identity(chunk)
        self.assertEqual(parent_identity.document_id, self.doc_id)
        self.assertEqual(parent_identity.root_section, "compound")
        self.assertFalse(fallback)

    def test_get_parent_identity_fallback_when_root_section_missing(self):
        """Missing root_section falls back to the chunk's section."""
        chunk = VectorDocument(
            chunk_id="legacy_chunk",
            document_id=self.doc_id,
            section="identifiers",
            text="legacy code chunk",
            embedding=[0.1] * 384,
            metadata_json={},
        )
        parent_identity, fallback = self.enricher.get_parent_identity(chunk)
        self.assertEqual(parent_identity.root_section, "identifiers")
        self.assertTrue(fallback)

    def test_get_parent_identity_fallback_when_empty_metadata(self):
        """Empty metadata falls back to overview."""
        chunk = VectorDocument(
            chunk_id="legacy_chunk",
            document_id=self.doc_id,
            section="",
            text="legacy chunk",
            embedding=[0.1] * 384,
            metadata_json={},
        )
        parent_identity, fallback = self.enricher.get_parent_identity(chunk)
        self.assertEqual(parent_identity.root_section, "overview")
        self.assertTrue(fallback)

    def test_reconstruct_parent_context_success(self):
        """Test reconstructing parent context from chunks."""
        # Create mock chunks that all belong to the "root" parent
        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=f"section_{i}",
                text=f"Text content for chunk {i}",
                embedding=[float(i)] * 384,
                metadata_json={"root_section": "root", "key": f"value_{i}"},
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

    def test_reconstruct_parent_context_sections_included_is_deterministic(self):
        """sections_included must preserve first-encounter order, not a set."""
        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=section,
                text=f"Text {i}",
                embedding=[float(i)] * 384,
                metadata_json={"root_section": "root"},
            )
            for i, section in enumerate(["codes", "names", "codes", "root", "names"])
        ]
        self.mock_db.get_documents.return_value = chunks

        parent_identity = ParentIdentity(
            document_id=self.doc_id, root_section="root"
        )
        context = self.enricher.reconstruct_parent_context(parent_identity)

        self.assertIsNotNone(context)
        self.assertEqual(
            context["sections_included"],
            ["codes", "names", "root"],
        )

    def test_reconstruct_parent_context_exclude_sections(self):
        """Test excluding sections from parent context."""
        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=f"section_{i}",
                text=f"Text {i}",
                embedding=[float(i)] * 384,
                metadata_json={"root_section": "root"},
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

        # New backends expose get_root_sections; use it when available.
        self.mock_db.get_root_sections.return_value = ["compound", "root"]

        parents = self.enricher.get_all_parents_in_document(self.doc_id)

        self.assertEqual(len(parents), 2)
        parent_sections = [p.root_section for p in parents]
        self.assertIn("root", parent_sections)
        self.assertIn("compound", parent_sections)

    def test_get_all_parents_empty_document(self):
        """Test getting parents for document with no chunks."""
        self.mock_db.get_root_sections.return_value = []

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
                    "display_name": "Acetylsalicylic Acid",
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
        parent_identity, _ = self.enricher.get_parent_identity(query_chunk)
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


class TestParentEnricherMetricsAndTruncation(unittest.TestCase):
    """Tests for enricher-level metrics and parent-text truncation flag."""

    def setUp(self):
        """Set up test fixtures with a real InMemoryMetrics instance."""
        from app.observability import InMemoryMetrics

        self.mock_db = MagicMock()
        self.metrics = InMemoryMetrics()
        self.enricher = ParentContextEnricher(self.mock_db, metrics=self.metrics)
        self.doc_id = uuid4()

    def test_reconstruct_increments_count_and_latency(self):
        """Each reconstruct call must increment count and observe latency."""
        self.mock_db.get_documents.return_value = [
            VectorDocument(
                chunk_id="c1",
                document_id=self.doc_id,
                section="names",
                text="name data",
                embedding=[0.1] * 384,
                metadata_json={"root_section": "root"},
            ),
        ]
        parent_identity = ParentIdentity(
            document_id=self.doc_id, root_section="root"
        )
        self.enricher.reconstruct_parent_context(parent_identity)

        snap = self.metrics.snapshot()
        self.assertEqual(snap["counters"].get("parent_rebuild.count"), 1)
        self.assertGreaterEqual(
            snap["latencies"]
            .get("parent_rebuild.latency_ms", {})
            .get("count", 0),
            1,
        )

    def test_enrich_search_results_dedups_shared_parents(self):
        """Shared parent identities are reconstructed only once per call."""
        from app.models import DBQueryResult

        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=f"section_{i}",
                text=f"Text {i}",
                embedding=[float(i)] * 384,
                metadata_json={"root_section": "root"},
            )
            for i in range(3)
        ]
        # Two results share a parent, the third is unique
        results = [
            DBQueryResult(document=chunks[0], score=0.9),
            DBQueryResult(document=chunks[1], score=0.8),
        ]

        parent_context = {
            "parent_identity": {
                "document_id": str(self.doc_id),
                "root_section": "root",
            },
            "num_chunks": 1,
            "sections_included": ["section_0"],
            "text_parts": [
                {"section": "section_0", "text": "Text 0", "source_url": "src"}
            ],
        }
        self.enricher.reconstruct_parent_context = Mock(
            return_value=parent_context
        )

        self.enricher.enrich_search_results(results)

        # Same parent identity (doc, root) appears twice -> only 1 rebuild.
        self.assertEqual(
            self.enricher.reconstruct_parent_context.call_count, 1
        )

    def test_truncation_flag_set_when_parent_text_exceeds_limit(self):
        """Aggregated parent text longer than the limit must set the flag."""
        chunk = VectorDocument(
            chunk_id="query_chunk",
            document_id=self.doc_id,
            section="query",
            text="query text",
            embedding=[0.1] * 384,
            metadata_json={"root_section": "root"},
        )
        # Build a parent context with text parts that exceed a small limit
        long_text = "x" * 600
        parent_context = {
            "parent_identity": {
                "document_id": str(self.doc_id),
                "root_section": "root",
            },
            "num_chunks": 2,
            "sections_included": ["section_0", "section_1"],
            "text_parts": [
                {"section": "section_0", "text": long_text, "source_url": "src"},
                {"section": "section_1", "text": long_text, "source_url": "src"},
            ],
        }

        enriched = self.enricher.enrich_chunk_with_parent(
            chunk,
            parent_context=parent_context,
            include_parent_text=True,
            parent_text_limit=200,
        )

        self.assertTrue(enriched.get("parent_text_truncated"))
        self.assertEqual(enriched.get("parent_text_truncated_chars"), 200)
        self.assertFalse(enriched.get("parent_grouping_fallback_used", False))
        snap = self.metrics.snapshot()
        self.assertGreaterEqual(
            snap["counters"].get("parent_text.truncated", 0), 1
        )

    def test_truncation_flag_false_when_under_limit(self):
        """Aggregated text under the limit sets parent_text_truncated=False."""
        chunk = VectorDocument(
            chunk_id="query_chunk",
            document_id=self.doc_id,
            section="query",
            text="query text",
            embedding=[0.1] * 384,
            metadata_json={"root_section": "root"},
        )
        parent_context = {
            "parent_identity": {
                "document_id": str(self.doc_id),
                "root_section": "root",
            },
            "num_chunks": 1,
            "sections_included": ["section_0"],
            "text_parts": [
                {"section": "section_0", "text": "short text", "source_url": "src"}
            ],
        }
        enriched = self.enricher.enrich_chunk_with_parent(
            chunk, parent_context=parent_context, include_parent_text=True
        )
        self.assertIn("parent_text_summary", enriched)
        self.assertFalse(enriched.get("parent_text_truncated", True))
        self.assertFalse(enriched.get("parent_grouping_fallback_used", False))

    def test_fallback_flag_set_for_legacy_chunk(self):
        """Legacy chunks without root_section flag the fallback when reconstructing."""
        chunk = VectorDocument(
            chunk_id="legacy_chunk",
            document_id=self.doc_id,
            section="identifiers",
            text="legacy code chunk",
            embedding=[0.1] * 384,
            metadata_json={},
        )
        sibling = VectorDocument(
            chunk_id="sibling_chunk",
            document_id=self.doc_id,
            section="identifiers",
            text="sibling chunk",
            embedding=[0.2] * 384,
            metadata_json={},
        )
        self.mock_db.get_documents.return_value = [chunk, sibling]

        enriched = self.enricher.enrich_chunk_with_parent(
            chunk, parent_context=None, include_parent_text=True
        )

        self.assertTrue(enriched.get("parent_grouping_fallback_used"))
        self.assertTrue(enriched["parent_context"].get("parent_grouping_fallback_used"))


class TestBackendSectionFilterAndExclusion(unittest.TestCase):
    """Tests that the enricher pushes the section filter to the backend
    and excludes the child by chunk_id, not by section name.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db = MagicMock()
        self.enricher = ParentContextEnricher(self.mock_db)
        self.doc_id = uuid4()

    def test_reconstruct_passes_sections_and_limit_to_backend(self):
        """The backend must receive a sections list and a limit."""
        from app.services.chunker import sections_in_root

        chunks = [
            VectorDocument(
                chunk_id=f"chunk_{i}",
                document_id=self.doc_id,
                section=("names" if i == 0 else "codes"),
                text=f"text {i}",
                embedding=[float(i)] * 384,
                metadata_json={"root_section": "names" if i == 0 else "codes"},
            )
            for i in range(2)
        ]
        self.mock_db.get_documents.return_value = chunks

        parent_identity = ParentIdentity(
            document_id=self.doc_id, root_section="names"
        )
        self.enricher.reconstruct_parent_context(parent_identity)

        # Inspect the call arguments: root_sections= must include the root
        # section name, limit= must be a positive int.
        call = self.mock_db.get_documents.call_args
        self.assertIsNotNone(call, "Expected get_documents to be called")
        kwargs = call.kwargs
        self.assertEqual(kwargs.get("substance_uuid"), self.doc_id)
        self.assertIn("names", kwargs.get("root_sections") or [])
        self.assertIsInstance(kwargs.get("limit"), int)
        self.assertGreater(kwargs["limit"], 0)

    def test_reconstruct_falls_back_when_backend_rejects_kwargs(self):
        """A backend that doesn't accept sections/limit must still work."""
        legacy_db = MagicMock()
        legacy_db.get_documents.side_effect = TypeError(
            "get_documents() got an unexpected keyword argument 'sections'"
        )
        # Second call (fallback) returns chunks
        legacy_db.get_documents.side_effect = [
            TypeError("nope"),
            [
                VectorDocument(
                    chunk_id="c1",
                    document_id=self.doc_id,
                    section="names",
                    text="data",
                    embedding=[0.1] * 384,
                    metadata_json={"root_section": "names"},
                )
            ],
        ]
        enricher = ParentContextEnricher(legacy_db)
        parent_identity = ParentIdentity(
            document_id=self.doc_id, root_section="names"
        )
        context = enricher.reconstruct_parent_context(parent_identity)
        self.assertIsNotNone(context)
        self.assertEqual(context["num_chunks"], 1)

    def test_child_excluded_by_chunk_id_not_section(self):
        """Siblings in the same section as the child must still appear."""
        # Two siblings in the "names" section, plus the child chunk.
        siblings = [
            VectorDocument(
                chunk_id="names_batch",
                document_id=self.doc_id,
                section="names",
                text="names batch",
                embedding=[0.1] * 384,
                metadata_json={"root_section": "names"},
            ),
            VectorDocument(
                chunk_id="names_atomic_0",
                document_id=self.doc_id,
                section="names",
                text="atomic 0",
                embedding=[0.2] * 384,
                metadata_json={"root_section": "names"},
            ),
        ]
        child = VectorDocument(
            chunk_id="names_atomic_1",
            document_id=self.doc_id,
            section="names",
            text="atomic 1",
            embedding=[0.3] * 384,
            metadata_json={"root_section": "names"},
        )
        self.mock_db.get_documents.return_value = siblings + [child]

        # Direct call with chunk_id exclusion: only the child is dropped.
        parent_identity = ParentIdentity(
            document_id=self.doc_id, root_section="names"
        )
        ctx = self.enricher.reconstruct_parent_context(
            parent_identity, exclude_chunk_ids={"names_atomic_1"}
        )
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["num_chunks"], 2)

        # Legacy section exclusion would drop ALL names chunks.
        ctx_legacy = self.enricher.reconstruct_parent_context(
            parent_identity, exclude_sections={"names"}
        )
        self.assertIsNone(ctx_legacy)

    def test_enrich_search_results_uses_cached_parent_and_excludes_child(self):
        """Within a batch, multiple children share a parent via cache."""
        from app.models import DBQueryResult

        # Mock reconstruct_parent_context to return the same context every
        # call so the cache hit/miss behaviour can be observed.
        sample_parent = {
            "parent_identity": {
                "document_id": str(self.doc_id),
                "root_section": "names",
            },
            "num_chunks": 2,
            "sections_included": ["names"],
            "text_parts": [
                {
                    "chunk_id": "names_batch",
                    "section": "names",
                    "text": "names batch",
                    "source_url": "src",
                },
                {
                    "chunk_id": "names_atomic_0",
                    "section": "names",
                    "text": "atomic 0",
                    "source_url": "src",
                },
            ],
        }
        self.enricher.reconstruct_parent_context = Mock(
            return_value=sample_parent
        )

        # Two child chunks: only the second should be excluded by chunk_id.
        child_0 = VectorDocument(
            chunk_id="names_batch",
            document_id=self.doc_id,
            section="names",
            text="names batch",
            embedding=[0.1] * 384,
            metadata_json={"root_section": "names"},
        )
        child_1 = VectorDocument(
            chunk_id="names_atomic_0",
            document_id=self.doc_id,
            section="names",
            text="atomic 0",
            embedding=[0.2] * 384,
            metadata_json={"root_section": "names"},
        )
        results = [
            DBQueryResult(document=child_0, score=0.9),
            DBQueryResult(document=child_1, score=0.8),
        ]
        enriched = self.enricher.enrich_search_results(results)

        # Both children share the same parent identity, so 1 miss + 1 hit.
        self.assertEqual(
            self.enricher.reconstruct_parent_context.call_count, 1
        )
        # First child: its own chunk_id is removed from the cached parent,
        # so the remaining part is "atomic 0".
        first_summary = enriched[0].get("parent_text_summary", "")
        self.assertIn("atomic 0", first_summary)
        self.assertNotIn("names batch", first_summary)
        # Second child: its own chunk_id is removed; "names batch" remains.
        second_summary = enriched[1].get("parent_text_summary", "")
        self.assertIn("names batch", second_summary)
        self.assertNotIn("atomic 0", second_summary)


if __name__ == "__main__":
    unittest.main()
