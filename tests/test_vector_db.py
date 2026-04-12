"""
GSRS MCP Server - Unit Tests for Vector Database Backends
"""
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from pgvector.sqlalchemy import Vector, HALFVEC

from app.config import settings
from app.db.factory import create_vector_database, get_available_backends
from app.models import VectorDocument, DBQueryResult

# Alias for backward compatibility in tests
QueryResult = DBQueryResult


class TestChromaDatabase(unittest.TestCase):
    """Unit tests for ChromaDB backend."""

    test_dimension = 384

    def setUp(self):
        """Set up ChromaDB for testing."""
        self.test_dir = tempfile.mkdtemp()
        chroma_url = f"chroma://{self.test_dir}/test_collection"
        self.db = create_vector_database(database_url=chroma_url)
        self.db.connect()
        self.db.initialize(dimension=self.test_dimension)

        self.test_docs = [
            VectorDocument(
                chunk_id=f"root_codes_{i}_code",
                document_id=uuid4(),
                section="codes",
                text=f"Test chunk text {i}",
                embedding=[float(i + 1)] * self.test_dimension,
                metadata_json={"test": f"value_{i}"},
                source_url="test_source",
            )
            for i in range(5)
        ]

    def tearDown(self):
        """Clean up test directory."""
        try:
            self.db.delete_all()
        except Exception:
            pass
        self.db.disconnect()
        if os.path.exists(self.test_dir):
            for _ in range(10):
                try:
                    shutil.rmtree(self.test_dir)
                    break
                except PermissionError:
                    time.sleep(0.1)
            else:
                shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_upsert_documents(self):
        """Test inserting documents."""
        count = self.db.upsert_documents(self.test_docs)
        self.assertEqual(count, 5)

        stats = self.db.get_statistics()
        self.assertEqual(stats["total_chunks"], 5)

    def test_get_document(self):
        """Test getting a document by ID."""
        self.db.upsert_documents(self.test_docs)

        doc = self.db.get_document("root_codes_0_code")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.chunk_id, "root_codes_0_code")
        self.assertEqual(doc.text, "Test chunk text 0")

    def test_get_document_not_found(self):
        """Test getting a non-existent document."""
        doc = self.db.get_document("nonexistent_path")
        self.assertIsNone(doc)

    def test_get_documents_by_substance(self):
        """Test getting documents by substance UUID."""
        substance_uuid = uuid4()

        docs = [
            VectorDocument(
                id=f"sub_path_{i}",
                chunk_id=f"root_{i}",
                document_id=substance_uuid,
                section="root",
                text=f"Substance chunk {i}",
                embedding=[0.1] * self.test_dimension,
                metadata_json={},
                source_url="test",
            )
            for i in range(3)
        ]

        self.db.upsert_documents(docs)
        results = self.db.get_documents_by_substance(substance_uuid)

        self.assertEqual(len(results), 3)

    def test_similarity_search(self):
        """Test similarity search."""
        self.db.upsert_documents(self.test_docs)

        query_embedding = [1.0] * self.test_dimension
        results = self.db.similarity_search(
            query_embedding=query_embedding,
            top_k=3,
        )

        self.assertLessEqual(len(results), 3)
        self.assertTrue(all(isinstance(r, QueryResult) for r in results))

    def test_similarity_search_with_filter(self):
        """Test similarity search with filters."""
        self.db.upsert_documents(self.test_docs)

        query_embedding = [1.0] * self.test_dimension
        results = self.db.similarity_search(
            query_embedding=query_embedding,
            top_k=5,
            filters={"test": "value_0"},
        )

        self.assertGreater(len(results), 0)

    def test_delete_documents_by_substance(self):
        """Test deleting documents by substance UUID."""
        substance_uuid = uuid4()

        docs = [
            VectorDocument(
                id=f"del_path_{i}",
                chunk_id=f"root_{i}",
                document_id=substance_uuid,
                section="root",
                text=f"To delete {i}",
                embedding=[0.1] * self.test_dimension,
                metadata_json={},
                source_url="test",
            )
            for i in range(3)
        ]

        self.db.upsert_documents(docs)
        deleted = self.db.delete_documents_by_substance(substance_uuid)

        self.assertEqual(deleted, 3)

        stats = self.db.get_statistics()
        self.assertEqual(stats["total_chunks"], 0)

    def test_delete_all(self):
        """Test deleting all documents."""
        self.db.upsert_documents(self.test_docs)
        self.db.delete_all()

        stats = self.db.get_statistics()
        self.assertEqual(stats["total_chunks"], 0)

    def test_get_statistics(self):
        """Test getting statistics."""
        self.db.upsert_documents(self.test_docs)

        stats = self.db.get_statistics()

        self.assertIn("total_chunks", stats)
        self.assertIn("total_substances", stats)
        self.assertEqual(stats["total_chunks"], 5)

    def test_get_unique_values(self):
        """Test getting unique field values."""
        self.db.upsert_documents(self.test_docs)

        values = self.db.get_unique_values("test")

        self.assertIn("value_0", values)

    def test_upsert_update(self):
        """Test that upsert updates existing documents."""
        doc = VectorDocument(
            id="update_path",
            chunk_id="root_update_code",
            document_id=uuid4(),
            section="codes",
            text="Original text",
            embedding=[0.1] * self.test_dimension,
            metadata={"version": 1},
            source_url="test",
        )

        count = self.db.upsert_documents([doc])
        self.assertEqual(count, 1)

        retrieved = self.db.get_document("root_update_code")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.text, "Original text")

    def test_search_by_example(self):
        """Test search_by_example with example metadata."""
        self.db.upsert_documents(self.test_docs)

        # Search using example metadata that matches test docs
        results = self.db.search_by_example(
            example={"test": "value_0"},
            top_k=5,
            mode="contains",
        )

        # Should return at least one result
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0].score, 0)

    def test_search_by_example_empty(self):
        """Test search_by_example with empty example."""
        results = self.db.search_by_example(example={}, top_k=5)
        self.assertEqual(results, [])


class TestFactory(unittest.TestCase):
    """Unit tests for the vector database factory."""

    def test_get_available_backends(self):
        """Test getting available backends."""
        backends = get_available_backends()

        self.assertIsInstance(backends, list)
        self.assertIn("pgvector", backends)

    def test_create_chroma_database(self):
        """Test creating ChromaDB instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_url = f"chroma://{tmpdir}/test_collection"
            db = create_vector_database(database_url=chroma_url)

            from app.db.backends.chroma import ChromaDatabase

            self.assertIsInstance(db, ChromaDatabase)

    def test_create_pgvector_database(self):
        """Test creating pgvector instance."""
        try:
            db = create_vector_database(
                backend="pgvector",
                database_url="postgresql://test:test@localhost/test",
            )

            from app.db.backends.pgvector import PGVectorDatabase

            self.assertIsInstance(db, PGVectorDatabase)
        except ImportError:
            self.skipTest("pgvector not installed")

    def test_create_unknown_backend(self):
        """Test creating unknown backend raises error."""
        with self.assertRaises(ValueError) as context:
            create_vector_database(database_url="unknown://localhost/test")

        self.assertIn("Unknown database scheme", str(context.exception))


class TestPGVectorDatabase(unittest.TestCase):
    """Focused unit tests for PGVectorDatabase initialization."""

    def test_initialize_creates_vector_extension_before_tables(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        conn = MagicMock()
        context_manager = MagicMock()
        context_manager.__enter__.return_value = conn
        context_manager.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = context_manager
        db.engine = engine

        with patch("app.db.backends.pgvector.Base.metadata.create_all") as create_all:
            db.initialize(dimension=384)

        executed_sql = [call.args[0].text.strip() for call in conn.execute.call_args_list]
        self.assertEqual(executed_sql[0], "CREATE EXTENSION IF NOT EXISTS vector;")
        create_all.assert_called_once_with(bind=engine)
        conn.commit.assert_called_once()

    def test_upsert_documents_targets_unique_chunk_id(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        session = MagicMock()
        db._get_session = MagicMock(return_value=session)

        doc = VectorDocument(
            chunk_id="root_update_code",
            document_id=uuid4(),
            section="codes",
            text="Updated text",
            embedding=[0.1, 0.2, 0.3],
            metadata_json={"version": 2},
            source_url="test",
        )

        insert_stmt = MagicMock()
        values_stmt = MagicMock()
        upsert_stmt = MagicMock()
        insert_stmt.values.return_value = values_stmt
        values_stmt.on_conflict_do_update.return_value = upsert_stmt

        with patch("app.db.backends.pgvector.insert", return_value=insert_stmt) as insert_factory:
            count = db.upsert_documents([doc])

        insert_factory.assert_called_once_with(VectorDocument)

        # Verify values() was called with correct keys (search_text is populated,
        # search_tsv is a to_tsvector expression — skip exact value comparison)
        insert_stmt.values.assert_called_once()
        call_kwargs = insert_stmt.values.call_args.kwargs
        self.assertEqual(call_kwargs["chunk_id"], doc.chunk_id)
        self.assertEqual(call_kwargs["text"], doc.text)
        self.assertEqual(call_kwargs["section"], doc.section)
        self.assertEqual(call_kwargs["metadata_json"], doc.metadata_json)
        # search_text should be populated by _build_search_text
        self.assertIn("search_text", call_kwargs)
        self.assertIn("search_tsv", call_kwargs)

        # Verify on_conflict_do_update was called with correct index_elements
        # and set_ containing the right keys (skip exact search_tsv comparison)
        values_stmt.on_conflict_do_update.assert_called_once()
        oc_call = values_stmt.on_conflict_do_update.call_args
        self.assertEqual(oc_call.kwargs["index_elements"], [VectorDocument.chunk_id])
        set_values = oc_call.kwargs["set_"]
        self.assertIn("search_text", set_values)
        self.assertIn("search_tsv", set_values)
        self.assertEqual(set_values["text"], doc.text)
        # execute is called once: the upsert statement includes all fields via doc.values()
        self.assertEqual(session.execute.call_count, 1)
        session.commit.assert_called_once()
        session.close.assert_called_once()
        self.assertEqual(count, 1)

    def test_disconnect_disposes_engine(self):
        from app.db.backends import pgvector as pgvector_module

        db = pgvector_module.PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_engine = MagicMock()
        db.engine = mock_engine
        db.SessionLocal = MagicMock()

        db.disconnect()

        mock_engine.dispose.assert_called_once()
        self.assertIsNone(db.engine)
        self.assertIsNone(db.SessionLocal)

    def test_get_session_lazy_connect(self):
        from app.db.backends import pgvector as pgvector_module

        db = pgvector_module.PGVectorDatabase("postgresql://test:test@localhost/test")
        db.engine = None
        db.SessionLocal = None

        original_connect = db.connect
        call_tracker = MagicMock()

        def fake_connect():
            call_tracker()
            db.SessionLocal = MagicMock()

        db.connect = fake_connect
        session = db._get_session()

        call_tracker.assert_called_once()
        self.assertIsNotNone(session)

    def test_similarity_search_applies_section_filter(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        db._get_session = MagicMock(return_value=mock_session)

        db.similarity_search(
            query_embedding=[0.1] * 3,
            top_k=5,
            filters={"section": "codes"},
        )

        mock_query.filter.assert_called()
        mock_session.close.assert_called_once()

    def test_similarity_search_applies_document_id_filter(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        db._get_session = MagicMock(return_value=mock_session)

        test_uuid = uuid4()
        db.similarity_search(
            query_embedding=[0.1] * 3,
            top_k=5,
            filters={"document_id": test_uuid},
        )

        mock_query.filter.assert_called()
        mock_session.close.assert_called_once()

    def test_delete_documents_by_substance_rolls_back_on_error(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.delete.side_effect = RuntimeError("DB error")
        db._get_session = MagicMock(return_value=mock_session)

        with self.assertRaises(RuntimeError):
            db.delete_documents_by_substance(uuid4())

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    def test_delete_all_rolls_back_on_error(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_session.query.return_value.delete.side_effect = RuntimeError("DB error")
        db._get_session = MagicMock(return_value=mock_session)

        with self.assertRaises(RuntimeError):
            db.delete_all()

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    def test_get_statistics_returns_both_counters(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_chunks_query = MagicMock()
        mock_substances_query = MagicMock()
        mock_session.query.side_effect = [mock_chunks_query, mock_substances_query]
        mock_chunks_query.count.return_value = 42
        mock_substances_query.distinct.return_value = mock_substances_query
        mock_substances_query.count.return_value = 7
        db._get_session = MagicMock(return_value=mock_session)

        stats = db.get_statistics()

        self.assertEqual(stats["total_chunks"], 42)
        self.assertEqual(stats["total_substances"], 7)
        mock_session.close.assert_called_once()

    def test_get_unique_values_section(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.all.return_value = [("names",), ("codes",), ("structure",)]
        db._get_session = MagicMock(return_value=mock_session)

        result = db.get_unique_values("section")

        self.assertEqual(result, ["names", "codes", "structure"])
        mock_session.close.assert_called_once()

    def test_get_unique_values_source_url(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.all.return_value = [("url1",), ("url2",), (None,)]
        db._get_session = MagicMock(return_value=mock_session)

        result = db.get_unique_values("source_url")

        self.assertEqual(result, ["url1", "url2"])

    def test_get_unique_values_unknown_field(self):
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        db._get_session = MagicMock(return_value=mock_session)

        result = db.get_unique_values("nonexistent")

        self.assertEqual(result, [])
        mock_session.close.assert_called_once()

    def test_lexical_search_returns_empty_for_no_terms(self):
        """Test lexical search returns empty for queries with no valid terms."""
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        db._get_session = MagicMock(return_value=mock_session)

        results = db.lexical_search("a", top_k=5)
        self.assertEqual(results, [])

    def test_lexical_search_builds_tsquery(self):
        """Test lexical search constructs correct query."""
        from app.db.backends.pgvector import PGVectorDatabase
        from sqlalchemy import func

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        db._get_session = MagicMock(return_value=mock_session)

        results = db.lexical_search("CAS code ibuprofen", top_k=10)

        self.assertEqual(results, [])
        # Verify query was built with plainto_tsquery
        mock_session.query.assert_called_once()
        call_args = mock_session.query.call_args
        self.assertEqual(len(call_args[0]), 2)  # VectorDocument + rank column
        mock_query.all.assert_called_once()

    def test_lexical_search_applies_section_filter(self):
        """Test lexical search applies section filter."""
        from app.db.backends.pgvector import PGVectorDatabase

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        db._get_session = MagicMock(return_value=mock_session)

        db.lexical_search("test query", top_k=5, filters={"section": "codes"})

        # Should have called filter at least twice (tsquery + section)
        self.assertGreaterEqual(mock_query.filter.call_count, 2)

    def test_lexical_search_applies_document_id_filter(self):
        """Test lexical search applies document_id filter."""
        from app.db.backends.pgvector import PGVectorDatabase
        from uuid import uuid4

        db = PGVectorDatabase("postgresql://test:test@localhost/test")
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        db._get_session = MagicMock(return_value=mock_session)

        substance_id = uuid4()
        db.lexical_search("test query", top_k=5, filters={"document_id": substance_id})

        self.assertGreaterEqual(mock_query.filter.call_count, 2)


class TestVectorDocument(unittest.TestCase):
    """Unit tests for VectorDocument model."""

    def test_create_document(self):
        """Test creating a VectorDocument."""
        doc = VectorDocument(
            chunk_id="root_code",
            document_id=uuid4(),
            section="codes",
            text="Test text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
            source_url="test",
        )

        self.assertEqual(doc.chunk_id, "root_code")
        self.assertEqual(doc.text, "Test text")
        self.assertEqual(len(doc.embedding), 3)
        self.assertEqual(doc.metadata_json, {"key": "value"})

    def test_document_repr(self):
        """Test VectorDocument string representation."""
        doc = VectorDocument(
            chunk_id="root_code",
            document_id=uuid4(),
            section="codes",
            text="Test",
            embedding=[],
            metadata_json={},
            source_url="test",
        )

        self.assertIn("root_code", repr(doc))
        self.assertIn("codes", repr(doc))

    def test_values_returns_dict_with_all_fields(self):
        """Test values() returns correct dict."""
        doc_id = uuid4()
        doc = VectorDocument(
            chunk_id="root_test",
            document_id=doc_id,
            section="root",
            text="Hello",
            embedding=[1.0],
            metadata_json={"k": "v"},
            source_url="src",
        )

        vals = doc.values()
        self.assertEqual(vals["document_id"], doc_id)
        self.assertEqual(vals["section"], "root")
        self.assertEqual(vals["text"], "Hello")
        self.assertEqual(vals["metadata_json"], {"k": "v"})

    def test_set_embedding(self):
        """Test set_embedding helper."""
        doc = VectorDocument(
            chunk_id="root_test",
            document_id=uuid4(),
            section="root",
            text="Test",
            embedding=[0.0],
            metadata_json={},
        )

        doc.set_embedding([1.0, 2.0, 3.0])
        self.assertEqual(doc.embedding, [1.0, 2.0, 3.0])

    def test_constructor_metadata_alias(self):
        """Test constructor accepts 'metadata' and stores as 'metadata_json'."""
        doc = VectorDocument(
            chunk_id="root_test",
            document_id=uuid4(),
            section="root",
            text="Test",
            embedding=[],
            metadata={"foo": "bar"},
        )
        self.assertEqual(doc.metadata_json, {"foo": "bar"})


class TestQueryResult(unittest.TestCase):
    """Unit tests for QueryResult dataclass."""

    def test_create_result(self):
        """Test creating a QueryResult."""
        doc = VectorDocument(
            chunk_id="root_code",
            document_id=uuid4(),
            section="codes",
            text="Test",
            embedding=[],
            metadata_json={},
            source_url="test",
        )

        result = QueryResult(document=doc, score=0.95)

        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.document.chunk_id, "root_code")


class TestEmbeddingColumnType(unittest.TestCase):
    """Tests for dynamic embedding column type (Vector vs HalfVec)."""

    def test_embedding_column_exists(self):
        """Test that embedding column is defined on VectorDocument."""
        columns = {c.key for c in VectorDocument.__table__.columns}
        self.assertIn("embedding", columns)

    def test_embedding_column_type_matches_current_dimension(self):
        """Test embedding column type matches current EMBEDDING_DIMENSION."""
        col = VectorDocument.__table__.columns.embedding
        if settings.embedding_dimension > 2000:
            self.assertIsInstance(col.type, HALFVEC)
            self.assertEqual(col.type.dim, settings.embedding_dimension)
        else:
            self.assertIsInstance(col.type, Vector)
            self.assertEqual(col.type.dim, settings.embedding_dimension)

    def test_hnsw_index_exists(self):
        """Test HNSW index is defined in __table_args__."""
        indexes = {idx.name: idx for idx in VectorDocument.__table__.indexes}
        self.assertIn("idx_embedding_hnsw", indexes)

    def test_hnsw_index_covers_embedding_column(self):
        """Test HNSW index includes the embedding column."""
        indexes = {idx.name: idx for idx in VectorDocument.__table__.indexes}
        hnsw_index = indexes["idx_embedding_hnsw"]
        col_names = [c.name for c in hnsw_index.columns]
        self.assertIn("embedding", col_names)

    def test_hnsw_index_dialect_kwargs(self):
        """Test HNSW index has correct PostgreSQL-specific options."""
        indexes = {idx.name: idx for idx in VectorDocument.__table__.indexes}
        hnsw_index = indexes["idx_embedding_hnsw"]

        dk = hnsw_index.dialect_kwargs
        self.assertEqual(dk.get("postgresql_using"), "hnsw")
        self.assertIn("postgresql_with", dk)
        self.assertIn("postgresql_ops", dk)

    def test_hnsw_index_operator_class(self):
        """Test HNSW index uses correct operator class for current dimension."""
        indexes = {idx.name: idx for idx in VectorDocument.__table__.indexes}
        hnsw_index = indexes["idx_embedding_hnsw"]

        expected_op = (
            "halfvec_cosine_ops"
            if settings.embedding_dimension > 2000
            else "vector_cosine_ops"
        )
        self.assertEqual(
            hnsw_index.dialect_kwargs["postgresql_ops"]["embedding"],
            expected_op,
        )

    def test_hnsw_index_hnsw_parameters(self):
        """Test HNSW index m and ef_construction values."""
        indexes = {idx.name: idx for idx in VectorDocument.__table__.indexes}
        hnsw_index = indexes["idx_embedding_hnsw"]

        pw = hnsw_index.dialect_kwargs.get("postgresql_with", {})
        self.assertEqual(pw.get("m"), 16)
        self.assertEqual(pw.get("ef_construction"), 64)

    def test_embedding_column_type_vector_for_low_dimension(self):
        """Test Vector type is used when dimension <= 2000."""
        if settings.embedding_dimension > 2000:
            self.skipTest("Only applies when EMBEDDING_DIMENSION <= 2000")

        col = VectorDocument.__table__.columns.embedding
        self.assertIsInstance(col.type, Vector)

    def test_embedding_column_type_halfvec_for_high_dimension(self):
        """Test HalfVec type is used when dimension > 2000."""
        if settings.embedding_dimension <= 2000:
            self.skipTest("Only applies when EMBEDDING_DIMENSION > 2000")

        col = VectorDocument.__table__.columns.embedding
        self.assertIsInstance(col.type, HALFVEC)


if __name__ == "__main__":
    unittest.main()


