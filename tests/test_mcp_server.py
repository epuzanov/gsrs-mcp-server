"""
GSRS MCP Server - Unit Tests for MCP Server
"""
import unittest
from uuid import uuid4

from app.config import settings
from app.models.api import (
    AskResponse,
    Citation,
    QueryResponse,
    QueryResult,
    SimilarSubstanceResponse,
    SimilarSubstanceResult,
)
from app.models.db import VectorDocument


class TestMCPConfig(unittest.TestCase):
    """Tests for MCP configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        self.assertEqual(settings.api_username, "admin")
        self.assertEqual(settings.api_password, "admin123")
        self.assertIsInstance(settings.default_top_k, int)
        self.assertGreater(settings.default_top_k, 0)


class TestMCPTools(unittest.TestCase):
    """Tests for MCP tool response formatters."""

    def test_format_ask_response_with_answer(self):
        """Test formatting a successful /ask response."""
        doc = VectorDocument(
            chunk_id="c1", document_id=uuid4(), section="codes",
            text="CAS: 50-78-2", embedding=[0.0]*10,
            metadata_json={"canonical_name": "Aspirin"},
        )
        resp = AskResponse(
            query="CAS for aspirin", rewritten_queries=[], applied_filters={},
            answer="The CAS number is 50-78-2.", citations=[
                Citation(chunk_id="c1", document_id=str(doc.document_id), section="codes"),
            ],
            evidence_chunks=[QueryResult(chunk=doc, score=0.9)],
            confidence=0.85, abstained=False,
        )
        # Manual format check
        self.assertIn("50-78-2", resp.answer)
        self.assertEqual(resp.confidence, 0.85)

    def test_format_ask_response_abstained(self):
        """Test formatting an abstained response."""
        resp = AskResponse(
            query="unknown", rewritten_queries=[], applied_filters={},
            answer=None, citations=[], evidence_chunks=[],
            confidence=0.1, abstained=True, abstain_reason="No evidence found.",
        )
        self.assertTrue(resp.abstained)
        self.assertIn("No evidence", resp.abstain_reason)

    def test_format_similarity_response(self):
        """Test formatting similarity search response."""
        doc = VectorDocument(
            chunk_id="c1", document_id=uuid4(), section="codes",
            text="CAS: 50-78-2", embedding=[0.0]*10,
            metadata_json={"canonical_name": "Aspirin"},
        )
        results = [
            SimilarSubstanceResult(
                substance_uuid="uuid-1", canonical_name="Acetylsalicylic acid",
                match_score=0.95, matched_fields=["codes", "names"],
                chunks=[QueryResult(chunk=doc, score=0.9)],
            ),
        ]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].match_score, 0.95)
        self.assertIn("codes", results[0].matched_fields)

    def test_format_similarity_response_empty(self):
        """Test formatting empty similarity response."""
        resp = SimilarSubstanceResponse(
            query_substance_name="Unknown", results=[],
            total_substances=0, total_chunks=0,
        )
        self.assertEqual(resp.total_substances, 0)

    def test_format_query_response(self):
        """Test formatting /query response."""
        doc = VectorDocument(
            chunk_id="c1", document_id=uuid4(), section="codes",
            text="Aspirin CAS 50-78-2", embedding=[0.0]*10,
            metadata_json={"canonical_name": "Aspirin"},
        )
        resp = QueryResponse(
            query="aspirin",
            results=[QueryResult(chunk=doc, score=0.9)],
            total_results=1,
        )
        self.assertEqual(resp.total_results, 1)
        self.assertEqual(resp.results[0].similarity_score, 0.9)

    def test_format_query_response_empty(self):
        """Test formatting empty /query response."""
        resp = QueryResponse(query="nonexistent", results=[], total_results=0)
        self.assertEqual(resp.total_results, 0)


class TestMCPServerRegistration(unittest.TestCase):
    """Tests that MCP server tools are properly registered."""

    def test_tools_registered(self):
        """Test that all MCP tools are registered."""
        from app.main import mcp
        tools = set(mcp._tool_manager._tools.keys())
        expected = {"gsrs_ask", "gsrs_similarity_search", "gsrs_retrieve",
                     "gsrs_ingest", "gsrs_delete", "gsrs_health", "gsrs_statistics",
                     "gsrs_aggregation", "gsrs_query_optimizer", "gsrs_get_document",
                     "gsrs_api_search", "gsrs_api_structure_search", "gsrs_api_sequence_search"}
        missing = expected - tools
        self.assertFalse(missing, f"Missing tools: {missing}")

    def test_server_has_instructions(self):
        """Test that server has instructions."""
        from app.main import mcp
        self.assertIsNotNone(mcp.instructions)
        self.assertIn("GSRS", mcp.instructions)


if __name__ == "__main__":
    unittest.main()
