"""
GSRS MCP Server - Unit Tests for Query Pipeline Components
"""
import unittest
from uuid import uuid4

from app.services.query_rewrite import QueryRewriteService, RewriteResult
from app.services.metadata_filters import MetadataFilterBuilder
from app.services.lexical_retrieval import LexicalRetriever
from app.services.hybrid_retrieval import HybridRetriever
from app.services.reranking import RerankerService
from app.services.evidence import EvidenceExtractor, EvidenceResult
from app.services.abstention import AbstentionPolicy, AbstentionDecision
from app.services.answering import AnswerGenerator
from app.services.aggregation import AggregationService, AggregationResult
from app.models.db import VectorDocument
from app.models.api import Citation


class TestQueryRewriteService(unittest.TestCase):
    """Unit tests for QueryRewriteService."""

    def setUp(self):
        self.service = QueryRewriteService()

    def test_normalize_query(self):
        """Test query normalization."""
        result = self.service._normalize("  What  is the   CAS code?  ")
        self.assertEqual(result, "what is the cas code?")

    def test_detect_identifier_lookup(self):
        """Test identifier lookup intent detection."""
        result = self.service.rewrite("What is the CAS code for ibuprofen?")
        self.assertEqual(result.intent, "identifier_lookup")
        self.assertIn("CAS", result.rewrites[1] if len(result.rewrites) > 1 else result.rewrites[0])

    def test_detect_relationship_query(self):
        """Test relationship query intent detection."""
        result = self.service.rewrite("What are the metabolites of aspirin?")
        self.assertEqual(result.intent, "relationship_query")

    def test_detect_general_intent(self):
        """Test general intent detection."""
        result = self.service.rewrite("Tell me about aspirin")
        self.assertIn(result.intent, ["general", "section_query"])

    def test_rewrites_are_non_empty(self):
        """Test that rewrites are generated."""
        result = self.service.rewrite("CAS code for ibuprofen")
        self.assertTrue(len(result.rewrites) >= 1)

    def test_filters_inferred_for_codes(self):
        """Test that code mentions infer section filters."""
        result = self.service.rewrite("What is the UNII code?")
        self.assertIn("sections", result.filters)
        self.assertIn("codes", result.filters["sections"])

    def test_filters_inferred_for_names(self):
        """Test that name mentions infer section filters."""
        result = self.service.rewrite("What is aspirin called?")
        self.assertIn("sections", result.filters)
        self.assertIn("names", result.filters["sections"])

    def test_detect_aggregation_identifiers(self):
        """Test aggregation intent for identifier counting."""
        result = self.service.rewrite("How many identifiers has Ibuprofen?")
        self.assertEqual(result.intent, "aggregation_identifiers")

    def test_detect_aggregation_general(self):
        """Test general aggregation intent."""
        result = self.service.rewrite("List all information about aspirin")
        self.assertEqual(result.intent, "aggregation_general")

    def test_aggregation_rewrites(self):
        """Test that aggregation queries generate appropriate rewrites."""
        result = self.service.rewrite("How many codes does Ibuprofen have?")
        self.assertEqual(result.intent, "aggregation_identifiers")
        self.assertTrue(len(result.rewrites) >= 2)
        # Should include substance name
        self.assertTrue(any("ibuprofen" in r.lower() for r in result.rewrites))


class TestAggregationService(unittest.TestCase):
    """Unit tests for AggregationService."""

    def setUp(self):
        self.service = AggregationService()

    def _make_doc(self, text: str, metadata: dict = None) -> VectorDocument:
        return VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="codes",
            text=text,
            embedding=[0.0] * 384,
            metadata_json=metadata or {},
        )

    def test_extract_codes_from_metadata(self):
        """Test extracting identifier codes from metadata."""
        metadata = {
            "canonical_name": "Ibuprofen",
            "codes": [
                {"codeSystem": "CAS", "code": "15687-27-1"},
                {"codeSystem": "UNII", "code": "XKG6425636"},
            ],
        }
        doc = self._make_doc("Ibuprofen codes section", metadata)
        candidates = [(doc, 0.9)]

        result = self.service.aggregate(candidates, "How many identifiers has Ibuprofen?", "aggregation_identifiers")
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.aggregation_type, "identifiers")
        self.assertEqual(result.substance_name, "Ibuprofen")

    def test_extract_names_from_metadata(self):
        """Test extracting names from metadata."""
        metadata = {
            "canonical_name": "Ibuprofen",
            "names": ["Advil", "Motrin", "Nurofen"],
        }
        doc = self._make_doc("Ibuprofen names section", metadata)
        candidates = [(doc, 0.9)]

        result = self.service.aggregate(candidates, "List all names of Ibuprofen", "aggregation_names")
        self.assertEqual(result.total_count, 4)  # canonical + 3 names
        self.assertEqual(result.aggregation_type, "names")

    def test_empty_aggregation(self):
        """Test aggregation with no data."""
        doc = self._make_doc("Some text", {})
        candidates = [(doc, 0.5)]

        result = self.service.aggregate(candidates, "How many identifiers?", "aggregation_identifiers")
        self.assertEqual(result.total_count, 0)
        self.assertIn("No identifier codes found", result.raw_text_summary)

    def test_summary_format(self):
        """Test that summary is formatted correctly."""
        metadata = {
            "canonical_name": "Aspirin",
            "codes": [
                {"codeSystem": "CAS", "code": "50-78-2"},
            ],
        }
        doc = self._make_doc("Aspirin codes", metadata)
        candidates = [(doc, 0.9)]

        result = self.service.aggregate(candidates, "How many identifiers has Aspirin?", "aggregation_identifiers")
        self.assertIn("Aspirin", result.raw_text_summary)
        self.assertIn("1", result.raw_text_summary)
        self.assertIn("CAS", result.raw_text_summary)
        self.assertIn("50-78-2", result.raw_text_summary)


class TestMetadataFilterBuilder(unittest.TestCase):
    """Unit tests for MetadataFilterBuilder."""

    def setUp(self):
        self.builder = MetadataFilterBuilder()

    def test_merge_request_filters(self):
        """Test merging request-provided filters."""
        result = self.builder.build(
            request_filters={"section": "codes"},
        )
        self.assertEqual(result.get("section"), "codes")

    def test_merge_substance_classes(self):
        """Test merging substance class filters."""
        result = self.builder.build(
            substance_classes=["Chemical", "Protein"],
        )
        self.assertIn("substance_classes", result)
        self.assertEqual(result["substance_classes"], ["Chemical", "Protein"])

    def test_merge_sections(self):
        """Test merging section filters."""
        result = self.builder.build(
            sections=["codes", "names"],
        )
        self.assertIn("sections", result)
        self.assertEqual(result["sections"], ["codes", "names"])

    def test_merge_inferred_filters(self):
        """Test merging rewrite-derived filters."""
        result = self.builder.build(
            inferred_filters={"sections": ["codes"]},
        )
        self.assertIn("sections", result)
        self.assertEqual(result["sections"], ["codes"])

    def test_deduplication(self):
        """Test that duplicate values are removed."""
        result = self.builder.build(
            substance_classes=["Chemical"],
            inferred_filters={"substance_classes": ["Chemical", "Protein"]},
        )
        # Should deduplicate
        chemical_count = result["substance_classes"].count("Chemical")
        self.assertEqual(chemical_count, 1)

    def test_empty_filters_removed(self):
        """Test that empty filter values are removed."""
        result = self.builder.build(
            request_filters={"empty": ""},
        )
        self.assertNotIn("empty", result)


class TestLexicalRetriever(unittest.TestCase):
    """Unit tests for LexicalRetriever."""

    def setUp(self):
        self.retriever = LexicalRetriever(top_k=10)

    def _make_doc(self, text: str, metadata: dict = None) -> VectorDocument:
        doc = VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="test",
            text=text,
            embedding=[0.0] * 384,
            metadata_json=metadata or {},
        )
        return doc

    def test_token_overlap_scoring(self):
        """Test that documents with more term overlap score higher."""
        doc1 = self._make_doc("The CAS code for aspirin is 50-78-2")
        doc2 = self._make_doc("Aspirin is a common medication")
        doc3 = self._make_doc("Protein structure analysis")

        candidates = [(doc1, 0.9), (doc2, 0.8), (doc3, 0.3)]
        results = self.retriever.search("CAS code aspirin", candidates)

        # doc1 should score highest due to term overlap
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0].chunk_id, doc1.chunk_id)

    def test_empty_candidates(self):
        """Test that empty candidates return empty results."""
        results = self.retriever.search("test", [])
        self.assertEqual(results, [])

    def test_empty_query(self):
        """Test that empty query returns empty results."""
        doc = self._make_doc("Some text")
        results = self.retriever.search("", [(doc, 0.9)])
        self.assertEqual(results, [])

    def test_metadata_included_in_search(self):
        """Test that metadata is included in search text."""
        doc = self._make_doc(
            "Some chunk text",
            metadata={"canonical_name": "Ibuprofen", "codes": [{"code": "15687-27-1"}]}
        )
        candidates = [(doc, 0.8)]
        results = self.retriever.search("ibuprofen 15687-27-1", candidates)
        self.assertTrue(len(results) > 0)
        self.assertGreater(results[0][1], 0)


class TestRerankerService(unittest.TestCase):
    """Unit tests for RerankerService."""

    def setUp(self):
        self.reranker = RerankerService()

    def _make_doc(self, text: str, metadata: dict = None) -> VectorDocument:
        return VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="codes",
            text=text,
            embedding=[0.0] * 384,
            metadata_json=metadata or {},
        )

    def test_rerank_order(self):
        """Test that more relevant documents rank higher."""
        doc1 = self._make_doc("CAS code for aspirin is 50-78-2")
        doc2 = self._make_doc("General information about proteins")

        candidates = [(doc1, 0.7), (doc2, 0.8)]
        reranked = self.reranker.rerank(candidates, "CAS code aspirin")

        # doc1 should rank higher due to term overlap
        self.assertEqual(reranked[0][0].chunk_id, doc1.chunk_id)

    def test_identifier_match_boost(self):
        """Test that identifier matches get boosted."""
        doc1 = self._make_doc(
            "Some text",
            metadata={"codes": [{"code": "50-78-2"}]}
        )
        doc2 = self._make_doc("Some other text", metadata={"codes": []})

        candidates = [(doc1, 0.5), (doc2, 0.6)]
        reranked = self.reranker.rerank(candidates, "CAS 50-78-2")

        # doc1 should rank higher due to identifier match
        self.assertEqual(reranked[0][0].chunk_id, doc1.chunk_id)

    def test_section_match_boost(self):
        """Test that section matches get boosted."""
        doc1 = self._make_doc("Some text about codes", metadata={},)
        doc2 = self._make_doc("Some text", metadata={})

        candidates = [(doc1, 0.5), (doc2, 0.55)]
        reranked = self.reranker.rerank(
            candidates, "code lookup",
            filters={"sections": ["codes"]}
        )

        # doc1 is in codes section, should get boost
        self.assertEqual(reranked[0][0].chunk_id, doc1.chunk_id)

    def test_empty_candidates(self):
        """Test that empty candidates return empty list."""
        reranked = self.reranker.rerank([], "test query")
        self.assertEqual(reranked, [])

    def test_scores_normalized(self):
        """Test that scores are normalized to [0, 1]."""
        docs = [self._make_doc(f"Document {i}") for i in range(3)]
        candidates = [(doc, 0.5 + i * 0.1) for i, doc in enumerate(docs)]
        reranked = self.reranker.rerank(candidates, "test")

        for _, score in reranked:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1.0)


class TestAbstentionPolicy(unittest.TestCase):
    """Unit tests for AbstentionPolicy."""

    def setUp(self):
        self.policy = AbstentionPolicy(min_score_threshold=0.3)

    def _make_evidence(self, text: str, score: float, metadata: dict = None) -> EvidenceResult:
        doc = VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="test",
            text=text,
            embedding=[0.0] * 384,
            metadata_json=metadata or {},
        )
        return EvidenceResult(
            document=doc,
            score=score,
            citation=Citation(
                chunk_id=doc.chunk_id,
                document_id=str(doc.document_id),
                section=doc.section,
            ),
            snippet=text[:200],
        )

    def test_no_evidence_abstains(self):
        """Test abstention with no evidence."""
        decision = self.policy.evaluate([], "test query")
        self.assertTrue(decision.abstained)
        self.assertIn("No relevant evidence", decision.abstain_reason)

    def test_low_score_abstains(self):
        """Test abstention when scores are below threshold."""
        evidence = [self._make_evidence("Tangential text", score=0.1)]
        decision = self.policy.evaluate(evidence, "CAS code")
        self.assertTrue(decision.abstained)

    def test_high_score_answers(self):
        """Test answering when scores are above threshold."""
        evidence = [self._make_evidence("CAS code for aspirin is 50-78-2", score=0.8)]
        decision = self.policy.evaluate(evidence, "CAS code aspirin", intent="identifier_query")
        self.assertFalse(decision.abstained)

    def test_insufficient_evidence_count(self):
        """Test abstention with insufficient evidence count."""
        policy = AbstentionPolicy(min_evidence_count=2)
        evidence = [self._make_evidence("Some text", score=0.5)]
        decision = policy.evaluate(evidence, "test query")
        self.assertTrue(decision.abstained)


class TestEvidenceExtractor(unittest.TestCase):
    """Unit tests for EvidenceExtractor."""

    def setUp(self):
        self.extractor = EvidenceExtractor(max_evidence_count=5)

    def _make_candidate(self, text: str, score: float) -> tuple:
        doc = VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="test",
            text=text,
            embedding=[0.0] * 384,
            metadata_json={},
        )
        return (doc, score)

    def test_extraction_limits(self):
        """Test that evidence extraction respects max count."""
        candidates = [self._make_candidate(f"Text {i}", 0.9 - i * 0.05) for i in range(10)]
        evidence = self.extractor.extract(candidates, "test query")
        self.assertLessEqual(len(evidence), 5)

    def test_empty_candidates(self):
        """Test extraction from empty candidates."""
        evidence = self.extractor.extract([], "test")
        self.assertEqual(evidence, [])

    def test_citations_generated(self):
        """Test that citations are generated for each evidence."""
        candidates = [self._make_candidate("Test text", 0.8)]
        evidence = self.extractor.extract(candidates, "test")
        self.assertEqual(len(evidence), 1)
        self.assertIsInstance(evidence[0].citation, Citation)


class TestAnswerGenerator(unittest.TestCase):
    """Unit tests for AnswerGenerator."""

    def setUp(self):
        self.generator = AnswerGenerator(use_llm=False)

    def _make_evidence(self, text: str, score: float) -> EvidenceResult:
        doc = VectorDocument(
            id=str(uuid4()),
            chunk_id=f"chunk_{uuid4()}",
            document_id=uuid4(),
            section="codes",
            text=text,
            embedding=[0.0] * 384,
            metadata_json={},
        )
        return EvidenceResult(
            document=doc,
            score=score,
            citation=Citation(
                chunk_id=doc.chunk_id,
                document_id=str(doc.document_id),
                section=doc.section,
            ),
            snippet=text[:200],
        )

    def test_no_evidence_fallback(self):
        """Test fallback message when no evidence."""
        answer, citations = self.generator.generate("test", [])
        self.assertIn("cannot answer", answer.lower())
        self.assertEqual(citations, [])

    def test_single_evidence(self):
        """Test answer with single evidence."""
        evidence = [self._make_evidence("CAS code for aspirin is 50-78-2", 0.9)]
        answer, citations = self.generator.generate("What is the CAS for aspirin?", evidence)
        self.assertIn("50-78-2", answer)
        self.assertEqual(len(citations), 1)

    def test_multiple_evidence(self):
        """Test answer with multiple evidence chunks."""
        evidence = [
            self._make_evidence("CAS code for aspirin is 50-78-2", 0.9),
            self._make_evidence("Aspirin molecular weight is 180.16", 0.8),
        ]
        answer, citations = self.generator.generate("Tell me about aspirin", evidence)
        self.assertIn("evidence", answer.lower())
        self.assertEqual(len(citations), 2)


if __name__ == "__main__":
    unittest.main()


class TestExampleScripts(unittest.TestCase):
    """Sanity tests for example scripts."""

    def test_gsrs_tool_file_exists(self):
        """Test that gsrs_tool.py exists."""
        import os
        tool_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "gsrs_tool.py"
        )
        self.assertTrue(os.path.isfile(tool_path))

    def test_gsrs_function_file_exists(self):
        """Test that gsrs_function.py exists."""
        import os
        func_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "gsrs_function.py"
        )
        self.assertTrue(os.path.isfile(func_path))

    def test_system_prompt_exists(self):
        """Test that system prompt files exist."""
        import os
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "gsrs_system_prompt.md"
        )
        self.assertTrue(os.path.isfile(prompt_path))

    def test_system_prompt_minimal_exists(self):
        """Test that minimal system prompt exists."""
        import os
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "gsrs_system_prompt_minimal.md"
        )
        self.assertTrue(os.path.isfile(prompt_path))

    def test_examples_readme_exists(self):
        """Test that examples README exists."""
        import os
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "README.md"
        )
        self.assertTrue(os.path.isfile(readme_path))

    def test_gsrs_tool_uses_ask_endpoint(self):
        """Test that gsrs_tool.py uses MCP tools and /ask endpoint."""
        import os
        tool_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "gsrs_tool.py"
        )
        with open(tool_path, "r") as f:
            content = f.read()
        self.assertIn("gsrs_ask", content)
        self.assertIn("gsrs_similarity_search", content)
        self.assertIn("/ask", content)
        self.assertIn("mcp", content.lower())

    def test_gsrs_function_supports_modes(self):
        """Test that gsrs_function.py supports evidence and answer_assist modes."""
        import os
        func_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "gsrs_function.py"
        )
        with open(func_path, "r") as f:
            content = f.read()
        self.assertIn("evidence", content)
        self.assertIn("answer_assist", content)
        self.assertIn("/ask", content)

    def test_system_prompt_non_empty(self):
        """Test that system prompt files are non-empty."""
        import os
        for filename in ["gsrs_system_prompt.md", "gsrs_system_prompt_minimal.md"]:
            prompt_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "examples", filename
            )
            with open(prompt_path, "r") as f:
                content = f.read()
            self.assertTrue(len(content.strip()) > 0, f"{filename} is empty")


class TestSimilarSubstanceSearch(unittest.TestCase):
    """Tests for the similar substance search feature."""

    def test_system_prompt_mentions_similar_search(self):
        """Test that system prompt documents the similarity search feature."""
        import os
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "gsrs_system_prompt.md"
        )
        with open(prompt_path, "r") as f:
            content = f.read()
        self.assertIn("similar substance", content.lower())
        self.assertIn("automatic detection", content.lower())
        self.assertIn("gsrs json", content.lower())

    def test_similar_substance_models_exist(self):
        """Test that the API models are defined."""
        from app.models.api import (
            SimilarSubstanceRequest,
            SimilarSubstanceResponse,
            SimilarSubstanceResult,
        )
        req = SimilarSubstanceRequest(substance={"uuid": "test", "names": [{"name": "Test"}]})
        self.assertEqual(req.substance["uuid"], "test")
        self.assertEqual(req.match_mode, "contains")
        self.assertTrue(req.exclude_self)

    def test_extract_search_criteria(self):
        """Test extraction of search criteria from substance JSON with priorities."""
        from app.main import _extract_search_criteria

        substance = {
            "uuid": "abc-123",
            "approvalID": "APP-001",
            "names": [
                {"name": "Aspirin", "type": "Systematic name"},
                {"name": "Acetylsalicylic acid", "type": "Official name"},
                {"name": "ASA", "type": "Common name"},
            ],
            "codes": [
                {"codeSystem": "UNII", "code": "WK2XYI10QM"},
                {"codeSystem": "CAS", "code": "50-78-2"},
                {"codeSystem": "SMS_ID", "code": "SMS-001"},
            ],
            "structure": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "inchi": "InChI=1S/C9H8O4/c1-6(10)11-7-4-2-3-5-8(7)9(12)13/h2-5H,1H3",
            },
            "classifications": [{"name": "WHO-ATC", "className": "N02BA01"}],
        }

        criteria = _extract_search_criteria(substance)

        # Priority 1: UUID
        self.assertEqual(criteria["uuid"], "abc-123")

        # Priority 2: approvalID
        self.assertEqual(criteria["approvalID"], "APP-001")

        # Priority 3: Reliable codes (UNII, SMS_ID are in default reliable list)
        self.assertIn("reliable_codes", criteria)
        self.assertEqual(criteria["reliable_codes"]["UNII"], "WK2XYI10QM")
        self.assertEqual(criteria["reliable_codes"]["SMS_ID"], "SMS-001")
        self.assertIn("all_codes", criteria)
        self.assertEqual(criteria["all_codes"]["CAS"], "50-78-2")

        # Priority 4: Structure
        self.assertIn("structure", criteria)
        self.assertIn("smiles", criteria["structure"])
        self.assertIn("inchi", criteria["structure"])

        # Priority 5-6: Names
        self.assertIn("systematic_names", criteria)
        self.assertEqual(criteria["systematic_names"][0], "Aspirin")
        self.assertIn("official_names", criteria)
        # "Acetylsalicylic acid" (Official name) and "ASA" (Common name) both match official/common
        self.assertEqual(criteria["official_names"][0], "Acetylsalicylic acid")
        self.assertEqual(criteria["official_names"][1], "ASA")
        self.assertEqual(criteria["canonical_name"], "Aspirin")

        # Classifications
        self.assertIn("WHO-ATC", criteria["classifications"])

    def test_extract_search_criteria_empty(self):
        """Test extraction with empty substance returns empty dict."""
        from app.main import _extract_search_criteria
        criteria = _extract_search_criteria({})
        self.assertEqual(criteria, {})
