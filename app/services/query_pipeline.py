"""
GSRS MCP Server - Query Pipeline Service
Orchestrates the full query -> answer pipeline.
"""
import time
from typing import Any, Dict, List, Optional

from app.models.api import (
    AskRequest,
    AskResponse,
    QueryResult,
)
from app.services.query_rewrite import QueryRewriteService
from app.services.metadata_filters import MetadataFilterBuilder
from app.services.hybrid_retrieval import HybridRetriever
from app.services.reranking import RerankerService
from app.services.evidence import EvidenceExtractor, EvidenceResult
from app.services.answering import AnswerGenerator
from app.services.abstention import AbstentionPolicy, AbstentionDecision
from app.services.aggregation import AggregationService, AggregationResult
from app.services.identifier_routing import IdentifierRouter
from app.services.vector_database import VectorDatabaseService
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService


class QueryPipelineService:
    """
    Orchestrates the full query pipeline:
    1. Rewrite query
    2. Build metadata filters
    3. Run hybrid retrieval
    4. Rerank
    5. Extract evidence
    6. Evaluate abstention
    7. If not abstaining, generate answer with citations
    8. Return AskResponse
    """

    def __init__(
        self,
        vector_db: VectorDatabaseService,
        embedding_service: EmbeddingService,
        llm_service: Optional[LLMService] = None,
        top_k: int = 20,
        semantic_top_k: int = 40,
        lexical_top_k: int = 40,
        fused_top_k: int = 25,
        max_evidence: int = 10,
        min_confidence: float = 0.0,
        use_llm: bool = True,
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service

        # Initialize pipeline components
        self.rewrite_service = QueryRewriteService()
        self.filter_builder = MetadataFilterBuilder()
        self.hybrid_retriever = HybridRetriever(
            vector_db=vector_db,
            embedding_service=embedding_service,
            top_k=top_k,
            semantic_top_k=semantic_top_k,
            lexical_top_k=lexical_top_k,
            fused_top_k=fused_top_k,
        )
        self.identifier_router = IdentifierRouter(vector_db=vector_db)
        self.reranker = RerankerService()
        self.aggregator = AggregationService()
        self.evidence_extractor = EvidenceExtractor(max_evidence_count=max_evidence)
        self.answer_generator = AnswerGenerator(
            llm_service=llm_service,
            use_llm=use_llm,
        )
        self.abstention_policy = AbstentionPolicy(
            min_confidence=min_confidence,
        )

    def set_answer_generation_enabled(
        self,
        enabled: bool,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        """Enable or disable answer generation without rebuilding the pipeline."""
        self.answer_generator.llm = llm_service
        self.answer_generator.use_llm = enabled and llm_service is not None

    def ask(self, request: AskRequest) -> AskResponse:
        """Execute the full query pipeline and return the public response only."""
        response, _ = self.ask_with_diagnostics(request)
        return response

    def ask_with_diagnostics(self, request: AskRequest) -> tuple[AskResponse, Dict[str, Any]]:
        """Execute the full query pipeline."""
        diagnostics: Dict[str, Any] = {
            "stages": [],
        }

        def record_stage(stage_name: str, *, outcome: str = "success", started_at: float | None = None, **fields: Any) -> None:
            stage_payload = {
                "stage": stage_name,
                "outcome": outcome,
                **fields,
            }
            if started_at is not None:
                stage_payload["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            diagnostics["stages"].append(stage_payload)

        # 1. Rewrite query
        rewrite_started = time.perf_counter()
        rewrite_result = self.rewrite_service.rewrite(request.query)
        queries = [request.query] + rewrite_result.rewrites
        diagnostics["canonical_query"] = rewrite_result.canonical_query
        diagnostics["intent"] = rewrite_result.intent
        record_stage(
            "rewrite",
            started_at=rewrite_started,
            canonical_query=rewrite_result.canonical_query,
            intent=rewrite_result.intent,
            rewrite_count=len(rewrite_result.rewrites),
        )
        debug_info: Dict[str, Any] = {
            "canonical_query": rewrite_result.canonical_query,
            "intent": rewrite_result.intent,
        } if request.debug else {}

        # 2. Build metadata filters
        filter_started = time.perf_counter()
        applied_filters = self.filter_builder.build(
            request_filters=request.filters,
            substance_classes=request.substance_classes,
            sections=request.sections,
            inferred_filters=rewrite_result.filters,
        )
        diagnostics["applied_filters"] = applied_filters
        record_stage(
            "filters",
            started_at=filter_started,
            filter_keys=sorted(applied_filters.keys()),
        )
        if request.debug:
            debug_info["applied_filters"] = applied_filters

        # 3. Prefer deterministic identifier or exact-name routing when possible.
        retrieval_started = time.perf_counter()
        route_result = self.identifier_router.route(request.query, top_k=request.top_k)
        routing_mode = "hybrid"
        if route_result is not None:
            routing_mode = f"identifier-first:{route_result.route}"
            candidates = [(result.document, max(result.score, 0.99)) for result in route_result.results]
            if request.debug:
                debug_info["deterministic_route"] = {
                    "route": route_result.route,
                    "matched_value": route_result.matched_value,
                    "result_count": len(route_result.results),
                    "example": route_result.example,
                }
        else:
            candidates = self.hybrid_retriever.retrieve(
                queries=queries,
                filters=applied_filters,
            )
        diagnostics["retrieval_mode"] = routing_mode
        diagnostics["candidate_count"] = len(candidates)
        record_stage(
            "retrieval",
            started_at=retrieval_started,
            retrieval_mode=routing_mode,
            candidate_count=len(candidates),
            top_chunks=self._chunk_refs(candidates[:5]),
        )
        if request.debug:
            debug_info["retrieval_mode"] = routing_mode
            debug_info["retrieved_chunks"] = self._chunk_refs(candidates[:10])

        # 4. Rerank
        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(
            candidates=candidates,
            query=request.query,
            rewritten_queries=rewrite_result.rewrites,
            filters=applied_filters,
        )
        diagnostics["reranked_count"] = len(reranked)
        diagnostics["top_reranked_chunks"] = self._chunk_refs(reranked[:5])
        record_stage(
            "reranking",
            started_at=rerank_started,
            reranked_count=len(reranked),
            top_chunks=self._chunk_refs(reranked[:5]),
        )
        if request.debug:
            debug_info["reranked_chunks"] = [
                {
                    "chunk_id": doc.chunk_id,
                    "document_id": str(doc.document_id),
                    "section": doc.section,
                    "score": round(score, 4),
                }
                for doc, score in reranked[:10]
            ]

        # 5. Extract evidence
        evidence_started = time.perf_counter()
        evidence = self.evidence_extractor.extract(
            candidates=reranked,
            query=request.query,
            intent=rewrite_result.intent,
        )
        diagnostics["evidence_count"] = len(evidence)
        diagnostics["citation_count"] = len([item for item in evidence if item.score > 0.3][:5])
        record_stage(
            "evidence",
            started_at=evidence_started,
            evidence_count=len(evidence),
            top_chunks=self._evidence_refs(evidence[:5]),
        )
        if request.debug:
            debug_info["evidence"] = [
                {
                    "chunk_id": item.document.chunk_id,
                    "section": item.citation.section,
                    "score": round(item.score, 4),
                }
                for item in evidence
            ]

        # 6. Evaluate abstention
        abstention_started = time.perf_counter()
        abstention = self.abstention_policy.evaluate(
            evidence=evidence,
            query=request.query,
            intent=rewrite_result.intent,
            applied_filters=applied_filters,
        )
        diagnostics["confidence"] = abstention.confidence
        diagnostics["abstained"] = abstention.abstained
        diagnostics["abstain_reason"] = abstention.abstain_reason
        record_stage(
            "abstention",
            started_at=abstention_started,
            outcome="abstained" if abstention.abstained else "success",
            confidence=round(abstention.confidence, 4),
            abstain_reason=abstention.abstain_reason,
        )

        # 6.5 For aggregation queries, also extract structured data
        aggregation_result = None
        if rewrite_result.intent.startswith("aggregation_"):
            aggregation_result = self.aggregator.aggregate(
                candidates=reranked,
                query=request.query,
                intent=rewrite_result.intent,
            )

        # 7. Generate answer or abstain
        if abstention.abstained:
            response = self._build_abstain_response(
                request=request,
                rewrite_result=rewrite_result,
                applied_filters=applied_filters,
                evidence=evidence,
                abstention=abstention,
                aggregation_result=aggregation_result,
                debug_info=debug_info if request.debug else None,
            )
        else:
            response = self._build_answer_response(
                request=request,
                rewrite_result=rewrite_result,
                applied_filters=applied_filters,
                evidence=evidence,
                abstention=abstention,
                aggregation_result=aggregation_result,
                debug_info=debug_info if request.debug else None,
            )
        answer_trace = self.answer_generator.last_trace.to_dict() if self.answer_generator.last_trace else None
        if response.abstained:
            answer_trace = {
                "mode": "abstained",
                "llm_attempted": False,
                "used_llm": False,
                "fallback_used": False,
                "evidence_count": len(evidence),
                "citation_count": len(response.citations),
                "error_type": None,
                "error_message": None,
            }
        diagnostics["answer_generation"] = answer_trace
        record_stage(
            "answer_generation",
            outcome="abstained" if response.abstained else ("degraded" if response.degraded else "success"),
            mode=answer_trace["mode"] if answer_trace else "unknown",
            used_llm=answer_trace["used_llm"] if answer_trace else False,
            fallback_used=answer_trace["fallback_used"] if answer_trace else False,
            error_type=answer_trace["error_type"] if answer_trace else None,
        )
        diagnostics["degraded"] = response.degraded
        diagnostics["degraded_reason"] = response.degraded_reason
        if request.debug:
            debug_info["answer_generation"] = answer_trace
            debug_info["stage_trace"] = diagnostics["stages"]
            response.debug = debug_info
        return response, diagnostics

    def _build_abstain_response(
        self,
        request: AskRequest,
        rewrite_result,
        applied_filters: Dict[str, Any],
        evidence: List[EvidenceResult],
        abstention: AbstentionDecision,
        aggregation_result: Optional[AggregationResult] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> AskResponse:
        """Build response when abstaining."""
        evidence_chunks = []
        citations = []

        if request.return_evidence:
            evidence_chunks = self._evidence_to_query_results(evidence)
            citations = [e.citation for e in evidence[:5]]

        # For aggregation queries, include structured data even when abstaining
        answer = None
        if aggregation_result:
            answer = aggregation_result.raw_text_summary

        return AskResponse(
            query=request.query,
            rewritten_queries=[request.query] + rewrite_result.rewrites,
            applied_filters=applied_filters,
            answer=answer,
            citations=citations,
            evidence_chunks=evidence_chunks,
            confidence=abstention.confidence,
            abstained=True,
            abstain_reason=abstention.abstain_reason,
            degraded=not self.answer_generator.use_llm,
            degraded_reason=None if self.answer_generator.use_llm else "Answer generation provider unavailable; returning retrieval-only response.",
            debug=debug_info,
        )

    def _build_answer_response(
        self,
        request: AskRequest,
        rewrite_result,
        applied_filters: Dict[str, Any],
        evidence: List[EvidenceResult],
        abstention: AbstentionDecision,
        aggregation_result: Optional[AggregationResult] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> AskResponse:
        """Build response with answer."""
        # For aggregation queries, prefer structured result
        if aggregation_result:
            answer = aggregation_result.raw_text_summary
            # Generate LLM answer if available, using aggregation data
            if self.answer_generator.use_llm and self.answer_generator.llm:
                answer, _ = self.answer_generator.generate(
                    query=request.query,
                    evidence=evidence,
                    answer_style=request.answer_style,
                )
                # Prepend aggregation summary
                answer = f"{aggregation_result.raw_text_summary}\n\n{answer}"
        else:
            answer, citations_from_llm = self.answer_generator.generate(
                query=request.query,
                evidence=evidence,
                answer_style=request.answer_style,
            )
            # Use LLM citations if available
            if citations_from_llm:
                pass  # citations already set below

        citations = [e.citation for e in evidence if e.score > 0.3][:5]

        evidence_chunks = []
        if request.return_evidence:
            evidence_chunks = self._evidence_to_query_results(evidence)

        generation_trace = self.answer_generator.last_trace
        degraded = not self.answer_generator.use_llm
        degraded_reason = None if self.answer_generator.use_llm else "Answer generation provider unavailable; returned retrieval-grounded fallback answer."
        if generation_trace and generation_trace.mode == "template_fallback":
            degraded = True
            degraded_reason = "Answer generation failed at runtime; returned retrieval-grounded template fallback answer."

        return AskResponse(
            query=request.query,
            rewritten_queries=[request.query] + rewrite_result.rewrites,
            applied_filters=applied_filters,
            answer=answer,
            citations=citations,
            evidence_chunks=evidence_chunks,
            confidence=abstention.confidence,
            abstained=False,
            abstain_reason=None,
            degraded=degraded,
            degraded_reason=degraded_reason,
            debug=debug_info,
        )

    def _evidence_to_query_results(self, evidence: List[EvidenceResult]) -> List[QueryResult]:
        """Convert evidence to QueryResult objects for API response."""
        results = []
        for e in evidence:
            # Create a QueryResult from the document
            qr = QueryResult(chunk=e.document, score=e.score)
            results.append(qr)
        return results

    def _chunk_refs(self, candidates: List[tuple]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for doc, score in candidates:
            refs.append(
                {
                    "chunk_id": doc.chunk_id,
                    "document_id": str(doc.document_id),
                    "section": doc.section,
                    "score": round(score, 4),
                }
            )
        return refs

    def _evidence_refs(self, evidence: List[EvidenceResult]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for item in evidence:
            refs.append(
                {
                    "chunk_id": item.document.chunk_id,
                    "document_id": str(item.document.document_id),
                    "section": item.citation.section,
                    "score": round(item.score, 4),
                }
            )
        return refs
