"""
GSRS MCP Server - Hybrid Retriever
Combines semantic and lexical retrieval using Reciprocal Rank Fusion.
"""
from typing import Any, Dict, List, Optional

from app.models.db import DBQueryResult
from app.services.lexical_retrieval import LexicalRetriever
from app.services.vector_database import VectorDatabaseService
from app.services.embedding import EmbeddingService


class HybridRetriever:
    """
    Hybrid retrieval combining semantic (vector) and lexical (keyword) search
    using Reciprocal Rank Fusion (RRF).

    Flow:
    1. Semantic retrieval for each rewritten query
    2. Lexical retrieval over semantic candidates
    3. Fuse results using RRF
    """

    def __init__(
        self,
        vector_db: VectorDatabaseService,
        embedding_service: EmbeddingService,
        top_k: int = 20,
        semantic_top_k: int = 40,
        lexical_top_k: int = 40,
        fused_top_k: int = 25,
        rrf_k: int = 60,
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.semantic_top_k = semantic_top_k
        self.lexical_top_k = lexical_top_k
        self.fused_top_k = fused_top_k
        self.rrf_k = rrf_k
        self.lexical_retriever = LexicalRetriever(top_k=lexical_top_k)

    def retrieve(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQueryResult]:
        """
        Perform hybrid retrieval.

        Args:
            queries: List of rewritten queries
            filters: Metadata filters

        Returns:
            List of DBQueryResult sorted by fused RRF score
        """
        # Collect all semantic results across queries
        semantic_results: Dict[str, DBQueryResult] = {}
        for query in queries:
            results = self._semantic_search(query, filters)
            for r in results:
                if r.document.chunk_id not in semantic_results:
                    semantic_results[r.document.chunk_id] = r

        # Convert to list for lexical scoring
        semantic_list = list(semantic_results.values())

        # Try native lexical search first (pgvector FTS)
        # Fall back to in-memory lexical retriever (Chroma)
        lexical_results_per_query: Dict[str, List[DBQueryResult]] = {}
        for query in queries:
            native_results = self._native_lexical_search(query, filters)
            if native_results:
                lexical_results_per_query[query] = native_results
            else:
                # Fallback to in-memory lexical scoring
                in_memory_results = self._lexical_retriever_fallback(
                    query, semantic_list, filters,
                )
                lexical_results_per_query[query] = in_memory_results

        # Build ranking lists for RRF
        # Semantic rankings (per query)
        semantic_rankings: List[List[str]] = []
        for query in queries:
            raw_results = self._semantic_search(query, filters)
            rankings = [r.document.chunk_id for r in raw_results]
            semantic_rankings.append(rankings)

        # Lexical rankings (per query)
        lexical_rankings: List[List[str]] = []
        for query in queries:
            ranked = lexical_results_per_query.get(query, [])
            rankings = [r.document.chunk_id for r in ranked]
            lexical_rankings.append(rankings)

        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        all_results: Dict[str, DBQueryResult] = {}

        # Merge all results
        for r in semantic_list:
            all_results[r.document.chunk_id] = r
        for query_results in lexical_results_per_query.values():
            for r in query_results:
                if r.document.chunk_id not in all_results:
                    all_results[r.document.chunk_id] = r

        # From semantic rankings
        for ranking in semantic_rankings:
            for rank, chunk_id in enumerate(ranking, 1):
                if chunk_id not in all_results:
                    existing = semantic_results.get(chunk_id)
                    if existing:
                        all_results[chunk_id] = existing
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (rank + self.rrf_k)

        # From lexical rankings
        for ranking in lexical_rankings:
            for rank, chunk_id in enumerate(ranking, 1):
                if chunk_id not in all_results:
                    # Find from lexical results
                    for q_results in lexical_results_per_query.values():
                        found = next((r for r in q_results if r.document.chunk_id == chunk_id), None)
                        if found:
                            all_results[chunk_id] = found
                            break
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (rank + self.rrf_k)

        # Sort by RRF score (descending) using DBQueryResult comparison
        sorted_results = sorted(
            [DBQueryResult(doc, rrf_scores[cid]) for cid, doc in
             ((cid, all_results[cid].document) for cid in all_results if cid in rrf_scores)],
            reverse=True,
        )

        return sorted_results[: self.fused_top_k]

    def _lexical_retriever_fallback(
        self,
        query: str,
        semantic_list: List[DBQueryResult],
        filters: Optional[Dict[str, Any]],
    ) -> List[DBQueryResult]:
        """In-memory lexical retrieval fallback."""
        return self.lexical_retriever.search(query, semantic_list, filters)

    def _native_lexical_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQueryResult]:
        """Try native lexical search (e.g., pgvector FTS)."""
        try:
            return self.vector_db.lexical_search(query, top_k=self.lexical_top_k, filters=filters)
        except (NotImplementedError, Exception):
            return []

    def _semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DBQueryResult]:
        """Perform semantic search for a single query."""
        embedding = self.embedding_service.embed(query)
        return self.vector_db.similarity_search(
            embedding, top_k=self.semantic_top_k, filters=filters
        )
    