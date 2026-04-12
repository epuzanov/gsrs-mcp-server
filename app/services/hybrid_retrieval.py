"""
GSRS MCP Server - Hybrid Retriever
Combines semantic and lexical retrieval using Reciprocal Rank Fusion.
"""
from typing import Any, Dict, List, Optional, Tuple

from app.models.db import VectorDocument
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
    ) -> List[Tuple[VectorDocument, float]]:
        """
        Perform hybrid retrieval.

        Args:
            queries: List of rewritten queries
            filters: Metadata filters

        Returns:
            List of (document, fused_score) tuples, sorted by fused score
        """
        # Collect all semantic results across queries
        semantic_results: Dict[str, Tuple[VectorDocument, float]] = {}
        for query in queries:
            results = self._semantic_search(query, filters)
            for doc, score in results:
                if doc.chunk_id not in semantic_results:
                    semantic_results[doc.chunk_id] = (doc, score)

        # Convert to list for lexical scoring
        semantic_list = list(semantic_results.values())

        # Try native lexical search first (pgvector FTS)
        # Fall back to in-memory lexical retriever (Chroma)
        lexical_results_per_query = {}
        for query in queries:
            native_results = self._native_lexical_search(query, filters)
            if native_results:
                lexical_results_per_query[query] = native_results
            else:
                # Fallback to in-memory lexical scoring
                in_memory_results = self.lexical_retriever.search(query, semantic_list, filters)
                lexical_results_per_query[query] = in_memory_results

        # Build ranking lists for RRF
        # Semantic rankings (per query)
        semantic_rankings: List[List[str]] = []
        for query in queries:
            raw_results = self._semantic_search(query, filters)
            rankings = [doc.chunk_id for doc, _ in raw_results]
            semantic_rankings.append(rankings)

        # Lexical rankings (per query)
        lexical_rankings: List[List[str]] = []
        for query in queries:
            ranked = lexical_results_per_query.get(query, [])
            rankings = [doc.chunk_id for doc, _ in ranked]
            lexical_rankings.append(rankings)

        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        all_chunk_ids: Dict[str, Tuple[VectorDocument, float]] = {}

        # Merge all results into all_chunk_ids
        for doc, score in semantic_list:
            all_chunk_ids[doc.chunk_id] = (doc, score)
        for query_results in lexical_results_per_query.values():
            for doc, score in query_results:
                if doc.chunk_id not in all_chunk_ids:
                    all_chunk_ids[doc.chunk_id] = (doc, score)

        # From semantic rankings
        for ranking in semantic_rankings:
            for rank, chunk_id in enumerate(ranking, 1):
                if chunk_id not in all_chunk_ids:
                    doc = semantic_results.get(chunk_id, (None, 0))[0]
                    s = semantic_results.get(chunk_id, (None, 0))[1]
                    if doc:
                        all_chunk_ids[chunk_id] = (doc, s)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (rank + self.rrf_k)

        # From lexical rankings
        for ranking in lexical_rankings:
            for rank, chunk_id in enumerate(ranking, 1):
                if chunk_id not in all_chunk_ids:
                    # Find from lexical results
                    for q_results in lexical_results_per_query.values():
                        found = next(((d, s) for d, s in q_results if d.chunk_id == chunk_id), None)
                        if found:
                            all_chunk_ids[chunk_id] = found
                            break
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (rank + self.rrf_k)

        # Sort by RRF score and return (document, rrf_score) tuples
        sorted_results = sorted(
            [(doc, rrf_scores[cid]) for cid, (doc, _score) in all_chunk_ids.items() if cid in rrf_scores],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results[: self.fused_top_k]

    def _native_lexical_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """Try native lexical search (e.g., pgvector FTS)."""
        try:
            return self.vector_db.lexical_search(query, top_k=self.lexical_top_k, filters=filters)
        except (NotImplementedError, Exception):
            return []

    def _semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """Perform semantic search for a single query."""
        embedding = self.embedding_service.embed(query)
        return self.vector_db.similarity_search(
            embedding, top_k=self.semantic_top_k, filters=filters
        )
