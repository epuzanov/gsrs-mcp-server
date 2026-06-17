# Changelog

## cleanup

- Reduced the MCP surface to RAG query/ingest, GSRS API lookup/search, health, and statistics.
- Removed the server-side answer synthesis pipeline, including rewrite, hybrid retrieval, reranking, evidence extraction, abstention, aggregation, and LLM services.
- Added `gsrs_get_summary`, which returns a markdown summary generated from GSRS substance JSON.
- Updated `scripts/json2md.py` to reuse the same markdown summary formatter.
- Retargeted loader and CLI helpers to `rag_ingest`, `rag_query`, `rag_query_chunks`, `health`, and `statistics`.
