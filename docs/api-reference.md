# API Reference

## HTTP Endpoints

### `GET /livez`

Process liveness only.

### `GET /readyz`

Runtime readiness. Returns HTTP `200` when retrieval dependencies are ready, otherwise `503`.
Ingest-specific chunker failures are reported as degraded component state in `/health`, not as retrieval readiness failures.

### `GET /health`

Combined snapshot of liveness, readiness, component state, backend, and light in-memory metrics.

### `POST /mcp`

Streamable HTTP MCP endpoint.

### `POST /eri/query`

Legacy ERI compatibility endpoint for older Open WebUI tools.

Behavior:

- accepts HTTP Basic auth using `MCP_USERNAME` and `MCP_PASSWORD`
- also accepts `Authorization: Bearer <MCP_PASSWORD>` for compatibility with newer deployments
- expects a JSON body with `query`, optional `top_k`, and optional `filters`
- returns `{"results": [...]}` with legacy `id`, `text`, `score`, and `metadata` fields
- uses the same identifier-first and semantic retrieval logic as `gsrs_retrieve`

## MCP Tools

### `gsrs_ask`

Grounded answering over GSRS chunks.

Arguments:

- `query`
- `top_k`
- `answer_style`
- `return_evidence`
- `min_confidence`
- `debug`

Behavior:

- expects a natural-language question, not a GSRS JSON payload
- returns a redirect message when a GSRS substance JSON document is passed
- prefers identifier-first deterministic lookup when possible
- abstains on low-confidence retrieval
- degrades to retrieval-only output if answer generation is unavailable

### `gsrs_retrieve`

Raw retrieval without answer synthesis.

Arguments:

- `query`
- `top_k`
- `filters`
- `debug`

### `gsrs_similarity_search`

Find similar substances from a GSRS JSON document.

Arguments:

- `substance_json`
- `top_k`
- `match_mode`
- `exclude_self`

### `gsrs_ingest`

Chunk, embed, and store a GSRS substance JSON document.

### `gsrs_delete`

Delete all chunks for a substance UUID.

### `gsrs_health`

Return the same structured runtime health snapshot exposed by `/health`.

### `gsrs_statistics`

Return database statistics.

### `gsrs_aggregation`

Aggregation-oriented retrieval for count/list-style questions.

### `gsrs_query_optimizer`

Show query rewrites, inferred filters, and optional translation output.

### `gsrs_get_document`

Fetch a full GSRS document by UUID from the GSRS upstream API.

### `gsrs_api_search`

Run GSRS upstream text search.

### `gsrs_api_structure_search`

Run GSRS upstream structure search.

### `gsrs_api_sequence_search`

Run GSRS upstream sequence search.
