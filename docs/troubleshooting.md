# Troubleshooting

## `/livez` works but `/readyz` returns `503`

Check `/health` for the failing component. Common causes:

- `EMBEDDING_API_KEY` missing
- `EMBEDDING_DIMENSION` does not match the configured backend collection
- database connection string is wrong

## Empty database looks unhealthy

It should not. An empty but connected backend is still ready. `/health` should show:

```json
{
  "ready": true,
  "statistics": {
    "total_chunks": 0,
    "total_substances": 0
  }
}
```

## `gsrs_ask` returns degraded output

That usually means the retrieval pipeline is healthy but answer generation is unavailable. Check:

- `LLM_API_KEY`
- `LLM_URL`
- `LLM_MODEL`

The fallback answer is expected behavior.

## `gsrs_ingest` says ingestion is unavailable

The message should point to the specific failed dependency. Common causes:

- embedding provider configuration is missing or invalid
- the vector backend failed startup
- chunker dependencies failed to initialize

## `/readyz` is healthy but ingest is degraded

That is expected when retrieval dependencies are ready but the optional ingest chunker failed startup.
Check `/health` for the `chunker` component and verify the GSRS model/chunker dependencies are installed correctly.

## `gsrs_api_*` tools fail while `/readyz` is healthy

That is expected when the core retrieval path is healthy but the optional GSRS upstream dependency is degraded. Check `/health` for the `gsrs_api` component and verify:

- `GSRS_API_URL`
- `GSRS_API_TIMEOUT`
- network reachability to the upstream GSRS service

## ChromaDB dimension mismatch

If the existing Chroma collection was created with a different embedding dimension, use a new path or align `EMBEDDING_DIMENSION`.

## Need more retrieval detail

Set `DEBUG_MODE=true` or pass `debug=true` to:

- `gsrs_ask`
- `gsrs_retrieve`

In `gsrs_ask`, debug output now includes:

- normalized query and intent
- deterministic routing decisions
- retrieved and reranked chunk IDs with scores
- stage trace for retrieval, evidence, abstention, and answer generation
- whether answer generation used the LLM, degraded to a template fallback, or abstained

## Identifier lookup abstains unexpectedly

If the query includes a concrete UUID, approval ID, code, or InChIKey, the server now prefers deterministic identifier-first routing. That means:

- a miss is treated as an explicit abstention, not a fuzzy fallback
- `/health` may still be healthy because this is an answer-quality decision, not an outage
- `debug=true` will show `retrieval_mode` and the deterministic route that was attempted

## Answer includes fewer evidence chunks than before

That is intentional. The evidence selector now keeps a tighter set of high-confidence chunks and drops weak tail chunks to reduce answer drift. Use `debug=true` to inspect retrieved vs selected chunks if you need to compare ranking behavior.
