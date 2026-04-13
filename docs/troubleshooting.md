# Troubleshooting

## `/livez` works but `/readyz` returns `503`

Check `/health` for the failing component. Common causes:

- `EMBEDDING_API_KEY` missing
- `EMBEDDING_DIMENSION` does not match the configured backend collection
- database connection string is wrong
- chunker dependencies are not installed correctly

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

## ChromaDB dimension mismatch

If the existing Chroma collection was created with a different embedding dimension, use a new path or align `EMBEDDING_DIMENSION`.

## Need more retrieval detail

Set `DEBUG_MODE=true` or pass `debug=true` to:

- `gsrs_ask`
- `gsrs_retrieve`
