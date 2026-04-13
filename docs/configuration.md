# Configuration

## Core Settings

### Vector backend

- `DATABASE_URL`
  - `chroma://./chroma_data/chunks`
  - `postgresql://user:pass@host:5432/gsrs_mcp`

### Embeddings

- `EMBEDDING_API_KEY`
- `EMBEDDING_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`
- `EMBEDDING_VERIFY_SSL`
- `EMBEDDING_TIMEOUT`
- `EMBEDDING_MAX_RETRIES`
- `EMBEDDING_RETRY_BACKOFF_MS`

Embeddings are required for retrieval and ingest. If they are misconfigured, `/readyz` will fail and retrieval tools will return specific error messages.

### Optional answer generation

- `LLM_API_KEY`
- `LLM_URL`
- `LLM_MODEL`
- `LLM_VERIFY_SSL`
- `LLM_TIMEOUT`
- `LLM_MAX_RETRIES`
- `LLM_RETRY_BACKOFF_MS`

If `LLM_API_KEY` is empty, `gsrs_ask` stays available in retrieval-only mode.

### MCP transport

- `MCP_TRANSPORT`
  - `streamable-http`
  - `stdio`
- `MCP_API`
- `MCP_PORT`

### Authentication

- `MCP_USERNAME`
- `MCP_PASSWORD`

For the HTTP MCP endpoint, bearer token verification uses `MCP_PASSWORD` as the token value.
`MCP_USERNAME` is kept for deployment consistency, but it is not part of the current bearer-token check.

### GSRS upstream tools

- `GSRS_API_URL`
- `GSRS_API_TIMEOUT`
- `GSRS_API_VERIFY_SSL`
- `GSRS_API_MAX_RETRIES`
- `GSRS_API_RETRY_BACKOFF_MS`
- `GSRS_API_PUBLIC_ONLY`

### Runtime / observability

- `DEFAULT_TOP_K`
- `IDENTIFIER_CODE_SYSTEMS`
- `ANSWER_CONFIDENCE_THRESHOLD`
- `MAX_ANSWER_EVIDENCE`
- `DEBUG_MODE`
- `STARTUP_VALIDATE_EXTERNAL`

`STARTUP_VALIDATE_EXTERNAL=false` keeps startup cheap and uses configuration validation for external providers. Set it to `true` to actively probe embedding, LLM, and GSRS upstream dependencies during startup.

## Recommended Profiles

### Local ChromaDB + OpenAI embeddings

```bash
DATABASE_URL=chroma://./chroma_data/chunks
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_URL=https://api.openai.com/v1/embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

### Local ChromaDB + Ollama embeddings

```bash
DATABASE_URL=chroma://./chroma_data/chunks
EMBEDDING_API_KEY=ollama
EMBEDDING_URL=http://localhost:11434/api/embed
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

### PostgreSQL + pgvector

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gsrs_mcp
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_URL=https://api.openai.com/v1/embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

## Readiness Notes

The server is ready when:

- the vector backend is initialized
- the embedding provider is configured correctly
- the query pipeline builds successfully

The server may still be available but degraded when:

- answer generation is disabled or unavailable
- GSRS upstream tools cannot reach the GSRS API
- the GSRS chunker cannot initialize, which affects ingest but not retrieval readiness
