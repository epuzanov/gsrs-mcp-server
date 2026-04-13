# GSRS MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/epuzanov/gsrs-mcp-server/actions/workflows/tests.yml/badge.svg)](https://github.com/epuzanov/gsrs-mcp-server/actions)

`gsrs-mcp-server` is an MCP (Model Context Protocol) server for GSRS substance data. It supports `pgvector` or `ChromaDB`, OpenAI-compatible embedding providers, optional answer generation, and MCP tools for grounded retrieval, similarity search, ingest, and deletion.

## What It Exposes

- MCP transport: `streamable-http` on `/mcp`, or `stdio`
- Health endpoints: `/livez`, `/readyz`, `/health`
- MCP tools:
  - `gsrs_ask`
  - `gsrs_similarity_search`
  - `gsrs_retrieve`
  - `gsrs_ingest`
  - `gsrs_delete`
  - `gsrs_health`
  - `gsrs_statistics`
  - `gsrs_aggregation`
  - `gsrs_query_optimizer`
  - `gsrs_get_document`
  - `gsrs_api_search`
  - `gsrs_api_structure_search`
  - `gsrs_api_sequence_search`

## Current Architecture

The server is MCP-first. `app/main.py` builds a `FastMCP` server, not a FastAPI REST application.

Runtime flow:

1. Startup builds a shared runtime with the configured vector backend, embedding client, optional LLM client, GSRS upstream client, and GSRS chunker.
2. `/livez` reports process liveness only.
3. `/readyz` reports whether the runtime is ready for core retrieval and ingest capabilities.
4. `gsrs_ask` uses query rewrite, metadata filter inference, identifier-first routing, hybrid retrieval, reranking, evidence extraction, abstention, and optional answer generation.
5. If answer generation is unavailable, `gsrs_ask` degrades to retrieval-grounded fallback output instead of failing.
6. Tool availability is capability-specific: similarity search only requires the vector backend, while `gsrs_api_*` tools depend on GSRS upstream readiness.

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### 2. Configure `.env`

Minimum required settings:

```bash
DATABASE_URL=chroma://./chroma_data/chunks
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_URL=https://api.openai.com/v1/embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
MCP_PASSWORD=change-me
```

Optional LLM-backed answering:

```bash
LLM_API_KEY=sk-your-key
LLM_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-4o-mini
```

### 3. Run the Server

Streamable HTTP:

```bash
gsrs-mcp-server
```

Or stdio for local MCP clients:

```bash
set MCP_TRANSPORT=stdio
gsrs-mcp-server
```

### 4. Check Health

```bash
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
curl http://localhost:8000/health
```

An empty but connected database is still considered ready.

## Authentication

When `MCP_USERNAME` and `MCP_PASSWORD` are set, the MCP endpoint uses bearer token verification based on `MCP_PASSWORD`.

- MCP HTTP auth: `Authorization: Bearer <MCP_PASSWORD>`
- Health endpoints: no auth
- Default credentials are for local development only

For stdio transport, auth is not used because the process is local.

## MCP Client Examples

### Claude Desktop / stdio

```json
{
  "mcpServers": {
    "gsrs": {
      "command": "gsrs-mcp-server",
      "env": {
        "MCP_TRANSPORT": "stdio",
        "DATABASE_URL": "chroma://./chroma_data/chunks",
        "EMBEDDING_API_KEY": "sk-your-key",
        "EMBEDDING_URL": "https://api.openai.com/v1/embeddings",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSION": "1536"
      }
    }
  }
}
```

### Streamable HTTP

```json
{
  "mcpServers": {
    "gsrs": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer change-me"
      }
    }
  }
}
```

## Loading Data

Use the bundled loader:

```bash
python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028
```

The loader now uses the MCP client library directly and can talk to the server
over streamable HTTP or `stdio`.

```bash
python scripts/load_data.py --transport stdio --command gsrs-mcp-server --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028
```

Or ingest through MCP with `gsrs_ingest`.

## Health and Readiness Semantics

- `/livez`: process is running
- `/readyz`: runtime is initialized and retrieval dependencies are usable
- `/health`: combined snapshot with component state and light in-memory metrics

`/health` returns a deterministic `status` field:

- `starting`: startup has not finished yet
- `ready`: required components are ready
- `ready_degraded`: required components are ready, but one or more optional components are unavailable
- `not_ready`: one or more required components failed initialization

Readiness depends on:

- vector backend initialization
- embedding provider configuration, and optional active probing if `STARTUP_VALIDATE_EXTERNAL=true`
- chunker initialization
- query pipeline construction

Optional components:

- answer generation provider
- GSRS upstream API tools

If optional components are unavailable, the server stays up and reports a degraded state.

Tool degradation is explicit:

- `gsrs_ask` falls back to retrieval-grounded output when answer generation is unavailable
- `gsrs_similarity_search` still works when the vector backend is healthy, even if embeddings are unavailable
- `gsrs_ingest` reports chunker or embedding failures specifically instead of returning a generic retrieval error
- `gsrs_api_*` tools fail fast with a GSRS-upstream-specific message when the upstream dependency is unavailable

## Conservative Retrieval Behavior

The retrieval path is intentionally conservative:

- explicit UUID, approval ID, code, and InChIKey queries prefer deterministic identifier-first lookup
- quoted or short exact-name queries receive stronger exact-name ranking boosts
- reranking prefers field-aware matches in `codes`, `names`, and `structure` sections instead of relying on fuzzy overlap alone
- evidence selection keeps the highest-confidence chunks and drops weak tail chunks that are likely to increase answer drift
- low-confidence or identifier-miss cases abstain with an explicit reason instead of fabricating an answer

When `gsrs_ask` answers successfully, the MCP response includes:

- a direct answer
- supporting evidence excerpts
- citations with chunk references
- an uncertainty note when the server abstains or degrades

## Debug and Observability

Structured JSON logs include fields such as:

- `request_id`
- `tool_name`
- `backend`
- `latency_ms`
- `outcome`
- `error_type`
- `result_count`
- `citation_count`

For `gsrs_ask`, logs now also emit stage-level events for:

- rewrite and normalization
- retrieval routing and candidate counts
- reranking
- evidence extraction
- abstention
- answer generation or template fallback

Set `DEBUG_MODE=true` to include extra runtime detail. You can also pass `debug=true` to `gsrs_ask` or `gsrs_retrieve` for internal diagnostics such as:

- query normalization
- deterministic identifier routing
- retrieved chunk IDs and scores
- reranked chunk IDs and scores
- applied filters
- degraded answer-generation state

Sensitive values such as API keys, passwords, bearer tokens, and authorization headers are redacted from structured logs.

## Docker

```bash
docker-compose --profile chroma up -d
docker-compose --profile postgres up -d
```

The server container exposes `http://localhost:8000/mcp`.

For Podman, a `podman kube play` manifest is included at
[deploy/podman-kube-play.yaml](/c:/Users/egor.puzanov/Projects/gsrs-mcp-server/deploy/podman-kube-play.yaml).
Build the local image first, then play the manifest:

```bash
podman build -t localhost/gsrs-mcp-server:latest .
podman kube play deploy/podman-kube-play.yaml
```

## Repository Layout

```text
app/
  main.py                MCP server and health routes
  runtime.py             Startup validation and shared runtime state
  db/                    pgvector and ChromaDB backends
  services/              retrieval, reranking, answering, GSRS API, embeddings
docs/
examples/
scripts/
tests/
```

## Known Limitations / Future Work

- Startup validation defaults to configuration checks for external providers; set `STARTUP_VALIDATE_EXTERNAL=true` for active probes.
- The metrics layer is intentionally lightweight and in-memory.
- Retrieval ranking is conservative and still heuristic-heavy.
- Example integrations focus on MCP usage rather than a polished Open WebUI package.
- CI covers smoke and failure modes, but not long-running backend soak tests.

## Development

```bash
python -m pytest tests -v
python -m compileall app tests
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Documentation

- [docs/README.md](docs/README.md)
- [docs/quickstart.md](docs/quickstart.md)
- [docs/configuration.md](docs/configuration.md)
- [docs/authentication.md](docs/authentication.md)
- [docs/api-reference.md](docs/api-reference.md)
- [docs/guides/mcp.md](docs/guides/mcp.md)
