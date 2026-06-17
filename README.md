# GSRS MCP Server

Compact MCP server for GSRS substance retrieval and live GSRS API lookup.

## Runtime

- MCP transport: `streamable-http` on `/mcp`, legacy `sse` on `/sse`, or local `stdio`
- Health endpoints: `/livez`, `/readyz`, `/health`
- Auth: HTTP Bearer token on `/mcp` or `/sse` with `Authorization: Bearer <MCP_PASSWORD>`
- CLI entry point: `gsrs-mcp-server`

## Tools

- `rag_query`: search locally ingested GSRS chunks and return evidence with parent context
- `rag_query_chunks`: search local chunks and return raw chunk evidence (no parent context)
- `rag_ingest`: ingest one GSRS substance JSON document into the local vector store
- `gsrs_get_substance`: fetch a full substance JSON document from the GSRS API
- `gsrs_get_summary`: fetch a substance and return a markdown summary
- `gsrs_parametric_search`: run GSRS API fielded/faceted search
- `gsrs_structure_search`: run GSRS API chemical structure search
- `gsrs_sequence_search`: run GSRS API sequence search
- `health`: return runtime health/readiness details
- `statistics`: return local vector-store statistics

The server does not synthesize final answers. `rag_query` returns grounded evidence so the MCP client or calling model can answer from the retrieved chunks.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
gsrs-mcp-server
curl http://localhost:8000/readyz
```

Minimum useful configuration:

```bash
DATABASE_URL=chroma://./chroma_data/chunks
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_URL=https://api.openai.com/v1/embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
MCP_PASSWORD=change-me
```

## Client Config

Stdio:

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

Streamable HTTP:

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

Load local `.gsrs` exports through MCP:

```bash
python scripts/load_data.py ./substances.gsrs --mcp-url http://localhost:8000/mcp --token change-me
```

Call a tool from the helper CLI:

```bash
python scripts/gsrs_mcp_cli.py --transport http --url http://localhost:8000/mcp --token change-me --tool rag_query --query aspirin
```

Generate a markdown summary from a local GSRS JSON file:

```bash
python scripts/json2md.py substance.json substance.md
```

## Deployment

Two manifests ship in the repository root:

- [`docker-compose.yaml`](docker-compose.yaml) — multi-service compose, suitable for a privileged host or a single-node rootful podman / docker setup.
- [`podman-kube-play.yaml`](podman-kube-play.yaml) — single-pod K8s manifest, intended for **`podman kube play` in rootless mode**. All env values live in a separate file, [`mcp-config.yaml`](mcp-config.yaml) (a `kind: ConfigMap gsrs-env` resource, generated from your project `.env`). The three containers pull them via `envFrom: configMapRef:`. Deploy with `podman kube play --configmap mcp-config.yaml podman-kube-play.yaml`. The full workflow is in the header comment of the manifest and in the deployment guide.

For the rootless podman path (caddy publishes on `hostPort 8080` / `8443`,
the host firewall forwards the standard `80` / `443`) see
[`docs/guides/rootless-podman-deployment.md`](docs/guides/rootless-podman-deployment.md).
