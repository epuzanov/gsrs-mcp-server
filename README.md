# GSRS MCP Server

Compact MCP server for GSRS substance retrieval and live GSRS API lookup.

## Runtime

- MCP transport: `streamable-http` on `/mcp`, legacy `sse` on `/sse`, or local `stdio`
- Health endpoints: `/livez`, `/readyz`, `/health`
- Auth: HTTP Bearer token on `/mcp` or `/sse` with `Authorization: Bearer <MCP_PASSWORD>`
- CLI entry point: `gsrs-mcp-server`

## Tools

- `rag_query`: child-level semantic search enriched with parent / root-section context. Each result shows the matching chunk plus reconstructed context from sibling chunks that share the same `(substance UUID, root_section)`.
- `rag_query_chunks`: plain child/chunk-level semantic retrieval without parent enrichment.
- `rag_ingest`: ingest one GSRS substance JSON document into the local vector store
- `gsrs_get_substance`: fetch a full substance JSON document from the GSRS API
- `gsrs_get_summary`: fetch a substance and return a markdown summary
- `gsrs_parametric_search`: run GSRS API fielded/faceted search
- `gsrs_structure_search`: run GSRS API chemical structure search
- `gsrs_sequence_search`: run GSRS API sequence search
- `health`: return runtime health/readiness details
- `statistics`: return local vector-store statistics

### When to use which RAG tool

Use `rag_query` when an answer may depend on nearby or sibling section context — for example, a `codes` chunk is more useful when it is shown alongside the substance overview, names, and definitions from the same record.

Use `rag_query_chunks` when you prefer minimal, focused chunk output and do not want sibling context mixed into the response.

The server does not synthesize final answers. Both tools return grounded evidence so the MCP client or calling model can answer from the retrieved chunks.

### RAG tool parameters

`rag_query_chunks(query, top_k=8, filters="")` — plain chunk retrieval.

`rag_query(query, top_k=8, include_parent_text=True, parent_text_limit=1000, filters="")` — chunk retrieval with parent context reconstruction. Parent context is rebuilt from all chunks that share the same `(document_id, root_section)`. The response exposes `Parent Context`, `Parent Summary`, and a `parent_text_truncated` flag when the summary is cut down to fit `parent_text_limit`.

Example `rag_query` response shape:

```text
Found 3 local RAG result(s) with parent context for "aspirin mechanism":

1. Score: `0.9523` | Section: `mechanisms`
   Substance UUID: `550e8400-e29b-41d4-a716-446655440000`
   Chunk: `mechanisms_550e8400-e29b-41d4-a716-446655440000_atomic_0`
   Text: Irreversible inhibitor of cyclooxygenase (COX) enzymes...
   **Parent Context**: 6 chunks in root, names, codes, definitions, mechanisms, references
   Parent Summary: [root] Substance: Acetylsalicylic Acid | Class: chemical
   parent_text_truncated: true
```

### Reindexing note

`rag_query` groups chunks by a dedicated `root_section` column (also mirrored
in chunk metadata). Vector stores indexed before this column existed will still
return results, but parent grouping may be less accurate until the data is
reingested. Re-ingest existing data with `rag_ingest` or the bulk loader to
populate the column and get correct hierarchical grouping.

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
