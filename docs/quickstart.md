# Quick Start

## 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## 2. Configure

Minimum settings:

```bash
DATABASE_URL=chroma://./chroma_data/chunks
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_URL=https://api.openai.com/v1/embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
MCP_PASSWORD=change-me
```

Optional grounded answer generation:

```bash
LLM_API_KEY=sk-your-key
LLM_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-4o-mini
```

## 3. Run

Streamable HTTP:

```bash
gsrs-mcp-server
```

stdio:

```bash
set MCP_TRANSPORT=stdio
gsrs-mcp-server
```

## 4. Verify

```bash
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
curl http://localhost:8000/health
```

Interpretation:

- `/livez` should return `{"status":"alive"}`
- `/readyz` should return HTTP `200` when retrieval dependencies are ready
- `/health` should include a `status` field of `ready`, `ready_degraded`, or `not_ready`
- an empty but connected vector database is still ready

## 5. Load Data

```bash
python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028
```

## 6. Connect an MCP Client

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

stdio:

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
