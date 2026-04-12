# GSRS MCP Server

The GSRS MCP server exposes substance search, Q&A, similarity search, and management
via the **Model Context Protocol (MCP)**.

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `gsrs_ask` | Full AI-answering pipeline with citations and evidence. Auto-detects GSRS JSON for similarity search. |
| `gsrs_similarity_search` | Find similar substances from a GSRS JSON document. Priority-based scoring. |
| `gsrs_retrieve` | Semantic chunk retrieval — raw results, no AI answer. |
| `gsrs_ingest` | Ingest a GSRS substance JSON into the database. |
| `gsrs_delete` | Delete all chunks for a substance UUID. |
| `gsrs_health` | Gateway health, model info, and database statistics. |
| `gsrs_statistics` | Raw database statistics as JSON. |

## Quick Start

### 1. Start the MCP Server

```bash
# SSE mode (HTTP endpoint — default)
gsrs-gateway

# stdio mode (for Claude Desktop, Cursor, etc.)
MCP_TRANSPORT=stdio gsrs-gateway
```

### 2. Connect an MCP Client

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gsrs-mcp": {
      "command": "gsrs-gateway",
      "args": [],
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

**Claude Desktop (SSE mode)**:

```json
{
  "mcpServers": {
    "gsrs-mcp": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `sse` | Transport: `stdio` or `sse` |
| `MCP_HOST` | `0.0.0.0` | SSE host |
| `MCP_PORT` | `8000` | SSE port |
| `MCP_REQUEST_TIMEOUT` | `60` | HTTP timeout (seconds) |
| `MCP_VERIFY_SSL` | `true` | SSL verification |

## Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector
gsrs-gateway
# Open MCP Inspector at http://localhost:6274
# Connect to http://localhost:8000/mcp
```

## Docker Deployment

```yaml
services:
  gsrs-mcp:
    image: gsrs-rag-gateway:latest
    environment:
      - DATABASE_URL=postgresql://...
      - EMBEDDING_API_KEY=...
      - MCP_TRANSPORT=sse
    ports:
      - "8000:8000"
```

## Open WebUI + Ollama + Remote MCP Server

This configuration runs Ollama locally for embeddings and LLM, with a **remote** GSRS MCP server for substance data.

### Architecture

```
User → Open WebUI → Ollama (local LLM)
                  ↘ GSRS MCP server (remote, SSE)
```

### docker-compose.yaml

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - open-webui_data:/app/backend/data
    restart: unless-stopped
    depends_on:
      - ollama

volumes:
  ollama_data:
  open-webui_data:
```

### .env

```bash
# Ollama
OLLAMA_BASE_URL=http://ollama:11434

# Embeddings (local Ollama)
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# LLM (local Ollama)
LLM_API_KEY=ollama
LLM_URL=http://host.docker.internal:11434/v1/chat/completions
LLM_MODEL=llama3.1

# Remote GSRS MCP Server
MCP_GATEWAY_URL=https://gsrs-mcp.example.com
MCP_API_USERNAME=admin
MCP_API_PASSWORD=your-secure-password
```

### Open WebUI MCP Tool Configuration

In the `gsrs_tool.py` valves:

```python
mcp_transport = "sse"
mcp_url = "https://gsrs-mcp.example.com/mcp"
api_username = "admin"
api_password = "your-secure-password"
top_k = 20
answer_style = "standard"
```

### Ollama Setup

```bash
# Pull embedding model
ollama pull nomic-embed-text

# Pull LLM
ollama pull llama3.1
```

### Usage

1. Open **http://localhost:3000**
2. Select model: `llama3.1`
3. Enable the GSRS MCP tool
4. Ask: *"What is the CAS code for aspirin?"*
5. The LLM uses the remote MCP server to look up substance data
6. Returns an answer with citations from the GSRS database

## Tool Reference

### `gsrs_ask`

Full AI-answering pipeline. Returns an answer with citations.

```python
gsrs_ask(query="CAS code for aspirin", top_k=10, answer_style="standard")
```

### `gsrs_similarity_search`

Find substances similar to a GSRS JSON document.

```python
gsrs_similarity_search(
    substance_json='{"uuid": "...", "names": [{"name": "Aspirin"}]}',
    top_k=10
)
```

### `gsrs_retrieve`

Raw semantic search — returns text chunks without AI answer.

```python
gsrs_retrieve(query="aspirin molecular weight", top_k=5)
```

### `gsrs_ingest`

Ingest a substance.

```python
gsrs_ingest(substance_json='{"uuid": "...", "names": [...], "codes": [...], ...}')
```

### `gsrs_delete`

Delete a substance by UUID.

```python
gsrs_delete(substance_uuid="abc-123")
```

### `gsrs_health`

Check gateway status.

```python
gsrs_health()
```
