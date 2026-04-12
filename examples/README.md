# GSRS MCP Server — Open WebUI Integration

This directory contains integration scripts for using the GSRS MCP Server with Open WebUI.

## Overview

The GSRS MCP server provides these tools via the Model Context Protocol:

| Tool | Purpose | Returns |
|------|---------|---------|
| `gsrs_ask` | Full AI answer with citations | Answer + evidence |
| `gsrs_similarity_search` | Find similar substances by JSON | Ranked substances |
| `gsrs_retrieve` | Raw semantic retrieval | Text chunks only |

## Files

### `gsrs_tool.py` — Open WebUI Tool Plugin

A tool that Open WebUI users can call directly in conversations.

**Features:**
- Uses MCP server (`gsrs_ask`, `gsrs_similarity_search`)
- Auto-detects uploaded GSRS JSON files for similarity search
- Auto-detects pasted GSRS JSON in the query
- Supports SSE and stdio transport
- Falls back to direct REST API for backward compatibility

**Valves:**
| Valve | Default | Description |
|-------|---------|-------------|
| `mcp_transport` | `sse` | `sse` (HTTP) or `stdio` (subprocess) |
| `mcp_url` | `http://gsrs-mcp-server:8000/mcp` | MCP server SSE URL |
| `mcp_command` | `gsrs-gateway` | MCP command for stdio mode |
| `api_username` | `admin` | Authentication username |
| `api_password` | `admin123` | Authentication password |
| `top_k` | `20` | Results to retrieve |
| `answer_style` | `standard` | `concise`, `standard`, `detailed` |
| `return_evidence` | `true` | Include evidence chunks |
| `evidence_limit` | `5` | Max evidence chunks to show |

**Usage:**
1. Import `gsrs_tool.py` into Open WebUI as a custom tool
2. Configure valves to match your MCP server
3. Users call `gsrs_substance_query("CAS code for aspirin")`

---

### `gsrs_function.py` — Open WebUI Filter Plugin

A filter that automatically injects GSRS context into user messages.

**Modes:**
- `evidence` — Injects evidence chunks only
- `answer_assist` — Injects draft answer + evidence

**Valves:**
| Valve | Default | Description |
|-------|---------|-------------|
| `mcp_url` | `http://gsrs-mcp-server:8000` | MCP server base URL |
| `api_username` | `admin` | Authentication username |
| `api_password` | `admin123` | Authentication password |
| `mode` | `evidence` | `evidence` or `answer_assist` |
| `top_k` | `10` | Results to retrieve |
| `max_evidence_chars` | `3000` | Max chars for evidence |
| `min_query_length` | `5` | Min query length to trigger lookup |

---

### `gsrs_system_prompt.md` — Full System Prompt

Comprehensive system prompt for GSRS-aware assistant behavior.

### `gsrs_system_prompt_minimal.md` — Minimal System Prompt

Short version: *"Use the GSRS MCP server for GSRS substance questions..."*

---

## Remote MCP Server Configuration (Open WebUI + Ollama)

Run Ollama locally for embeddings/LLM, connect to a **remote** GSRS MCP server:

### Architecture
```
User → Open WebUI → Ollama (local) → GSRS MCP server (remote, SSE)
```

### Environment
```bash
# Embeddings (local Ollama)
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# LLM (local Ollama)
LLM_API_KEY=ollama
LLM_URL=http://host.docker.internal:11434/v1/chat/completions
LLM_MODEL=llama3.1

# Remote GSRS MCP server
MCP_GATEWAY_URL=https://gsrs-mcp.example.com
MCP_API_USERNAME=admin
MCP_API_PASSWORD=your-secure-password
```

### Tool Valves
```python
mcp_transport = "sse"
mcp_url = "https://gsrs-mcp.example.com/mcp"
api_username = "admin"
api_password = "your-secure-password"
```

## Compatibility Notes

- The MCP server works without an LLM (template-based answers)
- If `LLM_API_KEY` is empty, template answers are used
- Both pgvector and ChromaDB backends are supported
- pgvector provides better lexical retrieval via PostgreSQL FTS
- **Aggregation queries** (e.g., "How many identifiers has Ibuprofen?") are auto-detected
- **Similarity search** triggers automatically when GSRS JSON is pasted or uploaded

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection error" | Verify `mcp_url` is correct and MCP server is running |
| "Error querying" | Check MCP server logs; verify auth credentials |
| Abstained response | Evidence was weak; try rephrasing or increasing `top_k` |
| Filter not injecting | Check `min_query_length` — short queries are skipped |
| 404 on MCP endpoint | Ensure `mcp_url` ends with `/mcp` for SSE transport |
| Aggregation incomplete | Increase `top_k` to retrieve more chunks |
