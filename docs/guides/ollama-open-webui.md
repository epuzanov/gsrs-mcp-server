# Ollama + Open WebUI Integration Guide with MCP Server

This guide shows you how to set up a **local** LLM environment with Ollama and Open WebUI, integrated with the GSRS MCP Server for answering questions about chemical substances.

## Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│  Open WebUI  │────▶│  MCP Server  │────▶│  pgvector   │
│  Browser    │     │   (Frontend) │     │   (API)      │     │  Database   │
└─────────────┘     └───────┬──────┘     └─────────────┘     └─────────────┘
                            │                                     ▲
                            │ MCP Tool                            │
                            ▼                                     │
                     ┌──────────────┐     ┌─────────────┐         │
                     │    Ollama    │────▶│  MCP Tool   │─────────┘
                     │  (Qwen3.5)   │     │  (Function) │
                     └──────────────┘     └─────────────┘
```

**Benefits:**
- ✅ **100% Local** - No API costs, full privacy
- ✅ **Qwen3.5 Models** - Latest powerful open-source LLMs from Alibaba
- ✅ **MCP Tools** - Open WebUI integration via MCP protocol
- ✅ **Web Interface** - User-friendly chat interface
- ✅ **Multiple Models** - Easy model switching
- ✅ **Similarity Search** - Find substances by JSON document
- ✅ **RAG Enhanced** - Accurate answers from GSRS database
- ✅ **GSRS API Tools** - Direct access to official GSRS database

## Prerequisites

1. **Docker** and **Docker Compose**
2. **Ollama** installed locally
3. **GSRS MCP Server** repository

## Step 1: Install Ollama

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### macOS

```bash
brew install ollama
```

### Windows

Download from [ollama.ai](https://ollama.ai)

### Verify Installation

```bash
ollama --version
```

## Step 2: Pull Qwen3.5 Models

```bash
# Qwen3.5 Chat model for answering questions (recommended: 7B or 14B)
ollama pull qwen3.5:7b

# Qwen3.5 Embedding model for RAG (recommended)
ollama pull qwen3-embedding:latest

# Alternative embedding models
# ollama pull nomic-embed-text
# ollama pull mxbai-embed-large

# Verify models
ollama list
```

**Recommended Qwen3.5 Models:**

| Purpose | Model | Size | Dimension | Command |
|---------|-------|------|-----------|---------|
| Chat | `qwen3.5:7b` | 7B | - | `ollama pull qwen3.5:7b` |
| Chat | `qwen3.5:14b` | 14B | - | `ollama pull qwen3.5:14b` |
| Chat | `qwen3.5:32b` | 32B | - | `ollama pull qwen3.5:32b` |
| Chat | `qwen3.5-coder:7b` | 7B | - | `ollama pull qwen3.5-coder:7b` |
| Embedding | `qwen3-embedding:latest` | ~7B | 1024 | `ollama pull qwen3-embedding` |
| Embedding | `nomic-embed-text` | 270MB | 768 | `ollama pull nomic-embed-text` |
| Embedding | `mxbai-embed-large` | 670MB | 1024 | `ollama pull mxbai-embed-large` |

### Using Qwen3.5 Embedding Model

**Yes, you can use `qwen3-embedding` as the embedding model!** This is the recommended embedding model for Qwen3.5 setups.

**Configuration for Qwen3.5 Embedding:**

```bash
# In your .env file:
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=qwen3-embedding:latest
EMBEDDING_DIMENSION=1024
```

**Benefits of Qwen3.5 Embedding:**
- **Higher dimension** (1024 vs 768) - Better semantic representation
- **Multi-language support** - Optimized for 100+ languages
- **Better alignment** with Qwen3.5 chat models
- **State-of-the-art** performance on MTEB benchmark

**Note:** Qwen3.5 embedding model is larger (~7B parameters) and requires more VRAM (~14 GB) compared to nomic-embed-text (~270MB). For resource-constrained systems, consider using `nomic-embed-text` instead.

### Model Selection Guide

| Use Case | Chat Model | Embedding Model | Total VRAM |
|----------|------------|-----------------|------------|
| Low VRAM | `qwen3.5:3b` | `nomic-embed-text` | ~4 GB |
| Balanced | `qwen3.5:7b` | `nomic-embed-text` | ~8 GB |
| Best Quality | `qwen3.5:14b` | `qwen3-embedding` | ~24 GB |
| Maximum | `qwen3.5:32b` | `qwen3-embedding` | ~34 GB |

## Step 3: Configure the MCP Server

Create or edit `.env` file:

```bash
# Use Qwen embeddings via Ollama
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# Database (ChromaDB for local development)
DATABASE_URL=chroma://./chroma_data/chunks

# Authentication (for ERI)
API_USERNAME=admin
API_PASSWORD=admin123
```

## Step 4: Start All Services

```bash
# Navigate to project
cd gsrs-rag-gateway

# Start MCP Server with Open WebUI
docker-compose --profile ollama up -d

# Verify services
docker-compose ps
```

**Services started:**
- `gsrs-postgres` - PostgreSQL with pgvector (or ChromaDB for local)
- `gsrs-rag-gateway` - MCP Server API
- `gsrs-open-webui` - Open WebUI interface

## Step 5: Load Substance Data

### Option A: Load Sample Substances from GSRS Server (Recommended)

```bash
# Load specific substances by UUID (fast, recommended for testing)
python scripts/load_data.py \
  --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9

# Verify data is loaded
curl -u admin:admin123 http://localhost:8000/statistics
```

### Option B: Load All Substances from GSRS Server

```bash
# Load first 100 substances (may take 10-30 minutes)
python scripts/load_data.py --all --max-results 100
```

### Option C: Load from .gsrs File

If you have substance data in `.gsrs` format:

```bash
python scripts/load_data.py data/substances.gsrs --batch-size 100
```

## Step 6: Integrate MCP Server with Open WebUI

Open WebUI provides **two native ways** to integrate external MCP Servers.
Both methods let the LLM call GSRS tools automatically during chat.

### Authentication

The MCP Server supports **Bearer token authentication**. Configure it in `.env`:

```bash
# Set API credentials (used for Bearer token validation)
API_USERNAME=admin
API_PASSWORD=your_secure_token_here
```

When connecting to the MCP Server, include the token in the `Authorization` header:

```
Authorization: Bearer your_secure_token_here
```

> **Note:** If `API_USERNAME` and `API_PASSWORD` are set, authentication is required.
> To disable authentication, leave both empty (not recommended for production).

### Method 1: Admin Panel → Settings → Integrations → "Manage Tool Servers"

This method connects Open WebUI to the MCP Server via the SSE protocol.
All 13 GSRS tools become instantly available to every model.

#### 1. Ensure MCP Server is Running

```bash
cd gsrs-rag-gateway
docker-compose up -d mcp-server
# Verify
curl http://localhost:8000/health
```

#### 2. Open Admin Panel

1. Open **http://localhost:3000** in your browser
2. Sign in (create an account if first time)
3. Click your **profile icon** (top-right) → **Admin Panel**

#### 3. Add MCP Server

1. Go to **Settings** → **Integrations**
2. Click **"Manage Tool Servers"**
3. Click **"Add Server"**
4. Fill in the connection details:

| Field | Value |
|-------|-------|
| **Server URL** | `http://host.docker.internal:8000/mcp` |
| **Server Name** | `GSRS MCP Server` |

> **Note:** The `/mcp` path uses the **Streamable HTTP** transport (JSON-RPC over HTTP POST).
> On Linux, you may need to use `http://gsrs-mcp-server:8000/mcp` instead
> (the Docker service name).

5. Click **"Connect"**

#### 4. Verify Connection

- Status should show **✅ Connected**
- Open WebUI lists all available tools:
  - `gsrs_ask` — Full AI answering pipeline with citations
  - `gsrs_similarity_search` — Find substances by JSON document
  - `gsrs_retrieve` — Semantic chunk retrieval
  - `gsrs_ingest` — Ingest a GSRS substance JSON
  - `gsrs_delete` — Delete substance chunks
  - `gsrs_health` — Gateway health and statistics
  - `gsrs_statistics` — Database statistics
  - `gsrs_aggregation` — Count identifiers, names, relationships
  - `gsrs_query_optimizer` — Rewrite, translate, optimise queries
  - `gsrs_get_document` — Fetch full GSRS JSON by UUID
  - `gsrs_api_search` — GSRS API text search
  - `gsrs_api_structure_search` — GSRS API chemical structure search
  - `gsrs_api_sequence_search` — GSRS API biological sequence search

#### 5. Use in Chat

1. Start a **new chat** with any model (e.g. `qwen3.5:7b`)
2. The MCP tools are **automatically available** — the LLM decides when to call them
3. Example prompts:

| Prompt | Tool Used |
|--------|-----------|
| *"What is the CAS code for Aspirin?"* | `gsrs_ask` |
| *"How many identifiers has Ibuprofen?"* | `gsrs_aggregation` |
| *"Show me substances similar to this JSON: {...}"* | `gsrs_similarity_search` |
| *"Search the GSRS API for 'Paracetamol'"* | `gsrs_api_search` |
| *"Find substances with SMILES CC(=O)O"* | `gsrs_api_structure_search` |
| *"Fetch the full document for UUID abc-123"* | `gsrs_get_document` |

#### 6. Configure Tool Valves (Optional)

Some tools have configurable parameters (valves):

1. In the chat, click the **tool icon** in the input area
2. Adjust valves like `top_k`, `answer_style`, etc.
3. Changes apply immediately

---

### Method 2: Workspace → Tools

This method uses Open WebUI's built-in **Tool Server** management from the
Workspace sidebar. It provides a visual interface to enable, disable, and
configure MCP tools per chat session.

#### 1. Open Workspace

1. Open **http://localhost:3000**
2. In the left sidebar, click **Workspace** (folder icon)
3. Click **Tools**

#### 2. Connect MCP Server

1. Click **"Add Tool Server"** or the **"+"** button
2. Select **"MCP Server"** as the type
3. Enter the connection details:

| Field | Value |
|-------|-------|
| **URL** | `http://host.docker.internal:8000/mcp` |
| **Name** | `GSRS MCP Server` |

> **Note:** The `/mcp` path uses the **Streamable HTTP** transport.

4. Click **"Connect"**

#### 3. Browse and Enable Tools

Open WebUI displays all available GSRS tools in a list. You can:

- **Toggle individual tools** ON/OFF per session
- **View tool descriptions** by hovering over the info icon
- **Configure default parameters** by clicking the settings gear

#### 4. Use Tools in Chat

1. Open or start a **new chat**
2. In the input area, click the **tools icon** (wrench or puzzle piece)
3. You'll see all enabled GSRS tools listed
4. The LLM will automatically call the appropriate tool based on your query

#### 5. Example Workflow

```
User: "Find substances similar to this GSRS JSON document:
       {\"uuid\":\"abc-123\",\"names\":[{\"name\":\"Test\"}]}"

→ LLM calls gsrs_similarity_search automatically
→ Returns ranked similar substances with match scores

User: "How many identifiers does Aspirin have?"

→ LLM calls gsrs_aggregation with aggregation_type=count
→ Returns: "Aspirin has 15 identifiers."

User: "Search the official GSRS API for 'Metformin'"

→ LLM calls gsrs_api_search
→ Returns list of matching substances with UUIDs
```

---

### Method 3: Custom Python Tool (Fallback / Legacy)

If your Open WebUI version doesn't support MCP Server integration,
use a custom Python tool script.

#### 1. Create Tool File

The tool file is provided in the examples directory: `examples/gsrs_tool.py`

**Tool Features:**
- Queries the GSRS MCP Server `/ask` endpoint
- Supports GSRS JSON similarity search
- Returns formatted search results with citations
- Configurable via valves (base URL, credentials, top_k)
- Error handling for timeouts and connection issues

#### 2. Mount Tool File in Docker

Update `docker-compose.yaml` to mount the tool file:

```yaml
services:
  open-webui:
    volumes:
      - ./examples/gsrs_tool.py:/app/backend/tools/gsrs_tool.py
```

#### 3. Restart Open WebUI

```bash
docker-compose restart open-webui
```

#### 4. Enable Tool in Open WebUI

1. Open **http://localhost:3000**
2. Go to **Workspace** → **Tools**
3. Find **GSRS MCP Server Tool** in the list
4. Click to enable it
5. Configure valves if needed:
   - **mcp_url**: `http://gsrs-rag-gateway:8000`
   - **api_username**: `admin`
   - **api_password**: `admin123`
   - **top_k**: `5`

#### 5. Use Tool in Chat

1. Start a new chat with Qwen model
2. The GSRS tool will be automatically available
3. Ask questions like:
   - "What is the CAS code for Aspirin?"
   - "Show me the molecular formula of Ibuprofen"
   - "What is the UNII code for Paracetamol?"

---

## Troubleshooting

### MCP Server Not Connecting

```bash
# Check MCP Server logs
docker-compose logs mcp-server

# Verify health endpoint
curl http://localhost:8000/health
```

### Tools Not Showing in Open WebUI

1. Ensure MCP Server is running: `docker-compose ps`
2. Check network connectivity: Open WebUI must reach `http://host.docker.internal:8000/mcp`
3. On Linux, try `http://gsrs-rag-gateway:8000/mcp` instead
4. Restart Open WebUI: `docker-compose restart open-webui`

### Tool Calls Fail

- Verify substances are loaded: `curl -u admin:admin123 http://localhost:8000/statistics`
- Check LLM can see tools: The model must support function/tool calling
- Try a simpler prompt first: "What is the CAS code for Aspirin?"

---

## Next Steps

- [ChatGPT Integration](chatgpt.md) - Cloud-based alternative
- [API Reference](../api-reference.md) - Complete API documentation
- [Chunking Guide](chunking.md) - Understand how substances are chunked
- [Data Loading](../data-loading.md) - Load your own substance data

## Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Open WebUI Documentation](https://docs.openwebui.com)
- [GSRS MCP Server API](http://localhost:8000/docs)
- [ChatGPT Integration](chatgpt.md) - Cloud-based alternative
- [API Reference](../api-reference.md) - Complete API documentation
- [Chunking Guide](chunking.md) - Understand how substances are chunked
- [Data Loading](../data-loading.md) - Load your own substance data
