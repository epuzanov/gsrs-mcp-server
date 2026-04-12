# Quick Start Guide

Get the GSRS MCP Server up and running in minutes.

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)
- API key for embeddings (OpenAI or Ollama)

## Option 1: Quick Start with ChromaDB (Recommended for Testing)

No Docker required! Uses ChromaDB for local vector storage.

```bash
# 1. Clone repository
cd gsrs-rag-gateway

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env:
#   DATABASE_URL=chroma://./chroma_data/chunks
#   EMBEDDING_API_KEY=sk-your-key-here

# 5. Set API key
export EMBEDDING_API_KEY="sk-your-openai-key"

# 6. Start the server
uvicorn app.main:app --reload

# 7. Verify
curl http://localhost:8000/health
```

## Option 2: Production with PostgreSQL + pgvector (Docker)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env:
#   DATABASE_URL=postgresql://postgres:postgres@postgres:5432/gsrs_rag
#   EMBEDDING_API_KEY=sk-your-key-here

# 2. Start all services
docker-compose up -d

# 3. Verify
curl http://localhost:8000/health
```

## Option 3: Local Ollama (Offline/Private)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull embedding model
ollama pull nomic-embed-text

# 3. Configure environment
cp .env.example .env
# Edit .env:
#   DATABASE_URL=chroma://./chroma_data/chunks
#   EMBEDDING_BASE_URL=http://localhost:11434/v1
#   EMBEDDING_MODEL=nomic-embed-text
#   EMBEDDING_DIMENSION=768

# 4. Start the server
uvicorn app.main:app --reload
```

## Load Sample Data

### Ingest into MCP Server

```bash
# Use the load script for bulk loading
python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9
```

### Check Statistics

```bash
curl http://localhost:8000/statistics
```

## Make Your First Query

```bash
curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{
        "query": "CAS code for Ibuprofen",
        "top_k": 3
    }'
```

Example response:
```json
{
    "query": "CAS code for Ibuprofen",
    "results": [
        {
            "element_path": "codes_cas_15687-27-1",
            "substance_uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
            "chunk_text": "Code: 15687-27-1 | System: CAS | Type: PRIMARY",
            "similarity_score": 0.92
        }
    ],
    "total_results": 1
}
```

## Access the API Documentation

Open in browser:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Authentication

The gateway uses HTTP Basic Authentication:

```bash
# Default credentials (change in production!)
# Username: admin
# Password: admin123

# Ingest with authentication
curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -u admin:admin123 \
    -d @substance.json

# Configure in .env:
#   API_USERNAME=admin
#   API_PASSWORD=admin123
```

## Next Steps

### Learn More

- [Data Loading Guide](data-loading.md) - How to load substances
- [Chunking Guide](guides/chunking.md) - How chunking works
- [Configuration](configuration.md) - Advanced settings
- [Troubleshooting](troubleshooting.md) - Common issues

### Integrations

- [ChatGPT Integration](guides/chatgpt.md) - Use with ChatGPT
- [Ollama + Open WebUI](guides/ollama-open-webui.md) - Local LLM interface

### Production Deployment

```bash
# Use pgvector for production
DATABASE_URL=postgresql://user:pass@host:5432/gsrs_rag

# Set strong credentials
API_USERNAME=your-username
API_PASSWORD=your-secure-password

# Configure for your embedding provider
EMBEDDING_API_KEY=sk-...
EMBEDDING_BASE_URL=https://api.openai.com/v1
```

## Common Commands

```bash
# Health check
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/statistics

# List substance classes
curl http://localhost:8000/substance-classes

# Query
curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "aspirin", "top_k": 5}'

# Delete a substance (requires auth)
curl -X DELETE http://localhost:8000/substances/0103a288-6eb6-4ced-b13a-849cd7edf028 \
    -u admin:admin123
```

## Environment Variables

Minimal configuration:

```bash
# Database (ChromaDB by default)
DATABASE_URL=chroma://./chroma_data/chunks

# Embeddings (OpenAI)
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Authentication
API_USERNAME=admin
API_PASSWORD=admin123
```

See [Configuration](configuration.md) for all options.
