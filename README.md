# GSRS MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/epuzanov/gsrs-rag-gateway/actions/workflows/tests.yml/badge.svg)](https://github.com/epuzanov/gsrs-rag-gateway/actions)

MCP (Model Context Protocol) server for GSRS (Global Substance Registration System) substances with **pgvector** or **ChromaDB** as vector database.

## Features

- 🧩 **Intelligent Chunking**: Automatic splitting of GSRS Substance JSON documents into element-based chunks
- 🔍 **Vector Search**: Semantic search with pgvector (Production) or ChromaDB (Development)
- 🤖 **MCP Tools**: `gsrs_ask`, `gsrs_similarity_search`, `gsrs_retrieve`, `gsrs_ingest`, `gsrs_delete`
- 🎯 **Element Path IDs**: Unique IDs for chunks based on element paths
- 📊 **Metadata Retention**: Complete metadata for each element in embeddings
- 🔄 **SubstanceClass Filter**: Filtering by substance type (chemical, protein, nucleicAcid, etc.)
- 🎨 **Embedding Provider**: OpenAI API, Azure OpenAI, Ollama and OpenAI-compatible APIs
- 🔐 **Authentication**: HTTP Basic Auth and API Key support
- 🗄️ **Multi-Backend**: pgvector (PostgreSQL) or ChromaDB (local, serverless)
- 🐳 **Docker Deployment**: Easy deployment with Docker Compose
- 📥 **Bulk Loading**: Loading script for JSONL files
- ✅ **Unit Tests**: Complete test coverage with pytest

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   GSRS JSON     │────▶│  ChunkerService  │────▶│ EmbeddingService│
│   (Substance)   │     │  (gsrs.model)    │     │ (OpenAI/Ollama) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Vector Database Backend                          │
│  ┌─────────────────┐                   ┌─────────────────┐          │
│  │   pgvector      │  (Production)     │    ChromaDB     │          │
│  │  (PostgreSQL)   │◀─────────────────▶│  (Development)  │          │
│  └─────────────────┘                   └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                            │
│  ┌─────────────────┐                   ┌─────────────────┐          │
│  │  /ingest        │                   │   /query        │          │
│  │  /ingest/batch  │◀───────┬─────────▶│   /statistics   │          │
│  │  /substances/*  │        │          │   /health       │          │
│  └─────────────────┘        │          └─────────────────┘          │
│                             │                                        │
│  ┌─────────────────┐        │                                        │
│  │  Auth Layer     │◀───────┘                                        │
│  │  (Basic/API Key)│                                                 │
│  └─────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Production with PostgreSQL + pgvector (Docker)

```bash
# Create .env file and configure
cp .env.example .env
# Edit .env and set:
#   DATABASE_URL=postgresql://gsrs:your_password@postgres:5432/gsrs_rag
#   EMBEDDING_API_KEY=sk-your-key

# Start all services (PostgreSQL + MCP Server)
docker-compose --profile postgres up -d

# With Open WebUI (for Ollama integration)
docker-compose --profile postgres --profile ollama up -d
```

### Option 2: Development with ChromaDB (Docker)

```bash
# Create .env file
cp .env.example .env
# DATABASE_URL is already preconfigured for ChromaDB

# Start ChromaDB + MCP Server
docker-compose --profile chroma up -d
```

### Option 3: Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Set EMBEDDING_API_KEY
export EMBEDDING_API_KEY="sk-your-key"

# Start gateway
uvicorn app.main:app --reload
```

### Authentication

```bash
# Default admin user
# Username: admin
# Password: admin123 (or change API_PASSWORD in .env)

# Use API with authentication
curl -u admin:admin123 http://localhost:8000/health
```

### Check Health

```bash
curl http://localhost:8000/health
```

### Load Sample Data

```bash
# Load substances from GSRS server
python scripts/load_data.py \
  --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9

# Check statistics
curl -u admin:admin123 http://localhost:8000/statistics
```

## API Endpoints

### Health Check

```bash
GET /health
```

### Ingest Substance

```bash
POST /ingest
Content-Type: application/json
Authorization: Basic YWRtaW46YWRtaW4xMjM=  # admin:admin123

{
    "substance": { /* GSRS Substance JSON */ }
}
```

### Batch Ingest

```bash
POST /ingest/batch
Content-Type: application/json

{
    "substances": [ /* Array of GSRS Substance JSON */ ]
}
```

### Semantic Search

```bash
POST /query
Content-Type: application/json

{
    "query": "CAS code for Aspirin",
    "top_k": 5,
    "filters": {}  // optional metadata filters
}
```

### Delete Substance

```bash
DELETE /substances/{substance_uuid}
Authorization: Basic YWRtaW46YWRtaW4xMjM=
```

### Available Embedding Models

```bash
GET /models
```

### Substance Classes

```bash
GET /substance-classes
```

### Statistics

```bash
GET /statistics
```

## Configuration

Environment variables (`.env` file):

```bash
# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# Database URL - Schema determines backend automatically:
# - PostgreSQL: postgresql://user:pass@host:port/dbname
# - ChromaDB: chroma://./chroma_data/chunks

# For ChromaDB (Development/Testing - Default):
DATABASE_URL=chroma://./chroma_data/chunks

# For PostgreSQL (Production - uncomment):
# DATABASE_URL=postgresql://gsrs:your_secure_password@localhost:5432/gsrs_rag

# =============================================================================
# EMBEDDING API CONFIGURATION
# =============================================================================
# Works with OpenAI, Azure OpenAI, Ollama and OpenAI-compatible APIs

# OpenAI (Production):
EMBEDDING_API_KEY=sk-your-api-key-here
EMBEDDING_BASE_URL=https://api.openai.com/v1

# Azure OpenAI (uncomment for Azure):
# EMBEDDING_API_KEY=your-azure-key
# EMBEDDING_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment

# Ollama (Local/Development - uncomment for local embeddings):
# EMBEDDING_API_KEY=ollama
# EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================
# OpenAI models:
#   - text-embedding-3-small (1536 dim, recommended)
#   - text-embedding-3-large (3072 dim, highest quality)
#   - text-embedding-ada-002 (1536 dim, legacy)

# Ollama models:
#   - nomic-embed-text (768 dim, lightweight)
#   - mxbai-embed-large (1024 dim, high quality)
#   - qwen3-embedding:latest (1024 dim, high quality)
#   - all-minilm (384 dim, smallest)

EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# For Ollama (uncomment for local embeddings):
# EMBEDDING_MODEL=nomic-embed-text
# EMBEDDING_DIMENSION=768

# =============================================================================
# AUTHENTICATION CONFIGURATION (HTTP Basic Auth)
# =============================================================================
# Change in Production!
API_USERNAME=admin
API_PASSWORD=admin123

# =============================================================================
# API CONFIGURATION
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
DEFAULT_TOP_K=5
```

### Embedding Providers

#### OpenAI (and compatible APIs)

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

**Supported models:**
- `text-embedding-3-small` (1536 dim) - Fast and efficient
- `text-embedding-3-large` (3072 dim) - Highest quality
- `text-embedding-ada-002` (1536 dim) - Legacy

**Azure OpenAI:**
```bash
EMBEDDING_API_KEY=your-azure-key
EMBEDDING_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
```

#### Ollama (Local Models)

```bash
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
EMBEDDING_BASE_URL=http://localhost:11434/v1
```

**Supported models:**
- `nomic-embed-text` (768 dim)
- `mxbai-embed-large` (1024 dim)
- `qwen3-embedding:latest` (1024 dim, high quality)
- `all-minilm` (384 dim, smallest)
- And all other Ollama embedding models

## Loading Data

### From JSON Files

```bash
curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -u admin:admin123 \
    -d @substance.json
```

### From .gsrs Files (JSONL.gz)

```bash
python scripts/load_data.py data/substances.gsrs --batch-size 100
```

### Open WebUI Integration

```bash
# Start with Open WebUI profile
docker-compose --profile ollama up -d

# Open Open WebUI in browser
# http://localhost:3000
```

## Chunking Strategy

Each GSRS substance document is split into chunks based on element paths:

### Example

**Input JSON:**
```json
{
    "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
    "substanceClass": "chemical",
    "codes": [
        {
            "code": "WK2XYI10QM",
            "codeSystem": "FDA UNII"
        },
        {
            "code": "CHEMBL521",
            "codeSystem": "ChEMBL"
        }
    ]
}
```

**Created Chunks:**
| Element Path | Chunk Text | Metadata |
|--------------|------------|----------|
| `root_codes_0_code` | code: WK2XYI10QM | {codeSystem: FDA UNII} |
| `root_codes_0_codeSystem` | codeSystem: FDA UNII | {} |
| `root_codes_1_code` | code: CHEMBL521 | {codeSystem: ChEMBL} |
| `root_codes_1_codeSystem` | codeSystem: ChEMBL | {} |

## Development

### Local Development without Docker

```bash
# Install PostgreSQL with pgvector
# Ubuntu: sudo apt install postgresql-16-pgvector

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Set OPENAI_API_KEY (for OpenAI embeddings)
export OPENAI_API_KEY="sk-..."

# Create database
createdb -U postgres gsrs_rag

# Start app
uvicorn app.main:app --reload
```

### Tests

```bash
# Unit tests for Vector Database Backends
python -m pytest tests/ -v

# Only ChromaDB tests
python -m pytest tests/test_vector_db.py -v

# Chunking tests
python -m pytest tests/test_chunking.py -v
```

## Project Structure

```
gsrs-rag-gateway/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── main.py                # FastAPI app
│   ├── models/
│   |   ├── __init__.py
│   │   ├── api.py             # API models
│   │   └── db.py              # SQLAlchemy models
│   ├── db/
│   |   ├── __init__.py
│   │   ├── base.py            # Vector Database Interface
│   │   ├── factory.py         # Backend Factory
│   │   └── backends/
│   |       ├── __init__.py
│   │       ├── chroma.py      # ChromaDB Backend
│   │       └── pgvector.py    # pgvector Backend
│   └── services/
│       ├── __init__.py
│       ├── chunking.py        # ChunkerService
│       ├── embedding.py       # EmbeddingService
│       └── vector_database.py # VectorDatabaseService
├── scripts/
│   └──load_data.py            # Loading script for .gsrs files
├── examples/
│   ├── gsrs_function.py       # ollama function script
│   └── gsrs_tool.py           # ollama tools script
├── tests/
│   ├── test_chunking.py
│   ├── test_load_data.py
│   └── test_vector_db.py
├── docs/
│   ├── api-reference.md
│   ├── authentication.md
│   ├── configuration.md
│   ├── data-loading.md
│   ├── quickstart.md
│   ├── README.md
│   ├── troubleshooting.md
│   ├── vector-databases.md
│   └── guides/
│       ├── chunking.md
│       ├── chatgpt.md
│       └── ollama-open-webui.md
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CONTRIBUTING.md
├── docker-compose.yaml
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-examples.txt
└── requirements.txt
```

## API Documentation

Full API documentation is available in Swagger UI:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Database Connection Failed

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready -U gsrs
```

### Embedding API Errors (OpenAI)

```bash
# Check API key
echo $EMBEDDING_API_KEY

# Test API availability
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $EMBEDDING_API_KEY"
```

### Embedding API Errors (Ollama)

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull model
ollama pull nomic-embed-text
```

### Vector Search Returns No Results

```bash
# Check if data is loaded
curl http://localhost:8000/statistics

# Check substance classes
curl http://localhost:8000/substance-classes
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please create an issue or pull request for improvements.

## Links

- [GitHub Repository](https://github.com/epuzanov/gsrs-rag-gateway)
- [GSRS Model Library](https://github.com/epuzanov/gsrs.model)
- [pgvector](https://github.com/pgvector/pgvector)
- [ChromaDB](https://docs.trychroma.com/)
