# Configuration Guide

Complete configuration reference for the GSRS MCP Server.

## Environment Variables

### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `chroma://./chroma_data/chunks` | Database connection URL (scheme determines backend) |

**Database URL Formats:**

```bash
# ChromaDB (local/embedded)
DATABASE_URL=chroma://./chroma_data/chunks

# PostgreSQL with pgvector
DATABASE_URL=postgresql://user:password@host:5432/gsrs_rag

# PostgreSQL with SSL
DATABASE_URL=postgresql://user:password@host:5432/gsrs_rag?sslmode=require
```

### Embedding API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_API_KEY` | - | API key for embedding service |
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | Base URL for embedding API |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_DIMENSION` | `1536` | Vector dimension (must match model) |

### LLM API Configuration (for /ask endpoint)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | `""` (empty) | API key; if empty, LLM-based answering is disabled |
| `LLM_URL` | `https://api.openai.com/v1/chat/completions` | Full URL to chat completions endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model name |
| `LLM_VERIFY_SSL` | `true` | SSL verification flag |
| `LLM_TIMEOUT` | `30` | HTTP timeout in seconds |

**LLM URL Examples:**

```bash
# OpenAI
LLM_URL=https://api.openai.com/v1/chat/completions

# Azure OpenAI (with API version in URL)
LLM_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2024-02-01

# Ollama (with OpenAI-compatible API)
LLM_URL=http://host.docker.internal:11434/v1/chat/completions

# LiteLLM proxy
LLM_URL=http://localhost:4000/chat/completions
```

> **Note:** `LLM_URL` must be the full path to the completions endpoint, not just a base URL. If `LLM_API_KEY` is empty, the `/ask` endpoint falls back to template-based answers.

### Authentication Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_USERNAME` | `admin` | HTTP Basic Auth username |
| `API_PASSWORD` | `admin123` | HTTP Basic Auth password |

### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |

### Vector Search Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_TOP_K` | `5` | Default number of search results |

## Example Configurations

### ChromaDB + OpenAI (Development)

```bash
# Database (local ChromaDB)
DATABASE_URL=chroma://./chroma_data/chunks

# Embeddings (OpenAI)
EMBEDDING_API_KEY=sk-your-openai-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# LLM for /ask endpoint (optional)
LLM_API_KEY=sk-your-openai-key
LLM_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-4o-mini

# Authentication
API_USERNAME=admin
API_PASSWORD=change-me-in-production
```

### PostgreSQL + OpenAI (Production)

```bash
# Database (PostgreSQL with pgvector)
DATABASE_URL=postgresql://gsrs:secure-password@postgres:5432/gsrs_rag

# Embeddings (OpenAI)
EMBEDDING_API_KEY=sk-your-openai-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Authentication
API_USERNAME=gsrs-admin
API_PASSWORD=very-secure-password
```

### Azure OpenAI

```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/gsrs_rag

# Embeddings (Azure OpenAI)
EMBEDDING_API_KEY=your-azure-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
```

### Ollama (Local/Offline)

```bash
# Database (local ChromaDB)
DATABASE_URL=chroma://./chroma_data/chunks

# Embeddings (Ollama)
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# LLM for /ask endpoint (Ollama with OpenAI-compatible API)
LLM_API_KEY=ollama
LLM_URL=http://localhost:11434/v1/chat/completions
LLM_MODEL=llama3.1

# Authentication
API_USERNAME=admin
API_PASSWORD=admin123
```

### Local Testing (No Auth)

For local testing without authentication, set simple credentials:

```bash
DATABASE_URL=chroma://./chroma_data/chunks
EMBEDDING_API_KEY=sk-test-key
API_USERNAME=test
API_PASSWORD=test
```

## Model Dimension Reference

### OpenAI Models

| Model | Dimension | Description |
|-------|-----------|-------------|
| `text-embedding-3-small` | 1536 | Fast and efficient (recommended) |
| `text-embedding-3-large` | 3072 | Highest quality |
| `text-embedding-ada-002` | 1536 | Legacy model |

### Ollama Models

| Model | Dimension | Description |
|-------|-----------|-------------|
| `nomic-embed-text` | 768 | Good balance (recommended) |
| `mxbai-embed-large` | 1024 | Large model |
| `all-minilm` | 384 | Small and fast |
| `snowflake-arctic-embed` | 1024 | High quality |

## Docker Compose Configuration

### Minimal Configuration

```yaml
services:
  rag-gateway:
    image: gsrs-rag-gateway:latest
    environment:
      - DATABASE_URL=chroma://./chroma_data/chunks
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY}
    ports:
      - "8000:8000"
```

### Full Production Configuration

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=gsrs
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=gsrs_rag
    volumes:
      - postgres_data:/var/lib/postgresql/data

  rag-gateway:
    image: gsrs-rag-gateway:latest
    environment:
      - DATABASE_URL=postgresql://gsrs:${POSTGRES_PASSWORD}@postgres:5432/gsrs_rag
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY}
      - EMBEDDING_MODEL=text-embedding-3-small
      - EMBEDDING_DIMENSION=1536
      - API_USERNAME=${API_USERNAME}
      - API_PASSWORD=${API_PASSWORD}
    ports:
      - "8000:8000"
    depends_on:
      - postgres

volumes:
  postgres_data:
```

### Using .env File

Create `.env` in project root:

```bash
# Database
POSTGRES_PASSWORD=secure-password
DATABASE_URL=postgresql://gsrs:secure-password@postgres:5432/gsrs_rag

# Embeddings
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Authentication
API_USERNAME=gsrs-admin
API_PASSWORD=very-secure-password
```

Docker Compose automatically loads these variables.

## Security Considerations

### Production Checklist

- [ ] Change default `API_USERNAME` and `API_PASSWORD`
- [ ] Use strong passwords (16+ characters)
- [ ] Store secrets in environment variables or secrets manager
- [ ] Enable HTTPS for API endpoint
- [ ] Restrict CORS origins if needed
- [ ] Use SSL for PostgreSQL connection

### Using Docker Secrets

```yaml
services:
  rag-gateway:
    secrets:
      - embedding_api_key
      - api_password

secrets:
  embedding_api_key:
    file: ./secrets/embedding_api_key.txt
  api_password:
    file: ./secrets/api_password.txt
```

Then in the container:

```bash
EMBEDDING_API_KEY=$(cat /run/secrets/embedding_api_key)
API_PASSWORD=$(cat /run/secrets/api_password)
```

## Performance Tuning

### ChromaDB Persistence

For better performance with ChromaDB:

```bash
# Use SSD for persist directory
DATABASE_URL=chroma:///ssd/chroma_data/chunks
```

### PostgreSQL Tuning

For pgvector:

```bash
# In postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 64MB

# HNSW index parameters
CREATE INDEX idx_embedding_hnsw
ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Batch Loading

For large data loads:

```bash
# Adjust batch size based on memory
python scripts/load_data.py data.gsrs --batch-size 100

# For systems with more memory
python scripts/load_data.py data.gsrs --batch-size 500
```

### Query Optimization

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "aspirin",
    "top_k": 5
  }'
```

## Troubleshooting Configuration

### Check Environment Variables

```bash
# Print current configuration
python -c "from app.config import settings; print(settings.database_url)"
```

### Validate Database URL

```bash
# ChromaDB URL
echo $DATABASE_URL
# Expected: chroma://./chroma_data/chunks

# PostgreSQL URL
echo $DATABASE_URL
# Expected: postgresql://user:pass@host:5432/dbname
```

### Test Embedding Configuration

```bash
# Test OpenAI connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $EMBEDDING_API_KEY"

# Test Ollama connection
curl http://localhost:11434/api/tags
```

## Related Documentation

- [Quick Start](quickstart.md) - Get started quickly
- [Data Loading](data-loading.md) - Load substances
- [Troubleshooting](troubleshooting.md) - Common issues
