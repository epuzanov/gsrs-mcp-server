# Troubleshooting Guide

Common issues and solutions for the GSRS MCP Server.

## Service Issues

### Gateway Won't Start

**Symptoms:**
```bash
docker-compose up -d
# Container exits immediately
```

**Solutions:**

1. Check logs:
```bash
docker-compose logs rag-gateway
```

2. Verify database connection:
```bash
docker-compose logs postgres
```

3. Check environment variables:
```bash
docker-compose exec rag-gateway env | grep EMBEDDING
```

### Database Connection Failed

**Symptoms:**
```json
{
  "status": "unhealthy",
  "database_connected": false
}
```

**Solutions:**

1. Check if PostgreSQL is running:
```bash
docker-compose ps postgres
```

2. Test connection:
```bash
docker-compose exec postgres pg_isready -U postgres
```

3. Check credentials:
```bash
# In .env file
POSTGRES_HOST=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

4. Restart database:
```bash
docker-compose restart postgres
```

## Embedding Issues

### OpenAI API Key Error

**Symptoms:**
```
Error: API key is required for OpenAI provider
```

**Solutions:**

1. Set API key:
```bash
export OPENAI_API_KEY="sk-..."
```

2. Verify key in .env:
```bash
grep OPENAI_API_KEY .env
```

3. Test API key:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Ollama Connection Refused

**Symptoms:**
```
Error: Connection refused to http://localhost:11434
```

**Solutions:**

1. Check if Ollama is running:
```bash
ollama list
```

2. Start Ollama:
```bash
ollama serve
```

3. Pull required model:
```bash
ollama pull nomic-embed-text
```

4. Docker network configuration:

**Linux:**
```bash
OLLAMA_BASE_URL=http://172.17.0.1:11434
```

**macOS/Windows:**
```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Embedding Dimension Mismatch

**Symptoms:**
```
Error: vector dimension 384 does not match column dimension 1536
```

**Solutions:**

1. Check configured dimension:
```bash
grep EMBEDDING_DIMENSION .env
```

2. Verify model dimension:
- `text-embedding-3-small`: 1536
- `text-embedding-3-large`: 3072
- `nomic-embed-text`: 768

3. Fix and recreate database:
```bash
# Stop services
docker-compose down

# Clear database (WARNING: deletes all data)
docker-compose down -v

# Update .env with correct dimension
# EMBEDDING_DIMENSION=1536 (for text-embedding-3-small)

# Restart
docker-compose up -d
```

## Query Issues

### No Results Found

**Symptoms:**
```json
{
  "query": "aspirin",
  "results": [],
  "total_results": 0
}
```

**Solutions:**

1. Check if data is loaded:
```bash
curl http://localhost:8000/statistics
```

2. Verify substance classes:
```bash
curl http://localhost:8000/substance-classes
```

3. Try different query:
```bash
# More specific
curl -X POST http://localhost:8000/query \
  -d '{"query": "CAS code", "top_k": 5}'
```

4. Check embedding model:
```bash
curl http://localhost:8000/models
```

### Slow Queries

**Symptoms:**
```
Query takes > 5 seconds
```

**Solutions:**

1. Check HNSW index exists:
```bash
docker-compose exec postgres psql -U postgres -d gsrs_rag \
  -c "\di idx_embedding_hnsw"
```

2. Create index if missing:
```sql
CREATE INDEX IF NOT EXISTS idx_embedding_hnsw 
ON chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

3. Reduce top_k:
```bash
curl -X POST http://localhost:8000/query \
  -d '{"query": "aspirin", "top_k": 3}'
```

## Data Loading Issues

### Load Script Fails

**Symptoms:**
```
HTTP Error: 500 Internal Server Error
```

**Solutions:**

1. Check API health:
```bash
curl http://localhost:8000/health
```

2. Reduce batch size:
```bash
python scripts/load_data.py data.gsrs --batch-size 25
```

3. Check file format:
```bash
# Test parse
python -c "
import gzip
import json
with gzip.open('data.gsrs', 'rt') as f:
    for i, line in enumerate(f):
        if i > 5: break
        print(f'Line {i}: {line[:100]}...')
"
```

### No Chunks Generated

**Symptoms:**
```json
{
  "chunks_created": 0
}
```

**Solutions:**

1. Verify substance JSON structure:
```bash
python -c "
import json
with open('substance.json') as f:
    data = json.load(f)
    print('uuid:', data.get('uuid'))
    print('substanceClass:', data.get('substanceClass'))
"
```

2. Check required fields:
- `uuid` - must be present
- `substanceClass` - must be present

3. Test with sample data:
```bash
python scripts/download_samples.py
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d @sample_data/*.json
```

## Open WebUI Issues

### Can't Connect to MCP Server

**Symptoms:**
```
Connection refused to http://gsrs-rag-gateway:8000
```

**Solutions:**

1. Check container network:
```bash
docker-compose ps
```

2. Test from inside container:
```bash
docker-compose exec open-webui \
  wget -qO- http://gsrs-rag-gateway:8000/health
```

3. Check Docker network:
```bash
docker network ls
docker-compose networks
```

### Model Not Available

**Symptoms:**
```
Error: model 'llama3.1' not found
```

**Solutions:**

1. Pull model:
```bash
ollama pull llama3.1
```

2. List available models:
```bash
ollama list
```

3. Select model in Open WebUI:
- Click model selector (top left)
- Choose available model

## Performance Issues

### High Memory Usage

**Symptoms:**
```
Container killed due to OOM
```

**Solutions:**

1. Increase Docker memory:
```bash
# Docker Desktop: Settings → Resources → Memory
# Or use --memory flag
docker-compose up -d --memory 4g
```

2. Reduce batch size for loading:
```bash
python scripts/load_data.py data.gsrs --batch-size 10
```

3. Use smaller embedding model:
```bash
# Ollama
EMBEDDING_MODEL=all-minilm
EMBEDDING_DIMENSION=384
```

### Slow Ingestion

**Symptoms:**
```
Ingestion takes > 1 minute per substance
```

**Solutions:**

1. Check API latency:
```bash
time curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d @substance.json
```

2. Use batch ingestion:
```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H "Content-Type: application/json" \
  -d @substances.json
```

3. Check embedding API:
```bash
# OpenAI
time curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-3-small", "input": "test"}'
```

## Diagnostic Commands

### Quick Health Check

```bash
# All services running?
docker-compose ps

# Gateway healthy?
curl http://localhost:8000/health

# Data loaded?
curl http://localhost:8000/statistics

# Embeddings working?
curl http://localhost:8000/models
```

### Log Inspection

```bash
# Gateway logs
docker-compose logs --tail=100 rag-gateway

# Database logs
docker-compose logs --tail=100 postgres

# Follow logs in real-time
docker-compose logs -f rag-gateway
```

### Database Inspection

```bash
# Connect to database
docker-compose exec postgres psql -U postgres -d gsrs_rag

# Count chunks
SELECT COUNT(*) FROM chunks;

# Check vector dimension
SELECT pg_typeof(embedding), vector_dims(embedding)
FROM chunks LIMIT 1;
```

## Getting Help

If you can't resolve the issue:

1. **Check existing issues** on the repository
2. **Create a new issue** with:
   - Error messages
   - `docker-compose logs` output
   - Configuration (.env with secrets redacted)
   - Steps to reproduce

## Related Documentation

- [Quick Start](quickstart.md)
- [Configuration](configuration.md)
- [API Reference](api-reference.md)
