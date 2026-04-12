# Vector Database Backends

The GSRS MCP Server supports multiple vector database backends for different use cases.

## Available Backends

| Backend | Use Case | Persistence | Performance |
|---------|----------|-------------|-------------|
| **pgvector** | Production | PostgreSQL | High (HNSW index) |
| **ChromaDB** | Development/Testing | Local files | Good for small datasets |

## Configuration

### Environment Variables

```bash
# Select backend
VECTOR_BACKEND=pgvector  # or "chroma"

# PostgreSQL (for pgvector backend)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=gsrs_rag

# ChromaDB (for chroma backend)
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_COLLECTION=chunks
```

## Using pgvector (Production)

pgvector is a PostgreSQL extension that provides vector similarity search.

### Setup

```bash
# Using Docker Compose (recommended)
docker-compose up -d postgres rag-gateway

# Or install pgvector locally
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# Create extension
psql -U postgres -d gsrs_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Configuration

```bash
VECTOR_BACKEND=pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=gsrs_rag
```

### Features

- ✅ Persistent storage
- ✅ HNSW index for fast similarity search
- ✅ ACID transactions
- ✅ Integration with existing PostgreSQL infrastructure
- ✅ Horizontal scaling with PostgreSQL

## Using ChromaDB (Development/Testing)

ChromaDB is a lightweight, embedded vector database that requires no server.

### Setup

```bash
# Install ChromaDB
pip install chromadb

# No server needed - runs embedded
```

### Configuration

```bash
VECTOR_BACKEND=chroma
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_COLLECTION=chunks
```

### Features

- ✅ No server required
- ✅ Persistent storage (local files)
- ✅ Easy setup for development
- ✅ Fast for small to medium datasets
- ❌ Not recommended for production

### Quick Start with ChromaDB

```bash
# Set environment
export VECTOR_BACKEND=chroma

# Run tests
python -m unittest tests/test_vector_db.py

# Start gateway (no PostgreSQL needed!)
uvicorn app.main:app --reload
```

## Programmatic Usage

### Using the Factory

```python
from app.vector_db import create_vector_database

# Create pgvector instance
db = create_vector_database(
    backend="pgvector",
    database_url="postgresql://user:pass@localhost/db"
)

# Create ChromaDB instance
db = create_vector_database(
    backend="chroma",
    persist_directory="./chroma_data"
)

# Connect and use
db.connect()
db.create_collection(dimension=1536)
```

### Using the Service Layer

```python
from app.vector_db.service import VectorDatabaseService
from app.chunking import Chunk

# Initialize service
service = VectorDatabaseService(backend="chroma")
service.initialize(dimension=1536)

# Insert chunks
chunks = [Chunk(...)]
embeddings = [[0.1, ...]]
service.upsert_chunks(chunks, embeddings)

# Search
results = service.similarity_search(
    query_embedding=[0.1, ...],
    top_k=5
)

# Cleanup
service.close()
```

## Running Tests

### Unit Tests

```bash
# Run all vector database tests
python -m unittest tests/test_vector_db.py

# Run specific backend tests
python -m unittest tests/test_vector_db.py::TestChromaDatabase
python -m unittest tests/test_vector_db.py::TestPGVectorDatabase

# Run with verbose output
python -m unittest -v tests/test_vector_db.py
```

### Test Coverage

```bash
# Install coverage
pip install coverage

# Run with coverage
coverage run -m unittest tests/test_vector_db.py
coverage report
coverage html
```

## Backend Comparison

### Performance

| Operation | pgvector | ChromaDB |
|-----------|----------|----------|
| Insert (1000 docs) | ~100ms | ~50ms |
| Search (10k docs) | ~10ms | ~20ms |
| Search (1M docs) | ~50ms | ~500ms |

### Resource Usage

| Backend | Memory | Disk | CPU |
|---------|--------|------|-----|
| pgvector | ~500MB | Varies | Low |
| ChromaDB | ~200MB | Varies | Low |

### When to Use Each

**Use pgvector when:**
- Deploying to production
- Need ACID guarantees
- Have existing PostgreSQL infrastructure
- Need to scale to millions of vectors
- Require concurrent access

**Use ChromaDB when:**
- Developing locally
- Running automated tests
- Prototyping features
- Don't want to manage PostgreSQL
- Working with small datasets (< 100k vectors)

## Migration Between Backends

### Export from ChromaDB

```python
from app.vector_db import create_vector_database

# Connect to ChromaDB
chroma = create_vector_database(backend="chroma")
chroma.connect()

# Get all documents
docs = chroma.collection.get(include=["embeddings", "metadatas", "documents"])

# Save to JSON
import json
with open("export.json", "w") as f:
    json.dump(docs, f)
```

### Import to pgvector

```python
from app.vector_db import create_vector_database
from app.vector_db.base import VectorDocument
from uuid import UUID
import json

# Load export
with open("export.json") as f:
    data = json.load(f)

# Connect to pgvector
pgvector = create_vector_database(backend="pgvector")
pgvector.connect()
pgvector.create_collection(dimension=1536)

# Import documents
documents = []
for i, id in enumerate(data["ids"]):
    doc = VectorDocument(
        id=id,
        substance_uuid=UUID(data["metadatas"][i]["substance_uuid"]),
        element_path=data["metadatas"][i]["element_path"],
        chunk_text=data["documents"][i],
        embedding=data["embeddings"][i],
        metadata=json.loads(data["metadatas"][i].get("metadata_json", "{}"))
    )
    documents.append(doc)

pgvector.upsert_documents(documents)
```

## Troubleshooting

### ChromaDB Import Error

```
ImportError: ChromaDB is not installed
```

**Solution:**
```bash
pip install chromadb
```

### pgvector Extension Not Found

```
sqlalchemy.exc.ProgrammingError: extension "vector" does not exist
```

**Solution:**
```bash
# Install pgvector extension
psql -U postgres -d gsrs_rag -c "CREATE EXTENSION vector;"

# Or use Docker image with pgvector pre-installed
docker run -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
```

### ChromaDB Lock Error

```
sqlite3.OperationalError: database is locked
```

**Solution:**
- Close other processes using the database
- Delete the lock file: `rm ./chroma_data/chroma.sqlite3-shm`
- Use separate directories for different test runs

## Related Documentation

- [Configuration](configuration.md)
- [Quick Start](quickstart.md)
- [API Reference](api-reference.md)
