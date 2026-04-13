# Vector Databases

## ChromaDB

Best for:

- local development
- tests
- low-setup deployments

Example:

```bash
DATABASE_URL=chroma://./chroma_data/chunks
```

Notes:

- startup no longer deletes the Chroma collection on every run
- collection dimension must match `EMBEDDING_DIMENSION`

## PostgreSQL + pgvector

Best for:

- shared environments
- production-like deployments
- lexical retrieval through PostgreSQL full-text search

Example:

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gsrs_rag
```

Notes:

- the backend uses `pool_pre_ping=True` for more reliable long-lived connections
- the server creates the `vector` extension if needed during initialization
