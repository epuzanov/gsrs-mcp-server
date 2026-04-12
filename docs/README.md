# GSRS MCP Server Documentation

Welcome to the GSRS MCP Server documentation. This server provides an MCP (Model Context Protocol) interface for GSRS (Global Substance Registration System) substances.

## Table of Contents

### Getting Started

- [Quick Start](quickstart.md) - Get up and running in minutes
- [Configuration](configuration.md) - Environment variables and settings
- [Authentication](authentication.md) - API authentication methods

### Core Concepts

- [Vector Databases](vector-databases.md) - pgvector vs ChromaDB backends
- [Chunking Strategy](guides/chunking.md) - How substance documents are chunked
- [Data Loading](data-loading.md) - Loading substances into the gateway

### Integration Guides

- [ChatGPT Integration](guides/chatgpt.md) - Use with ChatGPT
- [Ollama + Open WebUI](guides/ollama-open-webui.md) - Local LLM setup

### API Reference

- [API Reference](api-reference.md) - Complete API documentation

### Troubleshooting

- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Quick Links

- [GitHub Repository](https://github.com/epuzanov/gsrs-rag-gateway)
- [Swagger UI](http://localhost:8000/docs) (when running)
- [ReDoc](http://localhost:8000/redoc) (when running)

## Architecture Overview

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
```

## Services

The gateway uses three main services:

| Service | Location | Description |
|---------|----------|-------------|
| `ChunkerService` | `app/services/chunking.py` | Chunks GSRS JSON into VectorDocuments |
| `EmbeddingService` | `app/services/embedding.py` | Generates embeddings via OpenAI/Ollama |
| `VectorDatabaseService` | `app/services/vector_database.py` | Manages vector database operations |

## Supported GSRS Substance Classes

- **concept** - Base substance class
- **chemical** - Small molecule chemicals with structure
- **protein** - Proteins including monoclonal antibodies
- **nucleicAcid** - RNA/DNA therapeutics
- **polymer** - Synthetic and biosynthetic polymers
- **mixture** - Mixtures of multiple components
- **structurallyDiverse** - Natural products from organisms, minerals, etc.
- **specifiedSubstanceG1** - Specified substances with constituents

## License

MIT License - see [LICENSE](../LICENSE) for details.
