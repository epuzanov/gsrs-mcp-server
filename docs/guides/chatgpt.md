# ChatGPT Integration Guide

This guide shows you how to integrate the GSRS MCP Server with ChatGPT to answer questions about chemical substances using the GSRS database.

## Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│   ChatGPT    │────▶│  MCP Server │────▶│  pgvector   │
│  Question   │     │   (GPT-4)    │     │   (API)      │     │  Database   │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
```

The workflow:
1. User asks a question about a chemical substance
2. ChatGPT queries the MCP Server API for relevant context
3. MCP Server performs vector search in database
4. ChatGPT receives context and generates an informed answer

## Prerequisites

1. **GSRS MCP Server** running
2. **OpenAI API Key** with access to GPT models
3. **Python 3.9+** with required packages

## Step 1: Start the MCP Server

### Option A: Quick Start with ChromaDB (Recommended for Testing)

```bash
# Clone and navigate to the project
cd gsrs-rag-gateway

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env:
#   DATABASE_URL=chroma://./chroma_data/chunks
#   EMBEDDING_API_KEY=sk-your-openai-key

# Start the server
uvicorn app.main:app --reload

# Verify it's running
curl http://localhost:8000/health
```

### Option B: Production with PostgreSQL (Docker)

```bash
# Configure environment
cp .env.example .env
# Edit .env:
#   DATABASE_URL=postgresql://postgres:postgres@postgres:5432/gsrs_rag
#   EMBEDDING_API_KEY=sk-your-openai-key

# Start with Docker Compose
docker-compose up -d

# Verify it's running
curl http://localhost:8000/health
```

## Step 2: Load Substance Data

### Option A: Load Sample Substances from GSRS Server (Recommended)

Use the `load_data.py` script to download and ingest substances directly from the GSRS NCATS API:

```bash
# Load specific substances by UUID (recommended for testing)
python scripts/load_data.py \
  --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028,80edf0eb-b6c5-4a9a-adde-28c7254046d9

# Load all substances from GSRS server (may take hours)
python scripts/load_data.py --all --max-results 100

# Verify data is loaded
curl http://localhost:8000/statistics
```

### Option B: Load from .gsrs File

If you have substance data in `.gsrs` format (JSONL.gz):

```bash
python scripts/load_data.py data/substances.gsrs --batch-size 100
```

### Option C: Manual Download and Ingest

```bash
# Download single substance
curl -o substance.json \
  "https://gsrs.ncats.nih.gov/api/v1/substances(0103a288-6eb6-4ced-b13a-849cd7edf028)?view=full"

# Ingest into MCP Server
curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -u admin:admin123 \
    -d @substance.json
```

### Verify Data is Loaded

```bash
# Check statistics
curl http://localhost:8000/statistics

# Check available substance classes
curl http://localhost:8000/statistics
```

Expected output:
```json
{
  "total_chunks": 450,
  "total_substances": 18
}
```

## Step 3: Install Python Dependencies

```bash
pip install openai httpx
```

## Step 4: Create the Integration Script

Create a file named `ask_chatgpt.py`:

```python
#!/usr/bin/env python3
"""
Ask ChatGPT questions about GSRS substances using MCP Server.
"""

import os
import httpx
from openai import OpenAI

# Configuration
RAG_GATEWAY_URL = os.getenv("RAG_GATEWAY_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin123")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)


def query_rag(query: str, top_k: int = 5):
    """Query the MCP Server for relevant substance chunks."""
    url = f"{RAG_GATEWAY_URL}/query"

    payload = {
        "query": query,
        "top_k": top_k
    }

    response = httpx.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def create_context(results: list) -> str:
    """Format RAG results as context for ChatGPT."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"""
[Source {i}]
- Element Path: {r['element_path']}
- Substance UUID: {r['substance_uuid']}
- Content: {r['chunk_text']}
- Similarity Score: {r['similarity_score']:.3f}
""")
    return "\n".join(parts)


def ask_chatgpt(question: str, context: str, model: str = "gpt-4o") -> str:
    """Ask ChatGPT with RAG context."""
    system_prompt = """You are a helpful assistant specialized in chemical substances and GSRS (Global Substance Registration System) data.

Use the provided context from the GSRS database to answer questions accurately.
- Always base your answers on the provided context
- If the context doesn't contain enough information, say so
- Include references to the source elements when possible (e.g., [Source 1])
- Be precise with chemical names, codes, and identifiers (CAS, UNII, ChEMBL, etc.)
- Format your answer clearly with proper structure"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from GSRS database:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("GSRS MCP Server - ChatGPT Integration")
    print("=" * 60)
    print()
    print(f"MCP Server URL: {RAG_GATEWAY_URL}")
    print(f"Model: gpt-4o")
    print()

    while True:
        question = input("\nYour question (or 'quit' to exit): ").strip()

        if not question or question.lower() == 'quit':
            break

        print("\n🔍 Searching GSRS database...")
        results = query_rag(question, top_k=5)

        if not results:
            print("No relevant information found in the database.")
            print("Tip: Make sure you've loaded substance data first.")
            print("     Run: python scripts/load_data.py --uuids <UUID1>,<UUID2>")
            continue

        print(f"Found {len(results)} relevant chunks")

        context = create_context(results)

        print("\n🤔 Asking ChatGPT...")
        answer = ask_chatgpt(question, context)

        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        print("\n📚 Sources used:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['element_path']}")
            print(f"     Substance: {r['substance_uuid']}")
            print(f"     Score: {r['similarity_score']:.3f}")


if __name__ == "__main__":
    main()
```

## Step 5: Run the Integration

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run the script
python ask_chatgpt.py
```

## Example Session

```
============================================================
GSRS MCP Server - ChatGPT Integration
============================================================

MCP Server URL: http://localhost:8000
Model: gpt-4o

Your question (or 'quit' to exit): What is the CAS code for Aspirin?

🔍 Searching GSRS database...
Found 5 relevant chunks

🤔 Asking ChatGPT...

============================================================
Answer:
============================================================
The CAS code for Aspirin is **50-78-2** [Source 1].

Aspirin is also known by its UNII code **WK2XYI10QM** [Source 2]
and has the ChEMBL identifier **CHEMBL521** [Source 3].

Additional information:
- Molecular Formula: C9H8O4
- Molecular Weight: 180.16 Da
- ATC Code: N02BA01
============================================================

📚 Sources used:
  1. codes_cas_50-78-2
     Substance: 0103a288-6eb6-4ced-b13a-849cd7edf028
     Class: chemical
     Score: 0.892
  2. codes_fda_unii_WK2XYI10QM
     Substance: 0103a288-6eb6-4ced-b13a-849cd7edf028
     Class: chemical
     Score: 0.856
  3. codes_chembl_CHEMBL521
     Substance: 0103a288-6eb6-4ced-b13a-849cd7edf028
     Class: chemical
     Score: 0.823
```

## Advanced: Custom GPT with API Integration

You can create a Custom GPT that directly calls the MCP Server API:

### 1. Create a Custom GPT

1. Go to [chat.openai.com](https://chat.openai.com)
2. Click "Explore GPTs" → "Create"
3. Configure your Custom GPT

### 2. Add Action (API)

In the Custom GPT configuration:

1. Click "Create new action"
2. Select "Import from URL"
3. Enter your MCP Server OpenAPI schema URL: `http://your-server:8000/openapi.json`

### 3. Configure the Action

```yaml
openapi: 3.0.0
info:
  title: GSRS MCP Server
  version: 1.0.0
servers:
  - url: http://your-server:8000
paths:
  /query:
    post:
      summary: Search substance database
      operationId: query_substances
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  description: Search query
                top_k:
                  type: integer
                  default: 5
      responses:
        '200':
          description: Search results
```

### 4. Instructions for Your Custom GPT

```
You are a GSRS Substance Database Assistant.

When users ask about chemical substances:
1. Use the query_substances API to search the database
2. Base your answers on the returned results
3. Always cite the source element paths
4. Be precise with chemical identifiers (CAS, UNII, ChEMBL, etc.)

If no results are found, inform the user that the substance
may not be in the database.
```

## Tips for Better Results

### 1. Optimize Query Parameters

```python
# More results for broader questions
results = query_rag(question, top_k=10)
```

### 2. Use Specific Questions

- ✅ "What is the CAS registry number for Ibuprofen?"
- ✅ "Show me all codes for Aspirin"
- ✅ "What is the molecular weight of Paracetamol?"
- ❌ "Tell me about drugs" (too vague)

### 3. Adjust Temperature

```python
# More deterministic answers (recommended for factual queries)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.1  # Lower = more focused
)

# More creative answers
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.7  # Higher = more creative
)
```

### 4. Use ERI Endpoint for Open WebUI Integration

The MCP Server also provides an ERI (External Retrieval Interface) endpoint:

```bash
curl -X POST http://localhost:8000/eri/query \
  -H "Content-Type: application/json" \
  -u admin:admin123 \
  -d '{"query": "CAS code for Aspirin", "top_k": 3}'
```

## Troubleshooting

### No Results Found

```bash
# Check if data is loaded
curl http://localhost:8000/statistics

# Check available substance classes
curl http://localhost:8000/substance-classes

# Load sample data if empty
python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028
```

### API Key Error

```bash
# Verify your API key
echo $OPENAI_API_KEY

# Test OpenAI API
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Check balance/credits
curl https://api.openai.com/v1/usage \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Gateway Not Responding

```bash
# Check gateway health
curl http://localhost:8000/health

# Check Docker containers
docker-compose ps

# Check logs
docker-compose logs rag-gateway
```

### Authentication Errors

```bash
# Test with credentials
curl -u admin:admin123 http://localhost:8000/health

# Change credentials in .env
API_USERNAME=your-username
API_PASSWORD=your-password
```

## Related Documentation

- [Data Loading Guide](../data-loading.md) - Complete guide for loading substances
- [Ollama + Open WebUI](ollama-open-webui.md) - Local LLM alternative
- [API Reference](../api-reference.md) - Complete API documentation
- [Configuration](../configuration.md) - Advanced configuration options
