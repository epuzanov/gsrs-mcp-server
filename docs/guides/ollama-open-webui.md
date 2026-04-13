# Ollama and Open WebUI

For local-private setups:

1. Run `gsrs-mcp-server`
2. Configure Ollama for embeddings and optional LLM answers
3. Point your MCP-capable client or bridge at `http://localhost:8000/mcp`

Example embedding config:

```bash
EMBEDDING_API_KEY=ollama
EMBEDDING_URL=http://host.docker.internal:11434/api/embed
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

Optional local answer generation:

```bash
LLM_API_KEY=ollama
LLM_URL=http://host.docker.internal:11434/v1/chat/completions
LLM_MODEL=llama3.1
```

Health checks:

```bash
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
curl http://localhost:8000/health
```
