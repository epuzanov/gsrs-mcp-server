# MCP Guide

## Supported Transports

- `streamable-http`
- `stdio`

## Default HTTP Endpoint

```text
http://localhost:8000/mcp
```

Bearer token:

```http
Authorization: Bearer <API_PASSWORD>
```

## Claude Desktop Example

```json
{
  "mcpServers": {
    "gsrs": {
      "command": "gsrs-mcp-server",
      "env": {
        "MCP_TRANSPORT": "stdio",
        "DATABASE_URL": "chroma://./chroma_data/chunks",
        "EMBEDDING_API_KEY": "sk-your-key",
        "EMBEDDING_URL": "https://api.openai.com/v1/embeddings",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSION": "1536"
      }
    }
  }
}
```

## HTTP Client Example

```json
{
  "mcpServers": {
    "gsrs": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer change-me"
      }
    }
  }
}
```

## Tooling Notes

- `gsrs_ask` is the highest-level grounded-answer tool
- `gsrs_retrieve` is the safest raw retrieval path for diagnostics
- `gsrs_health` and `/health` help diagnose degraded mode
- `debug=true` on `gsrs_ask` or `gsrs_retrieve` exposes internal routing and ranking details
