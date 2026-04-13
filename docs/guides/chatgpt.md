# ChatGPT / MCP Clients

Use the GSRS server as an MCP tool source, not as a legacy REST API.

Recommended connection:

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

If your client supports local stdio MCP servers, prefer:

```json
{
  "mcpServers": {
    "gsrs": {
      "command": "gsrs-mcp-server",
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```
