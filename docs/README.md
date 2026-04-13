# GSRS MCP Server Docs

The repository is centered on an MCP server, not a REST API surface.

Start here:

- [quickstart.md](quickstart.md): install, configure, run, and check readiness
- [configuration.md](configuration.md): environment variables and degraded-mode behavior
- [authentication.md](authentication.md): bearer-token auth for the MCP HTTP transport
- [api-reference.md](api-reference.md): MCP tools plus health endpoints
- [guides/mcp.md](guides/mcp.md): client configuration examples

Reference links:

- Repository: https://github.com/epuzanov/gsrs-mcp-server
- MCP endpoint when running: `http://localhost:8000/mcp`
- Liveness: `http://localhost:8000/livez`
- Readiness: `http://localhost:8000/readyz`
