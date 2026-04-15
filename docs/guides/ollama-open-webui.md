# Ollama and Open WebUI

This guide focuses on local or private deployments where:

- Ollama provides embeddings and optional answer generation
- `gsrs-mcp-server` exposes the GSRS tools over MCP
- Open WebUI is the user-facing chat interface

Recommended GSRS configuration with Ollama:

```bash
EMBEDDING_API_KEY=ollama
EMBEDDING_URL=http://host.docker.internal:11434/api/embed
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

LLM_API_KEY=ollama
LLM_URL=http://host.docker.internal:11434/v1/chat/completions
LLM_MODEL=llama3.1

MCP_TRANSPORT=streamable-http
MCP_API=0.0.0.0
MCP_PORT=8000
MCP_USERNAME=admin
MCP_PASSWORD=change-me
```

Health checks:

```bash
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
curl http://localhost:8000/health
```

## Scenario 1: Admin Settings -> Settings -> Integrations -> Manage Tool Servers

Use this when your Open WebUI version supports MCP integrations directly.

This is the cleanest path because Open WebUI can talk to the GSRS server as an
MCP server instead of through a custom wrapper.

Typical setup:

1. Start `gsrs-mcp-server`
2. Open `Admin Settings -> Settings -> Integrations -> Manage Tool Servers`
3. Click `+` (`Add Server`)
4. Set `Type` to `MCP (Streamable HTTP)`
5. Enter the GSRS server endpoint:

```text
http://localhost:8000/mcp
```

6. Enter the auth details:

```http
Authorization: Bearer <MCP_PASSWORD>
```

7. Save

Recommended tools to expose first:

- `gsrs_ask`
- `gsrs_retrieve`
- `gsrs_similarity_search`
- `gsrs_health`

Use this path when:

- you want the most MCP-native setup
- you want Open WebUI to discover GSRS tools directly
- you do not need a custom Python wrapper layer

## Scenario 2: Admin Settings -> Functions

Use this when you want to wrap GSRS behavior in a curated Open WebUI function.

This is useful when:

- your Open WebUI deployment does not use MCP integrations directly
- you want a smaller, safer surface than all exposed MCP tools
- you want opinionated prompts or pre/post-processing

Good repo references for this approach:

- [examples/gsrs_function.py](../../examples/gsrs_function.py)
- [examples/gsrs_tool.py](../../examples/gsrs_tool.py)
- [examples/gsrs_system_prompt.md](../../examples/gsrs_system_prompt.md)
- [examples/gsrs_system_prompt_minimal.md](../../examples/gsrs_system_prompt_minimal.md)

Typical pattern:

1. Create an Open WebUI function
2. Import `examples/gsrs_function.py` as a Pipe Function
3. Configure the valves for `MCP_URL` and `MCP_TOKEN`
4. Select either the `GSRS/Ask` or `GSRS/Retrieve` Pipe model in chat
5. Add a system prompt that tells the assistant when to use the function

Use this path when:

- you want admin-managed behavior
- you want a single tool like "Ask GSRS" instead of the full MCP surface
- you want to keep Open WebUI prompts and routing tightly controlled

## Scenario 3: Workspace -> Tools

Use this when you want team- or workspace-level tool definitions inside Open
WebUI.

This is usually the best fit when:

- only one workspace needs GSRS tools
- you want different tool sets for different teams
- you want to expose a few specialized GSRS tools instead of everything

Recommended workspace tools:

- `GSRS Ask`
  Calls `gsrs_ask` for grounded answers
- `GSRS Retrieve`
  Calls `gsrs_retrieve` for source-focused lookups
- `GSRS Similarity`
  Calls `gsrs_similarity_search` for related substances
- `GSRS Health`
  Calls `gsrs_health` for diagnostics

Suggested tool behavior:

- default user-facing tool: `gsrs_ask`
- diagnostics/admin tool: `gsrs_health`
- expert/debug tool: `gsrs_retrieve` with `debug=true`

Implementation note:

- Import [examples/gsrs_tool.py](../../examples/gsrs_tool.py) into `Workspace -> Tools` to expose native methods such as `answer_question`, `retrieve_evidence`, `find_similar_substances`, `check_health`, `get_document`, and the `gsrs_api_*` lookups.

Use this path when:

- you want workspace-local governance
- you do not need global admin-level integration
- you want a curated tool catalog for a specific team

## Which Open WebUI path to prefer

Prefer `Admin Settings -> Integrations -> MCP` when available:

- it matches the server's MCP-native design
- it avoids extra wrapper code
- it keeps tool contracts closest to the GSRS server

Prefer `Admin Settings -> Functions` when:

- you want heavy customization
- you want to hide most tools behind a single curated function

Prefer `Workspace -> Tools` when:

- you want a limited per-workspace rollout
- different teams need different GSRS exposure
