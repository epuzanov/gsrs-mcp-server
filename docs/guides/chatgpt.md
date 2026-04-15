# ChatGPT

This project is MCP-first. For ChatGPT, the two main integration paths are:

1. ChatGPT with MCP directly, without writing API code
2. Your own application using the OpenAI API plus this MCP server

As of April 13, 2026, these paths align with OpenAI's official documentation for
[ChatGPT developer mode](https://platform.openai.com/docs/guides/developer-mode)
and
[remote MCP servers in the API](https://platform.openai.com/docs/guides/tools-remote-mcp?lang=python).

## Scenario 1: GPT + MCP, without using the API

Use this when you want ChatGPT itself to connect to the GSRS MCP server and use
its tools in chat.

High-level flow:

1. Run `gsrs-mcp-server` with `MCP_TRANSPORT=streamable-http`
2. Make the MCP endpoint reachable over HTTPS if ChatGPT needs to access it remotely
3. In ChatGPT, enable developer mode
4. Add the GSRS MCP server as an app/tool source
5. Start a chat and let ChatGPT call the GSRS tools

Minimal server settings:

```bash
MCP_TRANSPORT=streamable-http
MCP_API=0.0.0.0
MCP_PORT=8000
MCP_USERNAME=admin
MCP_PASSWORD=change-me
```

Remote MCP endpoint:

```text
https://your-host.example.com/mcp
```

Auth header:

```http
Authorization: Bearer <MCP_PASSWORD>
```

Notes:

- `stdio` is great for local desktop MCP clients, but ChatGPT developer mode is
  oriented around remote MCP servers, so an HTTPS endpoint is the practical path
  here.
- If your GSRS server is only on `localhost`, put it behind a reverse proxy,
  tunnel, or other secure externally reachable endpoint first.
- Start with `gsrs_ask`, `gsrs_retrieve`, `gsrs_similarity_search`, and
  `gsrs_health`.

## Scenario 2: Use the OpenAI API

Use this when you are building your own app and want the model to call the GSRS
MCP server through the OpenAI API.

Recommended path:

1. Keep `gsrs-mcp-server` running over `streamable-http`
2. Use the OpenAI Responses API
3. Register the GSRS server as a remote MCP tool
4. Let the model decide when to call GSRS tools

This is the best fit when you need:

- application-owned auth and rate limiting
- your own UI or backend workflow
- additional tools mixed with GSRS
- full control over prompts, approvals, and response shaping

### Example: OpenWebUI front end + OpenAI API + GSRS MCP

One practical pattern is:

1. OpenWebUI is the user-facing chat UI
2. Your backend calls the OpenAI Responses API
3. The Responses API attaches `gsrs-mcp-server` as a remote MCP tool
4. The model decides when to call GSRS tools

That gives you:

- an OpenWebUI chat experience
- OpenAI-hosted reasoning and tool orchestration
- GSRS retrieval through the MCP server
- one place in your backend to control prompts, approvals, and logging

Runnable API example:

- [examples/openai_responses_mcp.py](../../examples/openai_responses_mcp.py)

Lower-level direct MCP client examples:

- [scripts/gsrs_mcp_cli.py](../../scripts/gsrs_mcp_cli.py)
- [examples/gsrs_function.py](../../examples/gsrs_function.py)

Minimal command:

```bash
set OPENAI_API_KEY=sk-your-openai-key
python examples/openai_responses_mcp.py --mcp-url http://localhost:8000/mcp --mcp-token change-me --query "What is the approval ID for aspirin?"
```

## Which one to use

Use GPT + MCP without API when:

- you want the fastest path inside ChatGPT
- you do not need to write application code
- ChatGPT is the primary user interface

Use the API when:

- you are building your own product or workflow
- you need backend control over prompts, tools, and logging
- you want to combine GSRS with other OpenAI tools or your own functions

## Another possibility

Another option is to use ChatGPT Actions or a richer ChatGPT app integration.
That can work well, but it is not the best fit for this repo because this
project already exposes an MCP-native tool surface. If you want to stay close to
the current server architecture, MCP is the cleaner path.
