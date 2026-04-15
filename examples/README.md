# Examples

The examples in this folder show how to call the GSRS MCP Server as an MCP client. They do not rely on deprecated REST mirror endpoints.

Files:

- `../scripts/gsrs_mcp_cli.py`: end-to-end example client for query-oriented MCP tools over `stdio` or streamable HTTP
- `gsrs_function.py`: Open WebUI Pipe Function example that exposes GSRS Ask and GSRS Retrieve as selectable models
- `gsrs_tool.py`: Open WebUI Workspace Tool example that calls GSRS MCP tools from native tool methods
- `gsrs_tool_system_prompt.md`: system prompt for Open WebUI setups that use `gsrs_tool.py`
- `openai_responses_mcp.py`: OpenAI API example using the Responses API with the GSRS MCP server as a remote MCP tool
- `gsrs_system_prompt.md`
- `gsrs_system_prompt_minimal.md`

Typical direct MCP usage:

```bash
python scripts/gsrs_mcp_cli.py --transport http --url http://localhost:8000/mcp --token change-me --query "What is the CAS code for aspirin?"
```

For HTTP transport, `--token` is the bearer token value configured as `MCP_PASSWORD`.

Or:

```bash
python scripts/gsrs_mcp_cli.py --transport stdio --query "Show me the UUID for aspirin"
```

Select a different query-oriented tool:

```bash
python scripts/gsrs_mcp_cli.py --transport http --tool gsrs_retrieve --url http://localhost:8000/mcp --token change-me --query "What is the CAS code for aspirin?"
```

OpenAI API usage:

```bash
set OPENAI_API_KEY=sk-your-openai-key
python examples/openai_responses_mcp.py --mcp-url http://localhost:8000/mcp --mcp-token change-me --query "What is the approval ID for aspirin?"
```

Open WebUI imports:

- Import `gsrs_function.py` under `Admin Settings -> Functions` when you want a Pipe Function that behaves like a model.
- Import `gsrs_tool.py` under `Workspace -> Tools` when you want native Open WebUI tool methods such as `answer_question`, `retrieve_evidence`, `get_document`, and the `gsrs_api_*` lookups.

Scenario mapping:

- ChatGPT without writing API code:
  see [docs/guides/chatgpt.md](../docs/guides/chatgpt.md)
- OpenAI API with remote MCP:
  see `openai_responses_mcp.py`
- Open WebUI Pipe Functions:
  start from `gsrs_function.py`
- Open WebUI Workspace Tools:
  start from `gsrs_tool.py`
