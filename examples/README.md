# Examples

The examples in this folder show how to call the GSRS MCP Server as an MCP client. They do not rely on deprecated REST mirror endpoints.

Files:

- `gsrs_tool.py`: end-to-end example that calls `gsrs_ask` over `stdio` or streamable HTTP
- `gsrs_function.py`: reusable helper functions for calling MCP tools from Python, useful for wrappers such as Open WebUI Functions or Tools
- `openai_responses_mcp.py`: OpenAI API example using the Responses API with the GSRS MCP server as a remote MCP tool
- `gsrs_system_prompt.md`
- `gsrs_system_prompt_minimal.md`

Typical direct MCP usage:

```bash
python examples/gsrs_tool.py --transport http --url http://localhost:8000/mcp --token change-me --query "What is the CAS code for aspirin?"
```

Or:

```bash
python examples/gsrs_tool.py --transport stdio --query "Show me the UUID for aspirin"
```

OpenAI API usage:

```bash
set OPENAI_API_KEY=sk-your-openai-key
python examples/openai_responses_mcp.py --mcp-url http://localhost:8000/mcp --mcp-token change-me --query "What is the approval ID for aspirin?"
```

Scenario mapping:

- ChatGPT without writing API code:
  see [docs/guides/chatgpt.md](../docs/guides/chatgpt.md)
- OpenAI API with remote MCP:
  see `openai_responses_mcp.py`
- Open WebUI Functions or Tools wrappers:
  start from `gsrs_function.py`
