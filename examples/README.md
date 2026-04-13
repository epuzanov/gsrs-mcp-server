# Examples

The examples in this folder show how to call the GSRS MCP Server as an MCP client. They do not rely on deprecated REST mirror endpoints.

Files:

- `gsrs_tool.py`: end-to-end example that calls `gsrs_ask` over `stdio` or streamable HTTP
- `gsrs_function.py`: reusable helper functions for calling MCP tools from Python
- `gsrs_system_prompt.md`
- `gsrs_system_prompt_minimal.md`

Typical usage:

```bash
python examples/gsrs_tool.py --transport http --url http://localhost:8000/mcp --token change-me --query "What is the CAS code for aspirin?"
```

Or:

```bash
python examples/gsrs_tool.py --transport stdio --query "Show me the UUID for aspirin"
```
