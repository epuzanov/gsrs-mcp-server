"""Simple MCP client for the GSRS MCP Server."""
import argparse
import asyncio
import json

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client


def _result_to_text(result) -> str:
    if result.structuredContent:
        return json.dumps(result.structuredContent, indent=2)
    return "\n".join(
        getattr(block, "text", json.dumps(block.model_dump(), indent=2))
        for block in result.content
    )


async def call_stdio(query: str, tool_name: str, command: str = "gsrs-mcp-server") -> str:
    server = StdioServerParameters(command=command, env={"MCP_TRANSPORT": "stdio"})
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, {"query": query})
            return _result_to_text(result)


async def call_http(query: str, tool_name: str, url: str, token: str) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        async with streamable_http_client(url, http_client=client, terminate_on_close=False) as transport:
            read_stream, write_stream, _ = transport
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, {"query": query})
                return _result_to_text(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a query-oriented GSRS MCP tool.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--tool", default="gsrs_ask", help="MCP tool to call, for example gsrs_ask or gsrs_retrieve.")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--command", default="gsrs-mcp-server")
    parser.add_argument("--url", default="http://localhost:8000/mcp")
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    if args.transport == "stdio":
        output = asyncio.run(call_stdio(args.query, args.tool, command=args.command))
    else:
        output = asyncio.run(call_http(args.query, args.tool, url=args.url, token=args.token))

    print(output)


if __name__ == "__main__":
    main()
