"""Reusable Python helpers for calling GSRS MCP tools."""
import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def _result_to_text(result) -> str:
    if result.structuredContent:
        return json.dumps(result.structuredContent, indent=2)
    return "\n".join(
        getattr(block, "text", json.dumps(block.model_dump(), indent=2))
        for block in result.content
    )


async def call_tool(
    tool_name: str,
    arguments: dict,
    *,
    mcp_url: str = "http://localhost:8000/mcp",
    token: str = "",
) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        async with streamable_http_client(mcp_url, http_client=client, terminate_on_close=False) as transport:
            read_stream, write_stream, _ = transport
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return _result_to_text(result)


async def ask(query: str, *, mcp_url: str = "http://localhost:8000/mcp", token: str = "") -> str:
    return await call_tool("gsrs_ask", {"query": query}, mcp_url=mcp_url, token=token)


async def retrieve(query: str, *, mcp_url: str = "http://localhost:8000/mcp", token: str = "") -> str:
    return await call_tool("gsrs_retrieve", {"query": query}, mcp_url=mcp_url, token=token)
