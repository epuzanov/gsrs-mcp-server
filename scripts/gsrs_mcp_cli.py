"""Simple MCP client for the GSRS MCP Server."""
import argparse
import asyncio
import json
from typing import Any

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


def _parse_json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def _parse_parameter_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_parameter_assignment(raw: str) -> tuple[str, Any]:
    key, separator, value = raw.partition("=")
    if not separator or not key:
        raise ValueError(f"Parameter must use key=value syntax: {raw}")
    return key, _parse_parameter_value(value)


def build_tool_arguments(
    query: str | None,
    *,
    arguments_json: str = "",
    parameters: list[str] | None = None,
) -> dict[str, Any]:
    """Build MCP tool arguments from a query shortcut plus optional parameters."""
    arguments: dict[str, Any] = {}
    if arguments_json:
        arguments.update(_parse_json_object(arguments_json, label="--arguments"))
    for raw_parameter in parameters or []:
        key, value = _parse_parameter_assignment(raw_parameter)
        arguments[key] = value
    if query:
        arguments.setdefault("query", query)
    return arguments


async def call_stdio(
    tool_name: str,
    arguments: dict[str, Any],
    command: str = "gsrs-mcp-server",
) -> str:
    server = StdioServerParameters(command=command, env={"MCP_TRANSPORT": "stdio"})
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _result_to_text(result)


async def call_http(tool_name: str, arguments: dict[str, Any], url: str, token: str) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        async with streamable_http_client(url, http_client=client, terminate_on_close=False) as transport:
            read_stream, write_stream, _ = transport
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return _result_to_text(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a query-oriented GSRS MCP tool.")
    parser.add_argument("--query", help="Convenience shortcut for tools that accept a query argument.")
    parser.add_argument("--tool", default="rag_query", help="MCP tool to call, for example rag_query or gsrs_get_summary.")
    parser.add_argument(
        "--arguments",
        default="",
        help='Full JSON object of MCP tool arguments, for example \'{"query":"aspirin","top_k":3}\'.',
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Additional MCP tool argument in key=value form. Values are parsed as JSON when possible; repeat as needed.",
    )
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--command", default="gsrs-mcp-server")
    parser.add_argument("--url", default="http://localhost:8000/mcp")
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    try:
        arguments = build_tool_arguments(
            args.query,
            arguments_json=args.arguments,
            parameters=args.param,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.transport == "stdio":
        output = asyncio.run(call_stdio(args.tool, arguments, command=args.command))
    else:
        output = asyncio.run(call_http(args.tool, arguments, url=args.url, token=args.token))

    print(output)


if __name__ == "__main__":
    main()
