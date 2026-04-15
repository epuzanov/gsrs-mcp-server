"""
title: GSRS MCP Pipe
author: Egor Puzanov
author_url: https://github.com/epuzanov
git_url: https://github.com/epuzanov/gsrs-mcp-server
description: Open WebUI Pipe Function that routes chat prompts to GSRS MCP tools over streamable HTTP.
required_open_webui_version: 0.4.0
requirements: httpx,mcp
version: 0.1.0
license: MIT
"""

import json
from typing import Any, Awaitable, Callable, Literal, Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field


EventEmitter = Optional[Callable[[dict], Awaitable[None]]]


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="GSRS/",
            description="Prefix shown before Pipe model names in Open WebUI.",
        )
        MCP_URL: str = Field(
            default="http://localhost:8000/mcp",
            description="Streamable HTTP endpoint for the GSRS MCP server.",
        )
        MCP_TOKEN: str = Field(
            default="",
            description="Optional bearer token for MCP_PASSWORD-protected servers.",
        )
        DEFAULT_QUERY_TOOL: Literal["gsrs_ask", "gsrs_retrieve"] = Field(
            default="gsrs_ask",
            description="Fallback MCP tool when the selected Pipe model does not map cleanly.",
        )
        HTTP_TIMEOUT_SECONDS: int = Field(
            default=60,
            description="Timeout for MCP HTTP requests in seconds.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        prefix = self.valves.NAME_PREFIX
        return [
            {"id": "gsrs_ask", "name": f"{prefix}Ask"},
            {"id": "gsrs_retrieve", "name": f"{prefix}Retrieve"},
        ]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        tool_name = self._tool_name_from_body(body)
        query = self._extract_query(body)
        if not query:
            return "No user query was found in the Open WebUI request body."

        await self._emit_status(
            __event_emitter__,
            f"Calling {tool_name} on the GSRS MCP server...",
            done=False,
        )
        try:
            result = await self._call_tool(tool_name, {"query": query})
        except Exception as exc:
            await self._emit_status(
                __event_emitter__,
                f"GSRS MCP call failed: {exc}",
                done=True,
            )
            return f"GSRS MCP error: {exc}"

        await self._emit_status(
            __event_emitter__,
            f"{tool_name} completed.",
            done=True,
        )
        return result

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        headers = {"Authorization": f"Bearer {self.valves.MCP_TOKEN}"} if self.valves.MCP_TOKEN else {}
        timeout = float(self.valves.HTTP_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            async with streamable_http_client(
                self.valves.MCP_URL,
                http_client=client,
                terminate_on_close=False,
            ) as transport:
                read_stream, write_stream, _ = transport
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return self._result_to_text(result)

    def _tool_name_from_body(self, body: dict[str, Any]) -> str:
        model_name = str(body.get("model", "")).lower()
        if model_name.endswith("gsrs_retrieve") or model_name.endswith("retrieve"):
            return "gsrs_retrieve"
        if model_name.endswith("gsrs_ask") or model_name.endswith("ask"):
            return "gsrs_ask"
        return self.valves.DEFAULT_QUERY_TOOL

    def _extract_query(self, body: dict[str, Any]) -> str:
        messages = body.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")).strip())
                return "\n".join(part for part in parts if part).strip()
        return str(body.get("prompt", "")).strip()

    def _result_to_text(self, result: Any) -> str:
        if getattr(result, "structuredContent", None):
            return json.dumps(result.structuredContent, indent=2)
        return "\n".join(
            getattr(block, "text", json.dumps(block.model_dump(), indent=2))
            for block in result.content
        )

    async def _emit_status(
        self,
        emitter: EventEmitter,
        description: str,
        *,
        done: bool,
    ) -> None:
        if emitter is None:
            return
        await emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                    "hidden": False,
                },
            }
        )
