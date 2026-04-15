"""
title: GSRS MCP Tools
author: Egor Puzanov
author_url: https://github.com/epuzanov
git_url: https://github.com/epuzanov/gsrs-mcp-server
description: Open WebUI Workspace Tool that calls GSRS MCP tools over streamable HTTP.
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


class Tools:
    class Valves(BaseModel):
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
            description="Default MCP tool for question-answering methods.",
        )
        HTTP_TIMEOUT_SECONDS: int = Field(
            default=60,
            description="Timeout for MCP HTTP requests in seconds.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def answer_question(
        self,
        question: str,
        tool_name: Optional[Literal["gsrs_ask", "gsrs_retrieve"]] = None,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """
        Answer a GSRS question through a selected MCP query tool.

        Use `gsrs_ask` for grounded direct answers and `gsrs_retrieve` for raw evidence.
        """
        selected_tool = tool_name or self.valves.DEFAULT_QUERY_TOOL
        await self._emit_status(
            __event_emitter__,
            f"Calling {selected_tool}...",
            done=False,
        )
        result = await self._call_tool(selected_tool, {"query": question})
        await self._emit_status(
            __event_emitter__,
            f"{selected_tool} completed.",
            done=True,
        )
        return result

    async def retrieve_evidence(
        self,
        query: str,
        debug: bool = False,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Retrieve raw GSRS evidence chunks without answer synthesis."""
        await self._emit_status(
            __event_emitter__,
            "Calling gsrs_retrieve...",
            done=False,
        )
        result = await self._call_tool("gsrs_retrieve", {"query": query, "debug": debug})
        await self._emit_status(
            __event_emitter__,
            "gsrs_retrieve completed.",
            done=True,
        )
        return result

    async def find_similar_substances(
        self,
        substance_json: str,
        top_k: int = 10,
        match_mode: Literal["contains", "match"] = "contains",
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Find GSRS substances similar to a provided GSRS JSON payload."""
        await self._emit_status(
            __event_emitter__,
            "Calling gsrs_similarity_search...",
            done=False,
        )
        result = await self._call_tool(
            "gsrs_similarity_search",
            {
                "substance_json": substance_json,
                "top_k": top_k,
                "match_mode": match_mode,
            },
        )
        await self._emit_status(
            __event_emitter__,
            "gsrs_similarity_search completed.",
            done=True,
        )
        return result

    async def check_health(
        self,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Return the current GSRS MCP runtime health payload."""
        await self._emit_status(
            __event_emitter__,
            "Calling gsrs_health...",
            done=False,
        )
        result = await self._call_tool("gsrs_health", {})
        await self._emit_status(
            __event_emitter__,
            "gsrs_health completed.",
            done=True,
        )
        return result

    async def get_document(
        self,
        substance_uuid: str,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Fetch a full GSRS substance JSON document by UUID."""
        await self._emit_status(
            __event_emitter__,
            "Calling gsrs_get_document...",
            done=False,
        )
        result = await self._call_tool(
            "gsrs_get_document",
            {"substance_uuid": substance_uuid},
        )
        await self._emit_status(
            __event_emitter__,
            "gsrs_get_document completed.",
            done=True,
        )
        return result

    async def get_substance_schema(
        self,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Return the GSRS substance JSON schema exposed by the MCP server."""
        await self._emit_status(
            __event_emitter__,
            "Calling gsrs_api_substance_schema...",
            done=False,
        )
        result = await self._call_tool("gsrs_api_substance_schema", {})
        await self._emit_status(
            __event_emitter__,
            "gsrs_api_substance_schema completed.",
            done=True,
        )
        return result

    async def api_search(
        self,
        query: str,
        page: int = 1,
        size: int = 20,
        fields: str = "",
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Search the official GSRS API text endpoint."""
        await self._emit_status(
            __event_emitter__,
            "Calling gsrs_api_search...",
            done=False,
        )
        result = await self._call_tool(
            "gsrs_api_search",
            {
                "query": query,
                "page": page,
                "size": size,
                "fields": fields,
            },
        )
        await self._emit_status(
            __event_emitter__,
            "gsrs_api_search completed.",
            done=True,
        )
        return result

    async def api_structure_search(
        self,
        smiles: str = "",
        inchi: str = "",
        search_type: Literal["EXACT", "SIMILAR", "SUBSTRUCTURE", "SUPERSTRUCTURE"] = "EXACT",
        size: int = 20,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Search the official GSRS API by chemical structure."""
        await self._emit_status(
            __event_emitter__,
            "Calling gsrs_api_structure_search...",
            done=False,
        )
        result = await self._call_tool(
            "gsrs_api_structure_search",
            {
                "smiles": smiles,
                "inchi": inchi,
                "search_type": search_type,
                "size": size,
            },
        )
        await self._emit_status(
            __event_emitter__,
            "gsrs_api_structure_search completed.",
            done=True,
        )
        return result

    async def api_sequence_search(
        self,
        sequence: str,
        search_type: Literal["EXACT", "CONTAINS", "SIMILAR"] = "EXACT",
        sequence_type: Literal["PROTEIN", "NUCLEIC_ACID"] = "PROTEIN",
        size: int = 20,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Search the official GSRS API by protein or nucleic-acid sequence."""
        await self._emit_status(
            __event_emitter__,
            "Calling gsrs_api_sequence_search...",
            done=False,
        )
        result = await self._call_tool(
            "gsrs_api_sequence_search",
            {
                "sequence": sequence,
                "search_type": search_type,
                "sequence_type": sequence_type,
                "size": size,
            },
        )
        await self._emit_status(
            __event_emitter__,
            "gsrs_api_sequence_search completed.",
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
