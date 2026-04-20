"""
title: GSRS MCP Tools
author: Egor Puzanov
author_url: https://github.com/epuzanov
git_url: https://github.com/epuzanov/gsrs-mcp-server
description: Open WebUI Workspace Tool that calls GSRS MCP tools over streamable HTTP without the mcp client library.
required_open_webui_version: 0.6.X
requirements: httpx
version: 0.2.0
license: MIT
"""

import json
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_SESSION_ID_HEADER = "mcp-session-id"
MCP_PROTOCOL_VERSION_HEADER = "mcp-protocol-version"


class RawMCPError(RuntimeError):
    """Raised when the MCP server returns a protocol-level error."""


class RawMCPClient:
    """Minimal streamable HTTP MCP client for Open WebUI tool calls."""

    def __init__(self, url: str, token: str, timeout: float, client_name: str) -> None:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout)
        self._url = url
        self._client_name = client_name
        self._next_id = 1
        self._session_id: str | None = None
        self._protocol_version: str | None = None

    async def __aenter__(self) -> "RawMCPClient":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._client.aclose()

    async def initialize(self) -> None:
        response = await self._post_request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self._client_name,
                    "version": "0.2.0",
                },
            },
            include_protocol_header=False,
        )
        result = self._extract_result(response)
        if not isinstance(result, dict):
            raise RawMCPError("Invalid initialize response from MCP server.")
        self._protocol_version = str(result.get("protocolVersion") or MCP_PROTOCOL_VERSION)
        await self._post_notification("notifications/initialized")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = await self._post_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )
        result = self._extract_result(response)
        if not isinstance(result, dict):
            raise RawMCPError(f"Invalid tool response for {tool_name}.")
        return result

    async def _post_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        include_protocol_header: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "jsonrpc": JSONRPC_VERSION,
            "id": self._next_request_id(),
            "method": method,
            "params": params,
        }
        response = await self._client.post(self._url, json=payload, headers=self._mcp_headers(include_protocol_header))
        response.raise_for_status()
        self._remember_session(response)
        return self._parse_response_payload(response)

    async def _post_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        response = await self._client.post(self._url, json=payload, headers=self._mcp_headers())
        response.raise_for_status()
        self._remember_session(response)

    def _mcp_headers(self, include_protocol_header: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._session_id:
            headers[MCP_SESSION_ID_HEADER] = self._session_id
        if include_protocol_header and self._protocol_version:
            headers[MCP_PROTOCOL_VERSION_HEADER] = self._protocol_version
        return headers

    def _remember_session(self, response: httpx.Response) -> None:
        session_id = response.headers.get(MCP_SESSION_ID_HEADER)
        if session_id:
            self._session_id = session_id

    def _parse_response_payload(self, response: httpx.Response) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("application/json"):
            return response.json()
        if content_type.startswith("text/event-stream"):
            return self._parse_sse_payload(response.text)
        raise RawMCPError(f"Unsupported MCP response content type: {content_type or 'unknown'}")

    def _parse_sse_payload(self, body: str) -> dict[str, Any]:
        data_lines: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            raise RawMCPError("No JSON-RPC payload found in SSE response.")
        try:
            return json.loads("\n".join(data_lines))
        except json.JSONDecodeError as exc:
            raise RawMCPError(f"Invalid JSON in SSE response: {exc}") from exc

    def _extract_result(self, payload: dict[str, Any]) -> Any:
        if "error" in payload:
            error = payload["error"] or {}
            code = error.get("code", "unknown")
            message = error.get("message", "Unknown MCP error")
            raise RawMCPError(f"MCP error {code}: {message}")
        return payload.get("result")

    def _next_request_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id


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
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Answer a GSRS question through a selected MCP query tool."""
        selected_tool = tool_name or self.valves.DEFAULT_QUERY_TOOL
        await self._emit_status(__event_emitter__, f"Calling {selected_tool}...", done=False)
        result = await self._call_tool(selected_tool, {"query": question})
        await self._emit_status(__event_emitter__, f"{selected_tool} completed.", done=True)
        return result

    async def retrieve_evidence(
        self,
        query: str,
        debug: bool = False,
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Retrieve raw GSRS evidence chunks without answer synthesis."""
        await self._emit_status(__event_emitter__, "Calling gsrs_retrieve...", done=False)
        result = await self._call_tool("gsrs_retrieve", {"query": query, "debug": debug})
        await self._emit_status(__event_emitter__, "gsrs_retrieve completed.", done=True)
        return result

    async def find_similar_substances(
        self,
        substance_json: str,
        top_k: int = 10,
        match_mode: Literal["contains", "match"] = "contains",
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Find GSRS substances similar to a provided GSRS JSON payload."""
        await self._emit_status(__event_emitter__, "Calling gsrs_similarity_search...", done=False)
        result = await self._call_tool(
            "gsrs_similarity_search",
            {
                "substance_json": substance_json,
                "top_k": top_k,
                "match_mode": match_mode,
            },
        )
        await self._emit_status(__event_emitter__, "gsrs_similarity_search completed.", done=True)
        return result

    async def check_health(
        self,
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Return the current GSRS MCP runtime health payload."""
        await self._emit_status(__event_emitter__, "Calling gsrs_health...", done=False)
        result = await self._call_tool("gsrs_health", {})
        await self._emit_status(__event_emitter__, "gsrs_health completed.", done=True)
        return result

    async def get_document(
        self,
        substance_uuid: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Fetch a full GSRS substance JSON document by UUID."""
        await self._emit_status(__event_emitter__, "Calling gsrs_get_document...", done=False)
        result = await self._call_tool("gsrs_get_document", {"substance_uuid": substance_uuid})
        await self._emit_status(__event_emitter__, "gsrs_get_document completed.", done=True)
        return result

    async def get_substance_schema(
        self,
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Return the GSRS substance JSON schema exposed by the MCP server."""
        await self._emit_status(__event_emitter__, "Calling gsrs_api_substance_schema...", done=False)
        result = await self._call_tool("gsrs_api_substance_schema", {})
        await self._emit_status(__event_emitter__, "gsrs_api_substance_schema completed.", done=True)
        return result

    async def api_search(
        self,
        query: str,
        page: int = 1,
        size: int = 20,
        fields: str = "",
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Search the official GSRS API text endpoint."""
        await self._emit_status(__event_emitter__, "Calling gsrs_api_search...", done=False)
        result = await self._call_tool(
            "gsrs_api_search",
            {
                "query": query,
                "page": page,
                "size": size,
                "fields": fields,
            },
        )
        await self._emit_status(__event_emitter__, "gsrs_api_search completed.", done=True)
        return result

    async def api_structure_search(
        self,
        smiles: str = "",
        inchi: str = "",
        search_type: Literal["EXACT", "SIMILAR", "SUBSTRUCTURE", "SUPERSTRUCTURE"] = "EXACT",
        size: int = 20,
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Search the official GSRS API by chemical structure."""
        await self._emit_status(__event_emitter__, "Calling gsrs_api_structure_search...", done=False)
        result = await self._call_tool(
            "gsrs_api_structure_search",
            {
                "smiles": smiles,
                "inchi": inchi,
                "search_type": search_type,
                "size": size,
            },
        )
        await self._emit_status(__event_emitter__, "gsrs_api_structure_search completed.", done=True)
        return result

    async def api_sequence_search(
        self,
        sequence: str,
        search_type: Literal["EXACT", "CONTAINS", "SIMILAR"] = "EXACT",
        sequence_type: Literal["PROTEIN", "NUCLEIC_ACID"] = "PROTEIN",
        size: int = 20,
        __user__: Optional[dict] = None,
        __event_emitter__: Any = None,
    ) -> str:
        """Search the official GSRS API by protein or nucleic-acid sequence."""
        await self._emit_status(__event_emitter__, "Calling gsrs_api_sequence_search...", done=False)
        result = await self._call_tool(
            "gsrs_api_sequence_search",
            {
                "sequence": sequence,
                "search_type": search_type,
                "sequence_type": sequence_type,
                "size": size,
            },
        )
        await self._emit_status(__event_emitter__, "gsrs_api_sequence_search completed.", done=True)
        return result

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        timeout = float(self.valves.HTTP_TIMEOUT_SECONDS)
        async with RawMCPClient(
            self.valves.MCP_URL,
            self.valves.MCP_TOKEN,
            timeout,
            client_name="gsrs_tool.py",
        ) as client:
            result = await client.call_tool(tool_name, arguments)
            return self._result_to_text(result)

    def _result_to_text(self, result: dict[str, Any]) -> str:
        structured_content = result.get("structuredContent")
        if structured_content:
            return json.dumps(structured_content, indent=2)

        content = result.get("content", [])
        lines: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                lines.append(str(block.get("text", "")))
            else:
                lines.append(json.dumps(block, indent=2))
        return "\n".join(lines)

    async def _emit_status(
        self,
        emitter: Any,
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
