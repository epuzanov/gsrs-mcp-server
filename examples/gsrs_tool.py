"""
title: GSRS MCP Server Tool
author: Egor Puzanov
version: 1.0
description: Query GSRS substance database via the MCP server (stdio or SSE transport).
"""

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from typing import Optional
import httpx
import json
import re
import os


class Tools:
    class Valves(BaseModel):
        mcp_transport: str = Field(
            default="sse",
            description="MCP transport: 'sse' (HTTP) or 'stdio' (local subprocess)"
        )
        mcp_url: str = Field(
            default="http://gsrs-mcp-server:8000/mcp",
            description="MCP server SSE URL (used when transport=sse)"
        )
        mcp_command: str = Field(
            default="gsrs-gateway",
            description="MCP server command (used when transport=stdio)"
        )
        api_username: str = Field(
            default="admin",
            description="Username for MCP server authentication"
        )
        api_password: str = Field(
            default="admin123",
            description="Password for MCP server authentication"
        )
        top_k: int = Field(
            default=20,
            description="Number of retrieval results to fetch"
        )
        answer_style: str = Field(
            default="standard",
            description="Answer style: concise, standard, or detailed"
        )
        return_evidence: bool = Field(
            default=True,
            description="Include evidence chunks in the response"
        )
        evidence_limit: int = Field(
            default=5,
            description="Maximum number of evidence chunks to show"
        )
        request_timeout: int = Field(
            default=60,
            description="Request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()

    def gsrs_substance_query(
        self,
        query: str = Field(
            ...,
            description="Search query about chemical substances (e.g., 'CAS code for Aspirin', 'molecular weight of Ibuprofen')"
        ),
        top_k: Optional[int] = Field(
            default=None,
            description="Number of results to return (default: uses valve setting)"
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Query the GSRS substance database via the MCP server.

        Use this tool when the user asks about:
        - Chemical codes (CAS, UNII, ChEMBL, etc.)
        - Molecular properties (weight, formula, etc.)
        - Substance names and synonyms
        - Protein or nucleic acid structures
        - Any GSRS substance data

        **File Upload Support:**
        - Upload a GSRS JSON file (.json) via the Open WebUI file upload button
        - The tool will automatically detect the uploaded file and run similarity search
        - You can also paste GSRS JSON directly into the query

        Args:
            query: The search query about a chemical substance
            top_k: Optional number of results (default: 5)

        Returns:
            Formatted search results, a generated answer, or similar substances
        """
        if isinstance(query, FieldInfo):
            query = "substance query"
        if isinstance(top_k, FieldInfo):
            top_k = None

        try:
            k = top_k if top_k is not None else self.valves.top_k

            # Step 1: Check for uploaded files (Open WebUI file upload support)
            uploaded_substance = self._extract_uploaded_file(__user__)
            if uploaded_substance and self._is_gsrs_substance(uploaded_substance):
                return self._call_mcp_tool("gsrs_similarity_search", {
                    "substance_json": json.dumps(uploaded_substance),
                    "top_k": k,
                })

            # Step 2: Check if query contains GSRS JSON
            substance_data = self._try_parse_json(query)
            if substance_data and self._is_gsrs_substance(substance_data):
                return self._call_mcp_tool("gsrs_similarity_search", {
                    "substance_json": json.dumps(substance_data),
                    "top_k": k,
                })

            # Step 3: Use MCP server for Q&A
            return self._call_mcp_tool("gsrs_ask", {
                "query": query,
                "top_k": k,
                "answer_style": self.valves.answer_style,
                "return_evidence": self.valves.return_evidence,
            })

        except httpx.TimeoutException:
            return "Timeout: The GSRS MCP server query took too long."
        except httpx.ConnectError:
            return (f"Connection error: Could not connect to GSRS MCP server "
                    f"at {self.valves.mcp_url}. Make sure the service is running.")
        except Exception as e:
            return f"Error querying GSRS database: {str(e)}"

    def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call an MCP tool via SSE transport using HTTP POST.

        For SSE transport, we use the MCP JSON-RPC endpoint at /mcp/message.
        For stdio transport, falls back to direct REST API calls.
        """
        if self.valves.mcp_transport == "sse":
            return self._call_mcp_sse(tool_name, arguments)
        else:
            return self._call_mcp_rest(tool_name, arguments)

    def _call_mcp_sse(self, tool_name: str, arguments: dict) -> str:
        """Call MCP tool via SSE transport."""
        # MCP JSON-RPC over HTTP: POST to /mcp/message
        # For simplicity, we use the REST API that the MCP server exposes
        # alongside SSE, or fall back to direct HTTP calls to the FastAPI health endpoint
        # to verify connectivity first.
        url = self.valves.mcp_url.rstrip("/")
        base_url = url.rsplit("/mcp", 1)[0]

        # Verify server is up
        with httpx.Client() as client:
            resp = client.get(f"{base_url}/health", timeout=10)
            resp.raise_for_status()

        # Call the MCP tool via the server's tool endpoint
        # Since MCP tools are invoked by the LLM, we use the REST
        # API that mirrors MCP tool functionality.
        if tool_name == "gsrs_ask":
            endpoint = f"{base_url}/mcp/tools/gsrs_ask"
            payload = {
                "query": arguments.get("query", ""),
                "top_k": arguments.get("top_k", 10),
                "answer_style": arguments.get("answer_style", "standard"),
                "return_evidence": arguments.get("return_evidence", True),
            }
        elif tool_name == "gsrs_similarity_search":
            endpoint = f"{base_url}/mcp/tools/gsrs_similarity_search"
            payload = {
                "substance_json": arguments.get("substance_json", "{}"),
                "top_k": arguments.get("top_k", 10),
                "match_mode": arguments.get("match_mode", "contains"),
            }
        else:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        with httpx.Client() as client:
            resp = client.post(
                endpoint,
                json=payload,
                timeout=self.valves.request_timeout,
                auth=(self.valves.api_username, self.valves.api_password),
            )
            resp.raise_for_status()
            return resp.text

    def _call_mcp_rest(self, tool_name: str, arguments: dict) -> str:
        """Fallback: call the GSRS REST API directly (for backward compatibility)."""
        url = self.valves.mcp_url.rstrip("/")
        base_url = url.rsplit("/mcp", 1)[0]

        if tool_name == "gsrs_ask":
            return self._call_ask(base_url, arguments)
        elif tool_name == "gsrs_similarity_search":
            return self._call_similarity(base_url, arguments)
        return f"Unknown tool: {tool_name}"

    def _call_ask(self, base_url: str, arguments: dict) -> str:
        """Call /ask endpoint."""
        with httpx.Client() as client:
            resp = client.post(
                f"{base_url}/ask",
                json={
                    "query": arguments.get("query", ""),
                    "top_k": arguments.get("top_k", 10),
                    "answer_style": arguments.get("answer_style", "standard"),
                    "return_evidence": arguments.get("return_evidence", True),
                },
                timeout=self.valves.request_timeout,
                auth=(self.valves.api_username, self.valves.api_password),
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("abstained"):
            return f"GSRS could not answer: {data.get('abstain_reason', 'Insufficient evidence.')}"

        answer = data.get("answer", "No answer available.")
        citations = data.get("citations", [])
        evidence = data.get("evidence_chunks", [])

        result = [answer, f"\nConfidence: {data.get('confidence', 0):.2f}"]

        if citations:
            result.append("\nCitations:")
            for i, c in enumerate(citations[:self.valves.evidence_limit], 1):
                result.append(f"  [{i}] {c.get('section', 'N/A')}")

        if evidence and self.valves.return_evidence:
            result.append(f"\nEvidence (top {min(len(evidence), self.valves.evidence_limit)}):")
            for i, ch in enumerate(evidence[:self.valves.evidence_limit], 1):
                text = ch.get("text", "")[:200]
                result.append(f"  [{i}] ({ch.get('element_path', '')}) {text}")

        return "\n".join(result)

    def _call_similarity(self, base_url: str, arguments: dict) -> str:
        """Call /substances/similar endpoint."""
        import json as _json
        try:
            substance = _json.loads(arguments.get("substance_json", "{}"))
        except _json.JSONDecodeError:
            return "Error: Invalid JSON in substance_json."

        with httpx.Client() as client:
            resp = client.post(
                f"{base_url}/substances/similar",
                json={
                    "substance": substance,
                    "top_k": arguments.get("top_k", 10),
                    "match_mode": arguments.get("match_mode", "contains"),
                    "exclude_self": True,
                },
                timeout=self.valves.request_timeout,
                auth=(self.valves.api_username, self.valves.api_password),
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        name = data.get("query_substance_name", "the provided substance")

        if not results:
            return f"No similar substances found for {name}."

        lines = [f"Found {len(results)} substance(s) similar to **{name}**:\n"]
        for i, r in enumerate(results, 1):
            n = r.get("canonical_name", r.get("substance_uuid", "Unknown"))
            lines.append(f"{i}. **{n}** (score {r.get('match_score', 0):.2f})")
            matched = r.get("matched_fields", [])
            if matched:
                lines.append(f"   Matched: {', '.join(matched[:5])}")
            chunks = r.get("chunks", [])
            if chunks:
                lines.append(f"   Chunks: {len(chunks)} — {chunks[0].get('text', '')[:120]}...")

        return "\n".join(lines)

    def _extract_uploaded_file(self, user: Optional[dict]) -> Optional[dict]:
        """Extract GSRS substance JSON from uploaded files in Open WebUI."""
        if not user:
            return None

        files = user.get("files", [])
        if not isinstance(files, list):
            files = [files] if files else []

        files_content = user.get("files_content") or user.get("file_content")
        if files_content and isinstance(files_content, (dict, str)):
            if isinstance(files_content, str):
                files.append({"content": files_content, "filename": "uploaded.json"})
            elif isinstance(files_content, dict):
                files.append(files_content)

        for file_info in files:
            if not isinstance(file_info, dict):
                continue

            content = (
                file_info.get("content")
                or file_info.get("text")
                or file_info.get("data")
            )
            if content:
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="ignore")
                parsed = self._try_parse_json(content)
                if parsed and self._is_gsrs_substance(parsed):
                    return parsed

            file_path = file_info.get("path") or file_info.get("filename")
            if file_path:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        parsed = self._try_parse_json(content)
                        if parsed and self._is_gsrs_substance(parsed):
                            return parsed
                except (OSError, IOError, UnicodeDecodeError):
                    pass

        return None

    def _try_parse_json(self, text: str) -> Optional[dict]:
        """Try to parse text as JSON."""
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def _is_gsrs_substance(self, data: dict) -> bool:
        """Check if the provided JSON looks like a GSRS substance document."""
        gsrs_keys = {"uuid", "names", "codes", "classifications", "relationships",
                     "structure", "properties", "references", "notes"}
        return len(gsrs_keys.intersection(data.keys())) >= 2
