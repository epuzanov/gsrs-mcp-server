"""
title: GSRS MCP Server Filter
author: Egor Puzanov
version: 1.0
description: Inject GSRS context into prompts via the MCP server
"""

from pydantic import BaseModel, Field
from typing import Optional
import httpx


class Filter:
    class Valves(BaseModel):
        mcp_url: str = Field(
            default="http://gsrs-mcp-server:8000",
            description="MCP server base URL"
        )
        api_username: str = Field(
            default="admin",
            description="Username for MCP server authentication"
        )
        api_password: str = Field(
            default="admin123",
            description="Password for MCP server authentication"
        )
        mode: str = Field(
            default="evidence",
            description="Mode: 'evidence' (inject chunks) or 'answer_assist' (inject draft answer)"
        )
        top_k: int = Field(
            default=10,
            description="Number of retrieval results to fetch"
        )
        max_evidence_chars: int = Field(
            default=3000,
            description="Maximum characters for evidence injection"
        )
        min_query_length: int = Field(
            default=5,
            description="Minimum query length to trigger GSRS lookup"
        )
        include_citations: bool = Field(
            default=True,
            description="Include citation metadata in injected context"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Query GSRS MCP server and add context to the prompt."""
        messages = body.get("messages", [])
        if not messages:
            return body

        query = messages[-1].get("content", "")

        if len(query) < self.valves.min_query_length:
            return body

        try:
            if self.valves.mode == "answer_assist":
                context = await self._call_ask(query, answer_style="standard")
            else:
                context = await self._call_ask(query, answer_style="standard")

            if context:
                body["messages"][-1]["content"] = f"{context}\n\nUser Question: {query}"

        except Exception as e:
            print(f"GSRS MCP error: {e}")

        return body

    async def _call_ask(self, query: str, answer_style: str = "standard") -> Optional[str]:
        """Call the GSRS MCP server /ask endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.valves.mcp_url}/ask",
                    json={
                        "query": query,
                        "top_k": self.valves.top_k,
                        "answer_style": answer_style,
                        "return_evidence": True,
                    },
                    timeout=30,
                    auth=(self.valves.api_username, self.valves.api_password),
                )

                if response.status_code == 200:
                    data = response.json()
                    evidence_chunks = data.get("evidence_chunks", [])
                    citations = data.get("citations", [])
                    abstained = data.get("abstained", False)
                    abstain_reason = data.get("abstain_reason")

                    if abstained:
                        context = f"GSRS MCP note: {abstain_reason or 'Evidence may be weak.'}\n\n"
                    else:
                        context = ""

                    context += "GSRS Database Evidence (answer ONLY from this evidence):\n"
                    context += self._format_evidence_chunks(evidence_chunks, citations)
                    return context

                return None

        except Exception as e:
            print(f"GSRS MCP error: {e}")
            return None

    def _format_evidence_chunks(self, chunks: list, citations: list) -> str:
        """Format evidence chunks for injection."""
        context = ""
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            section = chunk.get('element_path', 'N/A')
            score = chunk.get('similarity_score', 0)
            chunk_text = f"[{i}] (Section: {section}, Score: {score:.2f}) {text}\n\n"

            if total_chars + len(chunk_text) > self.valves.max_evidence_chars:
                break

            context += chunk_text
            total_chars += len(chunk_text)

        if citations and self.valves.include_citations:
            context += "\nCitations:\n"
            for i, cit in enumerate(citations, 1):
                context += f"  [{i}] {cit.get('section', 'N/A')}"
                if cit.get('source_url'):
                    context += f" - {cit['source_url']}"
                context += "\n"

        return context
