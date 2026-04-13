"""Use the OpenAI Responses API with the GSRS MCP server as a remote MCP tool."""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call the OpenAI Responses API with the GSRS MCP server attached as a remote MCP tool."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--mcp-url", default="http://localhost:8000/mcp")
    parser.add_argument("--mcp-token", default="")
    parser.add_argument(
        "--allowed-tool",
        action="append",
        dest="allowed_tools",
        default=[],
        help="Limit the imported MCP tools. Repeat to allow multiple tools.",
    )
    parser.add_argument(
        "--require-approval",
        choices=["always", "never"],
        default="never",
        help="OpenAI API approval mode for MCP tool calls.",
    )
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    tool = {
        "type": "mcp",
        "server_label": "gsrs",
        "server_description": "GSRS substance retrieval and grounded-answer MCP server.",
        "server_url": args.mcp_url,
        "require_approval": args.require_approval,
    }
    if args.mcp_token:
        # OpenAI's remote MCP tool accepts an authorization value for upstream auth.
        # The GSRS server expects a bearer token, so we pass the full header value.
        tool["authorization"] = f"Bearer {args.mcp_token}"
    if args.allowed_tools:
        tool["allowed_tools"] = args.allowed_tools

    response = client.responses.create(
        model=args.model,
        input=args.query,
        tools=[tool],
    )

    print(response.output_text)
    print("\nTool trace:")
    print(json.dumps(response.output, indent=2))


if __name__ == "__main__":
    main()
