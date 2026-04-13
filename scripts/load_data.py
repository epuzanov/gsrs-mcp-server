#!/usr/bin/env python3
"""
GSRS MCP Server - Data Loader Script

Loads substances into the GSRS database via the MCP server's `gsrs_ingest`
tool using the official MCP Python client library.

Sources:
1. `*.gsrs` files (JSONL.gz format with two leading tabs per line)
2. UUID list - fetch substances from the official GSRS API and ingest
3. `--all` - fetch all substances from the GSRS API and ingest

Usage:
    # Load from .gsrs file over MCP HTTP
    python scripts/load_data.py data/substances.gsrs --batch-size 100

    # Load specific substances by UUID from GSRS server
    python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028

    # Load all substances from GSRS server
    python scripts/load_data.py --all

    # Dry run (parse only, don't upload)
    python scripts/load_data.py data/substances.gsrs --dry-run

    # Custom MCP Server URL
    python scripts/load_data.py --uuids abc-123 --mcp-url http://localhost:9000/mcp

    # Use stdio transport instead of HTTP
    python scripts/load_data.py --transport stdio --command gsrs-mcp-server --uuids abc-123

    # Bearer token authentication for streamable HTTP
    python scripts/load_data.py --uuids abc-123 --bearer-token my_token

    # Disable TLS certificate validation for HTTPS endpoints
    python scripts/load_data.py --all --insecure
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Official GSRS API Configuration
GSRS_BASE_URL = "https://gsrs.ncats.nih.gov/api/v1"
GSRS_SEARCH_URL = f"{GSRS_BASE_URL}/substances/search"
GSRS_SUBSTANCE_URL = f"{GSRS_BASE_URL}/substances"

# MCP Server defaults
DEFAULT_MCP_URL = "http://localhost:8000/mcp"
DEFAULT_MCP_COMMAND = "gsrs-mcp-server"
DEFAULT_MCP_TRANSPORT = "http"

INGEST_COUNT_PATTERN = re.compile(r"Ingested \*\*(.+?)\*\* - (\d+) chunks\.")


# ---------------------------------------------------------------------------
# GSRS API helpers (fetch substances from the official public API)
# ---------------------------------------------------------------------------

def gsrs_async_client(verify_ssl: bool, timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, verify=verify_ssl)


def parse_gsrs_file(file_path: str) -> Generator[dict[str, Any], None, None]:
    """Parse a .gsrs file (JSONL.gz format, lines prefixed with two tabs)."""
    logger.info("Opening file: %s", file_path)
    line_number = 0
    with gzip.open(file_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.lstrip("\t\t")
            if not stripped.strip():
                continue
            try:
                yield json.loads(stripped)
                if line_number % 1000 == 0:
                    logger.info("Processed %s lines...", line_number)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse line %s: %s", line_number, exc)
    logger.info("Finished parsing %s lines", line_number)


async def fetch_substance_by_uuid(
    uuid: str,
    session: httpx.AsyncClient,
) -> dict[str, Any] | None:
    """Fetch a single substance from the official GSRS API by UUID."""
    url = f"{GSRS_SUBSTANCE_URL}({uuid})?view=full"
    try:
        response = await session.get(url, timeout=30.0)
        if response.status_code == 200:
            return response.json()
        logger.warning("Failed to fetch %s: HTTP %s", uuid, response.status_code)
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to fetch %s: %s", uuid, exc)
        return None


async def fetch_all_substance_uuids(
    session: httpx.AsyncClient,
    max_results: int = 10000,
) -> list[str]:
    """Page through the GSRS search API to collect substance UUIDs."""
    uuids: list[str] = []
    page = 1
    page_size = 1000

    while len(uuids) < max_results:
        try:
            response = await session.get(
                GSRS_SEARCH_URL,
                params={"page": page, "size": page_size, "fields": "uuid"},
                timeout=30.0,
            )
            if response.status_code != 200:
                logger.warning("Search page %s failed: HTTP %s", page, response.status_code)
                break

            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            for item in results:
                substance_uuid = item.get("uuid")
                if substance_uuid:
                    uuids.append(substance_uuid)
                    if len(uuids) >= max_results:
                        break

            logger.info("Fetched page %s, total UUIDs: %s", page, len(uuids))
            if len(results) < page_size:
                break
            page += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Error fetching page %s: %s", page, exc)
            break

    logger.info("Total UUIDs fetched: %s", len(uuids))
    return uuids


# ---------------------------------------------------------------------------
# MCP client helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCPConnectionSettings:
    transport: str = DEFAULT_MCP_TRANSPORT
    mcp_url: str = DEFAULT_MCP_URL
    command: str = DEFAULT_MCP_COMMAND
    verify_ssl: bool = True
    bearer_token: str | None = None


class MCPToolClient:
    """Thin MCP client wrapper used by the loader."""

    def __init__(self, connection: MCPConnectionSettings):
        self.connection = connection
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tool_names: set[str] = set()

    async def __aenter__(self) -> "MCPToolClient":
        stack = AsyncExitStack()
        self._stack = stack

        if self.connection.transport == "stdio":
            if self.connection.bearer_token:
                logger.warning("Ignoring bearer token for stdio transport.")
            server = StdioServerParameters(
                command=self.connection.command,
                env={"MCP_TRANSPORT": "stdio"},
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server))
        elif self.connection.transport == "http":
            headers: dict[str, str] = {}
            if self.connection.bearer_token:
                headers["Authorization"] = f"Bearer {self.connection.bearer_token}"
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    headers=headers,
                    timeout=httpx.Timeout(120.0, read=120.0),
                    verify=self.connection.verify_ssl,
                )
            )
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(
                    self.connection.mcp_url,
                    http_client=http_client,
                    terminate_on_close=False,
                )
            )
        else:  # pragma: no cover - argparse enforces valid values
            raise ValueError(f"Unsupported MCP transport: {self.connection.transport}")

        self._session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        tools = await self._session.list_tools()
        self._tool_names = {tool.name for tool in tools.tools}
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self._stack is None:
            return False
        return await self._stack.__aexit__(exc_type, exc, tb)

    async def call_tool_text(self, tool_name: str, arguments: dict[str, Any]) -> str:
        session = self._require_session()
        result = await session.call_tool(tool_name, arguments)
        structured = getattr(result, "structuredContent", None)
        if structured:
            return json.dumps(structured, indent=2)
        return "\n".join(self._result_blocks_to_text(result)).strip()

    async def get_health_payload(self) -> dict[str, Any]:
        self._require_tool("gsrs_health")
        return self._parse_json_payload(
            await self.call_tool_text("gsrs_health", {}),
            "gsrs_health",
        )

    async def get_statistics(self) -> dict[str, Any]:
        self._require_tool("gsrs_statistics")
        text = await self.call_tool_text("gsrs_statistics", {})
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"message": text}

    async def ensure_ingest_available(self) -> None:
        self._require_tool("gsrs_ingest")
        if "gsrs_health" not in self._tool_names:
            return

        payload = await self.get_health_payload()
        components = payload.get("components", {})
        required_components = {
            "vector_db": "Vector backend",
            "embedding": "Embedding provider",
            "chunker": "Chunker",
        }
        for component_name, label in required_components.items():
            component = components.get(component_name, {})
            if component.get("ready"):
                continue
            error = component.get("error") or f"{label} is not ready."
            raise RuntimeError(error)

    async def ingest_substance(self, substance: dict[str, Any]) -> str:
        return await self.call_tool_text(
            "gsrs_ingest",
            {"substance_json": json.dumps(substance)},
        )

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP session has not been initialized.")
        return self._session

    def _require_tool(self, tool_name: str) -> None:
        if tool_name not in self._tool_names:
            raise RuntimeError(f"MCP server does not expose the `{tool_name}` tool.")

    @staticmethod
    def _parse_json_payload(payload: str, tool_name: str) -> dict[str, Any]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"`{tool_name}` returned non-JSON content: {payload}") from exc

    @staticmethod
    def _result_blocks_to_text(result: Any) -> list[str]:
        texts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif hasattr(block, "model_dump"):
                texts.append(json.dumps(block.model_dump(), indent=2))
            else:
                texts.append(str(block))
        return texts


def build_mcp_client(
    *,
    transport: str,
    mcp_url: str,
    command: str,
    verify_ssl: bool,
    bearer_token: str | None,
) -> MCPToolClient:
    return MCPToolClient(
        MCPConnectionSettings(
            transport=transport,
            mcp_url=mcp_url,
            command=command,
            verify_ssl=verify_ssl,
            bearer_token=bearer_token,
        )
    )


def _empty_stats(include_downloaded: bool = False) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total_substances": 0,
        "successful": 0,
        "failed": 0,
        "total_chunks": 0,
        "errors": [],
    }
    if include_downloaded:
        stats["downloaded"] = 0
    return stats


def _parse_ingest_result(result: str) -> int | None:
    match = INGEST_COUNT_PATTERN.search(result)
    if match:
        return int(match.group(2))
    return None


async def _ingest_substance_batch(
    client: MCPToolClient,
    substances: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = 0
    failed = 0
    total_chunks = 0
    errors: list[str] = []

    for index, substance in enumerate(substances):
        substance_label = substance.get("uuid") or f"index {index}"
        try:
            result = await client.ingest_substance(substance)
            chunk_count = _parse_ingest_result(result)
            if chunk_count is None:
                failed += 1
                errors.append(f"Substance {substance_label}: {result}")
                continue
            successful += 1
            total_chunks += chunk_count
        except Exception as exc:
            failed += 1
            errors.append(f"Substance {substance_label}: {exc}")

    return {
        "total_substances": len(substances),
        "total_chunks": total_chunks,
        "successful": successful,
        "failed": failed,
        "errors": errors,
    }


async def ingest_substance_batch(
    mcp_url: str,
    verify_ssl: bool,
    substances: list[dict[str, Any]],
    bearer_token: str | None = None,
    *,
    transport: str = DEFAULT_MCP_TRANSPORT,
    command: str = DEFAULT_MCP_COMMAND,
) -> dict[str, Any]:
    """Ingest a batch of substances through a single MCP session."""
    async with build_mcp_client(
        transport=transport,
        mcp_url=mcp_url,
        command=command,
        verify_ssl=verify_ssl,
        bearer_token=bearer_token,
    ) as client:
        await client.ensure_ingest_available()
        return await _ingest_substance_batch(client, substances)


def ingest_batch_via_mcp(
    mcp_url: str,
    verify_ssl: bool,
    substances: list[dict[str, Any]],
    bearer_token: str | None = None,
    *,
    transport: str = DEFAULT_MCP_TRANSPORT,
    command: str = DEFAULT_MCP_COMMAND,
) -> dict[str, Any]:
    """Synchronous wrapper for batch ingestion through the MCP client."""
    return asyncio.run(
        ingest_substance_batch(
            mcp_url,
            verify_ssl,
            substances,
            bearer_token=bearer_token,
            transport=transport,
            command=command,
        )
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

async def load_substances_from_api(
    mcp_url: str,
    uuids: list[str],
    batch_size: int = 100,
    dry_run: bool = False,
    verify_ssl: bool = True,
    bearer_token: str | None = None,
    *,
    transport: str = DEFAULT_MCP_TRANSPORT,
    command: str = DEFAULT_MCP_COMMAND,
) -> dict[str, Any]:
    """Fetch substances from GSRS API and ingest via MCP."""
    stats = _empty_stats(include_downloaded=True)

    logger.info("Downloading %s substances from GSRS API...", len(uuids))
    async with gsrs_async_client(verify_ssl=verify_ssl) as session:
        tasks = [fetch_substance_by_uuid(uuid, session) for uuid in uuids]
        results = await asyncio.gather(*tasks)

    substances = [substance for substance in results if substance is not None]
    stats["downloaded"] = len(substances)
    stats["total_substances"] = len(uuids)
    logger.info("Downloaded %s/%s substances", len(substances), len(uuids))

    if dry_run:
        stats["successful"] = len(substances)
        return stats

    try:
        async with build_mcp_client(
            transport=transport,
            mcp_url=mcp_url,
            command=command,
            verify_ssl=verify_ssl,
            bearer_token=bearer_token,
        ) as client:
            await client.ensure_ingest_available()
            statistics = await client.get_statistics()
            logger.info(
                "MCP Server: %s chunks, %s substances",
                statistics.get("total_chunks", 0),
                statistics.get("total_substances", 0),
            )

            for batch_start in range(0, len(substances), batch_size):
                batch = substances[batch_start: batch_start + batch_size]
                result = await _ingest_substance_batch(client, batch)
                stats["successful"] += result["successful"]
                stats["failed"] += result["failed"]
                stats["total_chunks"] += result["total_chunks"]
                stats["errors"].extend(result["errors"])
                logger.info(
                    "Batch: %s ok, %s chunks",
                    result["successful"],
                    result["total_chunks"],
                )
    except Exception as exc:
        logger.error("MCP server not available: %s", exc)
        stats["errors"].append(f"MCP server not available: {exc}")

    return stats


async def _load_from_file_async(
    mcp_url: str,
    file_path: str,
    batch_size: int = 100,
    dry_run: bool = False,
    verify_ssl: bool = True,
    bearer_token: str | None = None,
    *,
    transport: str = DEFAULT_MCP_TRANSPORT,
    command: str = DEFAULT_MCP_COMMAND,
) -> dict[str, Any]:
    stats = _empty_stats()

    client: MCPToolClient | None = None
    if not dry_run:
        try:
            client = await build_mcp_client(
                transport=transport,
                mcp_url=mcp_url,
                command=command,
                verify_ssl=verify_ssl,
                bearer_token=bearer_token,
            ).__aenter__()
            await client.ensure_ingest_available()
            statistics = await client.get_statistics()
            logger.info(
                "MCP Server: %s chunks, %s substances",
                statistics.get("total_chunks", 0),
                statistics.get("total_substances", 0),
            )
        except Exception as exc:
            logger.error("MCP server not available: %s", exc)
            stats["errors"].append(f"MCP server not available: {exc}")
            if client is not None:
                await client.__aexit__(None, None, None)
            return stats

    batch: list[dict[str, Any]] = []
    logger.info("Starting load with batch size: %s", batch_size)

    try:
        for substance in parse_gsrs_file(file_path):
            batch.append(substance)
            if len(batch) < batch_size:
                continue

            stats["total_substances"] += len(batch)
            if dry_run:
                logger.info("[DRY RUN] Would process batch of %s substances", len(batch))
                stats["successful"] += len(batch)
            else:
                result = await _ingest_substance_batch(client, batch)
                stats["successful"] += result["successful"]
                stats["failed"] += result["failed"]
                stats["total_chunks"] += result["total_chunks"]
                stats["errors"].extend(result["errors"])
                logger.info(
                    "Batch: %s ok, %s chunks",
                    result["successful"],
                    result["total_chunks"],
                )
            batch = []

        if batch:
            stats["total_substances"] += len(batch)
            if dry_run:
                logger.info("[DRY RUN] Would process final batch of %s substances", len(batch))
                stats["successful"] += len(batch)
            else:
                result = await _ingest_substance_batch(client, batch)
                stats["successful"] += result["successful"]
                stats["failed"] += result["failed"]
                stats["total_chunks"] += result["total_chunks"]
                stats["errors"].extend(result["errors"])
    finally:
        if client is not None:
            await client.__aexit__(None, None, None)

    return stats


def load_from_file(
    mcp_url: str,
    file_path: str,
    batch_size: int = 100,
    dry_run: bool = False,
    verify_ssl: bool = True,
    bearer_token: str | None = None,
    *,
    transport: str = DEFAULT_MCP_TRANSPORT,
    command: str = DEFAULT_MCP_COMMAND,
) -> dict[str, Any]:
    """Parse a .gsrs file and ingest substances via MCP."""
    return asyncio.run(
        _load_from_file_async(
            mcp_url,
            file_path,
            batch_size=batch_size,
            dry_run=dry_run,
            verify_ssl=verify_ssl,
            bearer_token=bearer_token,
            transport=transport,
            command=command,
        )
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(stats: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LOAD SUMMARY")
    print("=" * 60)
    if "downloaded" in stats:
        print(f"Substances downloaded: {stats['downloaded']}/{stats['total_substances']}")
    print(f"Total substances processed: {stats['total_substances']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    verify_ssl = not args.insecure

    if args.file:
        return await _load_from_file_async(
            mcp_url=args.mcp_url,
            file_path=args.file,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            verify_ssl=verify_ssl,
            bearer_token=args.bearer_token,
            transport=args.transport,
            command=args.command,
        )

    if args.uuids:
        uuid_list = [uuid.strip() for uuid in args.uuids.split(",") if uuid.strip()]
        if not uuid_list:
            raise ValueError("No valid UUIDs provided")
        logger.info("Loading %s substances from GSRS API...", len(uuid_list))
        return await load_substances_from_api(
            mcp_url=args.mcp_url,
            uuids=uuid_list,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            verify_ssl=verify_ssl,
            bearer_token=args.bearer_token,
            transport=args.transport,
            command=args.command,
        )

    if args.all:
        logger.info("Fetching all substance UUIDs from GSRS API (max: %s)...", args.max_results)
        async with gsrs_async_client(verify_ssl=verify_ssl) as session:
            uuids = await fetch_all_substance_uuids(session, args.max_results)

        if not uuids:
            logger.error("No UUIDs fetched")
            stats = _empty_stats(include_downloaded=True)
            stats["errors"].append("No UUIDs fetched")
            return stats

        return await load_substances_from_api(
            mcp_url=args.mcp_url,
            uuids=uuids,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            verify_ssl=verify_ssl,
            bearer_token=args.bearer_token,
            transport=args.transport,
            command=args.command,
        )

    raise ValueError("No input source provided")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load GSRS substances via the MCP client library.",
    )

    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default=DEFAULT_MCP_TRANSPORT,
        help="MCP transport to use (default: %(default)s)",
    )
    parser.add_argument(
        "--mcp-url",
        default=DEFAULT_MCP_URL,
        help="MCP streamable-http URL (used when --transport=http)",
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_MCP_COMMAND,
        help="MCP server command (used when --transport=stdio)",
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "file",
        nargs="?",
        type=str,
        help="Path to the .gsrs file (JSONL.gz format)",
    )
    source_group.add_argument(
        "--uuids",
        type=str,
        help="Comma-separated list of substance UUIDs to load from GSRS server",
    )
    source_group.add_argument(
        "--all",
        action="store_true",
        help="Load all substances from GSRS server",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Substances per batch (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse/download only, don't ingest via MCP",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10000,
        help="Max substances to fetch with --all (default: %(default)s)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate validation for GSRS and MCP HTTPS requests",
    )
    parser.add_argument(
        "--bearer-token",
        type=str,
        default=None,
        help="Bearer token for MCP streamable-http authentication",
    )

    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            sys.exit(1)
    elif not args.uuids and not args.all:
        parser.print_help()
        print("\nExamples:")
        print("  # Load from .gsrs file over MCP HTTP")
        print("  python scripts/load_data.py data/substances.gsrs")
        print()
        print("  # Load specific substances by UUID")
        print("  python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028")
        print()
        print("  # Load all substances from GSRS server")
        print("  python scripts/load_data.py --all")
        print()
        print("  # Use stdio transport")
        print("  python scripts/load_data.py --transport stdio --command gsrs-mcp-server --uuids abc-123")
        print()
        print("  # Custom MCP Server URL with bearer token")
        print(
            "  python scripts/load_data.py --uuids abc-123 "
            "--mcp-url http://myserver:9000/mcp --bearer-token my_token"
        )
        sys.exit(1)

    try:
        stats = asyncio.run(_run_cli(args))
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print_summary(stats)
    if stats["failed"] > 0 or stats["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
