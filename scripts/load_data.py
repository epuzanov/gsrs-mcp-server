#!/usr/bin/env python3
"""
GSRS MCP Server - Data Loader Script

Loads substances into the GSRS database via the MCP Server's **gsrs_ingest** tool
using the official MCP Python SDK client (streamable-http transport).

Sources:
1. *.gsrs files (JSONL.gz format with two leading tabs per line)
2. UUID list - fetch substances from the official GSRS API and ingest
3. --all - fetch all substances from the GSRS API and ingest

Usage:
    # Load from .gsrs file
    python scripts/load_data.py data/substances.gsrs --batch-size 100

    # Load specific substances by UUID from GSRS server
    python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028

    # Load all substances from GSRS server
    python scripts/load_data.py --all

    # Dry run (parse only, don't upload)
    python scripts/load_data.py data/substances.gsrs --dry-run

    # Custom MCP Server URL
    python scripts/load_data.py --uuids abc-123 --mcp-url http://localhost:9000/mcp

    # Bearer token authentication
    python scripts/load_data.py --uuids abc-123 --bearer-token my_token

    # Disable TLS certificate validation for HTTPS endpoints
    python scripts/load_data.py --all --insecure
"""

import argparse
import asyncio
import gzip
import httpx
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Official GSRS API Configuration
GSRS_BASE_URL = "https://gsrs.ncats.nih.gov/api/v1"
GSRS_SEARCH_URL = f"{GSRS_BASE_URL}/substances/search"
GSRS_SUBSTANCE_URL = f"{GSRS_BASE_URL}/substances"

# MCP Server default
DEFAULT_MCP_URL = "http://localhost:8000/mcp"


# ---------------------------------------------------------------------------
# GSRS API helpers (fetch substances from the official public API)
# ---------------------------------------------------------------------------

def gsrs_async_client(verify_ssl: bool, timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, verify=verify_ssl)


def parse_gsrs_file(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Parse a .gsrs file (JSONL.gz format, lines prefixed with two tabs)."""
    logger.info(f"Opening file: {file_path}")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.lstrip('\t\t')
            if not stripped.strip():
                continue
            try:
                yield json.loads(stripped)
                if line_number % 1000 == 0:
                    logger.info(f"Processed {line_number} lines...")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_number}: {e}")
    logger.info(f"Finished parsing {line_number} lines")


async def fetch_substance_by_uuid(
    uuid: str,
    session: httpx.AsyncClient,
) -> Optional[Dict[str, Any]]:
    """Fetch a single substance from the official GSRS API by UUID."""
    url = f"{GSRS_SUBSTANCE_URL}({uuid})?view=full"
    try:
        resp = await session.get(url, timeout=30.0)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Failed to fetch {uuid}: HTTP {resp.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch {uuid}: {e}")
        return None


async def fetch_all_substance_uuids(
    session: httpx.AsyncClient,
    max_results: int = 10000,
) -> List[str]:
    """Page through the GSRS search API to collect substance UUIDs."""
    uuids: List[str] = []
    page = 1
    page_size = 1000
    while len(uuids) < max_results:
        try:
            resp = await session.get(
                GSRS_SEARCH_URL,
                params={"page": page, "size": page_size, "fields": "uuid"},
                timeout=30.0,
            )
            if resp.status_code != 200:
                logger.warning(f"Search page {page} failed: HTTP {resp.status_code}")
                break
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for item in results:
                u = item.get("uuid")
                if u:
                    uuids.append(u)
            logger.info(f"Fetched page {page}, total UUIDs: {len(uuids)}")
            if len(results) < page_size:
                break
            page += 1
        except Exception as e:
            logger.warning(f"Error fetching page {page}: {e}")
            break
    logger.info(f"Total UUIDs fetched: {len(uuids)}")
    return uuids


# ---------------------------------------------------------------------------
# MCP SDK client helpers
# ---------------------------------------------------------------------------

async def mcp_health(mcp_url: str, verify_ssl: bool, bearer_token: Optional[str] = None) -> Dict[str, Any]:
    """Check MCP server health via /health endpoint."""
    base_url = mcp_url.rstrip("/").removesuffix("/mcp")
    headers = {"Accept": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    async with httpx.AsyncClient(verify=verify_ssl, timeout=30.0) as client:
        resp = await client.get(f"{base_url}/health", headers=headers)
        resp.raise_for_status()
        return resp.json()


async def ingest_substance(
    mcp_url: str,
    verify_ssl: bool,
    substance: Dict[str, Any],
    bearer_token: Optional[str] = None,
) -> str:
    """Ingest a single substance via the MCP SDK streamable-http client."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    async with httpx.AsyncClient(
        headers=headers,
        verify=verify_ssl,
        timeout=httpx.Timeout(120.0, read=120.0),
    ) as http_client:
        async with streamable_http_client(
            mcp_url,
            http_client=http_client,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "gsrs_ingest",
                    {"substance_json": json.dumps(substance)},
                )
    # Extract text from CallToolResult
    texts = []
    for block in result.content:
        if hasattr(block, 'text'):
            texts.append(block.text)
        else:
            texts.append(str(block))
    return "\n".join(texts)


async def ingest_substance_batch(
    mcp_url: str,
    verify_ssl: bool,
    substances: List[Dict[str, Any]],
    bearer_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version: ingest a batch via MCP SDK."""
    successful = 0
    failed = 0
    total_chunks = 0
    errors: List[str] = []

    tasks = [
        ingest_substance(mcp_url, verify_ssl, sub, bearer_token)
        for sub in substances
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            failed += 1
            errors.append(f"Substance {idx}: {result}")
            continue
        m = re.search(r"Ingested \*\*(.+?)\*\*.*?(\d+) chunks", result)
        if m:
            total_chunks += int(m.group(2))
            successful += 1
        else:
            failed += 1
            errors.append(f"Substance {idx}: {result}")

    return {
        "total_substances": len(substances),
        "total_chunks": total_chunks,
        "successful": successful,
        "failed": failed,
        "errors": errors,
    }


def ingest_batch_via_mcp(
    mcp_url: str,
    verify_ssl: bool,
    substances: List[Dict[str, Any]],
    bearer_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest a batch of substances via MCP SDK."""
    try:
        loop = asyncio.get_running_loop()
        # Inside running loop: use ensure_future + run_until_complete won't work,
        # so we create a new thread or use nest_asyncio. Simpler: create new loop
        # in a separate thread is overkill. Just use nest.
        # Fallback: create a new event loop in a subprocess-like fashion.
        # Actually the simplest: just use a new loop with nest_asyncio or
        # run the async work directly. But we can't await from sync.
        # Solution: run in a new thread with its own loop.
        import threading
        result_container: List[Dict[str, Any]] = [{}]

        def _run():
            new_loop = asyncio.new_event_loop()
            try:
                result_container[0] = new_loop.run_until_complete(
                    ingest_substance_batch(mcp_url, verify_ssl, substances, bearer_token)
                )
            finally:
                new_loop.close()

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        return result_container[0]
    except RuntimeError:
        # No running event loop – create one
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                ingest_substance_batch(mcp_url, verify_ssl, substances, bearer_token)
            )
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

async def load_substances_from_api(
    mcp_url: str,
    uuids: List[str],
    batch_size: int = 100,
    dry_run: bool = False,
    verify_ssl: bool = True,
    bearer_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch substances from GSRS API and ingest via MCP SDK."""
    stats = {
        "total_substances": 0,
        "downloaded": 0,
        "successful": 0,
        "failed": 0,
        "total_chunks": 0,
        "errors": [],
    }

    logger.info(f"Downloading {len(uuids)} substances from GSRS API...")
    async with gsrs_async_client(verify_ssl=verify_ssl) as session:
        tasks = [fetch_substance_by_uuid(u, session) for u in uuids]
        results = await asyncio.gather(*tasks)

    substances = [s for s in results if s is not None]
    stats["downloaded"] = len(substances)
    stats["total_substances"] = len(uuids)
    logger.info(f"Downloaded {len(substances)}/{len(uuids)} substances")

    if dry_run:
        stats["successful"] = len(substances)
        return stats

    # Check MCP server availability
    try:
        health = await mcp_health(mcp_url, verify_ssl, bearer_token)
        logger.info(
            f"MCP Server: {health.get('total_chunks', 0)} chunks, "
            f"{health.get('total_substances', 0)} substances"
        )
    except Exception as e:
        logger.error(f"MCP server not available: {e}")
        stats["errors"].append(f"MCP server not available: {e}")
        return stats

    # Ingest via MCP SDK
    batch: List[Dict[str, Any]] = []
    for substance in substances:
        batch.append(substance)
        if len(batch) >= batch_size:
            result = ingest_batch_via_mcp(mcp_url, verify_ssl, batch, bearer_token)
            stats["successful"] += result["successful"]
            stats["failed"] += result["failed"]
            stats["total_chunks"] += result["total_chunks"]
            stats["errors"].extend(result["errors"])
            logger.info(
                f"Batch: {result['successful']} ok, "
                f"{result['total_chunks']} chunks"
            )
            batch = []

    if batch:
        result = ingest_batch_via_mcp(mcp_url, verify_ssl, batch, bearer_token)
        stats["successful"] += result["successful"]
        stats["failed"] += result["failed"]
        stats["total_chunks"] += result["total_chunks"]
        stats["errors"].extend(result["errors"])

    return stats


def load_from_file(
    mcp_url: str,
    file_path: str,
    batch_size: int = 100,
    dry_run: bool = False,
    verify_ssl: bool = True,
    bearer_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a .gsrs file and ingest substances via MCP SDK."""
    stats = {
        "total_substances": 0,
        "successful": 0,
        "failed": 0,
        "total_chunks": 0,
        "errors": [],
    }

    if not dry_run:
        try:
            loop = asyncio.new_event_loop()
            health = loop.run_until_complete(
                mcp_health(mcp_url, verify_ssl, bearer_token)
            )
            loop.close()
            logger.info(
                f"MCP Server: {health.get('total_chunks', 0)} chunks, "
                f"{health.get('total_substances', 0)} substances"
            )
        except Exception as e:
            logger.error(f"MCP server not available: {e}")
            stats["errors"].append(f"MCP server not available: {e}")
            return stats

    batch: List[Dict[str, Any]] = []
    logger.info(f"Starting load with batch size: {batch_size}")

    for substance in parse_gsrs_file(file_path):
        batch.append(substance)
        if len(batch) >= batch_size:
            stats["total_substances"] += len(batch)
            if dry_run:
                logger.info(f"[DRY RUN] Would process batch of {len(batch)} substances")
                stats["successful"] += len(batch)
            else:
                result = ingest_batch_via_mcp(mcp_url, verify_ssl, batch, bearer_token)
                stats["successful"] += result["successful"]
                stats["failed"] += result["failed"]
                stats["total_chunks"] += result["total_chunks"]
                stats["errors"].extend(result["errors"])
                logger.info(
                    f"Batch: {result['successful']} ok, "
                    f"{result['total_chunks']} chunks"
                )
            batch = []

    if batch:
        stats["total_substances"] += len(batch)
        if dry_run:
            logger.info(f"[DRY RUN] Would process final batch of {len(batch)} substances")
            stats["successful"] += len(batch)
        else:
            result = ingest_batch_via_mcp(mcp_url, verify_ssl, batch, bearer_token)
            stats["successful"] += result["successful"]
            stats["failed"] += result["failed"]
            stats["total_chunks"] += result["total_chunks"]
            stats["errors"].extend(result["errors"])

    return stats


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(stats: Dict[str, Any]) -> None:
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load GSRS substances via MCP Server (MCP SDK client)"
    )

    # MCP Server connection
    parser.add_argument(
        "--mcp-url",
        default=DEFAULT_MCP_URL,
        help="MCP Server streamable-http URL (default: %(default)s)",
    )

    # Input source (mutually exclusive)
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
        help="Disable TLS certificate validation for GSRS HTTPS requests",
    )
    parser.add_argument(
        "--bearer-token",
        type=str,
        default=None,
        help="Bearer token for MCP Server authentication",
    )

    args = parser.parse_args()
    verify_ssl = not args.insecure

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)

    if args.file:
        stats = load_from_file(
            mcp_url=args.mcp_url,
            file_path=args.file,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            verify_ssl=verify_ssl,
            bearer_token=args.bearer_token,
        )
    elif args.uuids:
        uuid_list = [u.strip() for u in args.uuids.split(",") if u.strip()]
        if not uuid_list:
            logger.error("No valid UUIDs provided")
            sys.exit(1)
        logger.info(f"Loading {len(uuid_list)} substances from GSRS API...")
        loop = asyncio.new_event_loop()
        try:
            stats = loop.run_until_complete(load_substances_from_api(
                mcp_url=args.mcp_url,
                uuids=uuid_list,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                verify_ssl=verify_ssl,
                bearer_token=args.bearer_token,
            ))
        finally:
            loop.close()
    elif args.all:
        logger.info(f"Fetching all substance UUIDs from GSRS API (max: {args.max_results})...")

        async def fetch_and_load() -> Dict[str, Any]:
            async with gsrs_async_client(verify_ssl=verify_ssl) as session:
                uuids = await fetch_all_substance_uuids(session, args.max_results)
            if not uuids:
                logger.error("No UUIDs fetched")
                return {
                    "total_substances": 0, "downloaded": 0,
                    "successful": 0, "failed": 0,
                    "total_chunks": 0,
                    "errors": ["No UUIDs fetched"],
                }
            return await load_substances_from_api(
                mcp_url=args.mcp_url,
                uuids=uuids,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                verify_ssl=verify_ssl,
                bearer_token=args.bearer_token,
            )

        loop = asyncio.new_event_loop()
        try:
            stats = loop.run_until_complete(fetch_and_load())
        finally:
            loop.close()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Load from .gsrs file")
        print("  python scripts/load_data.py data/substances.gsrs")
        print()
        print("  # Load specific substances by UUID")
        print("  python scripts/load_data.py --uuids 0103a288-6eb6-4ced-b13a-849cd7edf028")
        print()
        print("  # Load all substances from GSRS server")
        print("  python scripts/load_data.py --all")
        print()
        print("  # Custom MCP Server URL with Bearer token")
        print("  python scripts/load_data.py --uuids abc-123 --mcp-url http://myserver:9000/mcp --bearer-token my_token")
        print()
        print("  # Disable TLS certificate validation")
        print("  python scripts/load_data.py --all --insecure")
        sys.exit(1)

    print_summary(stats)
    if stats["failed"] > 0 or stats["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
