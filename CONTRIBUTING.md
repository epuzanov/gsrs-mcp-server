# Contributing

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/gsrs-mcp-server.git
cd gsrs-mcp-server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
pip install pytest pytest-cov pytest-asyncio ruff mypy
copy .env.example .env
```

The editable install registers the `gsrs-mcp-server` CLI used throughout the docs and examples.

## Local Checks

```bash
python -m pytest tests -v
ruff check app tests
mypy app --ignore-missing-imports --disable-error-code=annotation-unchecked
python -m compileall app tests
```

## Development Notes

- This repository is MCP-first. Prefer updating MCP tool docs and health/readiness docs instead of describing REST endpoints.
- The public runtime contract is `gsrs-mcp-server` over `/mcp` or `stdio`, plus `/livez`, `/readyz`, and `/health`.
- HTTP auth on `/mcp` is bearer-token based via `MCP_PASSWORD`.
- Keep changes incremental and easy to review.
- Favor deterministic tests and failure-mode coverage over broad rewrites.
- If you change startup or dependency behavior, update `README.md`, `docs/`, and `CHANGELOG.md`.

## Architecture Pointers

- `app/main.py`: FastMCP server, tools, and health routes
- `app/runtime.py`: shared runtime, startup validation, readiness state
- `app/services/`: retrieval, reranking, answering, embeddings, GSRS API integration
- `app/db/`: vector backends
- `tests/`: smoke, unit, and failure-mode coverage
