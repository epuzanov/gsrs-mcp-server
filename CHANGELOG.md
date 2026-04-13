### Changed
- Standardized repository/package naming on `gsrs-mcp-server` while keeping the legacy `gsrs-gateway` console script for compatibility.
- Rewrote the README, configuration docs, API docs, and examples to describe the MCP-first runtime and current transport/auth model.
- Separated liveness and readiness with `/livez`, `/readyz`, and a structured `/health` snapshot.
- Added startup/runtime component validation, structured logging, light in-memory metrics, and clearer degraded-mode behavior.
- Tightened capability-specific degraded behavior so ingest, similarity search, vector-admin tools, and GSRS upstream tools now fail against the correct dependency set with more specific messages.
- Preserved retrieval-only fallback when answer generation is unavailable and made identifier-first routing more deterministic.
- Expanded tests toward readiness, degraded behavior, identifier routing, and MCP transport smoke coverage.

### Added
- Initial release
