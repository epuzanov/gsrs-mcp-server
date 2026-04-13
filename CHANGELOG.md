### Changed
- Standardized repository/package naming on `gsrs-mcp-server` and removed the legacy `gsrs-gateway` console-script alias.
- Renamed MCP auth and bind settings to `MCP_USERNAME`, `MCP_PASSWORD`, `mcp_api`, and `mcp_port` to match the server's MCP-first runtime contract.
- Rewrote the README, configuration docs, API docs, and examples to describe the MCP-first runtime and current transport/auth model.
- Separated liveness and readiness with `/livez`, `/readyz`, and a structured `/health` snapshot.
- Added startup/runtime component validation, structured logging, light in-memory metrics, and clearer degraded-mode behavior.
- Tightened capability-specific degraded behavior so ingest, similarity search, vector-admin tools, and GSRS upstream tools now fail against the correct dependency set with more specific messages.
- Preserved retrieval-only fallback when answer generation is unavailable and made identifier-first routing more deterministic.
- Added stage-level ask-path diagnostics, richer optional debug traces, and structured-log redaction for sensitive fields.
- Strengthened conservative retrieval with broader identifier-first routing, field-aware reranking boosts, tighter evidence selection, chunk-referenced citations, and clearer low-confidence abstention reasons.
- Added configurable shared identifier code-system support so ASK, ASKP, SMS_ID, SMSID, EVMPD, and xEVMPD behave consistently across rewrites, routing, reranking, abstention, and identifier aggregation.
- Rewrote `scripts/load_data.py` around a persistent MCP client session with MCP-native tool validation and optional `stdio` transport support.
- Switched GSRS upstream structure and sequence search to the current documentation example flow using `/ginas/app/api/v1/...Search?q=...` plus status/result polling.
- Expanded tests toward runtime startup/failure semantics, auth enforcement, degraded behavior, identifier routing, grounded golden-set coverage, and MCP transport smoke coverage.

### Added
- Initial release
