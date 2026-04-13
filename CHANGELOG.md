# Changelog

## 0.1.0 - 2026-04-13

### Docs / Runtime Contract

- Standardized repository and package naming on `gsrs-mcp-server`.
- Rewrote the root README and supporting docs to describe the MCP-first runtime instead of a legacy REST-style surface.
- Clarified the current auth contract around `MCP_PASSWORD`, the `gsrs-mcp-server` CLI, `/mcp` transport options, and the health endpoints `/livez`, `/readyz`, and `/health`.
- Removed the legacy `gsrs-gateway` console-script alias and aligned examples, guides, and loader docs with the current package identity.

### Reliability / Health

- Separated liveness and readiness semantics with deterministic `/livez`, `/readyz`, and `/health` behavior.
- Added startup and dependency validation for the vector backend, embeddings, optional answer generation, optional GSRS upstream access, and chunker startup.
- Made empty-but-connected vector stores report healthy and ready instead of disconnected.
- Tightened degraded-mode handling so answer generation, GSRS upstream access, and ingest-only chunker failures degrade explicitly instead of failing the full retrieval path.
- Added bounded retries and timeouts for outbound dependency calls and made startup/runtime errors more specific and actionable.

### Observability

- Added structured JSON logging for MCP tool calls with request-scoped fields such as `request_id`, `tool_name`, `backend`, `query_type`, `latency_ms`, `outcome`, `error_type`, `result_count`, and `citation_count`.
- Added stage-level diagnostics for `gsrs_ask`, including rewrite, retrieval, reranking, evidence extraction, abstention, and answer-generation stages.
- Added optional debug output that surfaces routing decisions, normalized queries, retrieved chunk IDs, reranked scores, applied filters, and degraded runtime state.
- Added light in-memory metrics and redaction for sensitive values such as API keys, passwords, bearer tokens, and authorization headers.

### Retrieval / Grounded Answers

- Strengthened identifier-first routing for UUIDs, approval IDs, exact names, InChIKeys, and configured code systems.
- Added configurable shared identifier code-system support for `ASK`, `ASKP`, `SMS_ID`, `SMSID`, `EVMPD`, and `xEVMPD` across rewrites, routing, reranking, abstention, and identifier aggregation.
- Improved field-aware reranking and evidence selection so fewer, higher-confidence chunks are passed into answer generation.
- Added clearer abstention behavior for low-confidence or identifier-miss cases.
- Normalized grounded answer output around direct answers, supporting evidence, citations, and uncertainty/degraded notes.

### GSRS Upstream / Tooling

- Reworked GSRS upstream structure and sequence search to match the current documented `/ginas/app/api/v1/...Search?q=...` flow, including status polling and results retrieval.
- Rewrote `scripts/load_data.py` to use the MCP client library directly with a persistent session and optional `stdio` transport.
- Added a Podman `kube play` manifest and aligned container settings with the current MCP config names and readiness semantics.

### Tests / Examples

- Expanded regression coverage for MCP smoke paths, auth enforcement, health/readiness semantics, degraded dependency behavior, identifier-first retrieval, grounded citations, and golden-set answer behavior.
- Added transport-level `gsrs_ask` coverage so the public MCP path is exercised end to end in tests.
- Updated examples and integration guides for ChatGPT, Open WebUI, Ollama, and OpenAI API remote-MCP usage so they match the current runtime contract.
