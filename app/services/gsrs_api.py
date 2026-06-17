"""
GSRS MCP Server - GSRS API Client Service

Provides access to the official GSRS REST API for fetching substance data,
searching by text, structure, and sequence.
"""
import time
from typing import Any, Dict, Optional

import httpx


# Official GSRS API endpoints
GSRS_BASE_URL = "https://gsrs.ncats.nih.gov/api/v1"
GSRS_SUBSTANCE_URL = f"{GSRS_BASE_URL}/substances"
GSRS_SEARCH_URL = f"{GSRS_BASE_URL}/substances/search"
GSRS_STRUCTURE_SEARCH_URL = f"{GSRS_BASE_URL}/substances/structureSearch"
GSRS_SEQUENCE_SEARCH_URL = f"{GSRS_BASE_URL}/substances/sequenceSearch"
GSRS_CV_URL = f"{GSRS_BASE_URL}/vocabularies"


class GsrsApiService:
    """HTTP client for the official GSRS REST API."""

    def __init__(
        self,
        base_url: str = GSRS_BASE_URL,
        timeout: int = 30,
        verify_ssl: bool = True,
        public_only: bool = False,
        max_retries: int = 1,
        retry_backoff_ms: int = 250,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.public_only = public_only
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms

    # ------------------------------------------------------------------
    # Public-only filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_public(data: Any) -> Any:
        """Recursively remove elements whose 'access' field is a non-empty list.

        GSRS marks restricted data with an ``access`` key containing a list of
        roles that are allowed to see it.  When ``access`` is non-empty the
        element is considered private and is removed.
        """
        if isinstance(data, dict):
            access = data.get("access")
            if isinstance(access, list) and len(access) > 0:
                return None
            return {
                k: v
                for k, v in (
                    (k, GsrsApiService._filter_public(v)) for k, v in data.items()
                )
                if v is not None
            }
        if isinstance(data, list):
            filtered = [GsrsApiService._filter_public(item) for item in data]
            return [item for item in filtered if item is not None]
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.timeout,
            verify=self.verify_ssl,
            headers={"Accept": "application/json"},
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._client() as client:
                    resp = client.request(method, url, **kwargs)
                    if resp.status_code == 404:
                        return resp
                    resp.raise_for_status()
                    return resp
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_ms / 1000)
        raise RuntimeError(
            f"GSRS upstream request failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        return self._request(method, url, **kwargs).json()

    def get_facets(
        self,
        query: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        size: int = 20,
    ) -> Dict[str, Any]:
        """Return available Lucene facet buckets for the current query context.

        GSRS returns facet groups in search responses under the "facets" key.
        This helper runs the same parametric query used by `parametric_search`
        but asks only for facet metadata, which is useful for discovering the
        facet names and values that can be passed to the `facets` argument of
        `gsrs_parametric_search`.

        Args:
            query: Free-text search context. Defaults to "*" so that all
                available system facets are returned when no context is given.
            filters: Fielded filters (same format as `parametric_search`).
            page: Page number (1-based).
            size: Results per page (kept small because only facets are needed).

        Returns:
            GSRS API search response dict; examine the "facets" list for
            available buckets. Each facet entry typically contains "name",
            "value", and "count" fields.
        """
        q = self._build_parametric_query(query=query, filters=filters or {})
        params: Dict[str, Any] = {
            "q": q,
            "page": page,
            "size": size,
        }
        envelope = self._request_json(
            "GET",
            f"{self.base_url}/substances/search",
            params=params,
        )
        return self._resolve_async_search(envelope, size)

    def _resolve_async_search(self, envelope: Dict[str, Any], size: int) -> Dict[str, Any]:
        status_payload = dict(envelope)
        if "content" in status_payload:
            return status_payload
        status_url = status_payload.get("url")
        results_url = status_payload.get("results")
        deadline = time.monotonic() + max(float(self.timeout), 1.0)

        while not (status_payload.get("finished") or status_payload.get("determined")):
            if not status_url or time.monotonic() >= deadline:
                break
            time.sleep(min(self.retry_backoff_ms / 1000, 1.0))
            status_payload = self._request_json("GET", status_url)
            results_url = status_payload.get("results") or results_url

        if not results_url:
            return {
                "results": [],
                "total": 0,
                "count": 0,
                "status": status_payload.get("status", "Unknown"),
                "finished": bool(status_payload.get("finished") or status_payload.get("determined")),
                "envelope": status_payload,
            }

        page_size = max(int(size or 0), 1)
        skip = 0
        content: list[Any] = []
        first_page: Dict[str, Any] | None = None
        total: int | None = None

        while total is None or skip < total:
            page = self._request_json(
                "GET",
                results_url,
                params={"top": page_size, "skip": skip},
            )
            if first_page is None:
                first_page = dict(page)

            page_content = page.get("content") or page.get("results") or []
            if not isinstance(page_content, list):
                page_content = []

            content.extend(page_content)

            raw_total = page.get("total")
            if isinstance(raw_total, int):
                total = raw_total
            else:
                try:
                    total = int(raw_total)
                except (TypeError, ValueError):
                    total = len(content)

            if not page_content:
                break
            skip += len(page_content)

        payload = first_page or {}
        payload["content"] = content
        if isinstance(payload.get("results"), list):
            payload["results"] = content
        payload["count"] = len(content)
        payload["total"] = total if total is not None else len(content)
        return payload

    def get_status(self) -> Dict[str, Any]:
        """Return non-sensitive configuration details."""
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "verify_ssl": self.verify_ssl,
            "public_only": self.public_only,
        }

    def ping(self) -> None:
        """Lightweight upstream probe used by readiness checks."""
        self._request("GET", f"{self.base_url}/substances/search", params={"query": "aspirin", "size": 1})

    # ------------------------------------------------------------------
    # Controlled vocabulary endpoints
    # ------------------------------------------------------------------

    def get_cv_domains(self, size: int = 200) -> Dict[str, Any]:
        """Return the list of available GSRS controlled vocabulary domains.

        GSRS exposes each controlled vocabulary at the ``/vocabularies``
        endpoint. Each entry contains a ``domain`` key identifying the
        vocabulary (e.g. ``NAME_TYPE``) and a ``terms`` list.

        Args:
            size: Max number of vocabularies to return (default 200 covers
                all current GSRS domains).

        Returns:
            A dict with ``content`` (list of vocabularies), ``total``, and
            ``count``.
        """
        envelope = self._request_json(
            "GET",
            f"{self.base_url}/vocabularies",
            params={"top": max(1, min(size, 500)), "skip": 0},
        )
        if isinstance(envelope, list):
            return {
                "content": envelope,
                "total": len(envelope),
                "count": len(envelope),
            }

        content = envelope.get("content") or envelope.get("results") or []
        total = envelope.get("total", len(content))
        return {
            "content": content,
            "total": total,
            "count": len(content),
        }

    def get_cv_terms(self, domain: str) -> Dict[str, Any]:
        """Return the terms for a single GSRS controlled vocabulary domain.

        Each term contains at least ``value`` (the stored code) and
        ``display`` (the human-readable label). This is useful for
        resolving short codes such as ``of`` -> ``Official Name`` in
        ``names.type`` or ``codes.type``.

        Args:
            domain: CV domain name, e.g. ``NAME_TYPE``, ``CODE_TYPE``,
                ``SUBSTANCE_CLASS``.

        Returns:
            Vocabulary dict with ``domain`` and ``terms`` list.
        """
        if not domain or not str(domain).strip():
            raise ValueError("domain is required")
        return self._request_json(
            "GET",
            f"{self.base_url}/vocabularies/{domain.strip()}",
        )

    # ------------------------------------------------------------------
    # Core substance endpoints
    # ------------------------------------------------------------------

    def get_substance_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Fetch a complete substance document by UUID."""
        url = f"{self.base_url}/substances({uuid})"
        resp = self._request("GET", url, params={"view": "full"})
        if resp.status_code == 404:
            return None
        data = resp.json()
        if self.public_only:
            data = self._filter_public(data)
        return data

    def get_substance(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Fetch a complete substance document by UUID or approval identifier."""
        return self.get_substance_by_uuid(identifier.strip())

    def text_search(
        self,
        query: str,
        page: int = 1,
        size: int = 20,
        fields: Optional[str] = None,
        facets: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search substances by free-text query.

        Args:
            query: Search text (names, codes, etc.)
            page: Page number (1-based)
            size: Results per page
            fields: Comma-separated field list to return (e.g. "uuid,name,code")

        Returns:
            GSRS API search response dict with "content" and "total" keys.
        """
        params: Dict[str, Any] = {
            "q": query,
            "page": page,
            "size": size,
        }
        if fields:
            params["fields"] = fields
        if facets:
            params["facet"] = facets

        envelope = self._request_json(
            "GET",
            f"{self.base_url}/substances/search",
            params=params,
        )
        return self._resolve_async_search(envelope, size)

    def parametric_search(
        self,
        query: str = "",
        filters: Optional[Dict[str, Any]] = None,
        facets: Optional[list[str]] = None,
        page: int = 1,
        size: int = 20,
        fields: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search substances with GSRS fielded query terms and optional facets."""
        q = self._build_parametric_query(query=query, filters=filters or {})
        return self.text_search(
            q,
            page=page,
            size=size,
            fields=fields,
            facets=facets,
        )

    def _build_parametric_query(self, query: str = "", filters: Optional[Dict[str, Any]] = None) -> str:
        """Build the GSRS search q value from free text plus field:value terms."""
        terms: list[str] = []
        if query and query.strip():
            terms.append(query.strip())

        for field, value in (filters or {}).items():
            if value is None or value == "":
                continue
            if isinstance(value, list):
                field_terms = [self._field_query_term(field, item) for item in value if item not in (None, "")]
                if field_terms:
                    terms.append("(" + " OR ".join(field_terms) + ")")
            else:
                terms.append(self._field_query_term(field, value))

        return " AND ".join(terms) if terms else "*"

    @staticmethod
    def _field_query_term(field: str, value: Any) -> str:
        """Return one GSRS fielded query term."""
        raw = str(value).strip()
        if raw.startswith('"') and raw.endswith('"'):
            rendered = raw
        elif any(char.isspace() for char in raw) or any(char in raw for char in ":/"):
            rendered = f'"{raw}"'
        else:
            rendered = raw
        return f"{field}:{rendered}"

    def structure_search(
        self,
        structure: Optional[str] = None,
        search_type: str = "exact",
        cutoff: float = 0.8,
        size: int = 20,
    ) -> Dict[str, Any]:
        """
        Search substances by chemical structure.

        Args:
            structure: Chemical structure string (SMILES or InChI)
            search_type: exact | exactplus | sim | substructure | flex | flexplus
            cutoff: Similarity cutoff for sim searches (0.0 - 1.0)
            size: Max results

        Returns:
            GSRS API search response dict.
        """
        if not structure:
            raise ValueError("A structure must be provided.")
        query = structure.strip()
        params: Dict[str, Any] = {
            "q": query,
            "size": size,
        }
        if search_type:
            params["type"] = search_type
        if search_type == "sim":
            params["cutoff"] = cutoff

        envelope = self._request_json(
            "GET",
            f"{self.base_url}/substances/structureSearch",
            params=params,
        )
        return self._resolve_async_search(envelope, size)

    def sequence_search(
        self,
        sequence: str,
        search_type: str = "exact",
        sequence_type: str = "nucleicAcid",
        cutoff: float = 0.95,
        size: int = 20,
    ) -> Dict[str, Any]:
        """
        Search substances by biological sequence.

        Args:
            sequence: Amino acid or nucleotide sequence string
            search_type: GLOBAL | SUB
            sequence_type: protein | nucleicAcid
            cutoff: Similarity cutoff for SUB searches (0.0 - 1.0)
            size: Max results

        Returns:
            GSRS API search response dict.
        """
        params: Dict[str, Any] = {
            "q": sequence,
            "type": search_type,
            "seqType": sequence_type,
            "cutoff": cutoff,
        }

        envelope = self._request_json(
            "POST",
            f"{self.base_url}/substances/sequenceSearch",
            data=params,
        )
        return self._resolve_async_search(envelope, size)
