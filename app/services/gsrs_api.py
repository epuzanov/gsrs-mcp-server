"""
GSRS MCP Server - GSRS API Client Service

Provides access to the official GSRS REST API for fetching substance data,
searching by text, structure, and sequence.
"""
import time
from typing import Any, Dict, List, Optional

import httpx


# Official GSRS API endpoints
GSRS_BASE_URL = "https://gsrs.ncats.nih.gov/api/v1"
GSRS_EXAMPLE_BASE_URL = "https://gsrs.ncats.nih.gov/ginas/app/api/v1"
GSRS_SUBSTANCE_URL = f"{GSRS_BASE_URL}/substances"
GSRS_SEARCH_URL = f"{GSRS_BASE_URL}/substances/search"
GSRS_STRUCTURE_SEARCH_URL = f"{GSRS_EXAMPLE_BASE_URL}/substances/structureSearch"
GSRS_SEQUENCE_SEARCH_URL = f"{GSRS_EXAMPLE_BASE_URL}/substances/sequenceSearch"


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
        self.example_base_url = self._derive_example_base_url(self.base_url)
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

    @staticmethod
    def _derive_example_base_url(base_url: str) -> str:
        if "/ginas/app/api/v1" in base_url:
            return base_url
        if base_url.endswith("/api/v1"):
            return f"{base_url[:-len('/api/v1')]}/ginas/app/api/v1"
        return f"{base_url}/ginas/app/api/v1"

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

    def _resolve_async_search(self, envelope: Dict[str, Any], size: int) -> Dict[str, Any]:
        status_payload = dict(envelope)
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

        return self._request_json(
            "GET",
            results_url,
            params={"top": size, "skip": 0},
        )

    @staticmethod
    def _map_structure_search_type(search_type: str) -> str:
        mapping = {
            "EXACT": "Exact",
            "SIMILAR": "Similarity",
            "SUBSTRUCTURE": "Substructure",
            "SUPERSTRUCTURE": "Superstructure",
        }
        return mapping.get(search_type.upper(), search_type.title())

    def get_status(self) -> Dict[str, Any]:
        """Return non-sensitive configuration details."""
        return {
            "base_url": self.base_url,
            "example_base_url": self.example_base_url,
            "timeout": self.timeout,
            "verify_ssl": self.verify_ssl,
            "public_only": self.public_only,
        }

    def ping(self) -> None:
        """Lightweight upstream probe used by readiness checks."""
        self._request("GET", f"{self.base_url}/substances/search", params={"query": "aspirin", "size": 1})

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

    def text_search(
        self,
        query: str,
        page: int = 1,
        size: int = 20,
        fields: Optional[str] = None,
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
            "query": query,
            "page": page,
            "size": size,
        }
        if fields:
            params["fields"] = fields

        resp = self._request("GET", f"{self.base_url}/substances/search", params=params)
        return resp.json()

    def structure_search(
        self,
        smiles: Optional[str] = None,
        inchi: Optional[str] = None,
        search_type: str = "EXACT",
        size: int = 20,
    ) -> Dict[str, Any]:
        """
        Search substances by chemical structure.

        Args:
            smiles: SMILES string
            inchi: InChI string
            search_type: EXACT | SIMILAR | SUBSTRUCTURE | SUPERSTRUCTURE
            size: Max results

        Returns:
            GSRS API search response dict.
        """
        if not smiles and not inchi:
            raise ValueError("Either smiles or inchi must be provided.")
        query = inchi or smiles or ""
        params: Dict[str, Any] = {
            "q": query,
        }
        if search_type:
            params["type"] = self._map_structure_search_type(search_type)

        envelope = self._request_json(
            "GET",
            f"{self.example_base_url}/substances/structureSearch",
            params=params,
        )
        return self._resolve_async_search(envelope, size)

    def sequence_search(
        self,
        sequence: str,
        search_type: str = "EXACT",
        sequence_type: str = "NUCLEIC_ACID",
        size: int = 20,
    ) -> Dict[str, Any]:
        """
        Search substances by biological sequence.

        Args:
            sequence: Amino acid or nucleotide sequence string
            search_type: EXACT | CONTAINS | SIMILAR
            sequence_type: PROTEIN | NUCLEIC_ACID
            size: Max results

        Returns:
            GSRS API search response dict.
        """
        envelope = self._request_json(
            "GET",
            f"{self.example_base_url}/substances/sequenceSearch",
            params={"q": sequence},
        )
        return self._resolve_async_search(envelope, size)
