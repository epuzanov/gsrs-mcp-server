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
GSRS_SUBSTANCE_URL = f"{GSRS_BASE_URL}/substances"
GSRS_SEARCH_URL = f"{GSRS_BASE_URL}/substances/search"
GSRS_STRUCTURE_SEARCH_URL = f"{GSRS_BASE_URL}/substances/structure-search"
GSRS_SEQUENCE_SEARCH_URL = f"{GSRS_BASE_URL}/substances/sequence-search"


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
            GSRS API search response dict with "results" and "total" keys.
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

        payload: Dict[str, Any] = {
            "searchType": search_type,
            "size": size,
        }
        if smiles:
            payload["smiles"] = smiles
        if inchi:
            payload["inchi"] = inchi

        resp = self._request(
            "POST",
            f"{self.base_url}/substances/structure-search",
            json=payload,
        )
        return resp.json()

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
        payload: Dict[str, Any] = {
            "sequence": sequence,
            "searchType": search_type,
            "sequenceType": sequence_type,
            "size": size,
        }

        resp = self._request(
            "POST",
            f"{self.base_url}/substances/sequence-search",
            json=payload,
        )
        return resp.json()
