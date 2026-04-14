"""Tests for the GSRS upstream API client service."""

import unittest
from unittest.mock import patch

from app.services.gsrs_api import GsrsApiService


class TestGsrsApiService(unittest.TestCase):
    def test_structure_search_uses_documented_example_endpoint_and_polls_results(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        calls = []

        def fake_request_json(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/ginas/app/api/v1/substances/structureSearch"):
                return {
                    "status": "Running",
                    "finished": False,
                    "determined": False,
                    "url": "https://gsrs.ncats.nih.gov/api/v1/status(structure-key)",
                    "results": "https://gsrs.ncats.nih.gov/api/v1/status(structure-key)/results",
                }
            if url.endswith("/api/v1/status(structure-key)"):
                return {
                    "status": "Done",
                    "finished": True,
                    "determined": True,
                    "url": "https://gsrs.ncats.nih.gov/api/v1/status(structure-key)",
                    "results": "https://gsrs.ncats.nih.gov/api/v1/status(structure-key)/results",
                }
            if url.endswith("/api/v1/status(structure-key)/results"):
                return {
                    "total": 1,
                    "count": 1,
                    "content": [{"uuid": "sub-1", "_name": "Example structure"}],
                }
            raise AssertionError(f"Unexpected URL: {url}")

        with patch.object(service, "_request_json", side_effect=fake_request_json), \
             patch("app.services.gsrs_api.time.sleep", return_value=None):
            payload = service.structure_search(
                smiles="COCN",
                search_type="SUBSTRUCTURE",
                size=7,
            )

        self.assertEqual(payload["content"][0]["uuid"], "sub-1")
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(
            calls[0],
            (
                "GET",
                "https://gsrs.ncats.nih.gov/ginas/app/api/v1/substances/structureSearch",
                {"params": {"q": "COCN", "type": "Substructure"}},
            ),
        )
        self.assertEqual(
            calls[-1],
            (
                "GET",
                "https://gsrs.ncats.nih.gov/api/v1/status(structure-key)/results",
                {"params": {"top": 7, "skip": 0}},
            ),
        )

    def test_sequence_search_uses_documented_example_endpoint(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        calls = []

        def fake_request_json(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/ginas/app/api/v1/substances/sequenceSearch"):
                return {
                    "status": "Done",
                    "finished": True,
                    "determined": True,
                    "url": "https://gsrs.ncats.nih.gov/api/v1/status(sequence-key)",
                    "results": "https://gsrs.ncats.nih.gov/api/v1/status(sequence-key)/results",
                }
            if url.endswith("/api/v1/status(sequence-key)/results"):
                return {
                    "total": 1,
                    "count": 1,
                    "content": [{"uuid": "seq-1", "_name": "Example sequence"}],
                }
            raise AssertionError(f"Unexpected URL: {url}")

        with patch.object(service, "_request_json", side_effect=fake_request_json):
            payload = service.sequence_search(
                sequence="MVLSPADKTNVKAAWGKVGA",
                search_type="SIMILAR",
                sequence_type="PROTEIN",
                size=3,
            )

        self.assertEqual(payload["results"][0]["uuid"], "seq-1")
        self.assertEqual(
            calls[0],
            (
                "GET",
                "https://gsrs.ncats.nih.gov/ginas/app/api/v1/substances/sequenceSearch",
                {"params": {"q": "MVLSPADKTNVKAAWGKVGA"}},
            ),
        )
        self.assertEqual(
            calls[-1],
            (
                "GET",
                "https://gsrs.ncats.nih.gov/api/v1/status(sequence-key)/results",
                {"params": {"top": 3, "skip": 0}},
            ),
        )


if __name__ == "__main__":
    unittest.main()
