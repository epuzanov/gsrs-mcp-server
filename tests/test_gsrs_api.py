"""Tests for the GSRS upstream API client service."""

import unittest
from unittest.mock import patch

from app.services.gsrs_api import GsrsApiService


class TestGsrsApiService(unittest.TestCase):
    def test_structure_search_uses_documented_endpoint_and_polls_results(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        calls = []

        def fake_request_json(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/api/v1/substances/structureSearch"):
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
                structure="COCN",
                search_type="substructure",
                size=7,
            )

        self.assertEqual(payload["content"][0]["uuid"], "sub-1")
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(
            calls[0],
            (
                "GET",
                "https://gsrs.ncats.nih.gov/api/v1/substances/structureSearch",
                {"params": {"q": "COCN", "size": 7, "type": "substructure"}},
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

    def test_sequence_search_uses_documented_endpoint(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        calls = []

        def fake_request_json(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/api/v1/substances/sequenceSearch"):
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
                search_type="GLOBAL",
                sequence_type="protein",
                size=3,
            )

        self.assertEqual(payload["content"][0]["uuid"], "seq-1")
        self.assertEqual(
            calls[0],
            (
                "POST",
                "https://gsrs.ncats.nih.gov/api/v1/substances/sequenceSearch",
                {
                    "data": {
                        "q": "MVLSPADKTNVKAAWGKVGA",
                        "type": "GLOBAL",
                        "seqType": "protein",
                        "cutoff": 0.95,
                    }
                },
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

    def test_async_search_fetches_pages_until_total_is_reached(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        calls = []

        def fake_request_json(method, url, **kwargs):
            calls.append((method, url, kwargs))
            skip = kwargs["params"]["skip"]
            if skip == 0:
                return {
                    "total": 5,
                    "count": 2,
                    "content": [{"uuid": "sub-1"}, {"uuid": "sub-2"}],
                }
            if skip == 2:
                return {
                    "total": 5,
                    "count": 2,
                    "content": [{"uuid": "sub-3"}, {"uuid": "sub-4"}],
                }
            if skip == 4:
                return {
                    "total": 5,
                    "count": 1,
                    "content": [{"uuid": "sub-5"}],
                }
            raise AssertionError(f"Unexpected skip: {skip}")

        with patch.object(service, "_request_json", side_effect=fake_request_json):
            payload = service._resolve_async_search(
                {
                    "status": "Done",
                    "finished": True,
                    "determined": True,
                    "results": "https://gsrs.ncats.nih.gov/api/v1/status(search-key)/results",
                },
                size=2,
            )

        self.assertEqual(
            [substance["uuid"] for substance in payload["content"]],
            ["sub-1", "sub-2", "sub-3", "sub-4", "sub-5"],
        )
        self.assertEqual(payload["total"], 5)
        self.assertEqual(payload["count"], 5)
        self.assertEqual(
            [call[2]["params"] for call in calls],
            [
                {"top": 2, "skip": 0},
                {"top": 2, "skip": 2},
                {"top": 2, "skip": 4},
            ],
        )

    def test_async_search_preserves_results_list_shape_when_pages_use_results(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)

        def fake_request_json(method, url, **kwargs):
            skip = kwargs["params"]["skip"]
            if skip == 0:
                return {
                    "total": 2,
                    "count": 1,
                    "results": [{"uuid": "sub-1"}],
                }
            if skip == 1:
                return {
                    "total": 2,
                    "count": 1,
                    "results": [{"uuid": "sub-2"}],
                }
            raise AssertionError(f"Unexpected skip: {skip}")

        with patch.object(service, "_request_json", side_effect=fake_request_json):
            payload = service._resolve_async_search(
                {
                    "status": "Done",
                    "finished": True,
                    "determined": True,
                    "results": "https://gsrs.ncats.nih.gov/api/v1/status(search-key)/results",
                },
                size=1,
            )

        expected = [{"uuid": "sub-1"}, {"uuid": "sub-2"}]
        self.assertEqual(payload["content"], expected)
        self.assertEqual(payload["results"], expected)
        self.assertEqual(payload["count"], 2)

    def test_get_cv_domains_lists_domains(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)

        def fake_request_json(method, url, **kwargs):
            self.assertEqual(method, "GET")
            self.assertTrue(url.endswith("/vocabularies"))
            self.assertEqual(kwargs.get("params"), {"top": 200, "skip": 0})
            return {
                "total": 3,
                "count": 3,
                "content": [
                    {"domain": "NAME_TYPE"},
                    {"domain": "CODE_TYPE"},
                    {"domain": "SUBSTANCE_CLASS"},
                ],
            }

        with patch.object(service, "_request_json", side_effect=fake_request_json):
            payload = service.get_cv_domains()

        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["count"], 3)
        self.assertEqual([d["domain"] for d in payload["content"]], ["NAME_TYPE", "CODE_TYPE", "SUBSTANCE_CLASS"])

    def test_get_cv_terms_returns_terms_for_domain(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)

        def fake_request_json(method, url, **kwargs):
            self.assertEqual(method, "GET")
            self.assertTrue(url.endswith("/vocabularies/NAME_TYPE"))
            return {
                "domain": "NAME_TYPE",
                "terms": [
                    {"value": "of", "display": "Official Name"},
                    {"value": "sys", "display": "Systematic Name"},
                    {"value": "cn", "display": "Common Name"},
                ],
            }

        with patch.object(service, "_request_json", side_effect=fake_request_json):
            payload = service.get_cv_terms("NAME_TYPE")

        self.assertEqual(payload["domain"], "NAME_TYPE")
        self.assertEqual(len(payload["terms"]), 3)
        self.assertEqual(payload["terms"][0]["display"], "Official Name")

    def test_get_cv_terms_rejects_empty_domain(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with self.assertRaises(ValueError):
            service.get_cv_terms("")


if __name__ == "__main__":
    unittest.main()
