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

    def test_looks_like_uuid(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        self.assertTrue(
            service.looks_like_uuid("0103a288-6eb6-4ced-b13a-849cd7edf028")
        )
        self.assertTrue(
            service.looks_like_uuid("0103A288-6EB6-4CED-B13A-849CD7EDF028")
        )
        self.assertFalse(service.looks_like_uuid("not-a-uuid"))
        self.assertFalse(service.looks_like_uuid("ASPIRIN"))
        self.assertFalse(service.looks_like_uuid(""))

    def test_resolve_substance_uuid_with_uuid_input(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with patch.object(service, "get_substance_by_uuid") as fake:
            fake.return_value = None
            resolved = service.resolve_substance_uuid(
                "0103a288-6eb6-4ced-b13a-849cd7edf028"
            )
        self.assertEqual(resolved["uuid"], "0103a288-6eb6-4ced-b13a-849cd7edf028")
        self.assertEqual(resolved["match_type"], "uuid")
        # No fallback to search should have happened.
        fake.assert_not_called()

    def test_resolve_substance_uuid_via_substance_endpoint(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with patch.object(service, "get_substance_by_uuid", return_value={
            "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
            "_name": "IBUPROFEN",
            "approvalID": "WK2XYI10QM",
        }) as fake_substance, patch.object(service, "parametric_search") as fake_search:
            resolved = service.resolve_substance_uuid("WK2XYI10QM")
        self.assertEqual(resolved["uuid"], "0103a288-6eb6-4ced-b13a-849cd7edf028")
        self.assertEqual(resolved["name"], "IBUPROFEN")
        self.assertEqual(resolved["approvalID"], "WK2XYI10QM")
        self.assertEqual(resolved["match_type"], "approval_id_or_uuid")
        fake_substance.assert_called_once()
        # Should not have fallen through to search.
        fake_search.assert_not_called()

    def test_resolve_substance_uuid_via_name_search(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with patch.object(service, "get_substance_by_uuid", return_value=None), \
             patch.object(service, "parametric_search", return_value={
                 "content": [
                     {
                         "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
                         "_name": "ASPIRIN",
                         "approvalID": "R16CO5Y76E",
                     }
                 ]
             }) as fake_search:
            resolved = service.resolve_substance_uuid("ASPIRIN")
        self.assertEqual(resolved["uuid"], "0103a288-6eb6-4ced-b13a-849cd7edf028")
        self.assertEqual(resolved["name"], "ASPIRIN")
        self.assertEqual(resolved["match_type"], "name")
        fake_search.assert_called_once()
        # Search should have been called with the name as the query.
        self.assertEqual(fake_search.call_args.kwargs.get("query"), "ASPIRIN")

    def test_resolve_substance_uuid_returns_none_when_nothing_found(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with patch.object(service, "get_substance_by_uuid", return_value=None), \
             patch.object(service, "parametric_search", return_value={"content": []}):
            resolved = service.resolve_substance_uuid("NOPE")
        self.assertIsNone(resolved)

    def test_resolve_substance_uuid_prefers_exact_name_match(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with patch.object(service, "get_substance_by_uuid", return_value=None), \
             patch.object(service, "parametric_search", return_value={
                 "content": [
                     {
                         "uuid": "first-uuid",
                         "_name": "Something else",
                     },
                     {
                         "uuid": "exact-uuid",
                         "_name": "ASPIRIN",
                     },
                 ]
             }):
            resolved = service.resolve_substance_uuid("aspirin")
        self.assertEqual(resolved["uuid"], "exact-uuid")
        self.assertEqual(resolved["match_type"], "name")

    def test_get_substance_details_with_filter(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)

        captured = {}

        class FakeResponse:
            status_code = 200
            def json(self_inner):
                return ["ASPIRIN", "ACETYLSALICYLIC ACID"]

        def fake_request(method, url, **kwargs):
            captured["method"] = method
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeResponse()

        with patch.object(service, "_request", side_effect=fake_request):
            result = service.get_substance_details(
                "0103a288-6eb6-4ced-b13a-849cd7edf028",
                "names(type:cn)!(name)",
            )

        self.assertEqual(captured["method"], "GET")
        self.assertEqual(
            captured["url"],
            "https://gsrs.ncats.nih.gov/api/v1/substances(0103a288-6eb6-4ced-b13a-849cd7edf028)/names(type:cn)!(name)",
        )
        self.assertEqual(result, ["ASPIRIN", "ACETYLSALICYLIC ACID"])

    def test_get_substance_details_with_filter_steps(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)

        captured = {}

        class FakeResponse:
            status_code = 200
            def json(self_inner):
                return ["ASPIRIN"]

        def fake_request(method, url, **kwargs):
            captured["url"] = url
            return FakeResponse()

        with patch.object(service, "_request", side_effect=fake_request):
            result = service.get_substance_details(
                "0103a288-6eb6-4ced-b13a-849cd7edf028",
                filter_steps=["names(type:cn)", "(name)", "limit(1)"],
            )

        self.assertEqual(
            captured["url"],
            "https://gsrs.ncats.nih.gov/api/v1/substances(0103a288-6eb6-4ced-b13a-849cd7edf028)/names(type:cn)!(name)!limit(1)",
        )
        self.assertEqual(result, ["ASPIRIN"])

    def test_get_substance_details_rejects_both_filter_forms(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with self.assertRaises(ValueError):
            service.get_substance_details(
                "0103a288-6eb6-4ced-b13a-849cd7edf028",
                filter_expression="names",
                filter_steps=["names"],
            )

    def test_join_filter_steps_joins_with_bang(self):
        self.assertEqual(
            GsrsApiService.join_filter_steps(
                ["names(type:of)", "(name)", "limit(1)"]
            ),
            "names(type:of)!(name)!limit(1)",
        )

    def test_join_filter_steps_handles_empty_and_none(self):
        for value in (None, [], ["", "  "]):
            with self.subTest(value=value):
                self.assertEqual(GsrsApiService.join_filter_steps(value), "")

    def test_join_filter_steps_strips_whitespace(self):
        self.assertEqual(
            GsrsApiService.join_filter_steps(["  names  ", "(name)"]),
            "names!(name)",
        )

    def test_get_substance_details_without_filter(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)

        captured = {}

        class FakeResponse:
            status_code = 200
            def json(self_inner):
                return {"uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028"}

        def fake_request(method, url, **kwargs):
            captured["url"] = url
            return FakeResponse()

        with patch.object(service, "_request", side_effect=fake_request):
            result = service.get_substance_details(
                "0103a288-6eb6-4ced-b13a-849cd7edf028"
            )
        self.assertEqual(
            captured["url"],
            "https://gsrs.ncats.nih.gov/api/v1/substances(0103a288-6eb6-4ced-b13a-849cd7edf028)",
        )
        self.assertEqual(result, {"uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028"})

    def test_get_substance_details_returns_none_on_404(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)

        class FakeResponse:
            status_code = 404

        with patch.object(service, "_request", return_value=FakeResponse()):
            result = service.get_substance_details(
                "0103a288-6eb6-4ced-b13a-849cd7edf028",
                "names",
            )
        self.assertIsNone(result)

    def test_get_substance_details_rejects_invalid_filter(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with self.assertRaises(ValueError):
            service.get_substance_details("uuid", "names(type:of")

    def test_get_substance_details_rejects_empty_uuid(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with self.assertRaises(ValueError):
            service.get_substance_details("", "names")

    def test_get_substance_details_by_identifier_resolves_then_calls(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with patch.object(service, "resolve_substance_uuid", return_value={
            "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
            "name": "ASPIRIN",
            "approvalID": "R16CO5Y76E",
            "match_type": "name",
        }) as fake_resolve, patch.object(service, "get_substance_details", return_value=[
            "ASPIRIN"
        ]) as fake_details:
            payload = service.get_substance_details_by_identifier(
                "ASPIRIN", "names(type:of)!(name)"
            )
        self.assertEqual(payload["resolved"]["uuid"], "0103a288-6eb6-4ced-b13a-849cd7edf028")
        self.assertEqual(payload["result"], ["ASPIRIN"])
        fake_resolve.assert_called_once_with("ASPIRIN")
        fake_details.assert_called_once_with(
            "0103a288-6eb6-4ced-b13a-849cd7edf028",
            filter_expression="names(type:of)!(name)",
            filter_steps=None,
        )

    def test_get_substance_details_by_identifier_with_filter_steps(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with patch.object(service, "resolve_substance_uuid", return_value={
            "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
            "name": "ASPIRIN",
            "approvalID": "R16CO5Y76E",
            "match_type": "name",
        }), patch.object(service, "get_substance_details", return_value=[
            "ASPIRIN"
        ]) as fake_details:
            payload = service.get_substance_details_by_identifier(
                "ASPIRIN", filter_steps=["names(type:of)", "(name)"]
            )
        self.assertEqual(payload["result"], ["ASPIRIN"])
        fake_details.assert_called_once_with(
            "0103a288-6eb6-4ced-b13a-849cd7edf028",
            filter_expression=None,
            filter_steps=["names(type:of)", "(name)"],
        )

    def test_get_substance_details_by_identifier_returns_none_when_unresolved(self):
        service = GsrsApiService(timeout=1, retry_backoff_ms=0)
        with patch.object(service, "resolve_substance_uuid", return_value=None):
            payload = service.get_substance_details_by_identifier("NOPE", "names")
        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
