"""Tests for the gsrs_get_substance_details MCP tool and resource."""

import json
import unittest
from unittest.mock import patch

import app.main
from app.main import gsrs_substance_details_resource, gsrs_get_substance_details


def _make_fake_api(get_substance_details_by_identifier=None):
    """Build a fake GsrsApiService with the methods the resource/tool use."""

    class FakeApi:
        def __init__(self):
            self.calls = []

        def gsrs_api_available(self):
            return True

        def gsrs_api_unavailable_reason(self):
            return "API down for testing"

        def join_filter_steps(self, steps):
            if not steps:
                return ""
            return "!".join(str(step).strip() for step in steps if str(step).strip())

        def get_substance_details_by_identifier(self, identifier, filter_expression):
            self.calls.append((identifier, filter_expression))
            if get_substance_details_by_identifier is not None:
                return get_substance_details_by_identifier(identifier, filter_expression)
            return {
                "resolved": {
                    "uuid": "0103a288-6eb6-4ced-b13a-849cd7edf028",
                    "name": "ASPIRIN",
                    "approvalID": "R16CO5Y76E",
                    "match_type": "name",
                },
                "result": ["ASPIRIN", "ACETYLSALICYLIC ACID"],
            }

    return FakeApi()


class _RuntimePatcher:
    """Context manager to swap runtime.gsrs_api and bypass initialization."""

    def __init__(self, api, *, api_available=True):
        self.api = api
        self._runtime = None
        self._original_api = None
        self._original_initialized = None
        self._original_components = None
        self.api_available = api_available

    def __enter__(self):
        from app.main import runtime as runtime_module

        self._runtime = runtime_module
        self._original_api = self._runtime.gsrs_api
        self._original_initialized = self._runtime.initialized
        self._original_components = self._runtime.components
        self._runtime.gsrs_api = self.api
        self._runtime.initialized = True
        # Build a minimal component table so runtime.gsrs_api_available()
        # returns the value we want.
        from app.runtime import ComponentStatus

        if self.api_available:
            self._runtime.components = {
                "gsrs_api": ComponentStatus(
                    name="gsrs_api", required=True, ready=True
                )
            }
        else:
            self._runtime.components = {
                "gsrs_api": ComponentStatus(
                    name="gsrs_api",
                    required=True,
                    ready=False,
                    error="API down for testing",
                )
            }
        return self

    def __exit__(self, *exc):
        self._runtime.gsrs_api = self._original_api
        self._runtime.initialized = self._original_initialized
        self._runtime.components = self._original_components


class TestSubstanceDetailsResource(unittest.TestCase):
    def test_resource_resolves_and_returns_json(self):
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(
                gsrs_substance_details_resource("ASPIRIN", "names(type:cn)!(name)")
            )
        payload = json.loads(result)
        self.assertEqual(
            payload["resolved"]["uuid"], "0103a288-6eb6-4ced-b13a-849cd7edf028"
        )
        self.assertEqual(payload["filter"], "names(type:cn)!(name)")
        self.assertEqual(payload["result"], ["ASPIRIN", "ACETYLSALICYLIC ACID"])
        self.assertEqual(fake.calls, [("ASPIRIN", "names(type:cn)!(name)")])

    def test_resource_rejects_invalid_filter(self):
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(gsrs_substance_details_resource("ASPIRIN", "names(type:of"))
        payload = json.loads(result)
        self.assertIn("error", payload)
        self.assertIn("Invalid filter", payload["error"])
        # The fake API should not have been called for an invalid filter.
        self.assertEqual(fake.calls, [])

    def test_resource_handles_unresolved_identifier(self):
        def fake_details(identifier, filter_expression):
            return None

        fake = _make_fake_api(get_substance_details_by_identifier=fake_details)
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(gsrs_substance_details_resource("NOPE", "names"))
        payload = json.loads(result)
        self.assertIn("error", payload)
        self.assertIn("Could not resolve", payload["error"])

    def test_resource_normalizes_prejoined_string_to_filter_steps(self):
        """A pre-joined URI filter is split back into steps before validation.

        The resource is documented to take a list of steps. The URI
        template's ``{filter}`` is a single pre-joined string, so
        the resource must split it back into steps (and rejoin +
        validate) so URI callers and in-process callers share one
        code path. The JSON response should also expose the
        ``filter_steps`` array for introspection.
        """
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(
                gsrs_substance_details_resource(
                    "ASPIRIN", "names(type:cn)!(name)!limit(1)"
                )
            )
        payload = json.loads(result)
        self.assertEqual(payload["filter"], "names(type:cn)!(name)!limit(1)")
        self.assertEqual(
            payload["filter_steps"],
            ["names(type:cn)", "(name)", "limit(1)"],
        )
        # The fake's join_filter_steps is the same as the real one,
        # so the call is the round-tripped joined form.
        self.assertEqual(
            fake.calls, [("ASPIRIN", "names(type:cn)!(name)!limit(1)")]
        )

    def test_resource_rejects_malformed_prejoined_string(self):
        """An invalid pre-joined filter still raises a clear error."""
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            # Unbalanced parens — the parser should reject this.
            result = asyncio.run(
                gsrs_substance_details_resource("ASPIRIN", "names(type:of")
            )
        payload = json.loads(result)
        self.assertIn("error", payload)
        self.assertIn("Invalid filter", payload["error"])
        self.assertEqual(fake.calls, [])

    def test_resource_with_empty_filter_returns_no_filter_steps(self):
        """An empty filter yields ``None`` for both fields."""
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(gsrs_substance_details_resource("ASPIRIN", ""))
        payload = json.loads(result)
        self.assertIsNone(payload["filter"])
        self.assertIsNone(payload["filter_steps"])
        # Empty filter is forwarded as ``None`` (no filter path).
        self.assertEqual(fake.calls, [("ASPIRIN", None)])

    def test_resource_strips_whitespace_in_filter_steps(self):
        """Leading/trailing whitespace around each step is normalized away."""
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(
                gsrs_substance_details_resource(
                    "ASPIRIN", "  names(type:of)  !  (name)  "
                )
            )
        payload = json.loads(result)
        self.assertEqual(
            payload["filter_steps"],
            ["names(type:of)", "(name)"],
        )
        self.assertEqual(fake.calls, [("ASPIRIN", "names(type:of)!(name)")])


class TestSubstanceDetailsTool(unittest.TestCase):
    def test_tool_returns_payload_json(self):
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(
                gsrs_get_substance_details("ASPIRIN", "names(type:cn)!(name)")
            )
        payload = json.loads(result)
        self.assertEqual(
            payload["resolved"]["uuid"], "0103a288-6eb6-4ced-b13a-849cd7edf028"
        )
        self.assertEqual(payload["result"], ["ASPIRIN", "ACETYLSALICYLIC ACID"])

    def test_tool_accepts_filter_steps_as_json_array(self):
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(
                gsrs_get_substance_details(
                    "ASPIRIN",
                    '["names(type:cn)", "(name)", "limit(1)"]',
                )
            )
        payload = json.loads(result)
        self.assertEqual(payload["result"], ["ASPIRIN", "ACETYLSALICYLIC ACID"])
        # The resource should have been called with the joined string.
        self.assertEqual(
            fake.calls,
            [("ASPIRIN", "names(type:cn)!(name)!limit(1)")],
        )

    def test_tool_accepts_filter_steps_as_comma_separated(self):
        fake = _make_fake_api()
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(
                gsrs_get_substance_details(
                    "ASPIRIN",
                    "names(type:of), (name)",
                )
            )
        payload = json.loads(result)
        self.assertEqual(payload["result"], ["ASPIRIN", "ACETYLSALICYLIC ACID"])
        self.assertEqual(fake.calls, [("ASPIRIN", "names(type:of)!(name)")])

    def test_tool_handles_unresolved_identifier(self):
        def fake_details(identifier, filter_expression):
            return None

        fake = _make_fake_api(get_substance_details_by_identifier=fake_details)
        with _RuntimePatcher(fake):
            import asyncio

            result = asyncio.run(gsrs_get_substance_details("NOPE", "names"))
        payload = json.loads(result)
        self.assertIn("error", payload)
        self.assertIn("Could not resolve", payload["error"])

    def test_tool_handles_api_unavailable(self):
        class UnavailableApi(_make_fake_api().__class__):
            def gsrs_api_available(self):
                return False

        fake = UnavailableApi()
        with _RuntimePatcher(fake, api_available=False):
            import asyncio

            result = asyncio.run(gsrs_get_substance_details("ASPIRIN", "names"))
        self.assertIn("GSRS API is currently unavailable", result)
        self.assertIn("API down for testing", result)


if __name__ == "__main__":
    unittest.main()
