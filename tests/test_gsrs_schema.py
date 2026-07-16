"""Tests for the ``gsrs_get_schema`` tool and ``gsrs://schema/{model}`` resource."""

import asyncio
import json
import unittest
from unittest.mock import patch

import app.main
from app.main import gsrs_get_schema, gsrs_schema_resource
from app.services.gsrs_schema import SUPPORTED_MODELS


class TestGsrsSchemaResource(unittest.TestCase):
    """Direct tests of the resource function (no runtime init needed)."""

    def test_default_substance_schema(self):
        """The default model returns the Substance JSON Schema."""
        result = asyncio.run(gsrs_schema_resource("Substance"))
        payload = json.loads(result)
        self.assertEqual(payload["model"], "Substance")
        schema = payload["schema"]
        self.assertEqual(schema["title"], "Substance")
        # Substance is the polymorphic base — it has 28 top-level props.
        self.assertGreaterEqual(len(schema.get("properties", {})), 20)
        # The schema includes a $defs section with nested models.
        self.assertIn("$defs", schema)

    def test_subclass_schema(self):
        """A specific subclass returns its own schema."""
        result = asyncio.run(gsrs_schema_resource("ChemicalSubstance"))
        payload = json.loads(result)
        self.assertEqual(payload["model"], "ChemicalSubstance")
        self.assertEqual(payload["schema"]["title"], "ChemicalSubstance")

    def test_empty_string_defaults_to_substance(self):
        """An empty model name falls back to Substance."""
        result = asyncio.run(gsrs_schema_resource(""))
        payload = json.loads(result)
        self.assertEqual(payload["model"], "Substance")
        self.assertEqual(payload["schema"]["title"], "Substance")

    def test_unknown_model_returns_error_envelope(self):
        """An unknown model returns a JSON error envelope with code."""
        result = asyncio.run(gsrs_schema_resource("DefinitelyNotARealModel"))
        payload = json.loads(result)
        self.assertIn("error", payload)
        self.assertEqual(payload["code"], "model_not_found")
        # Error message lists the supported models so callers can self-correct.
        for name in SUPPORTED_MODELS:
            self.assertIn(name, payload["error"])

    def test_missing_dependency_returns_error_envelope(self):
        """If gsrs-model is not installed, the error code reflects that."""
        # Simulate the package being uninstalled by patching the
        # import inside the service module.
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "gsrs.model" or name.startswith("gsrs.model"):
                raise ImportError("simulated missing dependency")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = asyncio.run(gsrs_schema_resource("Substance"))
        payload = json.loads(result)
        self.assertEqual(payload["code"], "dependency_missing")

    def test_supported_models_are_well_known(self):
        """The supported-model set is a stable, hand-curated list."""
        expected = {
            "Substance",
            "ChemicalSubstance",
            "ProteinSubstance",
            "NucleicAcidSubstance",
            "MixtureSubstance",
            "PolymerSubstance",
            "StructurallyDiverseSubstance",
        }
        self.assertEqual(SUPPORTED_MODELS, expected)


class TestGsrsSchemaTool(unittest.TestCase):
    """Tests for the ``gsrs_get_schema`` tool.

    The tool is a thin wrapper over the resource; the tests focus
    on telemetry classification and delegation behavior rather
    than re-testing the schema-lookup logic.
    """

    def _patch_runtime(self):
        """Bypass the runtime init by injecting a fake initialized runtime."""
        from app.main import runtime as runtime_module
        from app.runtime import ComponentStatus

        original_initialized = runtime_module.initialized
        original_components = runtime_module.components
        runtime_module.initialized = True
        runtime_module.components = {
            "gsrs_api": ComponentStatus(
                name="gsrs_api", required=True, ready=True
            )
        }
        return runtime_module, original_initialized, original_components

    def _restore_runtime(self, runtime_module, original_initialized, original_components):
        runtime_module.initialized = original_initialized
        runtime_module.components = original_components

    def test_tool_delegates_to_resource(self):
        """The tool returns exactly what the resource returns for a known model."""
        runtime_module, original_initialized, original_components = self._patch_runtime()
        try:
            result = asyncio.run(gsrs_get_schema("ProteinSubstance"))
            payload = json.loads(result)
            self.assertEqual(payload["model"], "ProteinSubstance")
            self.assertEqual(payload["schema"]["title"], "ProteinSubstance")
        finally:
            self._restore_runtime(
                runtime_module, original_initialized, original_components
            )

    def test_tool_default_returns_substance(self):
        """No argument defaults to the Substance model."""
        runtime_module, original_initialized, original_components = self._patch_runtime()
        try:
            result = asyncio.run(gsrs_get_schema())
            payload = json.loads(result)
            self.assertEqual(payload["model"], "Substance")
        finally:
            self._restore_runtime(
                runtime_module, original_initialized, original_components
            )

    def test_tool_returns_error_for_unknown_model(self):
        """Unknown model returns the same error envelope as the resource."""
        runtime_module, original_initialized, original_components = self._patch_runtime()
        try:
            result = asyncio.run(gsrs_get_schema("BogusModel"))
            payload = json.loads(result)
            self.assertIn("error", payload)
            self.assertEqual(payload["code"], "model_not_found")
        finally:
            self._restore_runtime(
                runtime_module, original_initialized, original_components
            )


class TestSchemaRegistration(unittest.TestCase):
    """Verify the tool + resource are registered with FastMCP."""

    def test_tool_registered(self):
        names = {t.name for t in app.main.mcp._tool_manager._tools.values()}
        self.assertIn("gsrs_get_schema", names)

    def test_resource_template_registered(self):
        templates = set(app.main.mcp._resource_manager._templates.keys())
        self.assertIn("gsrs://schema/{model}", templates)

    def test_instructions_reference_schema_resource(self):
        """The server-level instructions mention the new schema resource."""
        # The instructions string is built at module import time;
        # accessing the mcp instance gives us the live value.
        self.assertIn("gsrs://schema/{model}", app.main.mcp.instructions)


if __name__ == "__main__":
    unittest.main()
