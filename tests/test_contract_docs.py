"""Contract tests for public-facing runtime documentation and packaging."""
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestReadmeRuntimeContract(unittest.TestCase):
    def test_readme_front_page_mentions_current_mcp_contract(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        required_fragments = [
            "/mcp",
            "stdio",
            "/livez",
            "/readyz",
            "/health",
            "Authorization: Bearer <MCP_PASSWORD>",
            "gsrs-mcp-server",
        ]

        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, readme)

    def test_readme_explicitly_rejects_legacy_rest_contract(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("no longer documents or relies on legacy REST-style routes", readme)
        self.assertNotIn("HTTP Basic Auth and API Key support", readme)

    def test_examples_and_contributing_match_current_cli_contract(self):
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        examples = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")

        self.assertIn("gsrs-mcp-server", contributing)
        self.assertIn("/mcp", examples)
        self.assertIn("MCP_PASSWORD", examples)


class TestPackagingAndRuntimeArtifacts(unittest.TestCase):
    def test_pyproject_exposes_gsrs_mcp_server_entrypoint(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = pyproject["project"]["scripts"]
        self.assertEqual(scripts["gsrs-mcp-server"], "app.main:main")

    def test_container_healthchecks_follow_readiness_contract(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

        self.assertIn("/readyz", dockerfile)
        self.assertIn("/readyz", compose)
        self.assertNotIn("HTTP Basic", compose)

    def test_readme_references_current_podman_manifest_path(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("podman-kube-play.yaml", readme)
        self.assertNotIn("deploy/podman-kube-play.yaml", readme)
