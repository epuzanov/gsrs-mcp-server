"""Contract tests for the compact MCP tool surface."""
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestCompactToolContract(unittest.TestCase):
    def test_main_exposes_only_compact_tool_surface(self):
        """The server should not drift back toward the old broad tool set."""
        source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        tool_names = []
        for node in module.body:
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "tool"
                ):
                    tool_names.append(node.name)

        self.assertEqual(
            tool_names,
            [
                "rag_query_chunks",
                "rag_ingest",
                "rag_query",
                "get_parent_context",
                "gsrs_get_substance",
                "gsrs_get_summary",
                "gsrs_parametric_search",
                "gsrs_get_facets",
                "gsrs_structure_search",
                "gsrs_sequence_search",
                "health",
                "statistics",
            ],
        )
