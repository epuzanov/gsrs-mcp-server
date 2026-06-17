"""Contract tests for the compact MCP tool surface."""
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestCompactToolContract(unittest.TestCase):
    EXPECTED_TOOLS = [
        "rag_query_chunks",
        "rag_ingest",
        "rag_query",
        "get_parent_context",
        "gsrs_get_substance",
        "gsrs_get_summary",
        "gsrs_parametric_search",
        "gsrs_get_facets",
        "gsrs_get_cv_domains",
        "gsrs_get_cv_terms",
        "gsrs_structure_search",
        "gsrs_sequence_search",
        "health",
        "statistics",
    ]

    def _tool_decorators(self):
        """Yield every function decorated with @mcp.tool() in app/main.py."""
        source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in module.body:
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "tool"
                ):
                    yield node.name, decorator

    def test_main_exposes_only_compact_tool_surface(self):
        """The server should not drift back toward the old broad tool set."""
        tool_names = [name for name, _ in self._tool_decorators()]
        self.assertEqual(tool_names, self.EXPECTED_TOOLS)

    def test_all_tools_have_annotations(self):
        """Every MCP tool should declare ToolAnnotations hints."""
        annotated = set()
        for _, decorator in self._tool_decorators():
            for keyword in decorator.keywords or []:
                if keyword.arg == "annotations":
                    annotated.add(True)
                    break
            else:
                annotated.add(False)

        self.assertTrue(
            all(annotated),
            "Some @mcp.tool() decorators are missing annotations=ToolAnnotations(...)",
        )

    def test_read_only_tools_are_not_marked_destructive(self):
        """Tools that do not mutate state must have destructiveHint=False."""
        read_only_tools = {
            "rag_query_chunks",
            "rag_query",
            "get_parent_context",
            "gsrs_get_substance",
            "gsrs_get_summary",
            "gsrs_parametric_search",
            "gsrs_get_facets",
            "gsrs_get_cv_domains",
            "gsrs_get_cv_terms",
            "gsrs_structure_search",
            "gsrs_sequence_search",
            "health",
            "statistics",
        }
        destructive_tools = set()
        for name, decorator in self._tool_decorators():
            if name in read_only_tools:
                for keyword in decorator.keywords or []:
                    if keyword.arg == "annotations":
                        source = ast.unparse(keyword.value)
                        if "destructiveHint=True" in source:
                            destructive_tools.add(name)
                        break
        self.assertEqual(destructive_tools, set())

    def test_ingest_is_marked_non_readonly_and_non_idempotent(self):
        """rag_ingest writes to the vector store and is not idempotent."""
        for name, decorator in self._tool_decorators():
            if name == "rag_ingest":
                annotations = None
                for keyword in decorator.keywords or []:
                    if keyword.arg == "annotations":
                        annotations = keyword.value
                        break
                self.assertIsNotNone(annotations)
                self.assertIn("ToolAnnotations", ast.unparse(annotations))
                source = ast.unparse(annotations)
                self.assertIn("readOnlyHint=False", source)
                self.assertIn("destructiveHint=False", source)
                self.assertIn("idempotentHint=True", source)
                break
        else:
            self.fail("rag_ingest tool not found")
