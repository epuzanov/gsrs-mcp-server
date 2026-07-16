"""
Tests for ``ParentContextEnricher.summarize_parent``.

These exercise the chunk-driven markdown renderer that lives alongside
the existing parent-child retrieval helpers in
``app.services.parent_child_retrieval``. The renderer is a
parent-scoped view: it consumes the same parent context the enricher
already exposes to ``enrich_chunk_with_parent`` and turns it into
markdown, with no extra backend calls.
"""
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from app.models import VectorDocument
from app.services.parent_child_retrieval import (
    ParentContextEnricher,
    ParentIdentity,
)


def _chunk(
    *,
    document_id: UUID,
    section: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_id: Optional[str] = None,
    root_section: Optional[str] = None,
) -> VectorDocument:
    """Build a VectorDocument with chunker-compatible metadata."""
    meta = dict(metadata or {})
    if root_section is None:
        # Default: trust ``root_section_for``'s mapping for the most
        # common sub-sections so tests can stay terse.
        if section in {"identifiers", "classifications"}:
            root_section = "codes"
        elif section in {
            "activemoiety",
            "metabolites",
            "impurities",
            "constituents",
            "salts",
        }:
            root_section = "relationships"
        elif section in {
            "chemical",
            "moieties",
            "protein",
            "nucleicacid",
            "polymer",
            "mixture",
            "structurallydiverse",
            "specifiedsubstance",
            "modifications",
        }:
            root_section = "definitions"
        else:
            root_section = section
    meta.setdefault("root_section", root_section)
    return VectorDocument(
        chunk_id=chunk_id or f"{section}_{uuid4().hex[:8]}",
        document_id=document_id,
        section=section,
        root_section=root_section,
        text=text,
        embedding=[0.0] * 4,
        metadata_json=meta,
    )


class TestSummarizeParentBasics(unittest.TestCase):
    """Smoke tests for the renderer's contract."""

    def setUp(self) -> None:
        self.mock_db = MagicMock()
        self.enricher = ParentContextEnricher(self.mock_db)
        self.doc_id = uuid4()

    def test_returns_empty_string_when_no_chunks(self) -> None:
        self.mock_db.get_documents.return_value = []
        identity = ParentIdentity(document_id=self.doc_id, root_section="names")
        self.assertEqual(self.enricher.summarize_parent(identity), "")

    def test_returns_empty_string_when_only_excluded_chunks(self) -> None:
        chunk = _chunk(
            document_id=self.doc_id,
            section="names",
            text="Name: Aspirin",
            metadata={"chunk_type": "name", "name": "Aspirin"},
        )
        self.mock_db.get_documents.return_value = [chunk]
        identity = ParentIdentity(document_id=self.doc_id, root_section="names")
        self.assertEqual(
            self.enricher.summarize_parent(identity, exclude_chunk_ids={chunk.chunk_id}),
            "",
        )

    def test_uses_display_name_from_metadata(self) -> None:
        chunk = _chunk(
            document_id=self.doc_id,
            section="names",
            text="Name: Aspirin",
            metadata={
                "chunk_type": "name",
                "name": "Aspirin",
                "display_name": "Aspirin",
            },
        )
        self.mock_db.get_documents.return_value = [chunk]
        identity = ParentIdentity(document_id=self.doc_id, root_section="names")
        md = self.enricher.summarize_parent(identity)
        self.assertIn("# Parent: Aspirin — names", md)
        # Footer for the names root should flag the substance-level gap.
        self.assertIn("gsrs_get_summary", md)

    def test_falls_back_to_uuid_when_no_display_name(self) -> None:
        chunk = _chunk(
            document_id=self.doc_id,
            section="names",
            text="Name: Aspirin",
            metadata={"chunk_type": "name", "name": "Aspirin"},
        )
        self.mock_db.get_documents.return_value = [chunk]
        identity = ParentIdentity(document_id=self.doc_id, root_section="names")
        md = self.enricher.summarize_parent(identity)
        self.assertIn(f"# Parent: {self.doc_id} — names", md)


class TestSummarizeParentSections(unittest.TestCase):
    """Per-section rendering rules."""

    def setUp(self) -> None:
        self.mock_db = MagicMock()
        self.enricher = ParentContextEnricher(self.mock_db)
        self.doc_id = uuid4()

    def test_renders_overview_section(self) -> None:
        chunk = _chunk(
            document_id=self.doc_id,
            section="overview",
            text="Substance: Aspirin\nClass: chemical\nAccess: Public",
            metadata={
                "chunk_type": "overview",
                "display_name": "Aspirin",
                "substance_definition_type": "PRIMARY",
            },
        )
        self.mock_db.get_documents.return_value = [chunk]
        identity = ParentIdentity(document_id=self.doc_id, root_section="overview")
        md = self.enricher.summarize_parent(identity)
        self.assertIn("## Overview", md)
        self.assertIn("Aspirin", md)
        self.assertIn("**Definition Type:** PRIMARY", md)

    def test_renders_names_section_with_metadata_columns(self) -> None:
        chunks = [
            _chunk(
                document_id=self.doc_id,
                section="names",
                text="Name: Aspirin\nType: Official Name\nLanguages: en",
                metadata={
                    "chunk_type": "name",
                    "name": "Aspirin",
                    "name_type": "of",
                    "name_orgs": ["WHO INN"],
                    "name_type_label": "Official Name",
                    "access": ["protected"],
                    "display_name": "Aspirin",
                },
            ),
            _chunk(
                document_id=self.doc_id,
                section="names",
                text="Name: Acetylsalicylic Acid\nType: Systematic Name",
                metadata={
                    "chunk_type": "name",
                    "name": "Acetylsalicylic Acid",
                    "name_type": "sys",
                    "name_orgs": [],
                    "access": [],
                    "display_name": "Aspirin",
                },
            ),
        ]
        self.mock_db.get_documents.return_value = chunks
        identity = ParentIdentity(document_id=self.doc_id, root_section="names")
        md = self.enricher.summarize_parent(identity, include_text_parts=False)
        self.assertIn("## Names", md)
        # Both names should appear as table rows.
        self.assertIn("Aspirin", md)
        self.assertIn("Acetylsalicylic Acid", md)
        # ``name_type`` column is rendered via the per-part metadata.
        self.assertIn("| of |", md)
        self.assertIn("| sys |", md)
        # With include_text_parts=False, the text column is not rendered.
        self.assertNotIn("Languages: en", md)

    def test_codes_root_groups_identifiers_and_classifications(self) -> None:
        chunks = [
            _chunk(
                document_id=self.doc_id,
                section="identifiers",
                text="Identifier: CAS:50-78-2\nCode System: CAS",
                metadata={
                    "chunk_type": "identifier",
                    "code": "50-78-2",
                    "code_system": "CAS",
                    "code_type": "primary",
                    "display_name": "Aspirin",
                },
            ),
            _chunk(
                document_id=self.doc_id,
                section="classifications",
                text="Classification: ATC:N02BA01",
                metadata={
                    "chunk_type": "classification",
                    "code": "N02BA01",
                    "code_system": "ATC",
                    "display_name": "Aspirin",
                },
            ),
        ]
        self.mock_db.get_documents.return_value = chunks
        identity = ParentIdentity(document_id=self.doc_id, root_section="codes")
        md = self.enricher.summarize_parent(identity, include_text_parts=False)
        # Both sub-buckets should appear as their own H2 sections.
        self.assertIn("## Identifiers", md)
        self.assertIn("## Classifications", md)
        self.assertIn("CAS", md)
        self.assertIn("ATC", md)

    def test_definitions_root_groups_per_class_sub_sections(self) -> None:
        chunks = [
            _chunk(
                document_id=self.doc_id,
                section="chemical",
                text="Chemical Structure\nSMILES: CC(=O)Oc1ccccc1C(=O)O",
                metadata={
                    "chunk_type": "structure",
                    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "molecular_formula": "C9H8O4",
                    "inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                    "display_name": "Aspirin",
                },
            ),
            _chunk(
                document_id=self.doc_id,
                section="moieties",
                text="Moiety 1\nSMILES: CC(=O)Oc1ccccc1C(=O)O",
                metadata={
                    "chunk_type": "moiety",
                    "moiety_index": 0,
                    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "molecular_formula": "C9H8O4",
                    "display_name": "Aspirin",
                },
            ),
        ]
        self.mock_db.get_documents.return_value = chunks
        identity = ParentIdentity(document_id=self.doc_id, root_section="definitions")
        md = self.enricher.summarize_parent(identity, include_text_parts=False)
        # Both sub-sections nest under the single ``Definitions`` H2.
        self.assertIn("## Definitions", md)
        self.assertIn("### Chemical Structure", md)
        self.assertIn("### Moieties", md)
        self.assertIn("C9H8O4", md)
        # Footer for definitions root should flag the missing
        # substance-level metadata.
        self.assertIn("gsrs_get_summary", md)

    def test_relationship_sub_sections_rebucket_to_typed_h3(self) -> None:
        chunks = [
            _chunk(
                document_id=self.doc_id,
                section="activemoiety",
                text="Relationship: ACTIVE MOIETY\nRelated Substance: Salicylic Acid",
                metadata={
                    "chunk_type": "relationship",
                    "relationship_type": "ACTIVE MOIETY",
                    "related_substance_name": "Salicylic Acid",
                },
            ),
            _chunk(
                document_id=self.doc_id,
                section="metabolites",
                text="Relationship: METABOLITE INACTIVE->PARENT\nRelated Substance: Gentisic Acid",
                metadata={
                    "chunk_type": "relationship",
                    "relationship_type": "METABOLITE INACTIVE->PARENT",
                    "related_substance_name": "Gentisic Acid",
                },
            ),
            _chunk(
                document_id=self.doc_id,
                section="salts",
                text="Relationship: SALT/SOLVATE->PARENT\nRelated Substance: Lysine",
                metadata={
                    "chunk_type": "relationship",
                    "relationship_type": "SALT/SOLVATE->PARENT",
                    "related_substance_name": "Lysine",
                },
            ),
        ]
        self.mock_db.get_documents.return_value = chunks
        identity = ParentIdentity(document_id=self.doc_id, root_section="relationships")
        md = self.enricher.summarize_parent(identity, include_text_parts=False)
        # One H2, three H3s in the order they first appeared.
        self.assertIn("## Relationships", md)
        self.assertIn("### Active Moieties", md)
        self.assertIn("### Metabolites", md)
        self.assertIn("### Salts or Solvates", md)
        # The chunker's section name (``salts``) must not leak.
        self.assertNotIn("### Salts\n", md)
        # The relationships root does NOT get the substance-gap
        # footer — that footer is reserved for overview / definitions
        # / names where approvalID / status are missing.
        self.assertNotIn("gsrs_get_summary", md)

    def test_properties_and_references_render(self) -> None:
        chunks = [
            _chunk(
                document_id=self.doc_id,
                section="properties",
                text="Property: Melting Point\nType: PHYSICAL\nValue: 135 C",
                metadata={
                    "chunk_type": "property",
                    "property_name": "Melting Point",
                    "property_type": "PHYSICAL",
                },
            ),
            _chunk(
                document_id=self.doc_id,
                section="references",
                text="Reference\nType: CITATION\nID: PMID-12345",
                metadata={
                    "chunk_type": "reference",
                    "doc_type": "CITATION",
                    "reference_id": "PMID-12345",
                },
            ),
        ]
        self.mock_db.get_documents.return_value = chunks
        identity = ParentIdentity(document_id=self.doc_id, root_section="properties")
        md = self.enricher.summarize_parent(identity, include_text_parts=False)
        self.assertIn("## Properties", md)
        self.assertIn("Melting Point", md)
        self.assertIn("| PHYSICAL |", md)

    def test_include_text_parts_appends_text_column(self) -> None:
        chunk = _chunk(
            document_id=self.doc_id,
            section="names",
            text="Name: Aspirin\nType: Official Name",
            metadata={
                "chunk_type": "name",
                "name": "Aspirin",
                "name_type": "of",
                "display_name": "Aspirin",
            },
        )
        self.mock_db.get_documents.return_value = [chunk]
        identity = ParentIdentity(document_id=self.doc_id, root_section="names")
        md = self.enricher.summarize_parent(identity, include_text_parts=True)
        # Text column is present and carries the chunk's text.
        self.assertIn("| Text |", md)
        self.assertIn("Name: Aspirin", md)


class TestSummarizeParentLimits(unittest.TestCase):
    """Size / cap behaviour."""

    def setUp(self) -> None:
        self.mock_db = MagicMock()
        self.enricher = ParentContextEnricher(self.mock_db)
        self.doc_id = uuid4()

    def test_max_chars_truncates_and_increments_metric(self) -> None:
        # Build a single very long name so the rendered markdown is
        # forced over the cap.
        long_text = "X" * 5000
        chunk = _chunk(
            document_id=self.doc_id,
            section="names",
            text=long_text,
            metadata={
                "chunk_type": "name",
                "name": "Aspirin",
                "display_name": "Aspirin",
            },
        )
        self.mock_db.get_documents.return_value = [chunk]
        identity = ParentIdentity(document_id=self.doc_id, root_section="names")
        md = self.enricher.summarize_parent(identity, max_chars=200)
        self.assertLessEqual(len(md), 200)
        self.assertTrue(md.rstrip().endswith("..."))
        snapshot = self.enricher.metrics.snapshot()
        self.assertGreater(snapshot.get("counters", {}).get("parent_summary.truncated", 0), 0)

    def test_bucket_limit_caps_rendered_rows(self) -> None:
        chunks = [
            _chunk(
                document_id=self.doc_id,
                section="properties",
                text=f"Property: P{i}\nValue: v{i}",
                metadata={
                    "chunk_type": "property",
                    "property_name": f"P{i}",
                    "property_type": "PHYSICAL",
                },
            )
            for i in range(80)
        ]
        self.mock_db.get_documents.return_value = chunks
        identity = ParentIdentity(document_id=self.doc_id, root_section="properties")
        md = self.enricher.summarize_parent(identity, include_text_parts=False)
        # 50 rows + header + separator = 52 table lines for a 5-col
        # table — we just check the table line count is bounded.
        table_lines = [
            line for line in md.splitlines() if line.startswith("|")
        ]
        # header + separator + 50 rows
        self.assertEqual(len(table_lines), 52)


# ------------------------------------------------------------------
# Public-surface tests for the ``get_parent_summary`` tool and the
# ``gsrs://substances/{identifier}/parents/{root_section}/summary``
# resource. These exercise the wiring in ``app/main.py`` end-to-end
# (helper → runtime.parent_enricher.summarize_parent) without
# booting the MCP server.
# ------------------------------------------------------------------


class _FakeEnricher:
    """Minimal stand-in for ``runtime.parent_enricher`` used by the tool.

    The MCP tool/resource call :func:`runtime.parent_enricher.summarize_parent`,
    so the public-surface tests only need to intercept that single
    method.
    """

    def __init__(self, body: str = "# stub markdown\n") -> None:
        self.body = body
        self.calls: list[dict] = []

    def summarize_parent(
        self,
        parent_identity,
        *,
        max_chars: int = 4000,
        include_text_parts: bool = True,
        exclude_chunk_ids=None,
    ):
        from app.services.parent_child_retrieval import ParentIdentity

        assert isinstance(parent_identity, ParentIdentity)
        self.calls.append(
            {
                "document_id": str(parent_identity.document_id),
                "root_section": parent_identity.root_section,
                "max_chars": max_chars,
                "include_text_parts": include_text_parts,
                "exclude_chunk_ids": set(exclude_chunk_ids or ()),
            }
        )
        return self.body


class _RuntimePatcher:
    """Swap ``runtime.parent_enricher`` and bypass init for public tests."""

    def __init__(self, enricher):
        self.enricher = enricher
        self._runtime = None
        self._original_enricher = None
        self._original_initialized = None
        self._original_components = None

    def __enter__(self):
        from app.main import runtime as runtime_module
        from app.runtime import ComponentStatus

        self._runtime = runtime_module
        # ``parent_enricher`` is a @property with no setter, so the
        # canonical override point is the underlying ``_parent_enricher``
        # slot (see app/runtime.py).
        self._original_enricher = self._runtime._parent_enricher
        self._original_initialized = self._runtime.initialized
        self._original_components = self._runtime.components
        self._runtime._parent_enricher = self.enricher
        self._runtime.initialized = True
        self._runtime.components = {
            "vector_db": ComponentStatus(
                name="vector_db", required=True, ready=True
            ),
            "embedding": ComponentStatus(
                name="embedding", required=True, ready=True
            ),
        }
        return self

    def __exit__(self, *exc):
        self._runtime._parent_enricher = self._original_enricher
        self._runtime.initialized = self._original_initialized
        self._runtime.components = self._original_components


class TestRenderParentSummaryHelper(unittest.TestCase):
    """Direct tests for the canonical ``_render_parent_summary`` helper."""

    def test_returns_validation_error_for_empty_uuid(self) -> None:
        import app.main
        from app.main import _render_parent_summary

        enricher = _FakeEnricher()
        with _RuntimePatcher(enricher):
            body, error = _render_parent_summary(
                substance_uuid="",
                root_section="names",
                max_chars=4000,
                include_text_parts=True,
                exclude_chunk_id="",
            )
        self.assertEqual(body, "")
        self.assertIn("substance_uuid is required", error)
        self.assertEqual(enricher.calls, [])

    def test_returns_validation_error_for_invalid_uuid(self) -> None:
        import app.main
        from app.main import _render_parent_summary

        enricher = _FakeEnricher()
        with _RuntimePatcher(enricher):
            body, error = _render_parent_summary(
                substance_uuid="not-a-uuid",
                root_section="names",
                max_chars=4000,
                include_text_parts=True,
                exclude_chunk_id="",
            )
        self.assertEqual(body, "")
        self.assertIn("is not a valid UUID", error)
        self.assertEqual(enricher.calls, [])

    def test_returns_validation_error_for_empty_root_section(self) -> None:
        import app.main
        from app.main import _render_parent_summary

        enricher = _FakeEnricher()
        with _RuntimePatcher(enricher):
            body, error = _render_parent_summary(
                substance_uuid="0103a288-6eb6-4ced-b13a-849cd7edf028",
                root_section="   ",
                max_chars=4000,
                include_text_parts=True,
                exclude_chunk_id="",
            )
        self.assertEqual(body, "")
        self.assertIn("root_section is required", error)
        self.assertEqual(enricher.calls, [])

    def test_happy_path_returns_body_and_no_error(self) -> None:
        import app.main
        from app.main import _render_parent_summary

        enricher = _FakeEnricher(body="# Parent: ASPIRIN — names\n")
        with _RuntimePatcher(enricher):
            body, error = _render_parent_summary(
                substance_uuid="0103a288-6eb6-4ced-b13a-849cd7edf028",
                root_section="names",
                max_chars=2000,
                include_text_parts=False,
                exclude_chunk_id="names_xyz",
            )
        self.assertEqual(error, "")
        self.assertEqual(body, "# Parent: ASPIRIN — names\n")
        self.assertEqual(len(enricher.calls), 1)
        call = enricher.calls[0]
        self.assertEqual(
            call["document_id"], "0103a288-6eb6-4ced-b13a-849cd7edf028"
        )
        self.assertEqual(call["root_section"], "names")
        self.assertEqual(call["max_chars"], 2000)
        self.assertFalse(call["include_text_parts"])
        self.assertEqual(call["exclude_chunk_ids"], {"names_xyz"})

    def test_empty_body_when_no_chunks(self) -> None:
        import app.main
        from app.main import _render_parent_summary

        enricher = _FakeEnricher(body="")
        with _RuntimePatcher(enricher):
            body, error = _render_parent_summary(
                substance_uuid="0103a288-6eb6-4ced-b13a-849cd7edf028",
                root_section="references",
                max_chars=4000,
                include_text_parts=True,
                exclude_chunk_id="",
            )
        self.assertEqual(error, "")
        self.assertEqual(body, "")


class TestParentSummaryResource(unittest.TestCase):
    """Tests for the ``gsrs://substances/{uuid}/parents/{root}/summary`` resource."""

    def test_resource_returns_markdown(self) -> None:
        import asyncio

        import app.main
        from app.main import gsrs_parent_summary_resource

        enricher = _FakeEnricher(body="# Parent: ASPIRIN — definitions\n")
        with _RuntimePatcher(enricher):
            result = asyncio.run(
                gsrs_parent_summary_resource(
                    "0103a288-6eb6-4ced-b13a-849cd7edf028", "definitions"
                )
            )
        self.assertEqual(result, "# Parent: ASPIRIN — definitions\n")
        self.assertEqual(len(enricher.calls), 1)
        self.assertEqual(enricher.calls[0]["root_section"], "definitions")
        self.assertEqual(
            enricher.calls[0]["document_id"],
            "0103a288-6eb6-4ced-b13a-849cd7edf028",
        )

    def test_resource_rejects_non_uuid_identifier(self) -> None:
        import asyncio

        import app.main
        from app.main import gsrs_parent_summary_resource

        enricher = _FakeEnricher()
        with _RuntimePatcher(enricher):
            result = asyncio.run(
                gsrs_parent_summary_resource("ASPIRIN", "names")
            )
        self.assertIn("is not a valid UUID", result)
        self.assertEqual(enricher.calls, [])

    def test_resource_returns_no_parent_message_when_empty(self) -> None:
        import asyncio

        import app.main
        from app.main import gsrs_parent_summary_resource

        enricher = _FakeEnricher(body="")
        with _RuntimePatcher(enricher):
            result = asyncio.run(
                gsrs_parent_summary_resource(
                    "0103a288-6eb6-4ced-b13a-849cd7edf028", "references"
                )
            )
        self.assertIn("No parent context found", result)


if __name__ == "__main__":
    unittest.main()
