"""Tests for GSRS markdown summaries."""
import unittest

from app.services.summary import substance_to_markdown


class TestSubstanceSummary(unittest.TestCase):
    def test_substance_to_markdown_matches_json2md_shape(self):
        payload = {
            "_name": "Aspirin",
            "_approvalIDDisplay": "R16CO5Y76E",
            "substanceClass": "chemical",
            "status": "approved",
            "structure": {
                "formula": "C9H8O4",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "mwt": 180.16,
                "stereochemistry": "ACHIRAL",
            },
            "names": [
                {
                    "name": "Aspirin",
                    "type": "cn",
                    "domains": ["drug"],
                    "languages": ["en"],
                }
            ],
            "codes": [
                {
                    "codeSystem": "CAS",
                    "code": "50-78-2",
                    "comments": "primary",
                }
            ],
        }

        markdown = substance_to_markdown(payload)

        self.assertTrue(markdown.startswith("# Substance: Aspirin"))
        self.assertIn("**Approval ID:** R16CO5Y76E", markdown)
        self.assertIn("## Structure", markdown)
        self.assertIn("| Name | Type | Domains | Languages |", markdown)
        self.assertIn("| Aspirin | cn | drug | en |", markdown)
        self.assertIn("| CAS | 50-78-2 | primary |", markdown)
