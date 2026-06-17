"""Tests for GSRS markdown summaries."""
import json
import os
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

    def _load_example(self, filename: str) -> dict:
        root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(root, "examples", filename)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_chemical_summary_renders_structure_and_codes(self):
        data = self._load_example("0103a288-6eb6-4ced-b13a-849cd7edf028.json")
        md = substance_to_markdown(data)
        self.assertIn("## Structure", md)
        self.assertIn("C13H18O2", md)
        self.assertIn("## Codes", md)
        self.assertIn("WK2XYI10QM", md)
        self.assertIn("## Names", md)

    def test_mixture_summary_renders_components(self):
        data = self._load_example("34a6ff49-23f3-4079-8d1a-544146ac6d62.json")
        md = substance_to_markdown(data)
        self.assertIn("## Mixture Components", md)
        self.assertIn("MUST_BE_PRESENT", md)
        self.assertIn("2-Methylcinnamic acid", md)

    def test_nucleic_acid_summary_renders_subunits(self):
        data = self._load_example("820a818b-e5cd-4a2d-82a1-fd4e50b8da1e.json")
        md = substance_to_markdown(data)
        self.assertIn("## Nucleic Acid Details", md)
        self.assertIn("PLASMID", md)
        self.assertIn("## Subunits", md)
        self.assertIn("length 6236", md)

    def test_polymer_summary_renders_monomers(self):
        data = self._load_example("b7d919e4-7fef-4c9e-bb03-f8112608c050.json")
        md = substance_to_markdown(data)
        self.assertIn("## Polymer Classification", md)
        self.assertIn("HOMOPOLYMER", md)
        self.assertIn("## Polymer Structure", md)
        self.assertIn("## Monomers", md)
        self.assertIn(".ALPHA.-D-GLUCOPYRANOSE", md)

    def test_protein_summary_renders_subunits_and_glycosylation(self):
        data = self._load_example("d3547410-a29a-4634-9963-2ec80daba7ca.json")
        md = substance_to_markdown(data)
        self.assertIn("## Protein Details", md)
        self.assertIn("MONOCLONAL ANTIBODY", md)
        self.assertIn("## Subunits", md)
        self.assertIn("## Glycosylation", md)
        self.assertIn("NGlycosylationSites", md)

    def test_structurally_diverse_summary_renders_source_material(self):
        data = self._load_example("8a1c6912-c89e-48cf-b182-94622a9ac1a1.json")
        md = substance_to_markdown(data)
        self.assertIn("## Source Material", md)
        self.assertIn("RECOMBINANT VIRUS", md)
        self.assertIn("Dependoparvovirus", md)

    def test_specified_substance_g1_summary_renders_constituents(self):
        data = self._load_example("f05749eb-1559-43ed-8bba-253d04a082d6.json")
        md = substance_to_markdown(data)
        self.assertIn("## Specified Substance Constituents", md)
        self.assertIn("Anhydrous dextrose", md)
        self.assertIn("Starmarella bombicola", md)

    def test_concept_summary_renders_basic_header(self):
        data = self._load_example("84be3229-7c19-4a9f-bafb-608832e61888.json")
        md = substance_to_markdown(data)
        self.assertIn("# Substance: PAPAVER SOMNIFERUM PERICARP", md)
        self.assertIn("concept", md)
        self.assertIn("## Names", md)

    def test_properties_table_renders_amounts(self):
        data = self._load_example("0103a288-6eb6-4ced-b13a-849cd7edf028.json")
        md = substance_to_markdown(data)
        self.assertIn("## Properties", md)
        self.assertIn("Cmax", md)
        self.assertIn("microgram/mL", md)

    def test_chemical_summary_renders_moieties(self):
        data = self._load_example("0103a288-6eb6-4ced-b13a-849cd7edf028.json")
        md = substance_to_markdown(data)
        self.assertIn("## Moieties", md)
        self.assertIn("C13H18O2", md)
        self.assertIn("RACEMIC", md)

    def test_chemical_alternative_summary_renders_moieties(self):
        data = self._load_example("d4ee19a6-a33f-4d37-bf05-5feeaa938b83.json")
        md = substance_to_markdown(data)
        self.assertIn("## Moieties", md)
        self.assertIn("C10H10O2", md)
        self.assertIn("ACHIRAL", md)
