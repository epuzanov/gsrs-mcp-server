"""Tests for the GSRS details filter parser."""

import unittest

from app.services.details import (
    KNOWN_AGGREGATIONS,
    FilterError,
    FilterSegment,
    ParsedFilter,
    parse_filter,
    validate_filter,
)


class TestParseFilter(unittest.TestCase):
    def test_empty_filter_is_empty(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                result = parse_filter(value)
                self.assertTrue(result.empty)
                self.assertEqual(result.segments, [])

    def test_plain_field_segment(self):
        result = parse_filter("names")
        self.assertFalse(result.empty)
        self.assertEqual(len(result.segments), 1)
        segment = result.segments[0]
        self.assertEqual(segment.field_path, "names")
        self.assertEqual(segment.locators, [])
        self.assertEqual(segment.projections, [])
        self.assertEqual(segment.aggregations, [])

    def test_field_with_locator(self):
        result = parse_filter("names(type:of)")
        segment = result.segments[0]
        self.assertEqual(segment.field_path, "names")
        self.assertEqual(segment.locators, ["(type:of)"])
        self.assertEqual(segment.projections, [])

    def test_field_with_index_locator(self):
        result = parse_filter("codes($0)")
        segment = result.segments[0]
        self.assertEqual(segment.field_path, "codes")
        self.assertTrue(segment.is_index_locator)

    def test_field_with_nested_path(self):
        result = parse_filter("relationships/relatedSubstance")
        self.assertEqual(result.segments[0].field_path, "relationships/relatedSubstance")

    def test_multiple_locators(self):
        result = parse_filter("names(type:of)(deprecated:false)")
        segment = result.segments[0]
        self.assertEqual(segment.locators, ["(type:of)", "(deprecated:false)"])

    def test_chained_locators_with_arrow(self):
        result = parse_filter("relationships(type:IMPURITY->PARENT)")
        segment = result.segments[0]
        self.assertEqual(segment.locators, ["(type:IMPURITY->PARENT)"])

    def test_projection(self):
        result = parse_filter("names!(name)")
        segment = result.segments[0]
        self.assertEqual(segment.projections, ["name"])

    def test_limit_aggregation(self):
        result = parse_filter("names(type:of)!(name)!limit(1)")
        segment = result.segments[0]
        self.assertEqual(segment.aggregations, [("limit", "1")])
        self.assertEqual(segment.projections, ["name"])

    def test_count_aggregation(self):
        result = parse_filter("names!count()")
        segment = result.segments[0]
        self.assertEqual(segment.aggregations, [("count", "")])

    def test_skip_limit_chain(self):
        result = parse_filter("names!skip(5)!limit(10)")
        segment = result.segments[0]
        self.assertEqual(segment.aggregations, [("skip", "5"), ("limit", "10")])

    def test_sort_aggregation(self):
        result = parse_filter("codes!sort(codeSystem)")
        segment = result.segments[0]
        self.assertEqual(segment.aggregations, [("sort", "codeSystem")])

    def test_revsort_aggregation(self):
        result = parse_filter("names!revsort(created)")
        segment = result.segments[0]
        self.assertEqual(segment.aggregations, [("revsort", "created")])

    def test_complex_filter(self):
        result = parse_filter(
            "relationships(type:IMPURITY->PARENT)"
            "(relatedSubstance/approvalID:27794-1)!(relatedSubstance/refPname)"
        )
        segment = result.segments[0]
        self.assertEqual(segment.field_path, "relationships")
        self.assertEqual(
            segment.locators,
            [
                "(type:IMPURITY->PARENT)",
                "(relatedSubstance/approvalID:27794-1)",
            ],
        )
        self.assertEqual(segment.projections, ["relatedSubstance/refPname"])

    def test_raw_is_preserved(self):
        raw = "  names(type:of)!(name)  "
        result = parse_filter(raw)
        self.assertEqual(result.raw, "names(type:of)!(name)")

    def test_known_aggregations_set(self):
        # The list of well-known aggregations should include the
        # documented ones.
        for name in ("sort", "revsort", "skip", "limit", "distinct", "count", "group"):
            self.assertIn(name, KNOWN_AGGREGATIONS)


class TestParseFilterErrors(unittest.TestCase):
    def test_unbalanced_paren_in_locator(self):
        with self.assertRaises(FilterError):
            parse_filter("names(type:of")

    def test_unbalanced_paren_in_aggregation(self):
        with self.assertRaises(FilterError):
            parse_filter("names!limit(1")

    def test_filter_must_start_with_field(self):
        with self.assertRaises(FilterError):
            parse_filter("!(name)")

    def test_aggregation_requires_field_argument(self):
        with self.assertRaises(FilterError):
            parse_filter("names!sort()")

    def test_limit_requires_integer(self):
        with self.assertRaises(FilterError):
            parse_filter("names!limit(abc)")

    def test_skip_requires_integer(self):
        with self.assertRaises(FilterError):
            parse_filter("names!skip(1.5)")

    def test_bang_without_name_or_arg(self):
        # ``!()`` (no field inside) is a meaningless token.
        with self.assertRaises(FilterError):
            parse_filter("names!()")

    def test_validate_filter_does_not_raise_on_valid(self):
        # Should not raise.
        validate_filter("names(type:of)!(name)!limit(1)")

    def test_validate_filter_raises_on_invalid(self):
        with self.assertRaises(FilterError):
            validate_filter("names(type:of")


if __name__ == "__main__":
    unittest.main()
