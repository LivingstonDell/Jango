from __future__ import annotations

import unittest

from nbia.numbering import (
    CANONICAL_NUMBERING_SCHEME,
    anarci_sort_key,
    cdr_name,
    framework_or_cdr,
    imgt_region,
    parse_anarci_numbered_residues,
)


class NumberingTests(unittest.TestCase):
    def test_imgt_region_boundaries(self) -> None:
        self.assertEqual(imgt_region(26), "FR")
        self.assertEqual(imgt_region(27), "CDR1")
        self.assertEqual(imgt_region(38), "CDR1")
        self.assertEqual(imgt_region(39), "FR")
        self.assertEqual(imgt_region(56), "CDR2")
        self.assertEqual(imgt_region(65), "CDR2")
        self.assertEqual(imgt_region(105), "CDR3")
        self.assertEqual(imgt_region(117), "CDR3")
        self.assertEqual(imgt_region(118), "FR")

    def test_framework_and_cdr_labels(self) -> None:
        self.assertEqual(framework_or_cdr("FR"), "Framework")
        self.assertEqual(framework_or_cdr("CDR1"), "CDR")
        self.assertEqual(cdr_name("FR"), "")
        self.assertEqual(cdr_name("CDR2"), "CDR2")

    def test_parse_anarci_numbered_residues_skips_gaps_and_preserves_insertions(self) -> None:
        rows = parse_anarci_numbered_residues(
            [
                ((26, " "), "Q"),
                ((27, " "), "A"),
                ((111, "A"), "G"),
                ((112, " "), "-"),
                ((112, " "), "Y"),
            ]
        )
        self.assertEqual([row["sequence_index"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(rows[1]["imgt_region"], "CDR1")
        self.assertEqual(rows[2]["anarci_label"], "111A")
        self.assertEqual(rows[2]["numbering_scheme"], CANONICAL_NUMBERING_SCHEME)
        self.assertGreater(anarci_sort_key(111, "A"), 111.0)
        self.assertLess(anarci_sort_key(111, "A"), 112.0)


if __name__ == "__main__":
    unittest.main()
