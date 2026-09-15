from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
from pathlib import Path

import pandas as pd

from nbia.numbering import imgt_region
from jango.analysis.mutation_localization import (
    compute_mutation_localization_outputs,
    localize_mutations,
    save_mutation_position_frequency_plot,
    save_mutation_region_bar,
)


def numbering_fixture(structure_id: str = "s1", length: int = 117) -> pd.DataFrame:
    rows = []
    for index in range(1, length + 1):
        region = imgt_region(index)
        rows.append(
            {
                "structure_id": structure_id,
                "sequence_index": index,
                "anarci_number": index,
                "anarci_insertion": "",
                "anarci_label": str(index),
                "anarci_sort_key": float(index),
                "imgt_region": region,
            }
        )
    return pd.DataFrame(rows)


def design_sequence(native: str, replacements: dict[int, str]) -> str:
    letters = list(native)
    for position, aa in replacements.items():
        letters[position - 1] = aa
    return "".join(letters)


def base_table(designed: str, native: str | None = None) -> pd.DataFrame:
    native = native or "A" * 117
    return pd.DataFrame(
        [
            {
                "experiment_namespace": "boltz2_1_auto_by_length",
                "structure_id": "s1",
                "design_id": "d1",
                "native_nanobody_sequence": native,
                "designed_nanobody_sequence": designed,
                "fold_backend": "boltz2",
                "redesign_mode": "hotspot_only",
            }
        ]
    )


class MutationLocalizationTests(unittest.TestCase):
    def test_framework_and_cdr_assignments_for_multiple_mutations(self) -> None:
        native = "A" * 117
        designed = design_sequence(native, {10: "C", 27: "D", 56: "E", 105: "F"})
        locations, opportunities, skipped, numbering_status = localize_mutations(base_table(designed, native), numbering_fixture())
        self.assertTrue(skipped.empty)
        self.assertEqual(numbering_status.iloc[0]["numbering_status"], "ok")
        self.assertEqual(numbering_status.iloc[0]["numbering_provenance"], "precomputed_table")
        self.assertEqual(len(locations), 4)
        self.assertEqual(numbering_status.iloc[0]["numbering_status"], "ok")
        self.assertEqual(set(locations["region"]), {"Framework", "CDR1", "CDR2", "CDR3"})
        cdr3 = locations[locations["region"] == "CDR3"].iloc[0]
        self.assertEqual(cdr3["native_position"], 105)
        self.assertEqual(cdr3["canonical_position"], "105")
        self.assertEqual(cdr3["framework_or_cdr"], "CDR")
        self.assertEqual(cdr3["cdr_name"], "CDR3")
        self.assertEqual(len(opportunities), 117)

    def test_summary_counts_and_fractions(self) -> None:
        native = "A" * 117
        designed = design_sequence(native, {10: "C", 27: "D", 56: "E", 105: "F"})
        locations, summary, frequency, enrichment, skipped, numbering_status = compute_mutation_localization_outputs(base_table(designed, native), numbering_fixture())
        row = summary.iloc[0]
        self.assertEqual(row["framework_mutations"], 1)
        self.assertEqual(row["cdr1_mutations"], 1)
        self.assertEqual(row["cdr2_mutations"], 1)
        self.assertEqual(row["cdr3_mutations"], 1)
        self.assertEqual(row["total_mutations"], 4)
        self.assertAlmostEqual(row["framework_fraction"], 0.25)
        self.assertFalse(frequency.empty)
        self.assertFalse(enrichment.empty)
        self.assertTrue(skipped.empty)
        self.assertEqual(len(locations), 4)

    def test_no_mutations_still_writes_zero_summary(self) -> None:
        native = "A" * 117
        locations, summary, frequency, enrichment, skipped, numbering_status = compute_mutation_localization_outputs(base_table(native, native), numbering_fixture())
        self.assertTrue(locations.empty)
        self.assertEqual(summary.iloc[0]["total_mutations"], 0)
        self.assertEqual(summary.iloc[0]["cdr3_fraction"], 0.0)
        self.assertEqual(int(frequency["n_mutations"].sum()), 0)
        self.assertFalse(enrichment.empty)
        self.assertTrue(skipped.empty)
        self.assertEqual(numbering_status.iloc[0]["numbering_status"], "ok")

    def test_length_mismatch_or_indel_is_reported_as_skipped(self) -> None:
        native = "A" * 117
        designed = native + "C"
        locations, opportunities, skipped, numbering_status = localize_mutations(base_table(designed, native), numbering_fixture())
        self.assertTrue(locations.empty)
        self.assertTrue(opportunities.empty)
        self.assertEqual(skipped.iloc[0]["skip_reason"], "sequence_length_mismatch_or_indel")
        self.assertEqual(numbering_status.iloc[0]["numbering_status"], "not_attempted")

    def test_missing_explicit_nanobody_sequences_are_not_treated_as_antigen(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "structure_id": "s1",
                    "design_id": "d1",
                    "native_antigen_sequence": "AAAA",
                    "designed_sequence": "CCCC",
                }
            ]
        )
        locations, opportunities, skipped, numbering_status = localize_mutations(table, numbering_fixture(length=4))
        self.assertTrue(locations.empty)
        self.assertTrue(opportunities.empty)
        self.assertEqual(skipped.iloc[0]["skip_reason"], "missing_explicit_nanobody_sequence_columns")
        self.assertTrue(numbering_status.empty)

    def test_external_anarci_runtime_success_is_recorded(self) -> None:
        native = "A" * 117
        designed = design_sequence(native, {105: "F"})
        rows = numbering_fixture().to_dict("records")
        with patch("jango.analysis.mutation_localization.number_vhh_sequence_imgt", return_value=rows) as mocked:
            locations, opportunities, skipped, numbering_status = localize_mutations(
                base_table(designed, native),
                anarci_python="/opt/anarci/bin/python",
            )
        self.assertTrue(skipped.empty)
        self.assertEqual(len(locations), 1)
        self.assertEqual(locations.iloc[0]["region"], "CDR3")
        self.assertEqual(numbering_status.iloc[0]["numbering_status"], "ok")
        self.assertIn("/opt/anarci/bin/python", numbering_status.iloc[0]["numbering_provenance"])
        mocked.assert_called_once()

    def test_failed_numbering_is_preserved_explicitly(self) -> None:
        native = "A" * 117
        locations, opportunities, skipped, numbering_status = localize_mutations(base_table(native, native))
        self.assertTrue(locations.empty)
        self.assertTrue(opportunities.empty)
        self.assertEqual(skipped.iloc[0]["skip_reason"], "numbering_failed:AnarciUnavailableError")
        self.assertEqual(numbering_status.iloc[0]["numbering_status"], "failed")
        self.assertEqual(numbering_status.iloc[0]["numbering_failure_reason"], "numbering_failed:AnarciUnavailableError")

    def test_insertion_code_position_assignment(self) -> None:
        native = "A" * 117
        designed = design_sequence(native, {27: "D"})
        numbering = numbering_fixture()
        numbering.loc[numbering["sequence_index"] == 27, "anarci_number"] = 27
        numbering.loc[numbering["sequence_index"] == 27, "anarci_insertion"] = "A"
        numbering.loc[numbering["sequence_index"] == 27, "anarci_label"] = "27A"
        numbering.loc[numbering["sequence_index"] == 27, "anarci_sort_key"] = 27.01
        locations, opportunities, skipped, numbering_status = localize_mutations(base_table(designed, native), numbering)
        self.assertTrue(skipped.empty)
        self.assertEqual(locations.iloc[0]["canonical_position"], "27A")
        self.assertEqual(locations.iloc[0]["region"], "CDR1")
        self.assertEqual(numbering_status.iloc[0]["numbering_status"], "ok")

    def test_plotting_and_csv_generation_with_synthetic_data(self) -> None:
        native = "A" * 117
        designed = design_sequence(native, {27: "D", 105: "F"})
        locations, summary, frequency, enrichment, skipped, numbering_status = compute_mutation_localization_outputs(base_table(designed, native), numbering_fixture())
        root = Path(os.environ.get("JANGO_TEST_TMP", tempfile.gettempdir()))
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"mutation_localization_{uuid4().hex}"
        bar = root / f"{prefix}_mutation_region_bar.png"
        freq = root / f"{prefix}_mutation_position_frequency.png"
        loc_csv = root / f"{prefix}_mutation_locations.csv"
        summary_csv = root / f"{prefix}_mutation_region_summary.csv"
        save_mutation_region_bar(summary, bar, dpi=90)
        save_mutation_position_frequency_plot(frequency, freq, dpi=90)
        locations.to_csv(loc_csv, index=False)
        summary.to_csv(summary_csv, index=False)
        enrichment.to_csv(root / f"{prefix}_mutation_region_enrichment.csv", index=False)
        skipped.to_csv(root / f"{prefix}_mutation_localization_skipped_decoys.csv", index=False)
        numbering_status.to_csv(root / f"{prefix}_mutation_numbering_status.csv", index=False)
        self.assertGreater(bar.stat().st_size, 0)
        self.assertGreater(freq.stat().st_size, 0)
        self.assertIn("canonical_position", pd.read_csv(loc_csv).columns)
        self.assertIn("cdr3_mutations", pd.read_csv(summary_csv).columns)


if __name__ == "__main__":
    unittest.main()

