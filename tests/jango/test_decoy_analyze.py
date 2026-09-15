from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from jango.pipeline.decoy_analyze import ensure_dockq_table_has_successes, grafted_decoy_count, write_zero_survivor_landscape_outputs


class DecoyAnalyzeTests(unittest.TestCase):
    def test_grafted_decoy_count_handles_missing_empty_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp) / "tables"
            self.assertEqual(grafted_decoy_count(tables), 0)
            tables.mkdir(parents=True)
            pd.DataFrame(columns=["design_id"]).to_csv(tables / "decoy_grafted_complex_manifest.csv", index=False)
            self.assertEqual(grafted_decoy_count(tables), 0)
            pd.DataFrame([{"design_id": "d1"}]).to_csv(tables / "decoy_grafted_complex_manifest.csv", index=False)
            self.assertEqual(grafted_decoy_count(tables), 1)

    def test_dockq_success_guard_rejects_all_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decoy_dockq.csv"
            pd.DataFrame(columns=["design_id", "dockq_status", "dockq"]).to_csv(path, index=False)
            with self.assertRaises(SystemExit) as raised:
                ensure_dockq_table_has_successes(path, "mode DockQ table")
            self.assertIn("has no DockQ rows", str(raised.exception))

            pd.DataFrame([{"design_id": "d1", "dockq_status": "missing_native_relaxed_pdb", "dockq": None}]).to_csv(path, index=False)

            with self.assertRaises(SystemExit) as raised:
                ensure_dockq_table_has_successes(path, "mode DockQ table")

            self.assertIn("no successful numeric DockQ rows", str(raised.exception))

            pd.DataFrame([{"design_id": "d1", "dockq_status": "ok", "dockq": 0.62}]).to_csv(path, index=False)
            self.assertEqual(1, ensure_dockq_table_has_successes(path, "mode DockQ table"))

    def test_zero_survivor_landscape_outputs_are_header_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = root / "tables"
            figures = root / "figures"
            write_zero_survivor_landscape_outputs(
                tables_dir=tables,
                figures_dir=figures,
                mode="full_antigen",
                experiment="exp",
                structure_threshold=0.49,
            )
            landscape = tables / "landscape"
            ranking = pd.read_csv(landscape / "decoy_quadrant_ranking.csv")
            summary = pd.read_csv(landscape / "decoy_quadrant_summary.csv")
            mutations = pd.read_csv(landscape / "mutation_locations.csv")

            self.assertIn("coordinate_score", ranking.columns)
            self.assertTrue(ranking.empty)
            self.assertEqual(["Q1", "Q4", "Q2", "Q3"], summary["quadrant"].tolist())
            self.assertEqual(0, int(summary["n_decoys"].sum()))
            self.assertIn("canonical_position", mutations.columns)


if __name__ == "__main__":
    unittest.main()
