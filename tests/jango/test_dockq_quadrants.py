import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from jango.analysis.experiment import (
    QUADRANT_ORDER,
    build_quadrant_tables,
    classify_delphi_quadrant,
    save_landscape_plot,
)
from jango.analysis.thresholds import parse_dockq_preservation_threshold


class DelphiQuadrantTests(unittest.TestCase):
    def test_exact_threshold_behavior(self):
        quadrant, rank, _ = classify_delphi_quadrant(0.49, 0.0, 0.49, 0.0)
        self.assertEqual(quadrant, "Q1")
        self.assertEqual(rank, 1.0)

    def test_ranking_is_q1_only_and_ordered_by_coordinate_score(self):
        scored = pd.DataFrame(
            {
                "structure_id": ["s3", "s1", "s4", "s2", "s5"],
                "design_id": ["d3", "d1", "d4", "d2", "d5"],
                "structure_preservation": [0.20, 0.60, 0.90, 0.30, 0.75],
                "binding_perturbation": [-1.0, 2.0, -0.5, 1.5, 0.1],
                "structure_preservation_metric": ["dockq"] * 5,
                "dockq": [0.20, 0.60, 0.90, 0.30, 0.75],
                "dockq_quality": ["incorrect", "medium", "high", "acceptable", "medium"],
            }
        )
        ranking, summary, unclassified = build_quadrant_tables(
            scored,
            pd.DataFrame(),
            structure_threshold=0.49,
            binding_threshold=0.0,
        )
        self.assertEqual(["Q1", "Q1"], ranking["quadrant"].tolist())
        self.assertEqual(["d1", "d5"], ranking["design_id"].tolist())
        self.assertEqual([2.6, 0.85], ranking["coordinate_score"].round(6).tolist())
        self.assertEqual(ranking["coordinate_score"].tolist(), ranking["delphi_score"].tolist())
        self.assertEqual(QUADRANT_ORDER, summary["quadrant"].tolist())
        self.assertTrue(unclassified.empty)
        self.assertEqual([1, 2, 3, 4], summary["quadrant_rank"].tolist())
        self.assertEqual([2, 1, 1, 1], summary["decoy_count"].tolist())

    def test_unclassified_rows_are_separate(self):
        scored = pd.DataFrame(
            {
                "structure_id": ["s1"],
                "design_id": ["d1"],
                "structure_preservation": [0.7],
                "binding_perturbation": [1.0],
                "structure_preservation_metric": ["dockq"],
                "dockq": [0.7],
            }
        )
        unclassified_source = pd.DataFrame(
            {
                "structure_id": ["s2"],
                "design_id": ["d2"],
                "dockq": [np.nan],
                "landscape_unclassified_reason": ["dockq"],
            }
        )
        _, _, unclassified = build_quadrant_tables(scored, unclassified_source, structure_threshold=0.49, binding_threshold=0.0)
        self.assertEqual("unclassified", unclassified.loc[0, "quadrant"])
        self.assertTrue(pd.isna(unclassified.loc[0, "quadrant_rank"]))
        self.assertEqual("dockq", unclassified.loc[0, "quadrant_interpretation"])

    def test_configured_structure_threshold_reassigns_quadrants(self):
        scored = pd.DataFrame(
            {
                "structure_id": ["s1"],
                "design_id": ["d1"],
                "structure_preservation": [0.60],
                "binding_perturbation": [1.0],
                "structure_preservation_metric": ["dockq"],
                "dockq": [0.60],
            }
        )
        default_ranking, _, _ = build_quadrant_tables(scored, pd.DataFrame(), structure_threshold=0.49, binding_threshold=0.0)
        stricter_ranking, stricter_summary, _ = build_quadrant_tables(scored, pd.DataFrame(), structure_threshold=0.70, binding_threshold=0.0)

        self.assertEqual("Q1", default_ranking.loc[0, "quadrant"])
        self.assertTrue(stricter_ranking.empty)
        q2 = stricter_summary[stricter_summary["quadrant"] == "Q2"].iloc[0]
        self.assertEqual(1, q2["decoy_count"])
        self.assertEqual(0.70, q2["structure_preservation_threshold"])

    def test_empty_ranked_quadrants_keep_summary_and_unclassified_rows(self):
        scored = pd.DataFrame(columns=["structure_id", "design_id", "structure_preservation", "binding_perturbation"])
        unclassified_source = pd.DataFrame(
            {
                "structure_id": ["s1"],
                "design_id": ["d1"],
                "landscape_unclassified_reason": ["dockq"],
            }
        )

        ranking, summary, unclassified = build_quadrant_tables(scored, unclassified_source, structure_threshold=0.49, binding_threshold=0.0)

        self.assertTrue(ranking.empty)
        self.assertEqual(QUADRANT_ORDER, summary["quadrant"].tolist())
        self.assertEqual([0, 0, 0, 0], summary["decoy_count"].tolist())
        self.assertEqual(1, len(unclassified))
        self.assertEqual("unclassified", unclassified.loc[0, "quadrant"])

    def test_landscape_plot_does_not_write_quadrant_text(self):
        landscape = pd.DataFrame(
            {
                "structure_preservation": [0.6, 0.8],
                "binding_perturbation": [1.2, -0.2],
                "redesign_mode": ["full_antigen", "hotspot_only"],
                "structure_preservation_metric": ["dockq", "dockq"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "landscape.png"
            with mock.patch("matplotlib.axes.Axes.text") as text_call:
                save_landscape_plot(landscape, path, dpi=80, structure_threshold=0.49, binding_threshold=0.0)
            self.assertTrue(path.exists())
            text_call.assert_not_called()

    def test_threshold_parser_accepts_boundaries(self):
        self.assertEqual(0.0, parse_dockq_preservation_threshold("0"))
        self.assertEqual(1.0, parse_dockq_preservation_threshold("1"))

    def test_threshold_parser_rejects_invalid_values(self):
        for value in ["-0.01", "1.01", "nan", "not-a-number"]:
            with self.subTest(value=value):
                with self.assertRaises(Exception):
                    parse_dockq_preservation_threshold(value)


if __name__ == "__main__":
    unittest.main()
