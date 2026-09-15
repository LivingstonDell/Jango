from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from jango.pipeline.interface_analyzer import (
    build_reference_delta,
    detect_pdb_column,
    ensure_outputs_outside_repo,
    manifest_rows,
)


class InterfaceAnalyzerCliTests(unittest.TestCase):
    def test_detect_pdb_column_prefers_common_jango_columns(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "structure_id": "s1",
                    "relaxed_pdb": "/tmp/s1.pdb",
                    "scored_pdb": "/tmp/other.pdb",
                }
            ]
        )
        self.assertEqual("relaxed_pdb", detect_pdb_column(table))
        self.assertEqual("scored_pdb", detect_pdb_column(table, "scored_pdb"))

    def test_manifest_rows_resolves_ids_paths_and_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "structure_id": "10zo_H_A",
                        "design_id": "10zo_H_A_mpnn_0003",
                        "relaxed_pdb": "/data/example/10zo.pdb",
                        "nanobody_chain": "H",
                        "antigen_chain": "A",
                    }
                ]
            ).to_csv(manifest, index=False)

            rows = manifest_rows(
                manifest,
                pdb_column=None,
                state="decoy_relaxed",
                interface=None,
                nanobody_chain=None,
                antigen_chain=None,
                limit=None,
            )

        self.assertEqual(1, len(rows))
        self.assertEqual("10zo_H_A_mpnn_0003", rows[0]["design_id"])
        self.assertEqual("10zo_H_A", rows[0]["structure_id"])
        self.assertEqual("H_A", rows[0]["interface"])
        self.assertEqual("relaxed_pdb", rows[0]["source_pdb_column"])

    def test_manifest_rows_allows_interface_override_without_chain_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            pd.DataFrame([{"structure_id": "s1", "pdb_path": "/tmp/s1.pdb"}]).to_csv(manifest, index=False)

            rows = manifest_rows(
                manifest,
                pdb_column=None,
                state="custom",
                interface="N_A",
                nanobody_chain=None,
                antigen_chain=None,
                limit=None,
            )

        self.assertEqual("N_A", rows[0]["interface"])
        self.assertEqual("N", rows[0]["nanobody_chain"])
        self.assertEqual("A", rows[0]["antigen_chain"])

    def test_manifest_rows_treats_nan_chain_cells_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            pd.DataFrame([{"structure_id": "s1", "pdb_path": "/tmp/s1.pdb", "nanobody_chain": pd.NA}]).to_csv(
                manifest,
                index=False,
            )

            rows = manifest_rows(
                manifest,
                pdb_column=None,
                state="custom",
                interface=None,
                nanobody_chain="N",
                antigen_chain="A",
                limit=None,
            )

        self.assertEqual("N_A", rows[0]["interface"])
        self.assertEqual("N", rows[0]["nanobody_chain"])
        self.assertEqual("A", rows[0]["antigen_chain"])

    def test_build_reference_delta_adds_binding_perturbation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.csv"
            pd.DataFrame(
                [
                    {
                        "structure_id": "s1",
                        "dG_separated": -20.0,
                        "dSASA_int": 1000.0,
                        "sc_value": 0.6,
                    }
                ]
            ).to_csv(reference, index=False)
            scored = pd.DataFrame(
                [
                    {
                        "structure_id": "s1",
                        "design_id": "d1",
                        "dG_separated": -12.5,
                        "dSASA_int": 950.0,
                        "sc_value": 0.5,
                    }
                ]
            )

            delta = build_reference_delta(scored, reference)

        self.assertEqual(7.5, float(delta.loc[0, "delta_dG_separated"]))
        self.assertEqual(-50.0, float(delta.loc[0, "delta_dSASA_int"]))
        self.assertEqual(7.5, float(delta.loc[0, "binding_perturbation"]))

    def test_output_guard_blocks_source_repo_paths(self) -> None:
        repo_path = Path(__file__).resolve().parents[2] / "results" / "bad.csv"
        with self.assertRaisesRegex(ValueError, "Refusing to write InterfaceAnalyzer outputs"):
            ensure_outputs_outside_repo([repo_path], allow_repo_output=False)

        ensure_outputs_outside_repo([repo_path], allow_repo_output=True)


if __name__ == "__main__":
    unittest.main()
