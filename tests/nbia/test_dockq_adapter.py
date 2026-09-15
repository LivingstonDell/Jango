import os
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import pandas as pd

from nbia.quality.dockq import (
    DockQError,
    build_dockq_command,
    classify_dockq_quality,
    parse_dockq_json,
    run_dockq,
    run_dockq_rs_python,
    score_decoy_manifest,
    validate_chain_mapping,
)


TEST_TMP_ROOT = Path(os.environ.get("JANGO_TEST_TMP", Path(tempfile.gettempdir()) / "jango_test_tmp" / "dockq_adapter"))


def workspace_case_root(name: str) -> Path:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def pdb_text(nb_chain: str = "N", ag_chain: str = "A") -> str:
    return "\n".join(
        [
            f"ATOM      1  CA  ALA {nb_chain}   1       0.000   0.000   0.000  1.00 20.00           C",
            f"ATOM      2  CA  GLY {ag_chain}   1       4.000   0.000   0.000  1.00 20.00           C",
            "END",
            "",
        ]
    )


class DockQAdapterTests(unittest.TestCase):
    def test_quality_boundaries(self):
        self.assertEqual(classify_dockq_quality(0.229), "incorrect")
        self.assertEqual(classify_dockq_quality(0.23), "acceptable")
        self.assertEqual(classify_dockq_quality(0.49), "medium")
        self.assertEqual(classify_dockq_quality(0.80), "high")

    def test_command_uses_explicit_na_mapping(self):
        command = build_dockq_command(Path("model.pdb"), Path("native.pdb"), Path("out.json"))
        self.assertEqual(command[0], "DockQ")
        self.assertIn("--mapping", command)
        self.assertEqual(command[command.index("--mapping") + 1], "NA:NA")
        validate_chain_mapping("NA:NA")
        with self.assertRaises(DockQError):
            validate_chain_mapping("HA:NA")

    def test_parse_official_json_fields(self):
        root = workspace_case_root("dockq_json")
        path = root / "dockq.json"
        path.write_text(json.dumps({
            "best_dockq": 0.62,
            "GlobalDockQ": 0.61,
            "best_mapping_str": "NA:NA",
            "best_result": {
                "NA": {
                    "fnat": 0.7,
                    "fnonnat": 0.1,
                    "F1": 0.8,
                    "iRMSD": 1.2,
                    "LRMSD": 2.3,
                    "clashes": 0,
                    "nat_correct": 7,
                    "nat_total": 10,
                    "model_total": 11,
                }
            },
        }))
        parsed = parse_dockq_json(path)
        self.assertEqual(parsed["dockq"], 0.62)
        self.assertEqual(parsed["dockq_quality"], "medium")
        self.assertEqual(parsed["dockq_best_mapping"], "NA:NA")
        self.assertEqual(parsed["dockq_nat_total"], 10.0)

    def test_run_dockq_skip_existing_does_not_call_subprocess(self):
        root = workspace_case_root("dockq_skip")
        json_path = root / "result.json"
        json_path.write_text(json.dumps({"best_dockq": 0.81, "best_result": {"NA": {}}}))
        with patch("nbia.quality.dockq.subprocess.run") as run:
            result = run_dockq(Path("DockQ"), root / "model.pdb", root / "native.pdb", json_path, skip_existing=True)
        run.assert_not_called()
        self.assertEqual(result["dockq_status"], "ok")
        self.assertEqual(result["dockq_quality"], "high")

    def test_run_dockq_failure_records_error(self):
        root = workspace_case_root("dockq_failed")
        with patch("nbia.quality.dockq.subprocess.run", return_value=SimpleNamespace(returncode=2, stdout="", stderr="bad chains")):
            result = run_dockq(Path("DockQ"), root / "model.pdb", root / "native.pdb", root / "result.json")
        self.assertEqual(result["dockq_status"], "failed:2")
        self.assertIn("bad chains", result["dockq_error"])

    def test_run_dockq_rs_python_writes_and_parses_json(self):
        root = workspace_case_root("dockq_rs")
        json_path = root / "result.json"

        def fake_run(argv, **kwargs):
            self.assertEqual(argv[0], "/opt/dockq-rs/bin/python")
            self.assertEqual(argv[1], "-")
            Path(argv[5]).write_text(json.dumps({
                "best_dockq": 0.71,
                "GlobalDockQ": 0.70,
                "best_mapping_str": "NA:NA",
                "best_result": {"NA": {"fnat": 0.5, "iRMSD": 1.0, "LRMSD": 2.0}},
            }))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("nbia.quality.dockq.subprocess.run", side_effect=fake_run):
            result = run_dockq_rs_python(Path("/opt/dockq-rs/bin/python"), root / "model.pdb", root / "native.pdb", json_path)
        self.assertEqual(result["dockq_status"], "ok")
        self.assertEqual(result["dockq"], 0.71)
        self.assertEqual(result["dockq_quality"], "medium")

    def test_score_manifest_uses_dockq_rs_python_provider(self):
        root = workspace_case_root("dockq_manifest_rs")
        decoy_pdb = root / "decoy.pdb"
        native_pdb = root / "native_relaxed.pdb"
        dockq_python = root / "dockq_rs_python"
        decoy_pdb.write_text(pdb_text("N", "A"))
        native_pdb.write_text(pdb_text("H", "A"))
        dockq_python.write_text("#!/usr/bin/env python\n")
        decoy_manifest = root / "decoy_relax.csv"
        native_manifest = root / "native_relax.csv"
        case_manifest = root / "cases.csv"
        pd.DataFrame([{"design_id": "s1_mpnn_0001", "structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(decoy_pdb)}]).to_csv(decoy_manifest, index=False)
        pd.DataFrame([{"structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(native_pdb)}]).to_csv(native_manifest, index=False)
        pd.DataFrame([{"structure_id": "s1", "nanobody_chain": "H", "antigen_chain": "A"}]).to_csv(case_manifest, index=False)

        def fake_run(argv, **kwargs):
            if argv[1] == "-c":
                return SimpleNamespace(returncode=0, stdout="0.1.0\n", stderr="")
            Path(argv[5]).write_text(json.dumps({
                "best_dockq": 0.82,
                "GlobalDockQ": 0.81,
                "best_mapping_str": "NA:NA",
                "best_result": {"NA": {"fnat": 0.75, "iRMSD": 0.8, "LRMSD": 1.8}},
            }))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("nbia.quality.dockq.subprocess.run", side_effect=fake_run):
            out = score_decoy_manifest(
                decoy_relax_manifest=decoy_manifest,
                native_relax_manifest=native_manifest,
                case_manifest=case_manifest,
                dockq_provider="dockq_rs_python",
                dockq_python=dockq_python,
                work_dir=root / "dockq_work",
                out_csv=root / "dockq.csv",
            )
        self.assertEqual(out.loc[0, "dockq_status"], "ok")
        self.assertEqual(out.loc[0, "dockq_provider"], "dockq_rs_python")
        self.assertEqual(out.loc[0, "dockq_quality"], "high")

    def test_score_manifest_uses_already_canonical_native_reference(self):
        root = workspace_case_root("dockq_manifest_canonical_native")
        decoy_pdb = root / "decoy.pdb"
        native_pdb = root / "native_relaxed.pdb"
        dockq_python = root / "dockq_rs_python"
        decoy_pdb.write_text(pdb_text("N", "A"))
        native_pdb.write_text(pdb_text("N", "A"))
        dockq_python.write_text("#!/usr/bin/env python\n")
        decoy_manifest = root / "decoy_relax.csv"
        native_manifest = root / "native_relax.csv"
        case_manifest = root / "cases.csv"
        pd.DataFrame([{"design_id": "s1_mpnn_0001", "structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(decoy_pdb)}]).to_csv(decoy_manifest, index=False)
        pd.DataFrame([{"structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(native_pdb)}]).to_csv(native_manifest, index=False)
        pd.DataFrame([{"structure_id": "s1", "nanobody_chain": "H", "antigen_chain": "A"}]).to_csv(case_manifest, index=False)

        def fake_run(argv, **kwargs):
            if argv[1] == "-c":
                return SimpleNamespace(returncode=0, stdout="0.1.0\n", stderr="")
            self.assertEqual(Path(argv[3]), native_pdb)
            Path(argv[5]).write_text(json.dumps({
                "best_dockq": 0.82,
                "GlobalDockQ": 0.81,
                "best_mapping_str": "NA:NA",
                "best_result": {"NA": {"fnat": 0.75, "iRMSD": 0.8, "LRMSD": 1.8}},
            }))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("nbia.quality.dockq.subprocess.run", side_effect=fake_run):
            out = score_decoy_manifest(
                decoy_relax_manifest=decoy_manifest,
                native_relax_manifest=native_manifest,
                case_manifest=case_manifest,
                dockq_provider="dockq_rs_python",
                dockq_python=dockq_python,
                work_dir=root / "dockq_work",
                out_csv=root / "dockq.csv",
            )
        self.assertEqual(Path(out.loc[0, "native_relaxed_remapped_pdb"]), native_pdb)

    def test_score_manifest_rejects_missing_native_relaxed_path_before_scoring(self):
        root = workspace_case_root("dockq_missing_native_path")
        decoy_pdb = root / "decoy.pdb"
        decoy_pdb.write_text(pdb_text("N", "A"))
        decoy_manifest = root / "decoy_relax.csv"
        native_manifest = root / "native_relax.csv"
        case_manifest = root / "cases.csv"
        pd.DataFrame([{"design_id": "s1_mpnn_0001", "structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(decoy_pdb)}]).to_csv(decoy_manifest, index=False)
        pd.DataFrame([{"structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(root / "missing_native.pdb")}]).to_csv(native_manifest, index=False)
        pd.DataFrame([{"structure_id": "s1", "nanobody_chain": "H", "antigen_chain": "A"}]).to_csv(case_manifest, index=False)

        with self.assertRaisesRegex(DockQError, "native relax manifest has 1/1 required relaxed_pdb path"):
            score_decoy_manifest(
                decoy_relax_manifest=decoy_manifest,
                native_relax_manifest=native_manifest,
                case_manifest=case_manifest,
                dockq_bin=root / "missing_DockQ",
                work_dir=root / "dockq_work",
                out_csv=root / "dockq.csv",
                allow_failures=True,
            )
        self.assertFalse((root / "dockq.csv").exists())

    def test_score_manifest_records_missing_tool_and_remaps_native(self):
        root = workspace_case_root("dockq_manifest")
        decoy_pdb = root / "decoy.pdb"
        native_pdb = root / "native_relaxed.pdb"
        decoy_pdb.write_text(pdb_text("N", "A"))
        native_pdb.write_text(pdb_text("H", "A"))
        decoy_manifest = root / "decoy_relax.csv"
        native_manifest = root / "native_relax.csv"
        case_manifest = root / "cases.csv"
        pd.DataFrame([{"design_id": "s1_mpnn_0001", "structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(decoy_pdb)}]).to_csv(decoy_manifest, index=False)
        pd.DataFrame([{"structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(native_pdb)}]).to_csv(native_manifest, index=False)
        pd.DataFrame([{"structure_id": "s1", "nanobody_chain": "H", "antigen_chain": "A"}]).to_csv(case_manifest, index=False)
        with self.assertRaisesRegex(DockQError, "All 1 DockQ row"):
            score_decoy_manifest(
                decoy_relax_manifest=decoy_manifest,
                native_relax_manifest=native_manifest,
                case_manifest=case_manifest,
                dockq_bin=root / "missing_DockQ",
                work_dir=root / "dockq_work",
                out_csv=root / "dockq.csv",
                allow_failures=True,
            )
        out = pd.read_csv(root / "dockq.csv")
        self.assertEqual(out.loc[0, "dockq_status"], "missing_dockq_executable")
        self.assertEqual(out.loc[0, "dockq_provider"], "dockq_cli")
        self.assertEqual(out.loc[0, "dockq_reference_state"], "native_relaxed")
        self.assertTrue(Path(out.loc[0, "native_relaxed_remapped_pdb"]).exists())
        with self.assertRaises(DockQError):
            score_decoy_manifest(
                decoy_relax_manifest=decoy_manifest,
                native_relax_manifest=native_manifest,
                case_manifest=case_manifest,
                dockq_bin=root / "missing_DockQ",
                work_dir=root / "dockq_work2",
                out_csv=root / "dockq2.csv",
                allow_failures=False,
            )


if __name__ == "__main__":
    unittest.main()
