from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nbia.rosetta_qc import (
    QC_FAILED,
    QC_INCOMPLETE,
    QC_PARSE_FAILED,
    QC_SKIPPED_EXISTING_VALID,
    build_decoy_rosetta_qc,
    build_native_rosetta_qc,
)


TEST_TMP_ROOT = Path(os.environ.get("JANGO_TEST_TMP", Path(tempfile.gettempdir()) / "jango_test_tmp")) / "rosetta_qc"


def make_root(name: str) -> Path:
    root = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def scorefile(path: Path, *, parseable: bool = True, dg: float = -1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if parseable:
        path.write_text(
            "SCORE: dG_separated dSASA_int sc_value hbonds_int delta_unsatHbonds packstat description\n"
            f"SCORE: {dg:.3f} 100.000 0.500 2.000 1.000 0.100 model\n"
        )
    else:
        path.write_text("SCORE: description\n")
    return path


def nonempty(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub\n")
    return path


def native_tables(root: Path, *, duplicate: bool = False, parseable: bool = True, include_scorefile: bool = True):
    relaxed = nonempty(root / "work/native/rosetta/relaxed/s1_interface_relax_0001.pdb")
    score = scorefile(root / "work/native/rosetta/interface_analyzer/relaxed/s1.sc", parseable=parseable) if include_scorefile else root / "missing.sc"
    log = nonempty(root / "work/native/rosetta/interface_analyzer/relaxed/s1.log")
    relax = pd.DataFrame([{"structure_id": "s1", "relax_status": "skipped_existing", "relaxed_pdb": str(relaxed)}])
    interface_rows = [
        {
            "structure_id": "s1",
            "interface_analyzer_status": "skipped_existing",
            "regular_score_status": "not_backfilled",
            "dG_separated": -1.0,
            "dSASA_int": 100.0,
            "sc_value": 0.5,
            "hbonds_int": 2.0,
            "delta_unsatHbonds": 1.0,
            "packstat": 0.1,
        }
    ]
    if duplicate:
        interface_rows.append(interface_rows[0].copy())
    interface = pd.DataFrame(interface_rows)
    relax_path = root / "native_relax.csv"
    interface_path = root / "native_interface.csv"
    relax.to_csv(relax_path, index=False)
    interface.to_csv(interface_path, index=False)
    return relax_path, interface_path, root / "work/native/rosetta", score, log


def decoy_tables(root: Path, *, parseable: bool = True, include_scorefile: bool = True, duplicate: bool = False):
    relaxed = nonempty(root / "work/decoy/relaxed/d1_interface_relax_0001.pdb")
    score = scorefile(root / "work/decoy/rosetta/s1/d1.interface.sc", parseable=parseable) if include_scorefile else root / "missing.sc"
    log = nonempty(root / "work/decoy/rosetta/s1/d1.interface.log")
    relax_rows = [
        {"design_id": "d1", "structure_id": "s1", "relax_status": "ok", "relaxed_pdb": str(relaxed), "redesign_mode": "hotspot_only"}
    ]
    interface_rows = [
        {
            "design_id": "d1",
            "structure_id": "s1",
            "rosetta_status": "ok",
            "scored_pdb": str(relaxed),
            "interface_scorefile": str(score),
            "interface_log": str(log),
            "dG_separated": -1.0,
            "dSASA_int": 100.0,
            "sc_value": 0.5,
            "hbonds_int": 2.0,
            "delta_unsatHbonds": 1.0,
            "packstat": 0.1,
        }
    ]
    if duplicate:
        interface_rows.append(interface_rows[0].copy())
    slurm = pd.DataFrame([
        {"design_id": "d1", "submission_status": "pending", "completion_status": "COMPLETED", "slurm_job_id": "123", "exit_code": 0}
    ])
    relax_path = root / "decoy_relax.csv"
    interface_path = root / "decoy_interface.csv"
    slurm_path = root / "slurm.csv"
    pd.DataFrame(relax_rows).to_csv(relax_path, index=False)
    pd.DataFrame(interface_rows).to_csv(interface_path, index=False)
    slurm.to_csv(slurm_path, index=False)
    return relax_path, interface_path, slurm_path


class RosettaQcTests(unittest.TestCase):
    def test_native_stale_existing_status_recovers_to_valid(self):
        root = make_root("native_valid")
        relax, interface, work, _, _ = native_tables(root)
        qc = build_native_rosetta_qc(native_relax_manifest=relax, native_interface_table=interface, native_work_dir=work)
        self.assertEqual(qc.iloc[0]["qc_status"], QC_SKIPPED_EXISTING_VALID)
        self.assertTrue(qc.iloc[0]["required_terms_present"])
        self.assertTrue(qc.iloc[0]["metrics_agree_with_table"])

    def test_decoy_stale_slurm_pending_is_reconciled_from_outputs(self):
        root = make_root("decoy_valid")
        native_relax, native_interface, native_work, _, _ = native_tables(root)
        native_qc = build_native_rosetta_qc(native_relax_manifest=native_relax, native_interface_table=native_interface, native_work_dir=native_work)
        relax, interface, slurm = decoy_tables(root)
        qc = build_decoy_rosetta_qc(decoy_relax_manifest=relax, decoy_interface_table=interface, native_qc_table=native_qc, slurm_submission_manifest=slurm)
        self.assertEqual(qc.iloc[0]["qc_status"], QC_SKIPPED_EXISTING_VALID)
        self.assertEqual(qc.iloc[0]["reconciled_submission_status"], "completed_from_stale_pending")

    def test_missing_scorefile_is_incomplete(self):
        root = make_root("missing_score")
        relax, interface, work, _, _ = native_tables(root, include_scorefile=False)
        qc = build_native_rosetta_qc(native_relax_manifest=relax, native_interface_table=interface, native_work_dir=None)
        self.assertEqual(qc.iloc[0]["qc_status"], QC_INCOMPLETE)
        self.assertFalse(qc.iloc[0]["interface_scorefile_exists"])

    def test_parse_failure_is_explicit(self):
        root = make_root("parse_failed")
        relax, interface, work, _, _ = native_tables(root, parseable=False)
        qc = build_native_rosetta_qc(native_relax_manifest=relax, native_interface_table=interface, native_work_dir=work)
        self.assertEqual(qc.iloc[0]["qc_status"], QC_PARSE_FAILED)
        self.assertFalse(qc.iloc[0]["required_terms_present"])

    def test_duplicate_decoy_rows_fail_qc(self):
        root = make_root("duplicate_decoy")
        native_relax, native_interface, native_work, _, _ = native_tables(root)
        native_qc = build_native_rosetta_qc(native_relax_manifest=native_relax, native_interface_table=native_interface, native_work_dir=native_work)
        relax, interface, slurm = decoy_tables(root, duplicate=True)
        qc = build_decoy_rosetta_qc(decoy_relax_manifest=relax, decoy_interface_table=interface, native_qc_table=native_qc, slurm_submission_manifest=slurm)
        self.assertEqual(qc.iloc[0]["qc_status"], QC_FAILED)
        self.assertTrue(qc.iloc[0]["duplicate_design_rows"])


if __name__ == "__main__":
    unittest.main()
