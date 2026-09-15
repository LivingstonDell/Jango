import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from jango.pipeline.fold_qc import build_fold_qc_table, main


AA3 = {"A": "ALA", "G": "GLY"}


def write_pdb(path: Path, sequence: str, chain: str = "A") -> None:
    lines = []
    serial = 1
    for idx, aa in enumerate(sequence, start=1):
        resname = AA3[aa]
        lines.append(
            f"ATOM  {serial:5d}  CA  {resname:>3} {chain}{idx:4d}    "
            f"{float(idx):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_fold_manifest(fold_dir: Path, validations: dict[str, Path], work_root: Path | None = None) -> dict[str, Path]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    mode_outputs = {}
    work_dirs: dict[str, Path] = {}
    for mode, validation in validations.items():
        work_dir = (work_root or fold_dir.parent / "work" / "decoy" / "exp1") / mode
        work_dir.mkdir(parents=True, exist_ok=True)
        work_dirs[mode] = work_dir
        mode_outputs[mode] = {"validation": str(validation), "work_dir": str(work_dir)}
    (fold_dir / "fold_manifest.json").write_text(json.dumps({"experiment": "exp1", "backend": "mockfold", "mode_outputs": mode_outputs}))
    return work_dirs


def write_submission_manifest(work_root: Path, rows: list[dict[str, object]]) -> Path:
    path = work_root / "slurm" / "fold_submission_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class FoldQCTests(unittest.TestCase):
    def test_build_fold_qc_keeps_all_statuses_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            fold_dir.mkdir()
            tables = root / "tables" / "mode_a"
            tables.mkdir(parents=True)
            native = root / "native.pdb"
            passed = root / "passed.pdb"
            failed = root / "failed.pdb"
            write_pdb(native, "AG")
            write_pdb(passed, "AG")
            write_pdb(failed, "AG")
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "s1_native",
                        "fold_backend": "mockfold",
                        "sequence_role": "native_control",
                        "is_native_reference": True,
                        "validation_status": "native_reference",
                        "selected_decoy": False,
                        "prediction_path": str(native),
                        "sequence": "AG",
                        "sequence_identity": 1.0,
                        "msa_source": "native_cache",
                        "msa_status": "msa_cache_hit",
                        "msa_consumed": True,
                        "msa_cache_hit": True,
                        "esmfold2_mean_plddt": 91.5,
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "s1_design_ok",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(passed),
                        "sequence": "AG",
                        "sequence_identity": 1.0,
                        "msa_source": "native_reuse",
                        "msa_status": "msa_generated",
                        "msa_consumed": True,
                        "msa_cache_hit": False,
                        "monomer_tm_score": 0.9,
                        "monomer_aligned_fraction": 1.0,
                        "esmfold2_mean_plddt": 88.0,
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "s1_design_failed",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "failed_fold",
                        "selected_decoy": False,
                        "prediction_path": str(failed),
                        "sequence": "AG",
                        "sequence_identity": 1.0,
                        "msa_source": "native_reuse",
                        "msa_consumed": True,
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "s1_design_missing",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(root / "missing.pdb"),
                        "sequence": "AG",
                        "sequence_identity": 1.0,
                        "msa_source": "native_reuse",
                        "msa_consumed": True,
                    },
                ]
            ).to_csv(validation, index=False)
            (fold_dir / "fold_manifest.json").write_text(
                json.dumps(
                    {
                        "experiment": "exp1",
                        "backend": "mockfold",
                        "mode_outputs": {"mode_a": {"validation": str(validation)}},
                    }
                )
            )

            qc = build_fold_qc_table(fold_dir)

            self.assertEqual(4, len(qc))
            self.assertEqual({"native_control", "passed", "excluded", "incomplete"}, set(qc["final_qc_status"]))
            self.assertEqual(1, int(qc["graft_eligible"].sum()))
            ok = qc.loc[qc["design_id"] == "s1_design_ok"].iloc[0]
            self.assertEqual("completed", ok["fold_completion_status"])
            self.assertEqual("exact", ok["sequence_match_status"])
            self.assertEqual(2, int(ok["residue_count"]))
            self.assertEqual("A", ok["chain_ids"])
            self.assertIn("esmfold2_mean_plddt", ok["confidence_metrics_available"])

    def test_main_writes_default_qc_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            fold_dir.mkdir()
            tables = root / "input_tables"
            tables.mkdir()
            pdb = root / "fold.pdb"
            write_pdb(pdb, "AG")
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(pdb),
                        "sequence": "AG",
                        "designed_sequence": "AG",
                    }
                ]
            ).to_csv(validation, index=False)
            (fold_dir / "fold_manifest.json").write_text(
                json.dumps({"experiment": "exp1", "backend": "mockfold", "mode_outputs": {"mode_a": {"validation": str(validation)}}})
            )

            rc = main(["--fold-dir", str(fold_dir), "--output-root", str(root), "--skip-figure"])

            self.assertEqual(0, rc)
            out = root / "results" / "decoy" / "tables" / "exp1" / "all_modes" / "post_fold_qc.csv"
            self.assertTrue(out.exists())
            self.assertEqual(1, len(pd.read_csv(out)))

    def test_discovers_design_id_pdb_and_single_fallback_pdb_from_fold_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            tables.mkdir(parents=True)
            expected_dir = root / "outputs" / "expected"
            fallback_dir = root / "outputs" / "fallback"
            opendde_dir = root / "outputs" / "opendde"
            expected_dir.mkdir(parents=True)
            fallback_dir.mkdir(parents=True)
            (opendde_dir / "predictions").mkdir(parents=True)
            expected_pdb = expected_dir / "d_expected.pdb"
            fallback_pdb = fallback_dir / "backend_name.pdb"
            opendde_pdb = opendde_dir / "predictions" / "d_opendde_sample_0.pdb"
            later_opendde_pdb = opendde_dir / "predictions" / "d_opendde_sample_1.pdb"
            write_pdb(expected_pdb, "AG")
            write_pdb(fallback_pdb, "AG")
            write_pdb(opendde_pdb, "AG")
            write_pdb(later_opendde_pdb, "AG")
            pd.DataFrame(
                [
                    {"prediction_path": str(later_opendde_pdb), "sample_rank": 1},
                    {"prediction_path": str(opendde_pdb), "sample_rank": 0},
                ]
            ).to_csv(opendde_dir / "opendde_prediction_manifest.csv", index=False)
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d_expected",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "fold_output_dir": str(expected_dir),
                        "sequence": "AG",
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s2",
                        "design_id": "d_fallback",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "fold_output_dir": str(fallback_dir),
                        "sequence": "AG",
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s3",
                        "design_id": "d_opendde",
                        "fold_backend": "opendde",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "fold_output_dir": str(opendde_dir),
                        "sequence": "AG",
                    },
                ]
            ).to_csv(validation, index=False)
            write_fold_manifest(fold_dir, {"mode_a": validation})

            qc = build_fold_qc_table(fold_dir)

            by_id = qc.set_index("design_id")
            self.assertEqual(str(expected_pdb), by_id.loc["d_expected", "structure_path"])
            self.assertEqual(str(fallback_pdb), by_id.loc["d_fallback", "structure_path"])
            self.assertEqual(str(opendde_pdb), by_id.loc["d_opendde", "structure_path"])
            self.assertEqual({"passed"}, set(qc["final_qc_status"]))

    def test_missing_and_empty_outputs_remain_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            tables.mkdir(parents=True)
            empty_pdb = root / "empty.pdb"
            empty_pdb.touch()
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {"redesign_mode": "mode_a", "structure_id": "s1", "design_id": "missing", "fold_backend": "mockfold", "sequence_role": "redesigned_decoy", "validation_status": "pass", "selected_decoy": True, "prediction_path": str(root / "missing.pdb"), "sequence": "AG"},
                    {"redesign_mode": "mode_a", "structure_id": "s2", "design_id": "empty", "fold_backend": "mockfold", "sequence_role": "redesigned_decoy", "validation_status": "pass", "selected_decoy": True, "prediction_path": str(empty_pdb), "sequence": "AG"},
                ]
            ).to_csv(validation, index=False)
            write_fold_manifest(fold_dir, {"mode_a": validation})

            qc = build_fold_qc_table(fold_dir)

            self.assertEqual({"incomplete"}, set(qc["final_qc_status"]))
            self.assertEqual({"missing", "empty"}, set(qc["fold_completion_status"]))
            self.assertFalse(qc["graft_eligible"].any())

    def test_submitted_exact_redesign_supersedes_stale_missing_backend_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            work_root = root / "work" / "decoy" / "exp1"
            tables.mkdir(parents=True)
            out_dir = root / "outputs" / "d1"
            out_dir.mkdir(parents=True)
            pdb = out_dir / "d1.pdb"
            write_pdb(pdb, "AG")
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "missing_mockfold_output",
                        "selected_decoy": False,
                        "fold_eligible": True,
                        "fold_output_dir": str(out_dir),
                        "sequence": "AG",
                        "designed_sequence": "AG",
                    }
                ]
            ).to_csv(validation, index=False)
            write_fold_manifest(fold_dir, {"mode_a": validation}, work_root=work_root)
            write_submission_manifest(
                work_root,
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "sequence_role": "redesigned_decoy",
                        "fold_output_dir": str(out_dir),
                        "submission_status": "submitted",
                        "completion_status": "COMPLETED",
                    }
                ],
            )

            qc = build_fold_qc_table(fold_dir)
            row = qc.iloc[0]

            self.assertEqual("passed", row["final_qc_status"])
            self.assertEqual("pass", row["validation_status"])
            self.assertEqual("missing_mockfold_output", row["source_validation_status"])
            self.assertFalse(bool(row["source_selected_decoy"]))
            self.assertTrue(bool(row["selected_for_folding"]))
            self.assertTrue(bool(row["selected_decoy"]))
            self.assertTrue(bool(row["graft_eligible"]))
            self.assertEqual(str(pdb), row["prediction_path"])
            self.assertEqual("AG", row["designed_sequence"])
            self.assertEqual("AG", row["sequence"])

    def test_stale_missing_output_backfills_monomer_metrics_from_native_refold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            work_root = root / "work" / "decoy" / "exp1"
            tables.mkdir(parents=True)
            native_dir = root / "outputs" / "s1_native"
            design_dir = root / "outputs" / "d1"
            native_dir.mkdir(parents=True)
            design_dir.mkdir(parents=True)
            native_pdb = native_dir / "s1_native.pdb"
            design_pdb = design_dir / "d1.pdb"
            write_pdb(native_pdb, "AGAG")
            write_pdb(design_pdb, "AGAG")
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "s1_native",
                        "fold_backend": "mockfold",
                        "sequence_role": "native_control",
                        "is_native_reference": True,
                        "validation_status": "missing_mockfold_output",
                        "selected_decoy": False,
                        "fold_eligible": True,
                        "fold_output_dir": str(native_dir),
                        "sequence": "AGAG",
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "missing_mockfold_output",
                        "selected_decoy": False,
                        "fold_eligible": True,
                        "fold_output_dir": str(design_dir),
                        "sequence": "AGAG",
                        "designed_sequence": "AGAG",
                    },
                ]
            ).to_csv(validation, index=False)
            write_fold_manifest(fold_dir, {"mode_a": validation}, work_root=work_root)
            write_submission_manifest(
                work_root,
                [
                    {"redesign_mode": "mode_a", "structure_id": "s1", "design_id": "s1_native", "sequence_role": "native_control", "fold_output_dir": str(native_dir), "submission_status": "submitted", "completion_status": "COMPLETED"},
                    {"redesign_mode": "mode_a", "structure_id": "s1", "design_id": "d1", "sequence_role": "redesigned_decoy", "fold_output_dir": str(design_dir), "submission_status": "submitted", "completion_status": "COMPLETED"},
                ],
            )

            qc = build_fold_qc_table(fold_dir)
            by_id = qc.set_index("design_id")
            row = by_id.loc["d1"]

            self.assertEqual("passed", row["final_qc_status"])
            self.assertTrue(bool(row["graft_eligible"]))
            self.assertEqual("computed", row["monomer_metric_status"])
            self.assertEqual("native_refold", row["fold_reference"])
            self.assertAlmostEqual(1.0, float(row["monomer_tm_score"]))
            self.assertAlmostEqual(0.0, float(row["monomer_ca_rmsd"]))
            self.assertAlmostEqual(1.0, float(row["monomer_aligned_fraction"]))
            self.assertEqual(4, int(row["monomer_matched_ca"]))
            self.assertEqual("native_control", by_id.loc["s1_native", "final_qc_status"])
            self.assertFalse(bool(by_id.loc["s1_native", "graft_eligible"]))

    def test_invalid_status_sequence_mismatch_and_fold_ineligible_remain_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            work_root = root / "work" / "decoy" / "exp1"
            tables.mkdir(parents=True)
            ok_out = root / "outputs"
            ok_out.mkdir()
            for design_id, sequence in {"failed": "AG", "mismatch": "AA", "ineligible": "AG", "msa_bad": "AG"}.items():
                out_dir = ok_out / design_id
                out_dir.mkdir()
                write_pdb(out_dir / f"{design_id}.pdb", sequence)
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {"redesign_mode": "mode_a", "structure_id": "s1", "design_id": "failed", "fold_backend": "mockfold", "sequence_role": "redesigned_decoy", "validation_status": "failed_fold", "selected_decoy": True, "fold_eligible": True, "fold_output_dir": str(ok_out / "failed"), "sequence": "AG"},
                    {"redesign_mode": "mode_a", "structure_id": "s2", "design_id": "mismatch", "fold_backend": "mockfold", "sequence_role": "redesigned_decoy", "validation_status": "pass", "selected_decoy": True, "fold_eligible": True, "fold_output_dir": str(ok_out / "mismatch"), "sequence": "AG"},
                    {"redesign_mode": "mode_a", "structure_id": "s3", "design_id": "ineligible", "fold_backend": "mockfold", "sequence_role": "redesigned_decoy", "validation_status": "pass", "selected_decoy": True, "fold_eligible": False, "fold_exclusion_reason": "policy_excluded", "fold_output_dir": str(ok_out / "ineligible"), "sequence": "AG"},
                    {"redesign_mode": "mode_a", "structure_id": "s4", "design_id": "msa_bad", "fold_backend": "mockfold", "sequence_role": "redesigned_decoy", "validation_status": "msa_ineligible", "selected_decoy": True, "fold_eligible": False, "fold_exclusion_reason": "msa_missing", "fold_output_dir": str(ok_out / "msa_bad"), "sequence": "AG"},
                ]
            ).to_csv(validation, index=False)
            write_fold_manifest(fold_dir, {"mode_a": validation}, work_root=work_root)
            write_submission_manifest(
                work_root,
                [
                    {"redesign_mode": "mode_a", "structure_id": f"s{i}", "design_id": design_id, "sequence_role": "redesigned_decoy", "fold_output_dir": str(ok_out / design_id), "submission_status": "submitted", "completion_status": "COMPLETED"}
                    for i, design_id in enumerate(["failed", "mismatch", "ineligible", "msa_bad"], start=1)
                ],
            )

            qc = build_fold_qc_table(fold_dir)
            by_id = qc.set_index("design_id")

            self.assertEqual({"excluded"}, set(qc["final_qc_status"]))
            self.assertEqual("validation_status:failed_fold", by_id.loc["failed", "final_qc_exclusion_reason"])
            self.assertEqual("sequence_mismatch", by_id.loc["mismatch", "final_qc_exclusion_reason"])
            self.assertEqual("policy_excluded", by_id.loc["ineligible", "final_qc_exclusion_reason"])
            self.assertEqual("msa_missing", by_id.loc["msa_bad", "final_qc_exclusion_reason"])
            self.assertFalse(qc["selected_decoy"].any())
            self.assertFalse(qc["graft_eligible"].any())

    def test_duplicate_design_ids_across_modes_merge_by_mode_and_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            work_root = root / "work" / "decoy" / "exp1"
            validations = {}
            submission_rows = []
            for mode, structure_id in [("mode_a", "s1"), ("mode_b", "s2")]:
                tables = root / "tables" / mode
                tables.mkdir(parents=True)
                out_dir = root / "outputs" / mode / "shared_design"
                out_dir.mkdir(parents=True)
                write_pdb(out_dir / "shared_design.pdb", "AG")
                validation = tables / "decoy_validation.csv"
                pd.DataFrame(
                    [
                        {"redesign_mode": mode, "structure_id": structure_id, "design_id": "shared_design", "fold_backend": "mockfold", "sequence_role": "redesigned_decoy", "validation_status": "missing_mockfold_output", "selected_decoy": False, "fold_eligible": True, "fold_output_dir": str(out_dir), "sequence": "AG"}
                    ]
                ).to_csv(validation, index=False)
                validations[mode] = validation
                submission_rows.append({"redesign_mode": mode, "structure_id": structure_id, "design_id": "shared_design", "sequence_role": "redesigned_decoy", "fold_output_dir": str(out_dir), "submission_status": "submitted", "completion_status": "COMPLETED"})
            write_fold_manifest(fold_dir, validations, work_root=work_root)
            write_submission_manifest(work_root, submission_rows)

            qc = build_fold_qc_table(fold_dir)

            self.assertEqual(2, len(qc))
            self.assertEqual({"mode_a", "mode_b"}, set(qc["redesign_mode"]))
            self.assertEqual({"passed"}, set(qc["final_qc_status"]))
            self.assertTrue(qc["graft_eligible"].all())
    def test_fold_job_bundle_restores_sequence_metadata_for_rerun_qc_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            work_root = root / "work" / "decoy" / "exp1"
            tables.mkdir(parents=True)
            out_dir = root / "outputs" / "d1"
            out_dir.mkdir(parents=True)
            pdb = out_dir / "d1.pdb"
            write_pdb(pdb, "AG")
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "missing_mockfold_output",
                        "selected_decoy": False,
                        "fold_eligible": True,
                        "fold_submission_present": False,
                        "fold_input": pd.NA,
                        "msa_consumed": True,
                        "msa_cache_hit": False,
                    }
                ]
            ).to_csv(validation, index=False)
            work_dirs = write_fold_manifest(fold_dir, {"mode_a": validation}, work_root=work_root)
            job_bundle = work_dirs["mode_a"] / "job_bundle" / "mockfold_monomer_jobs.tsv"
            job_bundle.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "sequence_role": "redesigned_decoy",
                        "fold_backend": "mockfold",
                        "sequence": "AG",
                        "designed_sequence": pd.NA,
                        "fold_output_dir": str(out_dir),
                        "fold_input": str(root / "inputs" / "d1.fa"),
                        "msa_consumed": True,
                        "msa_cache_hit": False,
                    }
                ]
            ).to_csv(job_bundle, sep="\t", index=False)
            write_submission_manifest(
                work_root,
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "sequence_role": "redesigned_decoy",
                        "fold_output_dir": str(out_dir),
                        "submission_status": "submitted",
                        "completion_status": "COMPLETED",
                    }
                ],
            )

            qc = build_fold_qc_table(fold_dir)
            row = qc.iloc[0]

            self.assertEqual("passed", row["final_qc_status"])
            self.assertEqual("exact", row["sequence_match_status"])
            self.assertEqual("AG", row["sequence"])
            self.assertEqual("AG", row["designed_sequence"])
            self.assertTrue(bool(row["fold_submission_present"]))
            self.assertFalse(any(column.endswith(("_job_x", "_job_y")) for column in qc.columns))


    def test_submission_merge_handles_blank_mode_from_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            tables.mkdir(parents=True)
            out_dir = root / "fold_outputs" / "d1"
            out_dir.mkdir(parents=True)
            write_pdb(out_dir / "d1.pdb", "AG")
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "fold_backend": "esmfold2",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "sequence": "AG",
                    }
                ]
            ).to_csv(validation, index=False)
            work_dirs = write_fold_manifest(fold_dir, {"mode_a": validation})
            write_submission_manifest(
                work_dirs["mode_a"],
                [
                    {
                        "redesign_mode": "",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "sequence_role": "redesigned_decoy",
                        "fold_output_dir": str(out_dir),
                        "submission_status": "submitted",
                        "completion_status": "COMPLETED",
                    }
                ],
            )

            qc = build_fold_qc_table(fold_dir)

            row = qc.iloc[0]
            self.assertTrue(bool(row["fold_submission_present"]))
            self.assertEqual("passed", row["final_qc_status"])
            self.assertEqual(str(out_dir / "d1.pdb"), row["structure_path"])


    def test_esmfold2_fold_qc_marks_only_top1_redesign_per_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            tables.mkdir(parents=True)
            pdbs = {}
            for design_id in ["s1_native", "d_low_tm", "d_tie_worse_rmsd", "d_top", "d_single", "mock_1", "mock_2"]:
                path = root / f"{design_id}.pdb"
                write_pdb(path, "AG")
                pdbs[design_id] = path
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "s1_native",
                        "fold_backend": "esmfold2",
                        "sequence_role": "native_control",
                        "is_native_reference": True,
                        "validation_status": "native_reference",
                        "selected_decoy": False,
                        "prediction_path": str(pdbs["s1_native"]),
                        "sequence": "AG",
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d_low_tm",
                        "fold_backend": "esmfold2",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(pdbs["d_low_tm"]),
                        "sequence": "AG",
                        "monomer_tm_score": 0.80,
                        "monomer_ca_rmsd": 0.1,
                        "monomer_aligned_fraction": 1.0,
                        "mpnn_score": 1.0,
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d_tie_worse_rmsd",
                        "fold_backend": "esmfold2",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(pdbs["d_tie_worse_rmsd"]),
                        "sequence": "AG",
                        "monomer_tm_score": 0.95,
                        "monomer_ca_rmsd": 2.0,
                        "monomer_aligned_fraction": 1.0,
                        "mpnn_score": 0.1,
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d_top",
                        "fold_backend": "esmfold2",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(pdbs["d_top"]),
                        "sequence": "AG",
                        "monomer_tm_score": 0.95,
                        "monomer_ca_rmsd": 1.0,
                        "monomer_aligned_fraction": 1.0,
                        "mpnn_score": 5.0,
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s2",
                        "design_id": "d_single",
                        "fold_backend": "esmfold2",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(pdbs["d_single"]),
                        "sequence": "AG",
                        "monomer_tm_score": 0.7,
                        "monomer_ca_rmsd": 3.0,
                        "monomer_aligned_fraction": 1.0,
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s3",
                        "design_id": "mock_1",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(pdbs["mock_1"]),
                        "sequence": "AG",
                    },
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s3",
                        "design_id": "mock_2",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "pass",
                        "selected_decoy": True,
                        "prediction_path": str(pdbs["mock_2"]),
                        "sequence": "AG",
                    },
                ]
            ).to_csv(validation, index=False)
            write_fold_manifest(fold_dir, {"mode_a": validation})

            qc = build_fold_qc_table(fold_dir)
            by_id = qc.set_index("design_id")

            self.assertFalse(bool(by_id.loc["s1_native", "graft_eligible"]))
            self.assertTrue(bool(by_id.loc["d_top", "graft_eligible"]))
            self.assertEqual("passed", by_id.loc["d_top", "final_qc_status"])
            self.assertEqual(1, int(by_id.loc["d_top", "selection_rank"]))
            self.assertFalse(bool(by_id.loc["d_tie_worse_rmsd", "graft_eligible"]))
            self.assertEqual("not_top1", by_id.loc["d_tie_worse_rmsd", "final_qc_exclusion_reason"])
            self.assertEqual(2, int(by_id.loc["d_tie_worse_rmsd", "selection_rank"]))
            self.assertFalse(bool(by_id.loc["d_low_tm", "graft_eligible"]))
            self.assertEqual(3, int(by_id.loc["d_low_tm", "selection_rank"]))
            self.assertTrue(bool(by_id.loc["d_single", "graft_eligible"]))
            self.assertTrue(bool(by_id.loc["mock_1", "graft_eligible"]))
            self.assertTrue(bool(by_id.loc["mock_2", "graft_eligible"]))
            esm_selected = qc[(qc["fold_backend"] == "esmfold2") & (qc["sequence_role"] == "redesigned_decoy") & qc["graft_eligible"].map(bool)]
            self.assertEqual({"d_top", "d_single"}, set(esm_selected["design_id"]))


    def test_main_writes_per_mode_qc_and_zero_survivor_csv_has_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fold_dir = root / "fold"
            tables = root / "tables" / "mode_a"
            tables.mkdir(parents=True)
            pdb = root / "fold.pdb"
            write_pdb(pdb, "AG")
            validation = tables / "decoy_validation.csv"
            pd.DataFrame(
                [
                    {
                        "redesign_mode": "mode_a",
                        "structure_id": "s1",
                        "design_id": "d1",
                        "fold_backend": "mockfold",
                        "sequence_role": "redesigned_decoy",
                        "validation_status": "failed_fold",
                        "selected_decoy": False,
                        "prediction_path": str(pdb),
                        "sequence": "AG",
                    }
                ]
            ).to_csv(validation, index=False)
            write_fold_manifest(fold_dir, {"mode_a": validation})

            rc = main(["--fold-dir", str(fold_dir), "--output-root", str(root), "--skip-figure"])

            self.assertEqual(0, rc)
            combined = root / "results" / "decoy" / "tables" / "exp1" / "all_modes" / "post_fold_qc.csv"
            per_mode = root / "results" / "decoy" / "tables" / "exp1" / "mode_a" / "post_fold_qc.csv"
            self.assertTrue(combined.exists())
            self.assertTrue(per_mode.exists())
            table = pd.read_csv(combined)
            self.assertIn("final_qc_status", table.columns)
            self.assertEqual(0, int(table["graft_eligible"].sum()))


if __name__ == "__main__":
    unittest.main()
