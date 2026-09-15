from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from nbia.decoys import validate_decoy_designs
from nbia.folding import MSAError
from nbia.folding.msa import a3m_query_sequence, derive_a3m_from_native_query, read_fasta_like_records, sequence_sha256


NATIVE_SEQUENCES = {"s1": "ACDE", "s2": "FGHI"}
DECOY_SEQUENCES = {"s1": "YCDE", "s2": "YGHI"}


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


class MSAFailurePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw_dir = self.root / "raw"
        self.work_dir = self.root / "work"
        self.tables_dir = self.root / "tables"
        self.cache_dir = self.root / "msa_cache"
        self.case_manifest = self.root / "cases.csv"
        self.raw_dir.mkdir(parents=True)
        self._write_cases()
        self._write_mpnn_outputs()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_cases(self) -> None:
        pd.DataFrame(
            [
                {
                    "structure_id": structure_id,
                    "pdb_id": structure_id,
                    "nanobody_chain": "N",
                    "antigen_chain": "X",
                    "source_filename": f"{structure_id}.pdb",
                    "native_antigen_sequence": sequence,
                    "antigen_contact_positions": "1",
                }
                for structure_id, sequence in NATIVE_SEQUENCES.items()
            ]
        ).to_csv(self.case_manifest, index=False)

    def _write_mpnn_outputs(self) -> None:
        for structure_id, sequence in DECOY_SEQUENCES.items():
            out_dir = self.work_dir / "mpnn_outputs" / structure_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "candidates.fa").write_text(f">candidate score=0.1\n{sequence}\n")

    def _write_cached_a3m(self, sequence: str, hit_sequence: str | None = None) -> Path:
        path = self.cache_dir / "by_sequence_hash" / sequence_sha256(sequence) / "unpaired.a3m"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f">query\n{sequence}\n>hit\n{hit_sequence or sequence}\n")
        return path

    def _cache_all_except_s1_decoy(self) -> None:
        for sequence in [NATIVE_SEQUENCES["s1"], NATIVE_SEQUENCES["s2"], DECOY_SEQUENCES["s2"]]:
            self._write_cached_a3m(sequence)

    def _run_validate(self, policy: str, reuse_policy: str = "exact_sequence"):
        return validate_decoy_designs(
            case_manifest_csv=self.case_manifest,
            raw_dir=self.raw_dir,
            work_dir=self.work_dir,
            tables_dir=self.tables_dir,
            fold_backend="esmfold2",
            esmfold2_python="/usr/bin/python",
            esmfold2_root=self.root / "esmfold2",
            esmfold2_model_id_or_path="fixture-model",
            esmfold2_cache_dir=self.root / "hf_cache",
            esmfold2_device="cpu",
            msa_mode="required",
            msa_provider="abforge_get_or_build",
            msa_cache_dir=self.cache_dir,
            msa_format="a3m",
            msa_pairing="unpaired",
            msa_build_if_missing=False,
            msa_failure_policy=policy,
            msa_reuse_policy=reuse_policy,
            max_decoy_sequences_per_structure=1,
        )

    def test_abort_stops_on_first_missing_required_msa(self) -> None:
        self._write_cached_a3m(NATIVE_SEQUENCES["s1"])
        with self.assertRaises(MSAError):
            self._run_validate("abort")

    def test_skip_design_marks_one_design_ineligible_and_continues(self) -> None:
        self._cache_all_except_s1_decoy()
        _, fold_df, validation_df = self._run_validate("skip_design")

        self.assertEqual(len(fold_df), 4)
        s1_decoy = fold_df.set_index("design_id").loc["s1_mpnn_0000"]
        self.assertFalse(as_bool(s1_decoy["fold_eligible"]))
        self.assertEqual(s1_decoy["fold_exclusion_reason"], "msa_resolution_failed")
        self.assertEqual(s1_decoy["msa_status"], "msa_missing")
        self.assertIn("AbForge cached unpaired A3M was not found", s1_decoy["msa_error"])
        self.assertEqual(str(s1_decoy["fold_command"]), "")
        self.assertNotEqual(s1_decoy["msa_status"], "single_sequence_explicit_fallback")

        manifest = pd.read_csv(self.tables_dir / "decoy_esmfold2_monomer_manifest.csv")
        self.assertIn("s1_mpnn_0000", set(manifest["design_id"]))
        validation = validation_df.set_index("design_id").loc["s1_mpnn_0000"]
        self.assertEqual(validation["validation_status"], "msa_ineligible")

        jobs = pd.read_csv(self.work_dir / "job_bundle" / "esmfold2_monomer_jobs.tsv", sep="\t")
        self.assertNotIn("s1_mpnn_0000", set(jobs["design_id"]))
        self.assertIn("s1_native", set(jobs["design_id"]))
        self.assertIn("s2_mpnn_0000", set(jobs["design_id"]))

        exclusions = pd.read_csv(self.tables_dir / "decoy_msa_exclusion_summary.csv")
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(exclusions.iloc[0]["design_id"], "s1_mpnn_0000")

    def test_skip_case_excludes_all_candidates_for_failed_structure(self) -> None:
        self._cache_all_except_s1_decoy()
        _, fold_df, _ = self._run_validate("skip_case")
        by_design = fold_df.set_index("design_id")

        self.assertFalse(as_bool(by_design.loc["s1_native", "fold_eligible"]))
        self.assertFalse(as_bool(by_design.loc["s1_mpnn_0000", "fold_eligible"]))
        self.assertTrue(as_bool(by_design.loc["s2_native", "fold_eligible"]))
        self.assertTrue(as_bool(by_design.loc["s2_mpnn_0000", "fold_eligible"]))
        self.assertEqual(by_design.loc["s1_native", "fold_exclusion_reason"], "msa_failure_in_case")
        self.assertEqual(by_design.loc["s1_mpnn_0000", "fold_exclusion_reason"], "msa_resolution_failed")

        manifest = pd.read_csv(self.tables_dir / "decoy_esmfold2_monomer_manifest.csv")
        self.assertIn("s1_native", set(manifest["design_id"]))
        self.assertIn("s1_mpnn_0000", set(manifest["design_id"]))

        jobs = pd.read_csv(self.work_dir / "job_bundle" / "esmfold2_monomer_jobs.tsv", sep="\t")
        self.assertNotIn("s1_native", set(jobs["design_id"]))
        self.assertNotIn("s1_mpnn_0000", set(jobs["design_id"]))
        self.assertEqual(set(jobs["design_id"]), {"s2_native", "s2_mpnn_0000"})


    def test_native_reuse_transformation_preserves_homolog_rows_and_width(self) -> None:
        native_a3m = self.root / "native.a3m"
        native_a3m.write_text(">query\nAC-D\n>hit1\nA--D\n>hit2\naC-D\n")
        output = self.root / "derived.a3m"

        details = derive_a3m_from_native_query(
            native_a3m_path=native_a3m,
            native_sequence="ACD",
            design_sequence="YCD",
            output_path=output,
        )

        records = read_fasta_like_records(output)
        self.assertEqual(records[0], ("query", "YC-D"))
        self.assertEqual(records[1:], [("hit1", "A--D"), ("hit2", "aC-D")])
        self.assertEqual(len(records[0][1]), len("AC-D"))
        self.assertEqual(a3m_query_sequence(records[0][1]), "YCD")
        self.assertEqual(details["mutation_count"], 1)
        self.assertEqual(details["query_replacement_status"], "replaced")
        self.assertEqual(details["length_match_status"], "matched")
        self.assertEqual(details["msa_depth"], 3)

    def test_native_reuse_rejects_insertions_and_deletions(self) -> None:
        native_a3m = self.root / "native.a3m"
        native_a3m.write_text(">query\nACDE\n>hit\nAC-E\n")
        with self.assertRaisesRegex(MSAError, "same length"):
            derive_a3m_from_native_query(
                native_a3m_path=native_a3m,
                native_sequence="ACDE",
                design_sequence="ACDEF",
                output_path=self.root / "derived.a3m",
            )

    def test_native_reuse_derives_design_msa_and_keeps_native_controls_direct(self) -> None:
        self._write_cached_a3m(NATIVE_SEQUENCES["s1"], hit_sequence="A-DE")
        self._write_cached_a3m(NATIVE_SEQUENCES["s2"], hit_sequence="F-HI")

        _, fold_df, _ = self._run_validate("skip_design", reuse_policy="native_reuse")
        by_design = fold_df.set_index("design_id")

        s1_decoy = by_design.loc["s1_mpnn_0000"]
        self.assertTrue(as_bool(s1_decoy["fold_eligible"]))
        self.assertEqual(s1_decoy["msa_status"], "msa_generated")
        self.assertEqual(s1_decoy["msa_source"], "native_reuse")
        self.assertEqual(int(s1_decoy["msa_mutation_count"]), 1)
        self.assertEqual(s1_decoy["msa_query_replacement_status"], "replaced")
        self.assertEqual(s1_decoy["msa_length_match_status"], "matched")
        self.assertTrue(as_bool(s1_decoy["msa_model_consumed"]))
        self.assertTrue(as_bool(s1_decoy["msa_consumed"]))

        derived_path = Path(str(s1_decoy["msa_path"]))
        self.assertTrue(derived_path.is_file())
        records = read_fasta_like_records(derived_path)
        self.assertEqual(a3m_query_sequence(records[0][1]), DECOY_SEQUENCES["s1"])
        self.assertEqual(records[1], ("hit", "A-DE"))
        self.assertIn(str(derived_path), str(s1_decoy["fold_command"]))

        s1_native = by_design.loc["s1_native"]
        native_path = self.cache_dir / "by_sequence_hash" / sequence_sha256(NATIVE_SEQUENCES["s1"]) / "unpaired.a3m"
        self.assertEqual(Path(str(s1_native["msa_path"])), native_path.resolve())
        self.assertEqual(s1_native["msa_status"], "msa_cache_hit")
        self.assertEqual(s1_native["msa_source"], "native_cache")

    def test_native_reuse_missing_native_msa_follows_skip_design_policy(self) -> None:
        self._write_cached_a3m(NATIVE_SEQUENCES["s2"])

        _, fold_df, _ = self._run_validate("skip_design", reuse_policy="native_reuse")
        by_design = fold_df.set_index("design_id")

        self.assertFalse(as_bool(by_design.loc["s1_mpnn_0000", "fold_eligible"]))
        self.assertEqual(by_design.loc["s1_mpnn_0000", "fold_exclusion_reason"], "msa_resolution_failed")
        self.assertEqual(by_design.loc["s1_mpnn_0000", "msa_status"], "msa_missing")
        self.assertIn("native_reuse native A3M was not found", by_design.loc["s1_mpnn_0000", "msa_error"])
        self.assertTrue(as_bool(by_design.loc["s2_mpnn_0000", "fold_eligible"]))

    def test_invalid_policy_values_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid MSA failure policy"):
            self._run_validate("skip_everything")


if __name__ == "__main__":
    unittest.main()
