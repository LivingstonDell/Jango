import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from jango.pipeline import decoy_fold
from jango.pipeline.cases import ModeThresholds, select_decoy_source_cases, split_cases_by_mode
from jango.pipeline.labels import LabelConfig, add_label_columns, merge_label_columns


TEST_TMP_ROOT = Path(os.environ.get("JANGO_TEST_TMP", tempfile.gettempdir())) / "jango_phase2_tests"


def workspace_case_root(name: str) -> Path:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = TEST_TMP_ROOT / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Phase3PipelineContractTests(unittest.TestCase):
    def test_label_schema_uses_source_label_role_confidence(self):
        manifest = pd.DataFrame({"structure_id": ["s1"]})
        labels = LabelConfig(source="sabdab1_2", label="positive", role="native_complex", confidence="high")
        labeled = add_label_columns(manifest, labels)
        self.assertEqual(["source", "label", "role", "confidence"], list(labeled.columns[1:]))
        self.assertNotIn("label_type", labeled.columns)
        merged = merge_label_columns(table=pd.DataFrame({"structure_id": ["s1"], "x": [1]}), manifest=labeled)
        self.assertEqual(merged.loc[0, "role"], "native_complex")

    def test_source_case_selection_does_not_drop_duplicate_pdb_ids(self):
        root = workspace_case_root("case_selection")
        manifest = pd.DataFrame(
            {
                "structure_id": ["s1", "s2", "short"],
                "pdb_id": ["1abc", "1abc", "2def"],
                "nanobody_chain": ["H", "H", "H"],
                "antigen_chain": ["A", "B", "A"],
                "source_filename": ["s1.pdb", "s2.pdb", "short.pdb"],
                "nanobody_length": [120, 121, 122],
                "antigen_length": [100, 150, 30],
                "antigen_class": ["protein", "protein", "protein"],
            }
        )
        features = pd.DataFrame(
            {
                "structure_id": ["s1", "s2", "short"],
                "cdr3_interface_fraction": [0.2, 0.3, 0.1],
                "residue_contact_pairs_5A": [20, 30, 5],
            }
        )
        rosetta = pd.DataFrame(
            {
                "structure_id": ["s1", "s2", "short"],
                "dSASA_int": [100.0, 200.0, 50.0],
                "dG_separated": [-5.0, -7.0, -1.0],
                "sc_value": [0.6, 0.7, 0.2],
                "fullpose_total_score": [-100.0, -120.0, -50.0],
            }
        )
        manifest_path = root / "manifest.csv"
        features_path = root / "features.csv"
        rosetta_path = root / "rosetta.csv"
        out_path = root / "cases.csv"
        manifest.to_csv(manifest_path, index=False)
        features.to_csv(features_path, index=False)
        rosetta.to_csv(rosetta_path, index=False)

        selected = select_decoy_source_cases(
            manifest_csv=manifest_path,
            native_features_csv=features_path,
            native_relaxed_rosetta_csv=rosetta_path,
            out_csv=out_path,
            case_count=10,
        )

        self.assertEqual(["s1", "s2"], selected["structure_id"].tolist())
        self.assertEqual(["1abc", "1abc"], selected["pdb_id"].tolist())
        self.assertTrue((selected["selection_policy"] == "all_filtered_from_input_max_no_pdb_dedup").all())

        selected_legacy_limit = select_decoy_source_cases(
            manifest_csv=manifest_path,
            native_features_csv=features_path,
            native_relaxed_rosetta_csv=rosetta_path,
            out_csv=root / "cases_legacy_limit.csv",
            case_count=1,
        )
        self.assertEqual(["s1", "s2"], selected_legacy_limit["structure_id"].tolist())

    def test_mode_split_writes_manual_and_auto_manifests(self):
        root = workspace_case_root("mode_split")
        base = pd.DataFrame(
            {
                "structure_id": ["small", "medium", "large"],
                "antigen_length": [90, 150, 230],
                "antigen_contact_count": [5, 20, 20],
            }
        )
        base_path = root / "base.csv"
        base.to_csv(base_path, index=False)

        manual, _ = split_cases_by_mode(
            base_case_manifest=base_path,
            out_dir=root / "manual",
            mode_policy="manual",
            modes=["hotspot_only", "full_antigen"],
        )
        self.assertEqual({"hotspot_only", "full_antigen"}, set(manual))
        self.assertEqual(3, len(pd.read_csv(manual["hotspot_only"])))

        auto, _ = split_cases_by_mode(
            base_case_manifest=base_path,
            out_dir=root / "auto",
            mode_policy="auto_by_length",
            modes=["hotspot_only", "interface_only", "full_antigen"],
            thresholds=ModeThresholds(),
        )
        self.assertEqual({"hotspot_only", "interface_only", "full_antigen"}, set(auto))

    def test_esmfold2_required_msa_requires_cache_and_is_not_faked(self):
        parser = decoy_fold.build_parser()
        args = parser.parse_args(
            [
                "--prefold-dir", "prefold",
                "--atlas-root", "atlas",
                "--backend", "esmfold2",
                "--msa-policy", "required",
                "--output-root", "out",
                "--plan-only",
            ]
        )
        with self.assertRaises(SystemExit):
            decoy_fold.validate_stage_args(parser, args, "esmfold2_1_manual", "esmfold2")


    def test_decoy_fold_requires_explicit_msa_mode_or_legacy_policy(self):
        parser = decoy_fold.build_parser()
        args = parser.parse_args(
            [
                "--prefold-dir", "prefold",
                "--atlas-root", "atlas",
                "--backend", "boltz2",
                "--output-root", "out",
                "--plan-only",
            ]
        )
        with self.assertRaises(SystemExit):
            decoy_fold.validate_stage_args(parser, args, "boltz2_1_manual", "boltz2")

    def test_boltz_backend_managed_msa_plan_is_valid(self):
        parser = decoy_fold.build_parser()
        args = parser.parse_args(
            [
                "--prefold-dir", "prefold",
                "--atlas-root", "atlas",
                "--backend", "boltz2",
                "--msa-mode", "backend-managed",
                "--msa-provider", "backend-managed",
                "--output-root", "out",
                "--plan-only",
            ]
        )
        decoy_fold.validate_stage_args(parser, args, "boltz2_1_manual", "boltz2")
if __name__ == "__main__":
    unittest.main()
