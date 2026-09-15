import unittest
from pathlib import Path

from jango.config import ExperimentConfig, experiment_name
from jango.pipeline import decoy_analyze, decoy_fold, decoy_fold_relax, decoy_prefold, nanobody_pipeline
from jango.runtime import ensure_matching_rosetta_states


class Phase2DelphiTests(unittest.TestCase):
    def test_experiment_name_requires_backend(self):
        with self.assertRaises(ValueError):
            experiment_name(backend="", max_structures=10, policy="manual")
        self.assertEqual(
            experiment_name(backend="esmfold2", max_structures=1258, policy="auto_by_length"),
            "esmfold2_max1258_auto_by_length",
        )

    def test_experiment_config_namespace(self):
        cfg = ExperimentConfig(
            backend="boltz2",
            case_count=1258,
            policy="manual",
            output_root=Path("out"),
        )
        self.assertEqual(cfg.namespace, "boltz2_max1258_manual")

    def test_rosetta_state_gate(self):
        ensure_matching_rosetta_states("native_relaxed", "decoy_relaxed")
        with self.assertRaises(ValueError):
            ensure_matching_rosetta_states("native", "decoy_relaxed")

    def test_pipeline_parsers_have_expected_prog_names(self):
        self.assertEqual(nanobody_pipeline.build_parser().prog, "nanobody-pipeline")
        self.assertEqual(decoy_prefold.build_parser().prog, "decoy-prefold")
        self.assertEqual(decoy_fold.build_parser().prog, "decoy-fold")
        self.assertEqual(decoy_analyze.build_parser().prog, "decoy-analyze")

    def test_decoy_fold_relax_reads_dockq_rs_environment(self):
        args = decoy_fold_relax.build_parser().parse_args(
            [
                "--fold-dir", "fold",
                "--atlas-root", "atlas",
                "--native-relax-manifest", "native_relax.csv",
                "--rosetta-runtime", "apptainer",
                "--rosetta-image", "docker://rosetta",
                "--output-root", "out",
            ]
        )
        self.assertIsNone(args.dockq_bin)
        self.assertIsNone(args.dockq_provider)
        self.assertIsNone(args.dockq_python)

    def test_decoy_fold_defaults_to_five_decoy_sequences(self):
        args = decoy_fold.build_parser().parse_args(
            [
                "--prefold-dir", "prefold",
                "--atlas-root", "atlas",
                "--backend", "boltz2",
                "--msa-mode", "disabled",
                "--output-root", "out",
            ]
        )
        self.assertEqual(args.max_decoy_sequences, 5)

    def test_decoy_validate_command_passes_sequence_limit_and_mode_policy(self):
        args = decoy_fold.build_parser().parse_args(
            [
                "--prefold-dir", "prefold",
                "--atlas-root", "atlas",
                "--backend", "boltz2",
                "--msa-mode", "disabled",
                "--msa-provider", "none",
                "--max-decoy-sequences", "3",
                "--output-root", "out",
            ]
        )
        command = decoy_fold.validate_command(
            args,
            Path("cases.csv"),
            Path("raw"),
            Path("work"),
            Path("tables"),
            "boltz2_max10_manual",
            "manual",
        )
        argv = [str(part) for part in command.argv]
        self.assertIn("--max-decoy-sequences", argv)
        self.assertEqual(argv[argv.index("--max-decoy-sequences") + 1], "3")
        self.assertIn("--mode-policy", argv)
        self.assertEqual(argv[argv.index("--mode-policy") + 1], "manual")


if __name__ == "__main__":
    unittest.main()
