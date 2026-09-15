from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import unittest

import pandas as pd

from jango.pipeline import fett


class FettPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        from tempfile import TemporaryDirectory

        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "unit"
        self.configs = self.root / "configs"
        self.configs.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.old_path = os.environ.get("PATH", "")
        self.env_restore: dict[str, str | None] = {}
        os.environ["PATH"] = f"{self.bin}:{self.old_path}"
        for command in [
            "fett",
            "nanobody-pipeline",
            "decoy-prefold",
            "decoy-fold",
            "decoy-analyze",
            "jango",
            "sbatch",
            "squeue",
            "scontrol",
            "apptainer",
            "anarci",
            "boltz",
            "colabfold_search",
            "mmseqs",
        ]:
            self._fake_executable(self.bin / command)

        self.raw = self.root / "raw"
        self.raw.mkdir()
        (self.raw / "1abc.tar.gz").write_bytes(b"fixture")
        self.proteinmpnn_home = self.root / "ProteinMPNN"
        self.proteinmpnn_home.mkdir()
        self.proteinmpnn_runner = self.proteinmpnn_home / "protein_mpnn_run.py"
        self.proteinmpnn_runner.write_text("print('ProteinMPNN fixture')\n")
        self.esm_root = self.root / "esmfold2"
        self.esm_root.mkdir()
        self.esm_cache = self.root / "hf_cache"
        self.esm_cache.mkdir()
        self.esmc_model = self.esm_cache / "model"
        self.esmc_model.mkdir()
        self.msa_cache = self.root / "msa_cache"
        self.msa_cache.mkdir()
        self.boltz_model = self.root / "boltz_models"
        self.boltz_model.mkdir()
        self.boltz_cache_parent = self.root / "boltz_cache_parent"
        self.boltz_cache_parent.mkdir()
        self.apptainer_cache_parent = self.root / "apptainer_parent"
        self.apptainer_cache_parent.mkdir()
        self.user_root = self.root / "user-jango"
        self.anarci_bin = self.bin / "anarci"
        self.boltz = self.bin / "boltz"

        self.paths_config = self.configs / "paths.env"
        self.runtime_config = self.configs / "runtime.env"
        self.paths_config.write_text(
            "export JANGO_SOURCE=<JANGO_REPO>\n"
            "export JANGO_SOURCE_LABEL=sabdab_1_2\n"
            "export JANGO_RUN_ROOT=${JANGO_RUN_ROOT:?JANGO_RUN_ROOT must be provided by CLI}\n"
            "export JANGO_OUTPUT_ROOT=${JANGO_OUTPUT_ROOT:?JANGO_OUTPUT_ROOT must be provided by CLI}\n"
            "export JANGO_RAW_TARBALLS=${JANGO_RAW_TARBALLS:?JANGO_RAW_TARBALLS must be provided by CLI}\n"
        )
        self.runtime_config.write_text(self._esm_runtime_text())
        self.decoy_config = self.configs / "decoys.yml"
        self.decoy_config.write_text("redesign: fixture\n")
        self._set_env("JANGO_USER_ROOT", str(self.user_root))
        self._set_env("PROTEINMPNN_HOME", str(self.proteinmpnn_home))
        self._set_env("PROTEINMPNN_PYTHON", sys.executable)
        self._set_env("PROTEINMPNN_RUNNER", str(self.proteinmpnn_runner))
        self._set_env("ESMFOLD2_PYTHON", sys.executable)
        self._set_env("ESM_ROOT", str(self.esm_root))
        self._set_env("ESMFOLD2_CACHE", str(self.esm_cache))
        self._set_env("ESMC_MODEL_PATH", str(self.esmc_model))
        self._set_env("BOLTZ_EXECUTABLE", str(self.boltz))
        self._set_env("BOLTZ_PYTHON", sys.executable)
        self._set_env("BOLTZ_MODEL_ROOT", str(self.boltz_model))
        self._set_env("MSA_CACHE_ROOT", str(self.msa_cache))
        self._set_env("APPTAINER_BIN", str(self.bin / "apptainer"))
        self._set_env("DOCKQ_PYTHON", sys.executable)
        self._set_env("ANARCI_PYTHON", sys.executable)
        self._set_env("ANARCI_BIN", str(self.anarci_bin))

    def tearDown(self) -> None:
        os.environ["PATH"] = self.old_path
        for key, value in self.env_restore.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def _set_env(self, key: str, value: str) -> None:
        if key not in self.env_restore:
            self.env_restore[key] = os.environ.get(key)
        os.environ[key] = value

    def _fake_executable(self, path: Path) -> None:
        path.write_text("#!/usr/bin/env bash\necho fixture\n")
        path.chmod(0o755)

    def _esm_runtime_text(self) -> str:
        return "\n".join(
            [
                "export FOLDING_BACKEND=esmfold2",
                "export EXECUTION_MODE=slurm",
                "export SLURM_PARTITION=batch",
                "export SLURM_GPUS=1",
                "export SLURM_CPUS_PER_TASK=8",
                "export SLURM_MEM=20G",
                "export SLURM_TIME=08:00:00",
                "export SLURM_MAX_CONCURRENT=1",
                "export FOLD_WORKERS_PER_GPU=1",
                "export DOCKQ_HIGH_QUALITY_THRESHOLD=0.80",
                f"export PROTEINMPNN_HOME={self.proteinmpnn_home}",
                f"export PROTEINMPNN_PYTHON={sys.executable}",
                f"export PROTEINMPNN_RUNNER={self.proteinmpnn_runner}",
                f"export ESMFOLD2_PYTHON={sys.executable}",
                f"export ESM_ROOT={self.esm_root}",
                f"export ESMFOLD2_CACHE={self.esm_cache}",
                f"export ESMC_MODEL_PATH={self.esmc_model}",
                "export MSA_PROVIDER=abforge_get_or_build",
                "export MSA_REQUIRE=true",
                "export MSA_BUILD_IF_MISSING=false",
                f"export MSA_CACHE_ROOT={self.msa_cache}",
                "export ROSETTA_RUNTIME=apptainer",
                "export ROSETTA_IMAGE=docker://rosettacommons/rosetta:latest",
                f"export APPTAINER_BIN={self.bin / 'apptainer'}",
                f"export APPTAINER_CACHEDIR={self.apptainer_cache_parent / 'apptainer'}",
                "export DOCKQ_PROVIDER=dockq_rs_python",
                f"export DOCKQ_PYTHON={sys.executable}",
                f"export ANARCI_PYTHON={sys.executable}",
                f"export ANARCI_BIN={self.anarci_bin}",
                "",
            ]
        )

    def _boltz_runtime_text(self) -> str:
        return "\n".join(
            [
                line.replace("export FOLDING_BACKEND=esmfold2", "export FOLDING_BACKEND=boltz2")
                for line in self._esm_runtime_text().splitlines()
                if not line.startswith("export ESM")
            ]
            + [
                f"export BOLTZ_EXECUTABLE={self.boltz}",
                f"export BOLTZ_PYTHON={sys.executable}",
                f"export BOLTZ_MODEL_ROOT={self.boltz_model}",
                f"export BOLTZ_CACHE={self.boltz_cache_parent / 'boltz'}",
                "",
            ]
        )

    def preflight_runner(self, argv, **kwargs):
        argv = list(argv)
        if argv and argv[0] == "scontrol":
            return subprocess.CompletedProcess(argv, 0, stdout="PartitionName=batch\n", stderr="")
        if argv and argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.run(argv, **kwargs)

    def submit_args(self) -> list[str]:
        return [
            "submit",
            "--paths-config",
            str(self.paths_config),
            "--runtime-config",
            str(self.runtime_config),
            "--input-kind",
            "tarballs",
            "--input-dir",
            str(self.raw),
            "--output-root",
            str(self.output),
            "--decoy-config",
            str(self.decoy_config),
            "--max-structures",
            "5",
            "--mode-policy",
            "auto_by_length",
            "--run-id",
            "unit",
            "--label",
            "fixture",
            "--role",
            "candidate",
            "--confidence",
            "manual",
        ]

    def test_submit_writes_eight_stage_afterok_chain_and_scripts(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout=f"Submitted batch job {100 + len(calls)}\n", stderr="")

        args = fett.build_parser().parse_args(self.submit_args())
        rc = fett.submit_from_args(args, runner=fake_runner, preflight_runner=self.preflight_runner)
        self.assertEqual(0, rc)
        sbatch_calls = [call for call in calls if call[0] == "sbatch"]
        self.assertEqual(8, len(sbatch_calls))
        self.assertFalse(any(part.startswith("--dependency") for part in sbatch_calls[0]))
        for idx, job_id in enumerate(range(101, 108), start=1):
            self.assertIn(f"--dependency=afterok:{job_id}", sbatch_calls[idx])

        scripts = sorted((self.output / "work" / "fett" / "slurm").glob("*.sbatch"))
        self.assertEqual(8, len(scripts))
        by_name = {script.stem: script.read_text() for script in scripts}
        for stem, body in by_name.items():
            if stem == "fold":
                self.assertIn("#SBATCH --gres=gpu:1", body)
            else:
                self.assertNotIn("#SBATCH --gres", body)
        self.assertIn("#SBATCH --mem=24G", by_name["native"])
        self.assertIn("#SBATCH --mem=16G", by_name["prepare"])
        self.assertIn("#SBATCH --mem=16G", by_name["proteinmpnn"])
        self.assertIn("#SBATCH --mem=12G", by_name["sequence_validation"])
        self.assertIn("#SBATCH --mem=24G", by_name["msa_resolution"])
        self.assertIn("#SBATCH --mem=24G", by_name["fold"])
        self.assertIn("#SBATCH --mem=12G", by_name["fold_qc"])
        self.assertLess(by_name["native"].index("export JANGO_RUN_NAME="), by_name["native"].index(f"source {self.paths_config}"))
        self.assertIn(f"export JANGO_RAW_TARBALLS={self.raw}", by_name["native"])
        self.assertIn("nanobody-pipeline", by_name["native"])
        self.assertIn("decoy-prefold", by_name["native"])
        self.assertIn("--limit 5", by_name["native"])
        self.assertIn("--max-structures 5", by_name["native"])
        self.assertNotIn("--case-count", by_name["native"])
        self.assertIn("--experiment esmfold2_max5_auto_by_length", by_name["native"])
        self.assertIn("decoy-fold", by_name["prepare"])
        self.assertIn("--fett-stage prepare", by_name["prepare"])
        self.assertIn(str(self.output / "data" / "manifests" / "decoy" / "esmfold2_max5_auto_by_length"), by_name["prepare"])
        self.assertNotIn(str(self.output / "data" / "manifests" / "decoy" / "unit"), by_name["prepare"])
        self.assertIn("--fett-stage proteinmpnn", by_name["proteinmpnn"])
        self.assertIn("export CUDA_VISIBLE_DEVICES=""", by_name["proteinmpnn"])
        self.assertIn("--fett-stage sequence_validation", by_name["sequence_validation"])
        self.assertIn("--fett-stage msa_resolution", by_name["msa_resolution"])
        self.assertIn("jango fold-jobs prepare-bundle", by_name["fold"])
        self.assertNotIn("jango fold-jobs submit", by_name["fold"])
        self.assertNotIn("--dry-run", by_name["fold"])
        self.assertIn("jango fold-jobs run-bundle", by_name["fold"])
        self.assertNotIn("jango fold-qc", by_name["fold"])
        self.assertIn("jango fold-qc", by_name["fold_qc"])
        self.assertIn("decoy-fold-relax", by_name["analysis"])
        self.assertIn("decoy-results", by_name["analysis"])
        self.assertNotIn("decoy-analyze", by_name["analysis"])
        self.assertLess(
            by_name["analysis"].index("decoy-fold-relax"),
            by_name["analysis"].index("decoy-results"),
        )
        self.assertIn("jango rosetta-qc", by_name["analysis"])
        self.assertIn("fett export-final", by_name["analysis"])

    def test_repository_production_paths_config_is_sourceable_without_cli_values(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        config = repo / "configs" / "production" / "paths.env"
        env = dict(os.environ)
        for key in ["JANGO_RUN_NAME", "JANGO_RUN_ROOT", "JANGO_OUTPUT_ROOT", "JANGO_RAW_TARBALLS"]:
            env.pop(key, None)
        env["USER"] = "demo"

        bare = subprocess.run(
            ["bash", "-lc", f"set -euo pipefail; source {config}; test -z \"${{JANGO_RUN_ROOT+x}}\"; test \"$JANGO_RAW_TARBALLS\" = /data/demo/jango/data/tarballs"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(0, bare.returncode, bare.stderr)

        resolved = subprocess.run(
            [
                "bash",
                "-lc",
                f"set -euo pipefail; export JANGO_RUN_NAME=demo-run; export JANGO_RAW_TARBALLS=/data/demo/jango/data/tarballs; source {config}; "
                "test \"$JANGO_RUN_ROOT\" = /data/demo/jango/work/demo-run; "
                "test \"$JANGO_OUTPUT_ROOT\" = \"$JANGO_RUN_ROOT\"; "
                "test \"$JANGO_RAW_TARBALLS\" = /data/demo/jango/data/tarballs",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(0, resolved.returncode, resolved.stderr)

    def test_preflight_resolves_cli_env_before_sourcing_configs(self) -> None:
        args = fett.build_parser().parse_args(["preflight", *self.submit_args()[1:]])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertFalse(report.failed, report.render())
        self.assertEqual(str(self.raw), report.resolved_config["JANGO_RAW_TARBALLS"])
        script_text = (self.output / "work" / "fett" / "slurm" / "native.sbatch").read_text()
        self.assertLess(script_text.index("export JANGO_RUN_NAME="), script_text.index(f"source {self.paths_config}"))
        self.assertNotIn("CONFIG_REQUIRED_VARIABLE", {issue.code for issue in report.issues})

    def test_preflight_json_is_machine_readable(self) -> None:
        args = fett.build_parser().parse_args(["preflight", *self.submit_args()[1:], "--json"])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        payload = json.loads(report.to_json())
        self.assertEqual("PASSED", payload["status"])
        self.assertEqual("unit", payload["run_id"])
        self.assertIn("JANGO_RAW_TARBALLS", payload["resolved_config"])

    def test_submit_blocks_missing_decoy_config_before_sbatch(self) -> None:
        args_list = self.submit_args()
        idx = args_list.index("--decoy-config") + 1
        args_list[idx] = str(self.configs / "missing.yml")
        args = fett.build_parser().parse_args(args_list)

        def no_sbatch(argv, **kwargs):
            raise AssertionError("sbatch should not be called after fatal preflight")

        rc = fett.submit_from_args(args, runner=no_sbatch, preflight_runner=self.preflight_runner)
        self.assertEqual(1, rc)

    def test_placeholder_decoy_config_fails_preflight(self) -> None:
        args_list = self.submit_args()
        idx = args_list.index("--decoy-config") + 1
        args_list[idx] = "<your current decoy config>"
        args = fett.build_parser().parse_args(["preflight", *args_list[1:]])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertTrue(report.failed)
        self.assertIn("PLACEHOLDER_VALUE", {issue.code for issue in report.issues})

    def test_malformed_apptainer_cache_path_fails_preflight(self) -> None:
        self.runtime_config.write_text(
            self._esm_runtime_text()
            + "export APPTAINER_CACHEDIR=/data/demo/jango/cache/apptainer}/jango/cache/apptainer}\n"
        )
        args = fett.build_parser().parse_args(["preflight", *self.submit_args()[1:]])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertTrue(report.failed)
        self.assertIn("MALFORMED_PATH", {issue.code for issue in report.issues})

    def test_cache_only_msa_does_not_require_search_tools(self) -> None:
        for name in ("colabfold_search", "mmseqs"):
            (self.bin / name).unlink()
        args = fett.build_parser().parse_args(["preflight", *self.submit_args()[1:]])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertFalse(report.failed, report.render())
        self.assertNotIn("MSA_SCRIPT", "\n".join(issue.summary for issue in report.issues))

    def test_missing_msa_builder_fails_when_build_enabled(self) -> None:
        self.runtime_config.write_text(
            self._esm_runtime_text().replace("export MSA_BUILD_IF_MISSING=false", "export MSA_BUILD_IF_MISSING=true")
        )
        for name in ("colabfold_search", "mmseqs"):
            path = self.bin / name
            if path.exists():
                path.unlink()
        args = fett.build_parser().parse_args(["preflight", *self.submit_args()[1:]])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertTrue(report.failed)
        self.assertIn("MISSING_PATH", {issue.code for issue in report.issues})
        self.assertIn("COMMAND_NOT_FOUND", {issue.code for issue in report.issues})


    def test_stage_resource_memory_cap_uses_binary_slurm_units(self) -> None:
        self.runtime_config.write_text(self._esm_runtime_text() + "export FETT_MSA_MEM=25G\n")
        args = fett.build_parser().parse_args(["preflight", *self.submit_args()[1:]])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertTrue(report.failed)
        self.assertIn("RESOURCE_MEMORY_LIMIT", {issue.code for issue in report.issues})

        self.runtime_config.write_text(self._esm_runtime_text() + "export FETT_MSA_MEM=24576M\n")
        args = fett.build_parser().parse_args(["preflight", *self.submit_args()[1:]])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertFalse(report.failed, report.render())

        self.runtime_config.write_text(self._esm_runtime_text() + "export FETT_MSA_MEM=24577M\n")
        args = fett.build_parser().parse_args(["preflight", *self.submit_args()[1:]])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertTrue(report.failed)
        self.assertIn("RESOURCE_MEMORY_LIMIT", {issue.code for issue in report.issues})

    def test_normal_submit_help_hides_resource_controls(self) -> None:
        import io

        parser = fett.build_parser()
        submit = parser._subparsers._group_actions[0].choices["submit"]  # type: ignore[attr-defined]
        stream = io.StringIO()
        submit.print_help(stream)
        help_text = stream.getvalue()
        self.assertNotIn("FETT_", help_text)
        self.assertNotIn("SLURM_CPUS", help_text)
        self.assertNotIn("memory", help_text.lower())
        self.assertIn("--input", help_text)
        self.assertIn("--run-id", help_text)
        self.assertIn("--max-structures", help_text)
        self.assertNotIn("--case-count", help_text)

    def test_legacy_four_stage_manifest_is_reported_not_silently_expanded(self) -> None:
        old_root = self.root / "legacy-stage-schema"
        manifest = {
            "paths_config": str(self.paths_config),
            "runtime_config": str(self.runtime_config),
            "input_kind": "tarballs",
            "input_dir": str(self.raw),
            "metadata": "",
            "decoy_config": str(self.decoy_config),
            "case_count": 5,
            "mode_policy": "auto_by_length",
            "modes": [],
            "backend": "esmfold2",
            "experiment": "esmfold2_max5_auto_by_length",
            "experiment_key": "esmfold2_max5_auto_by_length",
            "run_id": "legacy-stage-schema",
            "source": "user_input",
            "label": "legacy-stage-schema",
            "role": "validation",
            "confidence": "medium",
            "max_decoy_sequences": 5,
            "stages": [{"name": "decoy_redesign", "status": "COMPLETED", "job_id": "22"}],
        }
        args = argparse.Namespace(output_root=old_root)
        manifest_args = fett._args_from_manifest(args, manifest)
        _, report = fett._run_preflight(manifest_args, manifest=manifest, runner=self.preflight_runner)
        codes = {issue.code for issue in report.issues}
        self.assertIn("LEGACY_STAGE_SCHEMA", codes)
        detail = "\n".join(issue.details or "" for issue in report.issues if issue.code == "LEGACY_STAGE_SCHEMA")
        self.assertIn("decoy_prepare", detail)
        self.assertIn("msa_resolution", detail)

    def test_boltz_runtime_preflight_is_supported(self) -> None:
        self.runtime_config.write_text(self._boltz_runtime_text())
        args_list = self.submit_args()
        args = fett.build_parser().parse_args(["preflight", *args_list[1:], "--backend", "boltz2"])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertFalse(report.failed, report.render())
        self.assertEqual("boltz2", report.resolved_config["FOLDING_BACKEND"])

    def test_minimal_submit_run_id_and_experiment_key_do_not_diverge(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout=f"Submitted batch job {700 + len(calls)}\n", stderr="")

        args = fett.build_parser().parse_args(
            ["submit", "--input", str(self.raw), "--run-id", "validation_50", "--max-structures", "50"]
        )
        rc = fett.submit_from_args(args, runner=fake_runner, preflight_runner=self.preflight_runner)
        self.assertEqual(0, rc)
        run_root = self.user_root / "work" / "validation_50"
        experiment = "esmfold2_max50_auto_by_length"
        manifest = json.loads((run_root / "data" / "manifests" / "fett_run_manifest.json").read_text())
        self.assertEqual("validation_50", manifest["run_id"])
        self.assertEqual(experiment, manifest["experiment"])
        self.assertEqual(experiment, manifest["experiment_key"])

        native_script = (run_root / "work" / "fett" / "slurm" / "native.sbatch").read_text()
        prepare_script = (run_root / "work" / "fett" / "slurm" / "prepare.sbatch").read_text()
        prefold_dir = run_root / "data" / "manifests" / "decoy" / experiment
        wrong_dir = run_root / "data" / "manifests" / "decoy" / "validation_50"
        self.assertIn(f"--experiment {experiment}", native_script)
        self.assertIn(str(prefold_dir), prepare_script)
        self.assertNotIn(str(wrong_dir), prepare_script)

    def test_case_count_alias_maps_to_max_structures(self) -> None:
        args = fett.build_parser().parse_args(["preflight", "--input", str(self.raw), "--run-id", "legacy_alias", "--case-count", "7"])
        ctx, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertFalse(report.failed, report.render())
        self.assertEqual(7, ctx.args.max_structures)
        self.assertEqual(7, ctx.args.case_count)
        self.assertEqual("esmfold2_max7_auto_by_length", ctx.experiment)

    def test_pipeline_contract_catches_mismatched_artifact_paths(self) -> None:
        report = fett.PreflightReport(run_id="unit")
        contracts = [
            fett.ArtifactContract(
                artifact="prefold manifest",
                producer="decoy-prefold",
                consumer="decoy-fold",
                producer_path="/run/data/manifests/decoy/esmfold2_50_auto_by_length/prefold_manifest.json",
                consumer_path="/run/data/manifests/decoy/validation_50/prefold_manifest.json",
                canonical_path="/run/data/manifests/decoy/esmfold2_50_auto_by_length/prefold_manifest.json",
            )
        ]
        fett.validate_artifact_contracts(report, contracts)
        self.assertTrue(report.failed)
        issue = report.issues[0]
        self.assertEqual("PIPELINE_CONTRACT_MISMATCH", issue.code)
        self.assertIn("validation_50", issue.details or "")

    def test_no_sbatch_when_pipeline_contract_is_inconsistent(self) -> None:
        original = fett._artifact_contracts

        def mismatched(output_root, experiment, *, backend):
            return [
                fett.ArtifactContract(
                    artifact="prefold manifest",
                    producer="decoy-prefold",
                    consumer="decoy-fold",
                    producer_path=str(output_root / "data" / "manifests" / "decoy" / "esmfold2_max5_auto_by_length" / "prefold_manifest.json"),
                    consumer_path=str(output_root / "data" / "manifests" / "decoy" / "unit" / "prefold_manifest.json"),
                    canonical_path=str(output_root / "data" / "manifests" / "decoy" / "esmfold2_max5_auto_by_length" / "prefold_manifest.json"),
                )
            ]

        fett._artifact_contracts = mismatched
        try:
            args = fett.build_parser().parse_args(self.submit_args())

            def no_sbatch(argv, **kwargs):
                raise AssertionError("sbatch should not be called when the pipeline contract fails")

            rc = fett.submit_from_args(args, runner=no_sbatch, preflight_runner=self.preflight_runner)
            self.assertEqual(1, rc)
        finally:
            fett._artifact_contracts = original

    def test_resume_uses_legacy_prefold_metadata_when_stored_experiment_is_run_id(self) -> None:
        old_root = self.root / "legacy"
        experiment = "esmfold2_50_auto_by_length"
        prefold_dir = old_root / "data" / "manifests" / "decoy" / experiment
        prefold_dir.mkdir(parents=True)
        (prefold_dir / "prefold_manifest.json").write_text(json.dumps({"experiment": experiment, "backend": "esmfold2"}))
        manifest = {
            "paths_config": str(self.paths_config),
            "runtime_config": str(self.runtime_config),
            "input_kind": "tarballs",
            "input_dir": str(self.raw),
            "metadata": "",
            "decoy_config": str(self.decoy_config),
            "case_count": 50,
            "mode_policy": "auto_by_length",
            "modes": [],
            "backend": "esmfold2",
            "experiment": "validation_50",
            "run_id": "validation_50",
            "source": "user_input",
            "label": "validation_50",
            "role": "validation",
            "confidence": "medium",
            "max_decoy_sequences": 5,
        }
        args = argparse.Namespace(output_root=old_root)
        recovered = fett._args_from_manifest(args, manifest)
        self.assertEqual(experiment, recovered.experiment)

    def test_minimal_submit_resolves_installation_defaults_and_run_layout(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout=f"Submitted batch job {200 + len(calls)}\n", stderr="")

        args = fett.build_parser().parse_args(["submit", "--input", str(self.raw), "--run-id", "validation_50", "--max-structures", "5"])
        rc = fett.submit_from_args(args, runner=fake_runner, preflight_runner=self.preflight_runner)
        self.assertEqual(0, rc)
        run_root = self.user_root / "work" / "validation_50"
        self.assertTrue((run_root / "run.json").exists())
        self.assertTrue((run_root / "resolved_config.json").exists())
        self.assertTrue((run_root / "provenance.json").exists())
        self.assertTrue((run_root / "inputs" / "decoy_config.yml").exists())
        self.assertEqual(run_root, Path(json.loads((run_root / "run.json").read_text())["output_root"]))
        resolved = json.loads((run_root / "resolved_config.json").read_text())
        self.assertEqual("esmfold2", resolved["FOLDING_BACKEND"])
        self.assertEqual(str(self.user_root / "cache" / "apptainer"), resolved["APPTAINER_CACHEDIR"])
        script = run_root / "work" / "fett" / "slurm" / "native.sbatch"
        text = script.read_text()
        self.assertIn(f"export APPTAINER_CACHEDIR={self.user_root / 'cache' / 'apptainer'}", text)
        self.assertIn("configs/production/paths.env", text)
        self.assertIn("configs/production/esmfold2.env", text)
        self.assertEqual(8, len([call for call in calls if call[0] == "sbatch"]))

    def test_minimal_boltz_alias_selects_boltz_runtime(self) -> None:
        args = fett.build_parser().parse_args(["preflight", "--input", str(self.raw), "--run-id", "boltz_run", "--max-structures", "5", "--backend", "boltz"])
        ctx, report = fett._run_preflight(args, runner=self.preflight_runner)
        self.assertFalse(report.failed, report.render())
        self.assertEqual("boltz2", ctx.runtime.backend)
        self.assertTrue(str(ctx.runtime_config).endswith("configs/production/boltz2.env"))

    def test_input_kind_detection_and_ambiguous_input(self) -> None:
        pdb_dir = self.root / "pdbs"
        pdb_dir.mkdir()
        (pdb_dir / "1abc.pdb").write_text("ATOM\n")
        self.assertEqual("sabdab-pdb", fett.detect_input_kind(pdb_dir))
        cif_dir = self.root / "cifs"
        cif_dir.mkdir()
        (cif_dir / "1abc.cif").write_text("data_1abc\n")
        self.assertEqual("sabdab2-cif", fett.detect_input_kind(cif_dir))
        (pdb_dir / "1abc.tar.gz").write_bytes(b"fixture")
        with self.assertRaises(ValueError):
            fett.detect_input_kind(pdb_dir)

    def test_invalid_run_id_and_conflicting_input_aliases_fail_preflight(self) -> None:
        other = self.root / "other_raw"
        other.mkdir()
        (other / "2def.tar.gz").write_bytes(b"fixture")
        args = fett.build_parser().parse_args([
            "preflight",
            "--input",
            str(self.raw),
            "--input-dir",
            str(other),
            "--run-id",
            "bad/run",
            "--max-structures",
            "5",
            "--output-root",
            str(self.root / "bad-run"),
        ])
        _, report = fett._run_preflight(args, runner=self.preflight_runner)
        codes = {issue.code for issue in report.issues}
        self.assertIn("INPUT_ALIAS_CONFLICT", codes)
        self.assertIn("RUN_ID_INVALID", codes)

    def test_run_management_commands_accept_run_id(self) -> None:
        args = fett.build_parser().parse_args(["submit", "--input", str(self.raw), "--run-id", "managed", "--max-structures", "5"])
        ids = iter(["301", "302", "303", "304", "305", "306", "307", "308"])
        fett.submit_from_args(
            args,
            runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=f"Submitted batch job {next(ids)}\n", stderr=""),
            preflight_runner=self.preflight_runner,
        )
        run_root = self.user_root / "work" / "managed"
        (run_root / "logs" / "slurm" / "fett-managed-native-301.out").write_text("native log\n")

        status_calls: list[list[str]] = []

        def status_runner(argv, **kwargs):
            status_calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="RUNNING\n", stderr="")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        self.assertEqual(0, fett.status_from_args(fett.build_parser().parse_args(["status", "managed"]), runner=status_runner))
        self.assertEqual(0, fett.logs_from_args(fett.build_parser().parse_args(["logs", "managed", "--lines", "1"])))

        cancel_calls: list[list[str]] = []

        def cancel_runner(argv, **kwargs):
            cancel_calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="RUNNING\n", stderr="")
            if argv[0] == "scancel":
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        self.assertEqual(0, fett.cancel_from_args(fett.build_parser().parse_args(["cancel", "managed"]), runner=cancel_runner))
        self.assertEqual(["301", "302", "303", "304", "305", "306", "307", "308"], [call[1] for call in cancel_calls if call[0] == "scancel"])

    def test_resume_does_not_duplicate_active_jobs(self) -> None:
        args = fett.build_parser().parse_args(self.submit_args())
        fett.submit_from_args(
            args,
            runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="Submitted batch job 501\n", stderr=""),
            preflight_runner=self.preflight_runner,
        )
        calls: list[list[str]] = []

        def active_runner(argv, **kwargs):
            calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="RUNNING\n", stderr="")
            raise AssertionError("resume should not submit while an existing stage is active")

        resume_args = fett.build_parser().parse_args(["resume", "--output-root", str(self.output)])
        fett.resume_from_args(resume_args, runner=active_runner, preflight_runner=self.preflight_runner)
        self.assertFalse(any(call[0] == "sbatch" for call in calls))

    def test_resume_restarts_from_failed_stage_and_rechains_downstream(self) -> None:
        args = fett.build_parser().parse_args(self.submit_args())
        ids = iter(["1", "2", "3", "4", "5", "6", "7", "8"])
        fett.submit_from_args(
            args,
            runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=f"Submitted batch job {next(ids)}\n", stderr=""),
            preflight_runner=self.preflight_runner,
        )
        calls: list[list[str]] = []
        new_ids = iter(["11", "12", "13", "14", "15", "16", "17", "18"])

        def resume_runner(argv, **kwargs):
            calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if argv[0] == "sacct":
                job_id = argv[2]
                state = "FAILED|1:0" if job_id == "1" else "CANCELLED|1:0"
                return subprocess.CompletedProcess(argv, 0, stdout=state + "\n", stderr="")
            if argv[0] == "sbatch":
                return subprocess.CompletedProcess(argv, 0, stdout=f"Submitted batch job {next(new_ids)}\n", stderr="")
            raise AssertionError(argv)

        resume_args = fett.build_parser().parse_args(["resume", "--output-root", str(self.output)])
        fett.resume_from_args(resume_args, runner=resume_runner, preflight_runner=self.preflight_runner)
        sbatch_calls = [call for call in calls if call[0] == "sbatch"]
        self.assertEqual(8, len(sbatch_calls))
        self.assertFalse(any(part.startswith("--dependency") for part in sbatch_calls[0]))
        for idx, job_id in enumerate(range(11, 18), start=1):
            self.assertIn(f"--dependency=afterok:{job_id}", sbatch_calls[idx])

    def test_export_final_copies_relaxed_pdbs_tables_and_figures(self) -> None:
        experiment = "esmfold2_max5_auto_by_length"
        all_tables = self.output / "results" / "decoy" / "tables" / experiment / "all_modes"
        landscape = all_tables / "landscape"
        landscape.mkdir(parents=True)
        fold_pdb = self.output / "work" / "decoy" / experiment / "full_antigen" / "relaxed" / "d1_relaxed.pdb"
        fold_pdb.parent.mkdir(parents=True)
        fold_pdb.write_text("ATOM fixture\n")
        pd.DataFrame(
            [
                {"structure_id": "s1", "design_id": "d1", "redesign_mode": "full_antigen", "relax_status": "completed", "relaxed_pdb": str(fold_pdb)},
                {"structure_id": "s2", "design_id": "d2", "redesign_mode": "hotspot_only", "relax_status": "failed", "relaxed_pdb": ""},
            ]
        ).to_csv(all_tables / "decoy_relax_manifest.csv", index=False)
        pd.DataFrame([{"structure_id": "s1", "design_id": "d1"}]).to_csv(all_tables / "decoy_native_comparison.csv", index=False)
        pd.DataFrame([{"structure_id": "s1", "design_id": "d1", "quadrant": "Q1"}]).to_csv(landscape / "decoy_quadrant_ranking.csv", index=False)
        figure = self.output / "results" / "decoy" / "figures" / experiment / "all_modes" / "landscape" / "delphi_landscape.png"
        figure.parent.mkdir(parents=True)
        figure.write_bytes(b"png")
        (self.output / "data" / "manifests").mkdir(parents=True)
        (self.output / "data" / "manifests" / "fett_run_manifest.json").write_text(
            json.dumps({"run_id": "run1", "experiment": experiment, "stages": [{"name": "native_analysis", "status": "COMPLETED"}]})
        )

        args = fett.build_parser().parse_args(["export-final", "--output-root", str(self.output)])
        fett.export_final_from_args(args)

        export_root = self.user_root / "outputs" / "run1"
        copied_pdb = export_root / "pdbs" / "relaxed_decoys" / fold_pdb.name
        self.assertTrue(copied_pdb.exists())
        exported_manifest = pd.read_csv(export_root / "results" / "tables" / "decoy_relax_manifest.csv")
        self.assertEqual(str(copied_pdb), exported_manifest.loc[0, "relaxed_pdb"])
        self.assertEqual("failed", exported_manifest.loc[1, "relax_status"])
        self.assertTrue((export_root / "results" / "tables" / "decoy_native_comparison.csv").exists())
        self.assertTrue((export_root / "results" / "tables" / "decoy_quadrant_ranking.csv").exists())
        self.assertTrue((export_root / "results" / "figures" / "all_modes" / "landscape" / "delphi_landscape.png").exists())


if __name__ == "__main__":
    unittest.main()
