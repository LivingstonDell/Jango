from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest import mock
import unittest

import pandas as pd

from jango.pipeline.fold_execution import (
    SlurmConfig,
    parse_sbatch_job_id,
    run_bundle_manifest,
    submit_combined_job_bundle,
    submit_job_bundle,
    update_submission_status,
)
from nbia.folding.esmfold2_backend import ESMFold2Backend


FOLD_COMMAND = "<CONDA_ROOT>/envs/esmfold2/bin/python <JANGO_REPO>/scripts/nbia/run_esmfold2_monomer.py --fasta in.fa --out-dir out --msa-a3m query.a3m --require-msa"


class SlurmFoldExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / "work" / "decoy" / "esmfold2_5_auto_by_length" / "full_antigen"
        self.work.mkdir(parents=True)
        self.jobs_tsv = self.work / "job_bundle" / "esmfold2_monomer_jobs.tsv"
        self.jobs_tsv.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "design_id": "s1_native",
                    "structure_id": "s1",
                    "fold_input": str(self.root / "inputs" / "s1.fa"),
                    "fold_output_dir": str(self.root / "outputs" / "s1_native"),
                    "fold_command": FOLD_COMMAND,
                    "fold_backend": "esmfold2",
                }
            ]
        ).to_csv(self.jobs_tsv, sep="\t", index=False)
        self.config = SlurmConfig.from_values(
            output_root=self.root,
            partition="batch",
            gpus=1,
            cpus_per_task=8,
            mem="64G",
            time="08:00:00",
            repo_root=Path("<JANGO_REPO>"),
            paths_config=Path("configs/test/paths.env"),
            runtime_config=Path("configs/test/esmfold2.env"),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_slurm_script_generation_has_required_directives_and_environment(self) -> None:
        manifest = submit_job_bundle(
            jobs_tsv=self.jobs_tsv,
            work_dir=self.work,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=self.config,
            dry_run=True,
        )
        self.assertEqual(manifest.iloc[0]["submission_status"], "dry_run")
        script = Path(manifest.iloc[0]["slurm_script"])
        text = script.read_text()
        self.assertIn("#SBATCH --partition=batch", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("#SBATCH --mem=64G", text)
        self.assertIn("#SBATCH --time=08:00:00", text)
        self.assertIn("#SBATCH --output=" + str(self.root / "logs" / "slurm" / "%x-%j.out"), text)
        self.assertIn("source <CONDA_ROOT>/etc/profile.d/conda.sh", text)
        self.assertIn("conda activate jango", text)
        self.assertIn("cd <JANGO_REPO>", text)
        self.assertIn("source configs/test/paths.env", text)
        self.assertIn("source configs/test/esmfold2.env", text)
        self.assertIn("export NBIA_ROOT=", text)
        self.assertIn("<CONDA_ROOT>/envs/esmfold2/bin/python <JANGO_REPO>/scripts/nbia/run_esmfold2_monomer.py", text)
        self.assertNotIn("${NBIA_ROOT}/scripts/nbia/run_esmfold2_monomer.py", text)

    def test_sbatch_job_id_parsing(self) -> None:
        self.assertEqual(parse_sbatch_job_id("Submitted batch job 12345\n"), "12345")
        with self.assertRaises(ValueError):
            parse_sbatch_job_id("no job here")

    def test_submit_uses_sbatch_and_records_job_id(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if argv[0] == "sbatch":
                return subprocess.CompletedProcess(argv, 0, stdout="Submitted batch job 777\n", stderr="")
            raise AssertionError(argv)

        manifest = submit_job_bundle(
            jobs_tsv=self.jobs_tsv,
            work_dir=self.work,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=self.config,
            runner=fake_runner,
        )
        self.assertIn(["sbatch", str(manifest.iloc[0]["slurm_script"])], calls)
        self.assertEqual(manifest.iloc[0]["slurm_job_id"], "777")
        self.assertEqual(manifest.iloc[0]["submission_status"], "submitted")
        self.assertEqual(
            str(self.root / "logs" / "slurm" / "jg-esmfold2-s1-s1_native-777.out"),
            manifest.iloc[0]["stdout_log"],
        )
        self.assertEqual(
            str(self.root / "logs" / "slurm" / "jg-esmfold2-s1-s1_native-777.err"),
            manifest.iloc[0]["stderr_log"],
        )

    def test_submit_preserves_fold_job_metadata_for_qc(self) -> None:
        pd.DataFrame(
            [
                {
                    "design_id": "s1_design",
                    "structure_id": "s1",
                    "fold_input": str(self.root / "inputs" / "s1.fa"),
                    "fold_output_dir": str(self.root / "outputs" / "s1_design"),
                    "fold_command": FOLD_COMMAND,
                    "fold_backend": "esmfold2",
                    "redesign_mode": "hotspot_only",
                    "sequence_role": "redesigned_decoy",
                }
            ]
        ).to_csv(self.jobs_tsv, sep="	", index=False)

        def fake_runner(argv, **kwargs):
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if argv[0] == "sbatch":
                return subprocess.CompletedProcess(argv, 0, stdout="Submitted batch job 778\n", stderr="")
            raise AssertionError(argv)

        manifest = submit_job_bundle(
            jobs_tsv=self.jobs_tsv,
            work_dir=self.work,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=self.config,
            runner=fake_runner,
        )
        row = manifest.iloc[0]
        self.assertEqual("hotspot_only", row["redesign_mode"])
        self.assertEqual("redesigned_decoy", row["sequence_role"])
        self.assertEqual(str(self.root / "outputs" / "s1_design"), row["fold_output_dir"])
        self.assertIn("run_esmfold2_monomer.py", row["fold_command"])

    def test_running_jobs_are_not_duplicated(self) -> None:
        manifest_path = self.work / "slurm" / "fold_submission_manifest.csv"
        manifest_path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "design_id": "s1_native",
                    "structure_id": "s1",
                    "backend": "esmfold2",
                    "execution_mode": "slurm",
                    "slurm_job_id": "42",
                    "slurm_script": "old.sbatch",
                    "stdout_log": "old.out",
                    "stderr_log": "old.err",
                    "submission_status": "submitted",
                    "completion_status": "RUNNING",
                    "exit_code": "",
                }
            ]
        ).to_csv(manifest_path, index=False)
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="42|RUNNING\n", stderr="")
            raise AssertionError("sbatch should not be called")

        manifest = submit_job_bundle(
            jobs_tsv=self.jobs_tsv,
            work_dir=self.work,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=self.config,
            runner=fake_runner,
        )
        self.assertEqual(manifest.iloc[0]["submission_status"], "skipped_active")
        self.assertFalse(any(call[0] == "sbatch" for call in calls))

    def test_completed_outputs_are_skipped_and_failed_outputs_are_resumable(self) -> None:
        out_dir = self.root / "outputs" / "s1_native"
        out_dir.mkdir(parents=True)
        (out_dir / "prediction.pdb").write_text("ATOM\nEND\n")
        manifest = submit_job_bundle(
            jobs_tsv=self.jobs_tsv,
            work_dir=self.work,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=self.config,
            dry_run=True,
        )
        self.assertEqual(manifest.iloc[0]["submission_status"], "skipped_completed")
        self.assertEqual(str(manifest.iloc[0]["exit_code"]), "0")

        (out_dir / "prediction.pdb").unlink()
        manifest = submit_job_bundle(
            jobs_tsv=self.jobs_tsv,
            work_dir=self.work,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=self.config,
            dry_run=True,
        )
        self.assertEqual(manifest.iloc[0]["submission_status"], "dry_run")
        self.assertEqual(manifest.iloc[0]["completion_status"], "NOT_SUBMITTED")

    def test_status_update_uses_squeue_and_sacct(self) -> None:
        manifest_path = self.work / "slurm" / "fold_submission_manifest.csv"
        manifest_path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {"design_id": "a", "slurm_job_id": "11", "completion_status": "SUBMITTED", "exit_code": ""},
                {"design_id": "b", "slurm_job_id": "12", "completion_status": "SUBMITTED", "exit_code": ""},
            ]
        ).to_csv(manifest_path, index=False)

        def fake_runner(argv, **kwargs):
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="11|RUNNING\n", stderr="")
            if argv[0] == "sacct":
                return subprocess.CompletedProcess(argv, 0, stdout="12|FAILED|1:0\n", stderr="")
            raise AssertionError(argv)

        df = update_submission_status(manifest_path, runner=fake_runner)
        by_id = df.set_index("slurm_job_id")
        self.assertEqual(by_id.loc[11, "completion_status"], "RUNNING")
        self.assertEqual(by_id.loc[12, "completion_status"], "FAILED")
        self.assertEqual(by_id.loc[12, "exit_code"], "1:0")

    def test_esmfold2_command_preserves_configured_python_and_resolves_runner_before_execution(self) -> None:
        repo = self.root / "repo"
        runner = repo / "scripts" / "nbia" / "run_esmfold2_monomer.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("#!/usr/bin/env python\n")
        with mock.patch.dict(
            os.environ,
            {"JANGO_SOURCE": str(repo), "ESMFOLD2_RUNNER": "${JANGO_SOURCE}/scripts/nbia/run_esmfold2_monomer.py"},
            clear=False,
        ):
            backend = ESMFold2Backend(
                esmfold2_python="<CONDA_ROOT>/envs/esmfold2/bin/python",
                esm_root=self.root / "esm",
                model_id_or_path="fixture",
                cache_dir=self.root / "cache",
                device="cpu",
            )
        command = backend.command(self.root / "in.fa", self.root / "out")
        self.assertTrue(command.startswith(f"<CONDA_ROOT>/envs/esmfold2/bin/python {runner}"))
        self.assertNotIn("${JANGO_SOURCE}", command)
        self.assertNotIn("${NBIA_ROOT}", command)


    def test_combined_bundle_can_be_prepared_inside_existing_slurm_job_without_nested_sbatch(self) -> None:
        work_root = self.root / "work" / "decoy" / "esmfold2_5_auto_by_length"
        jobs_tsv = work_root / "mode" / "job_bundle" / "esmfold2_monomer_jobs.tsv"
        jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "design_id": "design_1",
                    "structure_id": "s1",
                    "fold_output_dir": str(self.root / "outputs" / "design_1"),
                    "fold_command": FOLD_COMMAND,
                    "fold_backend": "esmfold2",
                    "sequence_role": "redesigned_decoy",
                    "redesign_mode": "mode",
                }
            ]
        ).to_csv(jobs_tsv, sep="\t", index=False)
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            raise AssertionError("prepare must not call sbatch")

        manifest = submit_combined_job_bundle(
            job_sources=[("mode", jobs_tsv)],
            work_dir=work_root,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=self.config,
            submit=False,
            execution_mode="fett_slurm_bundle",
            current_slurm_job_id="1234",
            runner=fake_runner,
        )
        self.assertFalse(any(call[0] == "sbatch" for call in calls))
        self.assertEqual(manifest.iloc[0]["execution_mode"], "fett_slurm_bundle")
        self.assertEqual(str(manifest.iloc[0]["slurm_job_id"]), "1234")
        self.assertEqual(manifest.iloc[0]["submission_status"], "submitted")
        self.assertEqual(manifest.iloc[0]["completion_status"], "SUBMITTED")
        self.assertNotIn("dry_run", set(manifest["submission_status"].astype(str)))
        self.assertNotIn("NOT_SUBMITTED", set(manifest["completion_status"].astype(str)))
        self.assertEqual(len(list((work_root / "slurm").glob("*.sbatch"))), 0)
        bundle_tsv = work_root / "slurm" / "esmfold2_5_auto_by_length-fold-bundle_jobs.tsv"
        self.assertEqual(len(pd.read_csv(bundle_tsv, sep="\t")), 1)

    def test_combined_bundle_generates_one_sbatch_and_excludes_completed_native_controls(self) -> None:
        work_root = self.root / "work" / "decoy" / "esmfold2_5_auto_by_length"
        job_sources: list[tuple[str, Path]] = []
        native_by_mode = {"full_antigen": 1, "hotspot_only": 1, "interface_only": 3}
        decoy_by_mode = {"full_antigen": 5, "hotspot_only": 5, "interface_only": 15}
        for mode, native_count in native_by_mode.items():
            rows = []
            for idx in range(native_count):
                design_id = f"{mode}_native_{idx}"
                out_dir = self.root / "outputs" / mode / design_id
                out_dir.mkdir(parents=True)
                (out_dir / "prediction.pdb").write_text("ATOM\nEND\n")
                rows.append(
                    {
                        "design_id": design_id,
                        "structure_id": f"{mode}_s{idx}",
                        "fold_output_dir": str(out_dir),
                        "fold_command": FOLD_COMMAND,
                        "fold_backend": "esmfold2",
                        "sequence_role": "native_control",
                        "redesign_mode": mode,
                    }
                )
            for idx in range(decoy_by_mode[mode]):
                design_id = f"{mode}_mpnn_{idx:04d}"
                rows.append(
                    {
                        "design_id": design_id,
                        "structure_id": f"{mode}_s{idx}",
                        "fold_output_dir": str(self.root / "outputs" / mode / design_id),
                        "fold_command": FOLD_COMMAND,
                        "fold_backend": "esmfold2",
                        "sequence_role": "redesigned_decoy",
                        "redesign_mode": mode,
                    }
                )
            jobs_tsv = work_root / mode / "job_bundle" / "esmfold2_monomer_jobs.tsv"
            jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(jobs_tsv, sep="\t", index=False)
            job_sources.append((mode, jobs_tsv))

        config = SlurmConfig.from_values(
            output_root=self.root,
            partition="batch",
            gpus=1,
            cpus_per_task=8,
            mem="20G",
            time="08:00:00",
            repo_root=Path("<JANGO_REPO>"),
            paths_config=Path("configs/test/paths.env"),
            runtime_config=Path("configs/test/esmfold2.env"),
            max_concurrent=1,
            workers_per_gpu=1,
        )
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            if argv[0] == "sbatch":
                return subprocess.CompletedProcess(argv, 0, stdout="Submitted batch job 888\n", stderr="")
            raise AssertionError(argv)

        manifest = submit_combined_job_bundle(
            job_sources=job_sources,
            work_dir=work_root,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=config,
            runner=fake_runner,
        )
        self.assertEqual(sum(1 for call in calls if call[0] == "sbatch"), 1)
        self.assertEqual(manifest["submission_status"].value_counts().to_dict(), {"submitted": 25, "skipped_completed": 5})
        self.assertNotIn("dry_run", set(manifest["submission_status"].astype(str)))
        self.assertNotIn("NOT_SUBMITTED", set(manifest["completion_status"].astype(str)))
        submitted = manifest[manifest["submission_status"] == "submitted"]
        self.assertEqual(set(submitted["slurm_job_id"].astype(str)), {"888"})
        self.assertEqual(set(submitted["sequence_role"]), {"redesigned_decoy"})
        self.assertEqual(len(set(submitted["slurm_script"])), 1)
        script = Path(submitted.iloc[0]["slurm_script"])
        text = script.read_text()
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("#SBATCH --mem=20G", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("export SLURM_MAX_CONCURRENT=1", text)
        self.assertIn("export FOLD_WORKERS_PER_GPU=1", text)
        self.assertIn("jango fold-jobs run-bundle --manifest", text)
        self.assertNotIn("run_esmfold2_monomer.py --fasta", text)
        bundle_tsv = script.parent / "esmfold2_5_auto_by_length-fold-bundle_jobs.tsv"
        self.assertEqual(len(pd.read_csv(bundle_tsv, sep="\t")), 25)

    def test_esmfold2_bundle_uses_one_batch_runner_and_keeps_per_design_status(self) -> None:
        work_root = self.root / "work" / "decoy" / "experiment"
        manifest_path = work_root / "slurm" / "fold_submission_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        repo = self.root / "repo"
        single_runner = repo / "scripts" / "nbia" / "run_esmfold2_monomer.py"
        batch_runner = repo / "scripts" / "nbia" / "run_esmfold2_monomer_jobs.py"
        batch_runner.parent.mkdir(parents=True, exist_ok=True)
        single_runner.write_text("#!/usr/bin/env python\n")
        batch_runner.write_text("#!/usr/bin/env python\n")
        source_jobs_tsv = work_root / "mode" / "job_bundle" / "esmfold2_monomer_jobs.tsv"
        source_jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        source_rows = []
        for design_id in ["a", "b", "c"]:
            out_dir = self.root / "outputs" / design_id
            fasta = self.root / "inputs" / f"{design_id}.fa"
            source_rows.append(
                {
                    "design_id": design_id,
                    "structure_id": "s1",
                    "fold_input": str(fasta),
                    "fold_output_dir": str(out_dir),
                    "fold_command": (
                        f"/opt/esmfold2/bin/python {single_runner} "
                        f"--fasta {fasta} --out-dir {out_dir} --model-id-or-path fixture-model "
                        f"--cache-dir {self.root / 'cache'} --device cuda --esm-root {self.root / 'esm'} "
                        "--msa-a3m query.a3m --require-msa"
                    ),
                    "fold_backend": "esmfold2",
                    "redesign_mode": "mode",
                    "sequence_role": "redesigned_decoy",
                    "msa_path": "query.a3m",
                }
            )
            rows.append(
                {
                    "design_id": design_id,
                    "structure_id": "s1",
                    "backend": "esmfold2",
                    "execution_mode": "slurm_bundle",
                    "slurm_job_id": "999",
                    "stdout_log": str(self.root / "logs" / f"{design_id}.out"),
                    "stderr_log": str(self.root / "logs" / f"{design_id}.err"),
                    "submission_status": "submitted",
                    "completion_status": "SUBMITTED",
                    "fold_output_dir": str(out_dir),
                    "fold_command": (
                        f"/opt/esmfold2/bin/python {single_runner} "
                        f"--fasta {fasta} --out-dir {out_dir} --model-id-or-path fixture-model "
                        f"--cache-dir {self.root / 'cache'} --device cuda --esm-root {self.root / 'esm'} "
                        "--msa-a3m query.a3m --require-msa"
                    ),
                    "redesign_mode": "mode",
                    "sequence_role": "redesigned_decoy",
                    "source_jobs_tsv": str(source_jobs_tsv),
                }
            )
        pd.DataFrame(source_rows).to_csv(source_jobs_tsv, sep="	", index=False)
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            jobs_tsv = Path(argv[argv.index("--jobs-tsv") + 1])
            status_csv = Path(argv[argv.index("--status-csv") + 1])
            jobs = pd.read_csv(jobs_tsv, sep="\t")
            status_rows = []
            for _, row in jobs.iterrows():
                design_id = str(row["design_id"])
                out_dir = Path(str(row["fold_output_dir"]))
                if design_id != "b":
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "prediction.pdb").write_text("ATOM\nEND\n")
                    status = "COMPLETED"
                    exit_code = "0"
                else:
                    status = "FAILED"
                    exit_code = "1"
                status_rows.append(
                    {
                        "design_id": design_id,
                        "structure_id": row.get("structure_id", ""),
                        "redesign_mode": row.get("redesign_mode", ""),
                        "sequence_role": row.get("sequence_role", ""),
                        "fold_output_dir": str(out_dir),
                        "completion_status": status,
                        "exit_code": exit_code,
                        "error": "",
                        "msa_path": "query.a3m",
                    }
                )
            pd.DataFrame(status_rows).to_csv(status_csv, index=False)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        out = run_bundle_manifest(manifest_path, runner=fake_runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "/opt/esmfold2/bin/python")
        self.assertEqual(Path(calls[0][1]), batch_runner)
        self.assertIn("--jobs-tsv", calls[0])
        self.assertIn("--status-csv", calls[0])
        self.assertIn("--model-id-or-path", calls[0])
        self.assertIn("fixture-model", calls[0])
        by_design = out.set_index("design_id")
        self.assertEqual(by_design.loc["a", "completion_status"], "COMPLETED")
        self.assertEqual(by_design.loc["b", "completion_status"], "FAILED")
        self.assertEqual(by_design.loc["c", "completion_status"], "COMPLETED")


    def test_opendde_bundle_uses_one_batch_runner_and_preserves_completed_rows(self) -> None:
        work_root = self.root / "work" / "decoy" / "opendde_experiment"
        manifest_path = work_root / "slurm" / "fold_submission_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        repo = self.root / "repo"
        single_runner = repo / "scripts" / "nbia" / "run_opendde_monomer.py"
        batch_runner = repo / "scripts" / "nbia" / "run_opendde_monomer_jobs.py"
        batch_runner.parent.mkdir(parents=True, exist_ok=True)
        single_runner.write_text("#!/usr/bin/env python\n")
        batch_runner.write_text("#!/usr/bin/env python\n")
        source_jobs_tsv = work_root / "hotspot_only" / "job_bundle" / "opendde_monomer_jobs.tsv"
        source_jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
        source_rows = []
        rows = []
        for design_id in ["done", "todo", "fail"]:
            out_dir = self.root / "outputs" / design_id
            json_path = self.root / "inputs" / f"{design_id}.json"
            command = (
                f"/opt/opendde/bin/python {single_runner} --json {json_path} --out-dir {out_dir} "
                f"--opendde-executable /opt/opendde/bin/opendde --root-dir {self.root / 'models'} "
                "--model-name opendde_v1 --seeds 101 --cycle 10 --step 200 --sample 5 --dtype fp32 "
                "--use-msa true --require-msa"
            )
            source_rows.append(
                {
                    "design_id": design_id,
                    "structure_id": "s1",
                    "fold_input": str(json_path),
                    "fold_output_dir": str(out_dir),
                    "fold_command": command,
                    "fold_backend": "opendde",
                    "redesign_mode": "hotspot_only",
                    "sequence_role": "redesigned_decoy",
                }
            )
            rows.append(
                {
                    "design_id": design_id,
                    "structure_id": "s1",
                    "backend": "opendde",
                    "execution_mode": "slurm_bundle",
                    "slurm_job_id": "1000",
                    "stdout_log": str(self.root / "logs" / f"{design_id}.out"),
                    "stderr_log": str(self.root / "logs" / f"{design_id}.err"),
                    "submission_status": "submitted",
                    "completion_status": "SUBMITTED",
                    "fold_output_dir": str(out_dir),
                    "fold_command": command,
                    "redesign_mode": "hotspot_only",
                    "sequence_role": "redesigned_decoy",
                    "source_jobs_tsv": str(source_jobs_tsv),
                }
            )
        completed = self.root / "outputs" / "done" / "predictions"
        completed.mkdir(parents=True)
        (completed / "done_sample_0.pdb").write_text("ATOM\nEND\n")
        pd.DataFrame(source_rows).to_csv(source_jobs_tsv, sep="\t", index=False)
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        calls: list[list[str]] = []

        def fake_runner(argv, **kwargs):
            calls.append(list(argv))
            jobs_tsv = Path(argv[argv.index("--jobs-tsv") + 1])
            status_csv = Path(argv[argv.index("--status-csv") + 1])
            jobs = pd.read_csv(jobs_tsv, sep="\t")
            self.assertEqual(set(jobs["design_id"].astype(str)), {"todo", "fail"})
            pd.DataFrame(
                [
                    {
                        "design_id": "todo",
                        "structure_id": "s1",
                        "redesign_mode": "hotspot_only",
                        "sequence_role": "redesigned_decoy",
                        "fold_output_dir": str(self.root / "outputs" / "todo"),
                        "completion_status": "COMPLETED",
                        "exit_code": "0",
                        "prediction_count": "5",
                        "error": "",
                    },
                    {
                        "design_id": "fail",
                        "structure_id": "s1",
                        "redesign_mode": "hotspot_only",
                        "sequence_role": "redesigned_decoy",
                        "fold_output_dir": str(self.root / "outputs" / "fail"),
                        "completion_status": "FAILED",
                        "exit_code": "1",
                        "prediction_count": "0",
                        "error": "fixture failure",
                    },
                ]
            ).to_csv(status_csv, index=False)
            out_dir = self.root / "outputs" / "todo" / "predictions"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "todo_sample_0.pdb").write_text("ATOM\nEND\n")
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

        with mock.patch.dict(os.environ, {"OPENDDE_BATCH_WORKERS": "2"}, clear=False):
            out = run_bundle_manifest(manifest_path, runner=fake_runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "/opt/opendde/bin/python")
        self.assertEqual(Path(calls[0][1]), batch_runner)
        self.assertIn("--workers", calls[0])
        self.assertIn("2", calls[0])
        by_design = out.set_index("design_id")
        self.assertEqual(by_design.loc["done", "submission_status"], "skipped_completed")
        self.assertEqual(by_design.loc["done", "completion_status"], "COMPLETED")
        self.assertEqual(by_design.loc["todo", "completion_status"], "COMPLETED")
        self.assertEqual(by_design.loc["fail", "completion_status"], "FAILED")

    def test_combined_bundle_treats_stale_individual_records_as_resumable(self) -> None:
        work_root = self.root / "work" / "decoy" / "experiment"
        jobs_tsv = work_root / "mode" / "job_bundle" / "esmfold2_monomer_jobs.tsv"
        jobs_tsv.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "design_id": "stale_design",
                    "structure_id": "s1",
                    "fold_output_dir": str(self.root / "outputs" / "stale_design"),
                    "fold_command": FOLD_COMMAND,
                    "fold_backend": "esmfold2",
                    "sequence_role": "redesigned_decoy",
                    "redesign_mode": "mode",
                }
            ]
        ).to_csv(jobs_tsv, sep="\t", index=False)
        old_manifest = work_root / "mode" / "slurm" / "fold_submission_manifest.csv"
        old_manifest.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "design_id": "stale_design",
                    "structure_id": "s1",
                    "slurm_job_id": "1442.0",
                    "submission_status": "submitted",
                    "completion_status": "SUBMITTED",
                }
            ]
        ).to_csv(old_manifest, index=False)

        def fake_runner(argv, **kwargs):
            if argv[0] == "squeue":
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            raise AssertionError("dry-run should not call sbatch")

        manifest = submit_combined_job_bundle(
            job_sources=[("mode", jobs_tsv)],
            work_dir=work_root,
            output_root=self.root,
            backend="esmfold2",
            slurm_config=self.config,
            existing_manifest_paths=[old_manifest],
            dry_run=True,
            runner=fake_runner,
        )
        self.assertEqual(manifest.iloc[0]["submission_status"], "dry_run")
        self.assertEqual(manifest.iloc[0]["completion_status"], "NOT_SUBMITTED")

    def test_bundle_runner_executes_sequentially_and_continues_after_failure(self) -> None:
        manifest_path = self.work / "slurm" / "fold_submission_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        order = self.root / "order.txt"
        out_a = self.root / "outputs" / "a"
        out_b = self.root / "outputs" / "b"
        out_c = self.root / "outputs" / "c"

        def success_command(label: str, out_dir: Path) -> str:
            return "; ".join(
                [
                    f"printf {shlex.quote(label)} >> {shlex.quote(str(order))}",
                    f"mkdir -p {shlex.quote(str(out_dir))}",
                    f"printf ATOM > {shlex.quote(str(out_dir / 'prediction.pdb'))}",
                ]
            )

        rows = [
            {
                "design_id": "a",
                "structure_id": "s1",
                "backend": "esmfold2",
                "execution_mode": "slurm_bundle",
                "slurm_job_id": "999",
                "stdout_log": str(self.root / "logs" / "a.out"),
                "stderr_log": str(self.root / "logs" / "a.err"),
                "submission_status": "submitted",
                "completion_status": "SUBMITTED",
                "fold_output_dir": str(out_a),
                "fold_command": success_command("a", out_a),
            },
            {
                "design_id": "b",
                "structure_id": "s1",
                "backend": "esmfold2",
                "execution_mode": "slurm_bundle",
                "slurm_job_id": "999",
                "stdout_log": str(self.root / "logs" / "b.out"),
                "stderr_log": str(self.root / "logs" / "b.err"),
                "submission_status": "submitted",
                "completion_status": "SUBMITTED",
                "fold_output_dir": str(out_b),
                "fold_command": f"printf b >> {shlex.quote(str(order))}; exit 7",
            },
            {
                "design_id": "c",
                "structure_id": "s1",
                "backend": "esmfold2",
                "execution_mode": "slurm_bundle",
                "slurm_job_id": "999",
                "stdout_log": str(self.root / "logs" / "c.out"),
                "stderr_log": str(self.root / "logs" / "c.err"),
                "submission_status": "submitted",
                "completion_status": "SUBMITTED",
                "fold_output_dir": str(out_c),
                "fold_command": success_command("c", out_c),
            },
        ]
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        out = run_bundle_manifest(manifest_path)
        self.assertEqual(order.read_text(), "abc")
        by_design = out.set_index("design_id")
        self.assertEqual(by_design.loc["a", "completion_status"], "COMPLETED")
        self.assertEqual(by_design.loc["b", "completion_status"], "FAILED")
        self.assertEqual(str(by_design.loc["b", "exit_code"]), "7")
        self.assertEqual(by_design.loc["c", "completion_status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
