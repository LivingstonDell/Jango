"""Fold-job execution helpers for local and Slurm-backed runs."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd


ACTIVE_SLURM_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
}
COMPLETED_SLURM_STATES = {"COMPLETED"}
FAILED_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
ROW_TERMINAL_STATES = COMPLETED_SLURM_STATES | FAILED_SLURM_STATES | {"FAILED_OUTPUT_MISSING"}
BUNDLE_RUN_SUBMISSION_STATES = {"submitted", "queued_bundle"}


@dataclass(frozen=True)
class SlurmConfig:
    partition: str
    gpus: str
    cpus_per_task: str
    mem: str
    time: str
    output_root: Path
    repo_root: Path = Path("<JANGO_REPO>")
    conda_sh: Path = Path("<CONDA_ROOT>/etc/profile.d/conda.sh")
    conda_env: str = "jango"
    paths_config: Path = Path("configs/test/paths.env")
    runtime_config: Path = Path("configs/test/esmfold2.env")
    max_concurrent: str = "1"
    workers_per_gpu: str = "1"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs" / "slurm"

    @classmethod
    def from_values(
        cls,
        *,
        output_root: Path,
        partition: str | None = None,
        gpus: str | int | None = None,
        cpus_per_task: str | int | None = None,
        mem: str | None = None,
        time: str | None = None,
        repo_root: Path | None = None,
        conda_sh: Path | None = None,
        conda_env: str | None = None,
        paths_config: Path | None = None,
        runtime_config: Path | None = None,
        max_concurrent: str | int | None = None,
        workers_per_gpu: str | int | None = None,
    ) -> "SlurmConfig":
        return cls(
            partition=str(partition or os.environ.get("SLURM_PARTITION") or "batch"),
            gpus=str(gpus or os.environ.get("SLURM_GPUS") or "1"),
            cpus_per_task=str(cpus_per_task or os.environ.get("SLURM_CPUS_PER_TASK") or "8"),
            mem=str(mem or os.environ.get("SLURM_MEM") or "64G"),
            time=str(time or os.environ.get("SLURM_TIME") or "08:00:00"),
            output_root=Path(output_root).expanduser(),
            repo_root=Path(repo_root or os.environ.get("JANGO_SOURCE") or cls.repo_root).expanduser(),
            conda_sh=Path(conda_sh or os.environ.get("CONDA_SH") or cls.conda_sh).expanduser(),
            conda_env=str(conda_env or os.environ.get("JANGO_CONDA_ENV") or cls.conda_env),
            paths_config=Path(paths_config or os.environ.get("JANGO_PATHS_CONFIG") or cls.paths_config),
            runtime_config=Path(runtime_config or os.environ.get("JANGO_RUNTIME_CONFIG") or cls.runtime_config),
            max_concurrent=str(max_concurrent or os.environ.get("SLURM_MAX_CONCURRENT") or cls.max_concurrent),
            workers_per_gpu=str(workers_per_gpu or os.environ.get("FOLD_WORKERS_PER_GPU") or cls.workers_per_gpu),
        )


def safe_job_name(*parts: object, limit: int = 128) -> str:
    raw = "-".join(str(part) for part in parts if str(part))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.") or "jango-fold"
    return safe[:limit]


def normalize_slurm_job_id(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def read_jobs_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"fold jobs TSV not found: {path}")
    return pd.read_csv(path, sep="\t").fillna("")


def prediction_outputs_complete(output_dir: str | Path) -> bool:
    path = Path(output_dir)
    if not path.is_dir():
        return False
    for suffix in ("*.pdb", "*.cif", "*.mmcif"):
        for candidate in path.rglob(suffix):
            if "raw_opendde" in candidate.parts:
                continue
            return True
    return False


def slurm_paths_for_job(row: Mapping[str, object], *, work_dir: Path, config: SlurmConfig) -> dict[str, Path | str]:
    backend = str(row.get("fold_backend") or row.get("backend") or "fold")
    design_id = str(row.get("design_id") or "design")
    structure_id = str(row.get("structure_id") or "structure")
    job_name = safe_job_name("jg", backend, structure_id, design_id, limit=120)
    script_dir = work_dir / "slurm"
    script = script_dir / f"{job_name}.sbatch"
    return {
        "job_name": job_name,
        "script": script,
        "stdout_log": config.logs_dir / "%x-%j.out",
        "stderr_log": config.logs_dir / "%x-%j.err",
    }


def concrete_slurm_log_path(template: Path | str, *, job_name: str, job_id: str) -> str:
    text = str(template)
    if not job_id:
        return text
    return text.replace("%x", job_name).replace("%j", job_id).replace("%A", job_id)


def write_slurm_script(row: Mapping[str, object], *, work_dir: Path, config: SlurmConfig) -> Path:
    paths = slurm_paths_for_job(row, work_dir=work_dir, config=config)
    script = Path(paths["script"])
    command = str(row.get("fold_command") or "").strip()
    if not command:
        raise ValueError(f"fold_command is empty for design_id={row.get('design_id')}")
    script.parent.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --gres=gpu:{config.gpus}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem={config.mem}",
            f"#SBATCH --time={config.time}",
            f"#SBATCH --job-name={paths['job_name']}",
            f"#SBATCH --output={paths['stdout_log']}",
            f"#SBATCH --error={paths['stderr_log']}",
            "",
            "set -euo pipefail",
            f"source {shlex.quote(str(config.conda_sh))}",
            f"conda activate {shlex.quote(config.conda_env)}",
            f"cd {shlex.quote(str(config.repo_root))}",
            "set -a",
            f"source {shlex.quote(str(config.paths_config))}",
            f"source {shlex.quote(str(config.runtime_config))}",
            "set +a",
            f"export NBIA_ROOT=\"${{JANGO_SOURCE:-{config.repo_root}}}\"",
            "export PYTHONPATH=\"${NBIA_ROOT}/src:${PYTHONPATH:-}\"",
            "",
            command,
            "",
        ]
    )
    script.write_text(text)
    script.chmod(0o750)
    return script


def slurm_paths_for_combined_bundle(*, bundle_id: str, work_dir: Path, config: SlurmConfig) -> dict[str, Path | str]:
    job_name = safe_job_name("jg", bundle_id, limit=120)
    script_dir = work_dir / "slurm"
    return {
        "job_name": job_name,
        "script": script_dir / f"{job_name}.sbatch",
        "stdout_log": config.logs_dir / "%x-%j.out",
        "stderr_log": config.logs_dir / "%x-%j.err",
    }


def design_log_paths(row: Mapping[str, object], *, bundle_id: str, config: SlurmConfig) -> tuple[Path, Path]:
    mode = str(row.get("redesign_mode") or row.get("mode") or "mode")
    structure_id = str(row.get("structure_id") or "structure")
    design_id = str(row.get("design_id") or "design")
    log_stem = safe_job_name(mode, structure_id, design_id, limit=180)
    log_dir = config.output_root / "logs" / "fold-bundle" / safe_job_name(bundle_id, limit=80)
    return log_dir / f"{log_stem}.out", log_dir / f"{log_stem}.err"


def write_slurm_bundle_script(*, bundle_id: str, manifest_path: Path, work_dir: Path, config: SlurmConfig) -> Path:
    paths = slurm_paths_for_combined_bundle(bundle_id=bundle_id, work_dir=work_dir, config=config)
    script = Path(paths["script"])
    script.parent.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --gres=gpu:{config.gpus}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem={config.mem}",
            f"#SBATCH --time={config.time}",
            f"#SBATCH --job-name={paths['job_name']}",
            f"#SBATCH --output={paths['stdout_log']}",
            f"#SBATCH --error={paths['stderr_log']}",
            "",
            "set -euo pipefail",
            f"source {shlex.quote(str(config.conda_sh))}",
            f"conda activate {shlex.quote(config.conda_env)}",
            f"cd {shlex.quote(str(config.repo_root))}",
            "set -a",
            f"source {shlex.quote(str(config.paths_config))}",
            f"source {shlex.quote(str(config.runtime_config))}",
            "set +a",
            f"export NBIA_ROOT=\"${{JANGO_SOURCE:-{config.repo_root}}}\"",
            "export PYTHONPATH=\"${NBIA_ROOT}/src:${PYTHONPATH:-}\"",
            f"export SLURM_MAX_CONCURRENT={shlex.quote(str(config.max_concurrent))}",
            f"export FOLD_WORKERS_PER_GPU={shlex.quote(str(config.workers_per_gpu))}",
            "",
            f"jango fold-jobs run-bundle --manifest {shlex.quote(str(manifest_path))}",
            "",
        ]
    )
    script.write_text(text)
    script.chmod(0o750)
    return script


def parse_sbatch_job_id(stdout: str) -> str:
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if not match:
        raise ValueError(f"could not parse sbatch job id from: {stdout.strip()!r}")
    return match.group(1)


def submit_sbatch(script: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str:
    result = runner(["sbatch", str(script)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "sbatch failed").strip()
        raise RuntimeError(detail)
    return parse_sbatch_job_id(result.stdout)


def _existing_by_design(existing: pd.DataFrame) -> dict[str, dict[str, object]]:
    if existing.empty or "design_id" not in existing.columns:
        return {}
    return {str(row["design_id"]): dict(row) for row in existing.to_dict("records")}


def _read_existing_manifests(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.is_file():
            frames.append(pd.read_csv(path).fillna(""))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).fillna("")


def query_squeue(job_ids: Iterable[str], *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, str]:
    ids = [normalize_slurm_job_id(job_id) for job_id in job_ids if normalize_slurm_job_id(job_id)]
    if not ids:
        return {}
    result = runner(["squeue", "-h", "-j", ",".join(ids), "-o", "%A|%T"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        return {}
    statuses: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        job_id, state = line.split("|", 1)
        statuses[normalize_slurm_job_id(job_id)] = state.strip()
    return statuses


def query_sacct(job_ids: Iterable[str], *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, tuple[str, str]]:
    ids = [normalize_slurm_job_id(job_id) for job_id in job_ids if normalize_slurm_job_id(job_id)]
    if not ids:
        return {}
    result = runner(["sacct", "-P", "-n", "-j", ",".join(ids), "--format=JobIDRaw,State,ExitCode"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        return {}
    statuses: dict[str, tuple[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id, state, exit_code = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if "." in job_id:
            continue
        statuses[normalize_slurm_job_id(job_id)] = (state, exit_code)
    return statuses


def classify_existing_job(row: Mapping[str, object], active: Mapping[str, str], completed_outputs: bool) -> tuple[str, str, str]:
    job_id = normalize_slurm_job_id(row.get("slurm_job_id"))
    if completed_outputs:
        return "skipped_completed", "COMPLETED", "0"
    if job_id and active.get(job_id) in ACTIVE_SLURM_STATES:
        return "skipped_active", active[job_id], ""
    return "eligible", "INCOMPLETE", ""


def submit_job_bundle(
    *,
    jobs_tsv: Path,
    work_dir: Path,
    output_root: Path,
    backend: str,
    slurm_config: SlurmConfig,
    manifest_path: Path | None = None,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> pd.DataFrame:
    jobs = read_jobs_tsv(jobs_tsv)
    manifest_path = manifest_path or (work_dir / "slurm" / "fold_submission_manifest.csv")
    existing = pd.read_csv(manifest_path).fillna("") if manifest_path.is_file() else pd.DataFrame()
    existing_by_design = _existing_by_design(existing)
    active = query_squeue((row.get("slurm_job_id", "") for row in existing_by_design.values()), runner=runner)
    rows: list[dict[str, object]] = []
    for job in jobs.to_dict("records"):
        design_id = str(job.get("design_id") or "")
        previous = existing_by_design.get(design_id, {})
        output_dir = str(job.get("fold_output_dir") or "")
        output_complete = prediction_outputs_complete(output_dir)
        submission_status, completion_status, exit_code = classify_existing_job(previous, active, output_complete)
        paths = slurm_paths_for_job(job, work_dir=work_dir, config=slurm_config)
        slurm_job_id = normalize_slurm_job_id(previous.get("slurm_job_id"))
        if submission_status == "eligible":
            script = write_slurm_script(job, work_dir=work_dir, config=slurm_config)
            if dry_run:
                submission_status = "dry_run"
                completion_status = "NOT_SUBMITTED"
            else:
                slurm_job_id = submit_sbatch(script, runner=runner)
                submission_status = "submitted"
                completion_status = "SUBMITTED"
        row = {
            "design_id": design_id,
            "structure_id": str(job.get("structure_id") or ""),
            "backend": backend,
            "execution_mode": "slurm",
            "slurm_job_id": slurm_job_id,
            "slurm_script": str(paths["script"]),
            "stdout_log": concrete_slurm_log_path(paths["stdout_log"], job_name=str(paths["job_name"]), job_id=slurm_job_id),
            "stderr_log": concrete_slurm_log_path(paths["stderr_log"], job_name=str(paths["job_name"]), job_id=slurm_job_id),
            "submission_status": submission_status,
            "completion_status": completion_status,
            "exit_code": exit_code,
            "redesign_mode": str(job.get("redesign_mode") or ""),
            "sequence_role": str(job.get("sequence_role") or ""),
            "fold_output_dir": output_dir,
            "fold_command": str(job.get("fold_command") or ""),
        }
        rows.append(row)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    out.to_csv(manifest_path, index=False)
    return out


def _read_mode_job_sources(job_sources: Sequence[tuple[str, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mode, jobs_tsv in job_sources:
        df = read_jobs_tsv(jobs_tsv)
        if "redesign_mode" not in df.columns:
            df["redesign_mode"] = mode
        else:
            df["redesign_mode"] = df["redesign_mode"].replace("", mode)
        df["source_jobs_tsv"] = str(jobs_tsv)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).fillna("")


def submit_combined_job_bundle(
    *,
    job_sources: Sequence[tuple[str, Path]],
    work_dir: Path,
    output_root: Path,
    backend: str,
    slurm_config: SlurmConfig,
    manifest_path: Path | None = None,
    bundle_id: str | None = None,
    existing_manifest_paths: Sequence[Path] = (),
    dry_run: bool = False,
    submit: bool = True,
    execution_mode: str = "slurm_bundle",
    current_slurm_job_id: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> pd.DataFrame:
    jobs = _read_mode_job_sources(job_sources)
    bundle_id = bundle_id or safe_job_name(work_dir.name, "fold-bundle", limit=80)
    manifest_path = manifest_path or (work_dir / "slurm" / "fold_submission_manifest.csv")
    bundle_jobs_tsv = manifest_path.parent / f"{safe_job_name(bundle_id, limit=80)}_jobs.tsv"
    existing_paths = [*existing_manifest_paths, manifest_path]
    existing = _read_existing_manifests(existing_paths)
    existing_by_design = _existing_by_design(existing)
    active = query_squeue((row.get("slurm_job_id", "") for row in existing_by_design.values()), runner=runner)
    rows: list[dict[str, object]] = []
    for job in jobs.to_dict("records"):
        design_id = str(job.get("design_id") or "")
        previous = existing_by_design.get(design_id, {})
        output_dir = str(job.get("fold_output_dir") or "")
        output_complete = prediction_outputs_complete(output_dir)
        submission_status, completion_status, exit_code = classify_existing_job(previous, active, output_complete)
        slurm_job_id = normalize_slurm_job_id(previous.get("slurm_job_id"))
        stdout_log, stderr_log = design_log_paths(job, bundle_id=bundle_id, config=slurm_config)
        if submission_status == "eligible":
            submission_status = "queued_bundle"
            completion_status = "QUEUED_BUNDLE"
            slurm_job_id = ""
        rows.append(
            {
                "design_id": design_id,
                "structure_id": str(job.get("structure_id") or ""),
                "backend": backend,
                "execution_mode": execution_mode,
                "slurm_job_id": slurm_job_id,
                "slurm_script": "",
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "submission_status": submission_status,
                "completion_status": completion_status,
                "exit_code": exit_code,
                "redesign_mode": str(job.get("redesign_mode") or ""),
                "sequence_role": str(job.get("sequence_role") or ""),
                "fold_output_dir": output_dir,
                "fold_command": str(job.get("fold_command") or ""),
                "bundle_id": bundle_id,
                "bundle_size": "",
                "source_jobs_tsv": str(job.get("source_jobs_tsv") or ""),
                "bundle_stdout_log": "",
                "bundle_stderr_log": "",
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    pending_mask = out["submission_status"].astype(str).eq("queued_bundle")
    pending_count = int(pending_mask.sum())
    if pending_count:
        out.loc[pending_mask, "bundle_size"] = str(pending_count)
        if submit or dry_run:
            script = write_slurm_bundle_script(bundle_id=bundle_id, manifest_path=manifest_path, work_dir=work_dir, config=slurm_config)
            bundle_paths = slurm_paths_for_combined_bundle(bundle_id=bundle_id, work_dir=work_dir, config=slurm_config)
            out.loc[pending_mask, "slurm_script"] = str(script)
            out.loc[pending_mask, "bundle_stdout_log"] = str(bundle_paths["stdout_log"])
            out.loc[pending_mask, "bundle_stderr_log"] = str(bundle_paths["stderr_log"])
        out.to_csv(manifest_path, index=False)
        out[pending_mask].to_csv(bundle_jobs_tsv, sep="\t", index=False)
        if dry_run:
            out.loc[pending_mask, "submission_status"] = "dry_run"
            out.loc[pending_mask, "completion_status"] = "NOT_SUBMITTED"
        elif submit:
            submitted_job_id = submit_sbatch(script, runner=runner)
            out.loc[pending_mask, "slurm_job_id"] = submitted_job_id
            out.loc[pending_mask, "submission_status"] = "submitted"
            out.loc[pending_mask, "completion_status"] = "SUBMITTED"
        else:
            current_job_id = normalize_slurm_job_id(current_slurm_job_id or os.environ.get("SLURM_JOB_ID"))
            out.loc[pending_mask, "slurm_job_id"] = current_job_id
            out.loc[pending_mask, "submission_status"] = "submitted"
            out.loc[pending_mask, "completion_status"] = "SUBMITTED"
    out.to_csv(manifest_path, index=False)
    if pending_count:
        out[out["submission_status"].astype(str).isin({"submitted", "dry_run", "queued_bundle"})].to_csv(bundle_jobs_tsv, sep="\t", index=False)
    return out


SUBMISSION_COLUMNS = [
    "design_id",
    "structure_id",
    "backend",
    "execution_mode",
    "slurm_job_id",
    "slurm_script",
    "stdout_log",
    "stderr_log",
    "submission_status",
    "completion_status",
    "exit_code",
    "redesign_mode",
    "sequence_role",
    "fold_output_dir",
    "fold_command",
    "bundle_id",
    "bundle_size",
    "source_jobs_tsv",
    "bundle_stdout_log",
    "bundle_stderr_log",
]


def _write_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    df.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)


def _status_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("design_id") or ""), str(row.get("redesign_mode") or ""))


def _is_esmfold2_single_fold_command(command: str) -> bool:
    return (
        "run_esmfold2_monomer.py" in command
        and "run_esmfold2_monomer_jobs.py" not in command
        and "--fasta" in command
        and "--out-dir" in command
    )


def _is_opendde_single_fold_command(command: str) -> bool:
    return (
        "run_opendde_monomer.py" in command
        and "run_opendde_monomer_jobs.py" not in command
        and "--json" in command
        and "--out-dir" in command
    )


def _esmfold2_jobs_runner_path() -> Path:
    repo_root = Path(
        os.environ.get("JANGO_SOURCE")
        or os.environ.get("NBIA_ROOT")
        or Path(__file__).resolve().parents[3]
    ).expanduser()
    runner_path = repo_root / "scripts" / "nbia" / "run_esmfold2_monomer_jobs.py"
    if not runner_path.is_file():
        raise FileNotFoundError(f"ESMFold2 batch runner not found: {runner_path}")
    return runner_path


def _opendde_jobs_runner_path() -> Path:
    repo_root = Path(
        os.environ.get("JANGO_SOURCE")
        or os.environ.get("NBIA_ROOT")
        or Path(__file__).resolve().parents[3]
    ).expanduser()
    runner_path = repo_root / "scripts" / "nbia" / "run_opendde_monomer_jobs.py"
    if not runner_path.is_file():
        raise FileNotFoundError(f"OpenDDE batch runner not found: {runner_path}")
    return runner_path


def _esmfold2_python() -> str:
    python = str(os.environ.get("ESMFOLD2_PYTHON") or "").strip()
    if not python:
        raise RuntimeError("ESMFOLD2_PYTHON is required for ESMFold2 bundle execution")
    return python


def _command_option(tokens: Sequence[str], name: str) -> str | None:
    try:
        index = list(tokens).index(name)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _nonblank(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _scientific_rows_for_bundle(df: pd.DataFrame, indices: list[object]) -> pd.DataFrame:
    source_cache: dict[Path, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    overlay_columns = [
        "stdout_log",
        "stderr_log",
        "slurm_job_id",
        "submission_status",
        "completion_status",
        "exit_code",
        "bundle_id",
        "bundle_size",
        "source_jobs_tsv",
    ]
    for idx in indices:
        submission_row = df.loc[idx].to_dict()
        job_row = dict(submission_row)
        source_value = str(submission_row.get("source_jobs_tsv") or "").strip()
        if source_value:
            source_path = Path(source_value).expanduser()
            if source_path not in source_cache:
                source_cache[source_path] = read_jobs_tsv(source_path)
            source = source_cache[source_path]
            design_id = str(submission_row.get("design_id") or "")
            redesign_mode = str(submission_row.get("redesign_mode") or "")
            matches = source[source["design_id"].astype(str) == design_id]
            if "redesign_mode" in source.columns:
                mode_matches = matches[matches["redesign_mode"].astype(str) == redesign_mode]
                if not mode_matches.empty:
                    matches = mode_matches
            if matches.empty:
                raise ValueError(f"source fold job row not found for design_id={design_id} redesign_mode={redesign_mode} in {source_path}")
            job_row = matches.iloc[-1].to_dict()
        tokens = shlex.split(str(job_row.get("fold_command") or ""))
        if not _nonblank(job_row.get("fold_input")):
            fasta = _command_option(tokens, "--fasta")
            if fasta:
                job_row["fold_input"] = fasta
        if not _nonblank(job_row.get("msa_path")):
            msa = _command_option(tokens, "--msa-a3m")
            if msa:
                job_row["msa_path"] = msa
        for column in overlay_columns:
            value = submission_row.get(column)
            if _nonblank(value):
                job_row[column] = value
        rows.append(job_row)
    return pd.DataFrame(rows)


def _esmfold2_batch_command(row: Mapping[str, object], *, jobs_tsv: Path, status_csv: Path) -> list[str]:
    tokens = shlex.split(str(row.get("fold_command") or ""))
    python = tokens[0] if tokens else _esmfold2_python()
    runner_path = _esmfold2_jobs_runner_path()
    if len(tokens) > 1:
        single_runner = Path(tokens[1]).expanduser()
        sibling_runner = single_runner.with_name("run_esmfold2_monomer_jobs.py")
        if sibling_runner.is_file():
            runner_path = sibling_runner
    command = [
        python,
        str(runner_path),
        "--jobs-tsv",
        str(jobs_tsv),
        "--status-csv",
        str(status_csv),
    ]
    for option in (
        "--model-id-or-path",
        "--cache-dir",
        "--device",
        "--esm-root",
        "--num-recycles",
        "--num-sampling-steps",
        "--num-diffusion-samples",
        "--seed",
        "--msa-max-sequences",
    ):
        value = _command_option(tokens, option)
        if value:
            command.extend([option, value])
    return command


def _opendde_batch_command(row: Mapping[str, object], *, jobs_tsv: Path, status_csv: Path) -> list[str]:
    tokens = shlex.split(str(row.get("fold_command") or ""))
    python = tokens[0] if tokens else str(os.environ.get("OPENDDE_PYTHON") or "python")
    runner_path = _opendde_jobs_runner_path()
    if len(tokens) > 1:
        single_runner = Path(tokens[1]).expanduser()
        sibling_runner = single_runner.with_name("run_opendde_monomer_jobs.py")
        if sibling_runner.is_file():
            runner_path = sibling_runner
    command = [
        python,
        str(runner_path),
        "--jobs-tsv",
        str(jobs_tsv),
        "--status-csv",
        str(status_csv),
    ]
    for option in (
        "--opendde-executable",
        "--root-dir",
        "--model-name",
        "--checkpoint",
        "--seeds",
        "--cycle",
        "--step",
        "--sample",
        "--dtype",
        "--use-msa",
    ):
        value = _command_option(tokens, option)
        if value:
            command.extend([option, value])
    if "--require-msa" in tokens:
        command.append("--require-msa")
    workers = str(os.environ.get("OPENDDE_BATCH_WORKERS") or os.environ.get("SLURM_MAX_CONCURRENT") or "").strip()
    if workers:
        command.extend(["--workers", workers])
    return command


def _read_esmfold2_batch_status(status_csv: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not status_csv.is_file():
        return {}
    table = pd.read_csv(status_csv).fillna("")
    return {_status_key(row): dict(row) for row in table.to_dict("records")}


def _read_opendde_batch_status(status_csv: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not status_csv.is_file():
        return {}
    table = pd.read_csv(status_csv).fillna("")
    return {_status_key(row): dict(row) for row in table.to_dict("records")}


def _run_esmfold2_bundle_rows(
    df: pd.DataFrame,
    indices: list[object],
    manifest_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[object]],
) -> pd.DataFrame:
    jobs_tsv = manifest_path.parent / f"{manifest_path.stem}_esmfold2_jobs.tsv"
    status_csv = manifest_path.parent / f"{manifest_path.stem}_esmfold2_status.csv"
    jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
    _scientific_rows_for_bundle(df, indices).to_csv(jobs_tsv, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    df.loc[indices, "completion_status"] = "RUNNING"
    df.loc[indices, "exit_code"] = pd.NA
    _write_manifest(df, manifest_path)

    command = _esmfold2_batch_command(df.loc[indices[0]], jobs_tsv=jobs_tsv, status_csv=status_csv)
    result = runner(command, text=True, check=False)
    bundle_returncode = int(getattr(result, "returncode", 1))
    status_by_key = _read_esmfold2_batch_status(status_csv)

    for idx in indices:
        row = df.loc[idx]
        output_dir = str(row.get("fold_output_dir") or "")
        status = status_by_key.get(_status_key(row), {})
        row_status = str(status.get("completion_status") or "")
        row_exit_code = str(status.get("exit_code") or "")
        output_ok = prediction_outputs_complete(output_dir)
        if row_status == "COMPLETED" and output_ok:
            df.at[idx, "completion_status"] = "COMPLETED"
            df.at[idx, "exit_code"] = "0"
        elif row_status == "COMPLETED":
            df.at[idx, "completion_status"] = "FAILED_OUTPUT_MISSING"
            df.at[idx, "exit_code"] = row_exit_code or "0"
        elif row_status:
            df.at[idx, "completion_status"] = row_status
            df.at[idx, "exit_code"] = row_exit_code or str(bundle_returncode or 1)
        elif output_ok:
            df.at[idx, "completion_status"] = "COMPLETED"
            df.at[idx, "exit_code"] = "0"
        else:
            df.at[idx, "completion_status"] = "FAILED"
            df.at[idx, "exit_code"] = str(bundle_returncode or 1)
        _write_manifest(df, manifest_path)
    return df


def _run_opendde_bundle_rows(
    df: pd.DataFrame,
    indices: list[object],
    manifest_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[object]],
) -> pd.DataFrame:
    jobs_tsv = manifest_path.parent / f"{manifest_path.stem}_opendde_jobs.tsv"
    status_csv = manifest_path.parent / f"{manifest_path.stem}_opendde_status.csv"
    jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
    _scientific_rows_for_bundle(df, indices).to_csv(jobs_tsv, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    df.loc[indices, "completion_status"] = "RUNNING"
    df.loc[indices, "exit_code"] = pd.NA
    _write_manifest(df, manifest_path)

    command = _opendde_batch_command(df.loc[indices[0]], jobs_tsv=jobs_tsv, status_csv=status_csv)
    result = runner(command, text=True, check=False)
    bundle_returncode = int(getattr(result, "returncode", 1))
    status_by_key = _read_opendde_batch_status(status_csv)

    for idx in indices:
        row = df.loc[idx]
        output_dir = str(row.get("fold_output_dir") or "")
        status = status_by_key.get(_status_key(row), {})
        row_status = str(status.get("completion_status") or "")
        row_exit_code = str(status.get("exit_code") or "")
        output_ok = prediction_outputs_complete(output_dir)
        if row_status == "COMPLETED" and output_ok:
            df.at[idx, "completion_status"] = "COMPLETED"
            df.at[idx, "exit_code"] = "0"
        elif row_status == "COMPLETED":
            df.at[idx, "completion_status"] = "FAILED_OUTPUT_MISSING"
            df.at[idx, "exit_code"] = row_exit_code or "0"
        elif row_status:
            df.at[idx, "completion_status"] = row_status
            df.at[idx, "exit_code"] = row_exit_code or str(bundle_returncode or 1)
        elif output_ok:
            df.at[idx, "completion_status"] = "COMPLETED"
            df.at[idx, "exit_code"] = "0"
        else:
            df.at[idx, "completion_status"] = "FAILED"
            df.at[idx, "exit_code"] = str(bundle_returncode or 1)
        _write_manifest(df, manifest_path)
    return df


def run_bundle_manifest(
    manifest_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Slurm bundle manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path).fillna("")
    if "exit_code" in df.columns:
        df["exit_code"] = df["exit_code"].astype("string")

    runnable_indices: list[object] = []
    for idx, row in df.iterrows():
        submission_status = str(row.get("submission_status") or "")
        if submission_status not in BUNDLE_RUN_SUBMISSION_STATES:
            continue
        output_dir = str(row.get("fold_output_dir") or "")
        if prediction_outputs_complete(output_dir):
            df.at[idx, "submission_status"] = "skipped_completed"
            df.at[idx, "completion_status"] = "COMPLETED"
            df.at[idx, "exit_code"] = "0"
            _write_manifest(df, manifest_path)
            continue
        command = str(row.get("fold_command") or "").strip()
        if not command:
            df.at[idx, "completion_status"] = "FAILED"
            df.at[idx, "exit_code"] = "missing_command"
            _write_manifest(df, manifest_path)
            continue
        runnable_indices.append(idx)

    if runnable_indices and all(_is_esmfold2_single_fold_command(str(df.at[idx, "fold_command"])) for idx in runnable_indices):
        return _run_esmfold2_bundle_rows(df, runnable_indices, manifest_path, runner=runner)
    if runnable_indices and all(_is_opendde_single_fold_command(str(df.at[idx, "fold_command"])) for idx in runnable_indices):
        return _run_opendde_bundle_rows(df, runnable_indices, manifest_path, runner=runner)

    for idx in runnable_indices:
        row = df.loc[idx]
        command = str(row.get("fold_command") or "").strip()
        output_dir = str(row.get("fold_output_dir") or "")
        stdout_log = Path(str(row.get("stdout_log") or ""))
        stderr_log = Path(str(row.get("stderr_log") or ""))
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        df.at[idx, "completion_status"] = "RUNNING"
        df.at[idx, "exit_code"] = pd.NA
        _write_manifest(df, manifest_path)
        with stdout_log.open("w") as stdout, stderr_log.open("w") as stderr:
            result = runner(command, shell=True, executable="/bin/bash", stdout=stdout, stderr=stderr, check=False)
        returncode = int(getattr(result, "returncode", 1))
        if returncode == 0 and prediction_outputs_complete(output_dir):
            df.at[idx, "completion_status"] = "COMPLETED"
            df.at[idx, "exit_code"] = "0"
        elif returncode == 0:
            df.at[idx, "completion_status"] = "FAILED_OUTPUT_MISSING"
            df.at[idx, "exit_code"] = "0"
        else:
            df.at[idx, "completion_status"] = "FAILED"
            df.at[idx, "exit_code"] = str(returncode)
        _write_manifest(df, manifest_path)
    return df


def update_submission_status(
    manifest_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Slurm submission manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path).fillna("")
    if "exit_code" in df.columns:
        df["exit_code"] = df["exit_code"].astype("string")
    job_ids = [normalize_slurm_job_id(job_id) for job_id in df.get("slurm_job_id", []) if normalize_slurm_job_id(job_id)]
    active = query_squeue(job_ids, runner=runner)
    historical = query_sacct(job_ids, runner=runner)
    for idx, row in df.iterrows():
        current_status = str(row.get("completion_status") or "")
        if current_status in ROW_TERMINAL_STATES:
            continue
        job_id = normalize_slurm_job_id(row.get("slurm_job_id"))
        if not job_id:
            continue
        if job_id in active:
            df.at[idx, "completion_status"] = active[job_id]
            df.at[idx, "exit_code"] = pd.NA
        elif job_id in historical:
            state, exit_code = historical[job_id]
            df.at[idx, "completion_status"] = state
            df.at[idx, "exit_code"] = exit_code
    df.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def resumable_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    active = manifest["completion_status"].astype(str).isin(ACTIVE_SLURM_STATES | {"SUBMITTED", "QUEUED_BUNDLE", "RUNNING"})
    completed = manifest["completion_status"].astype(str).isin(COMPLETED_SLURM_STATES)
    return manifest[~active & ~completed].copy()
