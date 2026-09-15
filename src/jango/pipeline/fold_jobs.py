"""CLI for fold-job Slurm submission and status updates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ._common import FOLD_BACKENDS, add_output_root, ensure_output_root_outside_repo, env_value
from .fold_execution import (
    SlurmConfig,
    run_bundle_manifest,
    submit_combined_job_bundle,
    submit_job_bundle,
    update_submission_status,
)
from .metadata import FOLD_METADATA, read_fold_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fold-jobs", description="Submit or inspect generated folding job bundles.")
    sub = parser.add_subparsers(dest="action", required=True)

    p_submit = sub.add_parser("submit", help="Submit incomplete fold jobs through the configured execution backend.")
    p_submit.add_argument("--fold-dir", type=Path, help="Directory containing fold_manifest.json; submits all mode job bundles.")
    p_submit.add_argument("--jobs-tsv", type=Path, help="Single backend job TSV to submit.")
    p_submit.add_argument("--work-dir", type=Path, help="Single-mode work directory containing job_bundle/.")
    p_submit.add_argument("--backend", choices=FOLD_BACKENDS, default=env_value("FOLDING_BACKEND"))
    p_submit.add_argument("--execution-mode", choices=["local", "slurm"], default=env_value("EXECUTION_MODE", "local"))
    add_output_root(p_submit)
    add_slurm_args(p_submit)
    p_submit.add_argument("--dry-run", action="store_true", help="Write scripts/manifests but do not call sbatch.")

    p_prepare = sub.add_parser("prepare-bundle", help="Prepare a combined fold bundle manifest for execution inside the current Slurm job.")
    p_prepare.add_argument("--fold-dir", required=True, type=Path, help="Directory containing fold_manifest.json; prepares all mode job bundles.")
    add_output_root(p_prepare)
    add_slurm_args(p_prepare)

    p_status = sub.add_parser("status", help="Refresh Slurm status for one manifest or all manifests under a fold directory.")
    p_status.add_argument("--fold-dir", type=Path, help="Directory containing fold_manifest.json.")
    p_status.add_argument("--manifest", type=Path, help="Single fold_submission_manifest.csv to update.")
    add_output_root(p_status)

    p_run = sub.add_parser("run-bundle", help="Run one Slurm bundle manifest sequentially inside an allocated Slurm job.")
    p_run.add_argument("--manifest", required=True, type=Path)
    p_run.add_argument("--allow-direct", action="store_true", help="Allow local execution for tests or CPU-only debugging.")
    return parser


def add_slurm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slurm-partition", default=env_value("SLURM_PARTITION"))
    parser.add_argument("--slurm-gpus", default=env_value("SLURM_GPUS"))
    parser.add_argument("--slurm-cpus-per-task", default=env_value("SLURM_CPUS_PER_TASK"))
    parser.add_argument("--slurm-mem", default=env_value("SLURM_MEM"))
    parser.add_argument("--slurm-time", default=env_value("SLURM_TIME"))
    parser.add_argument("--slurm-max-concurrent", default=env_value("SLURM_MAX_CONCURRENT"))
    parser.add_argument("--fold-workers-per-gpu", default=env_value("FOLD_WORKERS_PER_GPU"))
    parser.add_argument("--slurm-bundle-mode", choices=["combined", "per-design"], default=env_value("SLURM_BUNDLE_MODE", "combined"))
    parser.add_argument("--paths-config", type=Path, default=Path(env_value("JANGO_PATHS_CONFIG", "configs/test/paths.env")))
    parser.add_argument("--runtime-config", type=Path, default=Path(env_value("JANGO_RUNTIME_CONFIG", "configs/test/esmfold2.env")))


def slurm_config_from_args(args: argparse.Namespace, output_root: Path) -> SlurmConfig:
    return SlurmConfig.from_values(
        output_root=output_root,
        partition=getattr(args, "slurm_partition", None),
        gpus=getattr(args, "slurm_gpus", None),
        cpus_per_task=getattr(args, "slurm_cpus_per_task", None),
        mem=getattr(args, "slurm_mem", None),
        time=getattr(args, "slurm_time", None),
        paths_config=getattr(args, "paths_config", None),
        runtime_config=getattr(args, "runtime_config", None),
        max_concurrent=getattr(args, "slurm_max_concurrent", None),
        workers_per_gpu=getattr(args, "fold_workers_per_gpu", None),
    )


def fold_mode_jobs(fold_dir: Path) -> list[tuple[str, str, Path, Path]]:
    fold = read_fold_metadata(fold_dir / FOLD_METADATA)
    jobs: list[tuple[str, str, Path, Path]] = []
    for mode, outputs in fold.mode_outputs.items():
        work_dir = Path(outputs["work_dir"])
        jobs_tsv = work_dir / "job_bundle" / f"{fold.backend}_monomer_jobs.tsv"
        jobs.append((mode, fold.backend, work_dir, jobs_tsv))
    return jobs


def fold_bundle_work_dir(fold_dir: Path) -> Path:
    jobs = fold_mode_jobs(fold_dir)
    if not jobs:
        return fold_dir / "slurm"
    return jobs[0][2].parent


def mode_submission_manifests_for_fold(fold_dir: Path) -> list[Path]:
    return [work_dir / "slurm" / "fold_submission_manifest.csv" for _, _, work_dir, _ in fold_mode_jobs(fold_dir)]


def combined_submission_manifest_for_fold(fold_dir: Path) -> Path:
    return fold_bundle_work_dir(fold_dir) / "slurm" / "fold_submission_manifest.csv"


def submission_manifests_for_fold(fold_dir: Path) -> list[Path]:
    paths = [combined_submission_manifest_for_fold(fold_dir), *mode_submission_manifests_for_fold(fold_dir)]
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def submit_from_args(args: argparse.Namespace, output_root: Path) -> int:
    if args.execution_mode != "slurm":
        raise SystemExit("fold-jobs submit currently manages Slurm submissions; use decoy-fold for local execution")
    config = slurm_config_from_args(args, output_root)
    if args.fold_dir is not None and args.slurm_bundle_mode == "combined":
        tasks = fold_mode_jobs(args.fold_dir)
        if not tasks:
            raise SystemExit(f"no fold job bundles found under {args.fold_dir}")
        backends = {backend for _, backend, _, _ in tasks}
        if len(backends) != 1:
            raise SystemExit(f"combined Slurm bundle requires one backend, found: {sorted(backends)}")
        backend = next(iter(backends))
        work_dir = fold_bundle_work_dir(args.fold_dir)
        manifest = submit_combined_job_bundle(
            job_sources=[(mode, jobs_tsv) for mode, _, _, jobs_tsv in tasks],
            work_dir=work_dir,
            output_root=output_root,
            backend=backend,
            slurm_config=config,
            existing_manifest_paths=mode_submission_manifests_for_fold(args.fold_dir),
            dry_run=args.dry_run,
        )
        counts = manifest["submission_status"].value_counts(dropna=False).to_dict() if not manifest.empty else {}
        job_ids = sorted({str(job_id) for job_id in manifest.get("slurm_job_id", []) if str(job_id) and str(job_id) != "nan"})
        print(f"combined: wrote {len(manifest)} Slurm submission rows under {work_dir / 'slurm'} status_counts={counts} slurm_job_ids={job_ids}")
        print(f"fold-jobs submit complete: {len(manifest)} rows")
        return 0

    tasks: list[tuple[str, str, Path, Path]] = []
    if args.fold_dir is not None:
        tasks.extend(fold_mode_jobs(args.fold_dir))
    elif args.jobs_tsv is not None and args.work_dir is not None and args.backend is not None:
        tasks.append(("single", args.backend, args.work_dir, args.jobs_tsv))
    else:
        raise SystemExit("use either --fold-dir or all of --jobs-tsv, --work-dir, and --backend")

    total = 0
    for mode, backend, work_dir, jobs_tsv in tasks:
        manifest = submit_job_bundle(
            jobs_tsv=jobs_tsv,
            work_dir=work_dir,
            output_root=output_root,
            backend=backend,
            slurm_config=config,
            dry_run=args.dry_run,
        )
        total += len(manifest)
        counts = manifest["submission_status"].value_counts(dropna=False).to_dict() if not manifest.empty else {}
        print(f"{mode}: wrote {len(manifest)} Slurm submission rows for {jobs_tsv} status_counts={counts}")
    print(f"fold-jobs submit complete: {total} rows")
    return 0


def prepare_bundle_from_args(args: argparse.Namespace, output_root: Path) -> int:
    config = slurm_config_from_args(args, output_root)
    tasks = fold_mode_jobs(args.fold_dir)
    if not tasks:
        raise SystemExit(f"no fold job bundles found under {args.fold_dir}")
    backends = {backend for _, backend, _, _ in tasks}
    if len(backends) != 1:
        raise SystemExit(f"combined Slurm bundle requires one backend, found: {sorted(backends)}")
    backend = next(iter(backends))
    work_dir = fold_bundle_work_dir(args.fold_dir)
    manifest = submit_combined_job_bundle(
        job_sources=[(mode, jobs_tsv) for mode, _, _, jobs_tsv in tasks],
        work_dir=work_dir,
        output_root=output_root,
        backend=backend,
        slurm_config=config,
        existing_manifest_paths=mode_submission_manifests_for_fold(args.fold_dir),
        submit=False,
        execution_mode="fett_slurm_bundle",
    )
    counts = manifest["submission_status"].value_counts(dropna=False).to_dict() if not manifest.empty else {}
    print(f"combined: prepared {len(manifest)} fold bundle rows under {work_dir / 'slurm'} status_counts={counts}")
    print(f"fold-jobs prepare-bundle complete: {len(manifest)} rows")
    return 0


def status_from_args(args: argparse.Namespace) -> int:
    manifests: list[Path] = []
    if args.fold_dir is not None:
        manifests.extend(submission_manifests_for_fold(args.fold_dir))
    if args.manifest is not None:
        manifests.append(args.manifest)
    if not manifests:
        raise SystemExit("use --fold-dir or --manifest")

    for manifest_path in manifests:
        if not manifest_path.exists():
            print(f"{manifest_path}: missing")
            continue
        df = update_submission_status(manifest_path)
        counts = df["completion_status"].value_counts(dropna=False).to_dict() if not df.empty else {}
        active = {"COMPLETED", "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUBMITTED", "QUEUED_BUNDLE"}
        resumable = df[~df["completion_status"].astype(str).isin(active)]
        print(f"{manifest_path}: rows={len(df)} completion_counts={counts} resumable={len(resumable)}")
    return 0


def run_bundle_from_args(args: argparse.Namespace) -> int:
    if not args.allow_direct and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("fold-jobs run-bundle must run inside a Slurm allocation; use --allow-direct only for tests/debugging")
    df = run_bundle_manifest(args.manifest)
    counts = df["completion_status"].value_counts(dropna=False).to_dict() if not df.empty else {}
    print(f"fold-jobs run-bundle complete: rows={len(df)} completion_counts={counts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "run-bundle":
        return run_bundle_from_args(args)
    try:
        output_root = ensure_output_root_outside_repo(args.output_root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.action == "submit":
        return submit_from_args(args, output_root)
    if args.action == "prepare-bundle":
        return prepare_bundle_from_args(args, output_root)
    if args.action == "status":
        return status_from_args(args)
    parser.error(f"unsupported action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
