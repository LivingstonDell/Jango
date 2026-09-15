"""Artifact-oriented folding command group for Jango."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ._common import EXECUTION_MODES, FOLD_BACKENDS, add_atlas_root, add_output_root, add_plan_only, env_path, env_value
from . import decoy_fold, fold_jobs, fold_qc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jango fold",
        description="Prepare, submit, inspect, and QC backend folding from reusable artifacts.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_prepare = sub.add_parser("prepare", help="Reuse ProteinMPNN artifacts, resolve MSAs, and write fold job bundles.")
    p_prepare.add_argument("--prefold-dir", required=True, type=Path, help="Directory containing prefold_manifest.json.")
    add_atlas_root(p_prepare)
    p_prepare.add_argument("--backend", choices=FOLD_BACKENDS, default=env_value("FOLDING_BACKEND"), required=env_value("FOLDING_BACKEND") is None)
    p_prepare.add_argument("--mpnn-input", required=True, type=Path, help="ProteinMPNN artifact directory, either <root>/<mode> or a root containing <mode>/.")
    p_prepare.add_argument("--msa-cache-dir", "--msa-cache", dest="msa_cache_dir", type=Path, default=env_path("MSA_CACHE_ROOT") or env_path("MSA_CACHE_DIR"))
    p_prepare.add_argument("--max-decoy-sequences", type=int, default=5)
    p_prepare.add_argument("--paths-config", type=Path, default=Path(env_value("JANGO_PATHS_CONFIG", "configs/test/paths.env")))
    p_prepare.add_argument("--runtime-config", type=Path, default=Path(env_value("JANGO_RUNTIME_CONFIG", "configs/test/esmfold2.env")))
    add_output_root(p_prepare)
    add_plan_only(p_prepare)

    p_submit = sub.add_parser("submit", help="Submit prepared fold jobs through the configured execution backend.")
    p_submit.add_argument("--fold-dir", required=True, type=Path, help="Directory containing fold_manifest.json.")
    p_submit.add_argument("--execution-mode", choices=EXECUTION_MODES, default=env_value("EXECUTION_MODE", "slurm"))
    p_submit.add_argument("--backend", choices=FOLD_BACKENDS, default=env_value("FOLDING_BACKEND"))
    p_submit.add_argument("--slurm-partition", default=env_value("SLURM_PARTITION"))
    p_submit.add_argument("--slurm-gpus", default=env_value("SLURM_GPUS"))
    p_submit.add_argument("--slurm-cpus-per-task", default=env_value("SLURM_CPUS_PER_TASK"))
    p_submit.add_argument("--slurm-mem", default=env_value("SLURM_MEM"))
    p_submit.add_argument("--slurm-time", default=env_value("SLURM_TIME"))
    p_submit.add_argument("--slurm-max-concurrent", default=env_value("SLURM_MAX_CONCURRENT"))
    p_submit.add_argument("--fold-workers-per-gpu", default=env_value("FOLD_WORKERS_PER_GPU"))
    p_submit.add_argument("--slurm-bundle-mode", choices=("combined", "per-design"), default=env_value("SLURM_BUNDLE_MODE", "combined"))
    p_submit.add_argument("--paths-config", type=Path, default=Path(env_value("JANGO_PATHS_CONFIG", "configs/test/paths.env")))
    p_submit.add_argument("--runtime-config", type=Path, default=Path(env_value("JANGO_RUNTIME_CONFIG", "configs/test/esmfold2.env")))
    p_submit.add_argument("--dry-run", action="store_true")
    add_output_root(p_submit)

    p_status = sub.add_parser("status", help="Refresh fold submission status.")
    p_status.add_argument("--fold-dir", type=Path)
    p_status.add_argument("--manifest", type=Path)
    add_output_root(p_status)

    p_qc = sub.add_parser("qc", help="Validate existing fold outputs without rerunning folding.")
    p_qc.add_argument("--fold-dir", required=True, type=Path)
    p_qc.add_argument("--validation", action="append", type=Path)
    p_qc.add_argument("--out", type=Path)
    p_qc.add_argument("--figure-out", type=Path)
    p_qc.add_argument("--skip-figure", action="store_true")
    add_output_root(p_qc)
    return parser


def _append_optional(parts: list[str], option: str, value: object | None) -> None:
    if value is not None:
        parts.extend([option, str(value)])


def _prepare_args(args: argparse.Namespace, *, stage: str) -> list[str]:
    parts = [
        "--prefold-dir",
        str(args.prefold_dir),
        "--atlas-root",
        str(args.atlas_root),
        "--backend",
        str(args.backend),
        "--mpnn-input",
        str(args.mpnn_input),
        "--output-root",
        str(args.output_root),
        "--fett-stage",
        stage,
        "--skip-mpnn",
        "--skip-folding",
        "--max-decoy-sequences",
        str(args.max_decoy_sequences),
        "--paths-config",
        str(args.paths_config),
        "--runtime-config",
        str(args.runtime_config),
    ]
    _append_optional(parts, "--msa-cache-dir", args.msa_cache_dir)
    if args.plan_only:
        parts.append("--plan-only")
    return parts


def prepare_from_args(args: argparse.Namespace) -> int:
    decoy_fold.main(_prepare_args(args, stage="sequence_validation"))
    return int(decoy_fold.main(_prepare_args(args, stage="msa_resolution")) or 0)


def submit_from_args(args: argparse.Namespace) -> int:
    parts = [
        "submit",
        "--fold-dir",
        str(args.fold_dir),
        "--output-root",
        str(args.output_root),
        "--execution-mode",
        str(args.execution_mode),
        "--paths-config",
        str(args.paths_config),
        "--runtime-config",
        str(args.runtime_config),
        "--slurm-bundle-mode",
        str(args.slurm_bundle_mode),
    ]
    _append_optional(parts, "--backend", args.backend)
    _append_optional(parts, "--slurm-partition", args.slurm_partition)
    _append_optional(parts, "--slurm-gpus", args.slurm_gpus)
    _append_optional(parts, "--slurm-cpus-per-task", args.slurm_cpus_per_task)
    _append_optional(parts, "--slurm-mem", args.slurm_mem)
    _append_optional(parts, "--slurm-time", args.slurm_time)
    _append_optional(parts, "--slurm-max-concurrent", args.slurm_max_concurrent)
    _append_optional(parts, "--fold-workers-per-gpu", args.fold_workers_per_gpu)
    if args.dry_run:
        parts.append("--dry-run")
    return int(fold_jobs.main(parts) or 0)


def status_from_args(args: argparse.Namespace) -> int:
    parts = ["status", "--output-root", str(args.output_root)]
    _append_optional(parts, "--fold-dir", args.fold_dir)
    _append_optional(parts, "--manifest", args.manifest)
    return int(fold_jobs.main(parts) or 0)


def qc_from_args(args: argparse.Namespace) -> int:
    parts = ["--fold-dir", str(args.fold_dir), "--output-root", str(args.output_root)]
    for validation in args.validation or []:
        parts.extend(["--validation", str(validation)])
    _append_optional(parts, "--out", args.out)
    _append_optional(parts, "--figure-out", args.figure_out)
    if args.skip_figure:
        parts.append("--skip-figure")
    return int(fold_qc.main(parts) or 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "prepare":
        return prepare_from_args(args)
    if args.action == "submit":
        return submit_from_args(args)
    if args.action == "status":
        return status_from_args(args)
    if args.action == "qc":
        return qc_from_args(args)
    parser.error(f"unknown fold action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
