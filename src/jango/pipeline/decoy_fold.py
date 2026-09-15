"""Decoy folding and validation pipeline entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import time

from jango.paths import RunPaths

from ._common import (
    EXECUTION_MODES,
    FOLD_BACKENDS,
    MSA_FAILURE_POLICIES,
    MSA_FORMATS,
    MSA_MODES,
    MSA_PAIRINGS,
    MSA_POLICIES,
    MSA_PROVIDERS,
    add_atlas_root,
    env_int,
    add_output_root,
    add_plan_only,
    ensure_output_root_outside_repo,
    env_bool,
    env_path,
    env_value,
)
from .execution import CommandRunner, PipelineCommand, build_env, ensure_dir, ensure_file, nbia_command
from .jobs import patch_boltz_cache, patch_proteinmpnn_tsv
from .metadata import FOLD_METADATA, PREFOLD_METADATA, FoldMetadata, read_prefold_metadata, write_metadata


STAGE_DESCRIPTION = "ProteinMPNN preparation/execution, backend folding, and structural validation"
FETT_STAGE_CHOICES = ("all", "prepare", "proteinmpnn", "sequence_validation", "msa_resolution")


LEGACY_MSA_POLICY_TO_MODE = {
    "required": "required",
    "optional": "optional",
    "unsupported": "disabled",
}


def env_msa_mode() -> str | None:
    if env_bool("MSA_REQUIRE"):
        return "required"
    return env_value("MSA_MODE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decoy-fold",
        description="Run ProteinMPNN, backend decoy folding, and validation from pre-fold manifests.",
    )
    backend_default = env_value("FOLDING_BACKEND")
    msa_mode_default = env_msa_mode()
    parser.add_argument("--prefold-dir", required=True, type=Path)
    add_atlas_root(parser)
    parser.add_argument("--backend", required=backend_default is None, default=backend_default, choices=FOLD_BACKENDS, help="Backend name; defaults from FOLDING_BACKEND when set.")
    parser.add_argument("--msa-mode", choices=MSA_MODES, default=msa_mode_default, help="Required unless legacy --msa-policy is supplied; defaults from MSA_REQUIRE/MSA_MODE when set.")
    parser.add_argument("--msa-policy", choices=MSA_POLICIES, default=None, help="Legacy compatibility alias for --msa-mode.")
    parser.add_argument("--msa-provider", choices=MSA_PROVIDERS, default=env_value("MSA_PROVIDER"))
    parser.add_argument("--msa-cache-dir", "--msa-cache", dest="msa_cache_dir", type=Path, default=env_path("MSA_CACHE_ROOT") or env_path("MSA_CACHE_DIR"))
    parser.add_argument("--msa-input", type=Path, default=None)
    parser.add_argument("--msa-format", choices=MSA_FORMATS, default=env_value("MSA_FORMAT"))
    parser.add_argument("--msa-database", default=None)
    parser.add_argument("--msa-max-sequences", type=int, default=None)
    parser.add_argument("--msa-pairing", choices=MSA_PAIRINGS, default=env_value("MSA_PAIRING", "none"))
    parser.add_argument("--msa-reuse-policy", "--msa-design-policy", dest="msa_reuse_policy", default=env_value("MSA_DESIGN_POLICY", env_value("MSA_REUSE_POLICY", "exact_sequence")))
    parser.add_argument("--msa-script", type=Path, default=env_path("MSA_SCRIPT"))
    parser.add_argument("--msa-tool", default=env_value("MSA_TOOL", "esmfold2"))
    parser.add_argument("--msa-build-if-missing", action="store_true", default=env_bool("MSA_BUILD_IF_MISSING"))
    parser.add_argument("--msa-failure-policy", choices=MSA_FAILURE_POLICIES, default=env_value("MSA_FAILURE_POLICY", "abort"))
    parser.add_argument("--allow-single-sequence-fallback", action="store_true")
    parser.add_argument("--max-decoy-sequences", type=int, default=5, help="Maximum ProteinMPNN redesigned sequences sent to folding per structure/mode; native controls are added separately.")
    parser.add_argument("--proteinmpnn-home", type=Path, default=env_path("PROTEINMPNN_HOME"))
    parser.add_argument("--proteinmpnn-python", type=Path, default=env_path("PROTEINMPNN_PYTHON"))
    parser.add_argument("--mpnn-artifact-root", type=Path, default=env_path("JANGO_MPNN_ROOT"), help="Optional ProteinMPNN artifact root; completed outputs are mirrored to <root>/<mode>.")
    parser.add_argument("--mpnn-input", type=Path, default=None, help="Existing ProteinMPNN artifact directory to reuse as input. Accepts either <root>/<mode> or a root containing <mode>/.")
    parser.add_argument("--boltz-executable", type=Path, default=env_path("BOLTZ_EXECUTABLE") or env_path("BOLTZ"))
    parser.add_argument("--boltz-cache", type=Path, default=env_path("BOLTZ_CACHE"))
    parser.add_argument("--esmfold2-python", type=Path, default=env_path("ESMFOLD2_PYTHON"))
    parser.add_argument("--esmfold2-root", type=Path, default=env_path("ESM_ROOT") or env_path("ESMFOLD2_ROOT"))
    parser.add_argument("--esmfold2-model-id-or-path", default=env_value("ESMFOLD2_MODEL"))
    parser.add_argument("--esmfold2-cache-dir", type=Path, default=env_path("ESMFOLD2_CACHE"))
    parser.add_argument("--esmfold2-device", default=env_value("ESMFOLD2_DEVICE"))
    parser.add_argument("--esmfold2-num-sampling-steps", type=int, default=env_int("ESMFOLD2_NUM_SAMPLING_STEPS"))
    parser.add_argument("--esmfold2-num-diffusion-samples", type=int, default=env_int("ESMFOLD2_NUM_DIFFUSION_SAMPLES", 1))
    parser.add_argument("--esmfold2-seed", type=int, default=env_int("ESMFOLD2_SEED"))
    parser.add_argument("--opendde-python", type=Path, default=env_path("OPENDDE_PYTHON"))
    parser.add_argument("--opendde-executable", type=Path, default=env_path("OPENDDE_EXECUTABLE") or env_path("OPENDDE_BIN"))
    parser.add_argument("--opendde-root-dir", type=Path, default=env_path("OPENDDE_ROOT_DIR") or env_path("OPENDDE_MODEL_ROOT"))
    parser.add_argument("--opendde-model-name", default=env_value("OPENDDE_MODEL_NAME"))
    parser.add_argument("--opendde-checkpoint", type=Path, default=env_path("OPENDDE_CHECKPOINT"))
    parser.add_argument("--opendde-seeds", default=env_value("OPENDDE_SEEDS"))
    parser.add_argument("--opendde-cycle", default=env_value("OPENDDE_CYCLE"))
    parser.add_argument("--opendde-step", default=env_value("OPENDDE_STEP"))
    parser.add_argument("--opendde-samples", default=env_value("OPENDDE_SAMPLES"))
    parser.add_argument("--opendde-dtype", default=env_value("OPENDDE_DTYPE"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fett-stage", choices=FETT_STAGE_CHOICES, default="all", help=argparse.SUPPRESS)
    parser.add_argument("--skip-mpnn", action="store_true", help="Resume/debug only: do not launch generated ProteinMPNN jobs.")
    parser.add_argument("--skip-folding", action="store_true", help="Resume/debug only: do not launch generated backend folding jobs.")
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default=env_value("EXECUTION_MODE", "local"), help="Fold-job execution mode: local shell runner or Slurm submission.")
    parser.add_argument("--slurm-partition", default=env_value("SLURM_PARTITION"))
    parser.add_argument("--slurm-gpus", default=env_value("SLURM_GPUS"))
    parser.add_argument("--slurm-cpus-per-task", default=env_value("SLURM_CPUS_PER_TASK"))
    parser.add_argument("--slurm-mem", default=env_value("SLURM_MEM"))
    parser.add_argument("--slurm-time", default=env_value("SLURM_TIME"))
    parser.add_argument("--paths-config", type=Path, default=Path(env_value("JANGO_PATHS_CONFIG", "configs/test/paths.env")))
    parser.add_argument("--runtime-config", type=Path, default=Path(env_value("JANGO_RUNTIME_CONFIG", "configs/test/esmfold2.env")))
    add_output_root(parser)
    add_plan_only(parser)
    return parser


def fold_root(output_root: Path, experiment: str) -> Path:
    return RunPaths.from_output_root(output_root).decoy_layout(experiment).manifest_dir


def _safe_artifact_component(value: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"unsafe artifact component: {value!r}")
    return text


def _copy_file_if_present(source: Path, target_dir: Path) -> int:
    if not source.is_file():
        return 0
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_dir / source.name)
    return 1


def mirror_mpnn_artifacts(*, mode: str, work_dir: Path, tables_dir: Path, artifact_root: Path | None, experiment: str) -> None:
    if artifact_root is None:
        return
    target = artifact_root.expanduser() / _safe_artifact_component(mode)
    mpnn_source = work_dir / "mpnn_outputs"
    copied_fastas = 0
    if mpnn_source.is_dir():
        mpnn_target = target / "mpnn_outputs"
        shutil.copytree(mpnn_source, mpnn_target, dirs_exist_ok=True)
        copied_fastas = len(list(mpnn_target.rglob("*.fa")))
    copied_files = 0
    for source in [
        tables_dir / "decoy_mpnn_job_manifest.csv",
        tables_dir / "proteinmpnn_execution_status.csv",
        tables_dir / "decoy_prepare_status.csv",
        work_dir / "job_bundle" / "proteinmpnn_jobs.tsv",
        work_dir / "job_bundle" / "run_proteinmpnn_jobs.sh",
    ]:
        copied_files += _copy_file_if_present(source, target / ("job_bundle" if source.parent.name == "job_bundle" else "tables"))
    manifest = {
        "experiment": experiment,
        "redesign_mode": mode,
        "source_work_dir": str(work_dir),
        "source_tables_dir": str(tables_dir),
        "copied_fasta_files": copied_fastas,
        "copied_metadata_files": copied_files,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Mirrored ProteinMPNN artifacts for {mode}: {target} ({copied_fastas} FASTA files)")


MPNN_TABLE_FILES = (
    "decoy_mpnn_job_manifest.csv",
    "proteinmpnn_execution_status.csv",
    "decoy_prepare_status.csv",
)
MPNN_JOB_BUNDLE_FILES = ("proteinmpnn_jobs.tsv", "run_proteinmpnn_jobs.sh")


def resolve_mpnn_input_dir(mpnn_input: Path, mode: str) -> Path:
    """Return the concrete artifact directory for one redesign mode."""

    root = mpnn_input.expanduser()
    candidates = [root / _safe_artifact_component(mode), root]
    for candidate in candidates:
        if (candidate / "mpnn_outputs").is_dir():
            manifest_path = candidate / "artifact_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                artifact_mode = str(manifest.get("redesign_mode", ""))
                if artifact_mode and artifact_mode != mode:
                    raise SystemExit(f"ProteinMPNN artifact mode mismatch for {candidate}: expected {mode}, found {artifact_mode}")
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"ProteinMPNN artifact mpnn_outputs directory not found for mode {mode}: searched {searched}")


def _copy_artifact_file(source: Path, destination_dir: Path) -> bool:
    if not source.exists():
        return False
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_dir / source.name)
    return True


def seed_mpnn_input_artifacts(*, mode: str, work_dir: Path, tables_dir: Path, mpnn_input: Path | None, plan_only: bool = False) -> None:
    """Seed the run-local ProteinMPNN layout from a reusable artifact directory."""

    if mpnn_input is None:
        return
    artifact_dir = resolve_mpnn_input_dir(mpnn_input, mode)
    source_outputs = artifact_dir / "mpnn_outputs"
    target_outputs = work_dir / "mpnn_outputs"
    fasta_count = len(list(source_outputs.rglob("*.fa"))) + len(list(source_outputs.rglob("*.fasta")))
    if fasta_count == 0:
        raise SystemExit(f"ProteinMPNN artifact contains no FASTA files: {source_outputs}")
    if plan_only:
        print(f"[plan] seed ProteinMPNN artifacts for {mode}: {artifact_dir} -> {work_dir}")
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    source_resolved = source_outputs.expanduser().resolve()
    target_resolved = target_outputs.expanduser().resolve() if target_outputs.exists() else target_outputs.expanduser().absolute()
    if source_resolved != target_resolved:
        shutil.copytree(source_outputs, target_outputs, dirs_exist_ok=True)

    copied_tables = 0
    for name in MPNN_TABLE_FILES:
        copied_tables += int(_copy_artifact_file(artifact_dir / "tables" / name, tables_dir))
        if not (tables_dir / name).exists():
            copied_tables += int(_copy_artifact_file(artifact_dir / name, tables_dir))

    copied_bundle = 0
    bundle_dir = work_dir / "job_bundle"
    for name in MPNN_JOB_BUNDLE_FILES:
        copied_bundle += int(_copy_artifact_file(artifact_dir / "job_bundle" / name, bundle_dir))
        if not (bundle_dir / name).exists():
            copied_bundle += int(_copy_artifact_file(artifact_dir / name, bundle_dir))

    if not (tables_dir / "decoy_mpnn_job_manifest.csv").exists():
        expected = artifact_dir / "tables" / "decoy_mpnn_job_manifest.csv"
        raise SystemExit(f"ProteinMPNN artifact is missing required table: {expected}")
    print(f"Seeded ProteinMPNN artifacts for {mode}: {artifact_dir} -> {work_dir} ({fasta_count} FASTA files, {copied_tables} table files, {copied_bundle} job files)")


def normalize_msa_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.msa_policy is None and args.msa_mode is None:
        parser.error("--msa-mode is required; use disabled, backend-managed, optional, or required")
    if args.msa_policy is not None:
        legacy_mode = LEGACY_MSA_POLICY_TO_MODE[args.msa_policy]
        if args.msa_mode is not None and args.msa_mode != legacy_mode:
            parser.error(f"--msa-policy {args.msa_policy!r} conflicts with --msa-mode {args.msa_mode!r}")
        args.msa_mode = legacy_mode
        if args.msa_provider is None:
            if args.msa_policy in {"required", "optional"} and args.msa_cache_dir is not None:
                args.msa_provider = "precomputed"
            else:
                args.msa_provider = "none"
    assert args.msa_mode is not None
    if args.msa_provider is None:
        if args.msa_mode == "disabled":
            args.msa_provider = "none"
        elif args.msa_mode == "backend-managed":
            args.msa_provider = "backend-managed"
        else:
            parser.error("--msa-provider is required when --msa-mode is required or optional")


def validate_msa_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    normalize_msa_args(parser, args)
    if args.msa_mode == "disabled" and args.msa_provider != "none":
        parser.error("--msa-mode disabled requires --msa-provider none")
    if args.msa_mode == "backend-managed" and args.msa_provider != "backend-managed":
        parser.error("--msa-mode backend-managed requires --msa-provider backend-managed")
    if args.msa_mode in {"required", "optional"} and args.msa_provider == "none":
        if args.msa_mode == "optional" and args.allow_single_sequence_fallback:
            return
        parser.error("MSA was requested but no MSA provider was configured")
    if args.msa_provider == "precomputed":
        if args.msa_input is None:
            parser.error("--msa-input is required for --msa-provider precomputed")
        if args.msa_cache_dir is None:
            parser.error("--msa-cache-dir is required for --msa-provider precomputed")
    if args.msa_provider == "mmseqs2":
        parser.error("MMseqs2 MSA generation is deferred until database paths and commands are configured")
    if args.msa_max_sequences is not None and args.msa_max_sequences <= 0:
        parser.error("--msa-max-sequences must be positive")
    if args.msa_pairing == "paired":
        parser.error("Current DELPHI decoy folding is monomeric; paired MSA input is not valid here")
    if args.backend == "esmfold2" and args.msa_mode == "optional" and args.msa_provider == "none" and not args.allow_single_sequence_fallback:
        parser.error("ESMFold2 optional MSA without a provider requires --allow-single-sequence-fallback")
    if args.backend == "esmfold2" and args.msa_mode == "backend-managed":
        parser.error("ESMFold2 cannot use backend-managed MSA in the current local runner")
    if args.backend == "opendde" and args.msa_mode == "backend-managed":
        parser.error("OpenDDE backend-managed MSA search is not wired in Jango; use abforge_get_or_build or precomputed MSA")
    if args.msa_provider == "abforge_get_or_build" and args.msa_script is None:
        parser.error("--msa-script/MSA_SCRIPT is required for --msa-provider abforge_get_or_build")
    if args.backend == "boltz2" and args.msa_provider == "precomputed":
        parser.error("Boltz2 precomputed MSA input is deferred until local YAML/CLI support is proven")
    if args.backend == "boltz2" and args.msa_mode == "optional" and args.msa_provider == "backend-managed":
        parser.error("Use --msa-mode backend-managed for Boltz2 MSA-server runs")


def validate_stage_args(parser: argparse.ArgumentParser, args: argparse.Namespace, metadata_experiment: str, metadata_backend: str) -> None:
    if args.backend != metadata_backend:
        parser.error(f"--backend {args.backend!r} does not match prefold metadata backend {metadata_backend!r}")
    validate_msa_args(parser, args)
    if args.mpnn_input is not None and args.fett_stage in {"all", "proteinmpnn"} and not args.skip_mpnn:
        parser.error("--mpnn-input reuses existing ProteinMPNN outputs; use --skip-mpnn or a post-ProteinMPNN --fett-stage")
    if not args.skip_mpnn and not args.plan_only:
        if args.proteinmpnn_home is None or args.proteinmpnn_python is None:
            parser.error("--proteinmpnn-home and --proteinmpnn-python are required unless --skip-mpnn or --plan-only is used")
    if args.backend == "boltz2" and not args.skip_folding and not args.plan_only:
        if args.boltz_executable is None or args.boltz_cache is None:
            parser.error("--boltz-executable and --boltz-cache are required for boltz2 folding unless --skip-folding is used")
    if args.max_decoy_sequences <= 0:
        parser.error("--max-decoy-sequences must be positive")
    if args.backend == "esmfold2" and not args.skip_folding and not args.plan_only:
        missing = [
            name
            for name, value in (
                ("--esmfold2-python", args.esmfold2_python),
                ("--esmfold2-root", args.esmfold2_root),
                ("--esmfold2-model-id-or-path", args.esmfold2_model_id_or_path),
                ("--esmfold2-cache-dir", args.esmfold2_cache_dir),
                ("--esmfold2-device", args.esmfold2_device),
            )
            if value is None
        ]
        if missing:
            parser.error("required for esmfold2 folding: " + ", ".join(missing))
    if args.backend == "opendde" and not args.skip_folding and not args.plan_only:
        missing = [
            name
            for name, value in (
                ("--opendde-python", args.opendde_python),
                ("--opendde-executable", args.opendde_executable),
                ("--opendde-root-dir", args.opendde_root_dir),
            )
            if value is None
        ]
        if missing:
            parser.error("required for opendde folding: " + ", ".join(missing))


def add_msa_command_args(parts: list, args: argparse.Namespace, experiment: str) -> None:
    parts.extend(["--msa-mode", args.msa_mode, "--msa-provider", args.msa_provider])
    if args.msa_cache_dir is not None:
        parts.extend(["--msa-cache-dir", args.msa_cache_dir])
    if args.msa_input is not None:
        parts.extend(["--msa-input", args.msa_input])
    if args.msa_format is not None:
        parts.extend(["--msa-format", args.msa_format])
    if args.msa_database is not None:
        parts.extend(["--msa-database", args.msa_database])
    if args.msa_max_sequences is not None:
        parts.extend(["--msa-max-sequences", str(args.msa_max_sequences)])
    parts.extend(["--msa-pairing", args.msa_pairing, "--msa-reuse-policy", args.msa_reuse_policy])
    if args.msa_script is not None:
        parts.extend(["--msa-script", args.msa_script])
    if args.msa_tool is not None:
        parts.extend(["--msa-tool", args.msa_tool])
    if args.msa_build_if_missing:
        parts.append("--msa-build-if-missing")
    parts.extend(["--msa-failure-policy", args.msa_failure_policy])
    if args.allow_single_sequence_fallback:
        parts.append("--allow-single-sequence-fallback")
    parts.extend(["--experiment-namespace", experiment])


def validate_command(args: argparse.Namespace, case_manifest: Path, raw_dir: Path, work_dir: Path, tables_dir: Path, experiment: str, mode_policy: str, *, validation_phase: str = "full"):
    parts: list = [
        "decoy-validate",
        "--case-manifest", case_manifest,
        "--raw-dir", raw_dir,
        "--work-dir", work_dir,
        "--tables-dir", tables_dir,
        "--fold-backend", args.backend,
        "--validation-phase", validation_phase,
    ]
    add_msa_command_args(parts, args, experiment)
    parts.extend(["--max-decoy-sequences", str(args.max_decoy_sequences), "--mode-policy", mode_policy])
    if args.backend == "esmfold2":
        parts.extend([
            "--esmfold2-python", args.esmfold2_python or "<required-esmfold2-python>",
            "--esmfold2-root", args.esmfold2_root or "<required-esmfold2-root>",
            "--esmfold2-model-id-or-path", args.esmfold2_model_id_or_path or "<required-esmfold2-model>",
            "--esmfold2-cache-dir", args.esmfold2_cache_dir or "<required-esmfold2-cache>",
            "--esmfold2-device", args.esmfold2_device or "<required-esmfold2-device>",
        ])
        if args.esmfold2_num_sampling_steps is not None:
            parts.extend(["--esmfold2-num-sampling-steps", str(args.esmfold2_num_sampling_steps)])
        if args.esmfold2_num_diffusion_samples != 1:
            parts.extend(["--esmfold2-num-diffusion-samples", str(args.esmfold2_num_diffusion_samples)])
        if args.esmfold2_seed is not None:
            parts.extend(["--esmfold2-seed", str(args.esmfold2_seed)])
    if args.backend == "opendde":
        parts.extend([
            "--opendde-python", args.opendde_python or "<required-opendde-python>",
            "--opendde-executable", args.opendde_executable or "<required-opendde-executable>",
            "--opendde-root-dir", args.opendde_root_dir or "<required-opendde-root>",
        ])
        optional_pairs = [
            ("--opendde-model-name", args.opendde_model_name),
            ("--opendde-checkpoint", args.opendde_checkpoint),
            ("--opendde-seeds", args.opendde_seeds),
            ("--opendde-cycle", args.opendde_cycle),
            ("--opendde-step", args.opendde_step),
            ("--opendde-samples", args.opendde_samples),
            ("--opendde-dtype", args.opendde_dtype),
        ]
        for option, value in optional_pairs:
            if value is not None:
                parts.extend([option, value])
    return nbia_command(*parts, name="decoy-validate")



def _write_prepare_status(path: Path, jobs_tsv: Path, *, mode: str) -> None:
    rows: list[dict[str, object]] = []
    if jobs_tsv.exists():
        with jobs_tsv.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows.append({"mode": mode, "structure_id": row.get("structure_id", ""), "status": "prepared", "jobs_tsv": str(jobs_tsv)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "structure_id", "status", "jobs_tsv"])
        writer.writeheader()
        writer.writerows(rows)


def _write_proteinmpnn_status(path: Path, jobs_tsv: Path, *, mode: str, status: str, return_code: int, elapsed_seconds: float | None = None) -> None:
    rows: list[dict[str, object]] = []
    if jobs_tsv.exists():
        with jobs_tsv.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows.append(
                    {
                        "mode": mode,
                        "case_id": row.get("structure_id", ""),
                        "structure_id": row.get("structure_id", ""),
                        "job_status": status,
                        "return_code": return_code,
                        "output_dir": row.get("mpnn_output_dir", row.get("output_dir", "")),
                        "elapsed_seconds": f"{elapsed_seconds:.3f}" if elapsed_seconds is not None else "",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "case_id", "structure_id", "job_status", "return_code", "output_dir", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)


def run_folding_jobs(args: argparse.Namespace, runner: CommandRunner, work_dir: Path, output_root: Path) -> bool:
    if args.skip_folding:
        print("Skipping backend folding by request.")
        return True
    if args.execution_mode == "slurm":
        from .fold_execution import SlurmConfig, submit_job_bundle

        jobs_tsv = work_dir / "job_bundle" / f"{args.backend}_monomer_jobs.tsv"
        ensure_file(jobs_tsv, f"{args.backend} monomer TSV")
        slurm_config = SlurmConfig.from_values(
            output_root=output_root,
            partition=args.slurm_partition,
            gpus=args.slurm_gpus,
            cpus_per_task=args.slurm_cpus_per_task,
            mem=args.slurm_mem,
            time=args.slurm_time,
            paths_config=args.paths_config,
            runtime_config=args.runtime_config,
        )
        manifest = submit_job_bundle(
            jobs_tsv=jobs_tsv,
            work_dir=work_dir,
            output_root=output_root,
            backend=args.backend,
            slurm_config=slurm_config,
        )
        counts = manifest["submission_status"].value_counts(dropna=False).to_dict() if not manifest.empty else {}
        print(f"Recorded Slurm fold submission manifest for {len(manifest)} jobs: {counts}")
        return False

    if args.backend == "boltz2":
        monomer_sh = work_dir / "job_bundle" / "run_boltz2_monomer_jobs.sh"
        monomer_tsv = work_dir / "job_bundle" / "boltz2_monomer_jobs.tsv"
        if args.boltz_cache is not None:
            if monomer_sh.exists():
                patch_boltz_cache(monomer_sh, args.boltz_cache)
            if monomer_tsv.exists():
                patch_boltz_cache(monomer_tsv, args.boltz_cache)
        ensure_file(monomer_sh, "Boltz2 monomer runner")
        ensure_file(monomer_tsv, "Boltz2 monomer TSV")
        runner.run(PipelineCommand("boltz2-monomer-fold", ("bash", monomer_sh, monomer_tsv)))
        return True
    if args.backend == "esmfold2":
        monomer_sh = work_dir / "job_bundle" / "run_esmfold2_monomer_jobs.sh"
        monomer_tsv = work_dir / "job_bundle" / "esmfold2_monomer_jobs.tsv"
        ensure_file(monomer_sh, "ESMFold2 monomer runner")
        ensure_file(monomer_tsv, "ESMFold2 monomer TSV")
        runner.run(PipelineCommand("esmfold2-monomer-fold", ("bash", monomer_sh, monomer_tsv)))
        return True
    if args.backend == "opendde":
        monomer_sh = work_dir / "job_bundle" / "run_opendde_monomer_jobs.sh"
        monomer_tsv = work_dir / "job_bundle" / "opendde_monomer_jobs.tsv"
        ensure_file(monomer_sh, "OpenDDE monomer runner")
        ensure_file(monomer_tsv, "OpenDDE monomer TSV")
        runner.run(PipelineCommand("opendde-monomer-fold", ("bash", monomer_sh, monomer_tsv)))
        return True
    raise SystemExit(f"Unsupported backend: {args.backend}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_root = ensure_output_root_outside_repo(args.output_root)
    except ValueError as exc:
        parser.error(str(exc))

    prefold_metadata_path = args.prefold_dir / PREFOLD_METADATA
    ensure_file(prefold_metadata_path, "prefold metadata")
    prefold = read_prefold_metadata(prefold_metadata_path)
    validate_stage_args(parser, args, prefold.experiment, prefold.backend)

    extra_env = {
        "PROTEINMPNN_HOME": args.proteinmpnn_home,
        "PROTEINMPNN_PYTHON": args.proteinmpnn_python,
        "PROTEINMPNN_RUNNER": env_path("PROTEINMPNN_RUNNER"),
        "BOLTZ_EXECUTABLE": args.boltz_executable,
        "BOLTZ": args.boltz_executable,
        "BOLTZ_PYTHON": env_path("BOLTZ_PYTHON"),
        "BOLTZ_MODEL_ROOT": env_path("BOLTZ_MODEL_ROOT"),
        "BOLTZ_CACHE": args.boltz_cache,
        "ESMFOLD2_PYTHON": args.esmfold2_python,
        "ESM_ROOT": args.esmfold2_root,
        "ESMFOLD2_MODEL": args.esmfold2_model_id_or_path,
        "ESMFOLD2_CACHE": args.esmfold2_cache_dir,
        "ESMFOLD2_DEVICE": args.esmfold2_device,
        "ESMFOLD2_NUM_SAMPLING_STEPS": args.esmfold2_num_sampling_steps,
        "ESMFOLD2_NUM_DIFFUSION_SAMPLES": args.esmfold2_num_diffusion_samples,
        "ESMFOLD2_SEED": args.esmfold2_seed,
        "ESMC_MODEL_PATH": env_path("ESMC_MODEL_PATH"),
        "OPENDDE_PYTHON": args.opendde_python,
        "OPENDDE_EXECUTABLE": args.opendde_executable,
        "OPENDDE_BIN": args.opendde_executable,
        "OPENDDE_ROOT_DIR": args.opendde_root_dir,
        "OPENDDE_MODEL_ROOT": args.opendde_root_dir,
        "OPENDDE_MODEL_NAME": args.opendde_model_name,
        "OPENDDE_CHECKPOINT": args.opendde_checkpoint,
        "OPENDDE_SEEDS": args.opendde_seeds,
        "OPENDDE_CYCLE": args.opendde_cycle,
        "OPENDDE_STEP": args.opendde_step,
        "OPENDDE_SAMPLES": args.opendde_samples,
        "OPENDDE_DTYPE": args.opendde_dtype,
        "OPENDDE_RUNNER": env_path("OPENDDE_RUNNER"),
        "MSA_SCRIPT": env_path("MSA_SCRIPT"),
        "MSA_TOOL": env_value("MSA_TOOL"),
        "MSA_BUILD_IF_MISSING": env_value("MSA_BUILD_IF_MISSING"),
        "MSA_FAILURE_POLICY": args.msa_failure_policy,
        "MSA_DESIGN_POLICY": args.msa_reuse_policy,
        "EXECUTION_MODE": args.execution_mode,
        "DELPHI_MSA_MODE": args.msa_mode,
        "DELPHI_MSA_PROVIDER": args.msa_provider,
        "DELPHI_MSA_CACHE_DIR": args.msa_cache_dir,
        "DELPHI_MSA_INPUT": args.msa_input,
    }
    env = build_env(atlas_root=args.atlas_root, extra=extra_env)
    runner = CommandRunner(env=env, plan_only=args.plan_only)
    run_paths = RunPaths.from_output_root(output_root)
    decoy_layout = run_paths.decoy_layout(prefold.experiment)
    root = decoy_layout.manifest_dir
    raw_dir = Path(prefold.raw_dir)

    if args.plan_only:
        print(f"decoy-fold: {STAGE_DESCRIPTION}")
        print(f"experiment: {prefold.experiment}")
        print(f"backend: {args.backend}")
        print(f"msa mode: {args.msa_mode}")
        print(f"max decoy sequences per structure/mode: {args.max_decoy_sequences}")
        print(f"msa provider: {args.msa_provider}")
        print(f"msa failure policy: {args.msa_failure_policy}")
        print(f"msa design policy: {args.msa_reuse_policy}")
        print(f"execution mode: {args.execution_mode}")
        print(f"fold metadata dir: {root}")
    else:
        ensure_dir(args.atlas_root / "src", "Atlas src")
        ensure_dir(raw_dir, "raw tarball dir")
        root.mkdir(parents=True, exist_ok=True)

    mode_outputs: dict[str, dict[str, Path]] = {}
    for mode, manifest_text in prefold.mode_manifests.items():
        case_manifest = Path(manifest_text)
        tables_dir = decoy_layout.mode_tables_dir(mode)
        work_dir = decoy_layout.mode_work_dir(mode)
        if not args.plan_only:
            tables_dir.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)
        mpnn_manifest = tables_dir / "decoy_mpnn_job_manifest.csv"

        prepare = nbia_command(
            "decoy-prepare",
            "--case-manifest", case_manifest,
            "--raw-dir", raw_dir,
            "--work-dir", work_dir,
            "--out", mpnn_manifest,
            "--redesign-mode", mode,
            *( ["--limit", str(args.limit)] if args.limit is not None else [] ),
            name=f"decoy-prepare-{mode}",
        )

        proteinmpnn_tsv = work_dir / "job_bundle" / "proteinmpnn_jobs.tsv"
        proteinmpnn_sh = work_dir / "job_bundle" / "run_proteinmpnn_jobs.sh"
        prepare_status = tables_dir / "decoy_prepare_status.csv"
        proteinmpnn_status = tables_dir / "proteinmpnn_execution_status.csv"
        selected_designs = tables_dir / "decoy_selected_designs.csv"

        if args.fett_stage in {"sequence_validation", "msa_resolution"} or (args.fett_stage == "all" and args.skip_mpnn):
            seed_mpnn_input_artifacts(mode=mode, work_dir=work_dir, tables_dir=tables_dir, mpnn_input=args.mpnn_input, plan_only=args.plan_only)

        if args.fett_stage in {"all", "prepare"}:
            runner.run(prepare)
            if not args.plan_only:
                ensure_file(proteinmpnn_tsv, "ProteinMPNN TSV")
                ensure_file(proteinmpnn_sh, "ProteinMPNN runner")
                if args.proteinmpnn_python is not None:
                    patch_proteinmpnn_tsv(proteinmpnn_tsv, args.proteinmpnn_python)
                _write_prepare_status(prepare_status, proteinmpnn_tsv, mode=mode)

        if args.fett_stage in {"all", "proteinmpnn"}:
            if not args.plan_only:
                ensure_file(proteinmpnn_tsv, "ProteinMPNN TSV")
                ensure_file(proteinmpnn_sh, "ProteinMPNN runner")
                if args.proteinmpnn_python is not None:
                    patch_proteinmpnn_tsv(proteinmpnn_tsv, args.proteinmpnn_python)
            if args.skip_mpnn:
                print("Skipping ProteinMPNN by request.")
                if not args.plan_only:
                    _write_proteinmpnn_status(proteinmpnn_status, proteinmpnn_tsv, mode=mode, status="skipped", return_code=0)
                    mirror_mpnn_artifacts(mode=mode, work_dir=work_dir, tables_dir=tables_dir, artifact_root=args.mpnn_artifact_root, experiment=prefold.experiment)
            else:
                started = time.monotonic()
                runner.run(PipelineCommand(f"proteinmpnn-{mode}", ("bash", proteinmpnn_sh, proteinmpnn_tsv)))
                if not args.plan_only:
                    _write_proteinmpnn_status(proteinmpnn_status, proteinmpnn_tsv, mode=mode, status="completed", return_code=0, elapsed_seconds=time.monotonic() - started)
                    mirror_mpnn_artifacts(mode=mode, work_dir=work_dir, tables_dir=tables_dir, artifact_root=args.mpnn_artifact_root, experiment=prefold.experiment)

        if args.fett_stage in {"all", "sequence_validation"}:
            validate = validate_command(args, case_manifest, raw_dir, work_dir, tables_dir, prefold.experiment, prefold.mode_policy, validation_phase="sequence_only")
            runner.run(validate)

        if args.fett_stage in {"all", "msa_resolution"}:
            if args.fett_stage == "msa_resolution" and not args.plan_only:
                ensure_file(selected_designs, "selected design manifest from sequence_validation")
            validate = validate_command(args, case_manifest, raw_dir, work_dir, tables_dir, prefold.experiment, prefold.mode_policy, validation_phase="full")
            runner.run(validate)
            if args.fett_stage == "all":
                if not args.plan_only:
                    folding_completed = run_folding_jobs(args, runner, work_dir, output_root)
                    if folding_completed:
                        runner.run(validate)
                    else:
                        print(f"Slurm fold jobs recorded for mode {mode}; run fold-jobs status after Slurm completion before post-fold validation.")
                else:
                    runner.run(PipelineCommand(f"{args.backend}-monomer-fold-{mode}", ("bash", work_dir / "job_bundle" / f"run_{args.backend}_monomer_jobs.sh", work_dir / "job_bundle" / f"{args.backend}_monomer_jobs.tsv")))
                    runner.run(validate)

        mode_outputs[mode] = {
            "case_manifest": case_manifest,
            "tables_dir": tables_dir,
            "work_dir": work_dir,
            "mpnn_manifest": mpnn_manifest,
            "validation": tables_dir / "decoy_validation.csv",
        }

    metadata = FoldMetadata.create(
        experiment=prefold.experiment,
        backend=args.backend,
        execution_mode=args.execution_mode,
        msa_policy=args.msa_policy,
        msa_mode=args.msa_mode,
        msa_provider=args.msa_provider,
        msa_cache_dir=args.msa_cache_dir,
        msa_input=args.msa_input,
        msa_format=args.msa_format,
        msa_database=args.msa_database,
        msa_max_sequences=args.msa_max_sequences,
        msa_pairing=args.msa_pairing,
        msa_reuse_policy=args.msa_reuse_policy,
        msa_failure_policy=args.msa_failure_policy,
        allow_single_sequence_fallback=args.allow_single_sequence_fallback,
        max_decoy_sequences_per_structure=args.max_decoy_sequences,
        prefold_metadata=prefold_metadata_path,
        mode_outputs=mode_outputs,
    )
    metadata_path = root / FOLD_METADATA
    if args.plan_only:
        print(f"[plan] write fold metadata: {metadata_path}")
    else:
        write_metadata(metadata_path, metadata)
        print(f"Wrote fold metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
