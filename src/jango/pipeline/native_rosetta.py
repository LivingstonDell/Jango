#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from jango.paths import RunPaths


DEFAULT_ROSETTA_IMAGE = "rosettacommons/rosetta:latest"
DEFAULT_DOCKER_IMAGE = DEFAULT_ROSETTA_IMAGE
ROSETTA_RUNTIMES = ("docker", "apptainer", "local")


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


def env_path(name: str, required: bool = False) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if required:
        raise SystemExit(f"Required environment variable is not set: {name}")
    return None


def resolve_path(path: Path, root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).expanduser().resolve()


def run_command(cmd: list[str | Path], env: dict[str, str], cwd: Path | None = None) -> None:
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    result = subprocess.run([str(x) for x in cmd], env=env, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def add_ml_labels_from_manifest(
    table_csv: Path,
    ml_manifest_csv: Path,
    out_csv: Path,
) -> pd.DataFrame:
    table = pd.read_csv(table_csv)
    manifest = pd.read_csv(ml_manifest_csv)

    label_cols = [
        "structure_id",
        "source",
        "label",
        "label_type",
        "label_confidence",
    ]

    missing = [c for c in label_cols if c not in manifest.columns]
    if missing:
        raise ValueError(f"ML manifest missing columns: {missing}")

    labeled = table.merge(
        manifest[label_cols],
        on="structure_id",
        how="left",
        validate="one_to_one",
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_csv, index=False)
    return labeled


def rosetta_command_args(args: argparse.Namespace) -> list[str | Path]:
    image = args.rosetta_image or args.docker_image
    parts: list[str | Path] = ["--rosetta-runtime", args.rosetta_runtime]
    if image:
        parts.extend(["--rosetta-image", image])
    if args.rosetta_bin_dir is not None:
        parts.extend(["--rosetta-bin-dir", args.rosetta_bin_dir])
    if args.apptainer_cache_dir is not None:
        parts.extend(["--apptainer-cache-dir", args.apptainer_cache_dir])
    return parts


def validate_rosetta_args(args: argparse.Namespace) -> None:
    image = args.rosetta_image or args.docker_image
    if args.rosetta_runtime is None:
        raise SystemExit("--rosetta-runtime is required")
    if args.rosetta_runtime in {"docker", "apptainer"} and not image:
        raise SystemExit("--rosetta-image is required when --rosetta-runtime is docker or apptainer")
    if args.rosetta_runtime == "local" and args.rosetta_bin_dir is None:
        raise SystemExit("--rosetta-bin-dir is required when --rosetta-runtime local")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Atlas Rosetta native scoring and add ML labels. Prefer nanobody-pipeline for new runs.")

    parser.add_argument("--output-root", type=Path, default=None, help="User-selected Jango run root.")
    parser.add_argument("--project-root", type=Path, default=None, help="Deprecated alias for --output-root.")
    parser.add_argument("--atlas-root", type=Path, default=env_path("NBIA_ROOT", required=True))
    parser.add_argument("--source", default="sabdab_1_2")
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--ml-manifest", default=None, type=Path)
    parser.add_argument("--raw-dir", default=None, type=Path)

    parser.add_argument("--work-dir", default=None, type=Path)

    parser.add_argument("--relax-status-out", default=None, type=Path)
    parser.add_argument("--interface-out", default=None, type=Path)
    parser.add_argument("--interface-ml-out", default=None, type=Path)

    parser.add_argument("--rosetta-runtime", choices=ROSETTA_RUNTIMES, default=env_value("ROSETTA_RUNTIME"))
    parser.add_argument("--rosetta-image", default=env_value("ROSETTA_IMAGE"))
    parser.add_argument("--rosetta-bin-dir", type=Path, default=env_path("ROSETTA_BIN_DIR"))
    parser.add_argument("--apptainer-cache-dir", type=Path, default=env_path("APPTAINER_CACHEDIR"))
    parser.add_argument("--docker-image", default=None, help="Deprecated alias for --rosetta-image.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--skip-relax",
        action="store_true",
        help="Skip interface-shell FastRelax and only run InterfaceAnalyzer on native tarballs.",
    )
    parser.add_argument(
        "--state",
        choices=["native", "relaxed"],
        default="native",
        help="Which state to score with InterfaceAnalyzer.",
    )
    parser.add_argument(
        "--backfill-fullscore",
        action="store_true",
        help="Also run Rosetta score_jd2 to backfill regular score terms.",
    )

    return parser.parse_args()


def output_root_from_args(args: argparse.Namespace) -> Path:
    root = args.output_root or args.project_root
    if root is None:
        raise SystemExit("--output-root is required")
    return root


def main() -> None:
    args = parse_args()

    output_root = output_root_from_args(args)
    run = RunPaths.from_output_root(output_root)
    args.atlas_root = args.atlas_root.expanduser().resolve()

    args.manifest = resolve_path(args.manifest, run.root) if args.manifest is not None else run.manifests_native / "manifest.csv"
    args.ml_manifest = resolve_path(args.ml_manifest, run.root) if args.ml_manifest is not None else run.manifests_native / "ml_positive_manifest.csv"
    args.raw_dir = resolve_path(args.raw_dir, run.root) if args.raw_dir is not None else run.raw_source(args.source).tarballs
    args.work_dir = resolve_path(args.work_dir, run.root) if args.work_dir is not None else run.work_native / "rosetta"
    args.relax_status_out = resolve_path(args.relax_status_out, run.root) if args.relax_status_out is not None else run.results_native_tables / "rosetta_relax_status.csv"
    args.interface_out = resolve_path(args.interface_out, run.root) if args.interface_out is not None else run.results_native_tables / "rosetta_interface_native.csv"
    args.interface_ml_out = resolve_path(args.interface_ml_out, run.root) if args.interface_ml_out is not None else run.results_native_tables / "rosetta_interface_native_ml.csv"
    if args.rosetta_bin_dir is not None:
        args.rosetta_bin_dir = resolve_path(args.rosetta_bin_dir, run.root)
    if args.apptainer_cache_dir is not None:
        args.apptainer_cache_dir = resolve_path(args.apptainer_cache_dir, run.root)

    atlas_src = args.atlas_root / "src"
    if not atlas_src.exists():
        raise SystemExit(f"Atlas src directory not found: {atlas_src}")

    for path in [args.manifest, args.ml_manifest, args.raw_dir]:
        if not path.exists():
            raise SystemExit(f"Required path not found: {path}")

    validate_rosetta_args(args)
    rosetta_args = rosetta_command_args(args)

    env = os.environ.copy()
    env["NBIA_ROOT"] = str(args.atlas_root)
    env["DELPHI_ROOT"] = str(run.root)
    env["ROSETTA_RUNTIME"] = str(args.rosetta_runtime)
    image = args.rosetta_image or args.docker_image
    if image:
        env["ROSETTA_IMAGE"] = str(image)
    if args.rosetta_bin_dir is not None:
        env["ROSETTA_BIN_DIR"] = str(args.rosetta_bin_dir)
    if args.apptainer_cache_dir is not None:
        env["APPTAINER_CACHEDIR"] = str(args.apptainer_cache_dir)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(atlas_src) if not existing_pythonpath else str(atlas_src) + os.pathsep + existing_pythonpath

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.relax_status_out.parent.mkdir(parents=True, exist_ok=True)
    args.interface_out.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_relax:
        run_command(
            [
                sys.executable,
                "-m",
                "nbia.cli",
                "rosetta-relax",
                "--manifest",
                args.manifest,
                "--raw-dir",
                args.raw_dir,
                "--work-dir",
                args.work_dir,
                *rosetta_args,
                "--jobs",
                str(args.jobs),
                "--status-out",
                args.relax_status_out,
                *(["--limit", str(args.limit)] if args.limit is not None else []),
            ],
            env=env,
            cwd=run.root,
        )

    run_command(
        [
            sys.executable,
            "-m",
            "nbia.cli",
            "rosetta-interface",
            "--manifest",
            args.manifest,
            "--raw-dir",
            args.raw_dir,
            "--work-dir",
            args.work_dir,
            "--out",
            args.interface_out,
            "--state",
            args.state,
            *rosetta_args,
            "--jobs",
            str(args.jobs),
            *(["--limit", str(args.limit)] if args.limit is not None else []),
        ],
        env=env,
        cwd=run.root,
    )

    if args.backfill_fullscore:
        run_command(
            [
                sys.executable,
                "-m",
                "nbia.cli",
                "rosetta-fullscore-backfill",
                "--interface-csv",
                args.interface_out,
                "--manifest",
                args.manifest,
                "--raw-dir",
                args.raw_dir,
                "--work-dir",
                args.work_dir,
                "--out",
                args.interface_out,
                *rosetta_args,
                "--jobs",
                str(args.jobs),
                "--state",
                args.state,
                *(["--limit", str(args.limit)] if args.limit is not None else []),
            ],
            env=env,
            cwd=run.root,
        )

    labeled = add_ml_labels_from_manifest(
        table_csv=args.interface_out,
        ml_manifest_csv=args.ml_manifest,
        out_csv=args.interface_ml_out,
    )

    print(f"\nWrote Rosetta interface table: {args.interface_out}")
    print(f"Wrote ML-labeled Rosetta table: {args.interface_ml_out}")
    print(f"Rows: {len(labeled)}")


if __name__ == "__main__":
    main()