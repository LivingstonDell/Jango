#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from jango.paths import RunPaths


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


def add_ml_labels(
    input_csv: Path,
    output_csv: Path,
    source: str,
    label: int,
    label_type: str,
    label_confidence: str,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    df["source"] = source
    df["label"] = label
    df["label_type"] = label_type
    df["label_confidence"] = label_confidence

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Atlas manifest/features, then add ML labels. Prefer nanobody-pipeline for new runs."
    )

    parser.add_argument("--output-root", type=Path, default=None, help="User-selected Jango run root.")
    parser.add_argument("--project-root", type=Path, default=None, help="Deprecated alias for --output-root.")
    parser.add_argument("--atlas-root", type=Path, default=env_path("NBIA_ROOT", required=True))
    parser.add_argument("--raw-dir", type=Path, default=None)

    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--checksums-out", type=Path, default=None)
    parser.add_argument("--ml-manifest-out", type=Path, default=None)

    parser.add_argument("--features-out", type=Path, default=None)
    parser.add_argument("--ml-features-out", type=Path, default=None)

    parser.add_argument("--contact-cutoff", type=float, default=5.0)

    parser.add_argument("--source", default="sabdab_1_2")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--label-type", default="positive_true_complex")
    parser.add_argument("--label-confidence", default="high")

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

    args.raw_dir = resolve_path(args.raw_dir, run.root) if args.raw_dir is not None else run.raw_source(args.source).tarballs
    args.manifest_out = resolve_path(args.manifest_out, run.root) if args.manifest_out is not None else run.manifests_native / "manifest.csv"
    args.checksums_out = resolve_path(args.checksums_out, run.root) if args.checksums_out is not None else run.manifests_native / "checksums.sha256"
    args.ml_manifest_out = resolve_path(args.ml_manifest_out, run.root) if args.ml_manifest_out is not None else run.manifests_native / "ml_positive_manifest.csv"
    args.features_out = resolve_path(args.features_out, run.root) if args.features_out is not None else run.results_native_tables / "native_interface_features.csv"
    args.ml_features_out = resolve_path(args.ml_features_out, run.root) if args.ml_features_out is not None else run.results_native_tables / "native_interface_features_ml.csv"

    atlas_src = args.atlas_root / "src"
    if not atlas_src.exists():
        raise SystemExit(f"Atlas src directory not found: {atlas_src}")

    if not args.raw_dir.exists():
        raise SystemExit(f"Raw tarball directory not found: {args.raw_dir}")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NBIA_ROOT"] = str(args.atlas_root)
    env["DELPHI_ROOT"] = str(run.root)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(atlas_src) if not existing_pythonpath else str(atlas_src) + os.pathsep + existing_pythonpath

    run_command(
        [
            sys.executable,
            "-m",
            "nbia.cli",
            "manifest",
            "--raw-dir",
            args.raw_dir,
            "--out",
            args.manifest_out,
            "--checksums",
            args.checksums_out,
        ],
        env=env,
        cwd=run.root,
    )

    manifest = add_ml_labels(
        input_csv=args.manifest_out,
        output_csv=args.ml_manifest_out,
        source=args.source,
        label=args.label,
        label_type=args.label_type,
        label_confidence=args.label_confidence,
    )
    print(f"Wrote ML manifest: {args.ml_manifest_out} ({len(manifest)} rows)")

    run_command(
        [
            sys.executable,
            "-m",
            "nbia.cli",
            "features-native",
            "--raw-dir",
            args.raw_dir,
            "--out",
            args.features_out,
            "--contact-cutoff",
            str(args.contact_cutoff),
        ],
        env=env,
        cwd=run.root,
    )

    features = pd.read_csv(args.features_out)

    label_cols = [
        "structure_id",
        "source",
        "label",
        "label_type",
        "label_confidence",
    ]
    features_ml = features.merge(
        manifest[label_cols],
        on="structure_id",
        how="left",
        validate="one_to_one",
    )

    args.ml_features_out.parent.mkdir(parents=True, exist_ok=True)
    features_ml.to_csv(args.ml_features_out, index=False)

    print(f"Wrote native features: {args.features_out} ({len(features)} rows)")
    print(f"Wrote ML native features: {args.ml_features_out} ({len(features_ml)} rows)")


if __name__ == "__main__":
    main()