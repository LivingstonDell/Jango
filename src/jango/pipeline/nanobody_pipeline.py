"""Native nanobody structure pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jango.paths import RunPaths
from jango.preprocessing.native_inputs import InputKind, prepare_native_inputs, preprocessing_paths

from ._common import ROSETTA_RUNTIMES, add_atlas_root, add_output_root, add_plan_only, ensure_output_root_outside_repo, env_path, env_value
from .execution import CommandRunner, build_env, ensure_dir, nbia_command
from .labels import LabelConfig, add_label_columns, merge_label_columns


STAGE_DESCRIPTION = "native preprocessing, manifest/features, native Rosetta relax, and native relaxed interface scoring"
INPUT_KINDS = ("tarballs", "sabdab-pdb", "sabdab2-cif")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanobody-pipeline",
        description="Run native nanobody structure analysis through native relaxed Rosetta scoring.",
    )
    input_default = env_path("JANGO_RAW_TARBALLS")
    source_default = env_value("JANGO_SOURCE_LABEL")
    rosetta_runtime_default = env_value("ROSETTA_RUNTIME")
    parser.add_argument("--input", required=input_default is None, default=input_default, type=Path, help="Input tarball, raw PDB, or raw mmCIF directory.")
    parser.add_argument("--input-kind", required=True, choices=INPUT_KINDS)
    parser.add_argument("--metadata", type=Path, default=None, help="Required for sabdab-pdb and sabdab2-cif inputs.")
    add_output_root(parser)
    add_atlas_root(parser)
    parser.add_argument("--source", required=source_default is None, default=source_default, help="Dataset/source label, for example sabdab1_2.")
    parser.add_argument("--label", required=True, help="User-supplied label value for downstream ML/openDDE tables.")
    parser.add_argument("--role", required=True, help="User-supplied role value for the label schema.")
    parser.add_argument("--confidence", required=True, help="User-supplied confidence value for the label schema.")
    parser.add_argument("--rosetta-runtime", required=rosetta_runtime_default is None, default=rosetta_runtime_default, choices=ROSETTA_RUNTIMES)
    parser.add_argument("--rosetta-image", default=env_value("ROSETTA_IMAGE"), help="Rosetta container image when --rosetta-runtime docker/apptainer is used.")
    parser.add_argument("--rosetta-bin-dir", type=Path, default=env_path("ROSETTA_BIN_DIR"), help="Directory containing local Rosetta executables when --rosetta-runtime local is used.")
    parser.add_argument("--apptainer-cache-dir", type=Path, default=env_path("APPTAINER_CACHEDIR"), help="Optional Apptainer cache directory.")
    parser.add_argument("--docker-image", default=None, help="Deprecated alias for --rosetta-image.")
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument("--min-antigen-len", type=int, default=60)
    parser.add_argument("--max-antigen-len", type=int, default=250)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backfill-fullscore", action="store_true")
    add_plan_only(parser)
    return parser


def native_paths(output_root: Path) -> dict[str, Path]:
    run = RunPaths.from_output_root(output_root)
    manifest_dir = run.manifests_native
    tables_dir = run.results_native_tables
    work_dir = run.work_native / "rosetta"
    return {
        "manifest": manifest_dir / "manifest.csv",
        "manifest_labeled": manifest_dir / "manifest_labeled.csv",
        "checksums": manifest_dir / "checksums.sha256",
        "features": tables_dir / "native_interface_features.csv",
        "features_labeled": tables_dir / "native_interface_features_labeled.csv",
        "relax_status": tables_dir / "rosetta_relax_status.csv",
        "rosetta_relaxed": tables_dir / "rosetta_interface_native_relaxed.csv",
        "rosetta_relaxed_labeled": tables_dir / "rosetta_interface_native_relaxed_labeled.csv",
        "work": work_dir,
    }


def planned_raw_dir(args: argparse.Namespace, output_root: Path, source: str) -> Path:
    if args.input_kind == "tarballs" and args.limit is None:
        return args.input
    return preprocessing_paths(output_root, source).tarballs


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


def build_commands(args: argparse.Namespace, paths: dict[str, Path], raw_dir: Path) -> list:
    rosetta_args = rosetta_command_args(args)
    limit = ["--limit", str(args.limit)] if args.limit is not None else []
    commands = [
        nbia_command(
            "manifest",
            "--raw-dir", raw_dir,
            "--out", paths["manifest"],
            "--checksums", paths["checksums"],
            name="manifest",
        ),
        nbia_command(
            "features-native",
            "--raw-dir", raw_dir,
            "--out", paths["features"],
            "--contact-cutoff", str(args.contact_cutoff),
            name="features-native",
        ),
        nbia_command(
            "rosetta-relax",
            "--manifest", paths["manifest"],
            "--raw-dir", raw_dir,
            "--work-dir", paths["work"],
            *rosetta_args,
            "--jobs", str(args.jobs),
            "--status-out", paths["relax_status"],
            *limit,
            name="rosetta-relax-native",
        ),
        nbia_command(
            "rosetta-interface",
            "--manifest", paths["manifest"],
            "--raw-dir", raw_dir,
            "--work-dir", paths["work"],
            "--out", paths["rosetta_relaxed"],
            "--state", "relaxed",
            *rosetta_args,
            "--jobs", str(args.jobs),
            *limit,
            name="rosetta-interface-native-relaxed",
        ),
    ]
    if args.backfill_fullscore:
        commands.append(
            nbia_command(
                "rosetta-fullscore-backfill",
                "--interface-csv", paths["rosetta_relaxed"],
                "--manifest", paths["manifest"],
                "--raw-dir", raw_dir,
                "--work-dir", paths["work"],
                "--out", paths["rosetta_relaxed"],
                *rosetta_args,
                "--jobs", str(args.jobs),
                "--state", "relaxed",
                *limit,
                name="rosetta-fullscore-native-relaxed",
            )
        )
    return commands

def write_labeled_native_tables(paths: dict[str, Path], labels: LabelConfig) -> None:
    manifest = pd.read_csv(paths["manifest"])
    manifest_labeled = add_label_columns(manifest, labels)
    paths["manifest_labeled"].parent.mkdir(parents=True, exist_ok=True)
    manifest_labeled.to_csv(paths["manifest_labeled"], index=False)

    features = pd.read_csv(paths["features"])
    merge_label_columns(table=features, manifest=manifest_labeled).to_csv(paths["features_labeled"], index=False)

    rosetta = pd.read_csv(paths["rosetta_relaxed"])
    merge_label_columns(table=rosetta, manifest=manifest_labeled).to_csv(paths["rosetta_relaxed_labeled"], index=False)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, output_root: Path, labels: LabelConfig) -> None:
    try:
        from jango.preprocessing.native_inputs import validate_preprocessing_options

        validate_preprocessing_options(input_kind=args.input_kind, input_path=args.input, metadata=args.metadata)
        labels.validate()
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_antigen_len <= 0 or args.max_antigen_len < args.min_antigen_len:
        parser.error("expected 0 < --min-antigen-len <= --max-antigen-len")
    image = args.rosetta_image or args.docker_image
    if args.rosetta_runtime in {"docker", "apptainer"} and not image:
        parser.error("--rosetta-image is required when --rosetta-runtime is docker or apptainer")
    if args.rosetta_runtime == "local" and args.rosetta_bin_dir is None:
        parser.error("--rosetta-bin-dir is required when --rosetta-runtime local")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_root = ensure_output_root_outside_repo(args.output_root)
        labels = LabelConfig(args.source, args.label, args.role, args.confidence)
    except ValueError as exc:
        parser.error(str(exc))
    validate_args(parser, args, output_root, labels)

    paths = native_paths(output_root)
    raw_dir = planned_raw_dir(args, output_root, labels.source)
    commands = build_commands(args, paths, raw_dir)
    env = build_env(atlas_root=args.atlas_root)
    runner = CommandRunner(env=env, plan_only=args.plan_only)

    if args.plan_only:
        print(f"nanobody-pipeline: {STAGE_DESCRIPTION}")
        print(f"input kind: {args.input_kind}")
        if args.input_kind == "tarballs":
            print(f"preprocessing: use existing tarball directory {raw_dir}")
        else:
            print(f"preprocessing: {args.input_kind} -> {raw_dir}")
            print(f"preprocessing manifest: {preprocessing_paths(output_root, labels.source).manifest_labeled}")
        print("native Rosetta score state: native_relaxed")
        print(f"output root: {output_root}")
        runner.run_many(commands)
        print(f"label schema: source,label,role,confidence -> {paths['manifest_labeled']}")
        return 0


    if not args.input.exists() or not args.input.is_dir():
        parser.error("--input must be an existing directory")
    if args.metadata is not None and not args.metadata.exists():
        parser.error("--metadata path does not exist")
    ensure_dir(args.atlas_root / "src", "Atlas src")

    prepared = prepare_native_inputs(
        input_kind=args.input_kind,
        input_path=args.input,
        output_root=output_root,
        labels=labels,
        metadata=args.metadata,
        min_antigen_len=args.min_antigen_len,
        max_antigen_len=args.max_antigen_len,
        limit=args.limit,
    )
    raw_dir = prepared.raw_dir
    commands = build_commands(args, paths, raw_dir)

    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
    runner.run(commands[0])
    runner.run(commands[1])
    runner.run(commands[2])
    runner.run(commands[3])
    if args.backfill_fullscore:
        runner.run(commands[4])
    write_labeled_native_tables(paths, labels)
    print(f"Prepared native input directory: {raw_dir}")
    if prepared.manifest_labeled is not None:
        print(f"Wrote preprocessing manifest: {prepared.manifest_labeled}")
    print(f"Wrote native relaxed Rosetta table: {paths['rosetta_relaxed']}")
    print(f"Wrote labeled native tables using source,label,role,confidence under: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

