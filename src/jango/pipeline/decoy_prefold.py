"""Decoy pre-fold case-manifest pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from jango.config import experiment_name
from jango.paths import RunPaths

from ._common import FOLD_BACKENDS, MODE_POLICIES, add_atlas_root, add_output_root, add_plan_only, ensure_output_root_outside_repo, env_value, require_manual_modes
from .cases import ModeThresholds, REDESIGN_MODES, select_decoy_source_cases, split_cases_by_mode
from .execution import CommandRunner, build_env, ensure_dir, ensure_file, nbia_command
from .metadata import PREFOLD_METADATA, PrefoldMetadata, write_metadata


STAGE_DESCRIPTION = "source case selection, decoy contact manifest generation, and redesign-mode manifest split"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decoy-prefold",
        description="Prepare decoy source/base/mode case manifests before ProteinMPNN and folding.",
    )
    add_atlas_root(parser)
    parser.add_argument("--raw-dir", required=True, type=Path, help="Preprocessed native tarball directory used by NBIA decoy-cases.")
    parser.add_argument("--native-manifest", required=True, type=Path)
    parser.add_argument("--native-features", required=True, type=Path)
    parser.add_argument("--native-relaxed-rosetta", required=True, type=Path)
    parser.add_argument("--decoy-config", required=True, type=Path, help="Explicit antigen redesign decoy config.")
    backend_default = env_value("FOLDING_BACKEND")
    parser.add_argument("--backend", required=backend_default is None, default=backend_default, choices=FOLD_BACKENDS, help="Backend name; defaults from FOLDING_BACKEND when set.")
    parser.add_argument("--max-structures", type=int, default=None, help="Maximum raw input structures selected before native analysis.")
    parser.add_argument("--case-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode-policy", required=True, choices=MODE_POLICIES)
    parser.add_argument("--experiment", default=None, help="Canonical decoy experiment key; defaults to backend_maxN_policy.")
    parser.add_argument("--modes", nargs="+", choices=REDESIGN_MODES)
    parser.add_argument("--antigen-length-min", type=int, default=60)
    parser.add_argument("--antigen-length-max", type=int, default=250)
    parser.add_argument("--small-antigen-len", type=int, default=120)
    parser.add_argument("--medium-antigen-len", type=int, default=200)
    parser.add_argument("--max-interface-fraction", type=float, default=0.25)
    add_output_root(parser)
    add_plan_only(parser)
    return parser


def prefold_paths(output_root: Path, namespace: str) -> dict[str, Path]:
    layout = RunPaths.from_output_root(output_root).decoy_layout(namespace)
    return {
        "prefold_dir": layout.manifest_dir,
        "source_case_manifest": layout.source_case_manifest,
        "base_decoy_case_manifest": layout.base_decoy_case_manifest,
        "metadata": layout.prefold_manifest,
    }


def build_decoy_cases_command(args: argparse.Namespace, paths: dict[str, Path]):
    return nbia_command(
        "decoy-cases",
        "--case-manifest", paths["source_case_manifest"],
        "--raw-dir", args.raw_dir,
        "--out", paths["base_decoy_case_manifest"],
        "--config", args.decoy_config,
        name="decoy-cases-base",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = require_manual_modes(parser, mode_policy=args.mode_policy, modes=args.modes)
    max_structures = args.max_structures if args.max_structures is not None else args.case_count
    if args.max_structures is not None and args.case_count is not None and args.max_structures != args.case_count:
        parser.error("--max-structures and deprecated --case-count disagree")
    if max_structures is None:
        parser.error("--max-structures is required")
    args.max_structures = max_structures
    args.case_count = max_structures
    try:
        output_root = ensure_output_root_outside_repo(args.output_root)
        namespace = args.experiment or experiment_name(backend=args.backend, max_structures=args.max_structures, policy=args.mode_policy)
    except ValueError as exc:
        parser.error(str(exc))

    paths = prefold_paths(output_root, namespace)
    command = build_decoy_cases_command(args, paths)
    env = build_env(atlas_root=args.atlas_root)
    runner = CommandRunner(env=env, plan_only=args.plan_only)

    if args.plan_only:
        print(f"decoy-prefold: {STAGE_DESCRIPTION}")
        print(f"experiment namespace: {namespace}")
        print(f"prefold dir: {paths['prefold_dir']}")
        print("[plan] select source cases: all eligible cases from input-limited native outputs")
        runner.run(command)
        print("[plan] split base decoy manifest by redesign mode policy")
        return 0

    for label, path in (
        ("Atlas src", args.atlas_root / "src"),
        ("raw tarball dir", args.raw_dir),
    ):
        ensure_dir(path, label)
    for label, path in (
        ("native manifest", args.native_manifest),
        ("native features", args.native_features),
        ("native relaxed Rosetta table", args.native_relaxed_rosetta),
        ("decoy config", args.decoy_config),
    ):
        ensure_file(path, label)

    paths["prefold_dir"].mkdir(parents=True, exist_ok=True)
    selected = select_decoy_source_cases(
        manifest_csv=args.native_manifest,
        native_features_csv=args.native_features,
        native_relaxed_rosetta_csv=args.native_relaxed_rosetta,
        out_csv=paths["source_case_manifest"],
        max_structures=args.max_structures,
        antigen_length_min=args.antigen_length_min,
        antigen_length_max=args.antigen_length_max,
    )
    print(f"Wrote {len(selected)} source decoy cases to {paths['source_case_manifest']}")

    runner.run(command)
    thresholds = ModeThresholds(
        small_antigen_len=args.small_antigen_len,
        medium_antigen_len=args.medium_antigen_len,
        max_interface_fraction=args.max_interface_fraction,
    )
    mode_manifests, assignment_path = split_cases_by_mode(
        base_case_manifest=paths["base_decoy_case_manifest"],
        out_dir=paths["prefold_dir"],
        mode_policy=args.mode_policy,
        modes=modes,
        thresholds=thresholds,
    )
    metadata = PrefoldMetadata.create(
        experiment=namespace,
        backend=args.backend,
        case_count=args.max_structures,
        max_structures=args.max_structures,
        count_semantics="input_max",
        mode_policy=args.mode_policy,
        raw_dir=args.raw_dir,
        native_manifest=args.native_manifest,
        native_features=args.native_features,
        native_relaxed_rosetta=args.native_relaxed_rosetta,
        source_case_manifest=paths["source_case_manifest"],
        base_decoy_case_manifest=paths["base_decoy_case_manifest"],
        mode_manifests=mode_manifests,
    )
    write_metadata(paths["metadata"], metadata)
    print(f"Wrote mode assignments to {assignment_path}")
    print(f"Wrote prefold metadata to {paths['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())