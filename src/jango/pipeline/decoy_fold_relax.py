"""Run decoy grafting, Rosetta relax, and DockQ without result-table analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from jango.analysis.thresholds import default_dockq_preservation_threshold, parse_dockq_preservation_threshold
from jango.paths import RunPaths
from jango.runtime.rosetta import ensure_matching_rosetta_states

from ._common import (
    ROSETTA_RUNTIMES,
    add_atlas_root,
    add_output_root,
    add_plan_only,
    ensure_output_root_outside_repo,
    env_path,
    env_value,
)
from .decoy_analyze import build_mode_commands, rosetta_command_args, validate_rosetta_args, write_all_modes_tables
from .execution import CommandRunner, build_env, ensure_dir, ensure_file
from .metadata import FOLD_METADATA, read_fold_metadata, read_prefold_metadata


STAGE_DESCRIPTION = "decoy grafting, Rosetta relax, and DockQ scoring"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decoy-fold-relax",
        description="Graft folded decoy antigens, relax complexes, and optionally run DockQ.",
    )
    parser.add_argument("--fold-dir", required=True, type=Path)
    add_atlas_root(parser)
    parser.add_argument("--native-relax-manifest", type=Path, default=None)
    parser.add_argument("--experiment", default=None)
    rosetta_runtime_default = env_value("ROSETTA_RUNTIME")
    parser.add_argument("--rosetta-runtime", required=rosetta_runtime_default is None, default=rosetta_runtime_default, choices=ROSETTA_RUNTIMES)
    parser.add_argument("--rosetta-image", default=env_value("ROSETTA_IMAGE"))
    parser.add_argument("--rosetta-bin-dir", type=Path, default=env_path("ROSETTA_BIN_DIR"))
    parser.add_argument("--apptainer-cache-dir", type=Path, default=env_path("APPTAINER_CACHEDIR"))
    parser.add_argument("--docker-image", default=None, help="Deprecated alias for --rosetta-image.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dockq-bin", type=Path, default=env_path("DOCKQ_BIN"))
    parser.add_argument("--dockq-provider", default=env_value("DOCKQ_PROVIDER"))
    parser.add_argument("--dockq-python", type=Path, default=env_path("DOCKQ_PYTHON"))
    parser.add_argument("--allow-dockq-failures", action="store_true")
    parser.add_argument("--dockq-skip-existing", action="store_true")
    parser.add_argument("--skip-dockq", action="store_true")
    parser.add_argument("--structure-threshold", type=parse_dockq_preservation_threshold, default=None)
    parser.add_argument("--limit", type=int, default=None)
    add_output_root(parser)
    add_plan_only(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.structure_threshold is None:
        try:
            args.structure_threshold = default_dockq_preservation_threshold()
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    try:
        output_root = ensure_output_root_outside_repo(args.output_root)
        ensure_matching_rosetta_states("native_relaxed", "decoy_relaxed")
    except ValueError as exc:
        parser.error(str(exc))
    validate_rosetta_args(parser, args)
    if not args.skip_dockq and args.native_relax_manifest is None:
        parser.error("--native-relax-manifest is required unless --skip-dockq is used")

    fold_metadata_path = args.fold_dir / FOLD_METADATA
    ensure_file(fold_metadata_path, "fold metadata")
    fold = read_fold_metadata(fold_metadata_path)
    prefold = read_prefold_metadata(Path(fold.prefold_metadata))
    if args.experiment is not None and args.experiment != fold.experiment:
        parser.error(f"--experiment {args.experiment!r} does not match fold metadata experiment {fold.experiment!r}")

    run_paths = RunPaths.from_output_root(output_root)
    decoy_layout = run_paths.decoy_layout(fold.experiment)
    raw_dir = Path(prefold.raw_dir)
    runner = CommandRunner(env=build_env(atlas_root=args.atlas_root), plan_only=args.plan_only)
    rosetta_args = rosetta_command_args(args)

    if args.plan_only:
        print(f"decoy-fold-relax: {STAGE_DESCRIPTION}")
        print(f"experiment: {fold.experiment}")
    else:
        ensure_dir(args.atlas_root / "src", "Atlas src")
        ensure_dir(raw_dir, "raw tarball dir")
        if not args.skip_dockq:
            ensure_file(args.native_relax_manifest, "native relax manifest")

    completed_modes: list[str] = []
    for mode, outputs in fold.mode_outputs.items():
        case_manifest = Path(outputs["case_manifest"])
        tables_dir = decoy_layout.mode_tables_dir(mode)
        fold_qc_validation = tables_dir / "post_fold_qc.csv"
        validation = fold_qc_validation if fold_qc_validation.exists() else Path(outputs["validation"])
        figures_dir = decoy_layout.mode_figures_dir(mode)
        work_dir = decoy_layout.mode_work_dir(mode)
        if not args.plan_only:
            tables_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)
            ensure_file(case_manifest, f"{mode} case manifest")
            ensure_file(validation, f"{mode} validation table")

        commands = build_mode_commands(
            mode=mode,
            case_manifest=case_manifest,
            validation=validation,
            raw_dir=raw_dir,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            work_dir=work_dir,
            native_rosetta=Path("unused_native_rosetta.csv"),
            native_features=Path("unused_native_features.csv"),
            native_relax_manifest=args.native_relax_manifest,
            dockq_bin=args.dockq_bin,
            dockq_provider=args.dockq_provider,
            dockq_python=args.dockq_python,
            allow_dockq_failures=args.allow_dockq_failures,
            dockq_skip_existing=args.dockq_skip_existing,
            skip_dockq=args.skip_dockq,
            experiment=fold.experiment,
            rosetta_args=rosetta_args,
            jobs=args.jobs,
            limit=args.limit,
        )
        runner.run_many(
            [
                command
                for command in commands
                if command.name.startswith(("decoy-graft-", "decoy-relax-", "decoy-dockq-"))
            ]
        )
        completed_modes.append(mode)

    if not args.plan_only:
        write_all_modes_tables(decoy_layout=decoy_layout, modes=completed_modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
