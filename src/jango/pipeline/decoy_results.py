"""Run decoy metric tables and landscape tables without rerunning relax/DockQ."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from jango.analysis.thresholds import default_dockq_preservation_threshold, parse_dockq_preservation_threshold
from jango.paths import RunPaths

from ._common import (
    ROSETTA_RUNTIMES,
    add_atlas_root,
    add_output_root,
    add_plan_only,
    ensure_output_root_outside_repo,
    env_path,
    env_value,
)
from .decoy_analyze import (
    ensure_dockq_table_has_successes,
    grafted_decoy_count,
    prepare_landscape_inputs,
    rosetta_command_args,
    validate_rosetta_args,
    write_all_modes_tables,
    write_zero_survivor_landscape_outputs,
)
from .execution import CommandRunner, build_env, ensure_dir, ensure_file, nbia_command
from .metadata import FOLD_METADATA, read_fold_metadata, read_prefold_metadata


STAGE_DESCRIPTION = "decoy metric tables, landscape tables, and compact result summaries"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decoy-results",
        description="Build decoy result tables from relaxed decoys and DockQ outputs.",
    )
    parser.add_argument("--fold-dir", required=True, type=Path)
    add_atlas_root(parser)
    parser.add_argument("--native-relaxed-rosetta", required=True, type=Path)
    parser.add_argument("--native-features", required=True, type=Path)
    parser.add_argument("--experiment", default=None)
    rosetta_runtime_default = env_value("ROSETTA_RUNTIME")
    parser.add_argument("--rosetta-runtime", required=rosetta_runtime_default is None, default=rosetta_runtime_default, choices=ROSETTA_RUNTIMES)
    parser.add_argument("--rosetta-image", default=env_value("ROSETTA_IMAGE"))
    parser.add_argument("--rosetta-bin-dir", type=Path, default=env_path("ROSETTA_BIN_DIR"))
    parser.add_argument("--apptainer-cache-dir", type=Path, default=env_path("APPTAINER_CACHEDIR"))
    parser.add_argument("--docker-image", default=None, help="Deprecated alias for --rosetta-image.")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI symmetry.")
    parser.add_argument("--structure-threshold", type=parse_dockq_preservation_threshold, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bins", type=int, default=30)
    add_output_root(parser)
    add_plan_only(parser)
    return parser


def _metrics_command(
    *,
    mode: str,
    case_manifest: Path,
    tables_dir: Path,
    work_dir: Path,
    native_rosetta: Path,
    native_features: Path,
    rosetta_args: list[str | Path],
) -> object:
    return nbia_command(
        "decoy-metrics",
        "--relax-manifest",
        tables_dir / "decoy_relax_manifest.csv",
        "--case-manifest",
        case_manifest,
        "--tables-dir",
        tables_dir,
        "--work-dir",
        work_dir,
        *rosetta_args,
        "--native-rosetta",
        native_rosetta,
        "--native-protein-interface",
        native_features,
        name=f"decoy-metrics-{mode}",
    )


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
    except ValueError as exc:
        parser.error(str(exc))
    validate_rosetta_args(parser, args)

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
        print(f"decoy-results: {STAGE_DESCRIPTION}")
        print(f"experiment: {fold.experiment}")
    else:
        ensure_dir(args.atlas_root / "src", "Atlas src")
        ensure_dir(raw_dir, "raw tarball dir")
        ensure_file(args.native_relaxed_rosetta, "native relaxed Rosetta table")
        ensure_file(args.native_features, "native features table")

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
            ensure_file(tables_dir / "decoy_relax_manifest.csv", f"{mode} decoy relax manifest")
            ensure_file(tables_dir / "decoy_dockq.csv", f"{mode} DockQ table")
            if grafted_decoy_count(tables_dir) > 0:
                ensure_dockq_table_has_successes(tables_dir / "decoy_dockq.csv", f"{mode} DockQ table")

        runner.run(
            _metrics_command(
                mode=mode,
                case_manifest=case_manifest,
                tables_dir=tables_dir,
                work_dir=work_dir,
                native_rosetta=args.native_relaxed_rosetta,
                native_features=args.native_features,
                rosetta_args=rosetta_args,
            )
        )
        if args.plan_only:
            print(f"[plan] run landscape table analysis for mode {mode}")
            continue

        prepare_landscape_inputs(tables_dir=tables_dir, case_manifest=case_manifest, validation=validation)
        if grafted_decoy_count(tables_dir) > 0:
            ensure_dockq_table_has_successes(tables_dir / "decoy_dockq.csv", f"{mode} DockQ table")
        if grafted_decoy_count(tables_dir) == 0:
            write_zero_survivor_landscape_outputs(
                tables_dir=tables_dir,
                figures_dir=figures_dir,
                mode=mode,
                experiment=fold.experiment,
                structure_threshold=args.structure_threshold,
            )
            completed_modes.append(mode)
            continue

        from jango.analysis.experiment import main as analyze_experiment

        tmp_figures = work_dir / "landscape_figures_tmp"
        if tmp_figures.exists():
            shutil.rmtree(tmp_figures)
        rc = analyze_experiment(
            [
                "--input-dir",
                str(tables_dir),
                "--out-dir",
                str(work_dir / "landscape_analysis_tmp"),
                "--tables-dir",
                str(tables_dir / "landscape"),
                "--figures-dir",
                str(tmp_figures),
                "--dpi",
                str(args.dpi),
                "--bins",
                str(args.bins),
                "--structure-threshold",
                str(args.structure_threshold),
            ]
        )
        if tmp_figures.exists():
            shutil.rmtree(tmp_figures)
        if rc != 0:
            raise SystemExit(rc)
        completed_modes.append(mode)

    if not args.plan_only:
        write_all_modes_tables(decoy_layout=decoy_layout, modes=completed_modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
