"""Decoy Rosetta analysis and report-building pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from jango.analysis.thresholds import default_dockq_preservation_threshold, parse_dockq_preservation_threshold
from jango.paths import RunPaths
from jango.runtime.rosetta import ensure_matching_rosetta_states

from ._common import ROSETTA_RUNTIMES, add_atlas_root, add_output_root, add_plan_only, ensure_output_root_outside_repo, env_path, env_value
from .execution import CommandRunner, build_env, ensure_dir, ensure_file, nbia_command
from .metadata import FOLD_METADATA, read_fold_metadata, read_prefold_metadata


STAGE_DESCRIPTION = "decoy grafting, required Rosetta scoring, native-relaxed comparison, reports, and DELPHI landscape analysis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decoy-analyze",
        description="Analyze Rosetta-scored decoys against native relaxed Rosetta scores.",
    )
    parser.add_argument("--fold-dir", required=True, type=Path)
    add_atlas_root(parser)
    parser.add_argument("--native-relaxed-rosetta", required=True, type=Path)
    parser.add_argument("--native-relax-manifest", type=Path, default=None, help="Native Rosetta relax status manifest containing relaxed native PDB paths; required unless --skip-dockq.")
    parser.add_argument("--native-features", required=True, type=Path)
    parser.add_argument("--experiment", default=None, help="Optional guard; must match fold metadata when provided.")
    rosetta_runtime_default = env_value("ROSETTA_RUNTIME")
    parser.add_argument("--rosetta-runtime", required=rosetta_runtime_default is None, default=rosetta_runtime_default, choices=ROSETTA_RUNTIMES)
    parser.add_argument("--rosetta-image", default=env_value("ROSETTA_IMAGE"), help="Rosetta container image when --rosetta-runtime docker/apptainer is used.")
    parser.add_argument("--rosetta-bin-dir", type=Path, default=env_path("ROSETTA_BIN_DIR"), help="Directory containing local Rosetta executables when --rosetta-runtime local is used.")
    parser.add_argument("--apptainer-cache-dir", type=Path, default=env_path("APPTAINER_CACHEDIR"), help="Optional Apptainer cache directory.")
    parser.add_argument("--docker-image", default=None, help="Deprecated alias for --rosetta-image.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dockq-bin", type=Path, default=env_path("DOCKQ_BIN"), help="Official DockQ executable path; falls back to DOCKQ_BIN/PATH in NBIA.")
    parser.add_argument("--dockq-provider", default=env_value("DOCKQ_PROVIDER"), help="DockQ provider passed through to NBIA decoy-dockq.")
    parser.add_argument("--dockq-python", type=Path, default=env_path("DOCKQ_PYTHON"), help="Python interpreter for the dockq_rs_python provider.")
    parser.add_argument("--allow-dockq-failures", action="store_true", help="Write DockQ failure rows and continue analysis.")
    parser.add_argument("--dockq-skip-existing", action="store_true", help="Reuse existing DockQ JSON outputs when present.")
    parser.add_argument("--skip-dockq", action="store_true", help="Legacy/debug only: do not run canonical DockQ scoring.")
    parser.add_argument(
        "--structure-threshold",
        type=parse_dockq_preservation_threshold,
        default=None,
        help=(
            "Initial DockQ threshold for high structure preservation in quadrant "
            "tables. Defaults to DOCKQ_PRESERVATION_THRESHOLD or 0.49."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bins", type=int, default=30)
    add_output_root(parser)
    add_plan_only(parser)
    return parser


def analyze_root(output_root: Path, experiment: str) -> Path:
    return RunPaths.from_output_root(output_root).decoy_layout(experiment).manifest_dir


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


def validate_rosetta_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    image = args.rosetta_image or args.docker_image
    if args.rosetta_runtime in {"docker", "apptainer"} and not image:
        parser.error("--rosetta-image is required when --rosetta-runtime is docker or apptainer")
    if args.rosetta_runtime == "local" and args.rosetta_bin_dir is None:
        parser.error("--rosetta-bin-dir is required when --rosetta-runtime local")


def build_mode_commands(
    *,
    mode: str,
    case_manifest: Path,
    validation: Path,
    raw_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    work_dir: Path,
    native_rosetta: Path,
    native_features: Path,
    native_relax_manifest: Path | None,
    dockq_bin: Path | None,
    dockq_provider: str | None = None,
    dockq_python: Path | None = None,
    allow_dockq_failures: bool,
    dockq_skip_existing: bool,
    skip_dockq: bool,
    experiment: str,
    rosetta_args: list[str | Path],
    jobs: int,
    limit: int | None,
) -> list:
    limit_parts = ["--limit", str(limit)] if limit is not None else []
    graft_manifest = tables_dir / "decoy_grafted_complex_manifest.csv"
    relax_manifest = tables_dir / "decoy_relax_manifest.csv"
    commands = [
        nbia_command(
            "decoy-graft",
            "--validation", validation,
            "--case-manifest", case_manifest,
            "--raw-dir", raw_dir,
            "--work-dir", work_dir,
            "--out", graft_manifest,
            name=f"decoy-graft-{mode}",
        ),
        nbia_command(
            "decoy-relax",
            "--manifest", graft_manifest,
            "--work-dir", work_dir,
            "--out", relax_manifest,
            *rosetta_args,
            "--jobs", str(jobs),
            *limit_parts,
            name=f"decoy-relax-{mode}",
        ),
    ]
    if not skip_dockq:
        commands.append(
            nbia_command(
                "decoy-dockq",
                "--decoy-relax-manifest", relax_manifest,
                "--native-relax-manifest", native_relax_manifest or "<required-native-relax-manifest>",
                "--case-manifest", case_manifest,
                "--work-dir", work_dir / "dockq",
                "--out", tables_dir / "decoy_dockq.csv",
                "--jobs", str(jobs),
                "--experiment-namespace", experiment,
                *( ["--dockq-bin", dockq_bin] if dockq_bin is not None else [] ),
                *( ["--dockq-provider", dockq_provider] if dockq_provider is not None else [] ),
                *( ["--dockq-python", dockq_python] if dockq_python is not None else [] ),
                *( ["--allow-failures"] if allow_dockq_failures else [] ),
                *( ["--skip-existing"] if dockq_skip_existing else [] ),
                *limit_parts,
                name=f"decoy-dockq-{mode}",
            )
        )
    commands.extend(
        [
            nbia_command(
                "decoy-metrics",
                "--relax-manifest", relax_manifest,
                "--case-manifest", case_manifest,
                "--tables-dir", tables_dir,
                "--work-dir", work_dir,
                *rosetta_args,
                "--native-rosetta", native_rosetta,
                "--native-protein-interface", native_features,
                name=f"decoy-metrics-{mode}",
            ),
            nbia_command(
                "decoy-report",
                "--tables-dir", tables_dir,
                "--figures-dir", figures_dir,
                name=f"decoy-report-{mode}",
            ),
        ]
    )
    return commands

ALL_MODES_TABLES = (
    "decoy_redesign_case_manifest.csv",
    "decoy_validation.csv",
    "decoy_grafted_complex_manifest.csv",
    "decoy_relax_manifest.csv",
    "decoy_dockq.csv",
    "decoy_protein_interface.csv",
    "decoy_shape_complementarity.csv",
    "decoy_rosetta_interface.csv",
    "decoy_native_comparison.csv",
)


def write_all_modes_tables(*, decoy_layout, modes: list[str]) -> None:
    """Concatenate per-mode decoy analysis tables for downstream QC/export consumers."""

    out_dir = decoy_layout.all_modes_tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in ALL_MODES_TABLES:
        frames: list[pd.DataFrame] = []
        for mode in modes:
            path = decoy_layout.mode_tables_dir(mode) / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if not frame.empty and "redesign_mode" not in frame.columns:
                frame = frame.copy()
                frame["redesign_mode"] = mode
            frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True, sort=False).to_csv(out_dir / filename, index=False)


def prepare_landscape_inputs(*, tables_dir: Path, case_manifest: Path, validation: Path) -> None:
    ensure_file(tables_dir / "decoy_native_comparison.csv", "decoy/native comparison")
    ensure_file(tables_dir / "decoy_grafted_complex_manifest.csv", "grafted decoy manifest")
    ensure_file(case_manifest, "mode case manifest")
    ensure_file(validation, "decoy validation table")

    shutil.copyfile(case_manifest, tables_dir / "decoy_redesign_case_manifest.csv")
    validation_destination = tables_dir / "decoy_validation.csv"
    if validation.resolve() != validation_destination.resolve():
        shutil.copyfile(validation, validation_destination)

    cases = pd.read_csv(case_manifest)
    assignments = cases[["structure_id"]].drop_duplicates().copy()
    if "selected_redesign_mode" in cases.columns:
        mode_map = cases.drop_duplicates("structure_id").set_index("structure_id")["selected_redesign_mode"]
        assignments["selected_redesign_mode"] = assignments["structure_id"].map(mode_map)
    elif "redesign_mode" in cases.columns:
        mode_map = cases.drop_duplicates("structure_id").set_index("structure_id")["redesign_mode"]
        assignments["selected_redesign_mode"] = assignments["structure_id"].map(mode_map)
    else:
        assignments["selected_redesign_mode"] = ""
    assignments.to_csv(tables_dir / "auto_mode_assignments.csv", index=False)


def grafted_decoy_count(tables_dir: Path) -> int:
    path = tables_dir / "decoy_grafted_complex_manifest.csv"
    if not path.exists() or path.stat().st_size == 0:
        return 0
    table = pd.read_csv(path)
    return int(len(table))


def ensure_dockq_table_has_successes(path: Path, label: str) -> int:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} not found or empty: {path}")
    table = pd.read_csv(path)
    if table.empty:
        raise SystemExit(f"{label} has no DockQ rows; refusing to build landscape outputs: {path}")
    if "dockq_status" not in table.columns or "dockq" not in table.columns:
        raise SystemExit(f"{label} is missing required DockQ columns dockq_status/dockq: {path}")
    ok = table["dockq_status"].astype(str).str.lower().eq("ok") & table["dockq"].notna()
    ok_count = int(ok.sum())
    if ok_count == 0:
        status_counts = table["dockq_status"].value_counts(dropna=False).head(5).to_dict()
        raise SystemExit(
            f"{label} has no successful numeric DockQ rows; refusing to build landscape outputs. "
            f"Status counts: {status_counts}. Check native/decoy relaxed PDB paths before rerunning. Path: {path}"
        )
    return ok_count


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows or [], columns=columns).to_csv(path, index=False)


def write_zero_survivor_landscape_outputs(*, tables_dir: Path, figures_dir: Path, mode: str, experiment: str, structure_threshold: float) -> None:
    landscape_tables = tables_dir / "landscape"
    landscape_figures = figures_dir / "landscape"
    landscape_tables.mkdir(parents=True, exist_ok=True)
    landscape_figures.mkdir(parents=True, exist_ok=True)
    score_columns = [
        "experiment_namespace",
        "redesign_mode",
        "structure_id",
        "design_id",
        "quadrant",
        "structure_preservation",
        "binding_perturbation",
        "coordinate_score",
        "delphi_score",
        "classification_reason",
    ]
    _write_csv(landscape_tables / "dataset_counts.csv", ["metric", "count"], [{"metric": "analyzed_decoys", "count": 0}])
    _write_csv(landscape_tables / "component_metric_summary.csv", ["metric", "n", "mean", "median", "std", "min", "max"])
    _write_csv(landscape_tables / "delphi_landscape_scores.csv", score_columns)
    _write_csv(landscape_tables / "decoy_quadrant_ranking.csv", score_columns)
    _write_csv(
        landscape_tables / "decoy_quadrant_summary.csv",
        ["quadrant", "n_decoys", "structure_threshold"],
        [{"quadrant": quadrant, "n_decoys": 0, "structure_threshold": structure_threshold} for quadrant in ["Q1", "Q4", "Q2", "Q3"]],
    )
    _write_csv(landscape_tables / "decoy_quadrant_unclassified.csv", score_columns)
    _write_csv(
        landscape_tables / "mutation_locations.csv",
        ["experiment_namespace", "structure_id", "design_id", "native_position", "canonical_position", "region", "cdr_name", "native_amino_acid", "mutated_amino_acid", "backend", "redesign_mode"],
    )
    _write_csv(landscape_tables / "mutation_region_summary.csv", ["experiment_namespace", "backend", "redesign_mode", "mode_policy", "total_mutations"])
    _write_csv(landscape_tables / "mutation_position_frequency.csv", ["experiment_namespace", "canonical_position", "region", "mutation_count", "mutation_frequency"])
    _write_csv(landscape_tables / "mutation_region_enrichment.csv", ["experiment_namespace", "region", "observed_mutations", "expected_mutations", "fold_enrichment", "p_value"])
    _write_csv(landscape_tables / "mutation_localization_skipped_decoys.csv", ["experiment_namespace", "structure_id", "design_id", "reason"])
    _write_csv(landscape_tables / "mutation_numbering_status.csv", ["experiment_namespace", "structure_id", "design_id", "status", "failure_reason"])
    print(f"mode {mode}: no grafted decoys; wrote empty landscape tables for experiment {experiment}")


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

    fold_metadata_path = args.fold_dir / FOLD_METADATA
    ensure_file(fold_metadata_path, "fold metadata")
    fold = read_fold_metadata(fold_metadata_path)
    prefold = read_prefold_metadata(Path(fold.prefold_metadata))
    if args.experiment is not None and args.experiment != fold.experiment:
        parser.error(f"--experiment {args.experiment!r} does not match fold metadata experiment {fold.experiment!r}")

    run_paths = RunPaths.from_output_root(output_root)
    decoy_layout = run_paths.decoy_layout(fold.experiment)
    root = decoy_layout.manifest_dir
    raw_dir = Path(prefold.raw_dir)
    validate_rosetta_args(parser, args)
    if not args.skip_dockq and args.native_relax_manifest is None:
        parser.error("--native-relax-manifest is required unless --skip-dockq is used")
    rosetta_args = rosetta_command_args(args)
    env = build_env(atlas_root=args.atlas_root)
    runner = CommandRunner(env=env, plan_only=args.plan_only)

    if args.plan_only:
        print(f"decoy-analyze: {STAGE_DESCRIPTION}")
        print(f"experiment: {fold.experiment}")
        print("native Rosetta score state: native_relaxed")
        print("decoy Rosetta score state: decoy_relaxed")
        print("DockQ reference state: native_relaxed" if not args.skip_dockq else "DockQ stage: skipped by request")
        print(f"DockQ structure threshold: {args.structure_threshold}")
        print(f"analysis metadata dir: {root}")
    else:

        ensure_dir(args.atlas_root / "src", "Atlas src")
        ensure_dir(raw_dir, "raw tarball dir")
        ensure_file(args.native_relaxed_rosetta, "native relaxed Rosetta table")
        if not args.skip_dockq:
            ensure_file(args.native_relax_manifest, "native relax manifest")
        ensure_file(args.native_features, "native features table")
        root.mkdir(parents=True, exist_ok=True)

    completed_modes: list[str] = []
    for mode, outputs in fold.mode_outputs.items():
        case_manifest = Path(outputs["case_manifest"])
        tables_dir = decoy_layout.mode_tables_dir(mode)
        fold_qc_validation = decoy_layout.mode_tables_dir(mode) / "post_fold_qc.csv"
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
            native_rosetta=args.native_relaxed_rosetta,
            native_features=args.native_features,
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
        runner.run_many(commands)
        if args.plan_only:
            print(f"[plan] run DELPHI landscape analysis for mode {mode}")
            continue

        prepare_landscape_inputs(tables_dir=tables_dir, case_manifest=case_manifest, validation=validation)
        if not args.skip_dockq and grafted_decoy_count(tables_dir) > 0:
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

        rc = analyze_experiment([
            "--input-dir", str(tables_dir),
            "--out-dir", str(figures_dir / "landscape"),
            "--tables-dir", str(tables_dir / "landscape"),
            "--figures-dir", str(figures_dir / "landscape"),
            "--dpi", str(args.dpi),
            "--bins", str(args.bins),
            "--structure-threshold", str(args.structure_threshold),
        ])
        if rc != 0:
            raise SystemExit(rc)
        completed_modes.append(mode)

    if not args.plan_only:
        write_all_modes_tables(decoy_layout=decoy_layout, modes=completed_modes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
