"""Explicit Rosetta InterfaceAnalyzer entrypoint for arbitrary Jango PDB manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jango.paths import repository_root
from nbia.config import RosettaRuntimeConfig
from nbia.rosetta import (
    build_interface_analyzer_command,
    interface_score_has_required_terms,
    log_mentions_single_chain_pose,
    normalize_interface_score_columns,
    parse_scorefile,
    rosetta_runtime_from_options,
)
from nbia.runtime.rosetta import RosettaRuntime, RosettaRuntimeError

from ._common import ROSETTA_RUNTIMES, add_plan_only, env_path, env_value


AUTO_PDB_COLUMNS = (
    "pdb_path",
    "relaxed_pdb",
    "scored_pdb",
    "complex_pdb",
    "grafted_complex_path",
    "decoy_relaxed_pdb",
    "native_relaxed_pdb",
)
ROSETTA_REFERENCE_TERMS = (
    "dG_separated",
    "dG_separated/dSASAx100",
    "dSASA_int",
    "sc_value",
    "hbonds_int",
    "delta_unsatHbonds",
    "packstat",
)
TITLE_RED = "#C92A2A"
GRID_GREY = "#E6E6E6"
DARK_GREY = "#4A4A4A"
POINT_BLUE = "#2E6FDC"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface-analyzer",
        description=(
            "Run Rosetta InterfaceAnalyzer on a manifest of PDB complexes and write "
            "canonical Jango interface tables."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with structure/design IDs and a PDB path column.")
    parser.add_argument("--out", required=True, type=Path, help="Output Rosetta InterfaceAnalyzer CSV.")
    parser.add_argument("--work-dir", required=True, type=Path, help="User-space work directory for scorefiles and logs.")
    parser.add_argument(
        "--pdb-column",
        default=None,
        help="Manifest column containing PDB paths. Defaults to auto-detecting common Jango path columns.",
    )
    parser.add_argument("--state", default="interface_analyzed", help="Label written to the output `state` column.")
    parser.add_argument("--interface", default=None, help="Rosetta interface string, for example H_A or N_A. Overrides chain columns.")
    parser.add_argument("--nanobody-chain", default=None, help="Fallback nanobody chain if manifest lacks `nanobody_chain`.")
    parser.add_argument("--antigen-chain", default=None, help="Fallback antigen chain if manifest lacks `antigen_chain`.")
    parser.add_argument("--reference-rosetta", type=Path, default=None, help="Optional reference InterfaceAnalyzer CSV for delta tables.")
    parser.add_argument("--delta-out", type=Path, default=None, help="Optional output path for scored-vs-reference deltas.")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Optional figure directory. Defaults beside --out.")
    parser.add_argument("--skip-figures", action="store_true", help="Do not write InterfaceAnalyzer figures.")
    rosetta_runtime_default = env_value("ROSETTA_RUNTIME")
    parser.add_argument("--rosetta-runtime", required=rosetta_runtime_default is None, default=rosetta_runtime_default, choices=ROSETTA_RUNTIMES)
    parser.add_argument("--rosetta-image", default=env_value("ROSETTA_IMAGE"), help="Rosetta container image for docker/apptainer runtimes.")
    parser.add_argument("--rosetta-bin-dir", type=Path, default=env_path("ROSETTA_BIN_DIR"), help="Local Rosetta bin dir for --rosetta-runtime local.")
    parser.add_argument("--apptainer-cache-dir", type=Path, default=env_path("APPTAINER_CACHEDIR"), help="Optional Apptainer cache dir.")
    parser.add_argument("--docker-image", default=None, help="Deprecated alias for --rosetta-image.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true", help="Recompute even if a valid scorefile/log pair already exists.")
    parser.add_argument(
        "--allow-repo-output",
        action="store_true",
        help="Allow outputs under the source repository. Intended only for curated checked-in artifacts.",
    )
    add_plan_only(parser)
    return parser


def rosetta_runtime_config_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RosettaRuntimeConfig:
    image = args.rosetta_image or args.docker_image
    config = RosettaRuntimeConfig(
        kind=args.rosetta_runtime,
        image=image,
        bin_dir=args.rosetta_bin_dir,
        apptainer_cache_dir=args.apptainer_cache_dir,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return config


def ensure_outputs_outside_repo(paths: list[Path | None], *, allow_repo_output: bool) -> None:
    if allow_repo_output:
        return
    repo = repository_root().expanduser().resolve()
    violations: list[str] = []
    for path in paths:
        if path is None:
            continue
        resolved = path.expanduser().resolve()
        if resolved == repo or repo in resolved.parents:
            violations.append(str(resolved))
    if violations:
        raise ValueError(
            "Refusing to write InterfaceAnalyzer outputs inside the source repository. "
            "Use <JANGO_USER_ROOT>/... or pass --allow-repo-output for curated repo artifacts. "
            + "; ".join(violations)
        )


def detect_pdb_column(manifest: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in manifest.columns:
            raise ValueError(f"--pdb-column {requested!r} is not present in manifest")
        return requested
    for column in AUTO_PDB_COLUMNS:
        if column in manifest.columns:
            return column
    raise ValueError("manifest needs a PDB path column; tried: " + ", ".join(AUTO_PDB_COLUMNS))


def nonempty_text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text


def manifest_rows(
    manifest_csv: Path,
    *,
    pdb_column: str | None,
    state: str,
    interface: str | None,
    nanobody_chain: str | None,
    antigen_chain: str | None,
    limit: int | None,
) -> list[dict[str, object]]:
    manifest = pd.read_csv(manifest_csv)
    if limit is not None:
        manifest = manifest.head(limit)
    path_column = detect_pdb_column(manifest, pdb_column)
    rows: list[dict[str, object]] = []
    for idx, row in manifest.reset_index(drop=True).iterrows():
        structure_id = (
            nonempty_text(row.get("structure_id"))
            or nonempty_text(row.get("native_structure_id"))
            or nonempty_text(row.get("design_id"))
            or f"row_{idx + 1}"
        )
        design_id = nonempty_text(row.get("design_id"), structure_id)
        nb_chain = (
            nonempty_text(row.get("nanobody_chain"))
            or nonempty_text(row.get("native_nanobody_chain_original"))
            or nonempty_text(nanobody_chain)
        )
        ag_chain = (
            nonempty_text(row.get("antigen_chain"))
            or nonempty_text(row.get("native_antigen_chain_original"))
            or nonempty_text(antigen_chain)
        )
        row_interface = nonempty_text(interface) or nonempty_text(row.get("interface"))
        if not row_interface:
            if not nb_chain or not ag_chain:
                raise ValueError(
                    "interface cannot be resolved for "
                    f"{design_id}; provide --interface or nanobody/antigen chain columns/defaults"
                )
            row_interface = f"{nb_chain}_{ag_chain}"
        if ("_" in row_interface) and (not nb_chain or not ag_chain):
            pieces = row_interface.split("_", 1)
            nb_chain = nb_chain or pieces[0]
            ag_chain = ag_chain or pieces[1]
        pdb_path = Path(str(row[path_column])).expanduser()
        rows.append(
            {
                "design_id": design_id,
                "structure_id": structure_id,
                "input_pdb": str(pdb_path),
                "state": state,
                "interface": row_interface,
                "nanobody_chain": nb_chain,
                "antigen_chain": ag_chain,
                "source_pdb_column": path_column,
            }
        )
    return rows


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "row"


def score_manifest_rows(
    rows: list[dict[str, object]],
    *,
    work_dir: Path,
    runtime: RosettaRuntime,
    jobs: int,
    skip_existing: bool,
) -> pd.DataFrame:
    score_root = work_dir / "interface_analyzer" / safe_name(rows[0]["state"] if rows else "interface_analyzed")
    score_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = [
            executor.submit(_score_one, row, score_root=score_root, runtime=runtime, skip_existing=skip_existing)
            for row in rows
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['design_id']}: {result['interface_analyzer_status']}", flush=True)
    if not results:
        return pd.DataFrame(columns=["design_id", "structure_id", "interface_analyzer_status"])
    out = pd.DataFrame(results).sort_values(["structure_id", "design_id"]).reset_index(drop=True)
    return normalize_interface_score_columns(out)


def _score_one(row: dict[str, object], *, score_root: Path, runtime: RosettaRuntime, skip_existing: bool) -> dict[str, object]:
    base = {
        "design_id": row["design_id"],
        "structure_id": row["structure_id"],
        "scored_pdb": row["input_pdb"],
        "state": row["state"],
        "nanobody_chain": row["nanobody_chain"],
        "antigen_chain": row["antigen_chain"],
        "interface": row["interface"],
    }
    pdb_path = Path(str(row["input_pdb"]))
    if not pdb_path.exists() or pdb_path.stat().st_size == 0:
        return {**base, "interface_analyzer_status": "missing_pdb", "interface_scorefile": "", "interface_log": ""}
    structure_dir = score_root / safe_name(row["structure_id"])
    structure_dir.mkdir(parents=True, exist_ok=True)
    scorefile = structure_dir / f"{safe_name(row['design_id'])}.interface.sc"
    log = structure_dir / f"{safe_name(row['design_id'])}.interface.log"
    status_text = "not_run"
    parsed: dict[str, object] = {}
    try:
        if skip_existing and scorefile.exists() and scorefile.stat().st_size > 0:
            parsed = parse_scorefile(scorefile)
            if interface_score_has_required_terms(parsed) and log.exists() and log.stat().st_size > 0:
                status_text = "skipped_existing_valid"
        if status_text == "not_run":
            command = build_interface_analyzer_command(
                runtime,
                pdb=pdb_path,
                score_dir=structure_dir,
                scorefile=scorefile,
                interface=str(row["interface"]),
            )
            with log.open("w") as handle:
                status = runtime.run(command, stdout=handle, stderr=subprocess.STDOUT)
            forced_retry = False
            if status.returncode != 0 and log.exists() and log_mentions_single_chain_pose(log):
                forced_retry = True
                force_chains = str(row["interface"]).replace("_", "")
                retry = build_interface_analyzer_command(
                    runtime,
                    pdb=pdb_path,
                    score_dir=structure_dir,
                    scorefile=scorefile,
                    interface=str(row["interface"]),
                    force_separate_chains=force_chains,
                )
                with log.open("a") as handle:
                    handle.write("\n\n--- retrying with forced chain separation ---\n")
                    status = runtime.run(retry, stdout=handle, stderr=subprocess.STDOUT)
            parsed = parse_scorefile(scorefile)
            if status.returncode != 0:
                status_text = f"failed:{status.returncode}"
            elif not interface_score_has_required_terms(parsed):
                status_text = "parse_failed"
            elif forced_retry:
                status_text = "ok_forced_chain_separation"
            else:
                status_text = "ok"
    except Exception as exc:  # noqa: BLE001 - row-level failures belong in the manifest.
        return {
            **base,
            "interface_analyzer_status": f"exception:{type(exc).__name__}",
            "interface_error": str(exc),
            "interface_scorefile": str(scorefile),
            "interface_log": str(log),
        }
    return {
        **base,
        **parsed,
        "rosetta_runtime": runtime.config.kind,
        "interface_analyzer_status": status_text,
        "interface_scorefile": str(scorefile),
        "interface_log": str(log),
    }


def build_reference_delta(scored: pd.DataFrame, reference_csv: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_csv)
    if "structure_id" not in reference.columns:
        raise ValueError(f"reference table is missing structure_id: {reference_csv}")
    scored = scored.copy()
    terms = [term for term in ROSETTA_REFERENCE_TERMS if term in scored.columns and term in reference.columns]
    keep = reference[["structure_id", *terms]].copy()
    keep = keep.rename(columns={term: f"reference_{term}" for term in terms})
    out = scored.merge(keep, on="structure_id", how="left")
    for term in terms:
        out[f"delta_{term}"] = pd.to_numeric(out[term], errors="coerce") - pd.to_numeric(
            out[f"reference_{term}"],
            errors="coerce",
        )
    if "delta_dG_separated" in out.columns:
        out["binding_perturbation"] = out["delta_dG_separated"]
    return out


def write_summary(scored: pd.DataFrame, summary_out: Path, delta: pd.DataFrame | None = None) -> pd.DataFrame:
    status_counts = scored["interface_analyzer_status"].value_counts(dropna=False).to_dict() if "interface_analyzer_status" in scored else {}
    rows: list[dict[str, object]] = [{"metric": f"status:{key}", "value": value} for key, value in status_counts.items()]
    rows.append({"metric": "n_rows", "value": len(scored)})
    if "dG_separated" in scored.columns:
        dgs = pd.to_numeric(scored["dG_separated"], errors="coerce")
        rows.extend(
            [
                {"metric": "n_numeric_dG_separated", "value": int(dgs.notna().sum())},
                {"metric": "median_dG_separated", "value": dgs.median()},
                {"metric": "mean_dG_separated", "value": dgs.mean()},
            ]
        )
    if delta is not None and "delta_dG_separated" in delta.columns:
        values = pd.to_numeric(delta["delta_dG_separated"], errors="coerce")
        rows.extend(
            [
                {"metric": "n_numeric_delta_dG_separated", "value": int(values.notna().sum())},
                {"metric": "median_delta_dG_separated", "value": values.median()},
                {"metric": "mean_delta_dG_separated", "value": values.mean()},
            ]
        )
    summary = pd.DataFrame(rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    return summary


def style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.1)
    ax.tick_params(colors="black", labelsize=9)


def write_figures(scored: pd.DataFrame, figures_dir: Path, delta: pd.DataFrame | None = None) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if "dG_separated" in scored.columns:
        values = pd.to_numeric(scored["dG_separated"], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
        style_axis(ax)
        if values.empty:
            ax.text(0.5, 0.5, "No numeric dG_separated values", ha="center", va="center")
        else:
            ax.hist(values, bins=35, color=POINT_BLUE, alpha=0.82, edgecolor="white", linewidth=0.5)
            ax.axvline(values.median(), color=TITLE_RED, linestyle=(0, (5, 4)), linewidth=1.0, label="median")
            ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.set_title("InterfaceAnalyzer dG Distribution", color=TITLE_RED, fontweight="bold", fontsize=14)
        ax.set_xlabel("dG separated (REU)")
        ax.set_ylabel("Structures")
        fig.tight_layout()
        out = figures_dir / "interface_dg_distribution.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    if delta is not None and {"reference_dG_separated", "dG_separated", "delta_dG_separated"}.issubset(delta.columns):
        plot = delta.dropna(subset=["reference_dG_separated", "dG_separated", "delta_dG_separated"]).copy()
        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=220)
        style_axis(ax)
        if plot.empty:
            ax.text(0.5, 0.5, "No paired reference dG values", ha="center", va="center")
        else:
            scatter = ax.scatter(
                plot["reference_dG_separated"],
                plot["dG_separated"],
                c=plot["delta_dG_separated"],
                cmap="coolwarm",
                s=24,
                alpha=0.82,
                edgecolors="none",
                rasterized=True,
            )
            lo = float(np.nanmin([plot["reference_dG_separated"].min(), plot["dG_separated"].min()]))
            hi = float(np.nanmax([plot["reference_dG_separated"].max(), plot["dG_separated"].max()]))
            ax.plot([lo, hi], [lo, hi], color=DARK_GREY, linewidth=1.0, label="same dG")
            ax.legend(loc="upper left", frameon=False, fontsize=9)
            cbar = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.78)
            cbar.set_label("Delta dG separated (REU)", fontsize=10)
        ax.set_title("InterfaceAnalyzer dG vs Reference", color=TITLE_RED, fontweight="bold", fontsize=14)
        ax.set_xlabel("Reference dG separated (REU)")
        ax.set_ylabel("Scored dG separated (REU)")
        fig.tight_layout()
        out = figures_dir / "interface_delta_dg_vs_reference.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def default_delta_out(out: Path) -> Path:
    return out.with_name(f"{out.stem}_delta.csv")


def default_summary_out(out: Path) -> Path:
    return out.with_name(f"{out.stem}_summary.csv")


def run_interface_analyzer(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    figures_dir = args.figures_dir or args.out.parent / "figures" / "interface_analyzer"
    delta_out = args.delta_out or (default_delta_out(args.out) if args.reference_rosetta is not None else None)
    ensure_outputs_outside_repo(
        [args.out, args.work_dir, figures_dir if not args.skip_figures else None, delta_out, default_summary_out(args.out)],
        allow_repo_output=args.allow_repo_output,
    )
    rows = manifest_rows(
        args.manifest,
        pdb_column=args.pdb_column,
        state=args.state,
        interface=args.interface,
        nanobody_chain=args.nanobody_chain,
        antigen_chain=args.antigen_chain,
        limit=args.limit,
    )
    if args.plan_only:
        print("interface-analyzer: Rosetta InterfaceAnalyzer scoring")
        print(f"manifest: {args.manifest}")
        print(f"rows: {len(rows)}")
        print(f"pdb column: {rows[0]['source_pdb_column'] if rows else '<none>'}")
        print(f"out: {args.out}")
        print(f"work-dir: {args.work_dir}")
        if delta_out:
            print(f"delta-out: {delta_out}")
        if not args.skip_figures:
            print(f"figures-dir: {figures_dir}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config = rosetta_runtime_config_from_args(parser, args)
    try:
        runtime = rosetta_runtime_from_options(runtime_config=config, docker_image=None)
    except RosettaRuntimeError as exc:
        raise SystemExit(f"Rosetta runtime unavailable: {exc}") from exc
    scored = score_manifest_rows(
        rows,
        work_dir=args.work_dir,
        runtime=runtime,
        jobs=args.jobs,
        skip_existing=not args.no_skip_existing,
    )
    scored.to_csv(args.out, index=False)
    delta = None
    if args.reference_rosetta is not None:
        if delta_out is None:
            delta_out = default_delta_out(args.out)
        delta = build_reference_delta(scored, args.reference_rosetta)
        delta_out.parent.mkdir(parents=True, exist_ok=True)
        delta.to_csv(delta_out, index=False)
    write_summary(scored, default_summary_out(args.out), delta)
    if not args.skip_figures:
        write_figures(scored, figures_dir, delta)
    ok = int(scored["interface_analyzer_status"].astype(str).str.contains("ok|skipped_existing_valid", regex=True).sum())
    failures = len(scored) - ok
    print(f"wrote {len(scored)} InterfaceAnalyzer rows to {args.out} ({failures} non-ok)")
    if delta_out is not None:
        print(f"wrote dG delta rows to {delta_out}")
    if not args.skip_figures:
        print(f"wrote InterfaceAnalyzer figures to {figures_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_interface_analyzer(args, parser)
    except ValueError as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
