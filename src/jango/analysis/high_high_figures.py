#!/usr/bin/env python3
"""Canonical high-high DockQ/delta-dG evaluation figures.

This command does not change Jango's binding-perturbation quadrant definition.
Here, "high-high" means:

* DockQ >= --dockq-cutoff
* delta dG >= --delta-dg-cutoff

Inputs may be ordinary CSV files or tar members in the form
``/path/archive.tar.gz:member/path.csv``.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

DOCKQ_CUTOFF = 0.49
DELTA_DG_CUTOFF = 0.0
TM_SCORE_CUTOFF = 90.0
DELTA_DG_DISPLAY_MIN = -50.0
DELTA_DG_DISPLAY_MAX = 50.0
TM_BIN_WIDTH = 5.0

FIGSIZE = (10.8, 7.2)
DPI = 180
BORDER_WIDTH = 1.4
GUIDE_WIDTH = 0.9
POINT_SIZE = 14
HIST_BAR_WIDTH = 3.3
GROUP_BAR_WIDTH = 1.4

GRID_GREY = "#e3e3e3"
DARK_GREY = "#5f5f5f"
TITLE_RED = "#b00020"

MODE_BACKEND_COLORS: dict[tuple[str, str], tuple[str, str]] = {
    ("hotspot", "esm"): ("#f8d4b5", "#c96f1d"),
    ("hotspot", "esmfold2"): ("#f8d4b5", "#c96f1d"),
    ("hotspot", "opendde"): ("#c7d8ee", "#315f90"),
    ("interface", "esm"): ("#f2b6df", "#b21883"),
    ("interface", "esmfold2"): ("#f2b6df", "#b21883"),
    ("interface", "opendde"): ("#bfe7df", "#00897b"),
}
BACKEND_LABELS = {"esm": "ESM", "esmfold2": "ESM", "opendde": "OpenDDE"}
MODE_LABELS = {"hotspot": "Hotspot", "interface": "Interface"}
BACKEND_MARKERS = {"esm": "o", "esmfold2": "o", "opendde": "^"}



@dataclass(frozen=True)
class Config:
    output_root: Path
    dockq_cutoff: float = DOCKQ_CUTOFF
    delta_dg_cutoff: float = DELTA_DG_CUTOFF
    tm_score_cutoff: float = TM_SCORE_CUTOFF
    delta_dg_display_min: float = DELTA_DG_DISPLAY_MIN
    delta_dg_display_max: float = DELTA_DG_DISPLAY_MAX
    tm_bin_width: float = TM_BIN_WIDTH


@dataclass(frozen=True)
class Dataset:
    mode: str
    backend: str
    landscape: pd.DataFrame
    native_refold: pd.DataFrame
    high_high_refold: pd.DataFrame

    @property
    def label(self) -> str:
        return f"{MODE_LABELS[self.mode]} {BACKEND_LABELS.get(self.backend, self.backend)}"

    @property
    def colors(self) -> tuple[str, str]:
        if (self.mode, self.backend) in MODE_BACKEND_COLORS:
            return MODE_BACKEND_COLORS[(self.mode, self.backend)]
        if self.backend.startswith("esm"):
            return MODE_BACKEND_COLORS[(self.mode, "esm")]
        if self.backend.startswith("opendde"):
            return MODE_BACKEND_COLORS[(self.mode, "opendde")]
        raise KeyError(f"no color configured for {self.mode}/{self.backend}")


def marker_for(backend: str) -> str:
    return BACKEND_MARKERS.get(backend, "o")


def resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def split_archive_spec(value: str) -> tuple[str, str] | None:
    for marker in (".tar.gz:", ".tgz:"):
        if marker in value:
            archive, member = value.split(marker, 1)
            return archive + marker[:-1], member
    return None


def read_csv(spec: str) -> pd.DataFrame:
    archive_spec = split_archive_spec(spec)
    if archive_spec is None:
        path = Path(resolve_path(spec))
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)
    archive, member = archive_spec
    archive_path = Path(resolve_path(archive))
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            extracted = tar.extractfile(member)
        except KeyError as exc:
            raise FileNotFoundError(f"archive member not found: {archive_path}:{member}") from exc
        if extracted is None:
            raise FileNotFoundError(f"archive member is not a file: {archive_path}:{member}")
        return pd.read_csv(io.BytesIO(extracted.read()))


def numeric(df: pd.DataFrame, candidates: Iterable[str], label: str) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            data = df[column]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            return pd.to_numeric(data, errors="coerce")
    raise ValueError(f"missing {label} column; tried {', '.join(candidates)}")


def string_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            data = df[column]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            return data.astype(str)
    raise ValueError(f"missing {label} column; tried {', '.join(candidates)}")


def optional_numeric(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series | None:
    for column in candidates:
        if column in df.columns:
            data = df[column]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            return pd.to_numeric(data, errors="coerce")
    return None


def top1_per_structure(df: pd.DataFrame) -> pd.DataFrame:
    if "structure_id" not in df.columns:
        raise ValueError("landscape table requires structure_id")
    table = df.copy()
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "monomer_tm_score" in table.columns:
        table["_rank_tm"] = pd.to_numeric(table["monomer_tm_score"], errors="coerce")
        sort_columns.append("_rank_tm")
        ascending.append(False)
    if "monomer_ca_rmsd" in table.columns:
        table["_rank_rmsd"] = pd.to_numeric(table["monomer_ca_rmsd"], errors="coerce")
        sort_columns.append("_rank_rmsd")
        ascending.append(True)
    if "design_id" in table.columns:
        table["_rank_design"] = table["design_id"].astype(str)
        sort_columns.append("_rank_design")
        ascending.append(True)
    if sort_columns:
        table = table.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    return table.drop_duplicates(subset=["structure_id"], keep="first").drop(
        columns=[c for c in ["_rank_tm", "_rank_rmsd", "_rank_design"] if c in table.columns]
    )


def load_native_refold(path: str) -> pd.DataFrame:
    table = read_csv(path)
    if "similarity_status" in table.columns:
        table = table[table["similarity_status"].astype(str).str.lower().eq("ok")].copy()
    structure_id = string_column(table, ["structure_id", "native_id", "case_id", "pdb_id"], "native refold structure id")
    tm = numeric(
        table,
        ["tm_percent", "native_refold_tm_percent", "antigen_ca_tm_score", "tm_score", "tm_score_antigen"],
        "native-refold TM-score",
    )
    if tm.max(skipna=True) <= 1.5:
        tm = tm * 100.0
    out = pd.DataFrame({"structure_id": structure_id, "native_refold_tm_percent": tm})
    return out.dropna(subset=["structure_id", "native_refold_tm_percent"]).copy()


def load_landscape(path: str, mode: str, backend: str, native_refold: pd.DataFrame, config: Config) -> pd.DataFrame:
    raw = read_csv(path)
    table = top1_per_structure(raw)
    out = table.copy()
    out["structure_id"] = string_column(out, ["structure_id", "native_id", "case_id", "pdb_id"], "structure id")
    if "design_id" not in out.columns:
        out["design_id"] = out["structure_id"]
    out["design_id"] = out["design_id"].astype(str)
    out["redesign_mode"] = mode
    out["backend"] = backend
    out["fold_backend"] = backend
    out["dockq"] = numeric(out, ["dockq", "dockq_score", "DockQ"], "DockQ")
    out["delta_dg"] = numeric(
        out,
        ["delta_dg", "delta_vs_native_dG_separated", "delta_delta_g", "delta_dG", "ddg", "binding_delta_delta_g"],
        "delta dG",
    )
    structure = optional_numeric(out, ["structure_preservation"])
    out["structure_preservation"] = structure if structure is not None else out["dockq"]
    bp = optional_numeric(out, ["binding_perturbation", "Binding Perturbation Index"])
    out["binding_perturbation"] = bp if bp is not None else pd.Series(np.nan, index=out.index)
    out = out.dropna(subset=["structure_id", "dockq", "delta_dg"]).copy()
    out = out.drop(columns=["native_refold_tm_percent", "native_refold_tm_cutoff", "native_refold_quality_pass"], errors="ignore")
    out = out.merge(native_refold, on="structure_id", how="left")
    out["native_refold_tm_cutoff"] = config.tm_score_cutoff
    out["native_refold_quality_pass"] = out["native_refold_tm_percent"] >= config.tm_score_cutoff
    out["high_high"] = (out["dockq"] >= config.dockq_cutoff) & (out["delta_dg"] >= config.delta_dg_cutoff)
    out["dockq_threshold"] = config.dockq_cutoff
    out["delta_dg_threshold"] = config.delta_dg_cutoff
    out["high_high_definition"] = f"dockq>={config.dockq_cutoff};delta_dg>={config.delta_dg_cutoff}"
    out["high_high_coordinate_score"] = out["dockq"] + out["delta_dg"]
    out["_high_sort"] = (~out["high_high"]).astype(int)
    out["_score_sort"] = -pd.to_numeric(out["high_high_coordinate_score"], errors="coerce")
    out["_dockq_sort"] = -pd.to_numeric(out["dockq"], errors="coerce")
    out = out.sort_values(["_high_sort", "_score_sort", "_dockq_sort", "structure_id", "design_id"], kind="mergesort")
    out["high_high_rank"] = np.nan
    high_index = out.index[out["high_high"]]
    out.loc[high_index, "high_high_rank"] = np.arange(1, len(high_index) + 1)
    return out.drop(columns=["_high_sort", "_score_sort", "_dockq_sort"])


def setup_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TITLE_RED, fontweight="bold", fontsize=16, pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color=GRID_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(BORDER_WIDTH)


def legend(ax: plt.Axes, *, handles: Sequence[object] = (), ncol: int = 2) -> None:
    current_handles, current_labels = ax.get_legend_handles_labels()
    all_handles = list(handles) + current_handles
    all_labels = [h.get_label() for h in handles] + current_labels
    seen: set[str] = set()
    out_handles: list[object] = []
    out_labels: list[str] = []
    for handle, label in zip(all_handles, all_labels):
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        out_handles.append(handle)
        out_labels.append(label)
    ax.legend(
        out_handles,
        out_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.16),
        frameon=False,
        fontsize=10,
        ncol=ncol,
        borderaxespad=0,
        handlelength=2.4,
        columnspacing=1.2,
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def display_landscape(table: pd.DataFrame, config: Config) -> pd.DataFrame:
    return table[
        (table["delta_dg"] >= config.delta_dg_display_min)
        & (table["delta_dg"] <= config.delta_dg_display_max)
    ].copy()


def plot_landscape(data: Dataset, config: Config, out: Path) -> None:
    table = display_landscape(data.landscape, config)
    light, dark = data.colors
    high_refold = table["native_refold_quality_pass"].fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(table.loc[~high_refold, "dockq"], table.loc[~high_refold, "delta_dg"], s=POINT_SIZE, color=light, alpha=0.76, edgecolors="none", marker=marker_for(data.backend), label=f"{data.label} <90% native refold")
    ax.scatter(table.loc[high_refold, "dockq"], table.loc[high_refold, "delta_dg"], s=POINT_SIZE, color=dark, alpha=0.88, edgecolors="none", marker=marker_for(data.backend), label=f"{data.label} >=90% native refold")
    ax.axvline(config.dockq_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    ax.axhline(config.delta_dg_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    if len(table):
        ax.axvline(float(table["dockq"].median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
        ax.axhline(float(table["delta_dg"].median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
    setup_axis(ax, f"{data.label} Decoy vs Refold", "DockQ score", "Delta dG (REU)")
    ax.set_ylim(config.delta_dg_display_min, config.delta_dg_display_max)
    legend(ax, ncol=2)
    save(fig, out)


def plot_refold_hist(data: Dataset, config: Config, out: Path, *, high_high_only: bool) -> None:
    table = data.high_high_refold if high_high_only else data.native_refold
    values = table["native_refold_tm_percent"].dropna()
    bins = np.arange(0, 100 + config.tm_bin_width, config.tm_bin_width)
    centers = bins[:-1] + config.tm_bin_width / 2
    counts, _ = np.histogram(values, bins=bins)
    light, dark = data.colors
    colors = [dark if left >= config.tm_score_cutoff else light for left in bins[:-1]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(centers, counts, width=HIST_BAR_WIDTH, color=colors, alpha=0.88, edgecolor="white", linewidth=0.7)
    ax.axvline(config.tm_score_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    if len(values):
        ax.axvline(float(values.median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
    title = f"{data.label} High-High Native Refold Quality" if high_high_only else f"{data.label} Native Refold Quality"
    ylabel = "High-high native structures" if high_high_only else "Structures"
    setup_axis(ax, title, "TM-score (%)", ylabel)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 105, 5))
    handles = [
        mpatches.Patch(color=light, label=f"{data.label} <90% native refold", alpha=0.88),
        mpatches.Patch(color=dark, label=f"{data.label} >=90% native refold", alpha=0.88),
    ]
    legend(ax, handles=handles, ncol=2)
    save(fig, out)


def plot_comparison_landscape(items: Sequence[Dataset], config: Config, out: Path, title: str, *, marker_by_backend: bool = True) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for data in items:
        table = display_landscape(data.landscape, config)
        light, dark = data.colors
        marker = marker_for(data.backend) if marker_by_backend else ("o" if data.mode == "hotspot" else "s")
        high_refold = table["native_refold_quality_pass"].fillna(False).astype(bool)
        ax.scatter(table.loc[~high_refold, "dockq"], table.loc[~high_refold, "delta_dg"], s=POINT_SIZE, color=light, alpha=0.66, edgecolors="none", marker=marker, label=f"{data.label} <90% native refold")
        ax.scatter(table.loc[high_refold, "dockq"], table.loc[high_refold, "delta_dg"], s=POINT_SIZE, color=dark, alpha=0.82, edgecolors="none", marker=marker, label=f"{data.label} >=90% native refold")
    ax.axvline(config.dockq_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    ax.axhline(config.delta_dg_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    setup_axis(ax, title, "DockQ score", "Delta dG (REU)")
    ax.set_ylim(config.delta_dg_display_min, config.delta_dg_display_max)
    legend(ax, ncol=3)
    save(fig, out)


def plot_comparison_hist(items: Sequence[Dataset], config: Config, out: Path, *, high_high_only: bool, title: str) -> None:
    bins = np.arange(0, 100 + config.tm_bin_width, config.tm_bin_width)
    centers = bins[:-1] + config.tm_bin_width / 2
    offsets = np.linspace(-GROUP_BAR_WIDTH / 2, GROUP_BAR_WIDTH / 2, len(items)) if len(items) > 1 else [0]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles: list[object] = []
    for idx, data in enumerate(items):
        table = data.high_high_refold if high_high_only else data.native_refold
        values = table["native_refold_tm_percent"].dropna()
        counts, _ = np.histogram(values, bins=bins)
        light, dark = data.colors
        colors = [dark if left >= config.tm_score_cutoff else light for left in bins[:-1]]
        ax.bar(centers + offsets[idx], counts, width=GROUP_BAR_WIDTH, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
        handles.extend([
            mpatches.Patch(color=light, label=f"{data.label} <90%", alpha=0.88),
            mpatches.Patch(color=dark, label=f"{data.label} >=90%", alpha=0.88),
        ])
    ax.axvline(config.tm_score_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    ylabel = "High-high native structures" if high_high_only else "Structures"
    setup_axis(ax, title, "TM-score (%)", ylabel)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 105, 5))
    legend(ax, handles=handles, ncol=3)
    save(fig, out)


def summary_row(data: Dataset, config: Config) -> dict[str, object]:
    landscape = data.landscape
    display = display_landscape(landscape, config)
    high = landscape[landscape["high_high"]]
    high_tm = high[high["native_refold_tm_percent"] >= config.tm_score_cutoff]
    return {
        "redesign_mode": data.mode,
        "backend": data.backend,
        "n_total_decoys": int(len(landscape)),
        "n_plotted": int(len(display)),
        "n_hidden_display_outliers": int(len(landscape) - len(display)),
        "n_high_high_decoys": int(len(high)),
        "n_total_refolds": int(len(data.native_refold)),
        "n_refolds_tm_ge_90": int((data.native_refold["native_refold_tm_percent"] >= config.tm_score_cutoff).sum()),
        "median_refold_tm_percent": float(data.native_refold["native_refold_tm_percent"].median()) if len(data.native_refold) else None,
        "n_high_high_refold_tm_ge_90": int((high["native_refold_tm_percent"] >= config.tm_score_cutoff).sum()),
        "median_refold_tm_percent_high_high": float(high["native_refold_tm_percent"].median()) if len(high) else None,
        "high_high_dockq_median": float(high["dockq"].median()) if len(high) else None,
        "high_high_delta_dg_median": float(high["delta_dg"].median()) if len(high) else None,
        "high_high_dockq_median_tm_ge_90": float(high_tm["dockq"].median()) if len(high_tm) else None,
        "high_high_delta_dg_median_tm_ge_90": float(high_tm["delta_dg"].median()) if len(high_tm) else None,
        "dockq_cutoff": config.dockq_cutoff,
        "delta_dg_cutoff": config.delta_dg_cutoff,
        "native_refold_tm_cutoff": config.tm_score_cutoff,
    }


def write_tables(data: Dataset, base: Path, config: Config) -> pd.DataFrame:
    tables = base / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    high = data.landscape[data.landscape["high_high"]].copy()
    high.to_csv(tables / "high_high_decoys.csv", index=False)
    data.landscape.to_csv(tables / "high_high_decoy_ranked_ml_dataset.csv", index=False)
    data.native_refold.to_csv(tables / "native_refold_tm_scores.csv", index=False)
    data.high_high_refold.to_csv(tables / "high_high_native_refold_tm_scores.csv", index=False)
    summary = pd.DataFrame([summary_row(data, config)])
    summary.to_csv(tables / "high_high_core_figure_summary.csv", index=False)
    (tables / "high_high_core_figure_summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2) + "\n")
    return summary


def load_dataset(mode: str, backend: str, landscape_path: str, refold_path: str, config: Config) -> Dataset:
    native_refold = load_native_refold(refold_path)
    landscape = load_landscape(landscape_path, mode, backend, native_refold, config)
    high_ids = set(landscape.loc[landscape["high_high"], "structure_id"])
    high_refold = native_refold[native_refold["structure_id"].isin(high_ids)].copy()
    if landscape["structure_id"].duplicated().any():
        raise ValueError(f"duplicate structure_id after top1 selection for {mode}/{backend}")
    return Dataset(mode=mode, backend=backend, landscape=landscape, native_refold=native_refold, high_high_refold=high_refold)


def add_path_arg(parser: argparse.ArgumentParser, mode: str, backend: str) -> None:
    parser.add_argument(f"--{mode}-{backend}-landscape", required=True)


def add_refold_arg(parser: argparse.ArgumentParser, mode: str, backend: str) -> None:
    parser.add_argument(f"--{mode}-{backend}-native-refold", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="high-high-figures", description="Write compatibility Q1 DockQ/delta-dG figure sets.")
    parser.add_argument("--output-root", type=Path, default=Path("data/outputs/evaluation/q1_core"))
    parser.add_argument("--esm-backend-key", default="esm", help="Output key for the ESM-family backend, e.g. esmfold2.")
    parser.add_argument("--esm-label", default="ESM", help="Display label for the ESM-family backend.")
    parser.add_argument("--opendde-backend-key", default="opendde", help="Output key for the OpenDDE backend.")
    parser.add_argument("--opendde-label", default="OpenDDE", help="Display label for the OpenDDE backend.")
    for mode in ["hotspot", "interface"]:
        for backend in ["esm", "opendde"]:
            add_path_arg(parser, mode, backend)
            add_refold_arg(parser, mode, backend)
    parser.add_argument("--esm-native-refold", default=None, help="Native-refold TM/similarity CSV used for all ESM-family modes unless mode-specific paths are provided.")
    parser.add_argument("--opendde-native-refold", default=None, help="Native-refold TM/similarity CSV used for all OpenDDE modes unless mode-specific paths are provided.")
    parser.add_argument("--dockq-cutoff", type=float, default=DOCKQ_CUTOFF)
    parser.add_argument("--delta-dg-cutoff", type=float, default=DELTA_DG_CUTOFF)
    parser.add_argument("--tm-score-cutoff", type=float, default=TM_SCORE_CUTOFF)
    parser.add_argument("--delta-dg-display-min", type=float, default=DELTA_DG_DISPLAY_MIN)
    parser.add_argument("--delta-dg-display-max", type=float, default=DELTA_DG_DISPLAY_MAX)
    return parser


def refold_arg(args: argparse.Namespace, mode: str, backend: str) -> str:
    mode_specific = getattr(args, f"{mode}_{backend}_native_refold")
    if mode_specific:
        return mode_specific
    fallback = getattr(args, f"{backend}_native_refold")
    if fallback:
        return fallback
    raise ValueError(
        f"missing native-refold input for {mode}/{backend}; provide "
        f"--{mode}-{backend}-native-refold or --{backend}-native-refold"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(
        output_root=args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root,
        dockq_cutoff=args.dockq_cutoff,
        delta_dg_cutoff=args.delta_dg_cutoff,
        tm_score_cutoff=args.tm_score_cutoff,
        delta_dg_display_min=args.delta_dg_display_min,
        delta_dg_display_max=args.delta_dg_display_max,
    )
    backend_keys = {"esm": args.esm_backend_key, "opendde": args.opendde_backend_key}
    BACKEND_LABELS[args.esm_backend_key] = args.esm_label
    BACKEND_LABELS[args.opendde_backend_key] = args.opendde_label
    datasets: dict[tuple[str, str], Dataset] = {}
    for mode in ["hotspot", "interface"]:
        for input_backend in ["esm", "opendde"]:
            attr = f"{mode}_{input_backend}_landscape"
            datasets[(mode, input_backend)] = load_dataset(
                mode,
                backend_keys[input_backend],
                getattr(args, attr),
                refold_arg(args, mode, input_backend),
                config,
            )

    summary_parts: list[pd.DataFrame] = []
    for mode in ["hotspot", "interface"]:
        mode_summaries: list[pd.DataFrame] = []
        for input_backend in ["esm", "opendde"]:
            data = datasets[(mode, input_backend)]
            base = config.output_root / mode / data.backend
            summary = write_tables(data, base, config)
            summary_parts.append(summary)
            mode_summaries.append(summary)
            plot_landscape(data, config, base / "figures" / "landscape_dockq_delta_dg.png")
            plot_refold_hist(data, config, base / "figures" / "native_refold_tm_histogram.png", high_high_only=False)
            plot_refold_hist(data, config, base / "figures" / "high_high_native_refold_tm_histogram.png", high_high_only=True)

        mode_items = [datasets[(mode, "esm")], datasets[(mode, "opendde")]]
        comparison = config.output_root / mode / "comparison"
        (comparison / "tables").mkdir(parents=True, exist_ok=True)
        plot_comparison_landscape(mode_items, config, comparison / "figures" / "backend_comparison_landscape_dockq_delta_dg.png", f"{MODE_LABELS[mode]} Decoy vs Refold Comparison")
        plot_comparison_hist(mode_items, config, comparison / "figures" / "backend_comparison_native_refold_tm_histogram.png", high_high_only=False, title=f"{MODE_LABELS[mode]} Native Refold Quality Comparison")
        plot_comparison_hist(mode_items, config, comparison / "figures" / "backend_comparison_high_high_native_refold_tm_histogram.png", high_high_only=True, title=f"{MODE_LABELS[mode]} High-High Native Refold Quality Comparison")
        pd.concat(mode_summaries, ignore_index=True).to_csv(comparison / "tables" / "high_high_core_figure_summary.csv", index=False)

    for input_backend in ["esm", "opendde"]:
        items = [datasets[("hotspot", input_backend)], datasets[("interface", input_backend)]]
        out_dir = config.output_root / "comparison_by_mode" / items[0].backend
        (out_dir / "tables").mkdir(parents=True, exist_ok=True)
        plot_comparison_landscape(items, config, out_dir / "figures" / "redesign_mode_landscape_dockq_delta_dg.png", f"{BACKEND_LABELS.get(items[0].backend, items[0].backend)} Redesign Mode Decoy vs Refold", marker_by_backend=False)
        pd.DataFrame([summary_row(item, config) for item in items]).to_csv(
            out_dir / "tables" / "high_high_core_figure_summary.csv", index=False
        )

    combined = pd.concat(summary_parts, ignore_index=True)
    comparison_root = config.output_root / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(comparison_root / "high_high_core_figure_summary.csv", index=False)
    (comparison_root / "high_high_core_figure_summary.json").write_text(json.dumps(combined.to_dict(orient="records"), indent=2) + "\n")
    print(f"wrote high-high figures and tables to {config.output_root}")
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
