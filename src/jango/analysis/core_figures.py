"""Compact canonical Jango evaluation figures and graph-summary tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


ESM_COLOR = "#f8d4b5"
OPENDDE_COLOR = "#c7d8ee"
ESM_DARK = "#c96f1d"
OPENDDE_DARK = "#315f90"
FAIL_GREY = "#b8b8b8"
GRID_GREY = "#e3e3e3"
DARK_GREY = "#5f5f5f"
TITLE_RED = "#b00020"

DOCKQ_CUTOFF = 0.49
DELTA_DG_CUTOFF = 0.0
BINDING_PERTURBATION_CUTOFF = 0.0
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

ML_DATASET_CSV = "decoy_ranked_ml_dataset.csv"
BACKEND_TABLE_OUTPUTS = {
    ML_DATASET_CSV,
    "native_refold_tm_scores.csv",
    "q1_native_refold_tm_scores.csv",
    "core_figure_summary.csv",
    "core_figure_summary.json",
}
COMPARISON_TABLE_OUTPUTS = {"core_figure_summary.csv", "core_figure_summary.json"}
QUADRANT_ORDER = ("Q1", "Q4", "Q2", "Q3")
QUADRANT_PRIORITY = {quadrant: idx + 1 for idx, quadrant in enumerate(QUADRANT_ORDER)}
QUADRANT_INTERPRETATION = {
    "Q1": "structure_preserved_binding_perturbed",
    "Q4": "structure_preserved_binding_not_perturbed",
    "Q2": "structure_not_preserved_binding_perturbed",
    "Q3": "structure_not_preserved_binding_not_perturbed",
    "unclassified": "missing_or_invalid_structure_preservation_or_binding_perturbation",
}
Q1_SUMMARY_METRICS = [
    "dockq",
    "delta_dg",
    "binding_perturbation",
    "coordinate_score",
    "monomer_tm_score",
    "monomer_ca_rmsd",
    "dockq_fnat",
    "dockq_irmsd",
    "dockq_lrmsd",
    "n_residues_mutated",
    "mutation_fraction",
    "native_refold_tm_percent",
]


@dataclass(frozen=True)
class BackendSpec:
    key: str
    label: str
    color: str
    dark_color: str
    marker: str
    landscape_csv: Path
    native_refold_csv: Path


@dataclass(frozen=True)
class FigureConfig:
    evaluation_root: Path
    dockq_cutoff: float = DOCKQ_CUTOFF
    delta_dg_cutoff: float = DELTA_DG_CUTOFF
    binding_perturbation_cutoff: float = BINDING_PERTURBATION_CUTOFF
    tm_score_cutoff: float = TM_SCORE_CUTOFF
    delta_dg_display_min: float = DELTA_DG_DISPLAY_MIN
    delta_dg_display_max: float = DELTA_DG_DISPLAY_MAX
    tm_bin_width: float = TM_BIN_WIDTH


@dataclass(frozen=True)
class BackendData:
    spec: BackendSpec
    landscape: pd.DataFrame
    native_refold: pd.DataFrame
    q1_refold: pd.DataFrame


def _setup_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TITLE_RED, fontweight="bold", fontsize=16, pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color=GRID_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(BORDER_WIDTH)


def _legend(ax: plt.Axes, *, handles: Sequence[object] = (), ncol: int = 2) -> None:
    current_handles, current_labels = ax.get_legend_handles_labels()
    all_handles = list(handles) + current_handles
    all_labels = [handle.get_label() for handle in handles] + current_labels
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


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _numeric(df: pd.DataFrame, candidates: Iterable[str], label: str) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    raise ValueError(f"missing {label} column; tried {', '.join(candidates)}")


def _optional_numeric(df: pd.DataFrame, candidates: Iterable[str], fallback: pd.Series) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            return values.combine_first(fallback)
    return fallback


def _top1_per_structure(df: pd.DataFrame) -> pd.DataFrame:
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
        table = table.sort_values(sort_columns, ascending=ascending)
    return table.drop_duplicates(subset=["structure_id"], keep="first")


def load_landscape(path: Path) -> pd.DataFrame:
    table = _top1_per_structure(pd.read_csv(path))
    out = table.copy()
    out["structure_id"] = out["structure_id"].astype(str)
    out["dockq"] = _numeric(out, ["dockq", "dockq_score", "DockQ"], "DockQ")
    out["delta_dg"] = _numeric(
        out,
        [
            "delta_vs_native_dG_separated",
            "delta_delta_g",
            "delta_dG",
            "delta_dg",
            "ddg",
            "binding_delta_delta_g",
        ],
        "delta dG",
    )
    out["structure_preservation"] = _optional_numeric(out, ["structure_preservation"], out["dockq"])
    out["binding_perturbation"] = _optional_numeric(
        out,
        ["binding_perturbation", "Binding Perturbation Index"],
        out["delta_dg"],
    )
    return out.dropna(subset=["structure_id", "dockq", "delta_dg"]).copy()


def load_native_refold(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "similarity_status" in table.columns:
        table = table[table["similarity_status"].astype(str).str.lower().eq("ok")].copy()
    if "structure_id" not in table.columns:
        raise ValueError("native-refold table requires structure_id")
    if "tm_percent" in table.columns:
        tm_percent = _numeric(table, ["tm_percent"], "native-refold TM-score percent")
    else:
        tm_percent = _numeric(
            table,
            ["antigen_ca_tm_score", "tm_score", "tm_score_antigen"],
            "native-refold TM-score",
        ) * 100.0
    out = pd.DataFrame({"structure_id": table["structure_id"].astype(str), "tm_percent": tm_percent})
    return out.dropna(subset=["structure_id", "tm_percent"]).copy()


def _classify_quadrants(table: pd.DataFrame, config: FigureConfig) -> pd.DataFrame:
    out = table.copy()
    structure = pd.to_numeric(out["structure_preservation"], errors="coerce")
    delta_dg = pd.to_numeric(out["delta_dg"], errors="coerce")
    binding_perturbation = pd.to_numeric(out["binding_perturbation"], errors="coerce")
    valid = (
        structure.notna()
        & binding_perturbation.notna()
        & np.isfinite(structure)
        & np.isfinite(binding_perturbation)
    )
    structure_high = structure >= config.dockq_cutoff
    binding_high = binding_perturbation >= config.binding_perturbation_cutoff

    out["structure_preservation"] = structure
    out["delta_dg"] = delta_dg
    out["binding_perturbation"] = binding_perturbation
    out["structure_preservation_metric"] = out.get("structure_preservation_metric", "dockq")
    out["structure_preservation_threshold"] = config.dockq_cutoff
    out["delta_dg_threshold"] = config.delta_dg_cutoff
    out["binding_perturbation_threshold"] = config.binding_perturbation_cutoff
    out["binding_perturbation_metric"] = out.get(
        "binding_perturbation_metric",
        "delphi_binding_perturbation_index",
    )
    out["quadrant_binding_metric"] = out["binding_perturbation_metric"]
    out["landscape_axis_pass"] = structure_high & binding_high

    out["quadrant"] = "unclassified"
    out.loc[valid & structure_high & binding_high, "quadrant"] = "Q1"
    out.loc[valid & structure_high & ~binding_high, "quadrant"] = "Q4"
    out.loc[valid & ~structure_high & binding_high, "quadrant"] = "Q2"
    out.loc[valid & ~structure_high & ~binding_high, "quadrant"] = "Q3"
    out["quadrant_order"] = out["quadrant"].map(QUADRANT_PRIORITY)
    out["quadrant_interpretation"] = out["quadrant"].map(QUADRANT_INTERPRETATION)
    out["is_q1"] = out["quadrant"].eq("Q1")
    out["coordinate_score"] = structure + binding_perturbation
    out["delphi_score"] = out["coordinate_score"]

    out["_quadrant_sort"] = out["quadrant_order"].fillna(999).astype(float)
    out["_coordinate_sort"] = -pd.to_numeric(out["coordinate_score"], errors="coerce")
    out["_structure_sort"] = -pd.to_numeric(out["structure_preservation"], errors="coerce")
    out["_design_sort"] = out.get("design_id", pd.Series("", index=out.index)).astype(str)
    out = out.sort_values(
        ["_quadrant_sort", "_coordinate_sort", "_structure_sort", "structure_id", "_design_sort"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )
    out["quadrant_rank"] = out.groupby("quadrant", dropna=False).cumcount() + 1
    return out.drop(columns=["_quadrant_sort", "_coordinate_sort", "_structure_sort", "_design_sort"])


def _ordered_ml_columns(table: pd.DataFrame) -> list[str]:
    preferred = [
        "backend",
        "fold_backend",
        "structure_id",
        "design_id",
        "redesign_mode",
        "quadrant",
        "quadrant_rank",
        "quadrant_order",
        "quadrant_interpretation",
        "is_q1",
        "coordinate_score",
        "delphi_score",
        "dockq",
        "delta_dg",
        "delta_dg_threshold",
        "delta_vs_native_dG_separated",
        "structure_preservation",
        "structure_preservation_threshold",
        "quadrant_binding_metric",
        "binding_perturbation",
        "binding_perturbation_metric",
        "binding_perturbation_threshold",
        "landscape_axis_pass",
        "native_refold_tm_percent",
        "native_refold_tm_cutoff",
        "native_refold_quality_pass",
        "monomer_tm_score",
        "monomer_ca_rmsd",
        "dockq_fnat",
        "dockq_irmsd",
        "dockq_lrmsd",
        "n_residues_mutated",
        "mutation_fraction",
        "decoy_relaxed_pdb",
        "relaxed_pdb",
        "scored_pdb",
        "native_relaxed_pdb",
        "native_reference_pdb",
    ]
    return [column for column in preferred if column in table.columns] + [
        column for column in table.columns if column not in preferred
    ]


def _attach_native_refold_quality(
    landscape: pd.DataFrame,
    native_refold: pd.DataFrame,
    config: FigureConfig,
) -> pd.DataFrame:
    refold = native_refold.rename(columns={"tm_percent": "native_refold_tm_percent"})[
        ["structure_id", "native_refold_tm_percent"]
    ]
    out = landscape.drop(
        columns=["native_refold_tm_percent", "native_refold_tm_cutoff", "native_refold_quality_pass"],
        errors="ignore",
    ).merge(refold, on="structure_id", how="left")
    out["native_refold_tm_cutoff"] = config.tm_score_cutoff
    out["native_refold_quality_pass"] = out["native_refold_tm_percent"] >= config.tm_score_cutoff
    return out


def load_backend_data(spec: BackendSpec, config: FigureConfig) -> BackendData:
    landscape = load_landscape(spec.landscape_csv)
    landscape["backend"] = spec.key
    if "fold_backend" not in landscape.columns:
        landscape["fold_backend"] = spec.key
    native_refold = load_native_refold(spec.native_refold_csv)
    landscape = _classify_quadrants(landscape, config)
    landscape = _attach_native_refold_quality(landscape, native_refold, config)
    landscape = landscape[_ordered_ml_columns(landscape)]
    q1_ids = set(landscape.loc[landscape["is_q1"], "structure_id"])
    q1_refold = native_refold[native_refold["structure_id"].isin(q1_ids)].copy()
    return BackendData(spec=spec, landscape=landscape, native_refold=native_refold, q1_refold=q1_refold)


def _display_landscape(table: pd.DataFrame, config: FigureConfig) -> pd.DataFrame:
    return table[
        (table["delta_dg"] >= config.delta_dg_display_min)
        & (table["delta_dg"] <= config.delta_dg_display_max)
    ].copy()


def plot_landscape(data: BackendData, config: FigureConfig, out: Path) -> None:
    table = _display_landscape(data.landscape, config)
    high_refold = table["native_refold_quality_pass"].fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(table.loc[~high_refold, "dockq"], table.loc[~high_refold, "delta_dg"], s=POINT_SIZE, color=data.spec.color, alpha=0.76, edgecolors="none", marker=data.spec.marker, label=f"{data.spec.label} <90% native refold")
    ax.scatter(table.loc[high_refold, "dockq"], table.loc[high_refold, "delta_dg"], s=POINT_SIZE, color=data.spec.dark_color, alpha=0.88, edgecolors="none", marker=data.spec.marker, label=f"{data.spec.label} >=90% native refold")
    ax.axvline(config.dockq_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    ax.axhline(config.delta_dg_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    ax.axvline(float(table["dockq"].median()), color=data.spec.dark_color, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
    ax.axhline(float(table["delta_dg"].median()), color=data.spec.dark_color, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
    _setup_axis(ax, f"{data.spec.label} Decoy vs Refold", "DockQ score", "Delta dG (REU)")
    ax.set_ylim(config.delta_dg_display_min, config.delta_dg_display_max)
    _legend(ax, ncol=2)
    _save(fig, out)


def plot_refold_hist(data: BackendData, config: FigureConfig, out: Path, *, q1_only: bool) -> None:
    table = data.q1_refold if q1_only else data.native_refold
    values = table["tm_percent"].dropna()
    bins = np.arange(0, 100 + config.tm_bin_width, config.tm_bin_width)
    centers = bins[:-1] + config.tm_bin_width / 2
    counts, _ = np.histogram(values, bins=bins)
    colors = [data.spec.dark_color if left >= config.tm_score_cutoff else data.spec.color for left in bins[:-1]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(centers, counts, width=HIST_BAR_WIDTH, color=colors, alpha=0.88, edgecolor="white", linewidth=0.7)
    ax.axvline(config.tm_score_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    if len(values):
        ax.axvline(float(values.median()), color=data.spec.dark_color, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
    title = f"{data.spec.label} Q1 Native Refold Quality" if q1_only else f"{data.spec.label} Native Refold Quality"
    ylabel = "Q1-native structures" if q1_only else "Structures"
    _setup_axis(ax, title, "TM-score (%)", ylabel)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 105, 5))
    handles = [
        mpatches.Patch(color=data.spec.color, label=f"{data.spec.label} <90% native refold", alpha=0.88),
        mpatches.Patch(color=data.spec.dark_color, label=f"{data.spec.label} >=90% native refold", alpha=0.88),
    ]
    _legend(ax, handles=handles, ncol=2)
    _save(fig, out)


def plot_comparison_landscape(backends: Sequence[BackendData], config: FigureConfig, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for data in backends:
        table = _display_landscape(data.landscape, config)
        high_refold = table["native_refold_quality_pass"].fillna(False).astype(bool)
        ax.scatter(table.loc[~high_refold, "dockq"], table.loc[~high_refold, "delta_dg"], s=POINT_SIZE, color=data.spec.color, alpha=0.66, edgecolors="none", marker=data.spec.marker, label=f"{data.spec.label} <90% native refold")
        ax.scatter(table.loc[high_refold, "dockq"], table.loc[high_refold, "delta_dg"], s=POINT_SIZE, color=data.spec.dark_color, alpha=0.82, edgecolors="none", marker=data.spec.marker, label=f"{data.spec.label} >=90% native refold")
    ax.axvline(config.dockq_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    ax.axhline(config.delta_dg_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    _setup_axis(ax, "Decoy vs Refold Comparison", "DockQ score", "Delta dG (REU)")
    ax.set_ylim(config.delta_dg_display_min, config.delta_dg_display_max)
    _legend(ax, ncol=3)
    _save(fig, out)


def plot_comparison_hist(backends: Sequence[BackendData], config: FigureConfig, out: Path, *, q1_only: bool) -> None:
    bins = np.arange(0, 100 + config.tm_bin_width, config.tm_bin_width)
    centers = bins[:-1] + config.tm_bin_width / 2
    offsets = [-GROUP_BAR_WIDTH / 2, GROUP_BAR_WIDTH / 2]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles: list[object] = []
    for idx, data in enumerate(backends):
        table = data.q1_refold if q1_only else data.native_refold
        values = table["tm_percent"].dropna()
        counts, _ = np.histogram(values, bins=bins)
        colors = [data.spec.dark_color if left >= config.tm_score_cutoff else data.spec.color for left in bins[:-1]]
        ax.bar(centers + offsets[idx], counts, width=GROUP_BAR_WIDTH, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
        handles.extend(
            [
                mpatches.Patch(color=data.spec.color, label=f"{data.spec.label} <90%", alpha=0.88),
                mpatches.Patch(color=data.spec.dark_color, label=f"{data.spec.label} >=90%", alpha=0.88),
            ]
        )
    ax.axvline(config.tm_score_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    title = "Q1 Native Refold Quality Comparison" if q1_only else "Native Refold Quality Comparison"
    ylabel = "Q1-native structures" if q1_only else "Structures"
    _setup_axis(ax, title, "TM-score (%)", ylabel)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 105, 5))
    _legend(ax, handles=handles, ncol=3)
    _save(fig, out)


def _series_stats(values: pd.Series, cutoff: float | None = None) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    stats: dict[str, object] = {
        "n": int(len(clean)),
        "mean": float(clean.mean()) if len(clean) else None,
        "median": float(clean.median()) if len(clean) else None,
        "min": float(clean.min()) if len(clean) else None,
        "max": float(clean.max()) if len(clean) else None,
    }
    if cutoff is not None:
        stats["cutoff"] = cutoff
        stats["n_above_or_equal_cutoff"] = int((clean >= cutoff).sum())
        stats["n_below_cutoff"] = int((clean < cutoff).sum())
    return stats


def _q1_summary_stats(q1: pd.DataFrame) -> dict[str, object]:
    stats: dict[str, object] = {}
    for metric in Q1_SUMMARY_METRICS:
        if metric not in q1.columns:
            continue
        values = pd.to_numeric(q1[metric], errors="coerce").dropna()
        prefix = f"q1_{metric}"
        stats[f"{prefix}_n"] = int(len(values))
        stats[f"{prefix}_mean"] = float(values.mean()) if len(values) else None
        stats[f"{prefix}_median"] = float(values.median()) if len(values) else None
    return stats


def summary_rows(data: BackendData, config: FigureConfig) -> list[dict[str, object]]:
    landscape = data.landscape
    display = _display_landscape(landscape, config)
    q1 = landscape[landscape["is_q1"]]
    return [
        {
            "backend": data.spec.key,
            "figure": "landscape_dockq_delta_dg",
            "n_total": len(landscape),
            "n_plotted": len(display),
            "n_hidden_display_outliers": len(landscape) - len(display),
            "x_metric": "dockq",
            "x_mean": float(landscape["dockq"].mean()),
            "x_median": float(landscape["dockq"].median()),
            "x_cutoff": config.dockq_cutoff,
            "x_n_above_or_equal_cutoff": int((landscape["dockq"] >= config.dockq_cutoff).sum()),
            "x_n_below_cutoff": int((landscape["dockq"] < config.dockq_cutoff).sum()),
            "y_metric": "delta_dg",
            "y_mean": float(landscape["delta_dg"].mean()),
            "y_median": float(landscape["delta_dg"].median()),
            "y_cutoff": config.delta_dg_cutoff,
            "y_n_above_or_equal_cutoff": int((landscape["delta_dg"] >= config.delta_dg_cutoff).sum()),
            "y_n_below_cutoff": int((landscape["delta_dg"] < config.delta_dg_cutoff).sum()),
            "q1_count": int(len(q1)),
            "landscape_axis_pass_count": int(landscape["landscape_axis_pass"].sum()),
            "native_refold_quality_pass_count": int(landscape["native_refold_quality_pass"].sum()),
            "q1_native_refold_quality_pass_count": int(q1["native_refold_quality_pass"].sum()),
            "q1_native_refold_quality_missing_count": int(q1["native_refold_tm_percent"].isna().sum()),
            **_q1_summary_stats(q1),
        },
        {
            "backend": data.spec.key,
            "figure": "native_refold_tm_histogram",
            "n_total": len(data.native_refold),
            "n_plotted": len(data.native_refold),
            "n_hidden_display_outliers": 0,
            "x_metric": "native_refold_tm_percent",
            **{f"x_{key}": value for key, value in _series_stats(data.native_refold["tm_percent"], config.tm_score_cutoff).items()},
            "q1_count": "",
        },
        {
            "backend": data.spec.key,
            "figure": "q1_native_refold_tm_histogram",
            "n_total": len(data.q1_refold),
            "n_plotted": len(data.q1_refold),
            "n_hidden_display_outliers": 0,
            "x_metric": "q1_native_refold_tm_percent",
            **{f"x_{key}": value for key, value in _series_stats(data.q1_refold["tm_percent"], config.tm_score_cutoff).items()},
            "q1_count": int(len(q1)),
        },
    ]


def _remove_stale_files(directory: Path, keep_names: set[str], suffixes: set[str]) -> None:
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file() and path.suffix in suffixes and path.name not in keep_names:
            path.unlink()


def write_backend_outputs(data: BackendData, config: FigureConfig) -> pd.DataFrame:
    base = config.evaluation_root / data.spec.key
    figures = base / "figures"
    tables = base / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for stale in figures.glob("*.png"):
        stale.unlink()
    _remove_stale_files(tables, BACKEND_TABLE_OUTPUTS, {".csv", ".json"})
    plot_landscape(data, config, figures / "landscape_dockq_delta_dg.png")
    plot_refold_hist(data, config, figures / "native_refold_tm_histogram.png", q1_only=False)
    plot_refold_hist(data, config, figures / "q1_native_refold_tm_histogram.png", q1_only=True)
    data.landscape.to_csv(tables / ML_DATASET_CSV, index=False)
    data.native_refold.to_csv(tables / "native_refold_tm_scores.csv", index=False)
    data.q1_refold.to_csv(tables / "q1_native_refold_tm_scores.csv", index=False)
    summary = pd.DataFrame(summary_rows(data, config))
    summary.to_csv(tables / "core_figure_summary.csv", index=False)
    (tables / "core_figure_summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2) + "\n")
    return summary


def write_core_outputs(backends: Sequence[BackendData], config: FigureConfig) -> pd.DataFrame:
    summaries = [write_backend_outputs(data, config) for data in backends]
    comparison = config.evaluation_root / "comparison"
    figures = comparison / "figures"
    tables = comparison / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for stale in figures.glob("*.png"):
        stale.unlink()
    _remove_stale_files(tables, COMPARISON_TABLE_OUTPUTS, {".csv", ".json"})
    plot_comparison_landscape(backends, config, figures / "backend_comparison_landscape_dockq_delta_dg.png")
    plot_comparison_hist(backends, config, figures / "backend_comparison_native_refold_tm_histogram.png", q1_only=False)
    plot_comparison_hist(backends, config, figures / "backend_comparison_q1_native_refold_tm_histogram.png", q1_only=True)
    combined = pd.concat(summaries, ignore_index=True, sort=False)
    combined.to_csv(tables / "core_figure_summary.csv", index=False)
    (tables / "core_figure_summary.json").write_text(json.dumps(combined.to_dict(orient="records"), indent=2) + "\n")
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="core-figures", description="Write compact canonical Jango evaluation figures and summary tables.")
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--esm-landscape", required=True, type=Path)
    parser.add_argument("--esm-native-refold", required=True, type=Path)
    parser.add_argument("--opendde-landscape", required=True, type=Path)
    parser.add_argument("--opendde-native-refold", required=True, type=Path)
    parser.add_argument("--dockq-cutoff", type=float, default=DOCKQ_CUTOFF)
    parser.add_argument("--delta-dg-cutoff", type=float, default=DELTA_DG_CUTOFF)
    parser.add_argument("--binding-perturbation-cutoff", type=float, default=BINDING_PERTURBATION_CUTOFF)
    parser.add_argument("--tm-score-cutoff", type=float, default=TM_SCORE_CUTOFF)
    parser.add_argument("--delta-dg-display-min", type=float, default=DELTA_DG_DISPLAY_MIN)
    parser.add_argument("--delta-dg-display-max", type=float, default=DELTA_DG_DISPLAY_MAX)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FigureConfig(
        evaluation_root=args.evaluation_root,
        dockq_cutoff=args.dockq_cutoff,
        delta_dg_cutoff=args.delta_dg_cutoff,
        binding_perturbation_cutoff=args.binding_perturbation_cutoff,
        tm_score_cutoff=args.tm_score_cutoff,
        delta_dg_display_min=args.delta_dg_display_min,
        delta_dg_display_max=args.delta_dg_display_max,
    )
    specs = [
        BackendSpec("esm", "ESMFold2", ESM_COLOR, ESM_DARK, "o", args.esm_landscape, args.esm_native_refold),
        BackendSpec("opendde", "OpenDDE", OPENDDE_COLOR, OPENDDE_DARK, "^", args.opendde_landscape, args.opendde_native_refold),
    ]
    summary = write_core_outputs([load_backend_data(spec, config) for spec in specs], config)
    print(f"wrote core evaluation figures and tables to {config.evaluation_root}")
    print(f"summary rows: {len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
