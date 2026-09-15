#!/usr/bin/env python3
"""Publish the compact canonical Jango evaluation figure/table package.

This command is intentionally a report-facing publisher. It consumes canonical
analysis outputs, normalizes public labels, and writes only the concise figure
and table bundle used for handoff. Technical provenance can remain in source
work directories, but public figures use the backend labels ESM and OpenDDE.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]

BACKEND_DIRS = {"esmfold2": "esm", "opendde": "opendde"}
BACKEND_LABELS = {"esmfold2": "ESM", "esm": "ESM", "opendde": "OpenDDE"}
BACKEND_COLORS = {
    "esmfold2": ("#f8d4b5", "#c96f1d"),
    "esm": ("#f8d4b5", "#c96f1d"),
    "opendde": ("#c7d8ee", "#315f90"),
}
BACKEND_CMAPS = {"esmfold2": "Oranges", "esm": "Oranges", "opendde": "Blues"}
MODE_LABELS = {"hotspot": "Hotspot", "interface": "Interface"}
TITLE_RED = "#b00020"
GRID_GREY = "#e3e3e3"
DARK_GREY = "#5f5f5f"
DOCKQ_CUTOFF = 0.49
DELTA_DG_CUTOFF = 0.0
DELTA_DG_DISPLAY_MIN = -50.0
DELTA_DG_DISPLAY_MAX = 50.0
REFERENCE_BIAS_DISPLAY_MIN = -60.0
REFERENCE_BIAS_DISPLAY_MAX = 55.0
BIAS_EXCESS_DISPLAY_MIN = -50.0
BIAS_EXCESS_DISPLAY_MAX = 50.0
DECOY_ENERGY_AXIS_LABEL = "decoy vs refold energy (REU)"
DPI = 220
PUBLIC_DROP_COL_FRAGMENTS = (
    "path",
    "scorefile",
    "log",
    "source",
    "pdb",
)


@dataclass(frozen=True)
class PublishedArtifact:
    package: str
    artifact_type: str
    path: Path
    role: str
    source: str
    notes: str = ""


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    full = resolve(path)
    if not full.is_file():
        raise FileNotFoundError(f"missing CSV: {full}")
    return pd.read_csv(full)


def display_backend(backend: str) -> str:
    return BACKEND_LABELS.get(backend, backend)


def public_backend_dir(backend: str) -> str:
    return BACKEND_DIRS.get(backend, backend)


def backend_aliases(backend: str) -> list[str]:
    aliases = [backend, public_backend_dir(backend)]
    if backend == "esmfold2":
        aliases.append("esm")
    if backend == "esm":
        aliases.append("esmfold2")
    return list(dict.fromkeys(aliases))


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def format_p(value: float) -> str:
    if pd.isna(value):
        return "p=NA"
    if value < 0.001:
        return f"p={value:.1e}"
    return f"p={value:.3f}"


def spearman_summary(table: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, int]:
    sub = table[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 3 or sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
        return np.nan, np.nan, int(len(sub))
    rho, p_value = spearmanr(sub[x_col], sub[y_col])
    return float(rho), float(p_value), int(len(sub))


def bool_count(table: pd.DataFrame, col: str) -> int:
    return int(truthy(table[col]).sum()) if col in table.columns else 0


def high_low_bias_effect(table: pd.DataFrame) -> float:
    sub = table[["reference_bias_crystal_minus_refold", "delta_dg_refold"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 9 or sub["reference_bias_crystal_minus_refold"].nunique() < 3:
        return np.nan
    low_cut = sub["reference_bias_crystal_minus_refold"].quantile(1 / 3)
    high_cut = sub["reference_bias_crystal_minus_refold"].quantile(2 / 3)
    if not np.isfinite(low_cut) or not np.isfinite(high_cut) or low_cut >= high_cut:
        return np.nan
    low = sub.loc[sub["reference_bias_crystal_minus_refold"] <= low_cut, "delta_dg_refold"]
    high = sub.loc[sub["reference_bias_crystal_minus_refold"] >= high_cut, "delta_dg_refold"]
    if len(low) == 0 or len(high) == 0:
        return np.nan
    return float(high.median() - low.median())


def reference_bias_stat_lines(table: pd.DataFrame) -> list[str]:
    lines = ["Bias vs decoy energy association: Spearman rho and high-low bias tertile effect"]
    for subset_label, subset in [
        ("All", table),
        ("Q1", table[truthy(table["q1_refold"])] if "q1_refold" in table.columns else table.iloc[0:0]),
    ]:
        parts: list[str] = []
        for backend, label in [("esmfold2", "ESM"), ("opendde", "OpenDDE")]:
            sub = subset[subset["backend"].astype(str).isin(backend_aliases(backend))].copy()
            rho, p_value, _ = spearman_summary(sub, "reference_bias_crystal_minus_refold", "delta_dg_refold")
            effect = high_low_bias_effect(sub)
            parts.append(f"{label}: rho={rho:.2f}, {format_p(p_value)}, high-low={effect:+.1f} REU")
        lines.append(f"{subset_label} decoys: " + "  |  ".join(parts))
    return lines


def setup_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TITLE_RED, fontweight="bold", fontsize=16, pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color=GRID_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_with_footer(fig: plt.Figure, path: Path, footer_frac: float = 0.14) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, footer_frac, 1, 1))
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def add_inside_caption(ax: plt.Axes, text: str, *, y: float = 0.025) -> None:
    ax.text(
        0.02,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#c9c9c9", "alpha": 0.92},
    )


def clean_package(root: Path) -> None:
    for rel in [
        "decoys/figures",
        "decoys/tables",
        "validation/figures",
        "validation/tables",
    ]:
        target = root / rel
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def copy_figure(src: Path, dst: Path) -> None:
    full = resolve(src)
    if not full.is_file():
        raise FileNotFoundError(f"missing figure: {full}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(full, dst)


def write_manifest(path: Path, rows: Sequence[PublishedArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["package", "artifact_type", "path", "role", "source", "status", "notes"])
        for row in rows:
            writer.writerow(
                [
                    row.package,
                    row.artifact_type,
                    row.path.as_posix(),
                    row.role,
                    row.source,
                    "promoted",
                    row.notes,
                ]
            )


def combine_refold_tables(reference_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    decoy_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for mode in ["hotspot", "interface"]:
        for backend in ["esmfold2", "opendde"]:
            tables = reference_root / "reference_refold" / mode / backend / "tables"
            decoy = read_csv(tables / "decoy_ranked_ml_dataset.csv")
            decoy["reference_mode"] = "refold"
            decoy["redesign_mode"] = mode
            decoy["backend"] = backend
            decoy["backend_label"] = display_backend(backend)
            decoy["dataset_key"] = f"{public_backend_dir(backend)}_{mode}"
            decoy_frames.append(decoy)
            summary = read_csv(tables / "summary_stats.csv")
            summary["dataset_key"] = f"{public_backend_dir(backend)}_{mode}"
            summary["backend"] = backend
            summary["backend_label"] = display_backend(backend)
            summary["redesign_mode"] = mode
            summary_frames.append(summary)
    decoys = pd.concat(decoy_frames, ignore_index=True, sort=False)
    drop_cols = [
        col
        for col in decoys.columns
        if any(fragment in col.lower() for fragment in PUBLIC_DROP_COL_FRAGMENTS)
    ]
    decoys = decoys.drop(columns=drop_cols, errors="ignore")
    summaries = pd.concat(summary_frames, ignore_index=True, sort=False).drop(
        columns=["source_paths"], errors="ignore"
    )
    return decoys, summaries


def combine_native_refold(reference_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    stats: list[pd.DataFrame] = []
    for reference in ["relaxed", "crystal"]:
        for backend in ["esmfold2", "opendde"]:
            tables = reference_root / "native_refold_quality" / f"refold_vs_{reference}" / backend / "tables"
            native = read_csv(tables / "native_refold_tm_scores.csv")
            native["reference_mode"] = reference
            native["backend"] = backend
            native["backend_label"] = display_backend(backend)
            frames.append(native)
            summary = read_csv(tables / "summary_stats.csv")
            summary["reference_mode"] = reference
            summary["backend"] = backend
            summary["backend_label"] = display_backend(backend)
            stats.append(summary)
    native_stats = pd.concat(stats, ignore_index=True, sort=False).drop(columns=["source_paths"], errors="ignore")
    return pd.concat(frames, ignore_index=True, sort=False), native_stats


def publish_reference_figures(root: Path, reference_root: Path) -> list[PublishedArtifact]:
    rows: list[PublishedArtifact] = []
    for mode in ["hotspot", "interface"]:
        for backend in ["esmfold2", "opendde"]:
            public = public_backend_dir(backend)
            src = reference_root / "reference_refold" / mode / backend / "figures" / "landscape_dockq_delta_dg.png"
            dst = root / "decoys" / "figures" / mode / public / "decoy_vs_refold_dockq_delta_dg.png"
            copy_figure(src, dst)
            rows.append(
                PublishedArtifact(
                    "decoys",
                    "figure",
                    dst.relative_to(root / "decoys"),
                    f"{MODE_LABELS[mode]} {display_backend(backend)} decoy vs refold DockQ/delta-dG landscape",
                    str(src),
                )
            )
        src = reference_root / "reference_refold" / mode / "comparison" / "figures" / "backend_comparison_landscape_dockq_delta_dg.png"
        dst = root / "decoys" / "figures" / mode / "comparison" / "backend_comparison_decoy_vs_refold.png"
        copy_figure(src, dst)
        rows.append(
            PublishedArtifact(
                "decoys",
                "figure",
                dst.relative_to(root / "decoys"),
                f"{MODE_LABELS[mode]} backend comparison decoy vs refold landscape",
                str(src),
            )
        )

    for backend in ["esmfold2", "opendde"]:
        public = public_backend_dir(backend)
        src = (
            reference_root
            / "reference_refold"
            / "redesign_mode_comparison"
            / backend
            / "figures"
            / "redesign_mode_landscape_dockq_delta_dg.png"
        )
        dst = root / "decoys" / "figures" / "redesign_mode" / public / "redesign_mode_decoy_vs_refold.png"
        copy_figure(src, dst)
        rows.append(
            PublishedArtifact(
                "decoys",
                "figure",
                dst.relative_to(root / "decoys"),
                f"{display_backend(backend)} hotspot vs interface decoy landscape",
                str(src),
            )
        )

    for reference in ["relaxed", "crystal"]:
        for backend in ["esmfold2", "opendde"]:
            public = public_backend_dir(backend)
            src = (
                reference_root
                / "native_refold_quality"
                / f"refold_vs_{reference}"
                / backend
                / "figures"
                / "native_refold_tm_histogram.png"
            )
            dst = root / "decoys" / "figures" / "native_refold" / public / f"refold_vs_{reference}_tm_histogram.png"
            copy_figure(src, dst)
            rows.append(
                PublishedArtifact(
                    "decoys",
                    "figure",
                    dst.relative_to(root / "decoys"),
                    f"{display_backend(backend)} native-refold quality vs {reference}",
                    str(src),
                )
            )
        src = (
            reference_root
            / "native_refold_quality"
            / f"refold_vs_{reference}"
            / "comparison"
            / "figures"
            / "backend_comparison_native_refold_tm_histogram.png"
        )
        dst = root / "decoys" / "figures" / "native_refold" / "comparison" / f"refold_vs_{reference}_backend_comparison.png"
        copy_figure(src, dst)
        rows.append(
            PublishedArtifact(
                "decoys",
                "figure",
                dst.relative_to(root / "decoys"),
                f"Backend comparison native-refold quality vs {reference}",
                str(src),
            )
        )
    return rows


def plot_mutation_count_map(decoys: pd.DataFrame, backend: str, out: Path) -> None:
    table = decoys[
        decoys["redesign_mode"].astype(str).eq("hotspot")
        & decoys["backend"].astype(str).isin(backend_aliases(backend))
    ].copy()
    table["dockq"] = pd.to_numeric(table["dockq"], errors="coerce")
    table["delta_dg"] = pd.to_numeric(table["delta_dg"], errors="coerce")
    table["n_residues_mutated"] = pd.to_numeric(table["n_residues_mutated"], errors="coerce")
    stats_table = table.dropna(subset=["dockq", "delta_dg", "n_residues_mutated"])
    table = stats_table[(stats_table["delta_dg"] >= DELTA_DG_DISPLAY_MIN) & (stats_table["delta_dg"] <= DELTA_DG_DISPLAY_MAX)]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    hb = ax.hexbin(
        table["dockq"],
        table["delta_dg"],
        C=table["n_residues_mutated"],
        reduce_C_function=np.nanmean,
        gridsize=34,
        mincnt=1,
        cmap=BACKEND_CMAPS.get(backend, "viridis"),
        linewidths=0.0,
    )
    ax.scatter(table["dockq"], table["delta_dg"], s=6, color="#444444", alpha=0.18, edgecolors="none")
    ax.axvline(DOCKQ_CUTOFF, color=DARK_GREY, linewidth=0.9)
    ax.axhline(DELTA_DG_CUTOFF, color=DARK_GREY, linewidth=0.9)
    setup_axis(ax, f"Hotspot {display_backend(backend)} Mutation Count Map", "DockQ score", DECOY_ENERGY_AXIS_LABEL)
    ax.set_ylim(DELTA_DG_DISPLAY_MIN, DELTA_DG_DISPLAY_MAX)
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Mean mutated antigen residues", fontsize=11)
    rho_dockq, p_dockq, n_dockq = spearman_summary(stats_table, "n_residues_mutated", "dockq")
    rho_dg, p_dg, n_dg = spearman_summary(stats_table, "n_residues_mutated", "delta_dg")
    q1 = stats_table[truthy(stats_table["q1_tm90"])] if "q1_tm90" in stats_table.columns else stats_table.iloc[0:0]
    background = stats_table[~truthy(stats_table["q1_tm90"])] if "q1_tm90" in stats_table.columns else stats_table.iloc[0:0]
    q1_median = q1["n_residues_mutated"].median() if len(q1) else np.nan
    bg_median = background["n_residues_mutated"].median() if len(background) else np.nan
    footer = (
        f"Mutation count vs DockQ: Spearman rho={rho_dockq:.2f}, {format_p(p_dockq)}, n={n_dockq}\n"
        f"Mutation count vs decoy energy: rho={rho_dg:.2f}, {format_p(p_dg)}, n={n_dg}\n"
        f"Median mutations: Q1+TM90={q1_median:.0f}, background={bg_median:.0f}."
    )
    add_inside_caption(ax, footer)
    save(fig, out)


def plot_chemistry_enrichment(enrichment: pd.DataFrame, backend: str, out: Path) -> None:
    table = enrichment[
        enrichment["redesign_mode"].astype(str).eq("hotspot")
        & enrichment["backend"].astype(str).isin(backend_aliases(backend))
    ].copy()
    table["log2_odds_ratio"] = pd.to_numeric(table["log2_odds_ratio"], errors="coerce")
    table["q_value_bh"] = pd.to_numeric(table.get("q_value_bh", np.nan), errors="coerce")
    keep_features = {"blosum_class", "class_transition", "charge_transition", "polarity_transition"}
    table = table[table["feature"].isin(keep_features)].dropna(subset=["log2_odds_ratio"])
    if table.empty:
        raise ValueError(f"no chemistry enrichment rows available for {backend}")
    table["abs_effect"] = table["log2_odds_ratio"].abs()
    table = table.sort_values(["q_value_bh", "abs_effect"], ascending=[True, False]).head(12)
    table = table.sort_values("log2_odds_ratio", ascending=False)
    table["label"] = table["feature"].astype(str).str.replace("_", " ", regex=False) + ": " + table["category"].astype(str)
    light, dark = BACKEND_COLORS.get(backend, ("#cccccc", "#555555"))
    colors = [dark if value >= 0 else light for value in table["log2_odds_ratio"]]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    ax.barh(np.arange(len(table)), table["log2_odds_ratio"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table["label"], fontsize=9)
    ax.axvline(0, color=DARK_GREY, linewidth=0.9)
    setup_axis(
        ax,
        f"Hotspot {display_backend(backend)} Mutation Chemistry Enrichment",
        "log2 odds ratio vs background (BLOSUM62 / chemistry features)",
        "",
    )
    ax.invert_yaxis()
    save(fig, out)


def publish_mutation_profile(root: Path, mutation_root: Path) -> list[PublishedArtifact]:
    source_tables = mutation_root / "tables"
    decoys = read_csv(source_tables / "mutation_profile_per_decoy.csv")
    enrichment = read_csv(source_tables / "mutation_enrichment_summary.csv")
    summary = read_csv(source_tables / "mutation_profile_summary_stats.csv")
    tests = read_csv(source_tables / "mutation_decoy_feature_tests.csv")

    table_dir = root / "validation" / "tables"
    decoys.to_csv(table_dir / "mutation_landscape_decoys.csv", index=False)
    enrichment.to_csv(table_dir / "mutation_chemistry_enrichment.csv", index=False)
    tests.to_csv(table_dir / "mutation_landscape_correlation_summary.csv", index=False)
    summary.to_csv(table_dir / "mutation_profile_summary_stats.csv", index=False)

    rows: list[PublishedArtifact] = [
        PublishedArtifact(
            "validation",
            "table",
            Path("tables/mutation_landscape_decoys.csv"),
            "Per-decoy mutation counts and chemistry features for mutation landscape plots",
            str(source_tables / "mutation_profile_per_decoy.csv"),
        ),
        PublishedArtifact(
            "validation",
            "table",
            Path("tables/mutation_chemistry_enrichment.csv"),
            "Mutation chemistry enrichment table using BLOSUM62-derived classes",
            str(source_tables / "mutation_enrichment_summary.csv"),
        ),
        PublishedArtifact(
            "validation",
            "table",
            Path("tables/mutation_landscape_correlation_summary.csv"),
            "Support statistics for mutation landscape features",
            str(source_tables / "mutation_decoy_feature_tests.csv"),
        ),
    ]
    for backend in ["esmfold2", "opendde"]:
        public = public_backend_dir(backend)
        out = root / "validation" / "figures" / "mutation_profile" / public / "hotspot_mutation_count_map.png"
        plot_mutation_count_map(decoys, backend, out)
        rows.append(
            PublishedArtifact(
                "validation",
                "figure",
                out.relative_to(root / "validation"),
                f"Hotspot {display_backend(backend)} mutation count map",
                "tables/mutation_landscape_decoys.csv",
            )
        )
        out = root / "validation" / "figures" / "mutation_profile" / public / "hotspot_mutation_chemistry_enrichment.png"
        plot_chemistry_enrichment(enrichment, backend, out)
        rows.append(
            PublishedArtifact(
                "validation",
                "figure",
                out.relative_to(root / "validation"),
                f"Hotspot {display_backend(backend)} mutation chemistry enrichment",
                "tables/mutation_chemistry_enrichment.csv",
                "BLOSUM features use BLOSUM62.",
            )
        )
    return rows


def plot_bias_excess(table: pd.DataFrame, backend: str, out: Path) -> None:
    sub = table[table["backend"].astype(str).isin(backend_aliases(backend))].copy()
    for col in ["dockq_refold", "delta_dg_refold", "bias_excess_delta_dg"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=["dockq_refold", "delta_dg_refold", "bias_excess_delta_dg"])
    shown = sub[sub["delta_dg_refold"].between(BIAS_EXCESS_DISPLAY_MIN, BIAS_EXCESS_DISPLAY_MAX)].copy()
    hidden = len(sub) - len(shown)
    pass0 = shown["bias_excess_delta_dg"] > 0
    pass2 = truthy(shown["bias_excess_q1_margin2"]) if "bias_excess_q1_margin2" in shown.columns else pd.Series(False, index=shown.index)
    light, dark = BACKEND_COLORS.get(backend, ("#cccccc", "#555555"))
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    ax.scatter(
        shown.loc[~pass0, "dockq_refold"],
        shown.loc[~pass0, "delta_dg_refold"],
        s=18,
        color="#d9d9d9",
        alpha=0.58,
        edgecolors="none",
        label="bias excess <= 0",
    )
    positives = shown[pass0]
    sc = ax.scatter(
        positives["dockq_refold"],
        positives["delta_dg_refold"],
        c=positives["bias_excess_delta_dg"],
        s=22,
        cmap=BACKEND_CMAPS.get(backend, "viridis"),
        vmin=0,
        alpha=0.88,
        edgecolors="none",
        label="bias excess > 0",
    )
    strict = shown[pass2]
    ax.scatter(
        strict["dockq_refold"],
        strict["delta_dg_refold"],
        s=54,
        facecolors="none",
        edgecolors="#4d4d4d",
        linewidths=0.8,
        label="bias-excess Q1 >= 2",
    )
    ax.axvline(DOCKQ_CUTOFF, color=DARK_GREY, linewidth=0.9, linestyle="--")
    ax.axhline(DELTA_DG_CUTOFF, color=DARK_GREY, linewidth=0.9, linestyle="--")
    setup_axis(ax, f"Hotspot {display_backend(backend)} Bias-Excess Landscape", "DockQ vs refold", DECOY_ENERGY_AXIS_LABEL)
    ax.set_ylim(BIAS_EXCESS_DISPLAY_MIN, BIAS_EXCESS_DISPLAY_MAX)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Bias excess energy (REU)", fontsize=11)
    if hidden:
        ax.text(
            0.98,
            0.12,
            f"Display trimmed: {hidden} outliers hidden",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9d9d9", markeredgecolor="none", markersize=7, label="bias excess <= 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=dark, markeredgecolor="none", markersize=7, label="bias excess > 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#4d4d4d", markersize=8, label="bias-excess Q1 >= 2"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, -0.13), frameon=False, ncol=3, fontsize=10)
    footer = (
        "bias excess = decoy vs refold energy - expected energy from matched reference-bias bin\n"
        f"shown n={len(shown)}, >0 n={int(pass0.sum())}, Q1>=2 n={int(pass2.sum())}."
    )
    add_inside_caption(ax, footer)
    save(fig, out)


def plot_reference_bias_panel(table: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    table = table.copy()
    for col in [
        "reference_bias_crystal_minus_refold",
        "reference_bias_relaxed_minus_refold",
        "reference_bias_relaxed_minus_crystal",
        "delta_dg_refold",
    ]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    q1 = table[truthy(table["q1_refold"])].copy() if "q1_refold" in table.columns else table.copy()
    colors = {"ESM": BACKEND_COLORS["esm"][1], "OpenDDE": BACKEND_COLORS["opendde"][1], "Reference": "#9a9a9a"}
    bias_specs = [
        ("Crystal -\nESM refold", "reference_bias_crystal_minus_refold", "ESM", q1[q1["backend"].astype(str).isin(backend_aliases("esmfold2"))]),
        ("Crystal -\nOpenDDE refold", "reference_bias_crystal_minus_refold", "OpenDDE", q1[q1["backend"].astype(str).isin(backend_aliases("opendde"))]),
        ("Relaxed -\nESM refold", "reference_bias_relaxed_minus_refold", "ESM", q1[q1["backend"].astype(str).isin(backend_aliases("esmfold2"))]),
        ("Relaxed -\nOpenDDE refold", "reference_bias_relaxed_minus_refold", "OpenDDE", q1[q1["backend"].astype(str).isin(backend_aliases("opendde"))]),
        ("Relaxed -\ncrystal", "reference_bias_relaxed_minus_crystal", "Reference", q1.drop_duplicates("structure_id")),
    ]
    long_rows: list[dict[str, object]] = []
    hidden = 0
    for label, col, group, source in bias_specs:
        vals = source[col].dropna()
        hidden += int((~vals.between(REFERENCE_BIAS_DISPLAY_MIN, REFERENCE_BIAS_DISPLAY_MAX)).sum())
        vals = vals[vals.between(REFERENCE_BIAS_DISPLAY_MIN, REFERENCE_BIAS_DISPLAY_MAX)]
        long_rows.extend({"bias": label, "value": value, "group": group} for value in vals)
    long = pd.DataFrame(long_rows)

    filters = [
        ("Original Q1", "q1_refold"),
        ("Q1 + TM>=90", "q1_tm90"),
        ("Bias-excess Q1 >=0", "bias_excess_q1_margin0"),
        ("Bias-excess Q1 >=2", "bias_excess_q1_margin2"),
        ("Bias-excess Q1 >=5", "bias_excess_q1_margin5"),
        ("Reference-robust Q1", "reference_robust_q1"),
        ("Reference-robust Q1 + TM>=90", "reference_robust_q1_tm90"),
    ]
    funnel_rows: list[dict[str, object]] = []
    for backend, label in [("esmfold2", "ESM"), ("opendde", "OpenDDE")]:
        sub = table[table["backend"].astype(str).isin(backend_aliases(backend))]
        for filter_label, col in filters:
            funnel_rows.append({"backend": label, "filter": filter_label, "n": bool_count(sub, col)})
    funnel = pd.DataFrame(funnel_rows)

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.4), gridspec_kw={"width_ratios": [1.05, 1.15]})
    ax = axes[0]
    labels = [label for label, _, _, _ in bias_specs]
    data = [long.loc[long["bias"].eq(label), "value"].to_numpy() for label in labels]
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, (_, _, group, _) in zip(box["boxes"], bias_specs):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.58)
        patch.set_edgecolor("#444444")
    rng = np.random.default_rng(7)
    for idx, (vals, (_, _, group, _)) in enumerate(zip(data, bias_specs), start=1):
        if len(vals):
            ax.scatter(
                np.full(len(vals), idx) + rng.normal(0, 0.035, len(vals)),
                vals,
                s=7,
                alpha=0.22,
                color=colors[group],
                edgecolors="none",
            )
    ax.axhline(0, color=DARK_GREY, linewidth=0.9, linestyle="--")
    setup_axis(ax, "Q1 Reference Bias", "Reference contrast", "Reference energy shift (REU; left - right)")
    ax.set_ylim(REFERENCE_BIAS_DISPLAY_MIN, REFERENCE_BIAS_DISPLAY_MAX)
    ax.tick_params(axis="x", labelrotation=0)
    if hidden:
        ax.text(0.98, 0.04, f"{hidden} display outliers hidden", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#666666")

    ax = axes[1]
    y = np.arange(len(filters))
    height = 0.36
    for offset, label, color in [(-height / 2, "ESM", BACKEND_COLORS["esm"][1]), (height / 2, "OpenDDE", BACKEND_COLORS["opendde"][1])]:
        vals = [int(funnel[(funnel["backend"].eq(label)) & (funnel["filter"].eq(f[0]))]["n"].iloc[0]) for f in filters]
        bars = ax.barh(y + offset, vals, height=height, color=color, alpha=0.86, edgecolor="white", label=label)
        for bar, value in zip(bars, vals):
            ax.text(value + 4, bar.get_y() + bar.get_height() / 2, str(value), va="center", ha="left", fontsize=8.5, color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels([f[0] for f in filters], fontsize=9)
    ax.invert_yaxis()
    setup_axis(ax, "Bias-Aware Q1 Shrinkage", "Number of hotspot decoys", "")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    stat_lines = reference_bias_stat_lines(table)
    fig.text(0.06, 0.014, "\n".join(stat_lines), ha="left", va="bottom", fontsize=9.0, color="#333333",
             bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c9c9c9", "alpha": 0.95})
    save_with_footer(fig, out, footer_frac=0.135)


def publish_reference_bias(
    root: Path,
    table: pd.DataFrame,
    summary: pd.DataFrame,
    bins: pd.DataFrame | None,
    *,
    bias_table_source: str,
    bias_summary_source: str,
    bias_bins_source: str,
) -> list[PublishedArtifact]:
    table_dir = root / "validation" / "tables"
    table.to_csv(table_dir / "reference_bias_decoys.csv", index=False)
    summary.to_csv(table_dir / "reference_bias_summary.csv", index=False)
    if bins is not None:
        bins.to_csv(table_dir / "reference_bias_bins.csv", index=False)

    rows: list[PublishedArtifact] = [
        PublishedArtifact(
            "validation",
            "table",
            Path("tables/reference_bias_decoys.csv"),
            "Per-decoy reference-bias and bias-excess fields",
            bias_table_source,
        ),
        PublishedArtifact(
            "validation",
            "table",
            Path("tables/reference_bias_summary.csv"),
            "Reference-bias summary counts",
            bias_summary_source,
        ),
    ]
    if bins is not None:
        rows.append(
            PublishedArtifact(
                "validation",
                "table",
                Path("tables/reference_bias_bins.csv"),
                "Expected delta-dG by reference-bias bin",
                bias_bins_source,
            )
        )
    out = root / "validation" / "figures" / "reference_bias" / "hotspot_q1_reference_bias_boxplot_funnel.png"
    plot_reference_bias_panel(table, summary, out)
    rows.append(
        PublishedArtifact(
            "validation",
            "figure",
            out.relative_to(root / "validation"),
            "Hotspot reference-bias boxplot and Q1 shrinkage funnel",
            "tables/reference_bias_decoys.csv",
        )
    )
    for backend in ["esmfold2", "opendde"]:
        public = public_backend_dir(backend)
        out = root / "validation" / "figures" / "reference_bias" / public / "bias_excess_landscape.png"
        plot_bias_excess(table, backend, out)
        rows.append(
            PublishedArtifact(
                "validation",
                "figure",
                out.relative_to(root / "validation"),
                f"Hotspot {display_backend(backend)} bias-excess landscape",
                "tables/reference_bias_decoys.csv",
            )
        )
    return rows


def write_readmes(root: Path) -> None:
    (root / "README.md").write_text(
        "\n".join(
            [
                "# Jango Evaluation Outputs",
                "",
                "This directory contains the compact canonical report-facing Jango evaluation package.",
                "Figures are split into decoy results and validation analyses.",
                "Public labels use ESM and OpenDDE; technical runtime provenance remains in source work directories.",
                "",
                "Regeneration flow:",
                "1. Run `jango reference-benchmark-figures` to produce a temporary reference benchmark source package.",
                "2. Run `jango mutation-profile` against that source package to produce temporary mutation-profile source tables.",
                "3. Run `jango evaluation-package --clean-output` to publish this compact package.",
                "",
                "Only the compact package is kept here; intermediate source packages are regeneratable and may be removed after publishing.",
                "",
            ]
        )
    )
    (root / "decoys" / "README.md").write_text(
        "\n".join(
            [
                "# Jango Decoy Figures And Tables",
                "",
                "Decoy figures summarize DockQ vs delta dG landscapes for hotspot and interface redesigns.",
                "Q1 is DockQ >= 0.49 and delta dG >= 0.",
                "",
                "Primary tables:",
                "- `tables/decoy_landscape_ml_dataset.csv`",
                "- `tables/native_refold_quality.csv`",
                "- `tables/decoy_figure_stats.csv`",
                "",
            ]
        )
    )
    (root / "validation" / "README.md").write_text(
        "\n".join(
            [
                "# Jango Validation Figures And Tables",
                "",
                "Validation figures cover mutation chemistry/count patterns and reference-bias sensitivity.",
                "",
                "Primary tables:",
                "- `tables/mutation_landscape_decoys.csv`",
                "- `tables/mutation_chemistry_enrichment.csv`",
                "- `tables/mutation_landscape_correlation_summary.csv`",
                "- `tables/reference_bias_decoys.csv`",
                "- `tables/reference_bias_summary.csv`",
                "",
            ]
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish the compact canonical Jango evaluation package.")
    parser.add_argument("--output-root", type=Path, default=Path("data/outputs/evaluation"))
    parser.add_argument("--reference-root", type=Path, default=Path("data/outputs/evaluation/reference_benchmark"))
    parser.add_argument("--mutation-root", type=Path, default=Path("data/outputs/evaluation/mutation_profile"))
    parser.add_argument("--bias-table", type=Path, default=Path("data/outputs/evaluation/validation/tables/reference_bias_decoys.csv"))
    parser.add_argument("--bias-summary", type=Path, default=Path("data/outputs/evaluation/validation/tables/reference_bias_summary.csv"))
    parser.add_argument("--bias-bins", type=Path, default=Path("data/outputs/evaluation/validation/tables/reference_bias_bins.csv"))
    parser.add_argument("--clean-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = resolve(args.output_root)
    reference_root = resolve(args.reference_root)
    mutation_root = resolve(args.mutation_root)
    bias_table = read_csv(args.bias_table)
    bias_summary = read_csv(args.bias_summary) if args.bias_summary else pd.DataFrame()
    bias_bins = read_csv(args.bias_bins) if args.bias_bins and resolve(args.bias_bins).is_file() else None
    if args.clean_output:
        clean_package(root)
    else:
        for rel in ["decoys/figures", "decoys/tables", "validation/figures", "validation/tables"]:
            (root / rel).mkdir(parents=True, exist_ok=True)

    decoy_rows = publish_reference_figures(root, reference_root)
    decoys, decoy_stats = combine_refold_tables(reference_root)
    native, native_stats = combine_native_refold(reference_root)
    decoys.to_csv(root / "decoys" / "tables" / "decoy_landscape_ml_dataset.csv", index=False)
    native.to_csv(root / "decoys" / "tables" / "native_refold_quality.csv", index=False)
    pd.concat([decoy_stats, native_stats], ignore_index=True, sort=False).to_csv(
        root / "decoys" / "tables" / "decoy_figure_stats.csv", index=False
    )
    decoy_rows.extend(
        [
            PublishedArtifact(
                "decoys",
                "table",
                Path("tables/decoy_landscape_ml_dataset.csv"),
                "All decoy rows used for decoy-vs-refold landscape figures",
                str(reference_root),
            ),
            PublishedArtifact(
                "decoys",
                "table",
                Path("tables/native_refold_quality.csv"),
                "Native-refold TM-score rows used for refold quality figures",
                str(reference_root),
            ),
            PublishedArtifact(
                "decoys",
                "table",
                Path("tables/decoy_figure_stats.csv"),
                "Summary statistics for decoy and native-refold figures",
                str(reference_root),
            ),
        ]
    )

    validation_rows = []
    validation_rows.extend(publish_mutation_profile(root, mutation_root))
    validation_rows.extend(
        publish_reference_bias(
            root,
            bias_table,
            bias_summary,
            bias_bins,
            bias_table_source=str(args.bias_table),
            bias_summary_source=str(args.bias_summary) if args.bias_summary else "derived",
            bias_bins_source=str(args.bias_bins) if args.bias_bins else "derived",
        )
    )
    write_manifest(root / "decoys" / "manifest.csv", decoy_rows)
    write_manifest(root / "validation" / "manifest.csv", validation_rows)
    write_readmes(root)
    print(f"published compact evaluation package to {root}")
    print(f"decoy artifacts: {len(decoy_rows)}")
    print(f"validation artifacts: {len(validation_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
