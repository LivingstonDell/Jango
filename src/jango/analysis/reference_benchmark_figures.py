#!/usr/bin/env python3
"""Canonical reference-benchmark DockQ/delta-dG evaluation figures.

The command writes a compact, reference-aware figure bundle. It intentionally
keeps the high-high visual grammar while making the reference state explicit:
crystal, relaxed, or refold. Inputs may be prebuilt landscape tables or a
DockQ/Rosetta-source pair.
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
DECOY_ENERGY_AXIS_LABEL = "decoy vs refold energy (REU)"
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
REFERENCE_LABELS = {"crystal": "Crystal Reference", "relaxed": "Relaxed Reference", "refold": "Refold Reference"}
BACKEND_MARKERS = {"esm": "o", "esmfold2": "o", "opendde": "^"}
VALID_DOCKQ_STATUSES = {"ok"}
VALID_ANALYSIS_STATUSES = {"ok", "skipped_existing_valid"}
MISSING_INPUT_COLUMNS = ["reference_mode", "redesign_mode", "backend_input", "reason"]
PLOT_INVENTORY_COLUMNS = ["figure", "reference_mode", "redesign_mode", "backend", "figure_type", "source_kind", "source_paths"]


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
    reference_mode: str
    landscape: pd.DataFrame
    native_refold: pd.DataFrame
    high_high_refold: pd.DataFrame
    source_kind: str
    source_paths: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{MODE_LABELS[self.mode]} {BACKEND_LABELS.get(self.backend, self.backend)}"

    @property
    def reference_label(self) -> str:
        return REFERENCE_LABELS.get(self.reference_mode, self.reference_mode.title())

    @property
    def title(self) -> str:
        return f"{self.label} Decoy vs {self.reference_label}"

    @property
    def colors(self) -> tuple[str, str]:
        if (self.mode, self.backend) in MODE_BACKEND_COLORS:
            return MODE_BACKEND_COLORS[(self.mode, self.backend)]
        if self.backend.startswith("esm"):
            return MODE_BACKEND_COLORS[(self.mode, "esm")]
        if self.backend.startswith("opendde"):
            return MODE_BACKEND_COLORS[(self.mode, "opendde")]
        raise KeyError(f"no color configured for {self.mode}/{self.backend}")


@dataclass(frozen=True)
class RefoldQualityDataset:
    backend: str
    reference_mode: str
    native_refold: pd.DataFrame
    source_path: str

    @property
    def label(self) -> str:
        return BACKEND_LABELS.get(self.backend, self.backend)

    @property
    def colors(self) -> tuple[str, str]:
        if self.backend.startswith("esm"):
            return ("#f8d4b5", "#c96f1d")
        if self.backend.startswith("opendde"):
            return ("#c7d8ee", "#315f90")
        raise KeyError(f"no color configured for {self.backend}")


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


def filter_status(table: pd.DataFrame, column: str, valid_statuses: set[str]) -> pd.DataFrame:
    if column not in table.columns:
        return table.copy()
    status = table[column].astype(str).str.strip().str.lower()
    return table[status.isin(valid_statuses)].copy()
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


def optional_numeric(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series | None:
    for column in candidates:
        if column in df.columns:
            data = df[column]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            return pd.to_numeric(data, errors="coerce")
    return None


def string_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            data = df[column]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            return data.astype(str)
    raise ValueError(f"missing {label} column; tried {', '.join(candidates)}")


def top1_per_structure(df: pd.DataFrame) -> pd.DataFrame:
    if "structure_id" not in df.columns:
        raise ValueError("landscape table requires structure_id")
    table = df.copy()
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "selected_decoy" in table.columns:
        table["_rank_selected"] = table["selected_decoy"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
        sort_columns.append("_rank_selected")
        ascending.append(False)
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
        columns=[c for c in ["_rank_selected", "_rank_tm", "_rank_rmsd", "_rank_design"] if c in table.columns]
    )


def load_native_refold(path: str) -> pd.DataFrame:
    table = read_csv(path)
    if "similarity_status" in table.columns:
        table = table[table["similarity_status"].astype(str).str.lower().eq("ok")].copy()
    structure_id = string_column(table, ["structure_id", "native_id", "case_id", "pdb_id"], "native refold structure id")
    tm = numeric(
        table,
        [
            "tm_percent",
            "native_refold_tm_percent",
            "percent_structural_similarity",
            "antigen_ca_tm_score",
            "tm_score",
            "tm_score_antigen",
        ],
        "native-refold TM-score",
    )
    if tm.max(skipna=True) <= 1.5:
        tm = tm * 100.0
    out = pd.DataFrame({"structure_id": structure_id, "native_refold_tm_percent": tm})
    return out.dropna(subset=["structure_id", "native_refold_tm_percent"]).drop_duplicates("structure_id").copy()


def normalize_landscape(
    table: pd.DataFrame,
    *,
    mode: str,
    backend: str,
    reference_mode: str,
    native_refold: pd.DataFrame,
    config: Config,
    source_kind: str,
    source_paths: Sequence[str],
) -> Dataset:
    out = top1_per_structure(table.copy())
    out["structure_id"] = string_column(out, ["structure_id", "native_id", "case_id", "pdb_id"], "structure id")
    if "design_id" not in out.columns:
        out["design_id"] = out["structure_id"]
    out["design_id"] = out["design_id"].astype(str)
    out["redesign_mode"] = mode
    out["backend"] = backend
    out["fold_backend"] = backend
    out["reference_mode"] = reference_mode
    out["dockq"] = numeric(out, ["dockq", "dockq_score", "DockQ"], "DockQ")
    out["delta_dg"] = numeric(
        out,
        ["delta_dg", "delta_vs_native_dG_separated", "delta_delta_g", "delta_dG", "ddg", "binding_delta_delta_g"],
        "delta dG",
    )
    structure = optional_numeric(out, ["structure_preservation"])
    out["structure_preservation"] = structure if structure is not None else out["dockq"]
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
    out = out.drop(columns=["_high_sort", "_score_sort", "_dockq_sort"])
    high_refold = native_refold[native_refold["structure_id"].isin(set(out.loc[out["high_high"], "structure_id"]))].copy()
    return Dataset(
        mode=mode,
        backend=backend,
        reference_mode=reference_mode,
        landscape=out,
        native_refold=native_refold,
        high_high_refold=high_refold,
        source_kind=source_kind,
        source_paths=tuple(source_paths),
    )


def build_from_landscape(
    *,
    path: str,
    mode: str,
    backend: str,
    reference_mode: str,
    native_refold: pd.DataFrame,
    config: Config,
) -> Dataset:
    return normalize_landscape(
        read_csv(path),
        mode=mode,
        backend=backend,
        reference_mode=reference_mode,
        native_refold=native_refold,
        config=config,
        source_kind="landscape",
        source_paths=[path],
    )


def build_from_source_tables(
    *,
    dockq_path: str,
    decoy_rosetta_path: str,
    reference_rosetta_path: str,
    mode: str,
    backend: str,
    reference_mode: str,
    native_refold: pd.DataFrame,
    config: Config,
) -> Dataset:
    dockq = read_csv(dockq_path)
    rosetta = read_csv(decoy_rosetta_path)
    reference = read_csv(reference_rosetta_path)
    dockq = filter_status(dockq, "dockq_status", VALID_DOCKQ_STATUSES)
    rosetta = filter_status(rosetta, "rosetta_status", VALID_ANALYSIS_STATUSES)
    reference = filter_status(reference, "interface_analyzer_status", VALID_ANALYSIS_STATUSES)

    table = rosetta.copy()
    table["structure_id"] = string_column(table, ["structure_id", "native_id", "case_id", "pdb_id"], "structure id")
    table["design_id"] = string_column(table, ["design_id", "description", "structure_id"], "design id")
    table = table.merge(
        dockq[["structure_id", "design_id", *[c for c in ["dockq", "dockq_fnat", "dockq_irmsd", "dockq_lrmsd", "dockq_quality"] if c in dockq.columns]]],
        on=["structure_id", "design_id"],
        how="inner",
    )
    ref = reference.copy()
    ref["structure_id"] = string_column(ref, ["structure_id", "native_id", "case_id", "pdb_id"], "reference structure id")
    ref["reference_dG_separated"] = numeric(ref, ["dG_separated"], "reference dG")
    table = table.merge(ref[["structure_id", "reference_dG_separated"]], on="structure_id", how="left")
    table["delta_dg"] = numeric(table, ["dG_separated"], "decoy dG") - table["reference_dG_separated"]
    table["delta_vs_native_dG_separated"] = table["delta_dg"]
    table["native_dG_separated"] = table["reference_dG_separated"]
    return normalize_landscape(
        table,
        mode=mode,
        backend=backend,
        reference_mode=reference_mode,
        native_refold=native_refold,
        config=config,
        source_kind="source_tables",
        source_paths=[dockq_path, decoy_rosetta_path, reference_rosetta_path],
    )


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
    ax.scatter(
        table.loc[~high_refold, "dockq"],
        table.loc[~high_refold, "delta_dg"],
        s=POINT_SIZE,
        color=light,
        alpha=0.76,
        edgecolors="none",
        marker=marker_for(data.backend),
        label=f"{data.label} <90% native refold",
    )
    ax.scatter(
        table.loc[high_refold, "dockq"],
        table.loc[high_refold, "delta_dg"],
        s=POINT_SIZE,
        color=dark,
        alpha=0.88,
        edgecolors="none",
        marker=marker_for(data.backend),
        label=f"{data.label} >=90% native refold",
    )
    ax.axvline(config.dockq_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    ax.axhline(config.delta_dg_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    if len(table):
        ax.axvline(float(table["dockq"].median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
        ax.axhline(float(table["delta_dg"].median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
    setup_axis(ax, data.title, "DockQ score", DECOY_ENERGY_AXIS_LABEL)
    ax.set_ylim(config.delta_dg_display_min, config.delta_dg_display_max)
    legend(ax, ncol=2)
    save(fig, out)


def plot_comparison_landscape(items: Sequence[Dataset], config: Config, out: Path, title: str, *, marker_by_backend: bool = True) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for data in items:
        table = display_landscape(data.landscape, config)
        light, dark = data.colors
        marker = marker_for(data.backend) if marker_by_backend else ("o" if data.mode == "hotspot" else "s")
        high_refold = table["native_refold_quality_pass"].fillna(False).astype(bool)
        ax.scatter(
            table.loc[~high_refold, "dockq"],
            table.loc[~high_refold, "delta_dg"],
            s=POINT_SIZE,
            color=light,
            alpha=0.66,
            edgecolors="none",
            marker=marker,
            label=f"{data.label} <90% native refold",
        )
        ax.scatter(
            table.loc[high_refold, "dockq"],
            table.loc[high_refold, "delta_dg"],
            s=POINT_SIZE,
            color=dark,
            alpha=0.82,
            edgecolors="none",
            marker=marker,
            label=f"{data.label} >=90% native refold",
        )
        if len(table):
            ax.axvline(float(table["dockq"].median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label=f"{data.label} median")
            ax.axhline(float(table["delta_dg"].median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label=f"{data.label} median")
    ax.axvline(config.dockq_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    ax.axhline(config.delta_dg_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    setup_axis(ax, title, "DockQ score", DECOY_ENERGY_AXIS_LABEL)
    ax.set_ylim(config.delta_dg_display_min, config.delta_dg_display_max)
    legend(ax, ncol=3)
    save(fig, out)


def plot_refold_hist(data: Dataset, config: Config, out: Path, *, high_high_only: bool, title_suffix: str) -> None:
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
    prefix = f"{data.label} High-High" if high_high_only else data.label
    setup_axis(ax, f"{prefix} Native Refold Quality {title_suffix}", "TM-score (%)", "Structures")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 105, 5))
    handles = [
        mpatches.Patch(color=light, label=f"{data.label} <90% native refold", alpha=0.88),
        mpatches.Patch(color=dark, label=f"{data.label} >=90% native refold", alpha=0.88),
    ]
    legend(ax, handles=handles, ncol=2)
    save(fig, out)


def plot_refold_quality_hist(data: RefoldQualityDataset, config: Config, out: Path) -> None:
    values = data.native_refold["native_refold_tm_percent"].dropna()
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
    title = f"{data.label} Native Refold Quality vs {REFERENCE_LABELS.get(data.reference_mode, data.reference_mode.title())}"
    setup_axis(ax, title, "TM-score (%)", "Structures")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 105, 5))
    handles = [
        mpatches.Patch(color=light, label=f"{data.label} <90% native refold", alpha=0.88),
        mpatches.Patch(color=dark, label=f"{data.label} >=90% native refold", alpha=0.88),
    ]
    legend(ax, handles=handles, ncol=2)
    save(fig, out)


def plot_refold_quality_comparison(items: Sequence[RefoldQualityDataset], config: Config, out: Path, title: str) -> None:
    bins = np.arange(0, 100 + config.tm_bin_width, config.tm_bin_width)
    centers = bins[:-1] + config.tm_bin_width / 2
    offsets = np.linspace(-GROUP_BAR_WIDTH / 2, GROUP_BAR_WIDTH / 2, len(items)) if len(items) > 1 else [0]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles: list[object] = []
    for idx, data in enumerate(items):
        values = data.native_refold["native_refold_tm_percent"].dropna()
        counts, _ = np.histogram(values, bins=bins)
        light, dark = data.colors
        colors = [dark if left >= config.tm_score_cutoff else light for left in bins[:-1]]
        ax.bar(centers + offsets[idx], counts, width=GROUP_BAR_WIDTH, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
        handles.extend(
            [
                mpatches.Patch(color=light, label=f"{data.label} <90%", alpha=0.88),
                mpatches.Patch(color=dark, label=f"{data.label} >=90%", alpha=0.88),
            ]
        )
    ax.axvline(config.tm_score_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Cutoff")
    setup_axis(ax, title, "TM-score (%)", "Structures")
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
        "reference_mode": data.reference_mode,
        "redesign_mode": data.mode,
        "backend": data.backend,
        "n_refolded_natives": int(len(data.native_refold)),
        "median_TM": float(data.native_refold["native_refold_tm_percent"].median()) if len(data.native_refold) else None,
        "n_decoys": int(len(landscape)),
        "n_decoys_plotted": int(len(display)),
        "n_hidden_display_outliers": int(len(landscape) - len(display)),
        "n_decoys_in_Q1": int(len(high)),
        "Q1_median_dockQ": float(high["dockq"].median()) if len(high) else None,
        "Q1_median_dG": float(high["delta_dg"].median()) if len(high) else None,
        "n_decoys_w_TM_gt_90": int((landscape["native_refold_tm_percent"] >= config.tm_score_cutoff).sum()),
        "n_decoys_in_Q1_w_TM_gt_90": int((high["native_refold_tm_percent"] >= config.tm_score_cutoff).sum()),
        "Q1_median_dockQ_TM_gt_90": float(high_tm["dockq"].median()) if len(high_tm) else None,
        "Q1_median_dG_TM_gt_90": float(high_tm["delta_dg"].median()) if len(high_tm) else None,
        "dockq_cutoff": config.dockq_cutoff,
        "delta_dg_cutoff": config.delta_dg_cutoff,
        "native_refold_tm_cutoff": config.tm_score_cutoff,
        "source_kind": data.source_kind,
        "source_paths": ";".join(data.source_paths),
    }


def refold_quality_summary_row(data: RefoldQualityDataset, config: Config) -> dict[str, object]:
    values = data.native_refold["native_refold_tm_percent"].dropna()
    return {
        "reference_mode": data.reference_mode,
        "redesign_mode": "native_refold_quality",
        "backend": data.backend,
        "n_refolded_natives": int(len(values)),
        "median_TM": float(values.median()) if len(values) else None,
        "n_decoys": None,
        "n_decoys_plotted": None,
        "n_hidden_display_outliers": None,
        "n_decoys_in_Q1": None,
        "Q1_median_dockQ": None,
        "Q1_median_dG": None,
        "n_decoys_w_TM_gt_90": None,
        "n_decoys_in_Q1_w_TM_gt_90": None,
        "Q1_median_dockQ_TM_gt_90": None,
        "Q1_median_dG_TM_gt_90": None,
        "dockq_cutoff": config.dockq_cutoff,
        "delta_dg_cutoff": config.delta_dg_cutoff,
        "native_refold_tm_cutoff": config.tm_score_cutoff,
        "source_kind": "native_refold_quality",
        "source_paths": data.source_path,
    }


def write_tables(data: Dataset, base: Path, config: Config) -> pd.DataFrame:
    tables = base / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    high = data.landscape[data.landscape["high_high"]].copy()
    high.to_csv(tables / "q1_decoys.csv", index=False)
    data.landscape.to_csv(tables / "decoy_ranked_ml_dataset.csv", index=False)
    data.native_refold.to_csv(tables / "native_refold_tm_scores.csv", index=False)
    data.high_high_refold.to_csv(tables / "q1_native_refold_tm_scores.csv", index=False)
    summary = pd.DataFrame([summary_row(data, config)])
    summary.to_csv(tables / "summary_stats.csv", index=False)
    (tables / "summary_stats.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2) + "\n")
    return summary


def dataset_arg_prefix(reference: str, mode: str, backend: str) -> str:
    return f"{reference}-{mode}-{backend}"


def add_dataset_args(parser: argparse.ArgumentParser, reference: str, mode: str, backend: str) -> None:
    prefix = dataset_arg_prefix(reference, mode, backend)
    parser.add_argument(f"--{prefix}-landscape", default=None)
    parser.add_argument(f"--{prefix}-dockq", default=None)
    parser.add_argument(f"--{prefix}-decoy-rosetta", default=None)
    parser.add_argument(f"--{prefix}-reference-rosetta", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reference-benchmark-figures",
        description="Write canonical decoy-vs-reference DockQ/delta-dG figure packages.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/outputs/evaluation/reference_benchmark"))
    parser.add_argument("--references", nargs="+", default=["relaxed"], choices=["crystal", "relaxed", "refold"])
    parser.add_argument("--esm-backend-key", default="esmfold2")
    parser.add_argument("--esm-label", default="ESM")
    parser.add_argument("--opendde-backend-key", default="opendde")
    parser.add_argument("--opendde-label", default="OpenDDE")
    parser.add_argument("--esm-native-refold", required=True)
    parser.add_argument("--opendde-native-refold", required=True)
    parser.add_argument("--esm-native-refold-crystal", default=None)
    parser.add_argument("--opendde-native-refold-crystal", default=None)
    parser.add_argument("--dockq-cutoff", type=float, default=DOCKQ_CUTOFF)
    parser.add_argument("--delta-dg-cutoff", type=float, default=DELTA_DG_CUTOFF)
    parser.add_argument("--tm-score-cutoff", type=float, default=TM_SCORE_CUTOFF)
    parser.add_argument("--delta-dg-display-min", type=float, default=DELTA_DG_DISPLAY_MIN)
    parser.add_argument("--delta-dg-display-max", type=float, default=DELTA_DG_DISPLAY_MAX)
    for reference in ["crystal", "relaxed", "refold"]:
        for mode in ["hotspot", "interface"]:
            for backend in ["esm", "opendde"]:
                add_dataset_args(parser, reference, mode, backend)
    return parser


def get_arg(args: argparse.Namespace, reference: str, mode: str, backend: str, suffix: str) -> str | None:
    return getattr(args, f"{reference}_{mode}_{backend}_{suffix}")


def load_dataset_for(
    args: argparse.Namespace,
    *,
    reference: str,
    mode: str,
    backend_input: str,
    backend_key: str,
    native_refold: pd.DataFrame,
    config: Config,
) -> Dataset | None:
    landscape = get_arg(args, reference, mode, backend_input, "landscape")
    if landscape:
        return build_from_landscape(
            path=landscape,
            mode=mode,
            backend=backend_key,
            reference_mode=reference,
            native_refold=native_refold,
            config=config,
        )
    dockq = get_arg(args, reference, mode, backend_input, "dockq")
    decoy_rosetta = get_arg(args, reference, mode, backend_input, "decoy_rosetta")
    reference_rosetta = get_arg(args, reference, mode, backend_input, "reference_rosetta")
    values = [dockq, decoy_rosetta, reference_rosetta]
    if not any(values):
        return None
    if not all(values):
        missing = [name for name, value in zip(["dockq", "decoy-rosetta", "reference-rosetta"], values) if not value]
        raise ValueError(f"incomplete source-table inputs for {reference}/{mode}/{backend_input}; missing {', '.join(missing)}")
    return build_from_source_tables(
        dockq_path=dockq or "",
        decoy_rosetta_path=decoy_rosetta or "",
        reference_rosetta_path=reference_rosetta or "",
        mode=mode,
        backend=backend_key,
        reference_mode=reference,
        native_refold=native_refold,
        config=config,
    )


def write_readme(output_root: Path, summaries: pd.DataFrame, missing: list[dict[str, str]]) -> None:
    lines = [
        "# Jango Reference Benchmark Figures",
        "",
        "This package contains the canonical decoy-vs-reference DockQ/delta-dG figures.",
        "Q1 is defined as DockQ >= cutoff and delta dG >= cutoff.",
        "",
        "The `plot_inventory.csv` file records the exact source table(s) used for every generated plot.",
        "",
        "## Generated rows",
        "",
        f"- datasets: {len(summaries)}",
        f"- missing requested datasets: {len(missing)}",
    ]
    if missing:
        lines.extend(["", "## Missing requested datasets", ""])
        lines.extend([f"- {m['reference_mode']} / {m['redesign_mode']} / {m['backend_input']}: {m['reason']}" for m in missing])
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


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
    BACKEND_LABELS[args.esm_backend_key] = args.esm_label
    BACKEND_LABELS[args.opendde_backend_key] = args.opendde_label
    refold_quality_inputs = {
        ("relaxed", "esm"): args.esm_native_refold,
        ("relaxed", "opendde"): args.opendde_native_refold,
        ("crystal", "esm"): args.esm_native_refold_crystal,
        ("crystal", "opendde"): args.opendde_native_refold_crystal,
    }
    native_refolds_by_reference: dict[tuple[str, str], pd.DataFrame] = {}
    for backend_input in ["esm", "opendde"]:
        relaxed_refold = load_native_refold(refold_quality_inputs[("relaxed", backend_input)] or "")
        native_refolds_by_reference[("relaxed", backend_input)] = relaxed_refold
        native_refolds_by_reference[("refold", backend_input)] = relaxed_refold
        crystal_refold_path = refold_quality_inputs[("crystal", backend_input)]
        native_refolds_by_reference[("crystal", backend_input)] = (
            load_native_refold(crystal_refold_path) if crystal_refold_path else relaxed_refold
        )
    backend_keys = {"esm": args.esm_backend_key, "opendde": args.opendde_backend_key}
    datasets: dict[tuple[str, str, str], Dataset] = {}
    missing: list[dict[str, str]] = []
    for reference in args.references:
        for mode in ["hotspot", "interface"]:
            for backend_input in ["esm", "opendde"]:
                data = load_dataset_for(
                    args,
                    reference=reference,
                    mode=mode,
                    backend_input=backend_input,
                    backend_key=backend_keys[backend_input],
                    native_refold=native_refolds_by_reference[(reference, backend_input)],
                    config=config,
                )
                if data is None:
                    missing.append(
                        {
                            "reference_mode": reference,
                            "redesign_mode": mode,
                            "backend_input": backend_input,
                            "reason": "no landscape or source-table inputs supplied",
                        }
                    )
                    continue
                datasets[(reference, mode, backend_input)] = data

    summary_parts: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    refold_quality_items: dict[tuple[str, str], RefoldQualityDataset] = {}
    for reference in ["relaxed", "crystal"]:
        for backend_input in ["esm", "opendde"]:
            source_path = refold_quality_inputs[(reference, backend_input)]
            if not source_path:
                missing.append(
                    {
                        "reference_mode": reference,
                        "redesign_mode": "native_refold_quality",
                        "backend_input": backend_input,
                        "reason": "no native-refold quality input supplied",
                    }
                )
                continue
            item = RefoldQualityDataset(
                backend=backend_keys[backend_input],
                reference_mode=reference,
                native_refold=load_native_refold(source_path),
                source_path=source_path,
            )
            refold_quality_items[(reference, backend_input)] = item
            base = config.output_root / "native_refold_quality" / f"refold_vs_{reference}" / item.backend
            (base / "tables").mkdir(parents=True, exist_ok=True)
            (base / "figures").mkdir(parents=True, exist_ok=True)
            item.native_refold.to_csv(base / "tables" / "native_refold_tm_scores.csv", index=False)
            pd.DataFrame([refold_quality_summary_row(item, config)]).to_csv(base / "tables" / "summary_stats.csv", index=False)
            figure = base / "figures" / "native_refold_tm_histogram.png"
            plot_refold_quality_hist(item, config, figure)
            summary_parts.append(pd.DataFrame([refold_quality_summary_row(item, config)]))
            inventory.append(
                {
                    "figure": str(figure),
                    "reference_mode": reference,
                    "redesign_mode": "native_refold_quality",
                    "backend": item.backend,
                    "figure_type": "native_refold_quality",
                    "source_kind": "native_refold_quality",
                    "source_paths": source_path,
                }
            )
        comparison_items = [refold_quality_items[(reference, b)] for b in ["esm", "opendde"] if (reference, b) in refold_quality_items]
        if len(comparison_items) >= 2:
            out = (
                config.output_root
                / "native_refold_quality"
                / f"refold_vs_{reference}"
                / "comparison"
                / "figures"
                / "backend_comparison_native_refold_tm_histogram.png"
            )
            plot_refold_quality_comparison(
                comparison_items,
                config,
                out,
                f"Backend Native Refold Quality vs {REFERENCE_LABELS.get(reference, reference.title())}",
            )
            (out.parent.parent / "tables").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([refold_quality_summary_row(item, config) for item in comparison_items]).to_csv(
                out.parent.parent / "tables" / "summary_stats.csv",
                index=False,
            )
            inventory.append(
                {
                    "figure": str(out),
                    "reference_mode": reference,
                    "redesign_mode": "native_refold_quality",
                    "backend": "comparison",
                    "figure_type": "native_refold_quality_backend_comparison",
                    "source_kind": "derived",
                    "source_paths": ";".join(item.source_path for item in comparison_items),
                }
            )

    for key, data in sorted(datasets.items()):
        reference, mode, backend_input = key
        base = config.output_root / f"reference_{reference}" / mode / data.backend
        summary = write_tables(data, base, config)
        summary_parts.append(summary)
        figure = base / "figures" / "landscape_dockq_delta_dg.png"
        plot_landscape(data, config, figure)
        inventory.append(
            {
                "figure": str(figure),
                "reference_mode": reference,
                "redesign_mode": mode,
                "backend": data.backend,
                "figure_type": "decoy_vs_reference",
                "source_kind": data.source_kind,
                "source_paths": ";".join(data.source_paths),
            }
        )

    for reference in args.references:
        for mode in ["hotspot", "interface"]:
            mode_items = [datasets[(reference, mode, b)] for b in ["esm", "opendde"] if (reference, mode, b) in datasets]
            if len(mode_items) < 2:
                continue
            out = config.output_root / f"reference_{reference}" / mode / "comparison" / "figures" / "backend_comparison_landscape_dockq_delta_dg.png"
            plot_comparison_landscape(mode_items, config, out, f"{MODE_LABELS[mode]} Backend Comparison Decoy vs {REFERENCE_LABELS[reference]}")
            (out.parent.parent / "tables").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([summary_row(item, config) for item in mode_items]).to_csv(
                out.parent.parent / "tables" / "summary_stats.csv", index=False
            )
            inventory.append(
                {
                    "figure": str(out),
                    "reference_mode": reference,
                    "redesign_mode": mode,
                    "backend": "comparison",
                    "figure_type": "backend_comparison",
                    "source_kind": "derived",
                    "source_paths": ";".join(";".join(item.source_paths) for item in mode_items),
                }
            )

        for backend_input in ["esm", "opendde"]:
            items = [datasets[(reference, mode, backend_input)] for mode in ["hotspot", "interface"] if (reference, mode, backend_input) in datasets]
            if len(items) < 2:
                continue
            backend_label = BACKEND_LABELS.get(items[0].backend, items[0].backend)
            out = (
                config.output_root
                / f"reference_{reference}"
                / "redesign_mode_comparison"
                / items[0].backend
                / "figures"
                / "redesign_mode_landscape_dockq_delta_dg.png"
            )
            plot_comparison_landscape(items, config, out, f"{backend_label} Redesign Mode Decoy vs {REFERENCE_LABELS[reference]}", marker_by_backend=False)
            (out.parent.parent / "tables").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([summary_row(item, config) for item in items]).to_csv(out.parent.parent / "tables" / "summary_stats.csv", index=False)
            inventory.append(
                {
                    "figure": str(out),
                    "reference_mode": reference,
                    "redesign_mode": "hotspot_vs_interface",
                    "backend": items[0].backend,
                    "figure_type": "redesign_mode_comparison",
                    "source_kind": "derived",
                    "source_paths": ";".join(";".join(item.source_paths) for item in items),
                }
            )

    if summary_parts:
        combined = pd.concat(summary_parts, ignore_index=True)
    else:
        combined = pd.DataFrame()
    config.output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.output_root / "summary_stats.csv", index=False)
    pd.DataFrame(inventory, columns=PLOT_INVENTORY_COLUMNS).to_csv(config.output_root / "plot_inventory.csv", index=False)
    pd.DataFrame(missing, columns=MISSING_INPUT_COLUMNS).to_csv(config.output_root / "missing_inputs.csv", index=False)
    (config.output_root / "summary_stats.json").write_text(json.dumps(combined.to_dict(orient="records"), indent=2) + "\n")
    write_readme(config.output_root, combined, missing)
    print(f"wrote reference benchmark figures and tables to {config.output_root}")
    if missing:
        print(f"missing requested datasets: {len(missing)}")
    if len(combined):
        print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
