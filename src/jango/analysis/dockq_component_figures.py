#!/usr/bin/env python3
"""Canonical DockQ component vs delta-dG figure sets.

This command reads a compact high-high evaluation package and writes a separate
figure package where every DockQ-vs-delta-dG landscape is repeated for the three
official DockQ component metrics:

* dockq_fnat
* dockq_irmsd
* dockq_lrmsd

It does not change Jango's canonical quadrant definitions or the core figure
package.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from jango.analysis.high_high_figures import (
    BACKEND_LABELS,
    BORDER_WIDTH,
    DARK_GREY,
    DELTA_DG_CUTOFF,
    DELTA_DG_DISPLAY_MAX,
    DELTA_DG_DISPLAY_MIN,
    FIGSIZE,
    GRID_GREY,
    GUIDE_WIDTH,
    MODE_BACKEND_COLORS,
    MODE_LABELS,
    POINT_SIZE,
    REPO_ROOT,
    legend,
    marker_for,
    save,
    setup_axis,
)


DEFAULT_INPUT_ROOT = Path("data/outputs/evaluation/q1_core")
DEFAULT_OUTPUT_ROOT = Path("data/outputs/evaluation/dockq_components")


@dataclass(frozen=True)
class Config:
    input_root: Path
    output_root: Path
    delta_dg_cutoff: float = DELTA_DG_CUTOFF
    delta_dg_display_min: float = DELTA_DG_DISPLAY_MIN
    delta_dg_display_max: float = DELTA_DG_DISPLAY_MAX
    rmsd_display_quantile: float = 0.995


@dataclass(frozen=True)
class ComponentSpec:
    column: str
    slug: str
    title: str
    xlabel: str
    fixed_min: float = 0.0
    fixed_max: float | None = None


COMPONENTS = [
    ComponentSpec("dockq_fnat", "fnat", "DockQ Fnat", "DockQ Fnat", fixed_min=0.0, fixed_max=1.0),
    ComponentSpec("dockq_irmsd", "irmsd", "DockQ iRMSD", "DockQ iRMSD (A)", fixed_min=0.0),
    ComponentSpec("dockq_lrmsd", "lrmsd", "DockQ LRMSD", "DockQ LRMSD (A)", fixed_min=0.0),
]


@dataclass(frozen=True)
class Dataset:
    mode: str
    backend: str
    table: pd.DataFrame
    source_path: Path

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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def numeric(df: pd.DataFrame, column: str, label: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"missing {label} column: {column}")
    return pd.to_numeric(df[column], errors="coerce")


def truthy_series(raw: pd.Series) -> pd.Series:
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def bool_series(df: pd.DataFrame) -> pd.Series:
    if "native_refold_quality_pass" in df.columns:
        return truthy_series(df["native_refold_quality_pass"])
    if "native_refold_tm_percent" in df.columns:
        return pd.to_numeric(df["native_refold_tm_percent"], errors="coerce") >= 90.0
    return pd.Series(False, index=df.index)


def load_dataset(input_root: Path, mode: str, backend: str) -> Dataset:
    path = input_root / mode / backend / "tables" / "high_high_decoy_ranked_ml_dataset.csv"
    if not path.is_file():
        raise FileNotFoundError(f"input table not found: {path}")
    table = pd.read_csv(path)
    required = {"structure_id", "delta_dg", "dockq", "native_refold_quality_pass"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} missing required column(s): {', '.join(missing)}")
    for component in COMPONENTS:
        if component.column not in table.columns:
            raise ValueError(f"{path} missing DockQ component column: {component.column}")
    table = table.copy()
    table["delta_dg"] = numeric(table, "delta_dg", "delta dG")
    table["dockq"] = numeric(table, "dockq", "DockQ")
    table["native_refold_quality_pass"] = bool_series(table)
    for component in COMPONENTS:
        table[component.column] = numeric(table, component.column, component.title)
    return Dataset(mode=mode, backend=backend, table=table, source_path=path)


def y_display(table: pd.DataFrame, config: Config) -> pd.DataFrame:
    return table[
        (table["delta_dg"] >= config.delta_dg_display_min)
        & (table["delta_dg"] <= config.delta_dg_display_max)
    ].copy()


def component_x_limits(tables: Iterable[pd.DataFrame], component: ComponentSpec, config: Config) -> tuple[float, float]:
    if component.fixed_max is not None:
        return component.fixed_min, component.fixed_max
    values = []
    for table in tables:
        values.append(pd.to_numeric(y_display(table, config)[component.column], errors="coerce"))
    combined = pd.concat(values, ignore_index=True).dropna() if values else pd.Series(dtype=float)
    combined = combined[combined >= component.fixed_min]
    if combined.empty:
        return component.fixed_min, component.fixed_min + 1.0
    q = max(0.0, min(1.0, config.rmsd_display_quantile))
    xmax = float(combined.quantile(q)) if q < 1.0 else float(combined.max())
    if xmax <= component.fixed_min:
        xmax = float(combined.max())
    pad = max((xmax - component.fixed_min) * 0.06, 0.25)
    return component.fixed_min, xmax + pad


def component_display(table: pd.DataFrame, component: ComponentSpec, config: Config, x_limits: tuple[float, float]) -> pd.DataFrame:
    x_min, x_max = x_limits
    shown = y_display(table, config).dropna(subset=[component.column, "delta_dg"]).copy()
    return shown[(shown[component.column] >= x_min) & (shown[component.column] <= x_max)].copy()


def style_component_axis(ax: plt.Axes, title: str, xlabel: str, config: Config, x_limits: tuple[float, float]) -> None:
    setup_axis(ax, title, xlabel, "Delta dG (REU)")
    ax.set_xlim(*x_limits)
    ax.set_ylim(config.delta_dg_display_min, config.delta_dg_display_max)
    ax.grid(True, color=GRID_GREY, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(BORDER_WIDTH)


def plot_component(data: Dataset, component: ComponentSpec, config: Config, out: Path) -> None:
    x_limits = component_x_limits([data.table], component, config)
    table = component_display(data.table, component, config, x_limits)
    light, dark = data.colors
    high_refold = table["native_refold_quality_pass"].fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        table.loc[~high_refold, component.column],
        table.loc[~high_refold, "delta_dg"],
        s=POINT_SIZE,
        color=light,
        alpha=0.76,
        edgecolors="none",
        marker=marker_for(data.backend),
        label=f"{data.label} <90% native refold",
    )
    ax.scatter(
        table.loc[high_refold, component.column],
        table.loc[high_refold, "delta_dg"],
        s=POINT_SIZE,
        color=dark,
        alpha=0.88,
        edgecolors="none",
        marker=marker_for(data.backend),
        label=f"{data.label} >=90% native refold",
    )
    ax.axhline(config.delta_dg_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Delta dG cutoff")
    if len(table):
        ax.axvline(float(table[component.column].median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
        ax.axhline(float(table["delta_dg"].median()), color=dark, linewidth=GUIDE_WIDTH, linestyle="--", label="Median")
    style_component_axis(ax, f"{data.label} {component.title} vs Delta dG", component.xlabel, config, x_limits)
    legend(ax, ncol=2)
    save(fig, out)


def plot_component_comparison(
    items: Sequence[Dataset],
    component: ComponentSpec,
    config: Config,
    out: Path,
    title: str,
    *,
    marker_by_backend: bool = True,
) -> None:
    x_limits = component_x_limits([item.table for item in items], component, config)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for data in items:
        table = component_display(data.table, component, config, x_limits)
        light, dark = data.colors
        marker = marker_for(data.backend) if marker_by_backend else ("o" if data.mode == "hotspot" else "s")
        high_refold = table["native_refold_quality_pass"].fillna(False).astype(bool)
        ax.scatter(
            table.loc[~high_refold, component.column],
            table.loc[~high_refold, "delta_dg"],
            s=POINT_SIZE,
            color=light,
            alpha=0.66,
            edgecolors="none",
            marker=marker,
            label=f"{data.label} <90% native refold",
        )
        ax.scatter(
            table.loc[high_refold, component.column],
            table.loc[high_refold, "delta_dg"],
            s=POINT_SIZE,
            color=dark,
            alpha=0.82,
            edgecolors="none",
            marker=marker,
            label=f"{data.label} >=90% native refold",
        )
    ax.axhline(config.delta_dg_cutoff, color=DARK_GREY, linewidth=GUIDE_WIDTH, linestyle="-", label="Delta dG cutoff")
    style_component_axis(ax, title, component.xlabel, config, x_limits)
    legend(ax, ncol=3)
    save(fig, out)


def component_summary(data: Dataset, component: ComponentSpec, config: Config) -> dict[str, object]:
    x_limits = component_x_limits([data.table], component, config)
    numeric_rows = data.table.dropna(subset=[component.column, "delta_dg"])
    displayed = component_display(data.table, component, config, x_limits)
    high_high = displayed[truthy_series(displayed["high_high"])] if "high_high" in displayed.columns else displayed.iloc[0:0]
    high_refold = displayed[truthy_series(displayed["native_refold_quality_pass"])]
    return {
        "redesign_mode": data.mode,
        "backend": data.backend,
        "component": component.column,
        "component_label": component.title,
        "source_table": str(data.source_path),
        "n_total_decoys": int(len(data.table)),
        "n_plotted": int(len(displayed)),
        "n_hidden_display_outliers": int(len(numeric_rows) - len(displayed)),
        "x_display_min": float(x_limits[0]),
        "x_display_max": float(x_limits[1]),
        "component_median_plotted": float(displayed[component.column].median()) if len(displayed) else None,
        "delta_dg_median_plotted": float(displayed["delta_dg"].median()) if len(displayed) else None,
        "n_high_high_composite_plotted": int(len(high_high)),
        "component_median_high_high_composite": float(high_high[component.column].median()) if len(high_high) else None,
        "delta_dg_median_high_high_composite": float(high_high["delta_dg"].median()) if len(high_high) else None,
        "n_native_refold_tm_ge_90_plotted": int(len(high_refold)),
        "delta_dg_cutoff": config.delta_dg_cutoff,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dockq-component-figures", description="Write canonical DockQ component vs delta-dG figure package.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--esm-backend-key", default="esmfold2")
    parser.add_argument("--esm-label", default="ESM")
    parser.add_argument("--opendde-backend-key", default="opendde")
    parser.add_argument("--opendde-label", default="OpenDDE")
    parser.add_argument("--delta-dg-cutoff", type=float, default=DELTA_DG_CUTOFF)
    parser.add_argument("--delta-dg-display-min", type=float, default=DELTA_DG_DISPLAY_MIN)
    parser.add_argument("--delta-dg-display-max", type=float, default=DELTA_DG_DISPLAY_MAX)
    parser.add_argument("--rmsd-display-quantile", type=float, default=0.995)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    BACKEND_LABELS[args.esm_backend_key] = args.esm_label
    BACKEND_LABELS[args.opendde_backend_key] = args.opendde_label
    config = Config(
        input_root=resolve_path(args.input_root),
        output_root=resolve_path(args.output_root),
        delta_dg_cutoff=args.delta_dg_cutoff,
        delta_dg_display_min=args.delta_dg_display_min,
        delta_dg_display_max=args.delta_dg_display_max,
        rmsd_display_quantile=args.rmsd_display_quantile,
    )
    backend_keys = [args.esm_backend_key, args.opendde_backend_key]
    datasets: dict[tuple[str, str], Dataset] = {}
    for mode in ["hotspot", "interface"]:
        for backend in backend_keys:
            datasets[(mode, backend)] = load_dataset(config.input_root, mode, backend)

    summary_rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "components": [asdict(component) for component in COMPONENTS],
        "delta_dg_display_min": config.delta_dg_display_min,
        "delta_dg_display_max": config.delta_dg_display_max,
        "delta_dg_cutoff": config.delta_dg_cutoff,
        "rmsd_display_quantile": config.rmsd_display_quantile,
        "source_tables": {},
    }

    for mode in ["hotspot", "interface"]:
        for backend in backend_keys:
            data = datasets[(mode, backend)]
            manifest["source_tables"][f"{mode}/{backend}"] = str(data.source_path)
            base = config.output_root / mode / backend
            for component in COMPONENTS:
                plot_component(data, component, config, base / "figures" / f"{component.slug}_vs_delta_dg.png")
                summary_rows.append(component_summary(data, component, config))

        items = [datasets[(mode, backend)] for backend in backend_keys]
        comparison = config.output_root / mode / "comparison"
        for component in COMPONENTS:
            plot_component_comparison(
                items,
                component,
                config,
                comparison / "figures" / f"backend_comparison_{component.slug}_vs_delta_dg.png",
                f"{MODE_LABELS[mode]} {component.title} vs Delta dG Comparison",
            )

    for backend in backend_keys:
        items = [datasets[("hotspot", backend)], datasets[("interface", backend)]]
        out_dir = config.output_root / "comparison_by_mode" / backend
        for component in COMPONENTS:
            plot_component_comparison(
                items,
                component,
                config,
                out_dir / "figures" / f"redesign_mode_{component.slug}_vs_delta_dg.png",
                f"{BACKEND_LABELS.get(backend, backend)} Redesign Mode {component.title} vs Delta dG",
                marker_by_backend=False,
            )

    comparison_root = config.output_root / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(comparison_root / "dockq_component_figure_summary.csv", index=False)
    (comparison_root / "dockq_component_figure_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    (comparison_root / "dockq_component_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote DockQ component figures and summaries to {config.output_root}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
