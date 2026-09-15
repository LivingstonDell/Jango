from __future__ import annotations

from pathlib import Path

from .plots import set_publication_style

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


COMPARISON_METRICS = [
    "fullpose_total_score",
    "dG_separated",
    "dG_separated/dSASAx100",
    "dSASA_int",
    "sc_value",
    "hbonds_int",
    "delta_unsatHbonds",
    "nres_int",
]

METRIC_LABELS = {
    "fullpose_total_score": "Full-pose score",
    "dG_separated": "Interface dG",
    "dG_separated/dSASAx100": "Interface dG / dSASA",
    "dSASA_int": "Buried SASA",
    "sc_value": "Shape complementarity",
    "hbonds_int": "Interface H-bonds",
    "delta_unsatHbonds": "Buried unsatisfied H-bonds",
    "nres_int": "Interface residues",
}


def compare_rosetta_native_relaxed(
    native_csv: Path,
    relaxed_csv: Path,
    out_delta: Path,
    out_summary: Path,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_publication_style()
    native = pd.read_csv(native_csv)
    relaxed = pd.read_csv(relaxed_csv)
    paired = build_paired_delta_table(native, relaxed)
    summary = summarize_relaxation_effects(paired)

    out_delta.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_delta, index=False)
    summary.to_csv(out_summary, index=False)
    plot_native_relaxed_scatter(paired, figures_dir / "rosetta_native_vs_relaxed_scatter.png")
    plot_relaxation_deltas(paired, figures_dir / "rosetta_relaxation_delta_distributions.png")
    return paired, summary


def build_paired_delta_table(native: pd.DataFrame, relaxed: pd.DataFrame) -> pd.DataFrame:
    metadata = ["structure_id", "pdb_id", "nanobody_chain", "antigen_chain", "interface"]
    columns = metadata + COMPARISON_METRICS
    paired = native[columns].merge(
        relaxed[columns],
        on=metadata,
        how="inner",
        suffixes=("_native", "_relaxed"),
        validate="one_to_one",
    )
    for metric in COMPARISON_METRICS:
        paired[f"delta_{safe_metric_name(metric)}"] = paired[f"{metric}_relaxed"] - paired[f"{metric}_native"]
    paired["relaxation_energy_improved"] = paired["delta_fullpose_total_score"] < 0
    paired["interface_dg_improved"] = paired["delta_dG_separated"] < 0
    paired["shape_complementarity_improved"] = paired["delta_sc_value"] > 0
    paired["hbond_count_improved"] = paired["delta_hbonds_int"] > 0
    paired["buried_unsats_improved"] = paired["delta_delta_unsatHbonds"] < 0
    return paired


def summarize_relaxation_effects(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in COMPARISON_METRICS:
        native_col = f"{metric}_native"
        relaxed_col = f"{metric}_relaxed"
        delta_col = f"delta_{safe_metric_name(metric)}"
        subset = paired[[native_col, relaxed_col, delta_col]].dropna()
        delta = subset[delta_col]
        if delta.empty:
            p_value = pd.NA
        elif (delta != 0).any():
            p_value = float(stats.wilcoxon(delta).pvalue)
        else:
            p_value = 1.0
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "n": int(subset.shape[0]),
                "native_median": float(subset[native_col].median()),
                "relaxed_median": float(subset[relaxed_col].median()),
                "delta_median": float(delta.median()),
                "delta_mean": float(delta.mean()),
                "delta_p10": float(delta.quantile(0.10)),
                "delta_p90": float(delta.quantile(0.90)),
                "fraction_increased": float((delta > 0).mean()),
                "fraction_decreased": float((delta < 0).mean()),
                "spearman_r": spearman_or_na(subset[native_col], subset[relaxed_col]),
                "wilcoxon_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def spearman_or_na(native: pd.Series, relaxed: pd.Series) -> float | pd.NA:
    if native.nunique(dropna=True) < 2 or relaxed.nunique(dropna=True) < 2:
        return pd.NA
    return float(native.corr(relaxed, method="spearman"))


def safe_metric_name(metric: str) -> str:
    return metric.replace("/", "_per_")


def plot_native_relaxed_scatter(paired: pd.DataFrame, path: Path) -> None:
    metrics = [
        "fullpose_total_score",
        "dG_separated",
        "dG_separated/dSASAx100",
        "dSASA_int",
        "sc_value",
        "hbonds_int",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        x = paired[f"{metric}_native"]
        y = paired[f"{metric}_relaxed"]
        sns.scatterplot(x=x, y=y, s=14, linewidth=0, color="#2f6f8f", alpha=0.65, ax=ax)
        lo, hi = robust_limits(pd.concat([x, y]), lower=0.01, upper=0.99)
        ax.plot([lo, hi], [lo, hi], color="0.35", linewidth=0.8, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Native")
        ax.set_ylabel("Relaxed")
        ax.set_title(METRIC_LABELS.get(metric, metric))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_relaxation_deltas(paired: pd.DataFrame, path: Path) -> None:
    metrics = [
        "fullpose_total_score",
        "dG_separated",
        "dG_separated/dSASAx100",
        "dSASA_int",
        "sc_value",
        "hbonds_int",
        "delta_unsatHbonds",
        "nres_int",
    ]
    long_rows = []
    for metric in metrics:
        delta_col = f"delta_{safe_metric_name(metric)}"
        for value in paired[delta_col].dropna():
            long_rows.append({"metric": METRIC_LABELS.get(metric, metric), "delta": value})
    long = pd.DataFrame(long_rows)
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.8), constrained_layout=True)
    for ax, (metric_label, group) in zip(axes.flat, long.groupby("metric", sort=False)):
        lo, hi = robust_limits(group["delta"], lower=0.01, upper=0.99)
        sns.histplot(group["delta"], bins=28, binrange=(lo, hi), color="#4c78a8", edgecolor="white", linewidth=0.3, ax=ax)
        ax.axvline(0, color="0.25", linewidth=0.8)
        ax.axvline(group["delta"].median(), color="#b03a2e", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_title(metric_label)
        ax.set_xlabel("Relaxed - native")
        ax.set_ylabel("Interfaces")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def robust_limits(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> tuple[float, float]:
    finite = values.dropna()
    lo = float(finite.quantile(lower))
    hi = float(finite.quantile(upper))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
    else:
        pad = (hi - lo) * 0.06
    return lo - pad, hi + pad
