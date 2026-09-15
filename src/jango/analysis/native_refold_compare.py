"""Native-refold fidelity control analysis.

This command compares a relaxed native complex with a relaxed complex containing a
backend-native-refolded antigen. It is intentionally read-only with respect to
folding/scoring outputs: all generated tables and figures go under --out-dir.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nbia.pdbio import parse_atoms, read_pdb_text
from nbia.plots import set_publication_style
from nbia.residues import AA3_TO_1

ROSETTA_METRICS = [
    "dG_separated",
    "dSASA_int",
    "packstat",
    "sc_value",
    "hbonds_int",
    "delta_unsatHbonds",
    "fullpose_total_score",
    "dG_separated/dSASAx100",
]

METRIC_LABELS = {
    "dG_separated": "Interface dG",
    "dSASA_int": "Buried SASA",
    "packstat": "Packstat",
    "sc_value": "Shape complementarity",
    "hbonds_int": "Interface H-bonds",
    "delta_unsatHbonds": "Buried unsat H-bonds",
    "fullpose_total_score": "Full-pose score",
    "dG_separated/dSASAx100": "dG / dSASA",
}


@dataclass(frozen=True)
class ResidueCA:
    residue_number: int
    insertion_code: str
    residue_name: str
    aa: str
    coord: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="native-refold-compare",
        description="Compare relaxed native complexes to relaxed backend-native-refold controls.",
    )
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--pairs-csv", type=Path, help="Explicit pair table with structure_id, native_relaxed_pdb, refold_relaxed_pdb, and optional antigen_chain.")
    inputs.add_argument("--native-relax-manifest", type=Path, help="Native relaxed manifest/table containing structure_id and relaxed PDB path.")
    inputs.add_argument("--refold-relax-manifest", type=Path, help="Refold-relaxed manifest/table containing structure_id and relaxed PDB path.")
    inputs.add_argument("--native-metrics", type=Path, help="Optional native relaxed InterfaceAnalyzer/Rosetta metric table.")
    inputs.add_argument("--refold-metrics", type=Path, help="Optional refold relaxed InterfaceAnalyzer/Rosetta metric table.")
    inputs.add_argument("--native-path-column", default="", help="Native relaxed PDB column. Defaults to first recognized path column.")
    inputs.add_argument("--refold-path-column", default="", help="Refold relaxed PDB column. Defaults to first recognized path column.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Destination for generated control-analysis tables and figures.")
    parser.add_argument("--refold-label", default="refold", help="Label used for the folded-native/refold state in figures.")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _path_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() == "nan" else text


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _resolve_path_column(df: pd.DataFrame, explicit: str, role: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{role} path column not found: {explicit}")
        return explicit
    column = _first_present(
        list(df.columns),
        [
            f"{role}_relaxed_pdb",
            "relaxed_pdb",
            "relax_pdb",
            "pdb_path",
            "structure_path",
            "output_pdb",
        ],
    )
    if column is None:
        raise ValueError(f"could not identify {role} relaxed PDB column")
    return column


def infer_antigen_chain(structure_id: object) -> str:
    parts = str(structure_id or "").split("_")
    return parts[-1] if len(parts) >= 3 and parts[-1] else ""


def _require_unique(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "structure_id" not in df.columns:
        raise ValueError(f"{label} table requires structure_id")
    duplicates = df["structure_id"][df["structure_id"].duplicated()].astype(str).unique()
    if len(duplicates):
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{label} table has duplicate structure_id values: {preview}")
    return df.copy()


def build_pair_table(args: argparse.Namespace) -> pd.DataFrame:
    if args.pairs_csv is not None:
        pairs = pd.read_csv(args.pairs_csv).fillna("")
        required = {"structure_id", "native_relaxed_pdb", "refold_relaxed_pdb"}
        missing = required - set(pairs.columns)
        if missing:
            raise ValueError(f"pairs CSV missing required columns: {sorted(missing)}")
        out = pairs.copy()
    else:
        if args.native_relax_manifest is None or args.refold_relax_manifest is None:
            raise ValueError("provide either --pairs-csv or both --native-relax-manifest and --refold-relax-manifest")
        native = _require_unique(pd.read_csv(args.native_relax_manifest).fillna(""), "native relax")
        refold = _require_unique(pd.read_csv(args.refold_relax_manifest).fillna(""), "refold relax")
        native_path_col = _resolve_path_column(native, args.native_path_column, "native")
        refold_path_col = _resolve_path_column(refold, args.refold_path_column, "refold")
        native_cols = ["structure_id", native_path_col, *[c for c in ("antigen_chain", "nanobody_chain") if c in native.columns]]
        refold_cols = ["structure_id", refold_path_col]
        out = native[native_cols].merge(refold[refold_cols], on="structure_id", how="inner", validate="one_to_one")
        out = out.rename(columns={native_path_col: "native_relaxed_pdb", refold_path_col: "refold_relaxed_pdb"})
    if args.limit is not None:
        out = out.head(args.limit).copy()

    inferred = out["structure_id"].map(infer_antigen_chain)
    if "antigen_chain" in out.columns:
        antigen_chain = out["antigen_chain"].replace("", pd.NA).fillna(inferred)
    else:
        antigen_chain = inferred
    if "native_antigen_chain" in out.columns:
        out["native_antigen_chain"] = out["native_antigen_chain"].replace("", pd.NA).fillna(antigen_chain)
    else:
        out["native_antigen_chain"] = antigen_chain
    if "refold_antigen_chain" in out.columns:
        out["refold_antigen_chain"] = out["refold_antigen_chain"].replace("", pd.NA).fillna(antigen_chain)
    else:
        out["refold_antigen_chain"] = antigen_chain

    native_chain = out["native_antigen_chain"].astype(str)
    refold_chain = out["refold_antigen_chain"].astype(str)
    out["antigen_chain"] = native_chain.where(native_chain.eq(refold_chain), native_chain + "->" + refold_chain)
    return out.fillna("")


def chain_ca_residues(path: Path, chain_id: str) -> list[ResidueCA]:
    atoms = parse_atoms(read_pdb_text(path))
    residues: dict[tuple[int, str], ResidueCA] = {}
    for atom in atoms:
        if atom.chain_id != chain_id or atom.atom_name != "CA" or atom.residue_name not in AA3_TO_1:
            continue
        key = (atom.residue_number, atom.insertion_code)
        residues[key] = ResidueCA(
            residue_number=atom.residue_number,
            insertion_code=atom.insertion_code,
            residue_name=atom.residue_name,
            aa=AA3_TO_1.get(atom.residue_name, "X"),
            coord=np.array([atom.x, atom.y, atom.z], dtype=float),
        )
    return [residues[key] for key in sorted(residues)]


def align_residues(native: list[ResidueCA], refold: list[ResidueCA]) -> tuple[list[ResidueCA], list[ResidueCA], str]:
    if len(native) == len(refold):
        return native, refold, "positional_equal_length"
    native_by_key = {(res.residue_number, res.insertion_code): res for res in native}
    refold_by_key = {(res.residue_number, res.insertion_code): res for res in refold}
    keys = sorted(set(native_by_key) & set(refold_by_key))
    if len(keys) >= 3:
        return [native_by_key[key] for key in keys], [refold_by_key[key] for key in keys], "residue_number"
    n = min(len(native), len(refold))
    return native[:n], refold[:n], "positional_truncated"


def kabsch_superpose(reference: np.ndarray, model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_center = reference.mean(axis=0)
    model_center = model.mean(axis=0)
    ref0 = reference - ref_center
    model0 = model - model_center
    covariance = model0.T @ ref0
    u, _s, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    transformed = model0 @ rotation + ref_center
    return transformed, rotation


def tm_d0(length: int) -> float:
    if length <= 0:
        return 0.5
    return max(0.5, 1.24 * math.pow(max(length - 15, 1), 1.0 / 3.0) - 1.8)


def compare_antigen_pair(row: pd.Series) -> dict[str, object]:
    structure_id = str(row.get("structure_id") or "")
    native_chain_id = str(row.get("native_antigen_chain") or row.get("antigen_chain") or "")
    refold_chain_id = str(row.get("refold_antigen_chain") or row.get("antigen_chain") or native_chain_id or "")
    display_chain = native_chain_id if native_chain_id == refold_chain_id else f"{native_chain_id}->{refold_chain_id}"
    native_path = Path(_path_text(row.get("native_relaxed_pdb"))).expanduser()
    refold_path = Path(_path_text(row.get("refold_relaxed_pdb"))).expanduser()
    base = {
        "structure_id": structure_id,
        "antigen_chain": display_chain,
        "native_antigen_chain": native_chain_id,
        "refold_antigen_chain": refold_chain_id,
        "native_relaxed_pdb": str(native_path),
        "refold_relaxed_pdb": str(refold_path),
    }
    try:
        if not native_path.is_file():
            raise FileNotFoundError(f"native relaxed PDB not found: {native_path}")
        if not refold_path.is_file():
            raise FileNotFoundError(f"refold relaxed PDB not found: {refold_path}")
        if not native_chain_id or not refold_chain_id:
            raise ValueError("native_antigen_chain and refold_antigen_chain are required or must be inferable from structure_id")
        native = chain_ca_residues(native_path, native_chain_id)
        refold = chain_ca_residues(refold_path, refold_chain_id)
        if len(native) < 3 or len(refold) < 3:
            raise ValueError(
                f"need at least 3 CA atoms; got native chain {native_chain_id}={len(native)} "
                f"refold chain {refold_chain_id}={len(refold)}"
            )
        native_aligned, refold_aligned, alignment_method = align_residues(native, refold)
        if len(native_aligned) < 3:
            raise ValueError(
                f"aligned fewer than 3 CA atoms for native chain {native_chain_id} "
                f"and refold chain {refold_chain_id}"
            )
        reference = np.vstack([res.coord for res in native_aligned])
        model = np.vstack([res.coord for res in refold_aligned])
        transformed, _rotation = kabsch_superpose(reference, model)
        distances = np.linalg.norm(reference - transformed, axis=1)
        rmsd = float(np.sqrt(np.mean(np.square(distances))))
        d0 = tm_d0(len(native))
        tm_score = float(np.sum(1.0 / (1.0 + np.square(distances / d0))) / max(len(native), 1))
        sequence_matches = sum(a.aa == b.aa for a, b in zip(native_aligned, refold_aligned))
        return {
            **base,
            "similarity_status": "ok",
            "similarity_error": "",
            "alignment_method": alignment_method,
            "native_antigen_ca_residues": len(native),
            "refold_antigen_ca_residues": len(refold),
            "aligned_ca_residues": len(native_aligned),
            "aligned_fraction": float(len(native_aligned) / max(len(native), 1)),
            "antigen_ca_rmsd": rmsd,
            "antigen_ca_tm_score": tm_score,
            "percent_structural_similarity": tm_score * 100.0,
            "antigen_ca_sequence_identity": float(sequence_matches / max(len(native_aligned), 1)),
        }
    except Exception as exc:  # noqa: BLE001 - rows preserve explicit failures.
        return {
            **base,
            "similarity_status": "failed",
            "similarity_error": str(exc),
            "alignment_method": "",
            "native_antigen_ca_residues": pd.NA,
            "refold_antigen_ca_residues": pd.NA,
            "aligned_ca_residues": pd.NA,
            "aligned_fraction": pd.NA,
            "antigen_ca_rmsd": pd.NA,
            "antigen_ca_tm_score": pd.NA,
            "percent_structural_similarity": pd.NA,
            "antigen_ca_sequence_identity": pd.NA,
        }


def _load_metric_table(path: Path | None, label: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = _require_unique(pd.read_csv(path).fillna(""), label)
    keep = ["structure_id", *[metric for metric in ROSETTA_METRICS if metric in df.columns]]
    return df[keep].copy()


def merge_metric_tables(similarity: pd.DataFrame, native_metrics: Path | None, refold_metrics: Path | None) -> pd.DataFrame:
    out = similarity.copy()
    native = _load_metric_table(native_metrics, "native metrics")
    refold = _load_metric_table(refold_metrics, "refold metrics")
    if not native.empty:
        out = out.merge(native, on="structure_id", how="left", validate="one_to_one")
        out = out.rename(columns={metric: f"native_{metric}" for metric in ROSETTA_METRICS if metric in out.columns})
    if not refold.empty:
        out = out.merge(refold, on="structure_id", how="left", validate="one_to_one")
        out = out.rename(columns={metric: f"refold_{metric}" for metric in ROSETTA_METRICS if metric in out.columns})
    for metric in ROSETTA_METRICS:
        native_col = f"native_{metric}"
        refold_col = f"refold_{metric}"
        if native_col in out.columns and refold_col in out.columns:
            out[native_col] = pd.to_numeric(out[native_col], errors="coerce")
            out[refold_col] = pd.to_numeric(out[refold_col], errors="coerce")
            out[f"delta_{safe_metric_name(metric)}"] = out[refold_col] - out[native_col]
    return out


def safe_metric_name(metric: str) -> str:
    return metric.replace("/", "_per_")


def metric_long_table(table: pd.DataFrame, refold_label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ROSETTA_METRICS:
        native_col = f"native_{metric}"
        refold_col = f"refold_{metric}"
        if native_col not in table.columns or refold_col not in table.columns:
            continue
        for _, row in table.iterrows():
            rows.append({"structure_id": row["structure_id"], "metric": metric, "metric_label": METRIC_LABELS.get(metric, metric), "state": "native", "value": row[native_col]})
            rows.append({"structure_id": row["structure_id"], "metric": metric, "metric_label": METRIC_LABELS.get(metric, metric), "state": refold_label, "value": row[refold_col]})
    long = pd.DataFrame(rows)
    if not long.empty:
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ["percent_structural_similarity", "antigen_ca_rmsd", "aligned_fraction", "antigen_ca_sequence_identity"]:
        values = pd.to_numeric(table[metric], errors="coerce").dropna()
        rows.append(summary_row(metric, metric, values))
    for metric in ROSETTA_METRICS:
        native_col = f"native_{metric}"
        refold_col = f"refold_{metric}"
        delta_col = f"delta_{safe_metric_name(metric)}"
        if native_col in table.columns and refold_col in table.columns:
            native_values = pd.to_numeric(table[native_col], errors="coerce")
            refold_values = pd.to_numeric(table[refold_col], errors="coerce")
            delta = pd.to_numeric(table.get(delta_col), errors="coerce")
            paired = pd.DataFrame({"native": native_values, "refold": refold_values, "delta": delta}).dropna()
            rows.append(
                {
                    **summary_row(metric, METRIC_LABELS.get(metric, metric), paired["delta"] if not paired.empty else pd.Series(dtype=float)),
                    "native_mean": float(paired["native"].mean()) if not paired.empty else pd.NA,
                    "refold_mean": float(paired["refold"].mean()) if not paired.empty else pd.NA,
                    "native_median": float(paired["native"].median()) if not paired.empty else pd.NA,
                    "refold_median": float(paired["refold"].median()) if not paired.empty else pd.NA,
                }
            )
    return pd.DataFrame(rows)


def summary_row(metric: str, label: str, values: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"metric": metric, "label": label, "n": 0, "mean": pd.NA, "median": pd.NA, "p10": pd.NA, "p90": pd.NA}
    return {
        "metric": metric,
        "label": label,
        "n": int(values.shape[0]),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
    }


def robust_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return 0.0, 1.0
    lo = float(finite.quantile(0.01))
    hi = float(finite.quantile(0.99))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo else 1.0
    else:
        pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def plot_similarity(table: pd.DataFrame, path: Path) -> None:
    ok = table[table["similarity_status"] == "ok"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    if ok.empty:
        ax.text(0.5, 0.5, "No successful similarity rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.histplot(ok["percent_structural_similarity"], bins=30, color="#4c78a8", edgecolor="white", linewidth=0.3, ax=ax)
        ax.axvline(ok["percent_structural_similarity"].median(), color="#c0392b", linewidth=1.2, label="median")
        ax.set_xlabel("Antigen structural similarity (%)")
        ax.set_ylabel("Structures")
        ax.set_title("Native vs Folded-Native Antigen Similarity")
        ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_metric_pairs(long: pd.DataFrame, path: Path) -> None:
    if long.empty or long["value"].dropna().empty:
        return
    labels = list(dict.fromkeys(long["metric_label"].astype(str)))
    ncols = 3
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4, max(3.0, 2.6 * nrows)), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    palette = {"native": "#4c78a8"}
    states = list(dict.fromkeys(long["state"].astype(str)))
    for state in states:
        palette.setdefault(state, "#c0392b")
    for ax, label in zip(axes_array, labels):
        subset = long[long["metric_label"] == label]
        sns.boxplot(data=subset, x="state", y="value", hue="state", palette=palette, dodge=False, fliersize=1.5, linewidth=0.8, ax=ax)
        sns.stripplot(data=subset, x="state", y="value", color="0.15", size=1.5, alpha=0.35, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        if ax.legend_:
            ax.legend_.remove()
    for ax in axes_array[len(labels):]:
        ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_metric_scatter(table: pd.DataFrame, path: Path) -> None:
    metrics = [metric for metric in ROSETTA_METRICS if f"native_{metric}" in table.columns and f"refold_{metric}" in table.columns]
    if not metrics:
        return
    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4, max(3.0, 2.6 * nrows)), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes_array, metrics):
        x = pd.to_numeric(table[f"native_{metric}"], errors="coerce")
        y = pd.to_numeric(table[f"refold_{metric}"], errors="coerce")
        sns.scatterplot(x=x, y=y, s=12, linewidth=0, color="#2f6f8f", alpha=0.65, ax=ax)
        lo, hi = robust_limits(pd.concat([x, y]))
        ax.plot([lo, hi], [lo, hi], color="0.35", linewidth=0.8, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Native relaxed")
        ax.set_ylabel("Folded-native relaxed")
    for ax in axes_array[len(metrics):]:
        ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_publication_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    pairs = build_pair_table(args)
    similarity = pd.DataFrame([compare_antigen_pair(row) for _, row in pairs.iterrows()])
    table = merge_metric_tables(similarity, args.native_metrics, args.refold_metrics)
    summary = summarize(table)
    long = metric_long_table(table, args.refold_label)

    table.to_csv(args.out_dir / "native_refold_similarity.csv", index=False)
    summary.to_csv(args.out_dir / "native_refold_metric_summary.csv", index=False)
    long.to_csv(args.out_dir / "native_refold_metric_long.csv", index=False)
    plot_similarity(table, figures_dir / "native_refold_similarity_distribution.png")
    plot_metric_pairs(long, figures_dir / "native_refold_metric_pairs.png")
    plot_metric_scatter(table, figures_dir / "native_refold_metric_scatter.png")
    return table, summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table, _summary = run(args)
    counts = table["similarity_status"].value_counts(dropna=False).to_dict() if not table.empty else {}
    print(f"wrote native-refold comparison: {len(table)} rows to {args.out_dir}")
    print(f"similarity statuses: {counts}")
    return 0 if not table.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
