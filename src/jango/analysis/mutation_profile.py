from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu

try:
    from Bio.Align import substitution_matrices
except Exception:  # pragma: no cover
    substitution_matrices = None


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
AA_CLASS = {
    "A": "hydrophobic",
    "V": "hydrophobic",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "M": "hydrophobic",
    "F": "aromatic",
    "W": "aromatic",
    "Y": "aromatic",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "C": "polar",
    "G": "special",
    "P": "special",
    "D": "negative",
    "E": "negative",
    "K": "positive",
    "R": "positive",
    "H": "positive",
}
AA_CHARGE = {"D": "negative", "E": "negative", "K": "positive", "R": "positive", "H": "positive"}
HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "W", "Y"}
POLAR_OR_CHARGED = {"S", "T", "N", "Q", "C", "D", "E", "K", "R", "H"}

DEFAULT_DATASETS = {
    "esmfold2_hotspot": (
        "data/outputs/evaluation/reference_benchmark/reference_refold/hotspot/esmfold2/tables/decoy_ranked_ml_dataset.csv",
        "esmfold2",
        "hotspot",
    ),
    "esmfold2_interface": (
        "data/outputs/evaluation/reference_benchmark/reference_refold/interface/esmfold2/tables/decoy_ranked_ml_dataset.csv",
        "esmfold2",
        "interface",
    ),
    "opendde_hotspot": (
        "data/outputs/evaluation/reference_benchmark/reference_refold/hotspot/opendde/tables/decoy_ranked_ml_dataset.csv",
        "opendde",
        "hotspot",
    ),
    "opendde_interface": (
        "data/outputs/evaluation/reference_benchmark/reference_refold/interface/opendde/tables/decoy_ranked_ml_dataset.csv",
        "opendde",
        "interface",
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: Path
    backend: str
    redesign_mode: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile antigen mutations enriched in high-quality Jango decoys.")
    parser.add_argument("--output-root", type=Path, default=Path("data/outputs/evaluation/mutation_profile"))
    parser.add_argument("--dataset", action="append", default=[], metavar="KEY=CSV,BACKEND,MODE")
    parser.add_argument("--dockq-threshold", type=float, default=0.49)
    parser.add_argument("--delta-dg-threshold", type=float, default=0.0)
    parser.add_argument("--tm-threshold", type=float, default=90.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--top-enrichment", type=int, default=24)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args(argv)


def dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    if args.dataset:
        for raw in args.dataset:
            key, sep, rest = raw.partition("=")
            if not sep:
                raise ValueError(f"Expected KEY=CSV,BACKEND,MODE: {raw}")
            parts = [p.strip() for p in rest.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Expected KEY=CSV,BACKEND,MODE: {raw}")
            specs.append(DatasetSpec(key.strip(), Path(parts[0]), parts[1], parts[2]))
        return specs
    return [DatasetSpec(key, Path(path), backend, mode) for key, (path, backend, mode) in DEFAULT_DATASETS.items()]


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


def clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(ch for ch in str(value).strip().upper() if ch.isalpha())


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def load_blosum62():
    if substitution_matrices is None:
        return None
    try:
        return substitution_matrices.load("BLOSUM62")
    except Exception:
        return None


def blosum_score(matrix, native_aa: str, mutant_aa: str) -> float:
    if matrix is None or native_aa not in AA_ORDER or mutant_aa not in AA_ORDER:
        return math.nan
    try:
        return float(matrix[native_aa, mutant_aa])
    except Exception:
        try:
            return float(matrix[mutant_aa, native_aa])
        except Exception:
            return math.nan


def aa_class(aa: str) -> str:
    return AA_CLASS.get(aa, "unknown")


def aa_charge(aa: str) -> str:
    return AA_CHARGE.get(aa, "neutral")


def aa_polarity(aa: str) -> str:
    if aa in HYDROPHOBIC:
        return "hydrophobic"
    if aa in POLAR_OR_CHARGED:
        return "polar_or_charged"
    return "unknown"


def blosum_class(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= 1:
        return "conservative"
    if score >= -1:
        return "neutral"
    return "disruptive"


def parse_interface_entries(value: object) -> list[dict[str, str]]:
    text = "" if pd.isna(value) else str(value).strip()
    entries: list[dict[str, str]] = []
    for raw in text.split(";"):
        parts = raw.split(":")
        if len(parts) < 3:
            continue
        chain, residue_number, aa3 = parts[0], parts[1], parts[2].upper()
        entries.append(
            {
                "chain": chain,
                "residue_number": residue_number,
                "aa3": aa3,
                "aa1": AA3_TO_1.get(aa3, "X"),
                "label": f"{chain}:{residue_number}:{aa3}",
            }
        )
    return entries


def parse_contact_counts(value: object) -> Counter:
    counts: Counter = Counter()
    text = "" if pd.isna(value) else str(value).strip()
    for pair in text.split(";"):
        if "--" not in pair:
            continue
        parts = pair.split("--", 1)[1].split(":")
        if len(parts) >= 3:
            counts[f"{parts[0]}:{parts[1]}:{parts[2].upper()}"] += 1
    return counts


def map_interface_to_sequence(native_sequence: str, entries: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    mapped: dict[int, dict[str, str]] = {}
    start = 0
    for entry in entries:
        aa = entry.get("aa1", "X")
        if aa == "X":
            continue
        idx = native_sequence.find(aa, start)
        if idx < 0:
            idx = native_sequence.find(aa)
        if idx < 0:
            continue
        seq_pos = idx + 1
        start = idx + 1
        info = dict(entry)
        info["sequence_position"] = str(seq_pos)
        mapped[seq_pos] = info
    return mapped


def dist_to_interface(position: int, interface_positions: set[int]) -> float:
    if not interface_positions:
        return math.nan
    return float(min(abs(position - pos) for pos in interface_positions))


def distance_bin(distance: float) -> str:
    if pd.isna(distance):
        return "no_interface_map"
    if distance == 0:
        return "interface"
    if distance <= 3:
        return "near_interface_1_3"
    if distance <= 10:
        return "proximal_4_10"
    return "distal_gt_10"


def position_decile(frac: float) -> str:
    if pd.isna(frac):
        return "unknown"
    lo = (max(0, min(99, int(frac * 100))) // 10) * 10
    return f"{lo:02d}-{lo + 10:02d}%"


def annotate_quality(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["dockq"] = pd.to_numeric(out["dockq"], errors="coerce")
    out["delta_dg"] = pd.to_numeric(out["delta_dg"], errors="coerce")
    out["native_refold_tm_percent"] = pd.to_numeric(out["native_refold_tm_percent"], errors="coerce")
    high_high = out["high_high"].map(truthy) if "high_high" in out.columns else (
        out["dockq"].ge(args.dockq_threshold) & out["delta_dg"].ge(args.delta_dg_threshold)
    )
    tm90 = out["native_refold_tm_percent"].ge(args.tm_threshold)
    out["q1_high_high"] = high_high.astype(bool)
    out["tm90_pass"] = tm90.astype(bool)
    out["q1_tm90"] = (high_high & tm90).astype(bool)
    out["quality_group"] = np.where(out["q1_tm90"], "Q1_TM90", "background")
    return out


def load_inputs(specs: list[DatasetSpec], args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for spec in specs:
        path = spec.path if spec.path.is_absolute() else Path.cwd() / spec.path
        if not path.exists():
            raise FileNotFoundError(f"missing dataset {spec.key}: {path}")
        df = pd.read_csv(path)
        require_columns(
            df,
            ["structure_id", "design_id", "native_antigen_sequence", "designed_sequence", "dockq", "delta_dg", "native_refold_tm_percent"],
            spec.key,
        )
        df = annotate_quality(df, args)
        df["dataset_key"] = spec.key
        df["backend"] = spec.backend
        df["redesign_mode"] = spec.redesign_mode
        df["source_table"] = str(path)
        frames.append(df)
    return backfill_native_contact_annotations(pd.concat(frames, ignore_index=True, sort=False))


def backfill_native_contact_annotations(combined: pd.DataFrame) -> pd.DataFrame:
    """Fill missing native interface annotations from sibling datasets.

    Some legacy OpenDDE tables were generated before native-contact columns were
    carried through every final ML table. The annotations are native-structure
    properties keyed by structure_id, so using the populated sibling backend/mode
    row prevents false zeros in interface-mutation summaries.
    """

    out = combined.copy()
    annotation_cols = [
        "native_antigen_interface_positions",
        "native_contact_residue_pairs",
        "native_antigen_interface_sequence",
        "native_antigen_interface_residues",
        "native_antigen_chain",
    ]
    for col in annotation_cols:
        if col not in out.columns:
            out[col] = pd.NA
    for col in annotation_cols:
        valid = out.loc[out[col].notna() & out[col].astype(str).str.strip().ne(""), ["structure_id", col]]
        if valid.empty:
            continue
        lookup = valid.drop_duplicates("structure_id").set_index("structure_id")[col]
        missing = out[col].isna() | out[col].astype(str).str.strip().eq("")
        out.loc[missing, col] = out.loc[missing, "structure_id"].map(lookup)
    return out


def mutation_rows_for_decoy(row: pd.Series, matrix) -> list[dict[str, object]]:
    native = clean_sequence(row.get("native_antigen_sequence"))
    designed = clean_sequence(row.get("designed_sequence"))
    if not native or not designed or len(native) != len(designed):
        return []
    interface_map = map_interface_to_sequence(native, parse_interface_entries(row.get("native_antigen_interface_positions")))
    interface_positions = set(interface_map)
    contact_counts = parse_contact_counts(row.get("native_contact_residue_pairs"))
    seq_contact_counts = {pos: int(contact_counts.get(info["label"], 0)) for pos, info in interface_map.items()}
    rows = []
    for pos, (native_aa, mutant_aa) in enumerate(zip(native, designed), start=1):
        if native_aa == mutant_aa:
            continue
        frac = pos / len(native)
        distance = dist_to_interface(pos, interface_positions)
        bscore = blosum_score(matrix, native_aa, mutant_aa)
        native_cls, mutant_cls = aa_class(native_aa), aa_class(mutant_aa)
        native_charge, mutant_charge = aa_charge(native_aa), aa_charge(mutant_aa)
        native_polarity, mutant_polarity = aa_polarity(native_aa), aa_polarity(mutant_aa)
        interface_info = interface_map.get(pos, {})
        rows.append(
            {
                "dataset_key": row["dataset_key"],
                "backend": row["backend"],
                "redesign_mode": row["redesign_mode"],
                "structure_id": row["structure_id"],
                "design_id": row["design_id"],
                "sequence_position": pos,
                "position_fraction": frac,
                "position_decile": position_decile(frac),
                "native_aa": native_aa,
                "mutant_aa": mutant_aa,
                "mutation_label": f"{native_aa}{pos}{mutant_aa}",
                "native_class": native_cls,
                "mutant_class": mutant_cls,
                "class_transition": f"{native_cls}->{mutant_cls}",
                "native_charge": native_charge,
                "mutant_charge": mutant_charge,
                "charge_transition": f"{native_charge}->{mutant_charge}",
                "charge_changed": native_charge != mutant_charge,
                "native_polarity": native_polarity,
                "mutant_polarity": mutant_polarity,
                "polarity_transition": f"{native_polarity}->{mutant_polarity}",
                "polarity_changed": native_polarity != mutant_polarity,
                "blosum62": bscore,
                "blosum_class": blosum_class(bscore),
                "gly_or_pro_involved": native_aa in {"G", "P"} or mutant_aa in {"G", "P"},
                "aromatic_involved": native_aa in {"F", "W", "Y"} or mutant_aa in {"F", "W", "Y"},
                "is_interface_residue": pos in interface_positions,
                "sequence_distance_to_nearest_interface_position": distance,
                "interface_distance_bin": distance_bin(distance),
                "interface_contact_count": seq_contact_counts.get(pos, 0),
                "native_antigen_residue_label": interface_info.get("label", ""),
                "native_antigen_chain": interface_info.get("chain", row.get("native_antigen_chain", "")),
                "native_antigen_residue_number": interface_info.get("residue_number", ""),
                "dockq": row.get("dockq"),
                "delta_dg": row.get("delta_dg"),
                "native_refold_tm_percent": row.get("native_refold_tm_percent"),
                "q1_high_high": bool(row.get("q1_high_high")),
                "tm90_pass": bool(row.get("tm90_pass")),
                "q1_tm90": bool(row.get("q1_tm90")),
                "quality_group": row.get("quality_group"),
            }
        )
    return rows


def summarize_decoys(combined: pd.DataFrame, mutations: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset_key", "backend", "redesign_mode", "structure_id", "design_id"]
    base_cols = [
        *key_cols,
        "dockq",
        "delta_dg",
        "native_refold_tm_percent",
        "q1_high_high",
        "tm90_pass",
        "q1_tm90",
        "quality_group",
        "native_antigen_sequence",
        "designed_sequence",
    ]
    out = combined[base_cols].copy()
    out["native_antigen_length"] = out["native_antigen_sequence"].map(lambda x: len(clean_sequence(x)))
    out["designed_antigen_length"] = out["designed_sequence"].map(lambda x: len(clean_sequence(x)))
    out["sequence_lengths_match"] = out["native_antigen_length"].eq(out["designed_antigen_length"])
    if mutations.empty:
        grouped = pd.DataFrame(columns=key_cols)
    else:
        grouped = (
            mutations.groupby(key_cols, dropna=False)
            .agg(
                n_residues_mutated=("mutation_label", "count"),
                interface_mutation_count=("is_interface_residue", "sum"),
                near_interface_mutation_count=("interface_distance_bin", lambda s: s.isin(["interface", "near_interface_1_3"]).sum()),
                charge_change_count=("charge_changed", "sum"),
                polarity_change_count=("polarity_changed", "sum"),
                gly_or_pro_mutation_count=("gly_or_pro_involved", "sum"),
                aromatic_mutation_count=("aromatic_involved", "sum"),
                mean_sequence_distance_to_interface=("sequence_distance_to_nearest_interface_position", "mean"),
                median_sequence_distance_to_interface=("sequence_distance_to_nearest_interface_position", "median"),
                mean_blosum62=("blosum62", "mean"),
                disruptive_blosum_count=("blosum_class", lambda s: (s == "disruptive").sum()),
            )
            .reset_index()
        )
    out = out.merge(grouped, on=key_cols, how="left")
    zero_cols = [
        "n_residues_mutated",
        "interface_mutation_count",
        "near_interface_mutation_count",
        "charge_change_count",
        "polarity_change_count",
        "gly_or_pro_mutation_count",
        "aromatic_mutation_count",
        "disruptive_blosum_count",
    ]
    for col in zero_cols:
        if col not in out.columns:
            out[col] = 0
    out[zero_cols] = out[zero_cols].fillna(0).astype(int)
    out["mutation_fraction"] = np.where(out["native_antigen_length"].gt(0), out["n_residues_mutated"] / out["native_antigen_length"], np.nan)
    out["interface_mutation_fraction"] = np.where(out["n_residues_mutated"].gt(0), out["interface_mutation_count"] / out["n_residues_mutated"], 0.0)
    return out


def bh_qvalues(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        val = min(prev, p_values[idx] * n / (n - rank + 1))
        q[idx] = val
        prev = val
    return q.tolist()


def enrichment_for_feature(mutations: pd.DataFrame, feature: str) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in mutations.groupby("dataset_key", dropna=False):
        high = sub["q1_tm90"].astype(bool)
        for category, cat_sub in sub.groupby(feature, dropna=False):
            category = "missing" if pd.isna(category) else str(category)
            a = int(cat_sub["q1_tm90"].astype(bool).sum())
            b = int(len(cat_sub) - a)
            c = int(high.sum() - a)
            d = int((~high).sum() - b)
            odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "backend": str(sub["backend"].iloc[0]),
                    "redesign_mode": str(sub["redesign_mode"].iloc[0]),
                    "feature": feature,
                    "category": category,
                    "q1_tm90_mutation_count": a,
                    "background_mutation_count": b,
                    "q1_tm90_frequency": a / max(int(high.sum()), 1),
                    "background_frequency": b / max(int((~high).sum()), 1),
                    "odds_ratio": odds,
                    "p_value_fisher_exact": p,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh"] = np.nan
    for dataset_key, idx in out.groupby("dataset_key").groups.items():
        out.loc[idx, "q_value_bh"] = bh_qvalues(out.loc[idx, "p_value_fisher_exact"].fillna(1.0).tolist())
    out["log2_odds_ratio"] = out["odds_ratio"].replace([np.inf, -np.inf], np.nan).map(lambda x: np.log2(x) if pd.notna(x) and x > 0 else np.nan)
    return out.sort_values(["q_value_bh", "p_value_fisher_exact", "dataset_key", "feature", "category"], kind="mergesort")


def build_enrichment(mutations: pd.DataFrame) -> pd.DataFrame:
    features = [
        "is_interface_residue",
        "interface_distance_bin",
        "position_decile",
        "native_aa",
        "mutant_aa",
        "native_class",
        "mutant_class",
        "class_transition",
        "charge_transition",
        "polarity_transition",
        "blosum_class",
        "gly_or_pro_involved",
        "aromatic_involved",
    ]
    parts = [enrichment_for_feature(mutations, feature) for feature in features if feature in mutations.columns]
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def position_frequency(mutations: pd.DataFrame, decoys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    totals = decoys.groupby("dataset_key").agg(n_decoys=("design_id", "nunique"), n_q1_tm90_decoys=("q1_tm90", "sum"))
    for (dataset_key, pos), sub in mutations.groupby(["dataset_key", "sequence_position"], dropna=False):
        total = totals.loc[dataset_key]
        q1_designs = set(sub.loc[sub["q1_tm90"].astype(bool), "design_id"].astype(str))
        bg_designs = set(sub.loc[~sub["q1_tm90"].astype(bool), "design_id"].astype(str))
        rows.append(
            {
                "dataset_key": dataset_key,
                "backend": str(sub["backend"].iloc[0]),
                "redesign_mode": str(sub["redesign_mode"].iloc[0]),
                "sequence_position": int(pos),
                "position_fraction_median": float(sub["position_fraction"].median()),
                "native_aa_mode": sub["native_aa"].mode().iloc[0] if not sub["native_aa"].mode().empty else "",
                "q1_tm90_decoys_with_mutation": len(q1_designs),
                "background_decoys_with_mutation": len(bg_designs),
                "n_q1_tm90_decoys": int(total["n_q1_tm90_decoys"]),
                "n_decoys": int(total["n_decoys"]),
                "q1_tm90_position_frequency": len(q1_designs) / max(int(total["n_q1_tm90_decoys"]), 1),
                "overall_position_frequency": sub["design_id"].nunique() / max(int(total["n_decoys"]), 1),
                "interface_mutation_count": int(sub["is_interface_residue"].sum()),
                "mean_sequence_distance_to_interface": float(sub["sequence_distance_to_nearest_interface_position"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "sequence_position"], kind="mergesort")


def substitution_matrix(mutations: pd.DataFrame) -> pd.DataFrame:
    return (
        mutations.groupby(["dataset_key", "backend", "redesign_mode", "quality_group", "native_aa", "mutant_aa"], dropna=False)
        .size()
        .rename("mutation_count")
        .reset_index()
        .sort_values(["dataset_key", "quality_group", "native_aa", "mutant_aa"], kind="mergesort")
    )


def summary_stats(decoys: pd.DataFrame, mutations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, sub in decoys.groupby("dataset_key", dropna=False):
        high = sub[sub["q1_tm90"]]
        mut = mutations[mutations["dataset_key"].eq(dataset_key)]
        high_mut = mut[mut["q1_tm90"]]
        rows.append(
            {
                "dataset_key": dataset_key,
                "backend": str(sub["backend"].iloc[0]),
                "redesign_mode": str(sub["redesign_mode"].iloc[0]),
                "n_decoys": len(sub),
                "n_q1_high_high": int(sub["q1_high_high"].sum()),
                "n_tm90": int(sub["tm90_pass"].sum()),
                "n_q1_tm90": len(high),
                "q1_tm90_fraction": len(high) / max(len(sub), 1),
                "total_mutations": len(mut),
                "q1_tm90_mutations": len(high_mut),
                "median_mutations_per_decoy": float(sub["n_residues_mutated"].median()),
                "q1_tm90_median_mutations_per_decoy": float(high["n_residues_mutated"].median()) if len(high) else np.nan,
                "median_interface_mutation_fraction": float(sub["interface_mutation_fraction"].median()),
                "q1_tm90_median_interface_mutation_fraction": float(high["interface_mutation_fraction"].median()) if len(high) else np.nan,
                "q1_tm90_median_dockq": float(high["dockq"].median()) if len(high) else np.nan,
                "q1_tm90_median_delta_dg": float(high["delta_dg"].median()) if len(high) else np.nan,
                "q1_tm90_median_native_refold_tm": float(high["native_refold_tm_percent"].median()) if len(high) else np.nan,
                "q1_tm90_median_blosum62": float(high_mut["blosum62"].median()) if len(high_mut) else np.nan,
                "q1_tm90_interface_mutation_fraction": float(high_mut["is_interface_residue"].mean()) if len(high_mut) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["backend", "redesign_mode"], kind="mergesort")


def decoy_feature_tests(decoys: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "dockq",
        "delta_dg",
        "native_refold_tm_percent",
        "n_residues_mutated",
        "mutation_fraction",
        "interface_mutation_count",
        "interface_mutation_fraction",
        "mean_sequence_distance_to_interface",
        "mean_blosum62",
    ]
    rows = []
    for dataset_key, sub in decoys.groupby("dataset_key", dropna=False):
        high, bg = sub[sub["q1_tm90"]], sub[~sub["q1_tm90"]]
        for metric in metrics:
            h = pd.to_numeric(high.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            b = pd.to_numeric(bg.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            p = float(mannwhitneyu(h, b, alternative="two-sided").pvalue) if len(h) and len(b) else np.nan
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "backend": str(sub["backend"].iloc[0]),
                    "redesign_mode": str(sub["redesign_mode"].iloc[0]),
                    "metric": metric,
                    "q1_tm90_n": len(h),
                    "background_n": len(b),
                    "q1_tm90_median": float(h.median()) if len(h) else np.nan,
                    "background_median": float(b.median()) if len(b) else np.nan,
                    "median_difference_q1_minus_background": float(h.median() - b.median()) if len(h) and len(b) else np.nan,
                    "p_value_mann_whitney": p,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh"] = np.nan
    for dataset_key, idx in out.groupby("dataset_key").groups.items():
        out.loc[idx, "q_value_bh"] = bh_qvalues(out.loc[idx, "p_value_mann_whitney"].fillna(1.0).tolist())
    return out.sort_values(["q_value_bh", "dataset_key", "metric"], kind="mergesort")


def build_tables(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matrix = load_blosum62()
    rows: list[dict[str, object]] = []
    skipped_columns = [
        "dataset_key",
        "backend",
        "redesign_mode",
        "structure_id",
        "design_id",
        "skip_reason",
        "native_length",
        "designed_length",
    ]
    skipped = []
    for row in combined.itertuples(index=False):
        series = pd.Series(row._asdict())
        native, designed = clean_sequence(series.get("native_antigen_sequence")), clean_sequence(series.get("designed_sequence"))
        if not native or not designed or len(native) != len(designed):
            skipped.append(
                {
                    "dataset_key": series.get("dataset_key"),
                    "backend": series.get("backend"),
                    "redesign_mode": series.get("redesign_mode"),
                    "structure_id": series.get("structure_id"),
                    "design_id": series.get("design_id"),
                    "skip_reason": "missing_or_length_mismatched_sequence",
                    "native_length": len(native),
                    "designed_length": len(designed),
                }
            )
            continue
        rows.extend(mutation_rows_for_decoy(series, matrix))
    mutations = pd.DataFrame(rows)
    decoys = summarize_decoys(combined, mutations)
    return {
        "mutation_profile_per_mutation": mutations.sort_values(["dataset_key", "structure_id", "design_id", "sequence_position"], kind="mergesort"),
        "mutation_profile_per_decoy": decoys.sort_values(["dataset_key", "quality_group", "structure_id", "design_id"], kind="mergesort"),
        "mutation_enrichment_summary": build_enrichment(mutations),
        "mutation_position_frequency": position_frequency(mutations, decoys),
        "mutation_substitution_matrix": substitution_matrix(mutations),
        "mutation_profile_summary_stats": summary_stats(decoys, mutations),
        "mutation_decoy_feature_tests": decoy_feature_tests(decoys),
        "mutation_profile_skipped_decoys": pd.DataFrame(skipped, columns=skipped_columns),
    }


def save_fig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_global_figures(root: Path, tables: dict[str, pd.DataFrame], dpi: int, top_n: int) -> None:
    sns.set_theme(style="whitegrid")
    decoys, mutations = tables["mutation_profile_per_decoy"], tables["mutation_profile_per_mutation"]
    enrichment, matrix_table, pos = (
        tables["mutation_enrichment_summary"],
        tables["mutation_substitution_matrix"],
        tables["mutation_position_frequency"],
    )
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=decoys, x="dataset_key", y="n_residues_mutated", hue="quality_group", showfliers=False)
    sns.stripplot(data=decoys, x="dataset_key", y="n_residues_mutated", hue="quality_group", dodge=True, alpha=0.22, size=2, legend=False)
    plt.title("Mutation Counts In High-Quality Decoys", color="#b22222", fontweight="bold")
    plt.xlabel("Dataset")
    plt.ylabel("Mutated antigen residues")
    plt.xticks(rotation=25, ha="right")
    save_fig(root / "figures" / "mutation_count_distribution.png", dpi)

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=decoys, x="dataset_key", y="interface_mutation_fraction", hue="quality_group", showfliers=False)
    plt.title("Interface Mutation Fraction", color="#b22222", fontweight="bold")
    plt.xlabel("Dataset")
    plt.ylabel("Fraction of mutations at native interface")
    plt.xticks(rotation=25, ha="right")
    save_fig(root / "figures" / "interface_mutation_fraction_distribution.png", dpi)

    dist = mutations.dropna(subset=["sequence_distance_to_nearest_interface_position"]).copy()
    dist = dist[dist["sequence_distance_to_nearest_interface_position"] <= 60]
    if not dist.empty:
        g = sns.displot(
            data=dist,
            x="sequence_distance_to_nearest_interface_position",
            hue="quality_group",
            col="dataset_key",
            col_wrap=2,
            bins=30,
            height=3.2,
            aspect=1.35,
            facet_kws={"sharey": False},
        )
        g.fig.suptitle("Mutation Distance To Native Interface", color="#b22222", fontweight="bold", y=1.02)
        g.set_axis_labels("Sequence distance to nearest interface residue", "Mutation count")
        g.savefig(root / "figures" / "interface_distance_distribution.png", dpi=dpi, bbox_inches="tight")
        plt.close(g.fig)

    q1_matrix = matrix_table[matrix_table["quality_group"].eq("Q1_TM90")]
    if not q1_matrix.empty:
        pivot = q1_matrix.pivot_table(index="native_aa", columns="mutant_aa", values="mutation_count", aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(index=AA_ORDER, columns=AA_ORDER, fill_value=0)
        plt.figure(figsize=(9, 7))
        sns.heatmap(pivot, cmap="viridis", linewidths=0.2, linecolor="white")
        plt.title("Q1 TM>90% Substitution Matrix", color="#b22222", fontweight="bold")
        plt.xlabel("Mutant amino acid")
        plt.ylabel("Native amino acid")
        save_fig(root / "figures" / "substitution_heatmap_q1_tm90.png", dpi)

    enrich_plot = enrichment[enrichment["feature"].isin(["class_transition", "blosum_class", "interface_distance_bin", "is_interface_residue"])].copy()
    enrich_plot = enrich_plot.dropna(subset=["log2_odds_ratio"])
    if not enrich_plot.empty:
        enrich_plot["abs_log2_or"] = enrich_plot["log2_odds_ratio"].abs()
        enrich_plot = enrich_plot.sort_values(["q_value_bh", "abs_log2_or"], ascending=[True, False]).head(top_n)
        enrich_plot["label"] = enrich_plot["dataset_key"] + " | " + enrich_plot["feature"] + "=" + enrich_plot["category"]
        plt.figure(figsize=(11, max(5, 0.32 * len(enrich_plot))))
        sns.barplot(data=enrich_plot, y="label", x="log2_odds_ratio", hue="redesign_mode", dodge=False)
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Mutation Feature Enrichment In Q1 TM>90% Decoys", color="#b22222", fontweight="bold")
        plt.xlabel("log2 odds ratio vs background mutations")
        plt.ylabel("")
        save_fig(root / "figures" / "mutation_feature_enrichment.png", dpi)

    if not pos.empty:
        plot_pos = pos.copy()
        plot_pos["position_percent"] = plot_pos["position_fraction_median"] * 100
        g = sns.relplot(
            data=plot_pos,
            x="position_percent",
            y="q1_tm90_position_frequency",
            col="dataset_key",
            col_wrap=2,
            kind="scatter",
            height=3.2,
            aspect=1.35,
            size="interface_mutation_count",
            sizes=(10, 90),
            hue="mean_sequence_distance_to_interface",
            palette="mako_r",
            facet_kws={"sharey": False},
        )
        g.fig.suptitle("Antigen Position Mutation Frequency", color="#b22222", fontweight="bold", y=1.02)
        g.set_axis_labels("Relative antigen position (%)", "Q1 TM>90% mutation frequency")
        g.savefig(root / "figures" / "mutation_position_frequency.png", dpi=dpi, bbox_inches="tight")
        plt.close(g.fig)

    summary = tables["mutation_profile_summary_stats"]
    melted = summary.melt(
        id_vars=["dataset_key", "backend", "redesign_mode"],
        value_vars=["q1_tm90_fraction", "q1_tm90_median_interface_mutation_fraction", "q1_tm90_median_mutations_per_decoy"],
        var_name="metric",
        value_name="value",
    )
    g = sns.catplot(data=melted, x="redesign_mode", y="value", hue="backend", col="metric", kind="bar", height=3.2, aspect=1.05, sharey=False)
    g.fig.suptitle("Backend And Redesign Mode Mutation Profile", color="#b22222", fontweight="bold", y=1.08)
    g.savefig(root / "figures" / "backend_mode_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def plot_per_dataset(root: Path, tables: dict[str, pd.DataFrame], dpi: int) -> None:
    decoys, mutations = tables["mutation_profile_per_decoy"], tables["mutation_profile_per_mutation"]
    matrix_table, pos = tables["mutation_substitution_matrix"], tables["mutation_position_frequency"]
    for dataset_key in sorted(decoys["dataset_key"].unique()):
        ds_root = root / "by_dataset" / dataset_key / "figures"
        ddec = decoys[decoys["dataset_key"].eq(dataset_key)]
        dmut = mutations[mutations["dataset_key"].eq(dataset_key)]
        dmat = matrix_table[matrix_table["dataset_key"].eq(dataset_key)]
        dpos = pos[pos["dataset_key"].eq(dataset_key)]
        plt.figure(figsize=(7, 4.5))
        sns.histplot(data=ddec, x="n_residues_mutated", hue="quality_group", bins=25, alpha=0.55)
        plt.title(f"{dataset_key} Mutation Counts", color="#b22222", fontweight="bold")
        plt.xlabel("Mutated antigen residues")
        plt.ylabel("Decoy count")
        save_fig(ds_root / "mutation_count_distribution.png", dpi)
        q1 = dmat[dmat["quality_group"].eq("Q1_TM90")]
        if not q1.empty:
            pivot = q1.pivot_table(index="native_aa", columns="mutant_aa", values="mutation_count", aggfunc="sum", fill_value=0)
            pivot = pivot.reindex(index=AA_ORDER, columns=AA_ORDER, fill_value=0)
            plt.figure(figsize=(7, 6))
            sns.heatmap(pivot, cmap="viridis", linewidths=0.2, linecolor="white")
            plt.title(f"{dataset_key} Q1 TM>90% Substitutions", color="#b22222", fontweight="bold")
            plt.xlabel("Mutant amino acid")
            plt.ylabel("Native amino acid")
            save_fig(ds_root / "substitution_heatmap_q1_tm90.png", dpi)
        if not dmut.empty:
            order = dmut[dmut["q1_tm90"]]["class_transition"].value_counts().head(15).index.tolist()
            plt.figure(figsize=(7, 4.5))
            sns.countplot(data=dmut[dmut["q1_tm90"]], y="class_transition", order=order, color="#4c78a8")
            plt.title(f"{dataset_key} Q1 TM>90% Mutation Classes", color="#b22222", fontweight="bold")
            plt.xlabel("Mutation count")
            plt.ylabel("Class transition")
            save_fig(ds_root / "mutation_class_counts_q1_tm90.png", dpi)
        if not dpos.empty:
            plt.figure(figsize=(7, 4.5))
            plt.scatter(dpos["position_fraction_median"] * 100, dpos["q1_tm90_position_frequency"], s=18, alpha=0.75)
            plt.title(f"{dataset_key} Position Frequency", color="#b22222", fontweight="bold")
            plt.xlabel("Relative antigen position (%)")
            plt.ylabel("Q1 TM>90% mutation frequency")
            plt.grid(color="#dddddd")
            save_fig(ds_root / "mutation_position_frequency.png", dpi)


def write_tables(root: Path, tables: dict[str, pd.DataFrame]) -> None:
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    for dataset_key in sorted(tables["mutation_profile_per_decoy"]["dataset_key"].unique()):
        ds_dir = root / "by_dataset" / dataset_key / "tables"
        ds_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            if "dataset_key" in table.columns:
                table[table["dataset_key"].astype(str).eq(str(dataset_key))].to_csv(ds_dir / f"{name}.csv", index=False)


def write_readme(root: Path, specs: list[DatasetSpec], args: argparse.Namespace, summary: pd.DataFrame) -> None:
    lines = [
        "# Jango Mutation Profile Analysis",
        "",
        "This package profiles antigen sequence mutations associated with high-quality Jango decoys.",
        "",
        "High-quality decoys are defined as:",
        f"- DockQ >= {args.dockq_threshold}",
        f"- delta_dG >= {args.delta_dg_threshold}",
        f"- native refold TM >= {args.tm_threshold}%",
        "",
        "Inputs:",
    ]
    lines.extend(f"- `{spec.key}`: `{spec.path}`" for spec in specs)
    lines.extend(
        [
            "",
            "Primary tables:",
            "- `tables/mutation_profile_per_mutation.csv`: one row per antigen mutation.",
            "- `tables/mutation_profile_per_decoy.csv`: one row per decoy with aggregate mutation features.",
            "- `tables/mutation_enrichment_summary.csv`: feature enrichment in Q1/TM90 mutations against background mutations.",
            "- `tables/mutation_position_frequency.csv`: antigen-position mutation frequencies.",
            "- `tables/mutation_substitution_matrix.csv`: native amino acid to mutant amino acid counts.",
            "- `tables/mutation_profile_summary_stats.csv`: dataset-level summary statistics.",
            "",
            "Dataset summary:",
            "",
            "```csv",
            summary.to_csv(index=False).strip(),
            "```",
            "",
            "Note: interface location is derived from native antigen contact annotations already present in the ML tables. PDB residue labels are mapped onto antigen sequence positions by residue-order matching.",
        ]
    )
    (root / "README.md").write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = dataset_specs(args)
    root = args.output_root if args.output_root.is_absolute() else Path.cwd() / args.output_root
    if args.clean_output and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    combined = load_inputs(specs, args)
    tables = build_tables(combined)
    write_tables(root, tables)
    plot_global_figures(root, tables, args.dpi, args.top_enrichment)
    plot_per_dataset(root, tables, args.dpi)
    write_readme(root, specs, args, tables["mutation_profile_summary_stats"])
    print(f"wrote mutation profile package: {root}")
    print(tables["mutation_profile_summary_stats"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

