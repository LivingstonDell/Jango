#!/usr/bin/env python
"""Analyze a completed DELPHI-VHH decoy-generation experiment.

This standalone script is analysis-only. It reads completed DELPHI-VHH
experiment tables from ``--input-dir`` and writes reusable CSV tables plus
publication-quality Matplotlib figures to ``--out-dir``. It does not run
Rosetta, ProteinMPNN, folding, grafting, validation, or any upstream pipeline
step.

Input tables
------------
Required inputs are discovered by filename inside ``--input-dir`` only:

* ``decoy_native_comparison.csv``: decoy/native interface metrics and backend
  columns to preserve in the landscape score table.
* ``decoy_redesign_case_manifest.base.411.csv``: one row per antigen structure,
  including ``structure_id``, ``antigen_length``, and native antigen sequence
  metadata.
* ``decoy_validation.csv`` or ``decoy_validation_esm411_all_modes.csv``:
  validation rows, selected-decoy flags, fold status, and monomer validation
  metrics when not already present in the comparison table.
* ``decoy_grafted_complex_manifest_all_modes.<CPU_NODE>.csv`` or
  ``decoy_grafted_complex_manifest_all_modes.csv``: graft metadata and
  ``monomer_graft_rmsd`` when not already present in the comparison table.
* ``auto_mode_assignments.csv``: selected redesign mode by structure when not
  available at decoy level.
* ``vhh_anarci_numbering_per_structure.csv``: optional precomputed canonical
  ANARCI/IMGT nanobody numbering used for mutation localization. If absent,
  the analysis tries to number explicit nanobody/VHH sequence columns with ANARCI.
* ``decoy_dockq.csv``: official DockQ scores comparing relaxed decoy complexes
  to corresponding relaxed native complexes. Required by the default
  ``--structure-preservation-metric dockq`` mode.

Output tables
-------------
The script writes exactly these CSV outputs under ``--out-dir/tables``:

* ``dataset_counts.csv``
* ``antigen_length_summary.csv``
* ``redesign_mode_counts.csv``
* ``mutation_count_summary.csv``
* ``mutation_summary_by_antigen.csv``
* ``amino_acid_substitution_counts.csv``
* ``amino_acid_substitution_frequencies.csv``
* ``component_metric_summary.csv``
* ``delphi_landscape_scores.csv``
* ``decoy_quadrant_ranking.csv``
* ``decoy_quadrant_summary.csv``
* ``decoy_quadrant_unclassified.csv``
* ``mutation_locations.csv``
* ``mutation_region_summary.csv``
* ``mutation_position_frequency.csv``
* ``mutation_region_enrichment.csv``
* ``mutation_localization_skipped_decoys.csv``
* ``mutation_numbering_status.csv``

Mutation-count definition
-------------------------
For every analyzed designed decoy with equal-length native and designed
antigen sequences, ``n_residues_mutated`` is the number of aligned positions
where ``designed_sequence[position] != native_antigen_sequence[position]``.
``mutation_fraction`` is ``n_residues_mutated / antigen_length``.

Amino-acid substitution matrix definition
-----------------------------------------
Rows are redesigned amino acids and columns are native amino acids, both in the
canonical order ``A C D E F G H I K L M N P Q R S T V W Y``. By default only
changed positions are counted. Passing ``--include-unchanged`` includes
unchanged diagonal pairs. No BLOSUM, PAM, or other external substitution model
is used; this is an observed ProteinMPNN redesign substitution count matrix.

Native-delta equations
----------------------
The following derived columns are computed as decoy minus native:

* ``delta_vs_native_dG_separated = dG_separated - native_dG_separated``
* ``delta_vs_native_hbonds_int = hbonds_int - native_hbonds_int``
* ``delta_vs_native_sc_value = sc_value - native_sc_value``
* ``delta_vs_native_delta_unsatHbonds =
  delta_unsatHbonds - native_delta_unsatHbonds``
* ``delta_vs_native_dSASA_int = dSASA_int - native_dSASA_int``

Z-score definition
------------------
For each component metric, ``z = (value - dataset mean) / population standard
deviation`` with ``ddof = 0``. For components where lower values are better,
the negated raw value is standardized. Zero-variance components abort with a
clear error.

DELPHI Landscape axes
---------------------
The canonical structure-preservation axis is official DockQ by default:
``structure_preservation = dockq``.

The legacy ``Structural Preservation Index`` is still written for continuity.
It is the unweighted mean of:

* ``z(monomer_tm_score)``
* ``z(-monomer_ca_rmsd)``
* ``z(-monomer_graft_rmsd)``

``binding_perturbation`` is the Rosetta-native-relative predicted binding
perturbation. It is the same value as ``Binding Perturbation Index``, the
unweighted mean of:

* ``z(delta_vs_native_dG_separated)``
* ``z(-delta_vs_native_hbonds_int)``
* ``z(-delta_vs_native_sc_value)``
* ``z(delta_vs_native_delta_unsatHbonds)``

``delta_vs_native_dSASA_int`` is diagnostic only and is not included in the
Binding Perturbation Index.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jango.analysis.thresholds import default_dockq_preservation_threshold, parse_dockq_preservation_threshold

try:  # Optional dependency.
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover - behavior is validated at runtime.
    gaussian_kde = None

from jango.analysis.mutation_localization import (
    compute_mutation_localization_outputs,
    save_mutation_position_frequency_plot,
    save_mutation_region_bar,
)


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
REDESIGN_MODE_ORDER = ["hotspot_only", "interface_only", "full_antigen"]

INPUT_CANDIDATES = {
    "comparison": ["decoy_native_comparison.csv"],
    "case_manifest": [
        "decoy_redesign_case_manifest.csv",
        "decoy_redesign_case_manifest.base.csv",
        "decoy_redesign_case_manifest.base.411.csv",
    ],
    "validation": ["decoy_validation.csv", "decoy_validation_esm411_all_modes.csv"],
    "graft_manifest": [
        "decoy_grafted_complex_manifest.csv",
        "decoy_grafted_complex_manifest_all_modes.<CPU_NODE>.csv",
        "decoy_grafted_complex_manifest_all_modes.csv",
    ],
    "mode_assignments": ["auto_mode_assignments.csv"],
}

REQUIRED_LANDSCAPE_COLUMNS = [
    "design_id",
    "structure_id",
    "monomer_tm_score",
    "monomer_ca_rmsd",
    "monomer_graft_rmsd",
    "dG_separated",
    "dSASA_int",
    "sc_value",
    "hbonds_int",
    "delta_unsatHbonds",
    "native_dG_separated",
    "native_dSASA_int",
    "native_sc_value",
    "native_hbonds_int",
    "native_delta_unsatHbonds",
]

DELTA_COLUMNS = [
    "delta_vs_native_dG_separated",
    "delta_vs_native_hbonds_int",
    "delta_vs_native_sc_value",
    "delta_vs_native_delta_unsatHbonds",
    "delta_vs_native_dSASA_int",
]

COMPONENT_METRICS = [
    "monomer_tm_score",
    "monomer_ca_rmsd",
    "monomer_graft_rmsd",
    "delta_vs_native_dG_separated",
    "delta_vs_native_hbonds_int",
    "delta_vs_native_sc_value",
    "delta_vs_native_delta_unsatHbonds",
    "delta_vs_native_dSASA_int",
]

COMPONENT_UNITS = {
    "monomer_tm_score": "unitless",
    "monomer_ca_rmsd": "Angstrom",
    "monomer_graft_rmsd": "Angstrom",
    "delta_vs_native_dG_separated": "Rosetta energy units",
    "delta_vs_native_hbonds_int": "count",
    "delta_vs_native_sc_value": "unitless",
    "delta_vs_native_delta_unsatHbonds": "count",
    "delta_vs_native_dSASA_int": "Angstrom^2",
}

COMPONENT_LABELS = {
    "monomer_tm_score": "Monomer TM-score",
    "monomer_ca_rmsd": "Monomer CA RMSD (Angstrom)",
    "monomer_graft_rmsd": "Monomer graft RMSD (Angstrom)",
    "delta_vs_native_dG_separated": "Delta vs native dG_separated (REU)",
    "delta_vs_native_hbonds_int": "Delta vs native hbonds_int (count)",
    "delta_vs_native_sc_value": "Delta vs native sc_value",
    "delta_vs_native_delta_unsatHbonds": (
        "Delta vs native delta_unsatHbonds (count)"
    ),
    "delta_vs_native_dSASA_int": "Delta vs native dSASA_int (Angstrom^2)",
}

TOP_LEVEL_PNGS = [
    "antigen_length_distribution.png",
    "redesign_mode_counts.png",
    "mutation_count_distribution.png",
    "mutation_fraction_distribution.png",
    "mutations_by_redesign_mode.png",
    "amino_acid_substitution_counts_heatmap.png",
    "amino_acid_substitution_frequencies_heatmap.png",
    "delphi_landscape.png",
    "mutation_region_bar.png",
    "mutation_position_frequency.png",
]

CSV_OUTPUTS = [
    "dataset_counts.csv",
    "antigen_length_summary.csv",
    "redesign_mode_counts.csv",
    "mutation_count_summary.csv",
    "mutation_summary_by_antigen.csv",
    "amino_acid_substitution_counts.csv",
    "amino_acid_substitution_frequencies.csv",
    "component_metric_summary.csv",
    "delphi_landscape_scores.csv",
    "decoy_quadrant_ranking.csv",
    "decoy_quadrant_summary.csv",
    "decoy_quadrant_unclassified.csv",
    "mutation_locations.csv",
    "mutation_region_summary.csv",
    "mutation_position_frequency.csv",
    "mutation_region_enrichment.csv",
    "mutation_localization_skipped_decoys.csv",
    "mutation_numbering_status.csv",
]

ALLOW_EMPTY_CSV_OUTPUTS = {
    "delphi_landscape_scores.csv",
    "decoy_quadrant_ranking.csv",
    "decoy_quadrant_unclassified.csv",
    "mutation_locations.csv",
    "mutation_region_summary.csv",
    "mutation_position_frequency.csv",
    "mutation_region_enrichment.csv",
    "mutation_localization_skipped_decoys.csv",
    "mutation_numbering_status.csv",
}

HISTOGRAM_FILENAMES = [f"{metric}.png" for metric in COMPONENT_METRICS]

QUADRANT_ORDER = ["Q1", "Q4", "Q2", "Q3"]
QUADRANT_RANK = {"Q1": 1, "Q4": 2, "Q2": 3, "Q3": 4}
QUADRANT_INTERPRETATION = {
    "Q1": "Best: structurally preserved with strong predicted binding perturbation",
    "Q4": "Native-like",
    "Q2": "Likely nonsense, misfolded, or incorrectly positioned",
    "Q3": "Likely nonsense, misfolded, or uninformative",
}


class AnalysisError(RuntimeError):
    """Raised for clear, user-facing analysis failures."""


def parse_bool(value: str | bool) -> bool:
    """Parse CLI booleans such as true/false, yes/no, and 1/0."""

    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build and document the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a completed backend-agnostic DELPHI-VHH decoy-generation "
            "experiment. The script reads only CSV files inside --input-dir and "
            "writes only the specified CSV/PNG outputs inside --out-dir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analysis/analyze_delphi_experiment.py "
            "--input-dir results/tables/esm411 --out-dir results/analysis/esm411\n"
            "  python analysis/analyze_delphi_experiment.py "
            "--input-dir results/tables/boltz411 --out-dir results/analysis/boltz411\n"
            "  python analysis/analyze_delphi_experiment.py "
            "--input-dir results/tables/opendde411 --out-dir results/analysis/opendde411\n\n"
            "DELPHI Landscape axes:\n"
            "  structure_preservation = official DockQ by default "
            "(relaxed decoy complex vs relaxed native complex)\n"
            "  legacy_structure_preservation = mean(z(monomer_tm_score), "
            "z(-monomer_ca_rmsd), z(-monomer_graft_rmsd))\n"
            "  predicted binding perturbation = mean(z(delta_vs_native_dG_separated), "
            "z(-delta_vs_native_hbonds_int), z(-delta_vs_native_sc_value), "
            "z(delta_vs_native_delta_unsatHbonds))\n"
            "  delta_vs_native_dSASA_int is diagnostic only."
        ),
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing completed DELPHI-VHH experiment CSV tables.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Analysis root directory; CSV files are written to tables/ and PNG figures to figures/.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Optional explicit CSV output directory for orchestration wrappers.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Optional explicit PNG output directory for orchestration wrappers.",
    )
    parser.add_argument(
        "--dpi",
        default=300,
        type=int,
        help="DPI for all generated PNG figures.",
    )
    parser.add_argument(
        "--bins",
        default=30,
        type=int,
        help="Histogram bin count for distribution figures.",
    )
    parser.add_argument(
        "--selected-only",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help=(
            "Analyze only rows with selected_decoy == True when that column "
            "exists. Pass '--selected-only false' to analyze all valid designed "
            "decoys."
        ),
    )
    parser.add_argument(
        "--include-unchanged",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help=(
            "Include unchanged native-to-designed diagonal sequence pairs in "
            "the amino-acid substitution matrices."
        ),
    )
    parser.add_argument(
        "--structure-preservation-metric",
        choices=["dockq", "legacy"],
        default="dockq",
        help="Metric used for canonical landscape x-axis and quadrant classification.",
    )
    parser.add_argument(
        "--structure-threshold",
        type=parse_dockq_preservation_threshold,
        default=None,
        help=(
            "Initial high-preservation threshold for DockQ-based quadrant "
            "classification. Defaults to DOCKQ_PRESERVATION_THRESHOLD or 0.49."
        ),
    )
    parser.add_argument(
        "--binding-threshold",
        type=float,
        default=0.0,
        help="High predicted binding-perturbation threshold for quadrant classification.",
    )
    parser.add_argument(
        "--anarci-python",
        type=Path,
        default=Path(os.environ["ANARCI_PYTHON"]) if os.environ.get("ANARCI_PYTHON") else None,
        help=(
            "Explicit Python executable for canonical ANARCI/IMGT nanobody "
            "numbering. If omitted, a validated precomputed VHH numbering table "
            "must be present in --input-dir."
        ),
    )
    parser.add_argument(
        "--anarci-bin",
        type=Path,
        default=Path(os.environ["ANARCI_BIN"]) if os.environ.get("ANARCI_BIN") else None,
        help="Explicit ANARCI command-line executable for CSV-based IMGT numbering when --anarci-python is not supplied.",
    )
    return parser


def discover_input_files(input_dir: Path) -> dict[str, Path]:
    """Discover required input files inside the input directory only."""

    if not input_dir.exists() or not input_dir.is_dir():
        raise AnalysisError(f"Input directory does not exist: {input_dir}")

    resolved: dict[str, Path] = {}
    for role, candidates in INPUT_CANDIDATES.items():
        matches = [input_dir / candidate for candidate in candidates if (input_dir / candidate).exists()]
        if not matches:
            expected = ", ".join(candidates)
            raise AnalysisError(
                f"Missing required input for {role}. Expected one of: {expected}"
            )
        selected = matches[0]
        resolved[role] = selected
        if len(matches) > 1:
            print(
                f"Multiple candidates for {role}; selected {selected.name} "
                f"from {[path.name for path in matches]}"
            )
        else:
            print(f"Selected {role}: {selected}")
    return resolved


def load_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Read every required CSV exactly once and report table sizes."""

    tables: dict[str, pd.DataFrame] = {}
    for role, path in paths.items():
        table = pd.read_csv(path)
        tables[role] = table
        print(f"{role}: {len(table)} rows")
    return tables


def load_optional_dockq_table(input_dir: Path) -> pd.DataFrame:
    """Load decoy_dockq.csv when present; return an empty frame otherwise."""

    path = input_dir / "decoy_dockq.csv"
    if not path.exists():
        print("DockQ table not found: decoy_dockq.csv")
        return pd.DataFrame()
    table = pd.read_csv(path)
    print(f"dockq: {len(table)} rows")
    return table


def load_optional_vhh_numbering_table(input_dir: Path) -> pd.DataFrame:
    """Load optional canonical ANARCI/IMGT VHH numbering rows."""

    candidates = [
        "vhh_anarci_numbering_per_structure.csv",
        "vhh_numbering_per_structure.csv",
        "nanobody_anarci_numbering_per_structure.csv",
    ]
    for filename in candidates:
        path = input_dir / filename
        if path.exists():
            table = pd.read_csv(path)
            print(f"vhh_numbering: {len(table)} rows from {filename}")
            return table
    print("VHH numbering table not found; mutation localization will try explicit nanobody sequences with ANARCI if available.")
    return pd.DataFrame()


def print_available_columns(tables: dict[str, pd.DataFrame]) -> None:
    """Print available columns for every loaded input table."""

    for role, table in tables.items():
        print(f"{role} columns: {', '.join(table.columns.astype(str))}")


def require_columns(table: pd.DataFrame, columns: Iterable[str], role: str) -> None:
    """Abort if a table is missing required case-sensitive columns."""

    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise AnalysisError(
            f"{role} is missing required case-sensitive columns: {missing}"
        )


def report_identifier_quality(role: str, table: pd.DataFrame) -> None:
    """Report duplicate and missing design_id/structure_id values."""

    for column in ["design_id", "structure_id"]:
        if column not in table.columns:
            print(f"{role}: {column} column not present")
            continue
        missing = int(table[column].isna().sum())
        duplicate_rows = int(table[column].duplicated(keep=False).sum())
        duplicate_values = int(table.loc[table[column].duplicated(keep=False), column].nunique())
        print(
            f"{role}: missing {column}={missing}; duplicate {column} "
            f"rows={duplicate_rows}; duplicate values={duplicate_values}"
        )


def ensure_unique_key(table: pd.DataFrame, keys: list[str], role: str) -> None:
    """Verify that a table is unique on keys used for row-preserving joins."""

    require_columns(table, keys, role)
    duplicate_count = int(table.duplicated(subset=keys, keep=False).sum())
    if duplicate_count:
        raise AnalysisError(
            f"{role} has {duplicate_count} rows duplicated on expected unique "
            f"join keys {keys}; refusing a row-multiplying join."
        )


def key_overlap(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> int:
    """Return the number of overlapping key values between two tables."""

    left_keys = left[keys].dropna().drop_duplicates()
    right_keys = right[keys].dropna().drop_duplicates()
    if left_keys.empty or right_keys.empty:
        return 0
    merged = left_keys.merge(right_keys, on=keys, how="inner")
    return int(len(merged))


def report_overlap(
    left_name: str,
    left: pd.DataFrame,
    right_name: str,
    right: pd.DataFrame,
    keys: list[str],
    require_nonzero: bool = True,
) -> int:
    """Report and optionally require nonzero overlap for relevant tables."""

    require_columns(left, keys, left_name)
    require_columns(right, keys, right_name)
    overlap = key_overlap(left, right, keys)
    print(f"Overlap {left_name} vs {right_name} on {keys}: {overlap}")
    if require_nonzero and overlap == 0:
        raise AnalysisError(
            f"No overlap between {left_name} and {right_name} on keys {keys}."
        )
    return overlap


def merge_preserving_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    role: str,
    require_overlap: bool = True,
) -> pd.DataFrame:
    """Left-join a table and verify that the row count is preserved."""

    ensure_unique_key(right, keys, role)
    if require_overlap:
        report_overlap("analysis population", left, role, right, keys)

    before = len(left)
    try:
        merged = left.merge(
            right,
            on=keys,
            how="left",
            suffixes=("", f"__{role}"),
            validate="m:1",
        )
    except pd.errors.MergeError as exc:
        raise AnalysisError(f"Join with {role} failed validation: {exc}") from exc

    after = len(merged)
    print(f"Join {role} on {keys}: {before} -> {after} rows")
    if after != before:
        raise AnalysisError(
            f"Join with {role} changed row count from {before} to {after}."
        )
    return merged


def truthy_series(table: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    """Return a robust boolean interpretation of a column if it exists."""

    if column not in table.columns:
        return pd.Series(default, index=table.index, dtype=bool)
    values = table[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(1 if default else 0).astype(float).ne(0)
    normalized = values.astype(str).str.strip().str.lower()
    true_mask = normalized.isin({"true", "t", "1", "yes", "y", "selected"})
    false_mask = normalized.isin({"false", "f", "0", "no", "n", ""})
    result = pd.Series(default, index=table.index, dtype=bool)
    result.loc[true_mask] = True
    result.loc[false_mask] = False
    return result


def native_reference_mask(table: pd.DataFrame) -> pd.Series:
    """Identify native-reference rows without assuming a backend-specific flag."""

    mask = pd.Series(False, index=table.index, dtype=bool)
    for column in ["native_reference", "is_native_reference"]:
        if column in table.columns:
            mask |= truthy_series(table, column)
    for column in ["design_id", "validation_status"]:
        if column in table.columns:
            normalized = table[column].astype(str).str.strip().str.lower()
            mask |= normalized.str.contains("native_reference", regex=False)
            mask |= normalized.str.contains("native reference", regex=False)
    return mask


def failed_fold_mask(validation: pd.DataFrame) -> pd.Series:
    """Identify failed or unavailable fold-validation rows."""

    require_columns(validation, ["validation_status"], "validation")
    status = validation["validation_status"].astype(str).str.strip().str.lower()
    failed = status.eq("") | status.eq("nan")
    for token in ["fail", "error", "missing", "invalid", "unavailable"]:
        failed |= status.str.contains(token, regex=False, na=False)
    return failed


def success_from_status(
    table: pd.DataFrame,
    status_columns: list[str],
    fallback_columns: list[str],
) -> pd.Series:
    """Infer successful rows from status columns or non-missing fallback metrics."""

    for column in status_columns:
        if column in table.columns:
            status = table[column].astype(str).str.strip().str.lower()
            success = ~(status.eq("") | status.eq("nan"))
            for token in ["fail", "error", "missing", "invalid", "unavailable"]:
                success &= ~status.str.contains(token, regex=False, na=False)
            return success.fillna(False)
    for column in fallback_columns:
        if column in table.columns:
            return pd.to_numeric(table[column], errors="coerce").notna()
    return pd.Series(True, index=table.index, dtype=bool)


def coalesce_to_target(
    table: pd.DataFrame,
    target: str,
    candidates: list[str],
) -> None:
    """Create or replace a target column using the first non-null candidate."""

    present = [column for column in candidates if column in table.columns]
    if not present:
        return
    result = table[present[0]]
    for column in present[1:]:
        result = result.combine_first(table[column])
    table[target] = result


def numeric_column(table: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series and raise a clear error if absent."""

    if column not in table.columns:
        raise AnalysisError(f"Required numeric column is missing: {column}")
    return pd.to_numeric(table[column], errors="coerce")


def summary_for_series(values: pd.Series) -> dict[str, float | int]:
    """Compute summary statistics with count, quartiles, and standard deviation."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()) if len(numeric) else np.nan,
        "standard deviation": float(numeric.std()) if len(numeric) > 1 else 0.0,
        "minimum": float(numeric.min()) if len(numeric) else np.nan,
        "25th percentile": float(numeric.quantile(0.25)) if len(numeric) else np.nan,
        "median": float(numeric.median()) if len(numeric) else np.nan,
        "75th percentile": float(numeric.quantile(0.75)) if len(numeric) else np.nan,
        "maximum": float(numeric.max()) if len(numeric) else np.nan,
    }


def save_distribution_plot(
    values: pd.Series,
    path: Path,
    title: str,
    xlabel: str,
    bins: int,
    dpi: int,
) -> None:
    """Save a histogram with optional SciPy KDE, mean, median, and sample size."""

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    array = numeric.dropna().to_numpy(dtype=float)
    if array.size == 0:
        raise AnalysisError(f"Cannot plot empty distribution: {title}")

    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor="white")
    counts, edges, _ = ax.hist(
        array,
        bins=bins,
        color="#4C78A8",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.6,
    )

    if gaussian_kde is not None and array.size > 1 and np.nanstd(array) > 0:
        xs = np.linspace(float(np.nanmin(array)), float(np.nanmax(array)), 300)
        bin_width = float(np.mean(np.diff(edges))) if len(edges) > 1 else 1.0
        density = gaussian_kde(array)(xs) * array.size * bin_width
        ax.plot(xs, density, color="#F58518", linewidth=2.0, label="KDE")

    mean_value = float(np.nanmean(array))
    median_value = float(np.nanmedian(array))
    ax.axvline(mean_value, color="#D62728", linewidth=2.0, label="Mean")
    ax.axvline(median_value, color="#2CA02C", linewidth=2.0, linestyle="--", label="Median")
    stats_text = (
        f"n = {array.size}\n"
        f"mean = {mean_value:.4g}\n"
        f"median = {median_value:.4g}"
    )
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def save_redesign_mode_plot(counts: pd.DataFrame, path: Path, dpi: int) -> None:
    """Save grouped bars for structure, selected-decoy, and grafted-decoy counts."""

    modes = counts["redesign_mode"].tolist()
    x = np.arange(len(modes))
    width = 0.24
    columns = [
        ("structure_count", "Unique structures"),
        ("selected_decoy_count", "Selected decoys"),
        ("grafted_decoy_count", "Grafted decoys"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
    for offset, (column, label) in zip([-width, 0.0, width], columns):
        values = counts[column].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Redesign Mode Counts")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def save_mutations_by_mode_plot(
    mutation_table: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Save a box plot of mutated residues grouped by redesign mode."""

    if "redesign_mode" not in mutation_table.columns:
        raise AnalysisError("Cannot plot mutations by redesign mode without redesign_mode.")
    plot_table = mutation_table.dropna(subset=["redesign_mode", "n_residues_mutated"]).copy()
    if plot_table.empty:
        raise AnalysisError("No mutation rows available for redesign-mode box plot.")

    modes = [
        mode
        for mode in REDESIGN_MODE_ORDER
        if mode in set(plot_table["redesign_mode"].astype(str))
    ]
    extra_modes = sorted(set(plot_table["redesign_mode"].astype(str)) - set(modes))
    modes.extend(extra_modes)
    data = [
        plot_table.loc[plot_table["redesign_mode"].astype(str) == mode, "n_residues_mutated"]
        .astype(float)
        .to_numpy()
        for mode in modes
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor="white")
    ax.boxplot(data, tick_labels=modes, patch_artist=True, boxprops={"facecolor": "#D9E8F5"})
    rng = np.random.default_rng(7)
    for idx, values in enumerate(data, start=1):
        jitter = rng.normal(loc=idx, scale=0.035, size=len(values))
        ax.scatter(jitter, values, s=14, color="#333333", alpha=0.22, linewidths=0)

    ax.set_title("Mutated Residues by Redesign Mode")
    ax.set_xlabel("Redesign mode")
    ax.set_ylabel("Mutated residues (count)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    colorbar_label: str,
    dpi: int,
    annotate_counts: bool = False,
) -> None:
    """Save a square 20 by 20 Matplotlib heatmap."""

    if matrix.shape != (20, 20):
        raise AnalysisError(f"Substitution matrix must be 20 x 20, got {matrix.shape}.")

    fig, ax = plt.subplots(figsize=(8.0, 7.2), facecolor="white")
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(20))
    ax.set_yticks(np.arange(20))
    ax.set_xticklabels(AA_ORDER)
    ax.set_yticklabels(AA_ORDER)
    ax.set_xlabel("Native amino acid")
    ax.set_ylabel("Redesigned amino acid")
    ax.set_title(title, fontsize=14)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)

    if annotate_counts and float(np.nanmax(values)) <= 999:
        for row in range(20):
            for column in range(20):
                value = int(values[row, column])
                if value:
                    ax.text(
                        column,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value > np.nanmax(values) * 0.5 else "black",
                    )

    ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def save_landscape_plot(
    landscape: pd.DataFrame,
    path: Path,
    dpi: int,
    *,
    structure_threshold: float,
    binding_threshold: float,
) -> None:
    """Save the DELPHI Landscape scatter plot using canonical axis columns."""

    x_column = "structure_preservation"
    y_column = "binding_perturbation"
    if x_column not in landscape.columns or y_column not in landscape.columns:
        raise AnalysisError("Landscape axis columns are missing.")

    fig, ax = plt.subplots(figsize=(7.4, 6.1), facecolor="white")
    if "redesign_mode" in landscape.columns and landscape["redesign_mode"].notna().any():
        plotted_any = False
        for mode in REDESIGN_MODE_ORDER:
            subset = landscape[landscape["redesign_mode"].astype(str) == mode]
            if subset.empty:
                continue
            ax.scatter(
                subset[x_column],
                subset[y_column],
                s=34,
                alpha=0.62,
                label=mode,
                edgecolors="none",
            )
            plotted_any = True
        other = landscape[
            landscape["redesign_mode"].notna()
            & ~landscape["redesign_mode"].astype(str).isin(REDESIGN_MODE_ORDER)
        ]
        if not other.empty:
            ax.scatter(other[x_column], other[y_column], s=34, alpha=0.62, label="other", edgecolors="none")
            plotted_any = True
        if not plotted_any:
            ax.scatter(landscape[x_column], landscape[y_column], s=34, alpha=0.62)
        else:
            ax.legend(frameon=False, title="Redesign mode", fontsize=9, title_fontsize=10)
    else:
        ax.scatter(landscape[x_column], landscape[y_column], s=34, alpha=0.62)

    ax.axhline(binding_threshold, color="#666666", linewidth=1.0)
    ax.axvline(structure_threshold, color="#666666", linewidth=1.0)
    ax.set_xlabel("DockQ interface preservation" if (landscape.get("structure_preservation_metric", pd.Series(dtype=str)).astype(str) == "dockq").any() else "Legacy structural preservation index", fontsize=12)
    ax.set_ylabel("Delta dG (REU)", fontsize=12)
    ax.set_title("DELPHI Landscape", fontsize=15)
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def validate_input_schemas(tables: dict[str, pd.DataFrame]) -> None:
    """Validate required table schemas, identifiers, and overlaps."""

    comparison = tables["comparison"]
    case_manifest = tables["case_manifest"]
    validation = tables["validation"]
    graft_manifest = tables["graft_manifest"]
    mode_assignments = tables["mode_assignments"]

    require_columns(comparison, ["design_id", "structure_id"], "comparison")
    require_columns(case_manifest, ["structure_id", "antigen_length"], "case_manifest")
    require_columns(validation, ["design_id", "structure_id", "validation_status"], "validation")
    require_columns(graft_manifest, ["design_id", "structure_id"], "graft_manifest")
    require_columns(mode_assignments, ["structure_id"], "mode_assignments")

    if "native_antigen_sequence" not in case_manifest.columns and "native_antigen_sequence" not in validation.columns:
        raise AnalysisError(
            "native_antigen_sequence is required in the case manifest or validation table."
        )
    if "designed_sequence" not in validation.columns and "designed_sequence" not in graft_manifest.columns:
        raise AnalysisError(
            "designed_sequence is required in the validation table or graft manifest."
        )

    print_available_columns(tables)
    for role, table in tables.items():
        report_identifier_quality(role, table)

    ensure_unique_key(case_manifest, ["structure_id"], "case_manifest")
    ensure_unique_key(mode_assignments, ["structure_id"], "mode_assignments")

    report_overlap("case_manifest", case_manifest, "validation", validation, ["structure_id"])
    report_overlap("comparison", comparison, "validation", validation, ["design_id", "structure_id"])
    report_overlap("graft_manifest", graft_manifest, "validation", validation, ["design_id", "structure_id"])
    report_overlap("mode_assignments", mode_assignments, "case_manifest", case_manifest, ["structure_id"])


def select_analysis_population(
    validation: pd.DataFrame,
    selected_only: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select valid designed decoys for mutation and decoy-level analyses."""

    native_mask = native_reference_mask(validation)
    failed_mask = failed_fold_mask(validation) & ~native_mask
    non_native = ~native_mask
    valid_designed = non_native & ~failed_mask
    if selected_only and "selected_decoy" in validation.columns:
        selected_mask = truthy_series(validation, "selected_decoy") & valid_designed
    elif selected_only:
        print("selected_decoy column not present; analyzing all valid designed decoys.")
        selected_mask = valid_designed
    else:
        selected_mask = valid_designed

    population = validation.loc[selected_mask].copy()
    counts = {
        "total_validation_rows": int(len(validation)),
        "native_reference_rows": int(native_mask.sum()),
        "failed_fold_rows": int(failed_mask.sum()),
        "passed_fold_rows": int(valid_designed.sum()),
        "selected_decoy_rows": int(selected_mask.sum()),
    }
    print(f"total validation rows: {counts['total_validation_rows']}")
    print(f"selected decoy rows: {counts['selected_decoy_rows']}")
    print(f"excluded native-reference rows: {counts['native_reference_rows']}")
    print(f"excluded failed-fold rows: {counts['failed_fold_rows']}")
    if population.empty:
        raise AnalysisError("No decoy rows remain after analysis population selection.")
    return population, counts


def graft_success_mask(graft_manifest: pd.DataFrame) -> pd.Series:
    """Identify successfully grafted decoys."""

    if "monomer_graft_rmsd" in graft_manifest.columns:
        return pd.to_numeric(graft_manifest["monomer_graft_rmsd"], errors="coerce").notna()
    return success_from_status(
        graft_manifest,
        ["graft_status", "grafted_status", "status", "validation_status"],
        ["graft_rmsd", "ca_rmsd"],
    )


def rosetta_success_mask(comparison: pd.DataFrame) -> pd.Series:
    """Identify Rosetta-scored decoys."""

    return success_from_status(
        comparison,
        ["rosetta_status"],
        ["dG_separated", "dSASA_int", "hbonds_int", "delta_unsatHbonds"],
    )


def compute_dataset_counts(
    case_manifest: pd.DataFrame,
    validation: pd.DataFrame,
    comparison: pd.DataFrame,
    graft_manifest: pd.DataFrame,
    population: pd.DataFrame,
    validation_counts: dict[str, int],
) -> pd.DataFrame:
    """Compute dataset-level summary counts."""

    grafted = graft_manifest.loc[graft_success_mask(graft_manifest)]
    rosetta_scored = comparison.loc[rosetta_success_mask(comparison)]
    rows = [
        ("unique_structures_in_base_case_manifest", case_manifest["structure_id"].nunique()),
        ("structures_with_validation_rows", validation["structure_id"].nunique()),
        ("validation_rows", validation_counts["total_validation_rows"]),
        ("native_reference_rows", validation_counts["native_reference_rows"]),
        ("passed_folds", validation_counts["passed_fold_rows"]),
        ("failed_folds", validation_counts["failed_fold_rows"]),
        ("selected_decoys", validation_counts["selected_decoy_rows"]),
        ("successfully_grafted_decoys", len(grafted)),
        ("rosetta_scored_decoys", len(rosetta_scored)),
        ("unique_structures_among_selected_decoys", population["structure_id"].nunique()),
        ("unique_structures_among_grafted_decoys", grafted["structure_id"].nunique()),
        (
            "unique_structures_among_rosetta_scored_decoys",
            rosetta_scored["structure_id"].nunique(),
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def first_nonnull_by_structure(table: pd.DataFrame, column: str) -> dict[str, object]:
    """Return the first non-null column value for each structure_id."""

    if column not in table.columns or "structure_id" not in table.columns:
        return {}
    subset = table.loc[table[column].notna(), ["structure_id", column]].drop_duplicates("structure_id")
    return dict(zip(subset["structure_id"], subset[column]))


def resolve_structure_modes(
    case_manifest: pd.DataFrame,
    mode_assignments: pd.DataFrame,
    graft_manifest: pd.DataFrame,
) -> pd.Series:
    """Resolve one redesign mode per structure using the required priority."""

    structures = case_manifest[["structure_id"]].copy()
    mode = pd.Series(pd.NA, index=structures.index, dtype="object")
    if "redesign_mode" in case_manifest.columns:
        mode = case_manifest["redesign_mode"].astype("object")
    if "selected_redesign_mode" in mode_assignments.columns:
        assignment_map = first_nonnull_by_structure(mode_assignments, "selected_redesign_mode")
        mapped = structures["structure_id"].map(assignment_map)
        mode = mode.combine_first(mapped)
    if "redesign_mode" in graft_manifest.columns:
        graft_map = first_nonnull_by_structure(graft_manifest, "redesign_mode")
        mapped = structures["structure_id"].map(graft_map)
        mode = mode.combine_first(mapped)
    return mode


def attach_mode_to_table(
    table: pd.DataFrame,
    structure_modes: pd.Series,
    case_manifest: pd.DataFrame,
) -> pd.Series:
    """Attach structure-level redesign modes to a decoy-level table."""

    mapping = dict(zip(case_manifest["structure_id"], structure_modes))
    return table["structure_id"].map(mapping)


def compute_antigen_length_outputs(
    case_manifest: pd.DataFrame,
    out_dir: Path,
    bins: int,
    dpi: int,
) -> pd.DataFrame:
    """Compute antigen-length summary statistics and figure."""

    unique_cases = case_manifest.drop_duplicates("structure_id").copy()
    lengths = pd.to_numeric(unique_cases["antigen_length"], errors="coerce")
    if lengths.dropna().empty:
        raise AnalysisError("No numeric antigen_length values are available.")
    summary = pd.DataFrame([{"metric": "antigen_length", **summary_for_series(lengths)}])
    save_distribution_plot(
        lengths,
        out_dir / "antigen_length_distribution.png",
        "Antigen Length Distribution",
        "Antigen length (amino acids)",
        bins,
        dpi,
    )
    return summary


def compute_redesign_mode_counts(
    case_manifest: pd.DataFrame,
    population: pd.DataFrame,
    graft_manifest: pd.DataFrame,
    structure_modes: pd.Series,
    out_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    """Compute redesign-mode counts at structure, selected-decoy, and graft levels."""

    structure_table = case_manifest[["structure_id"]].drop_duplicates().copy()
    structure_table["redesign_mode"] = structure_modes.to_numpy()

    selected_columns = ["design_id", "structure_id"]
    if "redesign_mode" in population.columns:
        selected_columns.append("redesign_mode")
    selected_table = population[selected_columns].copy()
    selected_structure_mode = attach_mode_to_table(
        selected_table, structure_modes, case_manifest
    )
    if "redesign_mode" in selected_table.columns:
        selected_table["redesign_mode"] = selected_table["redesign_mode"].combine_first(
            selected_structure_mode
        )
    else:
        selected_table["redesign_mode"] = selected_structure_mode

    grafted_columns = ["design_id", "structure_id"]
    if "redesign_mode" in graft_manifest.columns:
        grafted_columns.append("redesign_mode")
    grafted_table = graft_manifest.loc[graft_success_mask(graft_manifest), grafted_columns].copy()
    grafted_structure_mode = attach_mode_to_table(
        grafted_table, structure_modes, case_manifest
    )
    if "redesign_mode" in grafted_table.columns:
        grafted_table["redesign_mode"] = grafted_table["redesign_mode"].combine_first(
            grafted_structure_mode
        )
    else:
        grafted_table["redesign_mode"] = grafted_structure_mode

    rows = []
    structure_denominator = max(int(structure_table["structure_id"].nunique()), 1)
    selected_denominator = max(int(len(selected_table)), 1)
    grafted_denominator = max(int(len(grafted_table)), 1)
    for mode in REDESIGN_MODE_ORDER:
        structure_count = int(
            structure_table.loc[structure_table["redesign_mode"].astype(str) == mode, "structure_id"].nunique()
        )
        selected_count = int((selected_table["redesign_mode"].astype(str) == mode).sum())
        grafted_count = int((grafted_table["redesign_mode"].astype(str) == mode).sum())
        rows.append(
            {
                "redesign_mode": mode,
                "structure_count": structure_count,
                "selected_decoy_count": selected_count,
                "grafted_decoy_count": grafted_count,
                "structure_fraction": structure_count / structure_denominator,
                "selected_decoy_fraction": selected_count / selected_denominator,
                "grafted_decoy_fraction": grafted_count / grafted_denominator,
            }
        )
    counts = pd.DataFrame(rows)
    save_redesign_mode_plot(counts, out_dir / "redesign_mode_counts.png", dpi)
    return counts


def prepare_analysis_table(
    population: pd.DataFrame,
    comparison: pd.DataFrame,
    case_manifest: pd.DataFrame,
    graft_manifest: pd.DataFrame,
    mode_assignments: pd.DataFrame,
    dockq: pd.DataFrame | None = None,
    require_dockq: bool = False,
) -> pd.DataFrame:
    """Join decoy and structure metadata while preserving row counts."""

    table = population.copy()
    table = merge_preserving_rows(table, comparison, ["design_id", "structure_id"], "comparison")
    if dockq is not None and not dockq.empty:
        table = merge_preserving_rows(table, dockq, ["design_id", "structure_id"], "dockq", require_overlap=require_dockq)
    elif require_dockq:
        raise AnalysisError("DockQ structure-preservation metric requested but decoy_dockq.csv is missing or empty.")
    table = merge_preserving_rows(table, graft_manifest, ["design_id", "structure_id"], "graft_manifest")
    table = merge_preserving_rows(table, case_manifest, ["structure_id"], "case_manifest")
    table = merge_preserving_rows(table, mode_assignments, ["structure_id"], "mode_assignments")

    for target in [
        "antigen_length",
        "native_antigen_sequence",
        "designed_sequence",
        "native_nanobody_sequence",
        "nanobody_sequence",
        "native_vhh_sequence",
        "vhh_sequence",
        "designed_nanobody_sequence",
        "mutated_nanobody_sequence",
        "designed_vhh_sequence",
        "mutated_vhh_sequence",
        "experiment_namespace",
        "backend",
        "fold_backend",
        "mode_policy",
        "monomer_tm_score",
        "monomer_ca_rmsd",
        "monomer_graft_rmsd",
        "dockq",
        "dockq_global",
        "dockq_quality",
        "dockq_fnat",
        "dockq_irmsd",
        "dockq_lrmsd",
        "dockq_reference_state",
        "dG_separated",
        "dSASA_int",
        "sc_value",
        "hbonds_int",
        "delta_unsatHbonds",
        "native_dG_separated",
        "native_dSASA_int",
        "native_sc_value",
        "native_hbonds_int",
        "native_delta_unsatHbonds",
    ]:
        coalesce_to_target(
            table,
            target,
            [
                f"{target}__dockq",
                f"{target}__comparison",
                target,
                f"{target}__graft_manifest",
                f"{target}__case_manifest",
                f"{target}__mode_assignments",
            ],
        )

    coalesce_to_target(
        table,
        "redesign_mode",
        [
            "redesign_mode",
            "redesign_mode__comparison",
            "selected_redesign_mode",
            "selected_redesign_mode__mode_assignments",
            "redesign_mode__graft_manifest",
        ],
    )
    return table


def compute_mutation_outputs(
    table: pd.DataFrame,
    out_dir: Path,
    bins: int,
    dpi: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Compute mutation counts, mutation summaries, and mutation figures."""

    require_columns(
        table,
        ["design_id", "structure_id", "native_antigen_sequence", "designed_sequence", "antigen_length"],
        "analysis table",
    )
    work = table.copy()
    missing_sequence_mask = work["native_antigen_sequence"].isna() | work["designed_sequence"].isna()
    native_sequences = work["native_antigen_sequence"].fillna("").astype(str)
    designed_sequences = work["designed_sequence"].fillna("").astype(str)
    native_lengths = native_sequences.str.len()
    designed_lengths = designed_sequences.str.len()
    mismatch_mask = missing_sequence_mask | native_lengths.ne(designed_lengths)
    mismatch_count = int(mismatch_mask.sum())
    print(f"sequence-length mismatches: {mismatch_count}")

    valid = work.loc[~mismatch_mask].copy()
    if valid.empty:
        raise AnalysisError("No selected designs have matching sequence lengths.")

    mutation_counts = []
    for native_sequence, designed_sequence in zip(
        valid["native_antigen_sequence"].astype(str),
        valid["designed_sequence"].astype(str),
    ):
        mutation_counts.append(
            sum(
                native_residue != designed_residue
                for native_residue, designed_residue in zip(native_sequence, designed_sequence)
            )
        )
    valid["n_residues_mutated"] = mutation_counts
    valid["antigen_length"] = pd.to_numeric(valid["antigen_length"], errors="coerce")
    if valid["antigen_length"].isna().any() or (valid["antigen_length"] <= 0).any():
        raise AnalysisError("antigen_length must be positive and numeric for mutation_fraction.")
    valid["mutation_fraction"] = valid["n_residues_mutated"] / valid["antigen_length"]

    print(f"selected designs with matching sequence lengths: {len(valid)}")
    print(f"selected designs excluded due to sequence-length mismatch: {mismatch_count}")

    summary = pd.DataFrame(
        [
            {"metric": "n_residues_mutated", **summary_for_series(valid["n_residues_mutated"])},
            {"metric": "mutation_fraction", **summary_for_series(valid["mutation_fraction"])},
        ]
    )

    group = valid.groupby("structure_id", dropna=False)
    by_antigen = group.agg(
        antigen_length=("antigen_length", "first"),
        n_selected_decoys=("design_id", "count"),
        mean_residues_mutated=("n_residues_mutated", "mean"),
        median_residues_mutated=("n_residues_mutated", "median"),
        minimum_residues_mutated=("n_residues_mutated", "min"),
        maximum_residues_mutated=("n_residues_mutated", "max"),
        mean_mutation_fraction=("mutation_fraction", "mean"),
        median_mutation_fraction=("mutation_fraction", "median"),
    ).reset_index()
    if "redesign_mode" in valid.columns:
        mode_by_antigen = group["redesign_mode"].agg(
            lambda values: values.dropna().astype(str).iloc[0] if values.dropna().size else pd.NA
        )
        by_antigen = by_antigen.merge(
            mode_by_antigen.rename("redesign_mode").reset_index(),
            on="structure_id",
            how="left",
        )
        ordered = [
            "structure_id",
            "antigen_length",
            "redesign_mode",
            "n_selected_decoys",
            "mean_residues_mutated",
            "median_residues_mutated",
            "minimum_residues_mutated",
            "maximum_residues_mutated",
            "mean_mutation_fraction",
            "median_mutation_fraction",
        ]
        by_antigen = by_antigen[ordered]

    save_distribution_plot(
        valid["n_residues_mutated"],
        out_dir / "mutation_count_distribution.png",
        "Mutation Count Distribution",
        "Mutated residues (count)",
        bins,
        dpi,
    )
    save_distribution_plot(
        valid["mutation_fraction"],
        out_dir / "mutation_fraction_distribution.png",
        "Mutation Fraction Distribution",
        "Mutation fraction",
        bins,
        dpi,
    )
    save_mutations_by_mode_plot(valid, out_dir / "mutations_by_redesign_mode.png", dpi)

    return valid, summary, by_antigen, mismatch_count


def build_substitution_matrices(
    mutation_table: pd.DataFrame,
    include_unchanged: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 20 by 20 amino-acid substitution count and frequency matrices."""

    counts = pd.DataFrame(0, index=AA_ORDER, columns=AA_ORDER, dtype=int)
    noncanonical_pairs = 0
    for native_sequence, designed_sequence in zip(
        mutation_table["native_antigen_sequence"].astype(str),
        mutation_table["designed_sequence"].astype(str),
    ):
        for native_residue, designed_residue in zip(native_sequence, designed_sequence):
            if not include_unchanged and native_residue == designed_residue:
                continue
            if native_residue not in AA_ORDER or designed_residue not in AA_ORDER:
                noncanonical_pairs += 1
                continue
            counts.loc[designed_residue, native_residue] += 1

    if noncanonical_pairs:
        print(f"noncanonical amino-acid pairs skipped from substitution matrix: {noncanonical_pairs}")
    column_totals = counts.sum(axis=0)
    frequencies = counts.astype(float).divide(column_totals.replace(0, np.nan), axis=1).fillna(0.0)
    return counts, frequencies


def compute_native_deltas(table: pd.DataFrame) -> None:
    """Compute decoy-minus-native DELPHI Landscape delta metrics in place."""

    required_pairs = [
        ("delta_vs_native_dG_separated", "dG_separated", "native_dG_separated"),
        ("delta_vs_native_hbonds_int", "hbonds_int", "native_hbonds_int"),
        ("delta_vs_native_sc_value", "sc_value", "native_sc_value"),
        (
            "delta_vs_native_delta_unsatHbonds",
            "delta_unsatHbonds",
            "native_delta_unsatHbonds",
        ),
        ("delta_vs_native_dSASA_int", "dSASA_int", "native_dSASA_int"),
    ]
    for output, decoy_column, native_column in required_pairs:
        table[output] = numeric_column(table, decoy_column) - numeric_column(table, native_column)


def missing_component_counts(table: pd.DataFrame, metrics: list[str] | None = None) -> pd.Series:
    """Return missing-value counts for each requested landscape metric."""

    selected = metrics or COMPONENT_METRICS
    return table[selected].apply(
        lambda column: pd.to_numeric(column, errors="coerce").isna().sum()
    )


def compute_component_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-z-score component metric summary statistics."""

    rows = []
    for metric in COMPONENT_METRICS:
        rows.append(
            {
                "metric": metric,
                "unit": COMPONENT_UNITS[metric],
                **summary_for_series(table[metric]),
            }
        )
    return pd.DataFrame(rows)


def zscore(values: pd.Series, label: str) -> pd.Series:
    """Compute population z-scores with ddof=0 and zero-variance protection."""

    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        raise AnalysisError(f"Metric {label} has zero or undefined population variance.")
    return (numeric - mean) / std


def compute_landscape_scores(table: pd.DataFrame, structure_metric: str = "dockq") -> pd.DataFrame:
    """Compute legacy structural preservation plus canonical landscape axes."""

    landscape = table.copy()
    legacy_components = pd.concat(
        [
            zscore(landscape["monomer_tm_score"], "monomer_tm_score"),
            zscore(-numeric_column(landscape, "monomer_ca_rmsd"), "-monomer_ca_rmsd"),
            zscore(-numeric_column(landscape, "monomer_graft_rmsd"), "-monomer_graft_rmsd"),
        ],
        axis=1,
    )
    landscape["legacy_structure_preservation"] = legacy_components.mean(axis=1)
    landscape["Structural Preservation Index"] = landscape["legacy_structure_preservation"]
    landscape["Binding Perturbation Index"] = pd.concat(
        [
            zscore(
                landscape["delta_vs_native_dG_separated"],
                "delta_vs_native_dG_separated",
            ),
            zscore(
                -numeric_column(landscape, "delta_vs_native_hbonds_int"),
                "-delta_vs_native_hbonds_int",
            ),
            zscore(
                -numeric_column(landscape, "delta_vs_native_sc_value"),
                "-delta_vs_native_sc_value",
            ),
            zscore(
                landscape["delta_vs_native_delta_unsatHbonds"],
                "delta_vs_native_delta_unsatHbonds",
            ),
        ],
        axis=1,
    ).mean(axis=1)
    if structure_metric == "dockq":
        landscape["structure_preservation"] = numeric_column(landscape, "dockq")
        landscape["structure_preservation_metric"] = "dockq"
    elif structure_metric == "legacy":
        landscape["structure_preservation"] = landscape["legacy_structure_preservation"]
        landscape["structure_preservation_metric"] = "legacy_structure_preservation"
    else:
        raise AnalysisError(f"Unsupported structure preservation metric: {structure_metric}")
    landscape["binding_perturbation"] = landscape["Binding Perturbation Index"]
    return landscape


def classify_delphi_quadrant(
    structure_preservation: object,
    binding_perturbation: object,
    structure_threshold: float,
    binding_threshold: float,
) -> tuple[str, float, str]:
    """Classify one row using authoritative DELPHI quadrant definitions."""

    sp = pd.to_numeric(pd.Series([structure_preservation]), errors="coerce").iloc[0]
    bp = pd.to_numeric(pd.Series([binding_perturbation]), errors="coerce").iloc[0]
    if pd.isna(sp) or pd.isna(bp) or not np.isfinite(float(sp)) or not np.isfinite(float(bp)):
        return "unclassified", np.nan, "missing or invalid structure_preservation/binding_perturbation"
    structure_high = float(sp) >= structure_threshold
    binding_high = float(bp) >= binding_threshold
    if structure_high and binding_high:
        quadrant = "Q1"
    elif structure_high and not binding_high:
        quadrant = "Q4"
    elif not structure_high and binding_high:
        quadrant = "Q2"
    else:
        quadrant = "Q3"
    return quadrant, float(QUADRANT_RANK[quadrant]), QUADRANT_INTERPRETATION[quadrant]


def _fold_validity_mask(table: pd.DataFrame) -> pd.Series:
    if "validation_status" not in table.columns:
        return pd.Series(True, index=table.index, dtype=bool)
    status = table["validation_status"].astype(str).str.strip().str.lower()
    invalid = status.eq("") | status.eq("nan")
    for token in ["fail", "error", "missing", "invalid", "unavailable"]:
        invalid |= status.str.contains(token, regex=False, na=False)
    return ~invalid


def landscape_required_metrics(structure_metric: str) -> list[str]:
    required = list(COMPONENT_METRICS)
    if structure_metric == "dockq":
        required.append("dockq")
    return required


def mark_landscape_eligibility(table: pd.DataFrame, structure_metric: str) -> pd.Series:
    required = landscape_required_metrics(structure_metric)
    missing_required = table[required].apply(lambda column: pd.to_numeric(column, errors="coerce").isna())
    complete = ~missing_required.any(axis=1)
    fold_valid = _fold_validity_mask(table)
    table["landscape_eligible"] = complete & fold_valid
    reasons = []
    for idx, row in table.iterrows():
        row_reasons = [metric for metric in required if pd.isna(pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0])]
        if not fold_valid.loc[idx]:
            row_reasons.append("invalid_fold")
        reasons.append(";".join(row_reasons))
    table["landscape_unclassified_reason"] = reasons
    return table["landscape_eligible"]


def build_quadrant_tables(
    landscape_scored: pd.DataFrame,
    unclassified_source: pd.DataFrame,
    *,
    structure_threshold: float,
    binding_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = landscape_scored.copy()
    classifications = ranked.apply(
        lambda row: classify_delphi_quadrant(
            row.get("structure_preservation"),
            row.get("binding_perturbation"),
            structure_threshold,
            binding_threshold,
        ),
        axis=1,
    )
    ranked["quadrant"] = [item[0] for item in classifications]
    ranked["quadrant_rank"] = [item[1] for item in classifications]
    ranked["quadrant_interpretation"] = [item[2] for item in classifications]
    ranked["structure_preservation_threshold"] = structure_threshold
    ranked["binding_perturbation_threshold"] = binding_threshold
    ranked["native_rosetta_state"] = ranked.get("native_rosetta_state", "native_relaxed")
    ranked["decoy_rosetta_state"] = ranked.get("decoy_rosetta_state", "decoy_relaxed")
    if "dockq_reference_state" not in ranked.columns:
        ranked["dockq_reference_state"] = "native_relaxed"
    ranked["coordinate_score"] = pd.to_numeric(ranked["structure_preservation"], errors="coerce") + pd.to_numeric(ranked["binding_perturbation"], errors="coerce")
    ranked["delphi_score"] = ranked["coordinate_score"]

    classified = ranked.copy()
    ranked = classified[classified["quadrant"] == "Q1"].copy()
    ranked["_coordinate_sort"] = -pd.to_numeric(ranked["coordinate_score"], errors="coerce")
    ranked = ranked.sort_values(
        ["_coordinate_sort", "structure_id", "design_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).drop(columns=["_coordinate_sort"])

    classified_count = len(classified)
    summary_rows = []
    for quadrant in QUADRANT_ORDER:
        subset = classified[classified["quadrant"] == quadrant]
        summary_rows.append(
            {
                "quadrant": quadrant,
                "quadrant_rank": QUADRANT_RANK[quadrant],
                "quadrant_interpretation": QUADRANT_INTERPRETATION[quadrant],
                "structure_preservation_threshold": structure_threshold,
                "binding_perturbation_threshold": binding_threshold,
                "decoy_count": int(len(subset)),
                "fraction_of_classified_decoys": float(len(subset) / classified_count) if classified_count else 0.0,
                "median_dockq": float(pd.to_numeric(subset.get("dockq", pd.Series(dtype=float)), errors="coerce").median()) if not subset.empty else np.nan,
                "mean_dockq": float(pd.to_numeric(subset.get("dockq", pd.Series(dtype=float)), errors="coerce").mean()) if not subset.empty else np.nan,
                "median_structure_preservation": float(pd.to_numeric(subset.get("structure_preservation", pd.Series(dtype=float)), errors="coerce").median()) if not subset.empty else np.nan,
                "median_binding_perturbation": float(pd.to_numeric(subset.get("binding_perturbation", pd.Series(dtype=float)), errors="coerce").median()) if not subset.empty else np.nan,
                "mean_binding_perturbation": float(pd.to_numeric(subset.get("binding_perturbation", pd.Series(dtype=float)), errors="coerce").mean()) if not subset.empty else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)

    unclassified = unclassified_source.copy()
    if not unclassified.empty:
        unclassified["quadrant"] = "unclassified"
        unclassified["quadrant_rank"] = np.nan
        unclassified["quadrant_interpretation"] = unclassified.get("landscape_unclassified_reason", "unclassified")
        unclassified["structure_preservation_threshold"] = structure_threshold
        unclassified["binding_perturbation_threshold"] = binding_threshold
    return ranked, summary, unclassified


def build_landscape_score_table(
    landscape: pd.DataFrame,
    comparison_columns: list[str],
) -> pd.DataFrame:
    """Build the ranked landscape score table with required and extra columns."""

    required_order = [
        "design_id",
        "structure_id",
        "redesign_mode",
        "antigen_length",
        "native_antigen_sequence",
        "designed_sequence",
        "n_residues_mutated",
        "mutation_fraction",
        "monomer_tm_score",
        "monomer_ca_rmsd",
        "monomer_graft_rmsd",
        "dockq",
        "dockq_quality",
        "dockq_fnat",
        "dockq_irmsd",
        "dockq_lrmsd",
        "dockq_reference_state",
        "dG_separated",
        "dSASA_int",
        "sc_value",
        "hbonds_int",
        "delta_unsatHbonds",
        "native_dG_separated",
        "native_dSASA_int",
        "native_sc_value",
        "native_hbonds_int",
        "native_delta_unsatHbonds",
        "delta_vs_native_dG_separated",
        "delta_vs_native_dSASA_int",
        "delta_vs_native_sc_value",
        "delta_vs_native_hbonds_int",
        "delta_vs_native_delta_unsatHbonds",
        "legacy_structure_preservation",
        "structure_preservation",
        "structure_preservation_metric",
        "structure_preservation_threshold",
        "binding_perturbation",
        "binding_perturbation_threshold",
        "coordinate_score",
        "delphi_score",
        "Structural Preservation Index",
        "Binding Perturbation Index",
    ]

    score = pd.DataFrame(index=landscape.index)
    for column in required_order:
        if column in landscape.columns:
            score[column] = landscape[column]

    for column in comparison_columns:
        if column in score.columns or column in {"design_id", "structure_id"}:
            continue
        source = column if column in landscape.columns else f"{column}__comparison"
        if source in landscape.columns:
            score[column] = landscape[source]

    for column in landscape.columns:
        if column in score.columns or "__" in column:
            continue
        if column in {"selected_decoy", "validation_status"}:
            score[column] = landscape[column]

    score = score.sort_values(
        ["Binding Perturbation Index", "Structural Preservation Index"],
        ascending=[False, False],
        kind="mergesort",
    )
    return score


def no_infinite_values(table: pd.DataFrame, columns: list[str]) -> bool:
    """Return True if all listed numeric columns contain no infinite values."""

    for column in columns:
        if column not in table.columns:
            continue
        numeric = pd.to_numeric(table[column], errors="coerce")
        if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            return False
    return True


def validate_outputs(
    table_dir: Path,
    figure_dir: Path,
    csv_tables: dict[str, pd.DataFrame],
    counts_matrix: pd.DataFrame,
    frequency_matrix: pd.DataFrame,
    score_table: pd.DataFrame,
    expected_landscape_rows: int,
    missing_counts: pd.Series,
    excluded_missing_metrics: int,
) -> None:
    """Validate required output files, matrix shapes, and derived values."""

    for filename in CSV_OUTPUTS:
        path = table_dir / filename
        if not path.exists():
            raise AnalysisError(f"Required CSV was not written: {path}")
        if filename not in csv_tables:
            raise AnalysisError(f"Required CSV table was not registered: {filename}")
        if filename not in ALLOW_EMPTY_CSV_OUTPUTS and csv_tables[filename].empty:
            raise AnalysisError(f"Required CSV has no data rows: {filename}")

    for filename in TOP_LEVEL_PNGS:
        path = figure_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            raise AnalysisError(f"Required PNG missing or empty: {path}")

    if expected_landscape_rows > 0:
        hist_dir = figure_dir / "component_histograms"
        for filename in HISTOGRAM_FILENAMES:
            path = hist_dir / filename
            if not path.exists() or path.stat().st_size == 0:
                raise AnalysisError(f"Required component histogram missing or empty: {path}")

    if counts_matrix.shape != (20, 20):
        raise AnalysisError(f"Count substitution matrix shape is {counts_matrix.shape}, not 20 x 20.")
    if frequency_matrix.shape != (20, 20):
        raise AnalysisError(
            f"Frequency substitution matrix shape is {frequency_matrix.shape}, not 20 x 20."
        )
    if len(score_table) != expected_landscape_rows:
        raise AnalysisError(
            "Landscape score rows do not equal the validated analyzed decoy population: "
            f"{len(score_table)} vs {expected_landscape_rows}."
        )
    if not no_infinite_values(
        score_table,
        DELTA_COLUMNS + ["Structural Preservation Index", "Binding Perturbation Index", "structure_preservation", "binding_perturbation"],
    ):
        raise AnalysisError("Infinite values detected in derived landscape metrics.")

    print("Missing-value counts for landscape components:")
    for metric, count in missing_counts.items():
        print(f"  {metric}: {int(count)}")
    print(f"Rows excluded from landscape due to missing metrics: {excluded_missing_metrics}")


def write_csv(table: pd.DataFrame, path: Path) -> None:
    """Write a CSV without an index."""

    table.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    """Run the DELPHI-VHH experiment analysis workflow."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.structure_threshold is None:
        try:
            args.structure_threshold = default_dockq_preservation_threshold()
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    start_time = time.perf_counter()

    input_dir = args.input_dir
    out_dir = args.out_dir
    table_dir = args.tables_dir or out_dir / "tables"
    figure_dir = args.figures_dir or out_dir / "figures"
    hist_dir = figure_dir / "component_histograms"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading input tables...")
        paths = discover_input_files(input_dir)
        tables = load_tables(paths)
        dockq = load_optional_dockq_table(input_dir)
        vhh_numbering = load_optional_vhh_numbering_table(input_dir)
        if not dockq.empty:
            tables["dockq"] = dockq
        if args.structure_preservation_metric == "dockq" and dockq.empty:
            raise AnalysisError("DockQ structure-preservation metric requested but decoy_dockq.csv is missing or empty.")

        print("Validating schemas...")
        validate_input_schemas(tables)

        comparison = tables["comparison"]
        case_manifest = tables["case_manifest"]
        validation = tables["validation"]
        graft_manifest = tables["graft_manifest"]
        mode_assignments = tables["mode_assignments"]
        dockq = tables.get("dockq", pd.DataFrame())

        print("Selecting analysis population...")
        population, validation_counts = select_analysis_population(
            validation, selected_only=args.selected_only
        )

        print("Joining case and decoy metadata...")
        analysis_table = prepare_analysis_table(
            population,
            comparison,
            case_manifest,
            graft_manifest,
            mode_assignments,
            dockq=dockq,
            require_dockq=args.structure_preservation_metric == "dockq",
        )
        structure_modes = resolve_structure_modes(
            case_manifest, mode_assignments, graft_manifest
        )
        if "redesign_mode" not in analysis_table.columns or analysis_table["redesign_mode"].isna().all():
            analysis_table["redesign_mode"] = attach_mode_to_table(
                analysis_table, structure_modes, case_manifest
            )

        print("Computing dataset counts...")
        dataset_counts = compute_dataset_counts(
            case_manifest,
            validation,
            comparison,
            graft_manifest,
            population,
            validation_counts,
        )

        print("Computing antigen-length statistics...")
        antigen_summary = compute_antigen_length_outputs(
            case_manifest, figure_dir, args.bins, args.dpi
        )

        print("Computing redesign-mode statistics...")
        redesign_counts = compute_redesign_mode_counts(
            case_manifest,
            population,
            graft_manifest,
            structure_modes,
            figure_dir,
            args.dpi,
        )

        print("Computing mutation statistics...")
        mutation_table, mutation_summary, mutation_by_antigen, mismatch_count = compute_mutation_outputs(
            analysis_table,
            figure_dir,
            args.bins,
            args.dpi,
        )

        print("Computing nanobody mutation localization...")
        mutation_locations, mutation_region_summary, mutation_position_frequency, mutation_region_enrichment, mutation_localization_skipped, mutation_numbering_status = compute_mutation_localization_outputs(
            mutation_table,
            numbering=vhh_numbering,
            anarci_python=args.anarci_python,
            anarci_bin=args.anarci_bin,
        )
        if mutation_locations.empty:
            print("No localized nanobody mutations were found; see mutation_localization_skipped_decoys.csv for skipped-row reasons.")

        print("Building amino-acid substitution matrices...")
        substitution_counts, substitution_frequencies = build_substitution_matrices(
            mutation_table,
            include_unchanged=args.include_unchanged,
        )
        save_heatmap(
            substitution_counts,
            figure_dir / "amino_acid_substitution_counts_heatmap.png",
            "Observed Amino-Acid Substitution Counts",
            "Count",
            args.dpi,
            annotate_counts=True,
        )
        save_heatmap(
            substitution_frequencies,
            figure_dir / "amino_acid_substitution_frequencies_heatmap.png",
            "Observed Amino-Acid Substitution Frequencies",
            "Proportion",
            args.dpi,
            annotate_counts=False,
        )

        print("Computing native delta metrics...")
        landscape_source = mutation_table.copy()
        for column in REQUIRED_LANDSCAPE_COLUMNS:
            if column not in landscape_source.columns:
                raise AnalysisError(
                    f"Required DELPHI Landscape metric cannot be constructed: {column}"
                )
        compute_native_deltas(landscape_source)
        required_metrics = landscape_required_metrics(args.structure_preservation_metric)
        for metric in required_metrics:
            if metric not in landscape_source.columns:
                raise AnalysisError(f"Required DELPHI Landscape metric cannot be constructed: {metric}")
            landscape_source[metric] = pd.to_numeric(landscape_source[metric], errors="coerce")

        missing_counts = missing_component_counts(landscape_source, required_metrics)
        complete_mask = mark_landscape_eligibility(landscape_source, args.structure_preservation_metric)
        excluded_missing_metrics = int((~complete_mask).sum())
        unclassified_source = landscape_source.loc[~complete_mask].copy()
        landscape_complete = landscape_source.loc[complete_mask].copy()

        print("Computing component statistics...")
        component_summary = compute_component_summary(landscape_complete if not landscape_complete.empty else landscape_source)

        if landscape_complete.empty:
            print("No analyzed decoys have complete DELPHI Landscape metrics; writing empty ranked outputs and unclassified rows.")
            landscape_scored = landscape_complete.copy()
            for column in [
                "legacy_structure_preservation",
                "Structural Preservation Index",
                "Binding Perturbation Index",
                "structure_preservation",
                "binding_perturbation",
            ]:
                if column not in landscape_scored.columns:
                    landscape_scored[column] = pd.Series(dtype=float)
            if "structure_preservation_metric" not in landscape_scored.columns:
                landscape_scored["structure_preservation_metric"] = args.structure_preservation_metric
        else:
            print("Generating component histograms...")
            for metric in COMPONENT_METRICS:
                save_distribution_plot(
                    landscape_complete[metric],
                    hist_dir / f"{metric}.png",
                    f"{COMPONENT_LABELS[metric]} Distribution",
                    COMPONENT_LABELS[metric],
                    args.bins,
                    args.dpi,
                )

            print("Computing DELPHI Landscape...")
            landscape_scored = compute_landscape_scores(landscape_complete, args.structure_preservation_metric)

        landscape_scored["structure_preservation_threshold"] = args.structure_threshold
        landscape_scored["binding_perturbation_threshold"] = args.binding_threshold
        score_table = build_landscape_score_table(
            landscape_scored, comparison_columns=list(comparison.columns)
        )
        quadrant_ranking, quadrant_summary, quadrant_unclassified = build_quadrant_tables(
            landscape_scored,
            unclassified_source,
            structure_threshold=args.structure_threshold,
            binding_threshold=args.binding_threshold,
        )

        print("Generating figures...")
        save_mutation_region_bar(
            mutation_region_summary,
            figure_dir / "mutation_region_bar.png",
            args.dpi,
        )
        save_mutation_position_frequency_plot(
            mutation_position_frequency,
            figure_dir / "mutation_position_frequency.png",
            args.dpi,
        )

        save_landscape_plot(
            landscape_scored,
            figure_dir / "delphi_landscape.png",
            args.dpi,
            structure_threshold=args.structure_threshold,
            binding_threshold=args.binding_threshold,
        )

        print("Saving outputs...")
        csv_tables = {
            "dataset_counts.csv": dataset_counts,
            "antigen_length_summary.csv": antigen_summary,
            "redesign_mode_counts.csv": redesign_counts,
            "mutation_count_summary.csv": mutation_summary,
            "mutation_summary_by_antigen.csv": mutation_by_antigen,
            "amino_acid_substitution_counts.csv": substitution_counts,
            "amino_acid_substitution_frequencies.csv": substitution_frequencies,
            "component_metric_summary.csv": component_summary,
            "delphi_landscape_scores.csv": score_table,
            "decoy_quadrant_ranking.csv": quadrant_ranking,
            "decoy_quadrant_summary.csv": quadrant_summary,
            "decoy_quadrant_unclassified.csv": quadrant_unclassified,
            "mutation_locations.csv": mutation_locations,
            "mutation_region_summary.csv": mutation_region_summary,
            "mutation_position_frequency.csv": mutation_position_frequency,
            "mutation_region_enrichment.csv": mutation_region_enrichment,
            "mutation_localization_skipped_decoys.csv": mutation_localization_skipped,
            "mutation_numbering_status.csv": mutation_numbering_status,
        }

        for filename, table in csv_tables.items():
            write_csv(table, table_dir / filename)

        print("Validating outputs...")
        validate_outputs(
            table_dir,
            figure_dir,
            csv_tables,
            substitution_counts,
            substitution_frequencies,
            score_table,
            expected_landscape_rows=len(landscape_scored),
            missing_counts=missing_counts,
            excluded_missing_metrics=excluded_missing_metrics,
        )

        elapsed = time.perf_counter() - start_time
        print("Done.")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {out_dir}")
        print(f"Table directory: {table_dir}")
        print(f"Figure directory: {figure_dir}")
        print(f"Number of unique structures: {case_manifest['structure_id'].nunique()}")
        print(f"Number of validation rows: {len(validation)}")
        print(f"Number of selected decoys analyzed: {len(landscape_scored)}")
        print(f"Number of successfully grafted decoys: {int(graft_success_mask(graft_manifest).sum())}")
        print(f"Number of Rosetta-scored decoys: {int(rosetta_success_mask(comparison).sum())}")
        print(f"Number of sequence-length mismatches excluded: {mismatch_count}")
        print(f"Dataset-count CSV path: {table_dir / 'dataset_counts.csv'}")
        print(f"Antigen-length summary path: {table_dir / 'antigen_length_summary.csv'}")
        print(f"Redesign-mode summary path: {table_dir / 'redesign_mode_counts.csv'}")
        print(f"Mutation summary path: {table_dir / 'mutation_count_summary.csv'}")
        print(f"Nanobody mutation-location table path: {table_dir / 'mutation_locations.csv'}")
        print(f"Nanobody mutation-region summary path: {table_dir / 'mutation_region_summary.csv'}")
        print(
            "Mutation heatmap paths: "
            f"{figure_dir / 'amino_acid_substitution_counts_heatmap.png'}, "
            f"{figure_dir / 'amino_acid_substitution_frequencies_heatmap.png'}"
        )
        print(f"Component-metric summary path: {table_dir / 'component_metric_summary.csv'}")
        print(f"Landscape score-table path: {table_dir / 'delphi_landscape_scores.csv'}")
        print(f"Landscape figure path: {figure_dir / 'delphi_landscape.png'}")
        print(f"Component-histogram directory: {hist_dir}")
        print(f"Elapsed runtime: {elapsed:.2f} seconds")
        return 0

    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())





