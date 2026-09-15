"""Case selection and redesign-mode manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REDESIGN_MODES = ("hotspot_only", "interface_only", "full_antigen")
SOURCE_CASE_COLUMNS = [
    "selection_rank",
    "structure_id",
    "pdb_id",
    "nanobody_chain",
    "antigen_chain",
    "source_filename",
    "nanobody_length",
    "antigen_length",
    "antigen_class",
    "dSASA_int",
    "dG_separated",
    "sc_value",
    "fullpose_total_score",
    "cdr3_interface_fraction",
    "residue_contact_pairs_5A",
    "selection_policy",
    "native_rosetta_state",
]


@dataclass(frozen=True)
class ModeThresholds:
    small_antigen_len: int = 120
    medium_antigen_len: int = 200
    max_interface_fraction: float = 0.25


def select_decoy_source_cases(
    *,
    manifest_csv: Path,
    native_features_csv: Path,
    native_relaxed_rosetta_csv: Path,
    out_csv: Path,
    max_structures: int | None = None,
    case_count: int | None = None,
    antigen_length_min: int = 60,
    antigen_length_max: int = 250,
) -> pd.DataFrame:
    """Build the source case manifest from the already input-limited native outputs."""

    requested = max_structures if max_structures is not None else case_count
    if requested is not None and requested <= 0:
        raise ValueError("max_structures must be positive")

    manifest = pd.read_csv(manifest_csv)
    features = pd.read_csv(native_features_csv)
    rosetta = pd.read_csv(native_relaxed_rosetta_csv)

    if "structure_id" not in manifest.columns:
        raise ValueError("manifest must contain structure_id")

    df = manifest.copy()
    df = _merge_nonduplicate_columns(df, features, "structure_id")
    df = _merge_nonduplicate_columns(
        df,
        rosetta,
        "structure_id",
        preferred=[
            "dSASA_int",
            "dG_separated",
            "sc_value",
            "fullpose_total_score",
            "hbonds_int",
            "delta_unsatHbonds",
            "nres_int",
        ],
    )

    if "antigen_class" in df.columns:
        df = df[df["antigen_class"].astype(str).str.lower().eq("protein")].copy()
    if "antigen_length" not in df.columns:
        raise ValueError("case selection requires antigen_length")
    df["antigen_length"] = pd.to_numeric(df["antigen_length"], errors="coerce")
    df = df[df["antigen_length"].between(antigen_length_min, antigen_length_max)].copy()
    df = df.sort_values(["structure_id"], kind="stable").copy()
    df.insert(0, "selection_rank", range(1, len(df) + 1))
    df["selection_policy"] = "all_filtered_from_input_max_no_pdb_dedup"
    df["max_structures_requested"] = requested if requested is not None else ""
    df["native_rosetta_state"] = "native_relaxed"

    ordered = [column for column in SOURCE_CASE_COLUMNS if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]
    out = df[ordered + remaining]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def split_cases_by_mode(
    *,
    base_case_manifest: Path,
    out_dir: Path,
    mode_policy: str,
    modes: list[str],
    thresholds: ModeThresholds = ModeThresholds(),
) -> tuple[dict[str, Path], Path]:
    """Write one decoy case manifest per redesign mode."""

    if not modes:
        raise ValueError("at least one redesign mode is required")
    unknown = sorted(set(modes) - set(REDESIGN_MODES))
    if unknown:
        raise ValueError(f"unknown redesign modes: {unknown}")

    df = pd.read_csv(base_case_manifest)
    if df.empty:
        raise ValueError(f"base decoy case manifest is empty: {base_case_manifest}")

    out_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = out_dir / "auto_mode_assignments.csv"
    if mode_policy == "auto_by_length":
        required = {"structure_id", "antigen_length", "antigen_contact_count"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"base decoy case manifest missing columns for auto mode: {sorted(missing)}")
        df["interface_fraction"] = df.apply(interface_fraction, axis=1)
        df["selected_redesign_mode"] = df.apply(
            lambda row: choose_mode_for_case(row, thresholds),
            axis=1,
        )
        assignments = df.copy()
    elif mode_policy == "manual":
        assignments = df[["structure_id"]].drop_duplicates().copy()
        assignments["selected_redesign_mode"] = ",".join(modes)
    else:
        raise ValueError(f"unsupported mode policy: {mode_policy}")
    assignments.to_csv(assignment_path, index=False)

    mode_manifests: dict[str, Path] = {}
    for mode in REDESIGN_MODES:
        if mode not in modes:
            continue
        if mode_policy == "auto_by_length":
            mode_df = df[df["selected_redesign_mode"] == mode].copy()
        else:
            mode_df = df.copy()
            mode_df["selected_redesign_mode"] = mode
        if mode_df.empty:
            continue
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = mode_dir / "decoy_redesign_case_manifest.csv"
        mode_df.to_csv(manifest_path, index=False)
        mode_manifests[mode] = manifest_path

    if not mode_manifests:
        raise ValueError("mode split produced no runnable case manifests")
    return mode_manifests, assignment_path


def choose_mode_for_case(row: pd.Series, thresholds: ModeThresholds = ModeThresholds()) -> str:
    antigen_len = int(row.get("antigen_length", 0) or 0)
    if antigen_len < thresholds.small_antigen_len:
        return "hotspot_only"
    if interface_fraction(row) > thresholds.max_interface_fraction:
        return "hotspot_only"
    if antigen_len < thresholds.medium_antigen_len:
        return "interface_only"
    return "full_antigen"


def interface_fraction(row: pd.Series) -> float:
    antigen_len = float(row.get("antigen_length", 0) or 0)
    contacts = float(row.get("antigen_contact_count", 0) or 0)
    return contacts / antigen_len if antigen_len else 1.0


def _merge_nonduplicate_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    preferred: list[str] | None = None,
) -> pd.DataFrame:
    if key not in right.columns:
        return left
    if preferred is None:
        cols = [key] + [column for column in right.columns if column != key and column not in left.columns]
    else:
        cols = [key] + [column for column in preferred if column in right.columns and column not in left.columns]
    if len(cols) == 1:
        return left
    return left.merge(right[cols], on=key, how="left", validate="one_to_one")
