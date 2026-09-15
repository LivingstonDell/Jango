"""Backend-neutral post-fold QC checkpoint for Jango fold outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from nbia.pdbio import parse_atoms, read_pdb_text
from nbia.residues import AA3_TO_1


CONFIDENCE_TOKENS = (
    "plddt",
    "ptm",
    "iptm",
    "pae",
    "confidence",
    "ranking_confidence",
    "predicted_lddt",
)
CONFIDENCE_PREFIXES = ("esmfold2_", "boltz_", "openfold_", "alphafold_", "colabfold_")
PASS_VALIDATION_STATUSES = {"pass", "passed", "ok", "valid"}

ESSENTIAL_COLUMNS = [
    "experiment_namespace",
    "redesign_mode",
    "structure_id",
    "design_id",
    "backend",
    "fold_backend",
    "sequence_role",
    "is_native_reference",
    "fold_completion_status",
    "validation_status",
    "source_validation_status",
    "selected_decoy",
    "source_selected_decoy",
    "selected_for_folding",
    "fold_submission_present",
    "fold_submission_status",
    "fold_submission_completion_status",
    "fold_eligible",
    "fold_exclusion_reason",
    "structure_path",
    "structure_exists",
    "structure_nonempty",
    "reported_sequence_identity",
    "structure_sequence_identity",
    "sequence_match_status",
    "residue_count",
    "chain_count",
    "chain_ids",
    "msa_source",
    "msa_status",
    "msa_consumed",
    "msa_cache_hit",
    "monomer_tm_score",
    "monomer_ca_rmsd",
    "monomer_aligned_fraction",
    "monomer_matched_ca",
    "fold_reference",
    "monomer_metric_status",
    "confidence_metrics_available",
    "native_or_redesign",
    "final_qc_status",
    "final_qc_exclusion_reason",
    "graft_eligible",
]


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return pd.isna(value)
    except (TypeError, ValueError):
        return False


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if _missing(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _blank(value: object) -> bool:
    if _missing(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none"}


def _explicit_false(value: object) -> bool:
    return not _blank(value) and not _as_bool(value)


def _clean_string(value: object) -> str:
    if _missing(value):
        return ""
    return str(value)


def _sequence_from_row(row: pd.Series) -> str:
    for column in ("sequence", "designed_sequence", "native_antigen_sequence"):
        value = row.get(column)
        if not _missing(value) and str(value).strip():
            return str(value).strip().replace(" ", "").upper()
    return ""


def _path_from_prediction_manifest(output_dir: Path) -> Path | None:
    manifest = output_dir / "opendde_prediction_manifest.csv"
    if not manifest.is_file():
        return None
    try:
        table = pd.read_csv(manifest)
    except Exception:
        return None
    if table.empty or "prediction_path" not in table.columns:
        return None
    if "sample_rank" in table.columns:
        table = table.assign(_sample_rank=pd.to_numeric(table["sample_rank"], errors="coerce")).sort_values(
            "_sample_rank", kind="stable", na_position="last"
        )
    for value in table["prediction_path"]:
        if _blank(value):
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = manifest.parent / path
        if path.is_file():
            return path
    return None


def _path_from_row(row: pd.Series) -> Path | None:
    for column in ("prediction_path", "structure_path", "fold_output_path", "pdb_path"):
        value = row.get(column)
        if not _missing(value) and str(value).strip():
            return Path(str(value))

    output_dir_value = row.get("fold_output_dir")
    if not _missing(output_dir_value) and str(output_dir_value).strip():
        output_dir = Path(str(output_dir_value))
        design_id_value = row.get("design_id")
        design_id = (
            ""
            if _missing(design_id_value)
            else str(design_id_value).strip()
        )

        if design_id:
            expected = output_dir / f"{design_id}.pdb"
            if expected.is_file():
                return expected

        pdbs = sorted(output_dir.glob("*.pdb"))
        if len(pdbs) == 1:
            return pdbs[0]

        manifest_prediction = _path_from_prediction_manifest(output_dir)
        if manifest_prediction is not None:
            return manifest_prediction

    return None


def _residue_key(atom: Any) -> tuple[str, int, str, str]:
    return (atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name)


def _inspect_structure(path: Path | None, expected_sequence: str) -> dict[str, object]:
    if path is None:
        return {
            "structure_path": "",
            "structure_exists": False,
            "structure_nonempty": False,
            "residue_count": 0,
            "chain_count": 0,
            "chain_ids": "",
            "structure_sequence_identity": pd.NA,
            "sequence_match_status": "missing_path",
        }
    exists = path.exists()
    nonempty = exists and path.stat().st_size > 0
    result: dict[str, object] = {
        "structure_path": str(path),
        "structure_exists": exists,
        "structure_nonempty": nonempty,
        "residue_count": 0,
        "chain_count": 0,
        "chain_ids": "",
        "structure_sequence_identity": pd.NA,
        "sequence_match_status": "missing_structure" if not exists else "empty_structure",
    }
    if not nonempty:
        return result
    try:
        atoms = parse_atoms(read_pdb_text(path))
    except Exception as exc:  # pragma: no cover - defensive for corrupt backend outputs
        result["sequence_match_status"] = f"parse_failed:{exc.__class__.__name__}"
        return result
    chain_ids = sorted({atom.chain_id for atom in atoms if atom.residue_name in AA3_TO_1})
    residues = sorted({_residue_key(atom) for atom in atoms if atom.residue_name in AA3_TO_1})
    result["residue_count"] = len(residues)
    result["chain_count"] = len(chain_ids)
    result["chain_ids"] = ";".join(chain_ids)
    if not expected_sequence:
        result["sequence_match_status"] = "not_checked:no_expected_sequence"
        return result
    chain_sequences: list[str] = []
    for chain in chain_ids:
        by_residue = {
            (atom.residue_number, atom.insertion_code): atom.residue_name
            for atom in atoms
            if atom.chain_id == chain and atom.residue_name in AA3_TO_1
        }
        sequence = "".join(AA3_TO_1.get(by_residue[key], "X") for key in sorted(by_residue))
        if sequence:
            chain_sequences.append(sequence)
    observed = chain_sequences[0] if len(chain_sequences) == 1 else "".join(chain_sequences)
    if not observed:
        result["sequence_match_status"] = "not_checked:no_parsed_sequence"
        return result
    matches = sum(1 for a, b in zip(expected_sequence, observed) if a == b)
    denom = max(len(expected_sequence), len(observed))
    identity = matches / denom if denom else pd.NA
    result["structure_sequence_identity"] = identity
    if observed == expected_sequence:
        result["sequence_match_status"] = "exact"
    elif len(observed) != len(expected_sequence):
        result["sequence_match_status"] = "length_mismatch"
    else:
        result["sequence_match_status"] = "sequence_mismatch"
    return result


def confidence_columns(table: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in table.columns:
        lower = column.lower()
        if lower.startswith(CONFIDENCE_PREFIXES) or any(token in lower for token in CONFIDENCE_TOKENS):
            columns.append(column)
    return columns


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validation_tables_from_fold_dir(fold_dir: Path) -> list[tuple[str, Path]]:
    manifest_path = fold_dir / "fold_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"fold manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    tables: list[tuple[str, Path]] = []
    for mode, outputs in sorted(manifest.get("mode_outputs", {}).items()):
        validation = outputs.get("validation")
        if validation:
            tables.append((mode, Path(validation)))
    if not tables:
        raise ValueError(f"no validation tables listed in {manifest_path}")
    return tables


def load_fold_submission_rows(fold_dir: Path) -> pd.DataFrame:
    manifest_path = fold_dir / "fold_manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame()
    manifest = _read_json(manifest_path)
    per_mode: list[Path] = []
    combined_paths: list[Path] = []
    for outputs in manifest.get("mode_outputs", {}).values():
        work_dir_value = outputs.get("work_dir")
        if not work_dir_value:
            continue
        work_dir = Path(work_dir_value)
        per_mode.append(work_dir / "slurm" / "fold_submission_manifest.csv")
        combined_paths.append(work_dir.parent / "slurm" / "fold_submission_manifest.csv")

    frames: list[pd.DataFrame] = []
    seen: set[Path] = set()
    for path in [*per_mode, *combined_paths]:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        table = pd.read_csv(path)
        if table.empty:
            continue
        table["fold_submission_manifest_path"] = str(path)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "design_id" in combined.columns:
        key = ["design_id"]
        if "redesign_mode" in combined.columns:
            key.insert(0, "redesign_mode")
        combined = combined.drop_duplicates(subset=key, keep="last")
    return combined


def load_fold_job_rows(fold_dir: Path) -> pd.DataFrame:
    manifest_path = fold_dir / "fold_manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame()
    manifest = _read_json(manifest_path)
    frames: list[pd.DataFrame] = []
    for mode, outputs in manifest.get("mode_outputs", {}).items():
        work_dir_value = outputs.get("work_dir")
        if not work_dir_value:
            continue
        for path in sorted((Path(work_dir_value) / "job_bundle").glob("*_monomer_jobs.tsv")):
            table = pd.read_csv(path, sep="\t")
            if table.empty:
                continue
            if "redesign_mode" not in table.columns:
                table["redesign_mode"] = mode
            table["fold_job_bundle_path"] = str(path)
            frames.append(table)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "design_id" in combined.columns:
        key = ["design_id"]
        if "redesign_mode" in combined.columns:
            key.insert(0, "redesign_mode")
        combined = combined.drop_duplicates(subset=key, keep="last")
    return combined


def _internal_merge_column(column: str) -> bool:
    return (
        column.endswith("_job")
        or column.endswith("_job_x")
        or column.endswith("_job_y")
        or column.endswith("_submission")
        or column.endswith("_submission_x")
        or column.endswith("_submission_y")
        or column in {
            "fold_submission_manifest_path",
            "fold_submission_manifest_path_x",
            "fold_submission_manifest_path_y",
            "fold_submission_slurm_job_id",
        }
    )


def _fill_from_suffixed_source(merged: pd.DataFrame, target: str, source: str) -> None:
    if source not in merged.columns:
        return
    if target not in merged.columns:
        merged[target] = ""
    mask = merged[target].map(_blank) & ~merged[source].map(_blank)
    if not bool(mask.any()):
        return
    if pd.api.types.is_bool_dtype(merged[target]):
        merged.loc[mask, target] = merged.loc[mask, source].map(_as_bool)
        return
    if not pd.api.types.is_object_dtype(merged[target]) and not pd.api.types.is_string_dtype(merged[target]):
        merged[target] = merged[target].astype("object")
    merged.loc[mask, target] = merged.loc[mask, source]


def _merge_fold_job_rows(validation: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    if validation.empty or jobs.empty or "design_id" not in validation.columns or "design_id" not in jobs.columns:
        return validation
    validation = validation.drop(columns=[column for column in validation.columns if _internal_merge_column(str(column))])
    key = ["design_id"]
    if "redesign_mode" in validation.columns and "redesign_mode" in jobs.columns:
        key.insert(0, "redesign_mode")
    job_columns = [column for column in jobs.columns if column not in key]
    renamed = jobs[[*key, *job_columns]].rename(columns={column: f"{column}_job" for column in job_columns})
    merged = validation.merge(renamed, on=key, how="left", validate="many_to_one")
    for column in [
        "structure_id",
        "sequence_role",
        "sequence",
        "designed_sequence",
        "sequence_hash",
        "fold_backend",
        "fold_input",
        "fold_output_dir",
        "fold_command",
        "msa_source",
        "msa_status",
        "msa_consumed",
        "msa_cache_hit",
    ]:
        _fill_from_suffixed_source(merged, column, f"{column}_job")
    return merged


def _merge_submission_rows(validation: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return validation
    merged = validation.copy()
    merged = merged.drop(columns=[column for column in merged.columns if _internal_merge_column(str(column))])
    submission_columns = [
        "fold_submission_present",
        "fold_submission_status",
        "fold_submission_completion_status",
        "fold_output_dir_submission",
        "sequence_role_submission",
        "structure_id_submission",
        "fold_submission_slurm_job_id",
    ]
    merged = merged.drop(columns=[column for column in submission_columns if column in merged.columns])
    if submission.empty or "design_id" not in merged.columns or "design_id" not in submission.columns:
        merged["fold_submission_present"] = False
        return merged
    key = ["design_id"]
    submission = submission.copy()
    if "redesign_mode" in merged.columns and "redesign_mode" in submission.columns:
        validation_modes = merged["redesign_mode"].fillna("").astype(str).str.strip()
        submission_modes = submission["redesign_mode"].fillna("").astype(str).str.strip()
        if bool(validation_modes.ne("").any()) and bool(submission_modes.ne("").any()):
            key.insert(0, "redesign_mode")
    for frame in (merged, submission):
        for column in key:
            if column in frame.columns:
                frame[column] = frame[column].fillna("").astype(str)
    import_columns = [
        column
        for column in [
            *key,
            "submission_status",
            "completion_status",
            "fold_output_dir",
            "sequence_role",
            "structure_id",
            "slurm_job_id",
            "fold_submission_manifest_path",
        ]
        if column in submission.columns
    ]
    sub = submission[import_columns].copy()
    rename = {
        "submission_status": "fold_submission_status",
        "completion_status": "fold_submission_completion_status",
        "fold_output_dir": "fold_output_dir_submission",
        "sequence_role": "sequence_role_submission",
        "structure_id": "structure_id_submission",
        "slurm_job_id": "fold_submission_slurm_job_id",
    }
    sub = sub.rename(columns=rename)
    merged = merged.merge(sub, on=key, how="left", validate="many_to_one")
    merged["fold_submission_present"] = ~merged.get("fold_submission_status", pd.Series(pd.NA, index=merged.index)).isna()
    for target, source in [("fold_output_dir", "fold_output_dir_submission"), ("sequence_role", "sequence_role_submission"), ("structure_id", "structure_id_submission")]:
        if source not in merged.columns:
            continue
        _fill_from_suffixed_source(merged, target, source)
    return merged


def load_validation_rows(fold_dir: Path, validation_paths: list[Path] | None = None) -> pd.DataFrame:
    if validation_paths:
        tables = [(path.parent.name, path) for path in validation_paths]
    else:
        tables = validation_tables_from_fold_dir(fold_dir)
    frames: list[pd.DataFrame] = []
    for mode, path in tables:
        if not path.exists():
            raise FileNotFoundError(f"fold validation table not found: {path}")
        table = pd.read_csv(path)
        if "redesign_mode" not in table.columns:
            table["redesign_mode"] = mode
        table["fold_validation_path"] = str(path)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "design_id" in combined.columns and "redesign_mode" in combined.columns:
        combined = combined.drop_duplicates(subset=["redesign_mode", "design_id"], keep="last")
    return combined


def _fold_completion_status(path: Path | None) -> str:
    if path is None:
        return "missing_path"
    if not path.exists():
        return "missing"
    if path.stat().st_size <= 0:
        return "empty"
    return "completed"


def _native_reference_key(row: pd.Series) -> tuple[str, str]:
    return (_clean_string(row.get("redesign_mode")).strip(), _clean_string(row.get("structure_id")).strip())


def _is_native_control(row: pd.Series) -> bool:
    return _clean_string(row.get("sequence_role")).strip().lower() == "native_control" or _as_bool(row.get("is_native_reference"))


def _native_reference_paths(table: pd.DataFrame) -> dict[tuple[str, str], Path]:
    references: dict[tuple[str, str], Path] = {}
    for _, row in table.iterrows():
        if not _is_native_control(row):
            continue
        key = _native_reference_key(row)
        if not all(key) or key in references:
            continue
        path = _path_from_row(row)
        if path is not None:
            references[key] = path
    return references


def _monomer_metric_values(
    row: pd.Series,
    structure_path: Path | None,
    reference_paths: dict[tuple[str, str], Path],
    qc_status: str,
    sequence_status: str,
) -> dict[str, object]:
    values: dict[str, object] = {
        "monomer_tm_score": row.get("monomer_tm_score", ""),
        "monomer_ca_rmsd": row.get("monomer_ca_rmsd", ""),
        "monomer_aligned_fraction": row.get("monomer_aligned_fraction", ""),
        "monomer_matched_ca": row.get("monomer_matched_ca", ""),
        "fold_reference": row.get("fold_reference", ""),
        "monomer_metric_status": row.get("monomer_metric_status", ""),
    }
    required = ("monomer_tm_score", "monomer_ca_rmsd", "monomer_aligned_fraction")
    if all(not _blank(values[column]) for column in required):
        if _blank(values["monomer_metric_status"]):
            values["monomer_metric_status"] = "copied_from_validation"
        return values
    if _is_native_control(row):
        values["monomer_metric_status"] = "native_control"
        values["fold_reference"] = values["fold_reference"] or "self"
        return values
    if qc_status != "passed" or sequence_status != "exact" or structure_path is None:
        values["monomer_metric_status"] = "not_computed"
        return values
    reference_path = reference_paths.get(_native_reference_key(row))
    if reference_path is None:
        values["monomer_metric_status"] = "missing_native_refold"
        return values
    if not reference_path.exists() or reference_path.stat().st_size <= 0:
        values["monomer_metric_status"] = "invalid_native_refold"
        values["fold_reference"] = "native_refold"
        return values
    try:
        from nbia.decoys import monomer_validation_metrics, normalize_monomer_chain

        reference_atoms = normalize_monomer_chain(parse_atoms(read_pdb_text(reference_path)))
        prediction_atoms = normalize_monomer_chain(parse_atoms(read_pdb_text(structure_path)))
        metrics = monomer_validation_metrics(reference_atoms, prediction_atoms)
    except Exception as exc:  # pragma: no cover - defensive for corrupt backend outputs
        values["monomer_metric_status"] = f"parse_failed:{exc.__class__.__name__}"
        values["fold_reference"] = "native_refold"
        return values
    values.update(metrics)
    values["fold_reference"] = "native_refold"
    values["monomer_metric_status"] = "computed"
    return values


def _row_selected_for_folding(row: pd.Series) -> bool:
    if _as_bool(row.get("selected_decoy")) or _as_bool(row.get("selected_for_folding")):
        return True
    sequence_role = _clean_string(row.get("sequence_role")).strip().lower()
    if sequence_role and sequence_role != "redesigned_decoy":
        return False
    if not _as_bool(row.get("fold_submission_present")):
        return False
    submission_status = _clean_string(row.get("fold_submission_status")).strip().lower()
    completion_status = _clean_string(row.get("fold_submission_completion_status")).strip().lower()
    if submission_status == "dry_run" or completion_status == "not_submitted":
        return False
    return True


def _current_validation_status(source_status: str, qc_status: str, exclusion_reason: str) -> str:
    if qc_status == "passed":
        return "pass"
    if qc_status == "native_control":
        return "native_reference"
    if exclusion_reason:
        return exclusion_reason
    return source_status


def _row_qc_status(row: pd.Series, completion_status: str, sequence_status: str) -> tuple[str, str, bool]:
    validation_status = _clean_string(row.get("validation_status")).strip().lower()
    sequence_role = _clean_string(row.get("sequence_role")).strip().lower()
    native_control = sequence_role == "native_control" or _as_bool(row.get("is_native_reference"))
    selected_for_folding = _row_selected_for_folding(row)
    if completion_status != "completed":
        return "incomplete", completion_status, False
    if native_control:
        return "native_control", "native_control", False
    if _explicit_false(row.get("fold_eligible")):
        reason = _clean_string(row.get("fold_exclusion_reason")).strip() or "fold_ineligible"
        return "excluded", reason, False
    backend = _clean_string(row.get("fold_backend") or row.get("backend")).strip().lower()
    stale_missing_output = bool(backend) and validation_status == f"missing_{backend}_output" and sequence_status == "exact"
    if validation_status and validation_status not in PASS_VALIDATION_STATUSES and not stale_missing_output:
        return "excluded", f"validation_status:{validation_status}", False
    if sequence_status in {"length_mismatch", "sequence_mismatch"}:
        return "excluded", sequence_status, False
    if selected_for_folding:
        return "passed", "", True
    return "excluded", "not_selected", False


def _float_or_none(value: object) -> float | None:
    if _blank(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _first_float(row: pd.Series, columns: tuple[str, ...]) -> float | None:
    for column in columns:
        if column in row.index:
            value = _float_or_none(row.get(column))
            if value is not None:
                return value
    return None


def _esmfold2_top1_rank_key(row: pd.Series) -> tuple[float, float, float, float, float, float, str]:
    tm_score = _first_float(row, ("monomer_tm_score",))
    ca_rmsd = _first_float(row, ("monomer_ca_rmsd",))
    ptm = _first_float(row, ("esmfold2_ptm", "ptm", "esmfold2_ranking_confidence", "ranking_confidence"))
    mean_plddt = _first_float(row, ("esmfold2_mean_plddt", "mean_plddt", "plddt", "predicted_lddt"))
    mean_pae = _first_float(row, ("esmfold2_mean_pae", "mean_pae", "pae"))
    mpnn_score = _first_float(row, ("mpnn_score",))
    return (
        -(tm_score if tm_score is not None else float("-inf")),
        ca_rmsd if ca_rmsd is not None else float("inf"),
        -(ptm if ptm is not None else float("-inf")),
        -(mean_plddt if mean_plddt is not None else float("-inf")),
        mean_pae if mean_pae is not None else float("inf"),
        mpnn_score if mpnn_score is not None else float("inf"),
        _clean_string(row.get("design_id")),
    )


def _apply_esmfold2_top1_selection(qc: pd.DataFrame) -> pd.DataFrame:
    if qc.empty or "structure_id" not in qc.columns:
        return qc
    index = qc.index
    backend = qc.get("fold_backend", qc.get("backend", pd.Series("", index=index))).fillna("").astype(str).str.lower()
    role = qc.get("sequence_role", pd.Series("", index=index)).fillna("").astype(str).str.lower()
    status = qc.get("final_qc_status", pd.Series("", index=index)).fillna("").astype(str).str.lower()
    graft_eligible = qc.get("graft_eligible", pd.Series(False, index=index)).map(_as_bool)
    candidates = backend.eq("esmfold2") & role.eq("redesigned_decoy") & status.eq("passed") & graft_eligible
    if not bool(candidates.any()):
        return qc

    out = qc.copy()
    if "selection_rank" not in out.columns:
        out["selection_rank"] = pd.NA
    non_top1 = pd.Series(False, index=out.index)
    group_cols = [column for column in ("redesign_mode", "structure_id") if column in out.columns]
    for _, group in out[candidates].groupby(group_cols, dropna=False, sort=False):
        ranked_indexes = sorted(group.index, key=lambda idx: _esmfold2_top1_rank_key(out.loc[idx]))
        for rank, idx in enumerate(ranked_indexes, start=1):
            out.at[idx, "selection_rank"] = rank
        for idx in ranked_indexes[1:]:
            non_top1.at[idx] = True

    out.loc[non_top1, "selected_decoy"] = False
    out.loc[non_top1, "graft_eligible"] = False
    out.loc[non_top1, "validation_status"] = "not_top1"
    out.loc[non_top1, "final_qc_status"] = "excluded"
    out.loc[non_top1, "final_qc_exclusion_reason"] = "not_top1"
    return out


def build_fold_qc_table(fold_dir: Path, validation_paths: list[Path] | None = None) -> pd.DataFrame:
    manifest_path = fold_dir / "fold_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    experiment = str(manifest.get("experiment") or fold_dir.name)
    backend = str(manifest.get("backend") or "")
    validation = load_validation_rows(fold_dir, validation_paths)
    validation = _merge_fold_job_rows(validation, load_fold_job_rows(fold_dir))
    validation = _merge_submission_rows(validation, load_fold_submission_rows(fold_dir))
    if validation.empty:
        return pd.DataFrame(columns=ESSENTIAL_COLUMNS)
    confidence = confidence_columns(validation)
    reference_paths = _native_reference_paths(validation)
    rows: list[dict[str, object]] = []
    for _, source in validation.iterrows():
        structure_path = _path_from_row(source)
        expected_sequence = _sequence_from_row(source)
        structure = _inspect_structure(structure_path, expected_sequence)
        completion = _fold_completion_status(structure_path)
        sequence_status = str(structure["sequence_match_status"])
        qc_status, exclusion_reason, graft_eligible = _row_qc_status(source, completion, sequence_status)
        source_validation_status = source.get("source_validation_status", source.get("validation_status", ""))
        source_selected_decoy = source.get("source_selected_decoy", source.get("selected_decoy", ""))
        selected_for_folding = _row_selected_for_folding(source)
        monomer_metrics = _monomer_metric_values(source, structure_path, reference_paths, qc_status, sequence_status)
        row: dict[str, object] = {
            "experiment_namespace": experiment,
            "redesign_mode": source.get("redesign_mode", ""),
            "structure_id": source.get("structure_id", ""),
            "design_id": source.get("design_id", ""),
            "backend": source.get("fold_backend", backend),
            "fold_backend": source.get("fold_backend", backend),
            "sequence_role": source.get("sequence_role", ""),
            "is_native_reference": source.get("is_native_reference", ""),
            "fold_completion_status": completion,
            "validation_status": _current_validation_status(str(source_validation_status).strip().lower(), qc_status, exclusion_reason),
            "source_validation_status": source_validation_status,
            "selected_decoy": bool(graft_eligible),
            "source_selected_decoy": source_selected_decoy,
            "selected_for_folding": bool(selected_for_folding),
            "fold_submission_present": bool(_as_bool(source.get("fold_submission_present"))),
            "fold_submission_status": source.get("fold_submission_status", ""),
            "fold_submission_completion_status": source.get("fold_submission_completion_status", ""),
            "fold_eligible": source.get("fold_eligible", ""),
            "fold_exclusion_reason": source.get("fold_exclusion_reason", ""),
            "reported_sequence_identity": source.get("sequence_identity", ""),
            "msa_source": source.get("msa_source", ""),
            "msa_status": source.get("msa_status", ""),
            "msa_consumed": source.get("msa_consumed", ""),
            "msa_cache_hit": source.get("msa_cache_hit", ""),
            **monomer_metrics,
            "confidence_metrics_available": ";".join(c for c in confidence if not _missing(source.get(c))),
            "native_or_redesign": "native_control" if _clean_string(source.get("sequence_role")).strip().lower() == "native_control" else "redesign",
            "final_qc_status": qc_status,
            "final_qc_exclusion_reason": exclusion_reason,
            "graft_eligible": graft_eligible,
        }
        row.update(structure)
        if row.get("structure_path"):
            row["prediction_path"] = row["structure_path"]
        else:
            row["prediction_path"] = source.get("prediction_path", "")
        for column in source.index:
            if column not in row and not _internal_merge_column(str(column)):
                row[column] = source.get(column, "")
        if (
            _clean_string(row.get("sequence_role")).strip().lower() == "redesigned_decoy"
            and _blank(row.get("designed_sequence"))
            and not _blank(row.get("sequence"))
        ):
            row["designed_sequence"] = row.get("sequence", "")
        for column in confidence:
            row[column] = source.get(column, "")
        rows.append(row)
    qc = _apply_esmfold2_top1_selection(pd.DataFrame(rows))
    ordered = [column for column in ESSENTIAL_COLUMNS if column in qc.columns]
    extras = [column for column in qc.columns if column not in ordered]
    return qc[ordered + extras]


def write_mode_qc_tables(qc: pd.DataFrame, output_root: Path, experiment: str) -> list[Path]:
    if qc.empty or "redesign_mode" not in qc.columns:
        return []
    written: list[Path] = []
    for mode, table in qc.groupby("redesign_mode", dropna=False, sort=False):
        mode_name = str(mode)
        if _blank(mode_name):
            continue
        path = output_root / "results" / "decoy" / "tables" / experiment / mode_name / "post_fold_qc.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        written.append(path)
    return written


def save_status_figure(qc: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    counts = qc.get("final_qc_status", pd.Series(dtype=str)).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_xlabel("QC status")
    ax.set_ylabel("Fold rows")
    ax.set_title("Post-fold QC status")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def default_output_paths(fold_dir: Path, output_root: Path, make_figure: bool) -> tuple[Path, Path | None]:
    manifest_path = fold_dir / "fold_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    experiment = str(manifest.get("experiment") or fold_dir.name)
    csv_path = output_root / "results" / "decoy" / "tables" / experiment / "all_modes" / "post_fold_qc.csv"
    figure_path = None
    if make_figure:
        figure_path = output_root / "results" / "decoy" / "figures" / experiment / "all_modes" / "post_fold_qc_status.png"
    return csv_path, figure_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fold-qc",
        description="Validate existing fold outputs without rerunning backend folding.",
    )
    parser.add_argument("--fold-dir", required=True, type=Path, help="Directory containing fold_manifest.json.")
    parser.add_argument("--validation", action="append", type=Path, default=None, help="Optional explicit fold validation CSV; may be repeated.")
    parser.add_argument("--output-root", type=Path, default=_env_path("JANGO_TEST_RUN_ROOT") or _env_path("JANGO_OUTPUT_ROOT"))
    parser.add_argument("--out", type=Path, default=None, help="Output QC CSV path.")
    parser.add_argument("--figure-out", type=Path, default=None, help="Optional compact QC status figure path.")
    parser.add_argument("--skip-figure", action="store_true", help="Do not write the default compact QC figure.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out is None and args.output_root is None:
        parser.error("--output-root is required unless --out is supplied")
    qc = build_fold_qc_table(args.fold_dir, args.validation)
    if args.out is None:
        out_path, figure_path = default_output_paths(args.fold_dir, args.output_root, not args.skip_figure)
    else:
        out_path = args.out
        figure_path = None if args.skip_figure else args.figure_out
    if args.figure_out is not None:
        figure_path = args.figure_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(out_path, index=False)
    mode_qc_paths: list[Path] = []
    if args.out is None and args.output_root is not None:
        manifest_path = args.fold_dir / "fold_manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        mode_qc_paths = write_mode_qc_tables(qc, args.output_root, str(manifest.get("experiment") or args.fold_dir.name))
    if figure_path is not None:
        save_status_figure(qc, figure_path)
    print(f"wrote post-fold QC: {len(qc)} rows to {out_path}")
    print(f"post-fold QC statuses: {qc['final_qc_status'].value_counts(dropna=False).to_dict() if 'final_qc_status' in qc.columns else {}}")
    print(f"graft eligible rows: {int(qc['graft_eligible'].fillna(False).astype(bool).sum()) if 'graft_eligible' in qc.columns else 0}")
    if mode_qc_paths:
        print(f"per-mode post-fold QC tables: {len(mode_qc_paths)}")
    if figure_path is not None:
        print(f"post-fold QC figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
