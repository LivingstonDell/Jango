from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbia.numbering import (
    CANONICAL_NUMBERING_SCHEME,
    IMGT_CDR_SPANS,
    canonical_position_label,
    cdr_name,
    display_region,
    framework_or_cdr,
    imgt_region,
    number_vhh_sequence_imgt,
)

try:  # Optional; enrichment p-values are reported only when SciPy is present.
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - absence is handled explicitly.
    binomtest = None


REGION_ORDER = ["Framework", "CDR1", "CDR2", "CDR3"]
GROUP_COLUMNS = ["experiment_namespace", "backend", "redesign_mode", "mode_policy"]
LOCATION_COLUMNS = [
    "experiment_namespace",
    "structure_id",
    "design_id",
    "native_residue",
    "mutated_residue",
    "native_position",
    "canonical_position",
    "region",
    "framework_or_cdr",
    "cdr_name",
    "native_amino_acid",
    "mutated_amino_acid",
    "backend",
    "redesign_mode",
    "mode_policy",
    "canonical_sort_key",
    "numbering_scheme",
    "numbering_provenance",
    "numbering_status",
    "numbering_chain",
    "numbering_failure_reason",
]
SUMMARY_COLUMNS = [
    "experiment_namespace",
    "backend",
    "redesign_mode",
    "mode_policy",
    "framework_mutations",
    "cdr1_mutations",
    "cdr2_mutations",
    "cdr3_mutations",
    "framework_fraction",
    "cdr1_fraction",
    "cdr2_fraction",
    "cdr3_fraction",
    "total_mutations",
]
POSITION_FREQUENCY_COLUMNS = [
    "canonical_position",
    "canonical_sort_key",
    "anarci_number",
    "anarci_insertion",
    "region",
    "framework_or_cdr",
    "cdr_name",
    "n_decoys_with_position",
    "n_mutations",
    "mutation_frequency",
]
ENRICHMENT_COLUMNS = [
    "experiment_namespace",
    "backend",
    "redesign_mode",
    "mode_policy",
    "region",
    "observed_mutations",
    "expected_mutations",
    "fold_enrichment",
    "region_positions_observed",
    "total_positions_observed",
    "total_mutations",
    "p_value_binomial_two_sided",
    "statistical_test",
]
SKIPPED_COLUMNS = [
    "experiment_namespace",
    "structure_id",
    "design_id",
    "backend",
    "redesign_mode",
    "mode_policy",
    "skip_reason",
    "numbering_status",
    "numbering_failure_reason",
    "numbering_provenance",
    "numbering_scheme",
    "numbering_chain",
]
NUMBERING_STATUS_COLUMNS = [
    "experiment_namespace",
    "structure_id",
    "design_id",
    "backend",
    "redesign_mode",
    "mode_policy",
    "numbering_status",
    "numbering_failure_reason",
    "numbering_provenance",
    "numbering_scheme",
    "numbering_chain",
    "sequence_length",
    "numbered_positions",
]
NATIVE_SEQUENCE_CANDIDATES = [
    "native_nanobody_sequence",
    "nanobody_sequence",
    "native_vhh_sequence",
    "vhh_sequence",
]
DESIGNED_SEQUENCE_CANDIDATES = [
    "designed_nanobody_sequence",
    "mutated_nanobody_sequence",
    "designed_vhh_sequence",
    "mutated_vhh_sequence",
]


def empty_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(columns=LOCATION_COLUMNS),
        pd.DataFrame(columns=SUMMARY_COLUMNS),
        pd.DataFrame(columns=POSITION_FREQUENCY_COLUMNS),
        pd.DataFrame(columns=ENRICHMENT_COLUMNS),
        pd.DataFrame(columns=SKIPPED_COLUMNS),
        pd.DataFrame(columns=NUMBERING_STATUS_COLUMNS),
    )


def first_present_column(table: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in table.columns:
            return column
    return None


def sequence_columns(table: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return explicit nanobody/VHH native and designed sequence columns.

    The generic ``designed_sequence`` column is intentionally not accepted here;
    in the canonical decoy pipeline it currently represents redesigned antigen
    sequence. Nanobody localization needs explicit VHH/nanobody sequence fields.
    """

    return (
        first_present_column(table, NATIVE_SEQUENCE_CANDIDATES),
        first_present_column(table, DESIGNED_SEQUENCE_CANDIDATES),
    )


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _metadata(row: pd.Series) -> dict[str, object]:
    namespace = _first_value(row, ["experiment_namespace", "experiment", "experiment__dockq"])
    backend = _first_value(row, ["backend", "fold_backend", "fold_backend__validation", "fold_backend__comparison"])
    mode_policy = _first_value(row, ["mode_policy", "policy"])
    if not backend and namespace:
        backend = str(namespace).split("_", 1)[0]
    if not mode_policy and namespace:
        text = str(namespace)
        if text.endswith("auto_by_length"):
            mode_policy = "auto_by_length"
        elif text.endswith("manual"):
            mode_policy = "manual"
    return {
        "experiment_namespace": namespace or "unknown",
        "backend": backend or "unknown",
        "redesign_mode": _first_value(row, ["redesign_mode", "selected_redesign_mode"]) or "unknown",
        "mode_policy": mode_policy or "unknown",
    }


def _first_value(row: pd.Series, columns: Iterable[str]) -> str:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if value is not None and not pd.isna(value) and str(value).strip():
                return str(value).strip()
    return ""


def _normal_region(region: object) -> str:
    text = str(region or "").strip()
    if text in {"CDR1", "CDR2", "CDR3"}:
        return text
    if text.upper() in {"FR", "FRAMEWORK"}:
        return "Framework"
    return display_region(text)


def _normalize_numbering_table(numbering: pd.DataFrame | None) -> pd.DataFrame:
    if numbering is None or numbering.empty:
        return pd.DataFrame()
    table = numbering.copy()
    if "sequence_index" not in table.columns:
        return pd.DataFrame()
    table["sequence_index"] = pd.to_numeric(table["sequence_index"], errors="coerce").astype("Int64")
    if "anarci_number" not in table.columns and "canonical_position" in table.columns:
        extracted = table["canonical_position"].astype(str).str.extract(r"^(?P<number>\d+)(?P<insertion>[A-Za-z]*)$")
        table["anarci_number"] = pd.to_numeric(extracted["number"], errors="coerce")
        table["anarci_insertion"] = extracted["insertion"].fillna("")
    if "anarci_insertion" not in table.columns:
        table["anarci_insertion"] = ""
    if "anarci_label" not in table.columns:
        table["anarci_label"] = [
            canonical_position_label(int(num), ins)
            if not pd.isna(num)
            else ""
            for num, ins in zip(table.get("anarci_number", pd.Series(dtype=float)), table["anarci_insertion"])
        ]
    if "imgt_region" not in table.columns:
        table["imgt_region"] = [imgt_region(int(num)) if not pd.isna(num) else "unmapped" for num in table.get("anarci_number", pd.Series(dtype=float))]
    if "anarci_sort_key" not in table.columns:
        table["anarci_sort_key"] = pd.to_numeric(table.get("anarci_number", pd.Series(dtype=float)), errors="coerce")
    if "numbering_scheme" not in table.columns:
        table["numbering_scheme"] = CANONICAL_NUMBERING_SCHEME
    return table.dropna(subset=["sequence_index"])


def _numbering_lookup(numbering: pd.DataFrame | None) -> dict[str, dict[int, dict[str, object]]]:
    table = _normalize_numbering_table(numbering)
    lookups: dict[str, dict[int, dict[str, object]]] = {}
    if table.empty:
        return lookups
    if "structure_id" not in table.columns:
        table["structure_id"] = "__global__"
    for structure_id, subset in table.groupby("structure_id", dropna=False):
        mapping: dict[int, dict[str, object]] = {}
        for row in subset.to_dict("records"):
            index = row.get("sequence_index")
            if pd.isna(index):
                continue
            mapping[int(index)] = row
        lookups[str(structure_id)] = mapping
    return lookups


def _number_native_sequence(
    sequence: str,
    *,
    anarci_python: str | Path | None = None,
    anarci_bin: str | Path | None = None,
) -> dict[int, dict[str, object]]:
    rows = number_vhh_sequence_imgt(sequence, anarci_python=anarci_python, anarci_bin=anarci_bin)
    return {int(row["sequence_index"]): row for row in rows}


def _runtime_provenance(
    *,
    source: str,
    anarci_python: str | Path | None = None,
    anarci_bin: str | Path | None = None,
) -> str:
    if source == "precomputed_table":
        return "precomputed_table"
    if anarci_python is not None and str(anarci_python).strip():
        return f"anarci_python:{Path(anarci_python).expanduser().resolve()}"
    if anarci_bin is not None and str(anarci_bin).strip():
        return f"anarci_bin:{Path(anarci_bin).expanduser().resolve()}"
    return "unavailable"


def _numbering_chain(row: dict[str, object]) -> str:
    series = pd.Series(row)
    return _first_value(
        series,
        [
            "nanobody_chain",
            "native_nanobody_chain",
            "native_nanobody_chain_original",
            "nanobody_chain__graft_manifest",
            "native_nanobody_chain__comparison",
        ],
    )


def _status_row(
    row: dict[str, object],
    metadata: dict[str, object],
    *,
    status: str,
    reason: str = "",
    provenance: str = "",
    scheme: str = CANONICAL_NUMBERING_SCHEME,
    numbered_positions: int = 0,
    sequence_length: int = 0,
) -> dict[str, object]:
    return {
        **metadata,
        "structure_id": _clean_text(row.get("structure_id")),
        "design_id": _clean_text(row.get("design_id")),
        "numbering_status": status,
        "numbering_failure_reason": reason,
        "numbering_provenance": provenance,
        "numbering_scheme": scheme,
        "numbering_chain": _numbering_chain(row),
        "sequence_length": sequence_length,
        "numbered_positions": numbered_positions,
    }


def _skip_row(
    row: dict[str, object],
    metadata: dict[str, object],
    *,
    reason: str,
    provenance: str = "",
    scheme: str = CANONICAL_NUMBERING_SCHEME,
    status: str = "failed",
) -> dict[str, object]:
    return {
        **metadata,
        "structure_id": _clean_text(row.get("structure_id")),
        "design_id": _clean_text(row.get("design_id")),
        "skip_reason": reason,
        "numbering_status": status,
        "numbering_failure_reason": reason,
        "numbering_provenance": provenance,
        "numbering_scheme": scheme,
        "numbering_chain": _numbering_chain(row),
    }


def _mapping_for_row(
    *,
    structure_id: str,
    native_sequence: str,
    precomputed: dict[str, dict[int, dict[str, object]]],
    cache: dict[tuple[str, str, str], dict[int, dict[str, object]]],
    anarci_python: str | Path | None = None,
    anarci_bin: str | Path | None = None,
) -> tuple[dict[int, dict[str, object]], str]:
    if structure_id in precomputed:
        return precomputed[structure_id], "precomputed_table"
    if "__global__" in precomputed:
        return precomputed["__global__"], "precomputed_table"
    runtime_key = str(Path(anarci_python).expanduser().resolve()) if anarci_python else str(Path(anarci_bin).expanduser().resolve()) if anarci_bin else ""
    key = (structure_id, native_sequence, runtime_key)
    if key not in cache:
        cache[key] = _number_native_sequence(native_sequence, anarci_python=anarci_python, anarci_bin=anarci_bin)
    return cache[key], "anarci_python" if anarci_python else "anarci_bin"


def _numbering_matches_native(mapping: dict[int, dict[str, object]], native_sequence: str) -> bool:
    for index, native_aa in enumerate(native_sequence, start=1):
        row = mapping.get(index, {})
        numbered_aa = str(row.get("anarci_aa", "") or "").strip().upper()
        if numbered_aa and numbered_aa != native_aa:
            return False
    return True


def localize_mutations(
    table: pd.DataFrame,
    numbering: pd.DataFrame | None = None,
    *,
    anarci_python: str | Path | None = None,
    anarci_bin: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Localize explicit nanobody/VHH substitutions to canonical IMGT regions.

    Returns ``mutation_locations``, ``position_opportunities``,
    ``skipped_decoys``, and ``numbering_status``. Insertions/deletions are not
    localized here; rows with unequal native/designed sequence lengths are
    reported in ``skipped_decoys``.
    """

    native_col, designed_col = sequence_columns(table)
    if native_col is None or designed_col is None:
        skipped = pd.DataFrame(
            [
                {
                    "experiment_namespace": "unknown",
                    "structure_id": "",
                    "design_id": "",
                    "backend": "unknown",
                    "redesign_mode": "unknown",
                    "mode_policy": "unknown",
                    "skip_reason": "missing_explicit_nanobody_sequence_columns",
                    "numbering_status": "not_attempted",
                    "numbering_failure_reason": "missing_explicit_nanobody_sequence_columns",
                    "numbering_provenance": "",
                    "numbering_scheme": CANONICAL_NUMBERING_SCHEME,
                    "numbering_chain": "",
                }
            ],
            columns=SKIPPED_COLUMNS,
        )
        status = pd.DataFrame(columns=NUMBERING_STATUS_COLUMNS)
        return (
            pd.DataFrame(columns=LOCATION_COLUMNS),
            pd.DataFrame(columns=[*GROUP_COLUMNS, "structure_id", "design_id", "native_position", "canonical_position", "canonical_sort_key", "anarci_number", "anarci_insertion", "region", "framework_or_cdr", "cdr_name"]),
            skipped,
            status,
        )

    precomputed = _numbering_lookup(numbering)
    cache: dict[tuple[str, str, str], dict[int, dict[str, object]]] = {}
    location_rows: list[dict[str, object]] = []
    opportunity_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []

    for row in table.to_dict("records"):
        series = pd.Series(row)
        metadata = _metadata(series)
        structure_id = _clean_text(row.get("structure_id"))
        design_id = _clean_text(row.get("design_id"))
        native_sequence = _clean_text(row.get(native_col)).upper()
        designed_sequence = _clean_text(row.get(designed_col)).upper()
        if not native_sequence or not designed_sequence:
            reason = "missing_nanobody_sequence"
            skipped_rows.append(_skip_row(row, metadata, reason=reason, status="not_attempted"))
            status_rows.append(_status_row(row, metadata, status="not_attempted", reason=reason))
            continue
        if len(native_sequence) != len(designed_sequence):
            reason = "sequence_length_mismatch_or_indel"
            skipped_rows.append(_skip_row(row, metadata, reason=reason, status="not_attempted"))
            status_rows.append(_status_row(row, metadata, status="not_attempted", reason=reason, sequence_length=len(native_sequence)))
            continue
        try:
            mapping, source = _mapping_for_row(
                structure_id=structure_id,
                native_sequence=native_sequence,
                precomputed=precomputed,
                cache=cache,
                anarci_python=anarci_python,
                anarci_bin=anarci_bin,
            )
        except Exception as exc:  # noqa: BLE001 - reported in output table.
            reason = f"numbering_failed:{type(exc).__name__}"
            provenance = _runtime_provenance(source="anarci_python" if anarci_python else "anarci_bin", anarci_python=anarci_python, anarci_bin=anarci_bin)
            skipped_rows.append(_skip_row(row, metadata, reason=reason, provenance=provenance))
            status_rows.append(_status_row(row, metadata, status="failed", reason=reason, provenance=provenance, sequence_length=len(native_sequence)))
            continue
        provenance = _runtime_provenance(source=source, anarci_python=anarci_python, anarci_bin=anarci_bin)
        scheme = str(next(iter(mapping.values())).get("numbering_scheme", CANONICAL_NUMBERING_SCHEME)) if mapping else CANONICAL_NUMBERING_SCHEME
        expected_positions = set(range(1, len(native_sequence) + 1))
        missing_positions = expected_positions.difference(set(mapping))
        mutated_positions = {
            index
            for index, (native_aa, designed_aa) in enumerate(zip(native_sequence, designed_sequence), start=1)
            if native_aa != designed_aa
        }
        if missing_positions.intersection(mutated_positions):
            reason = "mutation_at_unnumbered_position"
            skipped_rows.append(_skip_row(row, metadata, reason=reason, provenance=provenance, scheme=scheme))
            status_rows.append(_status_row(row, metadata, status="failed", reason=reason, provenance=provenance, scheme=scheme, numbered_positions=len(mapping), sequence_length=len(native_sequence)))
            continue
        if not _numbering_matches_native(mapping, native_sequence):
            reason = "numbering_native_sequence_mismatch"
            skipped_rows.append(_skip_row(row, metadata, reason=reason, provenance=provenance, scheme=scheme))
            status_rows.append(_status_row(row, metadata, status="failed", reason=reason, provenance=provenance, scheme=scheme, numbered_positions=len(mapping), sequence_length=len(native_sequence)))
            continue
        status_text = "partial" if missing_positions else "ok"
        status_reason = "" if not missing_positions else "unnumbered_positions:" + ";".join(str(pos) for pos in sorted(missing_positions))
        status_rows.append(_status_row(row, metadata, status=status_text, reason=status_reason, provenance=provenance, scheme=scheme, numbered_positions=len(mapping), sequence_length=len(native_sequence)))
        for index in [pos for pos in range(1, len(native_sequence) + 1) if pos in mapping]:
            numbered = mapping[index]
            internal_region = str(numbered.get("imgt_region", "FR"))
            region = _normal_region(internal_region)
            position_row = {
                **metadata,
                "structure_id": structure_id,
                "design_id": design_id,
                "native_position": index,
                "canonical_position": str(numbered.get("anarci_label", "")),
                "canonical_sort_key": float(numbered.get("anarci_sort_key", numbered.get("anarci_number", index))),
                "anarci_number": int(numbered.get("anarci_number", index)),
                "anarci_insertion": str(numbered.get("anarci_insertion", "") or ""),
                "region": region,
                "framework_or_cdr": framework_or_cdr(internal_region),
                "cdr_name": cdr_name(internal_region),
                "numbering_scheme": scheme,
                "numbering_provenance": provenance,
                "numbering_status": "ok",
                "numbering_chain": _numbering_chain(row),
                "numbering_failure_reason": "",
            }
            opportunity_rows.append(position_row)
            native_aa = native_sequence[index - 1]
            mutated_aa = designed_sequence[index - 1]
            if native_aa == mutated_aa:
                continue
            canonical_position = str(position_row["canonical_position"])
            location_rows.append(
                {
                    **metadata,
                    "structure_id": structure_id,
                    "design_id": design_id,
                    "native_residue": f"{native_aa}{canonical_position}",
                    "mutated_residue": f"{mutated_aa}{canonical_position}",
                    "native_position": index,
                    "canonical_position": canonical_position,
                    "region": region,
                    "framework_or_cdr": position_row["framework_or_cdr"],
                    "cdr_name": position_row["cdr_name"],
                    "native_amino_acid": native_aa,
                    "mutated_amino_acid": mutated_aa,
                    "backend": metadata["backend"],
                    "redesign_mode": metadata["redesign_mode"],
                    "mode_policy": metadata["mode_policy"],
                    "canonical_sort_key": position_row["canonical_sort_key"],
                    "numbering_scheme": scheme,
                    "numbering_provenance": provenance,
                    "numbering_status": "ok",
                    "numbering_chain": _numbering_chain(row),
                    "numbering_failure_reason": "",
                }
            )

    locations = pd.DataFrame(location_rows, columns=LOCATION_COLUMNS)
    opportunities = pd.DataFrame(opportunity_rows)
    skipped = pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS)
    status = pd.DataFrame(status_rows, columns=NUMBERING_STATUS_COLUMNS)
    return locations, opportunities, skipped, status

def summarize_regions(locations: pd.DataFrame, opportunities: pd.DataFrame) -> pd.DataFrame:
    if opportunities.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    groups = opportunities[GROUP_COLUMNS].drop_duplicates().copy()
    counts = (
        locations.groupby([*GROUP_COLUMNS, "region"], dropna=False)
        .size()
        .rename("mutations")
        .reset_index()
        if not locations.empty
        else pd.DataFrame(columns=[*GROUP_COLUMNS, "region", "mutations"])
    )
    rows: list[dict[str, object]] = []
    for group in groups.to_dict("records"):
        mask = pd.Series(True, index=counts.index)
        for column in GROUP_COLUMNS:
            mask &= counts[column].astype(str).eq(str(group[column])) if column in counts.columns else False
        subset = counts.loc[mask] if not counts.empty else pd.DataFrame(columns=counts.columns)
        region_counts = {region: 0 for region in REGION_ORDER}
        for record in subset.to_dict("records"):
            region = _normal_region(record.get("region"))
            if region in region_counts:
                region_counts[region] += int(record.get("mutations", 0))
        total = sum(region_counts.values())
        denom = total if total else 1
        rows.append(
            {
                **group,
                "framework_mutations": region_counts["Framework"],
                "cdr1_mutations": region_counts["CDR1"],
                "cdr2_mutations": region_counts["CDR2"],
                "cdr3_mutations": region_counts["CDR3"],
                "framework_fraction": region_counts["Framework"] / denom if total else 0.0,
                "cdr1_fraction": region_counts["CDR1"] / denom if total else 0.0,
                "cdr2_fraction": region_counts["CDR2"] / denom if total else 0.0,
                "cdr3_fraction": region_counts["CDR3"] / denom if total else 0.0,
                "total_mutations": total,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def position_frequency(locations: pd.DataFrame, opportunities: pd.DataFrame) -> pd.DataFrame:
    if opportunities.empty:
        return pd.DataFrame(columns=POSITION_FREQUENCY_COLUMNS)
    denom = (
        opportunities.groupby(["canonical_position", "canonical_sort_key", "anarci_number", "anarci_insertion", "region", "framework_or_cdr", "cdr_name"], dropna=False)
        .size()
        .rename("n_decoys_with_position")
        .reset_index()
    )
    if locations.empty:
        denom["n_mutations"] = 0
    else:
        counts = locations.groupby("canonical_position", dropna=False).size().rename("n_mutations").reset_index()
        denom = denom.merge(counts, on="canonical_position", how="left")
        denom["n_mutations"] = denom["n_mutations"].fillna(0).astype(int)
    denom["mutation_frequency"] = denom["n_mutations"] / denom["n_decoys_with_position"].replace(0, np.nan)
    denom["mutation_frequency"] = denom["mutation_frequency"].fillna(0.0)
    return denom.sort_values(["canonical_sort_key", "canonical_position"], kind="mergesort")[POSITION_FREQUENCY_COLUMNS]


def enrichment_table(locations: pd.DataFrame, opportunities: pd.DataFrame) -> pd.DataFrame:
    if opportunities.empty:
        return pd.DataFrame(columns=ENRICHMENT_COLUMNS)
    summary = summarize_regions(locations, opportunities)
    opportunity_counts = opportunities.groupby([*GROUP_COLUMNS, "region"], dropna=False).size().rename("region_positions_observed").reset_index()
    total_opportunities = opportunities.groupby(GROUP_COLUMNS, dropna=False).size().rename("total_positions_observed").reset_index()
    rows: list[dict[str, object]] = []
    for group in summary.to_dict("records"):
        total_mutations = int(group.get("total_mutations", 0))
        total_opp_row = total_opportunities.copy()
        for column in GROUP_COLUMNS:
            total_opp_row = total_opp_row[total_opp_row[column].astype(str).eq(str(group[column]))]
        total_opp = int(total_opp_row["total_positions_observed"].iloc[0]) if not total_opp_row.empty else 0
        for region, count_column in [
            ("Framework", "framework_mutations"),
            ("CDR1", "cdr1_mutations"),
            ("CDR2", "cdr2_mutations"),
            ("CDR3", "cdr3_mutations"),
        ]:
            opp = opportunity_counts.copy()
            for column in GROUP_COLUMNS:
                opp = opp[opp[column].astype(str).eq(str(group[column]))]
            opp = opp[opp["region"].astype(str).eq(region)]
            region_opp = int(opp["region_positions_observed"].iloc[0]) if not opp.empty else 0
            observed = int(group.get(count_column, 0))
            expected = (total_mutations * region_opp / total_opp) if total_opp and total_mutations else 0.0
            fold = observed / expected if expected > 0 else np.nan
            p_value = np.nan
            test_name = "not_run"
            if binomtest is not None and total_mutations > 0 and 0 < region_opp < total_opp:
                p = region_opp / total_opp
                p_value = float(binomtest(observed, total_mutations, p).pvalue)
                test_name = "binomial_two_sided"
            rows.append(
                {
                    **{column: group[column] for column in GROUP_COLUMNS},
                    "region": region,
                    "observed_mutations": observed,
                    "expected_mutations": expected,
                    "fold_enrichment": fold,
                    "region_positions_observed": region_opp,
                    "total_positions_observed": total_opp,
                    "total_mutations": total_mutations,
                    "p_value_binomial_two_sided": p_value,
                    "statistical_test": test_name,
                }
            )
    return pd.DataFrame(rows, columns=ENRICHMENT_COLUMNS)


def compute_mutation_localization_outputs(
    table: pd.DataFrame,
    numbering: pd.DataFrame | None = None,
    *,
    anarci_python: str | Path | None = None,
    anarci_bin: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locations, opportunities, skipped, numbering_status = localize_mutations(
        table,
        numbering,
        anarci_python=anarci_python,
        anarci_bin=anarci_bin,
    )
    summary = summarize_regions(locations, opportunities)
    frequency = position_frequency(locations, opportunities)
    enrichment = enrichment_table(locations, opportunities)
    return locations, summary, frequency, enrichment, skipped, numbering_status


def _group_label(row: pd.Series) -> str:
    pieces = [str(row.get(column, "")) for column in ["backend", "redesign_mode", "mode_policy"]]
    pieces = [piece for piece in pieces if piece and piece != "unknown"]
    return " / ".join(pieces) if pieces else str(row.get("experiment_namespace", "experiment"))


def save_mutation_region_bar(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor="white")
    if summary.empty:
        ax.text(0.5, 0.5, "No nanobody mutation localization data", ha="center", va="center")
        ax.set_axis_off()
    else:
        plot = summary.copy()
        plot["group_label"] = plot.apply(_group_label, axis=1)
        x = np.arange(len(plot))
        width = 0.18
        columns = ["framework_mutations", "cdr1_mutations", "cdr2_mutations", "cdr3_mutations"]
        colors = ["#5B6770", "#4C78A8", "#F58518", "#54A24B"]
        for offset, label, column, color in zip([-1.5, -0.5, 0.5, 1.5], REGION_ORDER, columns, colors):
            ax.bar(x + offset * width, plot[column].astype(float), width=width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(plot["group_label"], rotation=25, ha="right")
        ax.set_ylabel("Mutation count")
        ax.set_title("Nanobody Mutations by IMGT Region")
        ax.legend(frameon=False, ncol=4)
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def _add_imgt_bands(ax) -> None:
    colors = {"CDR1": "#d6eaf8", "CDR2": "#f8e2c2", "CDR3": "#d5f5e3"}
    for region, (start, stop) in IMGT_CDR_SPANS.items():
        ax.axvspan(start, stop, color=colors[region], alpha=0.45, lw=0)
        ax.text((start + stop) / 2, 0.98, region, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)


def save_mutation_position_frequency_plot(frequency: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.8), facecolor="white")
    if frequency.empty:
        ax.text(0.5, 0.5, "No canonical nanobody positions available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ordered = frequency.sort_values(["canonical_sort_key", "canonical_position"], kind="mergesort")
        x = ordered["canonical_sort_key"].astype(float)
        _add_imgt_bands(ax)
        ax.plot(x, ordered["mutation_frequency"].astype(float), color="#2F5597", linewidth=1.6)
        ax.scatter(x, ordered["mutation_frequency"].astype(float), color="#2F5597", s=14, zorder=3)
        min_pos = int(math.floor(float(x.min()) / 10.0) * 10)
        max_pos = int(math.ceil(float(x.max()) / 10.0) * 10)
        ax.set_xticks(list(range(min_pos, max_pos + 10, 10)))
        ax.set_xlim(max(0, float(x.min()) - 2), float(x.max()) + 2)
        upper = max(0.05, min(1.0, float(ordered["mutation_frequency"].max()) * 1.2 if not ordered.empty else 0.05))
        ax.set_ylim(0, upper)
        ax.set_xlabel("ANARCI IMGT nanobody position")
        ax.set_ylabel("Mutation frequency")
        ax.set_title("Nanobody Mutation Frequency by Canonical Position")
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
