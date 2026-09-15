"""Canonical DELPHI label schema helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LABEL_COLUMNS = ("source", "label", "role", "confidence")
LEGACY_LABEL_RENAMES = {
    "label_type": "role",
    "label_confidence": "confidence",
}


@dataclass(frozen=True)
class LabelConfig:
    """Required user-supplied label metadata."""

    source: str
    label: str
    role: str
    confidence: str

    def validate(self) -> None:
        for field, value in (
            ("source", self.source),
            ("label", self.label),
            ("role", self.role),
            ("confidence", self.confidence),
        ):
            if value is None or str(value).strip() == "":
                raise ValueError(f"{field} is required for the canonical label schema")


def normalize_label_schema(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with legacy label names normalized when present."""

    out = table.copy()
    for old, new in LEGACY_LABEL_RENAMES.items():
        if new not in out.columns and old in out.columns:
            out[new] = out[old]
    return out


def add_label_columns(table: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Attach canonical label columns to every row."""

    config.validate()
    out = table.copy()
    out["source"] = config.source
    out["label"] = config.label
    out["role"] = config.role
    out["confidence"] = config.confidence
    return out


def merge_label_columns(
    *,
    table: pd.DataFrame,
    manifest: pd.DataFrame,
    key: str = "structure_id",
) -> pd.DataFrame:
    """Merge canonical label columns from a manifest-like table."""

    labels = normalize_label_schema(manifest)
    missing = [column for column in (key, *LABEL_COLUMNS) if column not in labels.columns]
    if missing:
        raise ValueError(f"manifest is missing label columns: {missing}")
    return table.merge(
        labels[[key, *LABEL_COLUMNS]],
        on=key,
        how="left",
        validate="one_to_one",
    )
