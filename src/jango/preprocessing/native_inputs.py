"""Unified native-input preprocessing for DELPHI pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Literal

import pandas as pd

from jango.paths import RunPaths
from jango.pipeline.labels import LabelConfig, add_label_columns, normalize_label_schema

from . import sabdab2_cif, sabdab_pdb


InputKind = Literal["tarballs", "sabdab-pdb", "sabdab2-cif"]


@dataclass(frozen=True)
class PreprocessingPaths:
    root: Path
    tarballs: Path
    manifest: Path
    manifest_labeled: Path
    qc: Path


@dataclass(frozen=True)
class PreparedNativeInputs:
    input_kind: InputKind
    raw_dir: Path
    manifest: Path | None = None
    manifest_labeled: Path | None = None
    qc: Path | None = None


def preprocessing_paths(output_root: Path, source: str = "native") -> PreprocessingPaths:
    run = RunPaths.from_output_root(output_root)
    raw = run.raw_source(source)
    root = run.manifests_native
    return PreprocessingPaths(
        root=root,
        tarballs=raw.tarballs,
        manifest=root / "preprocessing_manifest.csv",
        manifest_labeled=root / "preprocessing_manifest_labeled.csv",
        qc=root / "preprocessing_qc.csv",
    )


def validate_preprocessing_options(
    *,
    input_kind: str,
    input_path: Path,
    metadata: Path | None,
) -> None:
    if input_kind not in {"tarballs", "sabdab-pdb", "sabdab2-cif"}:
        raise ValueError(f"unsupported input kind: {input_kind}")
    if input_kind == "tarballs":
        if metadata is not None:
            raise ValueError("--metadata is only used with sabdab-pdb or sabdab2-cif input")
        return
    if metadata is None:
        raise ValueError(f"--metadata is required with --input-kind {input_kind}")
    if input_path == metadata:
        raise ValueError("--input and --metadata must be different paths")


def canonicalize_preprocessing_manifest(
    manifest: pd.DataFrame,
    labels: LabelConfig,
) -> pd.DataFrame:
    """Normalize legacy labels and then enforce the canonical label schema."""

    normalized = normalize_label_schema(manifest)
    without_labels = normalized.drop(
        columns=[
            column
            for column in (
                "source",
                "label",
                "role",
                "confidence",
                "label_type",
                "label_confidence",
            )
            if column in normalized.columns
        ],
        errors="ignore",
    )
    return add_label_columns(without_labels, labels)


def _stage_tarball_subset(input_path: Path, output_root: Path, source: str, limit: int) -> Path:
    """Expose the first N raw tarballs in a run-local directory without duplicating them when possible."""

    if limit <= 0:
        raise ValueError("--limit must be positive")
    paths = preprocessing_paths(output_root, source)
    paths.tarballs.mkdir(parents=True, exist_ok=True)
    selected = sorted(path for path in input_path.glob("*.pdb.tar.gz") if not path.name.startswith("._"))[:limit]
    if not selected:
        raise ValueError(f"no .pdb.tar.gz files found in {input_path}")
    for stale in paths.tarballs.glob("*.pdb.tar.gz"):
        if stale.name not in {path.name for path in selected}:
            stale.unlink()
    for source_path in selected:
        target = paths.tarballs / source_path.name
        if target.exists():
            continue
        try:
            target.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target)
    return paths.tarballs


def prepare_native_inputs(
    *,
    input_kind: InputKind,
    input_path: Path,
    output_root: Path,
    labels: LabelConfig,
    metadata: Path | None = None,
    min_antigen_len: int = 60,
    max_antigen_len: int = 250,
    limit: int | None = None,
) -> PreparedNativeInputs:
    """Prepare native structures into Atlas-compatible tarballs when needed."""

    validate_preprocessing_options(input_kind=input_kind, input_path=input_path, metadata=metadata)
    labels.validate()

    if input_kind == "tarballs":
        raw_dir = _stage_tarball_subset(input_path, output_root, labels.source, limit) if limit is not None else input_path
        return PreparedNativeInputs(input_kind=input_kind, raw_dir=raw_dir)

    paths = preprocessing_paths(output_root, labels.source)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.tarballs.mkdir(parents=True, exist_ok=True)
    assert metadata is not None

    if input_kind == "sabdab-pdb":
        raw = sabdab_pdb.normalize_columns(sabdab_pdb.read_metadata(metadata))
        candidates, metadata_qc = sabdab_pdb.filter_and_normalize_metadata(raw)
        manifest, pdb_qc = sabdab_pdb.prepare_tarballs(
            metadata=candidates,
            raw_pdb_dir=input_path,
            output_dir=paths.tarballs,
        )
        metadata_qc_out = _metadata_qc_for_sabdab_pdb(metadata_qc)
        full_qc = pd.concat([metadata_qc_out, pdb_qc], ignore_index=True, sort=False)
        full_qc.to_csv(paths.qc, index=False)
    elif input_kind == "sabdab2-cif":
        raw = sabdab2_cif.read_table(metadata)
        sabdab2_cif.validate_required_columns(raw)
        candidates, metadata_qc = sabdab2_cif.filter_sabdab2_rows(
            raw,
            min_antigen_len=min_antigen_len,
            max_antigen_len=max_antigen_len,
        )
        sabdab2_cif.convert_cifs_to_tarballs(
            candidates=candidates,
            metadata_qc=metadata_qc,
            cif_dir=input_path,
            output_dir=paths.tarballs,
            manifest_out=paths.manifest,
            qc_out=paths.qc,
            limit=limit,
            clean_output_dir=False,
        )
        manifest = pd.read_csv(paths.manifest)
    else:  # pragma: no cover - validate_preprocessing_options guards this.
        raise ValueError(f"unsupported input kind: {input_kind}")

    if limit is not None and input_kind == "sabdab-pdb":
        manifest = manifest.head(limit).copy()

    labeled = canonicalize_preprocessing_manifest(manifest, labels)
    manifest.to_csv(paths.manifest, index=False)
    labeled.to_csv(paths.manifest_labeled, index=False)

    return PreparedNativeInputs(
        input_kind=input_kind,
        raw_dir=paths.tarballs,
        manifest=paths.manifest,
        manifest_labeled=paths.manifest_labeled,
        qc=paths.qc,
    )


def _metadata_qc_for_sabdab_pdb(metadata_qc: pd.DataFrame) -> pd.DataFrame:
    if metadata_qc.empty:
        return pd.DataFrame(
            columns=[
                "metadata_row",
                "pdb_id",
                "nanobody_chain",
                "antigen_chain",
                "structure_id",
                "status",
                "warnings",
            ]
        )
    keep = [
        "metadata_row",
        "pdb_id",
        "nanobody_chain",
        "antigen_chain",
        "structure_id",
        "metadata_status",
        "metadata_warnings",
    ]
    existing = [column for column in keep if column in metadata_qc.columns]
    return metadata_qc[existing].rename(
        columns={
            "metadata_status": "status",
            "metadata_warnings": "warnings",
        }
    )
