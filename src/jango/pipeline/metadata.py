"""Metadata files exchanged between split DELPHI pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


PREFOLD_METADATA = "prefold_manifest.json"
FOLD_METADATA = "fold_manifest.json"
METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PrefoldMetadata:
    schema_version: int
    experiment: str
    backend: str
    case_count: int
    max_structures: int
    count_semantics: str
    mode_policy: str
    raw_dir: str
    native_manifest: str
    native_features: str
    native_relaxed_rosetta: str
    source_case_manifest: str
    base_decoy_case_manifest: str
    mode_manifests: dict[str, str]

    @classmethod
    def create(
        cls,
        *,
        experiment: str,
        backend: str,
        case_count: int,
        mode_policy: str,
        raw_dir: Path,
        native_manifest: Path,
        native_features: Path,
        native_relaxed_rosetta: Path,
        source_case_manifest: Path,
        base_decoy_case_manifest: Path,
        mode_manifests: dict[str, Path],
        max_structures: int | None = None,
        count_semantics: str = "input_max",
    ) -> "PrefoldMetadata":
        return cls(
            schema_version=METADATA_SCHEMA_VERSION,
            experiment=experiment,
            backend=backend,
            case_count=case_count,
            max_structures=max_structures if max_structures is not None else case_count,
            count_semantics=count_semantics,
            mode_policy=mode_policy,
            raw_dir=str(raw_dir),
            native_manifest=str(native_manifest),
            native_features=str(native_features),
            native_relaxed_rosetta=str(native_relaxed_rosetta),
            source_case_manifest=str(source_case_manifest),
            base_decoy_case_manifest=str(base_decoy_case_manifest),
            mode_manifests={mode: str(path) for mode, path in mode_manifests.items()},
        )


@dataclass(frozen=True)
class FoldMetadata:
    schema_version: int
    experiment: str
    backend: str
    prefold_metadata: str
    mode_outputs: dict[str, dict[str, str]]
    execution_mode: str | None = None
    msa_policy: str | None = None
    msa_cache: str | None = None
    msa_mode: str | None = None
    msa_provider: str | None = None
    msa_cache_dir: str | None = None
    msa_input: str | None = None
    msa_format: str | None = None
    msa_database: str | None = None
    msa_max_sequences: int | None = None
    msa_pairing: str | None = None
    msa_reuse_policy: str | None = None
    msa_failure_policy: str | None = None
    allow_single_sequence_fallback: bool = False
    max_decoy_sequences_per_structure: int | None = None

    @classmethod
    def create(
        cls,
        *,
        experiment: str,
        backend: str,
        prefold_metadata: Path,
        mode_outputs: dict[str, dict[str, Path]],
        execution_mode: str | None = None,
        msa_policy: str | None = None,
        msa_mode: str | None = None,
        msa_provider: str | None = None,
        msa_cache_dir: Path | None = None,
        msa_input: Path | None = None,
        msa_format: str | None = None,
        msa_database: str | None = None,
        msa_max_sequences: int | None = None,
        msa_pairing: str | None = None,
        msa_reuse_policy: str | None = None,
        msa_failure_policy: str | None = None,
        allow_single_sequence_fallback: bool = False,
        max_decoy_sequences_per_structure: int | None = None,
    ) -> "FoldMetadata":
        cache_text = str(msa_cache_dir) if msa_cache_dir is not None else None
        return cls(
            schema_version=METADATA_SCHEMA_VERSION,
            experiment=experiment,
            backend=backend,
            prefold_metadata=str(prefold_metadata),
            execution_mode=execution_mode,
            mode_outputs={
                mode: {name: str(path) for name, path in outputs.items()}
                for mode, outputs in mode_outputs.items()
            },
            msa_policy=msa_policy,
            msa_cache=cache_text,
            msa_mode=msa_mode,
            msa_provider=msa_provider,
            msa_cache_dir=cache_text,
            msa_input=str(msa_input) if msa_input is not None else None,
            msa_format=msa_format,
            msa_database=msa_database,
            msa_max_sequences=msa_max_sequences,
            msa_pairing=msa_pairing,
            msa_reuse_policy=msa_reuse_policy,
            msa_failure_policy=msa_failure_policy,
            allow_single_sequence_fallback=allow_single_sequence_fallback,
            max_decoy_sequences_per_structure=max_decoy_sequences_per_structure,
        )


def write_metadata(path: Path, metadata: PrefoldMetadata | FoldMetadata) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")
    return path


def read_prefold_metadata(path: Path) -> PrefoldMetadata:
    data = json.loads(path.read_text())
    return PrefoldMetadata(**data)


def read_fold_metadata(path: Path) -> FoldMetadata:
    data = json.loads(path.read_text())
    return FoldMetadata(**data)

