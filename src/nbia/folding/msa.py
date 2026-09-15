"""Backend-neutral MSA models, validation, cache helpers, and provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar


MsaRequirement = Literal["required", "optional", "unsupported"]
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


class MSAError(ValueError):
    """Base error for invalid MSA configuration or artifacts."""


class MSAUnsupportedError(MSAError):
    """Raised when a backend cannot consume the requested MSA mode."""


class MSADeferredError(MSAUnsupportedError):
    """Raised for known MSA capabilities whose local API is not established yet."""


class MSACapability(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    UNSUPPORTED = "unsupported"
    BACKEND_MANAGED = "backend_managed"


class MSAMode(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    DISABLED = "disabled"
    BACKEND_MANAGED = "backend_managed"


class MSAProviderKind(str, Enum):
    PRECOMPUTED = "precomputed"
    ABFORGE_GET_OR_BUILD = "abforge_get_or_build"
    MMSEQS2 = "mmseqs2"
    BACKEND_MANAGED = "backend_managed"
    NONE = "none"


class MSAFormat(str, Enum):
    A3M = "a3m"
    FASTA = "fasta"
    CSV = "csv"
    BOLTZ_CSV = "boltz_csv"
    STOCKHOLM = "stockholm"
    UNKNOWN = "unknown"


class MSAPairing(str, Enum):
    AUTO = "auto"
    PAIRED = "paired"
    UNPAIRED = "unpaired"
    NONE = "none"


class MSAStatus(str, Enum):
    NOT_REQUESTED = "msa_not_requested"
    CACHE_HIT = "msa_cache_hit"
    GENERATED = "msa_generated"
    BACKEND_MANAGED = "msa_backend_managed"
    MISSING = "msa_missing"
    INVALID = "msa_invalid"
    UNSUPPORTED = "msa_unsupported"
    SINGLE_SEQUENCE_EXPLICIT_FALLBACK = "single_sequence_explicit_fallback"


EnumT = TypeVar("EnumT", bound=Enum)


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def parse_enum(enum_cls: type[EnumT], value: str | EnumT | None, field_name: str) -> EnumT:
    if isinstance(value, enum_cls):
        return value
    if value is None:
        raise MSAError(f"{field_name} is required")
    normalized = str(value).strip().lower().replace("-", "_")
    for item in enum_cls:
        if normalized == item.value or normalized == item.name.lower():
            return item
    allowed = ", ".join(item.value for item in enum_cls)
    raise MSAError(f"Invalid {field_name}: {value!r}. Allowed values: {allowed}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def ungapped_sequence(sequence: str) -> str:
    return normalize_sequence(sequence).replace("-", "").replace(".", "")


def a3m_query_sequence(sequence: str) -> str:
    """Return the biological query sequence represented by an A3M row."""

    chars: list[str] = []
    for char in str(sequence):
        if char.isspace() or char in "-." or char.islower():
            continue
        chars.append(char.upper())
    return "".join(chars)


def validate_canonical_amino_acids(sequence: str, *, label: str) -> str:
    normalized = normalize_sequence(sequence)
    invalid = sorted(set(normalized) - CANONICAL_AA)
    if invalid:
        raise MSAError(f"{label} contains invalid amino-acid characters: {''.join(invalid)}")
    return normalized


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(normalize_sequence(sequence).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items())}
    return value


def relative_to_or_none(path: Path, base_dir: Path | None) -> str | None:
    if base_dir is None:
        return None
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return None


@dataclass(frozen=True)
class BackendMSASupport:
    """MSA behavior declared by one folding backend."""

    capability: MSACapability
    accepts_precomputed: bool = False
    generates_msa: bool = False
    accepted_input_formats: tuple[MSAFormat, ...] = ()
    generated_artifact_formats: tuple[MSAFormat, ...] = ()
    paired_multichain: bool = False
    monomer: bool = True
    multichain: bool = False
    single_sequence_fallback_allowed: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MSAConfig:
    """Run-level MSA settings selected by the user or orchestrator."""

    mode: MSAMode
    provider: MSAProviderKind
    cache_dir: Path | None = None
    input_path: Path | None = None
    format: MSAFormat | None = None
    database: str | None = None
    max_sequences: int | None = None
    pairing: MSAPairing = MSAPairing.NONE
    reuse_policy: str = "exact_sequence"
    allow_single_sequence_fallback: bool = False
    script_path: Path | None = None
    tool: str | None = None
    build_if_missing: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        mode: str | MSAMode,
        provider: str | MSAProviderKind | None = None,
        cache_dir: str | Path | None = None,
        input_path: str | Path | None = None,
        format: str | MSAFormat | None = None,
        database: str | None = None,
        max_sequences: int | None = None,
        pairing: str | MSAPairing | None = None,
        reuse_policy: str | None = None,
        allow_single_sequence_fallback: bool = False,
        script_path: str | Path | None = None,
        tool: str | None = None,
        build_if_missing: bool = False,
    ) -> "MSAConfig":
        parsed_mode = parse_enum(MSAMode, mode, "msa mode")
        if provider is None:
            if parsed_mode == MSAMode.DISABLED:
                parsed_provider = MSAProviderKind.NONE
            elif parsed_mode == MSAMode.BACKEND_MANAGED:
                parsed_provider = MSAProviderKind.BACKEND_MANAGED
            else:
                parsed_provider = MSAProviderKind.NONE
        else:
            parsed_provider = parse_enum(MSAProviderKind, provider, "msa provider")
        parsed_format = parse_enum(MSAFormat, format, "msa format") if format is not None else None
        parsed_pairing = parse_enum(MSAPairing, pairing or MSAPairing.NONE, "msa pairing")
        return cls(
            mode=parsed_mode,
            provider=parsed_provider,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            input_path=Path(input_path) if input_path is not None else None,
            format=parsed_format,
            database=database,
            max_sequences=max_sequences,
            pairing=parsed_pairing,
            reuse_policy=reuse_policy or "exact_sequence",
            allow_single_sequence_fallback=allow_single_sequence_fallback,
            script_path=Path(script_path) if script_path is not None else None,
            tool=tool,
            build_if_missing=build_if_missing,
        )

    @classmethod
    def from_legacy_policy(
        cls,
        *,
        policy: str,
        cache_dir: str | Path | None = None,
        allow_single_sequence_fallback: bool = False,
    ) -> "MSAConfig":
        normalized = str(policy).strip().lower()
        if normalized == "unsupported":
            return cls.from_values(mode="disabled", provider="none", cache_dir=cache_dir)
        if normalized == "required":
            return cls.from_values(
                mode="required",
                provider="precomputed" if cache_dir else "none",
                cache_dir=cache_dir,
                allow_single_sequence_fallback=allow_single_sequence_fallback,
            )
        if normalized == "optional":
            return cls.from_values(
                mode="optional",
                provider="precomputed" if cache_dir else "none",
                cache_dir=cache_dir,
                allow_single_sequence_fallback=allow_single_sequence_fallback,
            )
        raise MSAError(f"Invalid legacy MSA policy: {policy!r}")

    def validate_basic(self) -> None:
        if self.max_sequences is not None and self.max_sequences <= 0:
            raise MSAError("msa max_sequences must be positive")
        if self.mode == MSAMode.DISABLED:
            if self.provider != MSAProviderKind.NONE:
                raise MSAError("MSA disabled requires provider=none")
            return
        if self.mode == MSAMode.BACKEND_MANAGED:
            if self.provider != MSAProviderKind.BACKEND_MANAGED:
                raise MSAError("MSA backend-managed mode requires provider=backend_managed")
            return
        if self.provider == MSAProviderKind.PRECOMPUTED:
            if self.input_path is None:
                raise MSAError("precomputed MSA provider requires --msa-input")
            if self.cache_dir is None:
                raise MSAError("precomputed MSA provider requires --msa-cache-dir for provenance manifests")
            return
        if self.provider == MSAProviderKind.ABFORGE_GET_OR_BUILD:
            if self.cache_dir is None:
                raise MSAError("AbForge MSA provider requires --msa-cache-dir/MSA_CACHE_ROOT")
            if self.format not in {None, MSAFormat.A3M, MSAFormat.UNKNOWN}:
                raise MSAError("AbForge ESMFold2 MSA provider currently requires A3M format")
            if self.pairing not in {MSAPairing.UNPAIRED, MSAPairing.NONE}:
                raise MSAError("AbForge ESMFold2 MSA provider currently requires unpaired MSA")
            return
        if self.provider == MSAProviderKind.MMSEQS2:
            if self.cache_dir is None:
                raise MSAError("mmseqs2 MSA provider requires --msa-cache-dir")
            if not self.database:
                raise MSAError("mmseqs2 MSA provider requires --msa-database")
            return
        if self.provider == MSAProviderKind.NONE and self.mode == MSAMode.REQUIRED:
            raise MSAError("MSA required mode needs a provider")

    def to_manifest_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MSARequest:
    """One MSA request for one native or designed sequence."""

    structure_id: str
    sequence: str
    backend: str
    provider: MSAProviderKind
    format: MSAFormat | None = None
    design_id: str | None = None
    chain_ids: tuple[str, ...] = ("A",)
    original_chain_ids: tuple[str, ...] = ()
    remapped_chain_ids: tuple[str, ...] = ("A",)
    pairing: MSAPairing = MSAPairing.NONE
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    source_database: str | None = None
    max_sequences: int | None = None
    reuse_policy: str = "exact_sequence"
    mode: MSAMode = MSAMode.DISABLED

    @property
    def query_sequence_hash(self) -> str:
        return sequence_sha256(self.sequence)

    @property
    def request_hash(self) -> str:
        return stable_json_hash(self.cache_key_payload())

    @property
    def msa_id(self) -> str:
        label = self.design_id or self.structure_id
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
        return f"{safe}_{self.request_hash[:16]}"

    def cache_key_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "provider": _enum_value(self.provider),
            "format": _enum_value(self.format) if self.format else None,
            "query_sequence_hash": self.query_sequence_hash,
            "chain_ids": list(self.chain_ids),
            "original_chain_ids": list(self.original_chain_ids),
            "remapped_chain_ids": list(self.remapped_chain_ids),
            "pairing": _enum_value(self.pairing),
            "source_database": self.source_database,
            "max_sequences": self.max_sequences,
            "reuse_policy": self.reuse_policy,
            "generation_parameters": _jsonable(self.generation_parameters),
            "mode": _enum_value(self.mode),
        }


@dataclass(frozen=True)
class MSAProvenance:
    provider: MSAProviderKind
    tool_name: str | None = None
    tool_version: str | None = None
    source_database: str | None = None
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MSAArtifact:
    msa_id: str
    sequence_hash: str
    query_sequence_hash: str
    structure_id: str
    backend: str
    provider: MSAProviderKind
    format: MSAFormat
    path: Path
    paired_or_unpaired: MSAPairing
    chain_ids: tuple[str, ...]
    sequence_count: int
    generation_status: MSAStatus
    provenance: MSAProvenance
    kind: str = ""
    model_consumed: bool = False
    design_id: str | None = None
    original_chain_ids: tuple[str, ...] = ()
    remapped_chain_ids: tuple[str, ...] = ()
    checksum: str | None = None
    cache_key: str | None = None
    manifest_path: Path | None = None

    def to_manifest_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        data = _jsonable(asdict(self))
        data["path"] = str(self.path)
        data["relative_path"] = relative_to_or_none(self.path, base_dir)
        data["manifest_path"] = str(self.manifest_path) if self.manifest_path else None
        data["relative_manifest_path"] = relative_to_or_none(self.manifest_path, base_dir) if self.manifest_path else None
        data["provenance"] = self.provenance.to_dict()
        return data


@dataclass(frozen=True)
class MSAResolution:
    status: MSAStatus
    config: MSAConfig
    request: MSARequest
    artifact: MSAArtifact | None = None
    message: str = ""

    def to_manifest_fields(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "msa_status": self.status.value,
            "msa_mode": self.config.mode.value,
            "msa_provider": self.config.provider.value,
            "msa_format": self.config.format.value if self.config.format else "",
            "msa_pairing": self.config.pairing.value,
            "msa_reuse_policy": self.config.reuse_policy,
            "msa_allow_single_sequence_fallback": self.config.allow_single_sequence_fallback,
            "msa_message": self.message,
            "msa_request_hash": self.request.request_hash,
            "msa_query_sequence_hash": self.request.query_sequence_hash,
            "msa_config_json": json.dumps(self.config.to_manifest_dict(), sort_keys=True),
            "msa_artifact_id": "",
            "msa_path": "",
            "msa_relative_path": "",
            "msa_sequence_count": "",
            "msa_depth": "",
            "msa_checksum": "",
            "msa_manifest_path": "",
            "msa_relative_manifest_path": "",
            "msa_kind": "",
            "msa_cache_status": self.status.value,
            "msa_cache_hit": False,
            "msa_model_consumed": False,
            "msa_consumed": False,
            "msa_source": "",
            "msa_native_sequence_hash": "",
            "msa_design_sequence_hash": "",
            "msa_native_a3m_path": "",
            "msa_derived_a3m_path": "",
            "msa_mutation_count": "",
            "msa_query_replacement_status": "",
            "msa_length_match_status": "",
            "msa_native_cache_hit": "",
            "msa_provenance_json": "",
        }
        if self.artifact is not None:
            art = self.artifact.to_manifest_dict(base_dir=base_dir)
            generation_parameters = self.artifact.provenance.generation_parameters
            fields.update(
                {
                    "msa_artifact_id": self.artifact.msa_id,
                    "msa_path": str(self.artifact.path),
                    "msa_relative_path": art.get("relative_path") or "",
                    "msa_format": self.artifact.format.value,
                    "msa_sequence_count": self.artifact.sequence_count,
                    "msa_depth": self.artifact.sequence_count,
                    "msa_checksum": self.artifact.checksum or "",
                    "msa_manifest_path": str(self.artifact.manifest_path) if self.artifact.manifest_path else "",
                    "msa_relative_manifest_path": art.get("relative_manifest_path") or "",
                    "msa_kind": self.artifact.kind or self.artifact.paired_or_unpaired.value,
                    "msa_cache_status": self.artifact.generation_status.value,
                    "msa_cache_hit": self.artifact.generation_status == MSAStatus.CACHE_HIT,
                    "msa_model_consumed": self.artifact.model_consumed,
                    "msa_consumed": self.artifact.model_consumed,
                    "msa_source": generation_parameters.get("msa_source", ""),
                    "msa_native_sequence_hash": generation_parameters.get("native_sequence_hash", ""),
                    "msa_design_sequence_hash": generation_parameters.get("design_sequence_hash", ""),
                    "msa_native_a3m_path": generation_parameters.get("native_a3m_path", ""),
                    "msa_derived_a3m_path": generation_parameters.get("derived_a3m_path", ""),
                    "msa_mutation_count": generation_parameters.get("mutation_count", ""),
                    "msa_query_replacement_status": generation_parameters.get("query_replacement_status", ""),
                    "msa_length_match_status": generation_parameters.get("length_match_status", ""),
                    "msa_native_cache_hit": generation_parameters.get("native_msa_cache_hit", ""),
                    "msa_provenance_json": json.dumps(self.artifact.provenance.to_dict(), sort_keys=True),
                }
            )
        return fields


@dataclass(frozen=True)
class MsaInput:
    """Backward-compatible phase-2 input type."""

    path: Path
    format: str
    source: str


@dataclass(frozen=True)
class MsaPolicy:
    """Backward-compatible phase-2 policy type."""

    requirement: MsaRequirement
    cache_dir: Path | None = None

    def validate_for_backend(self, backend_name: str, msa: MsaInput | None) -> None:
        if self.requirement == "required" and msa is None:
            raise ValueError(f"{backend_name} requires an MSA input.")
        if self.requirement == "unsupported" and msa is not None:
            raise ValueError(f"{backend_name} does not support MSA input.")


class MSACache:
    """Deterministic MSA cache rooted under a user-selected output directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cache_dir = self.root / "cache"
        self.manifest_dir = self.root / "manifests"
        self.log_dir = self.root / "logs"

    def ensure(self) -> None:
        for path in (self.cache_dir, self.manifest_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, request: MSARequest, suffix: str) -> Path:
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return self.cache_dir / f"{request.msa_id}{suffix}"

    def manifest_path(self, artifact: MSAArtifact) -> Path:
        return self.manifest_dir / f"{artifact.msa_id}.json"

    def write_manifest(self, artifact: MSAArtifact) -> MSAArtifact:
        self.ensure()
        manifest_path = self.manifest_path(artifact)
        artifact = replace(artifact, manifest_path=manifest_path)
        manifest_path.write_text(json.dumps(artifact.to_manifest_dict(base_dir=self.root), indent=2, sort_keys=True) + "\n")
        return artifact

    def load_precomputed(self, request: MSARequest, path: Path, format: MSAFormat | None = None) -> MSAArtifact:
        self.ensure()
        artifact = build_msa_artifact_from_file(
            request=request,
            path=path,
            format=format,
            status=MSAStatus.CACHE_HIT,
            provider=MSAProviderKind.PRECOMPUTED,
            tool_name="precomputed",
            strict_sequence_match=True,
        )
        return self.write_manifest(artifact)


def detect_msa_format(path: Path, requested: MSAFormat | None = None) -> MSAFormat:
    if requested is not None and requested != MSAFormat.UNKNOWN:
        return requested
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "a3m":
        return MSAFormat.A3M
    if suffix in {"fa", "faa", "fasta"}:
        return MSAFormat.FASTA
    if suffix == "csv":
        return MSAFormat.CSV
    if suffix in {"sto", "stockholm"}:
        return MSAFormat.STOCKHOLM
    return MSAFormat.UNKNOWN


def read_fasta_like_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
            current = []
        else:
            current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def read_fasta_like_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    current: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(current)))
            header = line[1:].strip() or f"record_{len(records) + 1}"
            current = []
        else:
            current.append(line)
    if header is not None:
        records.append((header, "".join(current)))
    if not records:
        raise MSAError(f"MSA contains no FASTA/A3M records: {path}")
    return records


def write_fasta_like_records(path: Path, records: Iterable[tuple[str, str]]) -> None:
    lines: list[str] = []
    for header, sequence in records:
        clean_header = str(header).strip() or "record"
        lines.extend([f">{clean_header}", str(sequence)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_csv_msa_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sequence" not in reader.fieldnames:
            raise MSAError(f"MSA CSV is missing a sequence column: {path}")
        for row in reader:
            seq = str(row.get("sequence", "")).strip()
            if seq:
                sequences.append(seq)
    return sequences


def extract_msa_sequences(path: Path, format: MSAFormat | None = None) -> list[str]:
    if not path.exists():
        raise MSAError(f"MSA file does not exist: {path}")
    detected = detect_msa_format(path, format)
    if detected in {MSAFormat.A3M, MSAFormat.FASTA}:
        sequences = read_fasta_like_sequences(path)
    elif detected in {MSAFormat.CSV, MSAFormat.BOLTZ_CSV}:
        sequences = read_csv_msa_sequences(path)
    else:
        raise MSAError(f"Unsupported MSA format for validation: {detected.value}")
    if not sequences:
        raise MSAError(f"MSA contains no sequences: {path}")
    return sequences


def validate_query_sequence_in_msa(query_sequence: str, msa_sequences: Iterable[str]) -> None:
    query = ungapped_sequence(query_sequence)
    for seq in msa_sequences:
        if ungapped_sequence(seq) == query or a3m_query_sequence(seq) == query:
            return
    raise MSAError("MSA does not contain the ungapped query sequence")


def _safe_path_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return token or "unknown"


def replace_a3m_query_sequence(native_query_row: str, design_sequence: str) -> str:
    design = validate_canonical_amino_acids(design_sequence, label="designed sequence")
    out: list[str] = []
    design_index = 0
    for char in native_query_row:
        if char.isspace():
            continue
        if char in "-." or char.islower():
            out.append(char)
            continue
        if design_index >= len(design):
            raise MSAError("designed sequence is shorter than the native A3M query row")
        out.append(design[design_index])
        design_index += 1
    if design_index != len(design):
        raise MSAError("designed sequence is longer than the native A3M query row")
    return "".join(out)


def derive_a3m_from_native_query(
    *,
    native_a3m_path: Path,
    native_sequence: str,
    design_sequence: str,
    output_path: Path,
) -> dict[str, Any]:
    native = validate_canonical_amino_acids(native_sequence, label="native sequence")
    design = validate_canonical_amino_acids(design_sequence, label="designed sequence")
    if len(native) != len(design):
        raise MSAError("native_reuse requires native and designed sequences to have the same length")

    native_a3m = Path(native_a3m_path).expanduser().resolve()
    if not native_a3m.is_file():
        raise MSAError(f"native_reuse native A3M was not found: {native_a3m}")
    records = read_fasta_like_records(native_a3m)
    query_header, query_row = records[0]
    normalized_query = a3m_query_sequence(query_row)
    if normalized_query != native:
        raise MSAError("native_reuse native A3M query does not match the native antigen sequence")

    derived_query = replace_a3m_query_sequence(query_row, design)
    if len(derived_query) != len(query_row):
        raise MSAError("native_reuse query replacement changed A3M alignment width")
    if a3m_query_sequence(derived_query) != design:
        raise MSAError("native_reuse derived A3M query does not match the designed sequence")

    output = Path(output_path).expanduser().resolve()
    write_fasta_like_records(output, [(query_header, derived_query), *records[1:]])
    mutation_count = sum(1 for native_aa, design_aa in zip(native, design) if native_aa != design_aa)
    return {
        "native_sequence_hash": sequence_sha256(native),
        "design_sequence_hash": sequence_sha256(design),
        "native_a3m_path": str(native_a3m),
        "derived_a3m_path": str(output),
        "mutation_count": mutation_count,
        "query_replacement_status": "replaced",
        "length_match_status": "matched",
        "msa_depth": len(records),
        "alignment_width": len(query_row),
        "homolog_rows_preserved": len(records) - 1,
    }


def build_msa_artifact_from_file(
    *,
    request: MSARequest,
    path: Path,
    format: MSAFormat | None,
    status: MSAStatus,
    provider: MSAProviderKind,
    tool_name: str | None,
    strict_sequence_match: bool,
) -> MSAArtifact:
    resolved_path = Path(path).expanduser().resolve()
    detected_format = detect_msa_format(resolved_path, format)
    sequences = extract_msa_sequences(resolved_path, detected_format)
    if strict_sequence_match:
        validate_query_sequence_in_msa(request.sequence, sequences)
    checksum = file_sha256(resolved_path)
    return MSAArtifact(
        msa_id=request.msa_id,
        sequence_hash=request.query_sequence_hash,
        query_sequence_hash=request.query_sequence_hash,
        structure_id=request.structure_id,
        design_id=request.design_id,
        backend=request.backend,
        provider=provider,
        format=detected_format,
        path=resolved_path,
        paired_or_unpaired=request.pairing,
        chain_ids=request.chain_ids,
        original_chain_ids=request.original_chain_ids,
        remapped_chain_ids=request.remapped_chain_ids,
        sequence_count=len(sequences),
        generation_status=status,
        provenance=MSAProvenance(
            provider=provider,
            tool_name=tool_name,
            source_database=request.source_database,
            generation_parameters=request.generation_parameters,
        ),
        kind=request.pairing.value if request.pairing != MSAPairing.NONE else "unpaired",
        checksum=checksum,
        cache_key=request.request_hash,
    )


def validate_config_for_backend(config: MSAConfig, support: BackendMSASupport, backend_name: str) -> None:
    config.validate_basic()
    if config.pairing == MSAPairing.PAIRED and not support.paired_multichain:
        raise MSAUnsupportedError(f"{backend_name} does not support paired multichain MSA input")
    if config.mode == MSAMode.DISABLED:
        return
    if support.capability == MSACapability.UNSUPPORTED:
        if config.mode == MSAMode.OPTIONAL and config.allow_single_sequence_fallback:
            return
        raise MSAUnsupportedError(f"{backend_name} does not support external MSA input")
    if config.mode == MSAMode.BACKEND_MANAGED:
        if not support.generates_msa:
            raise MSAUnsupportedError(f"{backend_name} cannot generate or manage its own MSA")
        return
    if config.provider in {MSAProviderKind.PRECOMPUTED, MSAProviderKind.ABFORGE_GET_OR_BUILD} and not support.accepts_precomputed:
        raise MSADeferredError(f"{backend_name} precomputed MSA input is not wired from local API evidence")
    if config.provider == MSAProviderKind.MMSEQS2:
        raise MSADeferredError("MMseqs2 MSA generation is not implemented in this local phase")


def resolve_msa_for_backend(
    *,
    config: MSAConfig,
    support: BackendMSASupport,
    request: MSARequest,
    backend_name: str,
) -> MSAResolution:
    try:
        validate_config_for_backend(config, support, backend_name)
    except MSAUnsupportedError as exc:
        if config.mode == MSAMode.OPTIONAL and config.allow_single_sequence_fallback:
            return MSAResolution(
                status=MSAStatus.SINGLE_SEQUENCE_EXPLICIT_FALLBACK,
                config=config,
                request=request,
                message=str(exc),
            )
        raise

    if config.mode == MSAMode.DISABLED:
        return MSAResolution(status=MSAStatus.NOT_REQUESTED, config=config, request=request)
    if config.mode == MSAMode.BACKEND_MANAGED:
        return MSAResolution(status=MSAStatus.BACKEND_MANAGED, config=config, request=request)
    if config.provider == MSAProviderKind.PRECOMPUTED:
        assert config.input_path is not None
        assert config.cache_dir is not None
        cache = MSACache(config.cache_dir)
        artifact = cache.load_precomputed(request, config.input_path, config.format)
        artifact = replace(artifact, model_consumed=True)
        return MSAResolution(status=MSAStatus.CACHE_HIT, config=config, request=request, artifact=artifact)
    if config.provider == MSAProviderKind.ABFORGE_GET_OR_BUILD:
        return resolve_abforge_get_or_build_msa(config=config, request=request)
    if config.mode == MSAMode.OPTIONAL and config.allow_single_sequence_fallback:
        return MSAResolution(
            status=MSAStatus.SINGLE_SEQUENCE_EXPLICIT_FALLBACK,
            config=config,
            request=request,
            message="No valid MSA provider was configured; explicit single-sequence fallback was allowed.",
        )
    raise MSAError("No MSA resolution path matched the requested configuration")


def discover_boltz_backend_managed_msa(
    *,
    output_dir: Path,
    request: MSARequest,
    config: MSAConfig,
) -> MSAResolution | None:
    if not output_dir.exists():
        return None
    exact = sorted(output_dir.rglob(f"{request.design_id or request.structure_id}_0.csv"))
    candidates = exact or sorted(output_dir.rglob("msa/*_0.csv"))
    if not candidates:
        return None
    try:
        artifact = build_msa_artifact_from_file(
            request=request,
            path=candidates[0],
            format=MSAFormat.BOLTZ_CSV,
            status=MSAStatus.BACKEND_MANAGED,
            provider=MSAProviderKind.BACKEND_MANAGED,
            tool_name="boltz2",
            strict_sequence_match=True,
        )
    except MSAError as exc:
        return MSAResolution(status=MSAStatus.INVALID, config=config, request=request, message=str(exc))
    return MSAResolution(status=MSAStatus.BACKEND_MANAGED, config=config, request=request, artifact=artifact)


class MSAProvider:
    name: MSAProviderKind

    def resolve(self, request: MSARequest, config: MSAConfig) -> MSAResolution:
        raise NotImplementedError


class NoMSAProvider(MSAProvider):
    name = MSAProviderKind.NONE

    def resolve(self, request: MSARequest, config: MSAConfig) -> MSAResolution:
        return MSAResolution(status=MSAStatus.NOT_REQUESTED, config=config, request=request)


class PrecomputedMSAProvider(MSAProvider):
    name = MSAProviderKind.PRECOMPUTED

    def resolve(self, request: MSARequest, config: MSAConfig) -> MSAResolution:
        config.validate_basic()
        if config.input_path is None or config.cache_dir is None:
            raise MSAError("precomputed provider requires input_path and cache_dir")
        artifact = MSACache(config.cache_dir).load_precomputed(request, config.input_path, config.format)
        return MSAResolution(status=MSAStatus.CACHE_HIT, config=config, request=request, artifact=artifact)


class BackendManagedMSAProvider(MSAProvider):
    name = MSAProviderKind.BACKEND_MANAGED

    def resolve(self, request: MSARequest, config: MSAConfig) -> MSAResolution:
        return MSAResolution(status=MSAStatus.BACKEND_MANAGED, config=config, request=request)


class MMseqs2MSAProvider(MSAProvider):
    name = MSAProviderKind.MMSEQS2

    def resolve(self, request: MSARequest, config: MSAConfig) -> MSAResolution:
        raise MSADeferredError("MMseqs2 generation is deferred until database paths and commands are configured")


class AbForgeGetOrBuildMSAProvider(MSAProvider):
    name = MSAProviderKind.ABFORGE_GET_OR_BUILD

    def resolve(self, request: MSARequest, config: MSAConfig) -> MSAResolution:
        return resolve_abforge_get_or_build_msa(config=config, request=request)


def _abforge_kind(config: MSAConfig) -> str:
    return "paired" if config.pairing == MSAPairing.PAIRED else "unpaired"


def _abforge_direct_cache_path(config: MSAConfig, request: MSARequest) -> Path:
    if config.cache_dir is None:
        raise MSAError("AbForge MSA provider requires cache_dir")
    return Path(config.cache_dir).expanduser().resolve() / "by_sequence_hash" / request.query_sequence_hash / f"{_abforge_kind(config)}.a3m"


def _is_redesigned_native_reuse_request(config: MSAConfig, request: MSARequest) -> bool:
    role = str(request.generation_parameters.get("sequence_role", ""))
    return config.reuse_policy == "native_reuse" and role == "redesigned_decoy"


def _native_reuse_derived_path(config: MSAConfig, request: MSARequest) -> Path:
    root = request.generation_parameters.get("derived_msa_dir")
    if not root:
        raise MSAError("native_reuse requires a run-local derived_msa_dir")
    return (
        Path(str(root)).expanduser().resolve()
        / _safe_path_token(request.structure_id)
        / _safe_path_token(request.design_id or request.msa_id)
        / f"{_abforge_kind(config)}.a3m"
    )


def resolve_abforge_native_reuse_msa(*, config: MSAConfig, request: MSARequest) -> MSAResolution:
    native_sequence = request.generation_parameters.get("native_sequence")
    if not native_sequence:
        raise MSAError("native_reuse requires native_sequence in the MSA request")

    validate_canonical_amino_acids(str(native_sequence), label="native sequence")
    validate_canonical_amino_acids(request.sequence, label="designed sequence")
    native_request = replace(
        request,
        sequence=str(native_sequence),
        design_id=f"{request.structure_id}_native",
        reuse_policy="exact_sequence",
        generation_parameters={
            **request.generation_parameters,
            "sequence_role": "native_control",
            "msa_source": "native_cache",
        },
    )
    native_a3m_path = _abforge_lookup_with_helper(config, native_request) or _abforge_direct_cache_path(config, native_request)
    if not native_a3m_path.is_file():
        raise MSAError(f"native_reuse native A3M was not found: {native_a3m_path}")

    derived_path = _native_reuse_derived_path(config, request)
    details = derive_a3m_from_native_query(
        native_a3m_path=native_a3m_path,
        native_sequence=str(native_sequence),
        design_sequence=request.sequence,
        output_path=derived_path,
    )
    enriched_request = replace(
        request,
        format=MSAFormat.A3M,
        pairing=MSAPairing.UNPAIRED if request.pairing == MSAPairing.NONE else request.pairing,
        generation_parameters={
            **request.generation_parameters,
            **details,
            "msa_source": "native_reuse",
            "native_msa_cache_hit": True,
            "abforge_cache_root": str(config.cache_dir) if config.cache_dir is not None else "",
            "abforge_script": str(config.script_path) if config.script_path is not None else "",
            "abforge_tool": config.tool or "esmfold2",
            "build_if_missing": config.build_if_missing,
            "msa_kind": _abforge_kind(config),
        },
    )
    artifact = build_msa_artifact_from_file(
        request=enriched_request,
        path=derived_path,
        format=MSAFormat.A3M,
        status=MSAStatus.GENERATED,
        provider=MSAProviderKind.ABFORGE_GET_OR_BUILD,
        tool_name="native_reuse",
        strict_sequence_match=True,
    )
    artifact = replace(artifact, model_consumed=True)
    return MSAResolution(status=MSAStatus.GENERATED, config=config, request=request, artifact=artifact)


def _abforge_lookup_with_helper(config: MSAConfig, request: MSARequest) -> Path | None:
    if config.script_path is None:
        return None
    script = Path(config.script_path).expanduser().resolve()
    if not script.is_file():
        raise MSAError(f"AbForge MSA helper does not exist: {script}")
    if config.cache_dir is None:
        raise MSAError("AbForge MSA provider requires cache_dir")
    cmd = [
        sys.executable,
        str(script),
        "--sequence",
        normalize_sequence(request.sequence),
        "--cache-root",
        str(Path(config.cache_dir).expanduser().resolve()),
        "--format",
        "json",
        "--tool",
        config.tool or "esmfold2",
    ]
    if config.build_if_missing:
        cmd.append("--build-if-missing")
    if config.mode == MSAMode.REQUIRED:
        cmd.append("--require")
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip().splitlines()
        message = details[0] if details else "no output"
        raise MSAError(f"AbForge MSA lookup failed: {message}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MSAError(f"AbForge MSA lookup returned invalid JSON: {exc}") from exc
    queries = payload.get("queries") or []
    if not queries:
        return None
    row = queries[0]
    if row.get("status") != "complete":
        return None
    field = "paired_msa_path" if _abforge_kind(config) == "paired" else "unpaired_msa_path"
    value = row.get(field)
    return Path(value).expanduser().resolve() if value else None


def resolve_abforge_get_or_build_msa(*, config: MSAConfig, request: MSARequest) -> MSAResolution:
    config.validate_basic()
    if _is_redesigned_native_reuse_request(config, request):
        return resolve_abforge_native_reuse_msa(config=config, request=request)
    msa_path = _abforge_lookup_with_helper(config, request) or _abforge_direct_cache_path(config, request)
    if not msa_path.is_file():
        if config.mode == MSAMode.OPTIONAL and config.allow_single_sequence_fallback:
            return MSAResolution(
                status=MSAStatus.SINGLE_SEQUENCE_EXPLICIT_FALLBACK,
                config=config,
                request=request,
                message=f"AbForge cached {_abforge_kind(config)} A3M was not found: {msa_path}",
            )
        raise MSAError(f"AbForge cached {_abforge_kind(config)} A3M was not found: {msa_path}")
    enriched_request = replace(
        request,
        format=MSAFormat.A3M,
        pairing=MSAPairing.UNPAIRED if request.pairing == MSAPairing.NONE else request.pairing,
        generation_parameters={
            **request.generation_parameters,
            "abforge_cache_root": str(config.cache_dir) if config.cache_dir is not None else "",
            "abforge_script": str(config.script_path) if config.script_path is not None else "",
            "abforge_tool": config.tool or "esmfold2",
            "build_if_missing": config.build_if_missing,
            "msa_kind": _abforge_kind(config),
            "msa_source": "native_cache",
        },
    )
    artifact = build_msa_artifact_from_file(
        request=enriched_request,
        path=msa_path,
        format=MSAFormat.A3M,
        status=MSAStatus.CACHE_HIT,
        provider=MSAProviderKind.ABFORGE_GET_OR_BUILD,
        tool_name="abforge_get_or_build_msa.py",
        strict_sequence_match=True,
    )
    artifact = replace(artifact, model_consumed=True)
    return MSAResolution(status=MSAStatus.CACHE_HIT, config=config, request=request, artifact=artifact)

