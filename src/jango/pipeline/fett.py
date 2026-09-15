"""User-facing Slurm orchestration for full Jango production runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Mapping, Sequence

from jango.config import experiment_name
from jango.paths import RunPaths, repository_root, resolve_output_root
from jango.pipeline._common import ensure_output_root_outside_repo, env_value
from jango.pipeline.fett_preflight import (
    ARCHIVE_SUFFIXES,
    PreflightReport,
    evaluate_configs,
    has_placeholder,
    validate_decoy_config,
    validate_generated_scripts,
    validate_git_state,
    validate_numeric_options,
    validate_output_root,
    validate_resolved_config,
    validate_runtime_dependencies,
    validate_slurm_resources,
    validate_tarball_input,
)

RUN_MANIFEST = Path("data/manifests/fett_run_manifest.json")
STAGE_MANIFEST = Path("data/manifests/fett_stage_manifest.csv")
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUBMITTED"}
SUCCESS_STATES = {"COMPLETED", "outputs_present"}
FAILURE_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED"}
CPU_STAGE_MEM_ENV = "FETT_CPU_MEM"
DEFAULT_BACKEND = "esmfold2"
DEFAULT_INPUT_KIND = "auto"
DEFAULT_MODE_POLICY = "auto_by_length"
DEFAULT_MAX_DECOY_SEQUENCES = 5
DEFAULT_ROLE = "validation"
DEFAULT_CONFIDENCE = "medium"
DEFAULT_SOURCE = "user_input"
DEFAULT_MAX_STRUCTURES: int | None = None
REDESIGN_MODES = ("hotspot_only", "interface_only", "full_antigen")
FOLD_DATA_BACKENDS = ("esmfold2", "opendde")
DEFAULT_PATHS_CONFIG = Path("configs/production/paths.env")
DEFAULT_RUNTIME_CONFIGS = {"esmfold2": Path("configs/production/esmfold2.env"), "boltz2": Path("configs/production/boltz2.env"), "opendde": Path("configs/production/opendde.env")}
DEFAULT_DECOY_CONFIG = Path("configs/production/antigen_redesign_decoys.yml")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PRE_SOURCE_EXPORT_KEYS = (
    "JANGO_RUN_NAME",
    "JANGO_RUN_ID",
    "JANGO_RUN_ROOT",
    "JANGO_OUTPUT_ROOT",
    "JANGO_WORK_ROOT",
    "JANGO_OUTPUTS_ROOT",
    "JANGO_DATA_ROOT",
    "JANGO_MPNN_ROOT",
    "JANGO_FOLD_ROOT",
    "JANGO_RAW_TARBALLS",
    "JANGO_RUN_EXPERIMENT",
    "JANGO_USER_ROOT",
    "JANGO_USER_CACHE_ROOT",
    "JANGO_USER_MSA_CACHE_ROOT",
    "FOLDING_BACKEND",
    "APPTAINER_CACHEDIR",
    "BOLTZ_CACHE",
)


MAX_STAGE_MEMORY_MIB = 24 * 1024


@dataclass(frozen=True)
class StageResources:
    cpus: str
    mem: str
    time: str
    gpus: str = "0"

    @property
    def uses_gpu(self) -> bool:
        try:
            return int(str(self.gpus)) > 0
        except ValueError:
            return bool(str(self.gpus).strip())


@dataclass(frozen=True)
class StageSpec:
    name: str
    slug: str
    description: str
    resource_prefix: str
    default_resources: StageResources

    @property
    def uses_gpu(self) -> bool:
        return self.default_resources.uses_gpu


STAGES = [
    StageSpec("native_analysis", "native", "Native preprocessing, manifests, features, Rosetta, and prefold case selection", "FETT_NATIVE", StageResources(cpus="8", mem="24G", time="08:00:00")),
    StageSpec("decoy_prepare", "prepare", "ProteinMPNN input/job preparation", "FETT_PREPARE", StageResources(cpus="8", mem="16G", time="04:00:00")),
    StageSpec("proteinmpnn", "proteinmpnn", "CPU-only ProteinMPNN generation", "FETT_PROTEINMPNN", StageResources(cpus="8", mem="16G", time="04:00:00")),
    StageSpec("sequence_validation", "sequence_validation", "ProteinMPNN sequence parsing, filtering, ranking, and selection", "FETT_SEQUENCE_VALIDATION", StageResources(cpus="4", mem="12G", time="02:00:00")),
    StageSpec("msa_resolution", "msa_resolution", "MSA lookup/reuse, fold eligibility, and fold bundle preparation", "FETT_MSA", StageResources(cpus="8", mem="24G", time="08:00:00")),
    StageSpec("decoy_folding", "fold", "Backend folding bundle execution", "FETT_FOLD", StageResources(cpus="8", mem="24G", time="08:00:00", gpus="1")),
    StageSpec("fold_qc", "fold_qc", "Post-fold QC and graft eligibility", "FETT_FOLD_QC", StageResources(cpus="4", mem="12G", time="02:00:00")),
    StageSpec("decoy_metrics_analysis", "analysis", "Grafting, decoy Rosetta/DockQ/landscape/mutation analysis, and final export", "FETT_ANALYSIS", StageResources(cpus="8", mem="24G", time="08:00:00")),
]
STAGE_NAMES = tuple(stage.name for stage in STAGES)
LEGACY_STAGE_MAP = {
    "native_analysis": ("native_analysis",),
    "decoy_redesign": ("decoy_prepare", "proteinmpnn", "sequence_validation", "msa_resolution"),
    "decoy_folding": ("decoy_folding", "fold_qc"),
    "decoy_metrics_analysis": ("decoy_metrics_analysis",),
}


@dataclass(frozen=True)
class InstallationLayout:
    jango_root: Path
    config_root: Path
    user_root_template: str = "/data/{user}/jango"

    @classmethod
    def current(cls) -> "InstallationLayout":
        root = repository_root()
        return cls(root, root / "configs")

    @property
    def paths_config(self) -> Path:
        return self.jango_root / DEFAULT_PATHS_CONFIG

    def runtime_config(self, backend: str) -> Path:
        return self.jango_root / DEFAULT_RUNTIME_CONFIGS[canonical_backend(backend)]

    @property
    def decoy_config(self) -> Path:
        return self.jango_root / DEFAULT_DECOY_CONFIG


@dataclass(frozen=True)
class UserWorkspace:
    root: Path
    work: Path
    outputs: Path
    data: Path
    cache: Path
    registry: Path
    tmp: Path

    @property
    def runs(self) -> Path:
        """Backward-compatible alias for the per-run work root."""

        return self.work

    @property
    def tarballs(self) -> Path:
        return self.data / "tarballs"

    @property
    def msa_cache(self) -> Path:
        return self.data / "msa"

    @property
    def mpnn_data(self) -> Path:
        return self.data / "mpnn"

    @property
    def folds_data(self) -> Path:
        return self.data / "folds"

    @property
    def apptainer_cache(self) -> Path:
        return self.cache / "apptainer"

    @property
    def metadata_cache(self) -> Path:
        return self.cache / "metadata"

    @property
    def boltz_cache(self) -> Path:
        return self.cache / "boltz"

    def create(self) -> None:
        directories = [
            self.work,
            self.outputs,
            self.tarballs,
            self.msa_cache / "by_sequence_hash",
            self.apptainer_cache,
            self.metadata_cache,
            self.boltz_cache,
            self.registry,
            self.tmp,
        ]
        directories.extend(self.mpnn_data / mode for mode in REDESIGN_MODES)
        directories.extend(self.folds_data / backend / mode for backend in FOLD_DATA_BACKENDS for mode in REDESIGN_MODES)
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def resolved_config(self) -> Path:
        return self.root / "resolved_config.json"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance.json"

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    def create(self) -> None:
        directories = [
            self.inputs,
            self.logs / "slurm",
            self.root / "data" / "manifests",
        ]
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RuntimeConfig:
    values: dict[str, str]

    @property
    def backend(self) -> str:
        return self.values.get("FOLDING_BACKEND", "")

    @property
    def partition(self) -> str:
        return self.values.get("SLURM_PARTITION", "batch")

    @property
    def gpus(self) -> str:
        return self.values.get("SLURM_GPUS", "1")

    @property
    def cpus(self) -> str:
        return self.values.get("SLURM_CPUS_PER_TASK", "8")

    @property
    def mem(self) -> str:
        return self.values.get("SLURM_MEM", "20G")

    @property
    def cpu_mem(self) -> str:
        return self.values.get(CPU_STAGE_MEM_ENV, self.mem)

    @property
    def time(self) -> str:
        return self.values.get("SLURM_TIME", "08:00:00")

    def resources_for(self, stage: StageSpec) -> StageResources:
        prefix = stage.resource_prefix
        defaults = stage.default_resources
        return StageResources(
            cpus=self.values.get(f"{prefix}_CPUS", defaults.cpus),
            mem=self.values.get(f"{prefix}_MEM", defaults.mem),
            time=self.values.get(f"{prefix}_TIME", defaults.time),
            gpus=self.values.get(f"{prefix}_GPUS", defaults.gpus),
        )


@dataclass(frozen=True)
class SubmitContext:
    repo_root: Path
    output_root: Path
    paths_config: Path
    runtime_config: Path
    runtime: RuntimeConfig
    experiment: str
    run_id: str
    args: argparse.Namespace


@dataclass(frozen=True)
class ArtifactContract:
    artifact: str
    producer: str
    consumer: str
    producer_path: str
    consumer_path: str
    canonical_path: str

    @property
    def consistent(self) -> bool:
        return self.producer_path == self.consumer_path == self.canonical_path

    def as_row(self) -> dict[str, str]:
        return {
            "artifact": self.artifact,
            "producer": self.producer,
            "consumer": self.consumer,
            "producer_path": self.producer_path,
            "consumer_path": self.consumer_path,
            "canonical_path": self.canonical_path,
            "consistent": str(self.consistent),
        }


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _safe_slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return value[:80] or "jango-run"


def canonical_backend(value: str | None) -> str:
    text = str(value or DEFAULT_BACKEND).strip().lower().replace("-", "_")
    aliases = {"esm": "esmfold2", "esmfold": "esmfold2", "esmfold2": "esmfold2", "boltz": "boltz2", "boltz2": "boltz2", "opendde": "opendde", "open_dde": "opendde"}
    if text not in aliases:
        raise ValueError(f"unsupported backend: {value!r}; expected esmfold2, boltz2, or opendde")
    return aliases[text]


def _current_user() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or getpass.getuser()


def validate_run_id(run_id: str) -> str:
    text = str(run_id or "").strip()
    if not text:
        raise ValueError("run ID is required")
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError("run ID must be a safe name, not a path")
    if has_placeholder(text):
        raise ValueError("run ID contains placeholder text")
    if text != str(run_id) or any(ord(ch) < 32 for ch in text):
        raise ValueError("run ID must not contain leading/trailing whitespace or control characters")
    if not RUN_ID_RE.fullmatch(text):
        raise ValueError("run ID may contain only letters, numbers, dash, underscore, and period")
    return text


def resolve_user_workspace(user: str | None = None) -> UserWorkspace:
    override = os.environ.get("JANGO_USER_ROOT")
    root = Path(override).expanduser() if override else Path("/data") / (user or _current_user()) / "jango"
    root = root.resolve() if root.exists() else root.expanduser().absolute()
    work = Path(os.environ.get("JANGO_WORK_ROOT", root / "work")).expanduser()
    outputs = Path(os.environ.get("JANGO_OUTPUTS_ROOT", root / "outputs")).expanduser()
    data = Path(os.environ.get("JANGO_DATA_ROOT", root / "data")).expanduser()
    return UserWorkspace(root=root, work=work, outputs=outputs, data=data, cache=root / "cache", registry=root / "registry", tmp=root / "tmp")


def resolve_run_layout(run_id: str, output_override: Path | None = None, *, user: str | None = None) -> RunLayout:
    safe_run_id = validate_run_id(run_id)
    if output_override is not None:
        return RunLayout(resolve_output_root(output_override, repo_root=repository_root()))
    workspace = resolve_user_workspace(user)
    return RunLayout(resolve_output_root(workspace.runs / safe_run_id, repo_root=repository_root()))


def _detect_directory_kind(path: Path) -> set[str]:
    kinds: set[str] = set()
    if not path.exists() or not path.is_dir():
        return kinds
    for child in path.iterdir():
        if not child.is_file():
            continue
        name = child.name.lower()
        if any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES):
            kinds.add("tarballs")
        elif name.endswith((".pdb", ".ent")):
            kinds.add("sabdab-pdb")
        elif name.endswith((".cif", ".mmcif")):
            kinds.add("sabdab2-cif")
    return kinds


def detect_input_kind(path: Path) -> str:
    kinds = _detect_directory_kind(path)
    if len(kinds) == 1:
        return next(iter(kinds))
    if not kinds:
        raise ValueError(f"could not detect input kind from {path}; pass --input-kind")
    raise ValueError(f"ambiguous input kind in {path}: {', '.join(sorted(kinds))}; pass --input-kind")


def _resolve_run_root_from_reference(run_id: str | None, output_root: Path | None) -> Path:
    if run_id:
        validate_run_id(run_id)
    if output_root is not None:
        resolved = resolve_output_root(output_root, repo_root=repository_root())
        if run_id and resolved.name != run_id:
            manifest = resolved / RUN_MANIFEST
            manifest_run_id = ""
            if manifest.exists():
                try:
                    manifest_run_id = str(json.loads(manifest.read_text()).get("run_id", ""))
                except json.JSONDecodeError:
                    manifest_run_id = ""
            if manifest_run_id and manifest_run_id != run_id:
                raise SystemExit(f"run ID {run_id!r} conflicts with manifest run_id {manifest_run_id!r} in {resolved}")
            if not manifest_run_id:
                raise SystemExit(f"run ID {run_id!r} conflicts with output root name {resolved.name!r}")
        return resolved
    if not run_id:
        raise SystemExit("provide a run ID or --output-root")
    return resolve_run_layout(run_id).root


def _config_path_or_default(value: Path | None, default: Path, repo_root: Path) -> Path:
    path = value or default
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _version() -> str:
    try:
        return metadata.version("jango")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _expand_env_value(value: str, env: dict[str, str]) -> str:
    value = _strip_quotes(value.strip())

    def default_repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        current = env.get(name)
        return current if current not in {None, ""} else default

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-]-(.*?)\}", default_repl, value)
    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:\?[^}]*\}", lambda m: env.get(m.group(1), ""), value)
    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: env.get(m.group(1), ""), value)
    value = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda m: env.get(m.group(1), ""), value)
    return value


def _parse_env_exports(path: Path, env: dict[str, str]) -> dict[str, str]:
    values = dict(env)
    if not path.exists():
        return values
    common = path.parent / "runtime.<SLURM_NODE>.common.env"
    if common.exists() and common != path:
        values.update(_parse_env_exports(common, values))
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("source "):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        elif "=" not in line or line.startswith("_"):
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            values[name] = _expand_env_value(value, values)
    return values


def _absolute_config_path(path: Path, repo_root: Path | None = None) -> Path:
    if path.is_absolute():
        return path
    return (repo_root or repository_root()) / path


def _normalize_submission_args(args: argparse.Namespace, *, repo_root: Path, report: PreflightReport) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    install = InstallationLayout.current()
    try:
        backend = canonical_backend(getattr(ns, "backend", None) or DEFAULT_BACKEND)
    except ValueError as exc:
        backend = DEFAULT_BACKEND
        report.fail("BACKEND_INVALID", str(exc), remediation="Use esmfold2, boltz2, or opendde.")
    ns.backend = backend

    legacy_case_count = getattr(ns, "case_count", None)
    max_structures = getattr(ns, "max_structures", None)
    if max_structures is not None and legacy_case_count is not None and max_structures != legacy_case_count:
        report.fail("MAX_STRUCTURES_CONFLICT", "--max-structures and deprecated --case-count disagree.", remediation="Pass only --max-structures N.")
    if max_structures is None:
        max_structures = legacy_case_count
    if max_structures is None:
        report.fail("MAX_STRUCTURES_REQUIRED", "--max-structures is required for normal Fett runs.", remediation="Pass --max-structures N.")
        max_structures = 0
    ns.max_structures = int(max_structures)
    ns.case_count = ns.max_structures

    input_value = getattr(ns, "input", None)
    input_dir = getattr(ns, "input_dir", None)
    if input_value is not None and input_dir is not None:
        if Path(input_value).expanduser().resolve() != Path(input_dir).expanduser().resolve():
            report.fail("INPUT_ALIAS_CONFLICT", "--input and --input-dir refer to different paths.", details=f"--input={input_value}\n--input-dir={input_dir}")
        else:
            report.warn("INPUT_ALIAS_REDUNDANT", "--input and --input-dir are identical; --input is enough for normal use.")
    if input_dir is None and input_value is not None:
        input_dir = Path(input_value)
    if input_dir is None:
        report.fail("INPUT_REQUIRED", "--input is required for normal Fett runs.", remediation="Pass --input PATH, or advanced --input-dir PATH.")
        input_dir = Path("")
    ns.input_dir = Path(input_dir)

    if getattr(ns, "input_kind", None) in {None, DEFAULT_INPUT_KIND}:
        try:
            ns.input_kind = detect_input_kind(ns.input_dir)
        except ValueError as exc:
            report.fail("INPUT_KIND_UNDETECTED", str(exc), remediation="Pass --input-kind tarballs, sabdab-pdb, or sabdab2-cif.")
            ns.input_kind = "tarballs"

    run_id = getattr(ns, "run_id", None)
    output_override = getattr(ns, "output_root", None)
    if not run_id and output_override is not None:
        run_id = _safe_slug(Path(output_override).name)
    if not run_id:
        report.fail("RUN_ID_REQUIRED", "--run-id is required unless advanced --output-root is supplied.", remediation="Pass --run-id NAME.")
        run_id = "invalid-run"
    try:
        run_id = validate_run_id(str(run_id))
    except ValueError as exc:
        report.fail("RUN_ID_INVALID", str(exc), remediation="Use letters, numbers, dash, underscore, and period only.")
        run_id = _safe_slug(str(run_id))
    ns.run_id = run_id

    workspace = resolve_user_workspace()
    ns.user_workspace = workspace.root
    ns.user_cache_root = workspace.cache
    ns.user_msa_cache_root = workspace.msa_cache
    if output_override is not None:
        output_path = Path(output_override)
        if output_path.name != run_id:
            manifest = output_path / RUN_MANIFEST
            manifest_run_id = ""
            if manifest.exists():
                try:
                    manifest_run_id = str(json.loads(manifest.read_text()).get("run_id", ""))
                except json.JSONDecodeError:
                    manifest_run_id = ""
            if manifest_run_id and manifest_run_id != run_id:
                report.fail("RUN_REFERENCE_CONFLICT", f"--run-id {run_id!r} conflicts with existing manifest run_id {manifest_run_id!r}.")
            elif not manifest_run_id:
                report.fail("RUN_REFERENCE_CONFLICT", f"--run-id {run_id!r} conflicts with --output-root name {output_path.name!r}.")
        ns.output_root = output_path
    else:
        ns.output_root = workspace.work / run_id

    ns.paths_config = _config_path_or_default(getattr(ns, "paths_config", None), DEFAULT_PATHS_CONFIG, repo_root)
    ns.runtime_config = _config_path_or_default(getattr(ns, "runtime_config", None), DEFAULT_RUNTIME_CONFIGS[backend], repo_root)
    ns.decoy_config = _config_path_or_default(getattr(ns, "decoy_config", None), DEFAULT_DECOY_CONFIG, repo_root)
    ns.mode_policy = getattr(ns, "mode_policy", None) or DEFAULT_MODE_POLICY
    ns.label = getattr(ns, "label", None) or run_id
    ns.role = getattr(ns, "role", None) or DEFAULT_ROLE
    ns.confidence = getattr(ns, "confidence", None) or DEFAULT_CONFIDENCE
    ns.source = getattr(ns, "source", None) or DEFAULT_SOURCE
    ns.max_decoy_sequences = getattr(ns, "max_decoy_sequences", None) or DEFAULT_MAX_DECOY_SEQUENCES
    ns.modes = getattr(ns, "modes", None)
    ns.metadata = getattr(ns, "metadata", None)
    ns.experiment = getattr(ns, "experiment", None)
    return ns


def _base_run_env(args: argparse.Namespace, *, repo_root: Path, output_root: Path, run_id: str, experiment: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    workspace = resolve_user_workspace()
    env.update(
        {
            "JANGO_SOURCE": str(repo_root),
            "JANGO_RUN_NAME": run_id,
            "JANGO_RUN_ID": run_id,
            "JANGO_RUN_ROOT": str(output_root),
            "JANGO_OUTPUT_ROOT": str(output_root),
            "JANGO_DATA_ROOT": str(workspace.data),
            "JANGO_MPNN_ROOT": str(workspace.mpnn_data),
            "JANGO_FOLD_ROOT": str(workspace.folds_data),
        }
    )
    if experiment:
        env["JANGO_RUN_EXPERIMENT"] = experiment
    if getattr(args, "user_workspace", None):
        workspace_root = Path(args.user_workspace)
        env["JANGO_USER_ROOT"] = str(workspace_root)
        env["JANGO_WORK_ROOT"] = str(workspace_root / "work")
        env["JANGO_OUTPUTS_ROOT"] = str(workspace_root / "outputs")
        env["JANGO_DATA_ROOT"] = str(workspace_root / "data")
        env["JANGO_MPNN_ROOT"] = str(workspace_root / "data" / "mpnn")
        env["JANGO_FOLD_ROOT"] = str(workspace_root / "data" / "folds")
    if getattr(args, "user_cache_root", None):
        env["JANGO_USER_CACHE_ROOT"] = str(args.user_cache_root)
        env["APPTAINER_CACHEDIR"] = str(Path(args.user_cache_root) / "apptainer")
        env["BOLTZ_CACHE"] = str(Path(args.user_cache_root) / "boltz")
    if getattr(args, "user_msa_cache_root", None):
        env["JANGO_USER_MSA_CACHE_ROOT"] = str(args.user_msa_cache_root)
    if getattr(args, "source", None):
        env["JANGO_SOURCE_LABEL"] = str(args.source)
    if getattr(args, "input_dir", None) is not None:
        env["JANGO_RAW_TARBALLS"] = str(args.input_dir)
    if getattr(args, "backend", None):
        env["FOLDING_BACKEND"] = str(args.backend)
    return env


def load_runtime(
    paths_config: Path,
    runtime_config: Path,
    *,
    base_env: Mapping[str, str] | None = None,
    runner: Runner = subprocess.run,
    report: PreflightReport | None = None,
) -> RuntimeConfig:
    values, issues = evaluate_configs(
        paths_config=paths_config,
        runtime_config=runtime_config,
        base_env=base_env or os.environ,
        runner=runner,
    )
    if report is not None:
        report.issues.extend(issues)
    return RuntimeConfig(values)


def _effective_config_for_manifest(values: Mapping[str, str]) -> dict[str, str]:
    prefixes = ("JANGO_", "SLURM_", "MSA_")
    keep = {
        "FOLDING_BACKEND",
        "PROTEINMPNN_HOME",
        "PROTEINMPNN_PYTHON",
        "PROTEINMPNN_RUNNER",
        "ESMFOLD2_PYTHON",
        "ESM_ROOT",
        "ESMFOLD2_MODEL",
        "ESMFOLD2_CACHE",
        "ESMC_MODEL_PATH",
        "ESMFOLD2_DEVICE",
        "BOLTZ_EXECUTABLE",
        "BOLTZ_PYTHON",
        "BOLTZ_MODEL_ROOT",
        "BOLTZ_CACHE",
        "ROSETTA_RUNTIME",
        "ROSETTA_IMAGE",
        "APPTAINER_BIN",
        "APPTAINER_CACHEDIR",
        "DOCKQ_PROVIDER",
        "DOCKQ_PYTHON",
        "ANARCI_PYTHON",
        "ANARCI_BIN",
        "EXECUTION_MODE",
        "FOLD_WORKERS_PER_GPU",
    }
    return {key: str(value) for key, value in sorted(values.items()) if key in keep or key.startswith(prefixes)}




def _simple_config_export(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    pattern = re.compile(rf"^(?:export\s+)?{re.escape(name)}=(.*)$")
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip()
        if "#" in value and not value.startswith(("'", '"')):
            value = value.split("#", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return os.path.expandvars(value)
    return None


def _initial_backend(args: argparse.Namespace, runtime_config: Path) -> str:
    return str(args.backend or env_value("FOLDING_BACKEND") or _simple_config_export(runtime_config, "FOLDING_BACKEND") or "")


def _resolve_experiment_key(args: argparse.Namespace, backend: str, report: PreflightReport | None = None) -> str:
    explicit = str(getattr(args, "experiment", "") or "").strip()
    if explicit:
        try:
            RunPaths.from_output_root(args.output_root).decoy_layout(explicit)
        except ValueError as exc:
            if report is not None:
                report.fail("EXPERIMENT_INVALID", str(exc), remediation="Use a safe experiment key or omit --experiment.")
            return _safe_slug(explicit)
        return explicit
    try:
        return experiment_name(backend=backend, max_structures=int(args.max_structures), policy=str(args.mode_policy))
    except (TypeError, ValueError) as exc:
        if report is not None:
            report.fail("EXPERIMENT_INVALID", f"Could not derive experiment key: {exc}")
        return _safe_slug(f"{backend or 'backend'}_max{getattr(args, 'max_structures', getattr(args, 'case_count', 'unknown'))}_{getattr(args, 'mode_policy', 'policy')}")


def _build_context_with_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> tuple[SubmitContext, PreflightReport]:
    repo = repository_root()
    provisional_run_id = str(getattr(args, "run_id", "") or (Path(str(getattr(args, "output_root", "") or "")).name if getattr(args, "output_root", None) else "jango-run"))
    report = PreflightReport(run_id=_safe_slug(provisional_run_id))
    args = _normalize_submission_args(args, repo_root=repo, report=report)
    output_root = ensure_output_root_outside_repo(args.output_root)
    paths_config = _absolute_config_path(args.paths_config, repo)
    runtime_config = _absolute_config_path(args.runtime_config, repo)
    run_id = args.run_id
    report.run_id = run_id
    backend = _initial_backend(args, runtime_config) or args.backend
    experiment = _resolve_experiment_key(args, backend, report)
    base_env = _base_run_env(args, repo_root=repo, output_root=output_root, run_id=run_id, experiment=experiment)
    if backend:
        base_env["FOLDING_BACKEND"] = backend
    runtime = load_runtime(paths_config, runtime_config, base_env=base_env, runner=runner, report=report)
    backend = args.backend or runtime.backend or backend
    if not backend:
        backend = ""
        report.fail("BACKEND_MISSING", "No folding backend could be resolved.", remediation="Pass --backend or use a backend runtime config.")
    elif args.backend and runtime.backend and runtime.backend != args.backend:
        report.fail(
            "BACKEND_CONFLICT",
            f"CLI backend {args.backend!r} conflicts with runtime config FOLDING_BACKEND={runtime.backend!r}.",
            remediation="Use the matching runtime config or remove the conflicting --backend.",
        )
    if backend:
        runtime.values["FOLDING_BACKEND"] = backend
    runtime.values["JANGO_RUN_NAME"] = run_id
    runtime.values["JANGO_RUN_ID"] = run_id
    runtime.values["JANGO_RUN_ROOT"] = str(output_root)
    runtime.values["JANGO_OUTPUT_ROOT"] = str(output_root)
    workspace = resolve_user_workspace()
    runtime.values.setdefault("JANGO_WORK_ROOT", str(workspace.work))
    runtime.values.setdefault("JANGO_OUTPUTS_ROOT", str(workspace.outputs))
    runtime.values.setdefault("JANGO_DATA_ROOT", str(workspace.data))
    runtime.values.setdefault("JANGO_MPNN_ROOT", str(workspace.mpnn_data))
    runtime.values.setdefault("JANGO_FOLD_ROOT", str(workspace.folds_data))
    runtime.values["JANGO_RUN_EXPERIMENT"] = experiment
    runtime.values["JANGO_RAW_TARBALLS"] = str(args.input_dir)
    if getattr(args, "user_workspace", None):
        workspace_root = Path(args.user_workspace)
        runtime.values["JANGO_USER_ROOT"] = str(workspace_root)
        runtime.values["JANGO_WORK_ROOT"] = str(workspace_root / "work")
        runtime.values["JANGO_OUTPUTS_ROOT"] = str(workspace_root / "outputs")
        runtime.values["JANGO_DATA_ROOT"] = str(workspace_root / "data")
        runtime.values["JANGO_MPNN_ROOT"] = str(workspace_root / "data" / "mpnn")
        runtime.values["JANGO_FOLD_ROOT"] = str(workspace_root / "data" / "folds")
    if getattr(args, "user_cache_root", None):
        runtime.values["JANGO_USER_CACHE_ROOT"] = str(args.user_cache_root)
        runtime.values.setdefault("APPTAINER_CACHEDIR", str(Path(args.user_cache_root) / "apptainer"))
        if backend == "boltz2":
            runtime.values.setdefault("BOLTZ_CACHE", str(Path(args.user_cache_root) / "boltz"))
    if getattr(args, "user_msa_cache_root", None):
        runtime.values["JANGO_USER_MSA_CACHE_ROOT"] = str(args.user_msa_cache_root)
    ctx = SubmitContext(repo, output_root, paths_config, runtime_config, runtime, experiment, run_id, args)
    return ctx, report


def _build_context(args: argparse.Namespace) -> SubmitContext:
    ctx, report = _build_context_with_report(args)
    fatal = [issue for issue in report.issues if issue.severity == "FAIL"]
    if fatal:
        raise SystemExit(fatal[0].summary)
    return ctx


def _run_preflight(args: argparse.Namespace, *, manifest: dict[str, object] | None = None, runner: Runner = subprocess.run) -> tuple[SubmitContext, PreflightReport]:
    ctx, report = _build_context_with_report(args, runner=runner)
    args = ctx.args
    validate_numeric_options(report, max_structures=args.max_structures, max_decoy_sequences=args.max_decoy_sequences)
    validate_resolved_config(report, ctx.runtime.values, cli_backend=args.backend)
    if ctx.experiment:
        report.pass_("EXPERIMENT_RESOLVED", f"experiment={ctx.experiment}")
    if args.input_kind == "tarballs":
        input_dir = args.input_dir or Path(ctx.runtime.values.get("JANGO_RAW_TARBALLS", ""))
        validate_tarball_input(report, Path(input_dir))
    elif args.input_dir is not None:
        if Path(args.input_dir).is_dir():
            report.pass_("INPUT_DIR_READABLE", f"Input directory readable: {args.input_dir}")
        else:
            report.fail("INPUT_DIR_NOT_FOUND", f"Input directory does not exist: {args.input_dir}")
    validate_decoy_config(report, Path(args.decoy_config))
    validate_output_root(report, ctx.output_root, ctx.repo_root, ctx.run_id)
    existing_manifest = _existing_run_manifest(ctx.output_root)
    if manifest is None and existing_manifest is not None:
        existing_run_id = str(existing_manifest.get("run_id", ""))
        condition = _stage_status_summary(existing_manifest)
        if existing_run_id and existing_run_id != ctx.run_id:
            report.fail("OUTPUT_ROOT_RUN_CONFLICT", f"Output root contains run_id {existing_run_id!r}, not {ctx.run_id!r}.")
        else:
            report.fail("OUTPUT_ROOT_RUN_EXISTS", f"Run {ctx.run_id!r} already exists and is {condition}; use fett resume {ctx.run_id}.")
    _prepare_workspace(ctx, report)
    validate_runtime_dependencies(report, ctx.runtime.values, backend=ctx.runtime.values.get("FOLDING_BACKEND", ctx.runtime.backend), runner=runner)
    validate_slurm_resources(report, ctx.runtime.values, runner=runner)
    _validate_stage_architecture(report, ctx)
    validate_git_state(report, ctx.repo_root, runner=runner)
    if manifest is not None:
        if manifest.get("paths_config_sha256") and manifest.get("paths_config_sha256") != _sha256(ctx.paths_config):
            report.warn("CONFIG_DRIFT", "paths config hash differs from the original run manifest.", remediation="Review config changes before resume.")
        if manifest.get("runtime_config_sha256") and manifest.get("runtime_config_sha256") != _sha256(ctx.runtime_config):
            report.warn("CONFIG_DRIFT", "runtime config hash differs from the original run manifest.", remediation="Review config changes before resume.")
        if manifest.get("git_commit") and manifest.get("git_commit") != _git_commit(ctx.repo_root):
            report.warn("SOURCE_REVISION_DRIFT", "Source revision differs from the original run manifest.")
        legacy_names = _legacy_stage_names(manifest)
        if legacy_names:
            report.warn(
                "LEGACY_STAGE_SCHEMA",
                "Run manifest predates the expanded Fett stage model.",
                details=_legacy_stage_mapping_details(legacy_names),
                remediation="Resume will use artifact proof where possible and restart from the earliest safe expanded stage otherwise.",
            )
        stale = [row for row in _stage_rows(manifest) if str(row.get("status")) in FAILURE_STATES or str(row.get("completion_status")) in FAILURE_STATES]
        if stale:
            report.warn("STALE_DEPENDENCY_CHAIN", f"Existing run manifest contains {len(stale)} failed/cancelled stage record(s).")
        stored_experiment = str(manifest.get("experiment_key") or manifest.get("experiment") or "")
        legacy_experiment = _legacy_prefold_experiment(ctx.output_root, stored_experiment)
        if legacy_experiment:
            report.warn(
                "LEGACY_DECOY_LAYOUT_DETECTED",
                f"Stored experiment {stored_experiment!r} does not contain prefold metadata; using existing prefold experiment {legacy_experiment!r}.",
                remediation="Resume is supported without moving files; future submissions will store experiment_key explicitly.",
            )
    if not any(issue.severity == "FAIL" and issue.code.startswith("OUTPUT_ROOT") for issue in report.issues):
        _write_run_snapshots(ctx, status="preflight")
        validate_pipeline_contract(report, ctx)
        scripts = [write_stage_script(ctx, stage) for stage in STAGES]
        validate_generated_scripts(report, scripts, runner=runner)
        _validate_generated_stage_scripts(report, scripts)
    return ctx, report


def _emit_preflight(report: PreflightReport, *, json_output: bool = False) -> None:
    if json_output:
        print(report.to_json())
    else:
        print(report.render())



def _parse_slurm_memory_mib(value: str) -> int | None:
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"(\d+)([KMGTP]?)", text)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2) or "M"
    factors = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024 * 1024 * 1024}
    return int(amount * factors[unit])


def _resource_plan_rows(ctx: SubmitContext) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for order, stage in enumerate(STAGES, start=1):
        resources = ctx.runtime.resources_for(stage)
        rows.append(
            {
                "stage_order": str(order),
                "stage": stage.name,
                "cpus": resources.cpus,
                "mem": resources.mem,
                "gpus": resources.gpus,
                "time": resources.time,
                "uses_gpu": str(resources.uses_gpu),
            }
        )
    return rows


def _format_resource_plan(ctx: SubmitContext) -> str:
    lines = ["RESOURCE PLAN", ""]
    for stage in STAGES:
        resources = ctx.runtime.resources_for(stage)
        lines.append(f"{stage.name:<24} {resources.cpus:>2} CPU   {resources.mem:<5} GPU {resources.gpus:<1}   {resources.time}")
    return "\n".join(lines)


def _validate_stage_architecture(report: PreflightReport, ctx: SubmitContext) -> None:
    expected = ("native_analysis", "decoy_prepare", "proteinmpnn", "sequence_validation", "msa_resolution", "decoy_folding", "fold_qc", "decoy_metrics_analysis")
    if STAGE_NAMES == expected:
        report.pass_("FETT_STAGE_ORDER_OK", "Expanded Fett stage order is valid.", details=" -> ".join(STAGE_NAMES))
    else:
        report.fail("FETT_STAGE_ORDER_INVALID", "Expanded Fett stage order is invalid.", details=" -> ".join(STAGE_NAMES))

    gpu_stages = [stage.name for stage in STAGES if ctx.runtime.resources_for(stage).uses_gpu]
    if gpu_stages == ["decoy_folding"]:
        report.pass_("FETT_GPU_SCOPE_OK", "Only decoy_folding requests a GPU.")
    else:
        report.fail("FETT_GPU_SCOPE_INVALID", "Only decoy_folding may request a GPU.", details=", ".join(gpu_stages) or "none")

    invalid_cpus = [stage.name for stage in STAGES if not re.fullmatch(r"\d+", ctx.runtime.resources_for(stage).cpus) or int(ctx.runtime.resources_for(stage).cpus) <= 0]
    if invalid_cpus:
        report.fail("RESOURCE_CPUS_INVALID", "All Fett stage CPU requests must be positive integers.", details=", ".join(invalid_cpus))
    else:
        report.pass_("RESOURCE_CPUS_OK", "All Fett stage CPU requests are positive.")

    invalid_times = [stage.name for stage in STAGES if not re.fullmatch(r"(\d+-)?\d{1,2}:\d{2}:\d{2}", ctx.runtime.resources_for(stage).time)]
    if invalid_times:
        report.fail("RESOURCE_TIME_INVALID", "All Fett stage wall times must use Slurm time format.", details=", ".join(invalid_times))
    else:
        report.pass_("RESOURCE_TIME_OK", "All Fett stage wall times are valid.")

    over_limit: list[str] = []
    invalid_mem: list[str] = []
    for stage in STAGES:
        mem = ctx.runtime.resources_for(stage).mem
        mib = _parse_slurm_memory_mib(mem)
        if mib is None:
            invalid_mem.append(f"{stage.name}={mem}")
        elif mib > MAX_STAGE_MEMORY_MIB:
            over_limit.append(f"{stage.name}={mem}")
    if invalid_mem:
        report.fail("RESOURCE_MEMORY_INVALID", "One or more Fett stage memory requests have invalid Slurm format.", details="\n".join(invalid_mem))
    if over_limit:
        first = over_limit[0]
        stage_name, requested = first.split("=", 1)
        report.fail(
            "RESOURCE_MEMORY_LIMIT",
            f"Stage {stage_name} requests {requested}, but the installation limit is 24G.",
            details=f"Resolved allocation:\n  stage: {stage_name}\n  requested: {requested}\n  maximum: 24G",
            remediation="No jobs submitted. Reduce the administrator-owned Fett stage resource override.",
            stage=stage_name,
        )
    if not invalid_mem and not over_limit:
        report.pass_("RESOURCE_MEMORY_OK", "No Fett stage requests more than 24G.")

    report.pass_("RESOURCE_PLAN", "Resolved Fett resource plan.", details=_format_resource_plan(ctx))


def _q(path: str | Path) -> str:
    text = str(path)
    if text.startswith("${") and text.endswith("}"):
        return text
    return shlex.quote(text)


def _line(*parts: str | Path) -> str:
    return " ".join(_q(part) for part in parts)


def _native_manifest(run: RunPaths) -> Path:
    return run.manifests_native / "manifest.csv"


def _native_features(run: RunPaths) -> Path:
    return run.results_native_tables / "native_interface_features.csv"


def _native_rosetta(run: RunPaths) -> Path:
    return run.results_native_tables / "rosetta_interface_native_relaxed.csv"


def _native_relax_manifest(run: RunPaths) -> Path:
    return run.results_native_tables / "rosetta_relax_status.csv"


def _fold_dir(run: RunPaths, experiment: str) -> Path:
    return run.decoy_layout(experiment).manifest_dir


def _all_mode_tables(run: RunPaths, experiment: str) -> Path:
    return run.decoy_layout(experiment).all_modes_tables_dir


def _artifact_contracts(output_root: Path, experiment: str, *, backend: str) -> list[ArtifactContract]:
    run = RunPaths.from_output_root(output_root)
    layout = run.decoy_layout(experiment)
    mode = "{mode}"
    mode_tables = layout.mode_tables_dir(mode)
    mode_work = layout.mode_work_dir(mode)
    mode_landscape = mode_tables / "landscape"
    rows = [
        ("native manifest", "native_analysis:nanobody-pipeline", "native_analysis:decoy-prefold --native-manifest", _native_manifest(run)),
        ("native feature table", "native_analysis:nanobody-pipeline", "native_analysis:decoy-prefold --native-features; decoy_metrics_analysis", _native_features(run)),
        ("native relaxed Rosetta table", "native_analysis:nanobody-pipeline", "native_analysis:decoy-prefold --native-relaxed-rosetta; decoy_metrics_analysis", _native_rosetta(run)),
        ("native relax status", "native_analysis:nanobody-pipeline", "decoy_metrics_analysis:rosetta-qc", _native_relax_manifest(run)),
        ("source case manifest", "native_analysis:decoy-prefold", "native_analysis:decoy-cases-base", layout.source_case_manifest),
        ("mode assignments", "native_analysis:decoy-prefold", "decoy_metrics_analysis:landscape", layout.mode_assignments),
        ("prefold manifest", "native_analysis:decoy-prefold", "decoy_prepare:decoy-fold --prefold-dir", layout.prefold_manifest),
        ("per-mode redesign manifest", "native_analysis:decoy-prefold", "decoy_prepare/decoy_metrics_analysis", layout.mode_case_manifest(mode)),
        ("ProteinMPNN output manifest", "decoy_prepare:decoy-prepare", "proteinmpnn:run_proteinmpnn_jobs", mode_tables / "decoy_mpnn_job_manifest.csv"),
        ("ProteinMPNN jobs TSV", "decoy_prepare:decoy-prepare", "proteinmpnn:run_proteinmpnn_jobs", mode_work / "job_bundle" / "proteinmpnn_jobs.tsv"),
        ("ProteinMPNN runner", "decoy_prepare:decoy-prepare", "proteinmpnn:run_proteinmpnn_jobs", mode_work / "job_bundle" / "run_proteinmpnn_jobs.sh"),
        ("ProteinMPNN execution status", "proteinmpnn", "sequence_validation", mode_tables / "proteinmpnn_execution_status.csv"),
        ("selected design manifest", "sequence_validation:decoy-validate", "msa_resolution:decoy-validate", mode_tables / "decoy_selected_designs.csv"),
        ("sequence validation manifest", "msa_resolution:decoy-validate", "fold_qc; decoy_metrics_analysis:graft", mode_tables / "decoy_validation.csv"),
        ("MSA eligibility manifest", "msa_resolution:decoy-validate", "decoy_folding:backend fold bundle", mode_tables / "decoy_msa_resolution_status.csv"),
        ("fold preparation manifest", "msa_resolution:decoy-validate", "decoy_folding:fold-jobs", mode_work / "job_bundle" / f"{backend}_monomer_jobs.tsv"),
        ("fold output manifest", "msa_resolution:decoy-validate", "decoy_folding/fold_qc/decoy_metrics_analysis", layout.fold_manifest),
        ("fold Slurm submission manifest", "decoy_folding:fold-jobs", "fold_qc", layout.combined_fold_submission_manifest),
        ("fold QC table", "fold_qc", "decoy_metrics_analysis/export", layout.all_modes_tables_dir / "post_fold_qc.csv"),
        ("grafted complex manifest", "decoy_metrics_analysis:decoy-graft", "decoy_metrics_analysis:decoy-relax/export", mode_tables / "decoy_grafted_complex_manifest.csv"),
        ("decoy relaxed Rosetta table", "decoy_metrics_analysis:decoy-relax", "decoy_metrics_analysis:decoy-metrics/DockQ", mode_tables / "decoy_relax_manifest.csv"),
        ("combined decoy relaxed Rosetta table", "decoy_metrics_analysis:all-mode aggregation", "decoy_metrics_analysis:rosetta-qc", layout.all_modes_tables_dir / "decoy_relax_manifest.csv"),
        ("decoy InterfaceAnalyzer table", "decoy_metrics_analysis:decoy-metrics", "decoy_metrics_analysis:landscape/export", mode_tables / "decoy_rosetta_interface.csv"),
        ("combined decoy InterfaceAnalyzer table", "decoy_metrics_analysis:all-mode aggregation", "decoy_metrics_analysis:rosetta-qc", layout.all_modes_tables_dir / "decoy_rosetta_interface.csv"),
        ("DockQ table", "decoy_metrics_analysis:decoy-dockq", "decoy_metrics_analysis:landscape/export", mode_tables / "decoy_dockq.csv"),
        ("landscape table", "decoy_metrics_analysis:landscape", "fett export-final", mode_landscape / "delphi_landscape_scores.csv"),
        ("quadrant table", "decoy_metrics_analysis:landscape", "fett export-final", mode_landscape / "decoy_quadrant_ranking.csv"),
        ("mutation localization table", "decoy_metrics_analysis:landscape", "fett export-final", mode_landscape / "mutation_locations.csv"),
        ("run manifest", "fett submit/resume", "fett status/logs/cancel/export", output_root / RUN_MANIFEST),
        ("stage manifest", "fett submit/resume/status", "fett status/logs/cancel", output_root / STAGE_MANIFEST),
        ("Slurm logs", "all Fett stage scripts", "fett logs", output_root / "logs" / "slurm" / "%x-%j.out"),
    ]
    return [
        ArtifactContract(name, producer, consumer, str(path), str(path), str(path))
        for name, producer, consumer, path in rows
    ]


def _contract_table(contracts: Sequence[ArtifactContract]) -> str:
    lines = ["artifact\tproducer\tconsumer\tcanonical_resolved_path\tconsistent"]
    for contract in contracts:
        lines.append(
            "\t".join(
                [
                    contract.artifact,
                    contract.producer,
                    contract.consumer,
                    contract.canonical_path,
                    str(contract.consistent),
                ]
            )
        )
    return "\n".join(lines)


def validate_artifact_contracts(report: PreflightReport, contracts: Sequence[ArtifactContract]) -> None:
    mismatches = [contract for contract in contracts if not contract.consistent]
    if mismatches:
        first = mismatches[0]
        report.fail(
            "PIPELINE_CONTRACT_MISMATCH",
            f"{len(mismatches)} producer/consumer artifact path mismatch(es) detected.",
            details=(
                f"Artifact: {first.artifact}\n"
                f"Producer declares: {first.producer_path}\n"
                f"Consumer requires: {first.consumer_path}\n"
                f"Canonical path: {first.canonical_path}\n\n"
                + _contract_table(mismatches)
            ),
            remediation="Regenerate stage scripts from one resolved RunPaths/DecoyLayout; do not submit Slurm jobs.",
        )
    else:
        report.pass_("PIPELINE_CONTRACT_OK", f"{len(contracts)} producer/consumer artifact paths are consistent.", details=_contract_table(contracts))


def validate_pipeline_contract(report: PreflightReport, ctx: SubmitContext) -> None:
    validate_artifact_contracts(report, _artifact_contracts(ctx.output_root, ctx.experiment, backend=ctx.runtime.backend or ctx.args.backend))


def _legacy_prefold_experiment(output_root: Path, stored_experiment: str) -> str | None:
    try:
        run = RunPaths.from_output_root(output_root)
    except ValueError:
        return None
    expected = run.decoy_layout(stored_experiment).prefold_manifest if stored_experiment else None
    if expected is not None and expected.exists():
        return None
    decoy_root = run.manifests_decoy
    if not decoy_root.exists():
        return None
    candidates = sorted(decoy_root.glob("*/prefold_manifest.json"))
    if len(candidates) != 1:
        return None
    try:
        data = json.loads(candidates[0].read_text())
    except (OSError, json.JSONDecodeError):
        return None
    experiment = str(data.get("experiment") or candidates[0].parent.name)
    return experiment if experiment and experiment != stored_experiment else None


def _mode_names_from_prefold(output_root: Path, experiment: str) -> list[str]:
    prefold = RunPaths.from_output_root(output_root).decoy_layout(experiment).prefold_manifest
    if not prefold.exists():
        return []
    try:
        data = json.loads(prefold.read_text())
    except json.JSONDecodeError:
        return []
    modes = data.get("mode_manifests")
    return sorted(str(mode) for mode in modes) if isinstance(modes, dict) else []


def _stage_expected_outputs(stage: StageSpec, output_root: Path, experiment: str) -> list[Path]:
    run = RunPaths.from_output_root(output_root)
    layout = run.decoy_layout(experiment)
    modes = _mode_names_from_prefold(output_root, experiment)
    if stage.name == "native_analysis":
        return [_native_manifest(run), _native_features(run), _native_rosetta(run), layout.prefold_manifest]
    if stage.name == "decoy_prepare":
        return [path for mode in modes for path in [layout.mode_tables_dir(mode) / "decoy_prepare_status.csv", layout.mode_work_dir(mode) / "job_bundle" / "proteinmpnn_jobs.tsv", layout.mode_work_dir(mode) / "job_bundle" / "run_proteinmpnn_jobs.sh"]]
    if stage.name == "proteinmpnn":
        return [layout.mode_tables_dir(mode) / "proteinmpnn_execution_status.csv" for mode in modes]
    if stage.name == "sequence_validation":
        return [layout.mode_tables_dir(mode) / "decoy_selected_designs.csv" for mode in modes]
    if stage.name == "msa_resolution":
        return [layout.fold_manifest, *[layout.mode_tables_dir(mode) / "decoy_msa_resolution_status.csv" for mode in modes]]
    if stage.name == "decoy_folding":
        return [layout.combined_fold_submission_manifest]
    if stage.name == "fold_qc":
        return [layout.all_modes_tables_dir / "post_fold_qc.csv"]
    if stage.name == "decoy_metrics_analysis":
        return [layout.all_modes_tables_dir / "decoy_rosetta_post_qc.csv"]
    return []


def _stage_is_complete(stage: StageSpec, output_root: Path, experiment: str) -> bool:
    paths = _stage_expected_outputs(stage, output_root, experiment)
    return bool(paths) and all(path.exists() and path.stat().st_size > 0 for path in paths)


def _script_header(ctx: SubmitContext, stage: StageSpec) -> list[str]:
    job_name = _safe_slug(f"fett-{ctx.run_id}-{stage.slug}")
    resources = ctx.runtime.resources_for(stage)
    lines = ["#!/usr/bin/env bash", f"#SBATCH --partition={ctx.runtime.partition}"]
    if resources.uses_gpu:
        lines.append(f"#SBATCH --gres=gpu:{resources.gpus}")
    lines.extend(
        [
            f"#SBATCH --cpus-per-task={resources.cpus}",
            f"#SBATCH --mem={resources.mem}",
            f"#SBATCH --time={resources.time}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={ctx.output_root / 'logs' / 'slurm' / '%x-%j.out'}",
            f"#SBATCH --error={ctx.output_root / 'logs' / 'slurm' / '%x-%j.err'}",
            "",
            "set -euo pipefail",
            "source <CONDA_ROOT>/etc/profile.d/conda.sh",
            "conda activate jango",
            f"cd {_q(ctx.repo_root)}",
            *[
                f"export {key}={_q(ctx.runtime.values[key])}"
                for key in PRE_SOURCE_EXPORT_KEYS
                if ctx.runtime.values.get(key)
            ],
            "set -a",
            f"source {_q(ctx.paths_config)}",
            f"source {_q(ctx.runtime_config)}",
            "set +a",
        ]
    )
    if stage.name == "proteinmpnn":
        lines.extend(
            [
                'export CUDA_VISIBLE_DEVICES=""',
                'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                'export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                'export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
                'echo "ProteinMPNN CPU mode: CUDA visibility disabled"',
            ]
        )
    lines.append("")
    return lines


def _stage_body(ctx: SubmitContext, stage: StageSpec) -> list[str]:
    args = ctx.args
    run = RunPaths.from_output_root(ctx.output_root)
    source_label = args.source or ctx.runtime.values.get("JANGO_SOURCE_LABEL", "sabdab_1_2")
    raw_dir = args.input_dir or Path(ctx.runtime.values.get("JANGO_RAW_TARBALLS", ""))
    rosetta_runtime = ctx.runtime.values.get("ROSETTA_RUNTIME")
    rosetta_image = ctx.runtime.values.get("ROSETTA_IMAGE")
    apptainer_cache = ctx.runtime.values.get("APPTAINER_CACHEDIR")

    native = [
        "nanobody-pipeline",
        "--input",
        raw_dir,
        "--input-kind",
        args.input_kind,
        "--output-root",
        ctx.output_root,
        "--atlas-root",
        ctx.repo_root,
        "--source",
        source_label,
        "--label",
        args.label,
        "--role",
        args.role,
        "--confidence",
        args.confidence,
        "--jobs",
        "${SLURM_CPUS_PER_TASK:-1}",
        "--limit",
        str(args.max_structures),
    ]
    if args.metadata is not None:
        native.extend(["--metadata", args.metadata])
    if rosetta_runtime:
        native.extend(["--rosetta-runtime", rosetta_runtime])
    if rosetta_image:
        native.extend(["--rosetta-image", rosetta_image])
    if apptainer_cache:
        native.extend(["--apptainer-cache-dir", apptainer_cache])

    if stage.name == "native_analysis":
        prefold = [
            "decoy-prefold",
            "--raw-dir",
            run.raw_source(source_label).tarballs,
            "--native-manifest",
            _native_manifest(run),
            "--native-features",
            _native_features(run),
            "--native-relaxed-rosetta",
            _native_rosetta(run),
            "--decoy-config",
            args.decoy_config,
            "--backend",
            ctx.runtime.backend,
            "--max-structures",
            str(args.max_structures),
            "--mode-policy",
            args.mode_policy,
            "--output-root",
            ctx.output_root,
            "--atlas-root",
            ctx.repo_root,
        ]
        prefold.extend(["--experiment", ctx.experiment])
        if args.modes:
            prefold.append("--modes")
            prefold.extend(args.modes)
        return [_line(*native), _line(*prefold)]

    decoy_fold_common = [
        "decoy-fold",
        "--prefold-dir",
        _fold_dir(run, ctx.experiment),
        "--atlas-root",
        ctx.repo_root,
        "--backend",
        ctx.runtime.backend,
        "--output-root",
        ctx.output_root,
        "--execution-mode",
        "slurm",
        "--paths-config",
        ctx.paths_config,
        "--runtime-config",
        ctx.runtime_config,
        "--max-decoy-sequences",
        str(args.max_decoy_sequences),
    ]

    if stage.name == "decoy_prepare":
        return [_line(*decoy_fold_common, "--fett-stage", "prepare", "--skip-mpnn", "--skip-folding")]

    if stage.name == "proteinmpnn":
        return [_line(*decoy_fold_common, "--fett-stage", "proteinmpnn", "--skip-folding")]

    if stage.name == "sequence_validation":
        return [_line(*decoy_fold_common, "--fett-stage", "sequence_validation", "--skip-mpnn", "--skip-folding")]

    if stage.name == "msa_resolution":
        return [_line(*decoy_fold_common, "--fett-stage", "msa_resolution", "--skip-mpnn", "--skip-folding")]

    if stage.name == "decoy_folding":
        manifest = run.decoy_layout(ctx.experiment).combined_fold_submission_manifest
        return [
            _line(
                "jango",
                "fold-jobs",
                "prepare-bundle",
                "--fold-dir",
                _fold_dir(run, ctx.experiment),
                "--output-root",
                ctx.output_root,
                "--paths-config",
                ctx.paths_config,
                "--runtime-config",
                ctx.runtime_config,
            ),
            _line("jango", "fold-jobs", "run-bundle", "--manifest", manifest),
        ]

    if stage.name == "fold_qc":
        return [_line("jango", "fold-qc", "--fold-dir", _fold_dir(run, ctx.experiment), "--output-root", ctx.output_root)]

    if stage.name == "decoy_metrics_analysis":
        all_tables = _all_mode_tables(run, ctx.experiment)
        fold_relax = [
            "decoy-fold-relax",
            "--fold-dir",
            _fold_dir(run, ctx.experiment),
            "--atlas-root",
            ctx.repo_root,
            "--native-relax-manifest",
            _native_relax_manifest(run),
            "--rosetta-runtime",
            ctx.runtime.values.get("ROSETTA_RUNTIME", "apptainer"),
            "--output-root",
            ctx.output_root,
            "--jobs",
            "${SLURM_CPUS_PER_TASK:-1}",
            "--allow-dockq-failures",
            "--dockq-skip-existing",
        ]
        results = [
            "decoy-results",
            "--fold-dir",
            _fold_dir(run, ctx.experiment),
            "--atlas-root",
            ctx.repo_root,
            "--native-relaxed-rosetta",
            _native_rosetta(run),
            "--native-features",
            _native_features(run),
            "--rosetta-runtime",
            ctx.runtime.values.get("ROSETTA_RUNTIME", "apptainer"),
            "--output-root",
            ctx.output_root,
            "--jobs",
            "${SLURM_CPUS_PER_TASK:-1}",
        ]
        if rosetta_image:
            fold_relax.extend(["--rosetta-image", rosetta_image])
            results.extend(["--rosetta-image", rosetta_image])
        if apptainer_cache:
            fold_relax.extend(["--apptainer-cache-dir", apptainer_cache])
            results.extend(["--apptainer-cache-dir", apptainer_cache])
        dockq_bin = ctx.runtime.values.get("DOCKQ_BIN")
        if dockq_bin:
            fold_relax.extend(["--dockq-bin", dockq_bin])
        rosetta_qc = [
            "jango",
            "rosetta-qc",
            "--native-relax-manifest",
            _native_relax_manifest(run),
            "--native-interface",
            _native_rosetta(run),
            "--decoy-relax-manifest",
            all_tables / "decoy_relax_manifest.csv",
            "--decoy-interface",
            all_tables / "decoy_rosetta_interface.csv",
            "--native-work-dir",
            run.work_native / "rosetta",
            "--source-root",
            ctx.repo_root,
            "--native-out",
            run.results_native_tables / "native_rosetta_post_qc.csv",
            "--decoy-out",
            all_tables / "decoy_rosetta_post_qc.csv",
        ]
        export = ["fett", "export-final", "--output-root", ctx.output_root, "--experiment", ctx.experiment]
        return [_line(*fold_relax), _line(*results), _line(*rosetta_qc), _line(*export)]
    raise ValueError(stage.name)



def _validate_generated_stage_scripts(report: PreflightReport, scripts: Sequence[Path]) -> None:
    scripts_by_stem = {script.stem: script for script in scripts}
    missing = [stage.slug for stage in STAGES if stage.slug not in scripts_by_stem]
    if missing:
        report.fail("FETT_STAGE_SCRIPT_MISSING", "One or more expanded Fett stage scripts were not generated.", details=", ".join(missing))
        return
    report.pass_("FETT_STAGE_SCRIPTS_OK", f"Generated {len(scripts)} expanded Fett stage script(s).")

    for stage in STAGES:
        script = scripts_by_stem[stage.slug]
        text = script.read_text(errors="ignore")
        has_gpu_directive = "#SBATCH --gres=" in text
        if stage.name == "decoy_folding":
            if "#SBATCH --gres=gpu:1" in text:
                report.pass_("FETT_FOLD_GPU_OK", "decoy_folding requests exactly one GPU.", stage=stage.name)
            else:
                report.fail("FETT_FOLD_GPU_INVALID", "decoy_folding must request #SBATCH --gres=gpu:1.", stage=stage.name)
        elif has_gpu_directive:
            report.fail("FETT_CPU_STAGE_GPU_REQUEST", f"CPU-only stage {stage.name} requests a GPU.", stage=stage.name)

        if stage.name == "proteinmpnn":
            if 'export CUDA_VISIBLE_DEVICES=""' in text:
                report.pass_("PROTEINMPNN_CPU_FORCED", "ProteinMPNN stage clears CUDA_VISIBLE_DEVICES.", stage=stage.name)
            else:
                report.fail("PROTEINMPNN_GPU_NOT_DISABLED", "ProteinMPNN stage must clear CUDA_VISIBLE_DEVICES.", stage=stage.name)
        if stage.name == "decoy_folding" and "fold-jobs submit" in text:
            report.fail("FETT_FOLD_NESTED_SUBMIT", "decoy_folding must not submit a nested Slurm fold bundle.", stage=stage.name)
        if stage.name == "decoy_folding" and "fold-jobs prepare-bundle" not in text:
            report.fail("FETT_FOLD_PREPARE_MISSING", "decoy_folding must prepare the fold bundle inside the Fett Slurm stage.", stage=stage.name)
        if stage.name == "decoy_folding" and "fold-qc" in text:
            report.fail("FOLD_QC_IN_GPU_STAGE", "fold QC must not run inside the GPU folding stage.", stage=stage.name)
        if stage.name != "decoy_folding" and "fold-jobs run-bundle" in text:
            report.fail("FOLDING_IN_CPU_STAGE", f"CPU-only stage {stage.name} contains fold execution.", stage=stage.name)


def write_stage_script(ctx: SubmitContext, stage: StageSpec) -> Path:
    script_dir = ctx.output_root / "work" / "fett" / "slurm"
    script_dir.mkdir(parents=True, exist_ok=True)
    path = script_dir / f"{stage.slug}.sbatch"
    path.write_text("\n".join(_script_header(ctx, stage) + _stage_body(ctx, stage)) + "\n")
    path.chmod(0o750)
    return path


def parse_sbatch_job_id(text: str) -> str:
    match = re.search(r"Submitted batch job\s+(\S+)", text)
    if not match:
        raise ValueError(f"could not parse sbatch job id from: {text!r}")
    return match.group(1)


def _run_sbatch(script: Path, dependency: str | None, runner: Runner = subprocess.run) -> tuple[str, str]:
    argv = ["sbatch"]
    if dependency:
        argv.append(f"--dependency=afterok:{dependency}")
    argv.append(str(script))
    result = runner(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {script}: {result.stderr.strip()}")
    return parse_sbatch_job_id(result.stdout), " ".join(argv)


def _stage_rows(manifest: dict[str, object]) -> list[dict[str, object]]:
    stages = manifest.get("stages")
    return [dict(row) for row in stages] if isinstance(stages, list) else []



def _legacy_stage_names(manifest: dict[str, object]) -> list[str]:
    names = [str(row.get("name", "")) for row in _stage_rows(manifest)]
    return [name for name in names if name and name not in STAGE_NAMES]


def _legacy_stage_mapping_details(names: Sequence[str]) -> str:
    lines: list[str] = []
    for name in names:
        mapped = LEGACY_STAGE_MAP.get(name, ())
        lines.append(f"{name} -> {', '.join(mapped) if mapped else 'no safe automatic mapping'}")
    return "\n".join(lines)


def _write_manifest(output_root: Path, manifest: dict[str, object]) -> None:
    path = output_root / RUN_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    rows = _stage_rows(manifest)
    if rows:
        csv_path = output_root / STAGE_MANIFEST
        keys = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    _write_run_snapshots_from_manifest(output_root, manifest)


def _load_manifest(output_root: Path) -> dict[str, object]:
    path = output_root / RUN_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"missing Fett run manifest: {path}")
    return json.loads(path.read_text())


def _existing_run_manifest(output_root: Path) -> dict[str, object] | None:
    path = output_root / RUN_MANIFEST
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"run_id": "", "manifest_error": "invalid_json"}


def _stage_status_summary(manifest: dict[str, object] | None) -> str:
    if not manifest:
        return "new"
    rows = _stage_rows(manifest)
    if any(str(row.get("status")) in ACTIVE_STATES for row in rows):
        return "active"
    if rows and all(str(row.get("status")) in SUCCESS_STATES for row in rows):
        return "completed"
    if any(str(row.get("status")) in FAILURE_STATES or str(row.get("status")) == "INCOMPLETE" for row in rows):
        return "resumable"
    return str(manifest.get("current_status") or manifest.get("status") or "existing")


def _workspace_from_args(args: argparse.Namespace) -> UserWorkspace:
    override = getattr(args, "user_workspace", None)
    if override:
        root = Path(override)
        return UserWorkspace(root=root, work=root / "work", outputs=root / "outputs", data=root / "data", cache=root / "cache", registry=root / "registry", tmp=root / "tmp")
    return resolve_user_workspace()


def _prepare_workspace(ctx: SubmitContext, report: PreflightReport | None = None) -> None:
    workspace = _workspace_from_args(ctx.args)
    layout = RunLayout(ctx.output_root)
    try:
        workspace.create()
        layout.create()
    except OSError as exc:
        if report is not None:
            report.fail("WORKSPACE_CREATE_FAILED", f"Could not create run workspace under {ctx.output_root}.", details=str(exc))
        else:
            raise
    else:
        if report is not None:
            report.pass_("WORKSPACE_READY", f"Workspace ready: {workspace.root}")
            report.pass_("RUN_LAYOUT_READY", f"Run layout ready: {layout.root}")


def _copy_decoy_snapshot(ctx: SubmitContext) -> str:
    source = Path(ctx.args.decoy_config)
    if not source.exists() or not source.is_file():
        return ""
    target = RunLayout(ctx.output_root).inputs / "decoy_config.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return str(target)


def _git_dirty(repo_root: Path) -> bool:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return False
    return bool(result.stdout.strip()) if result.returncode == 0 else False


def _write_run_snapshots(ctx: SubmitContext, *, manifest: dict[str, object] | None = None, status: str = "prepared") -> None:
    layout = RunLayout(ctx.output_root)
    layout.create()
    decoy_snapshot = _copy_decoy_snapshot(ctx)
    rows = _stage_rows(manifest or {})
    run_data = {
        "run_id": ctx.run_id,
        "user": _current_user(),
        "created_at": (manifest or {}).get("created_at", _now()),
        "updated_at": _now(),
        "current_status": status,
        "input_path": str(ctx.args.input_dir),
        "input_kind": ctx.args.input_kind,
        "case_count": ctx.args.case_count,
        "max_structures": ctx.args.max_structures,
        "count_semantics": "input_max",
        "backend": ctx.runtime.backend,
        "output_root": str(ctx.output_root),
        "experiment": ctx.experiment,
        "experiment_key": ctx.experiment,
        "label": ctx.args.label,
        "source": ctx.args.source,
        "role": ctx.args.role,
        "confidence": ctx.args.confidence,
        "stage_job_ids": {str(row.get("name")): str(row.get("job_id", "")) for row in rows},
        "stage_statuses": {str(row.get("name")): str(row.get("status", "")) for row in rows},
        "resource_plan": _resource_plan_rows(ctx),
        "proteinmpnn_forced_cpu": True,
    }
    layout.run_json.write_text(json.dumps(run_data, indent=2, sort_keys=True) + "\n")
    resolved = _effective_config_for_manifest(ctx.runtime.values)
    resolved.update(
        {
            "paths_config": str(ctx.paths_config),
            "runtime_config": str(ctx.runtime_config),
            "decoy_config": str(ctx.args.decoy_config),
            "decoy_config_snapshot": decoy_snapshot,
            "input_kind": str(ctx.args.input_kind),
            "case_count": str(ctx.args.case_count),
            "max_structures": str(ctx.args.max_structures),
            "count_semantics": "input_max",
            "mode_policy": str(ctx.args.mode_policy),
            "experiment_key": str(ctx.experiment),
            "max_decoy_sequences": str(ctx.args.max_decoy_sequences),
            "resource_plan": _resource_plan_rows(ctx),
            "proteinmpnn_forced_cpu": "true",
        }
    )
    layout.resolved_config.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    provenance = {
        "jango_git_revision": _git_commit(ctx.repo_root),
        "nbia_revision": _git_commit(ctx.repo_root),
        "git_dirty": _git_dirty(ctx.repo_root),
        "paths_config_sha256": _sha256(ctx.paths_config),
        "runtime_config_sha256": _sha256(ctx.runtime_config),
        "decoy_config_sha256": _sha256(Path(ctx.args.decoy_config)),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "submission_command": " ".join(sys.argv),
        "resolved_command": f"fett submit --input {ctx.args.input_dir} --run-id {ctx.run_id} --max-structures {ctx.args.max_structures} --backend {ctx.runtime.backend}",
        "resource_plan": _resource_plan_rows(ctx),
        "proteinmpnn_forced_cpu": True,
    }
    layout.provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    input_manifest = {
        "input_path": str(ctx.args.input_dir),
        "input_kind": ctx.args.input_kind,
        "raw_data_copied": False,
        "metadata": str(ctx.args.metadata or ""),
    }
    (layout.inputs / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2, sort_keys=True) + "\n")
    workspace = _workspace_from_args(ctx.args)
    try:
        workspace.registry.mkdir(parents=True, exist_ok=True)
        (workspace.registry / f"{ctx.run_id}.json").write_text(
            json.dumps({"run_id": ctx.run_id, "output_root": str(ctx.output_root), "status": status, "updated_at": _now()}, indent=2, sort_keys=True) + "\n"
        )
    except OSError:
        pass


def _write_run_snapshots_from_manifest(output_root: Path, manifest: dict[str, object], *, status: str | None = None) -> None:
    try:
        args = argparse.Namespace(output_root=output_root)
        ctx = _context_from_manifest(args, manifest)
    except Exception:
        return
    _write_run_snapshots(ctx, manifest=manifest, status=status or _stage_status_summary(manifest))


def _new_manifest(ctx: SubmitContext) -> dict[str, object]:
    return {
        "schema_version": 2,
        "stage_schema": "expanded_v2",
        "run_id": ctx.run_id,
        "version": _version(),
        "git_commit": _git_commit(ctx.repo_root),
        "repo_root": str(ctx.repo_root),
        "output_root": str(ctx.output_root),
        "paths_config": str(ctx.paths_config),
        "paths_config_sha256": _sha256(ctx.paths_config),
        "runtime_config": str(ctx.runtime_config),
        "runtime_config_sha256": _sha256(ctx.runtime_config),
        "input_kind": ctx.args.input_kind,
        "input_dir": str(ctx.args.input_dir or ""),
        "metadata": str(ctx.args.metadata or ""),
        "decoy_config": str(ctx.args.decoy_config),
        "experiment": ctx.experiment,
        "experiment_key": ctx.experiment,
        "artifact_contract": [contract.as_row() for contract in _artifact_contracts(ctx.output_root, ctx.experiment, backend=ctx.runtime.backend)],
        "resource_plan": _resource_plan_rows(ctx),
        "stage_order": list(STAGE_NAMES),
        "proteinmpnn_forced_cpu": True,
        "backend": ctx.runtime.backend,
        "case_count": ctx.args.case_count,
        "max_structures": ctx.args.max_structures,
        "count_semantics": "input_max",
        "mode_policy": ctx.args.mode_policy,
        "label": ctx.args.label,
        "role": ctx.args.role,
        "confidence": ctx.args.confidence,
        "source": ctx.args.source or "",
        "modes": ctx.args.modes or [],
        "max_decoy_sequences": ctx.args.max_decoy_sequences,
        "effective_config": _effective_config_for_manifest(ctx.runtime.values),
        "created_at": _now(),
        "updated_at": _now(),
        "stages": [],
    }


def _submit_chain(
    ctx: SubmitContext,
    stages: Sequence[StageSpec],
    manifest: dict[str, object],
    dependency: str | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, object]:
    rows = [row for row in _stage_rows(manifest) if row.get("name") not in {stage.name for stage in stages}]
    previous_job = dependency
    stage_index = {stage.name: idx for idx, stage in enumerate(STAGES, start=1)}
    for stage in stages:
        script = write_stage_script(ctx, stage)
        resources = ctx.runtime.resources_for(stage)
        completion_artifacts = [str(path) for path in _stage_expected_outputs(stage, ctx.output_root, ctx.experiment)]
        job_id, command = _run_sbatch(script, previous_job, runner=runner)
        rows.append(
            {
                "name": stage.name,
                "slug": stage.slug,
                "stage_order": stage_index.get(stage.name, ""),
                "description": stage.description,
                "job_id": job_id,
                "dependency_job_id": previous_job or "",
                "dependency": f"afterok:{previous_job}" if previous_job else "",
                "cpus": resources.cpus,
                "mem": resources.mem,
                "gpus": resources.gpus,
                "time": resources.time,
                "uses_gpu": str(resources.uses_gpu),
                "completion_artifact": ";".join(completion_artifacts),
                "status": "SUBMITTED",
                "completion_status": "SUBMITTED",
                "exit_code": "",
                "script": str(script),
                "stdout_log": str(ctx.output_root / "logs" / "slurm" / "%x-%j.out"),
                "stderr_log": str(ctx.output_root / "logs" / "slurm" / "%x-%j.err"),
                "sbatch_command": command,
                "submitted_at": _now(),
                "started_at": "",
                "ended_at": "",
                "updated_at": _now(),
            }
        )
        previous_job = job_id
    manifest["stages"] = rows
    manifest["updated_at"] = _now()
    _write_manifest(ctx.output_root, manifest)
    return manifest


def _query_job(job_id: str, runner: Runner = subprocess.run) -> tuple[str, str]:
    if not job_id:
        return "NOT_SUBMITTED", ""
    try:
        squeue = runner(["squeue", "-j", str(job_id), "-h", "-o", "%T"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return "UNKNOWN", ""
    if squeue.returncode == 0 and squeue.stdout.strip():
        return squeue.stdout.strip().splitlines()[0].strip(), ""
    sacct = runner(["sacct", "-j", str(job_id), "--format=State,ExitCode", "-n", "-P"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if sacct.returncode == 0 and sacct.stdout.strip():
        state, _, exit_code = sacct.stdout.strip().splitlines()[0].partition("|")
        return state.strip(), exit_code.strip()
    return "UNKNOWN", ""


def _stage_by_name(name: str) -> StageSpec | None:
    return next((stage for stage in STAGES if stage.name == name), None)


def refresh_status(output_root: Path, runner: Runner = subprocess.run) -> dict[str, object]:
    manifest = _load_manifest(output_root)
    experiment = str(manifest.get("experiment", ""))
    rows = []
    for row in _stage_rows(manifest):
        stage = _stage_by_name(str(row.get("name", "")))
        status, exit_code = _query_job(str(row.get("job_id", "")), runner=runner)
        complete_outputs = stage is not None and _stage_is_complete(stage, output_root, experiment)
        if stage is not None and status == "COMPLETED" and not complete_outputs:
            status = "INCOMPLETE"
        elif complete_outputs and status not in ACTIVE_STATES and status not in FAILURE_STATES:
            status = "COMPLETED"
            exit_code = exit_code or "0:0"
        row["status"] = status
        row["exit_code"] = exit_code
        row["updated_at"] = _now()
        rows.append(row)
    manifest["stages"] = rows
    manifest["updated_at"] = _now()
    _write_manifest(output_root, manifest)
    return manifest


def _add_submission_args(parser: argparse.ArgumentParser) -> None:
    normal = parser.add_argument_group("Normal use")
    normal.add_argument("--input", type=Path, default=None, help="Raw input directory containing SAbDab tarballs, PDBs, or mmCIFs.")
    normal.add_argument("--run-id", default=None, help="Safe run identifier; creates <JANGO_USER_ROOT>/work/<run-id> by default.")
    normal.add_argument("--max-structures", type=int, default=DEFAULT_MAX_STRUCTURES, help="Maximum number of raw input structures to process before native analysis.")
    normal.add_argument("--backend", default=None, help="Folding backend override: esmfold2 (default), boltz2, or opendde.")

    advanced = parser.add_argument_group("Advanced overrides")
    advanced.add_argument("--paths-config", type=Path, default=None)
    advanced.add_argument("--runtime-config", type=Path, default=None)
    advanced.add_argument("--input-kind", choices=("tarballs", "sabdab-pdb", "sabdab2-cif"), default=None)
    advanced.add_argument("--input-dir", type=Path, default=None, help="Legacy alias for --input.")
    advanced.add_argument("--case-count", type=int, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--metadata", type=Path, default=None)
    advanced.add_argument("--output-root", type=Path, default=None)
    advanced.add_argument("--decoy-config", type=Path, default=None)
    advanced.add_argument("--mode-policy", choices=("manual", "auto_by_length"), default=None)
    advanced.add_argument("--modes", nargs="+", default=None)
    advanced.add_argument("--experiment", default=None)
    advanced.add_argument("--source", default=None)
    advanced.add_argument("--label", default=None)
    advanced.add_argument("--role", default=None)
    advanced.add_argument("--confidence", default=None)
    advanced.add_argument("--max-decoy-sequences", type=int, default=DEFAULT_MAX_DECOY_SEQUENCES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fett", description="Submit, inspect, resume, and preflight full Jango Slurm pipeline runs.")
    sub = parser.add_subparsers(dest="action", required=True)

    p_preflight = sub.add_parser("preflight", help="Validate a Fett run without submitting Slurm jobs.")
    _add_submission_args(p_preflight)
    p_preflight.add_argument("--json", action="store_true", help="Emit a machine-readable JSON preflight report.")

    p_submit = sub.add_parser("submit", help="Preflight and submit the Jango Slurm dependency chain.")
    _add_submission_args(p_submit)
    p_submit.add_argument("--skip-preflight", action="store_true", help="Expert-only: submit without preflight validation. Strongly discouraged.")

    p_status = sub.add_parser("status", help="Refresh Slurm status and output-based completion state.")
    p_status.add_argument("run_id", nargs="?", help="Run ID under <JANGO_USER_ROOT>/work.")
    p_status.add_argument("--output-root", type=Path, default=None)

    p_resume = sub.add_parser("resume", help="Preflight and submit the first incomplete stage and downstream dependencies.")
    p_resume.add_argument("run_id", nargs="?", help="Run ID under <JANGO_USER_ROOT>/work.")
    p_resume.add_argument("--output-root", type=Path, default=None)
    p_resume.add_argument("--skip-preflight", action="store_true", help="Expert-only: resume without preflight validation. Strongly discouraged.")

    p_logs = sub.add_parser("logs", help="Show recent Fett/Slurm logs for a run.")
    p_logs.add_argument("run_id", nargs="?", help="Run ID under <JANGO_USER_ROOT>/work.")
    p_logs.add_argument("--output-root", type=Path, default=None)
    p_logs.add_argument("--stage", choices=[stage.slug for stage in STAGES] + [stage.name for stage in STAGES], default=None)
    p_logs.add_argument("--errors", action="store_true", help="Prefer stderr/error logs.")
    p_logs.add_argument("--follow", action="store_true", help="Follow the selected log with tail -f.")
    p_logs.add_argument("--lines", type=int, default=80)

    p_cancel = sub.add_parser("cancel", help="Cancel active Slurm jobs recorded for a run.")
    p_cancel.add_argument("run_id", nargs="?", help="Run ID under <JANGO_USER_ROOT>/work.")
    p_cancel.add_argument("--output-root", type=Path, default=None)

    p_export = sub.add_parser("export-final", help="Copy final relaxed decoy PDBs and analysis outputs into the user outputs tree.")
    p_export.add_argument("run_id", nargs="?", help="Run ID under <JANGO_USER_ROOT>/work.")
    p_export.add_argument("--output-root", type=Path, default=None)
    p_export.add_argument("--experiment", default=None)
    p_export.add_argument("--export-root", type=Path, default=None, help="Destination root; defaults to <JANGO_USER_ROOT>/outputs/<run-id>.")
    return parser


def preflight_from_args(args: argparse.Namespace, runner: Runner = subprocess.run) -> int:
    _, report = _run_preflight(args, runner=runner)
    _emit_preflight(report, json_output=getattr(args, "json", False))
    return 1 if report.failed else 0


def submit_from_args(args: argparse.Namespace, runner: Runner = subprocess.run, preflight_runner: Runner = subprocess.run) -> int:
    if getattr(args, "skip_preflight", False):
        print("WARNING: --skip-preflight was used; Fett will submit without validating the execution contract.")
        ctx = _build_context(args)
    else:
        ctx, report = _run_preflight(args, runner=preflight_runner)
        _emit_preflight(report)
        if report.failed:
            return 1
    _prepare_workspace(ctx)
    manifest = _submit_chain(ctx, STAGES, _new_manifest(ctx), runner=runner)
    _write_run_snapshots(ctx, manifest=manifest, status="submitted")
    print(f"submitted Fett run {ctx.run_id} for experiment {ctx.experiment}")
    for row in _stage_rows(manifest):
        print(f"{row['name']}: job_id={row['job_id']} dependency={row.get('dependency_job_id', '')}")
    print(f"run manifest: {ctx.output_root / RUN_MANIFEST}")
    return 0


def _management_output_root(args: argparse.Namespace) -> Path:
    return _resolve_run_root_from_reference(getattr(args, "run_id", None), getattr(args, "output_root", None))


def status_from_args(args: argparse.Namespace, runner: Runner = subprocess.run) -> int:
    output_root = ensure_output_root_outside_repo(_management_output_root(args))
    manifest = refresh_status(output_root, runner=runner)
    print(f"Run: {manifest.get('run_id', output_root.name)}")
    print(f"Root: {output_root}")
    print(f"Backend: {manifest.get('backend', '')}")
    print(f"Max structures requested: {manifest.get('max_structures', manifest.get('case_count', ''))}")
    print("")
    for row in _stage_rows(manifest):
        dependency = f"  afterok:{row.get('dependency_job_id')}" if row.get("dependency_job_id") else ""
        print(f"{str(row.get('name', '')):<24} {str(row.get('status', '')):<12} job {row.get('job_id', '')}{dependency}")
    return 0


def _manifest_experiment(args: argparse.Namespace, manifest: dict[str, object]) -> str:
    stored = str(manifest.get("experiment_key") or manifest.get("experiment") or "")
    output_root = getattr(args, "output_root", None)
    if output_root is not None:
        legacy = _legacy_prefold_experiment(Path(output_root), stored)
        if legacy:
            return legacy
    return stored


def _args_from_manifest(args: argparse.Namespace, manifest: dict[str, object]) -> argparse.Namespace:
    return argparse.Namespace(
        paths_config=Path(str(manifest["paths_config"])),
        runtime_config=Path(str(manifest["runtime_config"])),
        input_kind=manifest.get("input_kind", "tarballs"),
        input_dir=Path(str(manifest["input_dir"])) if manifest.get("input_dir") else None,
        metadata=Path(str(manifest["metadata"])) if manifest.get("metadata") else None,
        output_root=args.output_root,
        decoy_config=Path(str(manifest.get("decoy_config", "configs/decoys.yml"))),
        max_structures=int(manifest.get("max_structures", manifest["case_count"])),
        case_count=int(manifest.get("case_count", manifest.get("max_structures", 0))),
        mode_policy=str(manifest["mode_policy"]),
        modes=manifest.get("modes") or None,
        backend=str(manifest["backend"]),
        experiment=_manifest_experiment(args, manifest),
        run_id=str(manifest["run_id"]),
        source=str(manifest.get("source", "")) or None,
        label=str(manifest.get("label", "fett")),
        role=str(manifest.get("role", "candidate")),
        confidence=str(manifest.get("confidence", "user_supplied")),
        max_decoy_sequences=int(manifest.get("max_decoy_sequences", 5)),
    )


def _context_from_manifest(args: argparse.Namespace, manifest: dict[str, object]) -> SubmitContext:
    return _build_context(_args_from_manifest(args, manifest))


def resume_from_args(args: argparse.Namespace, runner: Runner = subprocess.run, preflight_runner: Runner = subprocess.run) -> int:
    output_root = ensure_output_root_outside_repo(_management_output_root(args))
    args_dict = vars(args).copy()
    args_dict["output_root"] = output_root
    args = argparse.Namespace(**args_dict)
    manifest = _load_manifest(output_root)
    manifest_args = _args_from_manifest(args, manifest)
    if not getattr(args, "skip_preflight", False):
        _, report = _run_preflight(manifest_args, manifest=manifest, runner=preflight_runner)
        _emit_preflight(report)
        if report.failed:
            return 1
    else:
        print("WARNING: --skip-preflight was used; Fett will resume without validating the execution contract.")
    manifest = refresh_status(output_root, runner=runner)
    legacy_names = _legacy_stage_names(manifest)
    if legacy_names:
        print("legacy Fett stage schema detected:")
        print(_legacy_stage_mapping_details(legacy_names))
    for row in _stage_rows(manifest):
        if str(row.get("name", "")) not in STAGE_NAMES and str(row.get("status")) in ACTIVE_STATES:
            print(f"legacy stage {row.get('name')} is already active as job {row.get('job_id')}; not submitting duplicates")
            return 0
    rows_by_name = {str(row.get("name")): row for row in _stage_rows(manifest)}
    dependency: str | None = None
    experiment = str(manifest.get("experiment_key") or manifest.get("experiment", ""))
    first_incomplete_index: int | None = None
    for index, stage in enumerate(STAGES):
        row = rows_by_name.get(stage.name)
        complete_outputs = _stage_is_complete(stage, output_root, experiment)
        if row and str(row.get("status")) in ACTIVE_STATES:
            print(f"stage {stage.name} is already active as job {row.get('job_id')}; not submitting duplicates")
            return 0
        if complete_outputs and (not row or str(row.get("status")) not in FAILURE_STATES):
            dependency = str((row or {}).get("job_id") or dependency or "") or dependency
            continue
        if row and str(row.get("status")) in SUCCESS_STATES and complete_outputs:
            dependency = str(row.get("job_id") or dependency or "") or dependency
            continue
        if index > 0:
            prev = STAGES[index - 1]
            if not _stage_is_complete(prev, output_root, experiment):
                prev_row = rows_by_name.get(prev.name, {})
                if str(prev_row.get("status")) not in SUCCESS_STATES:
                    raise SystemExit(f"cannot resume downstream of failed/incomplete stage {prev.name}")
        first_incomplete_index = index
        break
    if first_incomplete_index is None:
        print("all Fett stages are complete")
        return 0
    ctx = _build_context(manifest_args)
    pending = STAGES[first_incomplete_index:]
    manifest = _submit_chain(ctx, pending, manifest, dependency=dependency, runner=runner)
    print(f"resubmitted Fett stages from {pending[0].name}")
    for row in _stage_rows(manifest):
        print(f"{row['name']}: job_id={row.get('job_id')} status={row.get('status')}")
    return 0


def _candidate_log_files(output_root: Path, *, stage: str | None = None, errors: bool = False) -> list[Path]:
    candidates: list[Path] = []
    for root in [output_root / "logs" / "slurm", output_root / "logs" / "fett", output_root / "logs" / "tools"]:
        if root.exists():
            candidates.extend(path for path in root.glob("*") if path.is_file())
    if stage:
        candidates = [path for path in candidates if stage in path.name]
    if errors:
        preferred = [path for path in candidates if path.suffix == ".err" or "err" in path.name.lower()]
        if preferred:
            candidates = preferred
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def logs_from_args(args: argparse.Namespace, runner: Runner = subprocess.run) -> int:
    output_root = ensure_output_root_outside_repo(_management_output_root(args))
    print(f"Run root: {output_root}")
    logs = _candidate_log_files(output_root, stage=args.stage, errors=args.errors)
    if not logs:
        print("No log files found yet.")
        return 0
    print("Logs:")
    for path in logs[:20]:
        print(f"  {path}")
    selected = logs[0]
    print(f"\n==> {selected} <==")
    if args.follow:
        return runner(["tail", "-n", str(args.lines), "-f", str(selected)], check=False).returncode
    try:
        lines = selected.read_text(errors="replace").splitlines()[-max(args.lines, 1) :]
    except OSError as exc:
        print(f"could not read log: {exc}")
        return 1
    for line in lines:
        print(line)
    return 0


def cancel_from_args(args: argparse.Namespace, runner: Runner = subprocess.run) -> int:
    output_root = ensure_output_root_outside_repo(_management_output_root(args))
    manifest = _load_manifest(output_root)
    rows = []
    cancelled: list[str] = []
    inactive: list[str] = []
    for row in _stage_rows(manifest):
        row = dict(row)
        job_id = str(row.get("job_id", ""))
        status, exit_code = _query_job(job_id, runner=runner)
        row["status"] = status
        row["exit_code"] = exit_code
        if job_id and status in ACTIVE_STATES:
            result = runner(["scancel", job_id], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode == 0:
                row["status"] = "CANCEL_REQUESTED"
                cancelled.append(job_id)
            else:
                row["status"] = "CANCEL_FAILED"
                row["cancel_error"] = (result.stderr or result.stdout).strip()
        else:
            inactive.append(job_id or str(row.get("name", "unknown")))
        row["updated_at"] = _now()
        rows.append(row)
    manifest["stages"] = rows
    manifest["updated_at"] = _now()
    manifest["current_status"] = "cancel_requested" if cancelled else _stage_status_summary(manifest)
    _write_manifest(output_root, manifest)
    print(f"Run: {manifest.get('run_id', output_root.name)}")
    print(f"Cancelled jobs: {', '.join(cancelled) if cancelled else 'none'}")
    print(f"Already inactive: {', '.join(inactive) if inactive else 'none'}")
    return 0


RELAX_SUCCESS_STATUSES = {"completed", "success", "ok", "skipped_existing_valid"}
PDB_PATH_COLUMNS = ("relaxed_pdb", "output_pdb", "decoy_relaxed_pdb", "relax_pdb", "pdb_path")
RELAX_STATUS_COLUMNS = ("relax_status", "status", "completion_status")


def _first_present(row: Mapping[str, object], names: Sequence[str]) -> str:
    return next((name for name in names if str(row.get(name, "")).strip()), "")


def _copy_files(source_dir: Path, target_dir: Path, suffixes: set[str], *, relative: bool = False, skip: set[Path] | None = None) -> int:
    if not source_dir.exists():
        return 0
    skipped = {path.resolve() for path in (skip or set())}
    count = 0
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes):
        if source.resolve() in skipped:
            continue
        target = target_dir / (source.relative_to(source_dir) if relative else source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def export_final_from_args(args: argparse.Namespace) -> int:
    output_root = ensure_output_root_outside_repo(_management_output_root(args))
    manifest = _load_manifest(output_root)
    experiment = args.experiment or str(manifest.get("experiment", ""))
    if not experiment:
        raise SystemExit("could not determine experiment; pass --experiment")
    run_id = str(manifest.get("run_id") or output_root.name)
    export_root = Path(args.export_root).expanduser() if args.export_root else resolve_user_workspace().outputs / run_id
    export_root = ensure_output_root_outside_repo(export_root)
    pdb_dir = export_root / "pdbs" / "relaxed_decoys"
    table_dir = export_root / "results" / "tables"
    figure_dir = export_root / "results" / "figures"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = RunPaths.from_output_root(output_root)
    layout = paths.decoy_layout(experiment)
    relax_manifest = layout.all_modes_tables_dir / "decoy_relax_manifest.csv"
    copied_pdbs = 0
    wrote_relax_manifest = False
    if relax_manifest.exists() and relax_manifest.stat().st_size > 0:
        with relax_manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            fieldnames = list(reader.fieldnames or [])
        for row in rows:
            path_col = _first_present(row, PDB_PATH_COLUMNS)
            if not path_col:
                continue
            status_col = _first_present(row, RELAX_STATUS_COLUMNS)
            status = str(row.get(status_col, "")).strip().lower() if status_col else "completed"
            source = Path(str(row.get(path_col, ""))).expanduser()
            if status not in RELAX_SUCCESS_STATUSES or not source.is_file() or source.stat().st_size == 0:
                continue
            target = pdb_dir / source.name
            if target.exists() and target.resolve() != source.resolve():
                prefix = str(row.get("design_id") or row.get("structure_id") or source.stem)
                target = pdb_dir / f"{_safe_slug(prefix)}_{source.name}"
            shutil.copy2(source, target)
            row[path_col] = str(target)
            copied_pdbs += 1
        if fieldnames:
            with (table_dir / relax_manifest.name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            wrote_relax_manifest = True
    copied_tables = _copy_files(layout.all_modes_tables_dir, table_dir, {".csv", ".json"}, skip={relax_manifest})
    copied_tables += _copy_files(layout.analysis_tables_dir, table_dir, {".csv", ".json"})
    copied_figures = _copy_files(paths.results_decoy_figures / _safe_slug(experiment), figure_dir, {".png", ".jpg", ".jpeg", ".pdf", ".svg"}, relative=True)
    print(f"export root: {export_root}")
    print(f"relaxed decoy PDBs: {copied_pdbs}")
    print(f"tables copied: {copied_tables + int(wrote_relax_manifest)}")
    print(f"figures copied: {copied_figures}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "preflight":
        return preflight_from_args(args)
    if args.action == "submit":
        return submit_from_args(args)
    if args.action == "status":
        return status_from_args(args)
    if args.action == "resume":
        return resume_from_args(args)
    if args.action == "logs":
        return logs_from_args(args)
    if args.action == "cancel":
        return cancel_from_args(args)
    if args.action == "export-final":
        return export_final_from_args(args)
    parser.error(f"unsupported action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
