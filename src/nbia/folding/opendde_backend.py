from __future__ import annotations

import csv
import json
import os
import re
import shlex
import stat
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd

from .base import FoldJob, FoldPrediction
from .msa import BackendMSASupport, MSAArtifact, MSACapability, MSAFormat, MSAUnsupportedError


DEFAULT_OPENDDE_MODEL = "opendde_v1"
DEFAULT_OPENDDE_SEEDS = "101"
DEFAULT_OPENDDE_CYCLE = "10"
DEFAULT_OPENDDE_STEP = "200"
DEFAULT_OPENDDE_SAMPLES = "5"
DEFAULT_OPENDDE_DTYPE = "fp32"
DEFAULT_OPENDDE_RUNNER = "${JANGO_SOURCE}/scripts/nbia/run_opendde_monomer.py"
OPEN_DDE_CHAIN_ANTIGEN = "A"


def _env_value(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        value = default
    if required and (value is None or value == ""):
        raise RuntimeError(
            f"Required environment variable is not set: {name}. "
            "Set it in the deployment environment or pass the corresponding backend argument."
        )
    return value


def _env_path(name: str, default: str | Path | None = None, required: bool = False) -> Path | None:
    value = _env_value(name, str(default) if default is not None else None, required=required)
    if value is None:
        return None
    return Path(value).expanduser()


def _module_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runner_template_env(script_name: str) -> dict[str, str]:
    env = dict(os.environ)
    source = env.get("JANGO_SOURCE") or env.get("NBIA_ROOT")
    if not source:
        candidate = _module_repo_root()
        if (candidate / "scripts" / "nbia" / script_name).is_file():
            source = str(candidate)
    if source:
        env.setdefault("JANGO_SOURCE", source)
        env.setdefault("NBIA_ROOT", source)
    return env


def _resolve_runner_path(runner_path: str | Path | None = None) -> Path:
    raw = str(runner_path or _env_value("OPENDDE_RUNNER", DEFAULT_OPENDDE_RUNNER))
    env = _runner_template_env("run_opendde_monomer.py")
    expanded = Template(raw).safe_substitute(env)
    if "$" in expanded:
        raise RuntimeError(
            f"OpenDDE runner path could not be resolved from {raw!r}; "
            "set OPENDDE_RUNNER or JANGO_SOURCE/NBIA_ROOT"
        )
    path = Path(expanded).expanduser()
    if not path.is_absolute() and env.get("JANGO_SOURCE"):
        path = Path(env["JANGO_SOURCE"]).expanduser() / path
    if not path.is_file():
        raise FileNotFoundError(f"OpenDDE runner script not found: {path}")
    return path.resolve()


def _json_safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return safe or "opendde_job"


def _parse_int_list(value: str | None) -> list[int]:
    seeds = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds or [101]


class OpenDDEBackend:
    """OpenDDE monomer folding backend for redesigned antigen sequences."""

    name = "opendde"
    msa_support = BackendMSASupport(
        capability=MSACapability.OPTIONAL,
        accepts_precomputed=True,
        generates_msa=False,
        accepted_input_formats=(MSAFormat.A3M,),
        monomer=True,
        multichain=False,
        single_sequence_fallback_allowed=True,
        notes="OpenDDE consumes validated unpaired A3M through proteinChain.unpairedMsaPath in its JSON input.",
    )

    def __init__(
        self,
        *,
        opendde_python: str | Path | None = None,
        opendde_executable: str | Path | None = None,
        root_dir: str | Path | None = None,
        model_name: str | None = None,
        checkpoint: str | Path | None = None,
        seeds: str | None = None,
        cycle: int | str | None = None,
        step: int | str | None = None,
        samples: int | str | None = None,
        dtype: str | None = None,
        runner_path: str | Path | None = None,
    ) -> None:
        self.opendde_python = Path(opendde_python).expanduser() if opendde_python else _env_path(
            "OPENDDE_PYTHON",
            required=True,
        )
        self.opendde_executable = Path(opendde_executable).expanduser() if opendde_executable else _env_path(
            "OPENDDE_EXECUTABLE",
            _env_value("OPENDDE_BIN"),
            required=True,
        )
        self.root_dir = Path(root_dir).expanduser() if root_dir else _env_path(
            "OPENDDE_ROOT_DIR",
            _env_value("OPENDDE_MODEL_ROOT"),
            required=True,
        )
        checkpoint_value = checkpoint or _env_value("OPENDDE_CHECKPOINT")
        self.checkpoint = Path(checkpoint_value).expanduser() if checkpoint_value else None
        self.model_name = model_name or _env_value("OPENDDE_MODEL_NAME", DEFAULT_OPENDDE_MODEL)
        self.seeds = seeds or _env_value("OPENDDE_SEEDS", DEFAULT_OPENDDE_SEEDS)
        self.cycle = str(cycle or _env_value("OPENDDE_CYCLE", DEFAULT_OPENDDE_CYCLE))
        self.step = str(step or _env_value("OPENDDE_STEP", DEFAULT_OPENDDE_STEP))
        self.samples = str(samples or _env_value("OPENDDE_SAMPLES", DEFAULT_OPENDDE_SAMPLES))
        self.dtype = dtype or _env_value("OPENDDE_DTYPE", DEFAULT_OPENDDE_DTYPE)
        self.runner_path = _resolve_runner_path(runner_path)

        assert self.opendde_python is not None
        assert self.opendde_executable is not None
        assert self.root_dir is not None
        assert self.model_name is not None
        assert self.seeds is not None
        assert self.dtype is not None

    def write_input(
        self,
        *,
        design_id: str,
        structure_id: str,
        sequence: str,
        work_dir: Path,
        msa_artifact: MSAArtifact | None = None,
    ) -> tuple[Path, Path]:
        input_dir = work_dir / "inputs" / structure_id / "opendde_monomer" / design_id
        output_dir = work_dir / "opendde_monomer_outputs" / structure_id / design_id
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        protein_chain: dict[str, Any] = {
            "sequence": sequence,
            "count": 1,
            "id": [OPEN_DDE_CHAIN_ANTIGEN],
        }
        json_path = input_dir / f"{design_id}.json"
        if msa_artifact is not None:
            if msa_artifact.format != MSAFormat.A3M:
                raise MSAUnsupportedError("OpenDDE accepts only A3M MSA artifacts")
            if not msa_artifact.path.exists():
                raise MSAUnsupportedError(f"OpenDDE MSA artifact does not exist: {msa_artifact.path}")
            protein_chain["unpairedMsaPath"] = str(msa_artifact.path)
            self.msa_sidecar_path(json_path).write_text(
                json.dumps(
                    {
                        "provider": msa_artifact.provider.value,
                        "path": str(msa_artifact.path),
                        "kind": msa_artifact.kind or msa_artifact.paired_or_unpaired.value,
                        "sequence_hash": msa_artifact.sequence_hash,
                        "query_sequence_hash": msa_artifact.query_sequence_hash,
                        "cache_status": msa_artifact.generation_status.value,
                        "sequence_count": msa_artifact.sequence_count,
                        "checksum": msa_artifact.checksum,
                        "model_consumed": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

        payload = [
            {
                "name": _json_safe_name(design_id),
                "modelSeeds": _parse_int_list(self.seeds),
                "sequences": [{"proteinChain": protein_chain}],
            }
        ]
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return json_path, output_dir

    @staticmethod
    def msa_sidecar_path(input_path: Path) -> Path:
        return input_path.with_suffix(".msa.json")

    def command(self, input_path: Path, output_dir: Path) -> str:
        has_msa = self.msa_sidecar_path(input_path).exists()
        parts = [
            shlex.quote(str(self.opendde_python)),
            shlex.quote(str(self.runner_path)),
            "--json",
            shlex.quote(str(input_path)),
            "--out-dir",
            shlex.quote(str(output_dir)),
            "--opendde-executable",
            shlex.quote(str(self.opendde_executable)),
            "--root-dir",
            shlex.quote(str(self.root_dir)),
            "--model-name",
            shlex.quote(str(self.model_name)),
            "--seeds",
            shlex.quote(str(self.seeds)),
            "--cycle",
            shlex.quote(str(self.cycle)),
            "--step",
            shlex.quote(str(self.step)),
            "--sample",
            shlex.quote(str(self.samples)),
            "--dtype",
            shlex.quote(str(self.dtype)),
            "--use-msa",
            "true" if has_msa else "false",
        ]
        if self.checkpoint is not None:
            parts.extend(["--checkpoint", shlex.quote(str(self.checkpoint))])
        if has_msa:
            parts.append("--require-msa")
        return " ".join(parts)

    def write_job_bundle(self, jobs: list[FoldJob], bundle_dir: Path) -> None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        path = bundle_dir / "opendde_monomer_jobs.tsv"
        cols = [
            "design_id",
            "structure_id",
            "fold_input",
            "fold_output_dir",
            "fold_command",
            "experiment_namespace",
            "redesign_mode",
            "mode_policy",
            "sequence_role",
            "sequence",
            "sequence_hash",
            "mpnn_score",
            "mpnn_rank",
            "fold_backend",
            "fold_job_id",
            "requested_mpnn_sequences",
            "max_decoy_sequences_for_folding",
            "actual_decoy_sequences_for_folding",
            "native_controls_selected_for_folding",
            "msa_status",
            "msa_mode",
            "msa_provider",
            "msa_path",
            "msa_kind",
            "msa_query_sequence_hash",
            "msa_cache_status",
            "msa_cache_hit",
            "msa_depth",
            "msa_model_consumed",
            "msa_consumed",
            "msa_source",
            "msa_native_sequence_hash",
            "msa_design_sequence_hash",
            "msa_native_a3m_path",
            "msa_derived_a3m_path",
            "msa_mutation_count",
            "msa_query_replacement_status",
            "msa_length_match_status",
            "msa_native_cache_hit",
        ]
        rows = []
        for job in jobs:
            metadata = dict(job.metadata)
            row = {
                "design_id": job.design_id,
                "structure_id": job.structure_id,
                "fold_input": str(job.input_path),
                "fold_output_dir": str(job.output_dir),
                "fold_command": job.command,
            }
            for column in cols[5:]:
                row[column] = metadata.get(column, "")
            row["fold_backend"] = metadata.get("fold_backend", job.backend)
            rows.append(row)
        pd.DataFrame(rows, columns=cols).to_csv(path, sep="\t", index=False)

        script = bundle_dir / "run_opendde_monomer_jobs.sh"
        script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "",
                    ": ${OPENDDE_PYTHON:?Set OPENDDE_PYTHON to the OpenDDE Python interpreter}",
                    ": ${OPENDDE_EXECUTABLE:?Set OPENDDE_EXECUTABLE to the OpenDDE CLI executable}",
                    ": ${OPENDDE_ROOT_DIR:?Set OPENDDE_ROOT_DIR to the OpenDDE model root}",
                    "",
                    "JOBS_TSV=${1:-work/decoy/job_bundle/opendde_monomer_jobs.tsv}",
                    "",
                    "tail -n +2 \"$JOBS_TSV\" | while IFS=$'\\t' read -r design_id structure_id json_path output_dir fold_command metadata_rest; do",
                    "  mkdir -p \"$output_dir\"",
                    "  echo \"[$design_id] $fold_command\"",
                    "  eval \"$fold_command\"",
                    "done",
                    "",
                ]
            )
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def discover_predictions(self, output_dir: Path) -> list[FoldPrediction]:
        if not output_dir.exists():
            return []
        manifest = output_dir / "opendde_prediction_manifest.csv"
        predictions: list[FoldPrediction] = []
        if manifest.is_file():
            with manifest.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    path = Path(str(row.get("prediction_path") or ""))
                    if not path.is_file():
                        continue
                    predictions.append(
                        FoldPrediction(
                            prediction_path=path,
                            confidence=_parse_opendde_confidence(row.get("confidence_json")),
                            backend=self.name,
                        )
                    )
            return predictions
        for path in sorted((output_dir / "predictions").glob("*.pdb")):
            predictions.append(FoldPrediction(prediction_path=path, confidence={}, backend=self.name))
        return predictions


def _parse_opendde_confidence(path_value: object) -> dict[str, object]:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    confidence: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            confidence[f"opendde_{key}"] = value
    return confidence
