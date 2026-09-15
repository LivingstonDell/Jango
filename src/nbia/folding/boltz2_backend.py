from __future__ import annotations

import json
import os
import shlex
import stat
from pathlib import Path

import pandas as pd

from .base import FoldJob, FoldPrediction
from .msa import BackendMSASupport, MSAArtifact, MSACapability, MSAFormat, MSAUnsupportedError
from ..boltz2 import BOLTZ_CHAIN_ANTIGEN, discover_boltz_predictions, parse_confidence_json


DEFAULT_BOLTZ_MODEL = "boltz2"


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
    return Path(value).expanduser().resolve()


class Boltz2Backend:
    """Boltz2 monomer folding backend for redesigned antigen sequences.

    Machine-specific deployment paths are supplied through environment variables
    or constructor arguments.

    Required when executing generated jobs:

        BOLTZ_EXECUTABLE or BOLTZ

    Optional:

        BOLTZ_CACHE
    """

    name = "boltz2"
    msa_support = BackendMSASupport(
        capability=MSACapability.BACKEND_MANAGED,
        generates_msa=True,
        accepts_precomputed=False,
        generated_artifact_formats=(MSAFormat.BOLTZ_CSV, MSAFormat.A3M),
        paired_multichain=True,
        monomer=True,
        multichain=False,
        single_sequence_fallback_allowed=True,
        notes="Local source proves backend-managed MSA server output, but not precomputed YAML input.",
    )

    def __init__(
        self,
        *,
        boltz_executable: str | Path | None = None,
        cache_dir: str | Path | None = None,
        model: str | None = None,
        use_msa_server: bool = True,
        use_potentials: bool = True,
        diffusion_samples: int = 5,
        recycling_steps: int = 3,
        output_format: str = "pdb",
        write_full_pae: bool = True,
        write_full_pde: bool = True,
    ) -> None:
        self.boltz_executable = (
            Path(boltz_executable).expanduser().resolve()
            if boltz_executable
            else _env_path("BOLTZ_EXECUTABLE", os.environ.get("BOLTZ"), required=False)
        )
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir
            else _env_path("BOLTZ_CACHE", required=False)
        )
        self.model = model or _env_value("BOLTZ_MODEL", DEFAULT_BOLTZ_MODEL)
        self.use_msa_server = use_msa_server
        self.use_potentials = use_potentials
        self.diffusion_samples = diffusion_samples
        self.recycling_steps = recycling_steps
        self.output_format = output_format
        self.write_full_pae = write_full_pae
        self.write_full_pde = write_full_pde

    def write_input(
        self,
        *,
        design_id: str,
        structure_id: str,
        sequence: str,
        work_dir: Path,
        msa_artifact: MSAArtifact | None = None,
    ) -> tuple[Path, Path]:
        if msa_artifact is not None:
            raise MSAUnsupportedError("Boltz2 precomputed MSA input is not wired from local API evidence")

        input_dir = work_dir / "inputs" / structure_id / "boltz2_monomer" / design_id
        output_dir = work_dir / "boltz2_monomer_outputs" / structure_id / design_id
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = input_dir / f"{design_id}.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "sequences:",
                    "  - protein:",
                    f"      id: {BOLTZ_CHAIN_ANTIGEN}",
                    f"      sequence: {sequence}",
                    "",
                ]
            )
        )
        return yaml_path, output_dir

    def _command_parts(self, input_path: Path, output_dir: Path) -> list[str]:
        boltz = str(self.boltz_executable) if self.boltz_executable else "${BOLTZ:-boltz}"

        parts = [
            boltz,
            "predict",
        ]

        if self.cache_dir is not None:
            parts.extend(["--cache", str(self.cache_dir)])

        parts.extend(
            [
                str(input_path),
                "--model",
                str(self.model),
            ]
        )

        if self.use_msa_server:
            parts.append("--use_msa_server")
        if self.use_potentials:
            parts.append("--use_potentials")

        parts.extend(
            [
                "--diffusion_samples",
                str(self.diffusion_samples),
                "--recycling_steps",
                str(self.recycling_steps),
                "--output_format",
                str(self.output_format),
            ]
        )

        if self.write_full_pae:
            parts.append("--write_full_pae")
        if self.write_full_pde:
            parts.append("--write_full_pde")

        parts.extend(["--out_dir", str(output_dir)])
        return parts

    def command(self, input_path: Path, output_dir: Path) -> str:
        return " ".join(shlex.quote(part) for part in self._command_parts(input_path, output_dir))

    def write_job_bundle(self, jobs: list[FoldJob], bundle_dir: Path) -> None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        path = bundle_dir / "boltz2_monomer_jobs.tsv"
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

        script = bundle_dir / "run_boltz2_monomer_jobs.sh"
        script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "",
                    "BOLTZ=${BOLTZ_EXECUTABLE:-${BOLTZ:-boltz}}",
                    "",
                    "if [ -n \"${BOLTZ_CACHE:-}\" ]; then",
                    "  mkdir -p \"$BOLTZ_CACHE\"",
                    "fi",
                    "",
                    "JOBS_TSV=${1:-work/decoys/job_bundle/boltz2_monomer_jobs.tsv}",
                    "",
                    "tail -n +2 \"$JOBS_TSV\" | while IFS=$'\\t' read -r design_id structure_id yaml_path output_dir fold_command metadata_rest; do",
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
        predictions = []
        for path in discover_boltz_predictions(output_dir):
            predictions.append(
                FoldPrediction(
                    prediction_path=Path(path),
                    confidence=parse_confidence_json(Path(path)),
                    backend=self.name,
                )
            )
        return predictions
