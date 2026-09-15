from __future__ import annotations

import json
import os
import shlex
import stat
from string import Template
from pathlib import Path

import pandas as pd

from .base import FoldJob, FoldPrediction
from .msa import BackendMSASupport, MSAArtifact, MSACapability, MSAFormat, MSAUnsupportedError


DEFAULT_ESMFOLD2_MODEL = "Biohub/ESMFold2"
DEFAULT_ESMFOLD2_DEVICE = "cuda"
DEFAULT_ESMFOLD2_RUNNER = "${JANGO_SOURCE}/scripts/nbia/run_esmfold2_monomer.py"


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


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _module_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runner_template_env() -> dict[str, str]:
    env = dict(os.environ)
    source = env.get("JANGO_SOURCE") or env.get("NBIA_ROOT")
    if not source:
        candidate = _module_repo_root()
        if (candidate / "scripts" / "nbia" / "run_esmfold2_monomer.py").is_file():
            source = str(candidate)
    if source:
        env.setdefault("JANGO_SOURCE", source)
        env.setdefault("NBIA_ROOT", source)
    return env


def _resolve_runner_path(runner_path: str | Path | None = None) -> Path:
    raw = str(runner_path or _env_value("ESMFOLD2_RUNNER", DEFAULT_ESMFOLD2_RUNNER))
    env = _runner_template_env()
    expanded = Template(raw).safe_substitute(env)
    if "$" in expanded:
        raise RuntimeError(
            f"ESMFold2 runner path could not be resolved from {raw!r}; "
            "set ESMFOLD2_RUNNER or JANGO_SOURCE/NBIA_ROOT"
        )
    path = Path(expanded).expanduser()
    if not path.is_absolute() and env.get("JANGO_SOURCE"):
        path = Path(env["JANGO_SOURCE"]).expanduser() / path
    if not path.is_file():
        raise FileNotFoundError(f"ESMFold2 runner script not found: {path}")
    return path.resolve()


class ESMFold2Backend:
    """Local-only ESMFold2 monomer folding backend.

    This backend does not use the Biohub API and does not require BIOHUB_TOKEN.
    It runs local ESMFold2 through a deployment-provided Python interpreter and
    local ESM repository.

    Required deployment variables unless explicit constructor arguments are passed:

        ESMFOLD2_PYTHON
        ESM_ROOT
        ESMFOLD2_CACHE
        ESMFOLD2_RUNNER

    Optional deployment variables:

        ESMFOLD2_MODEL
        ESMFOLD2_DEVICE
    """

    name = "esmfold2"
    msa_support = BackendMSASupport(
        capability=MSACapability.OPTIONAL,
        accepts_precomputed=True,
        generates_msa=False,
        accepted_input_formats=(MSAFormat.A3M,),
        monomer=True,
        multichain=False,
        single_sequence_fallback_allowed=True,
        notes="Local runner accepts validated unpaired A3M input and passes it through ESMFold2InputBuilder to model.forward MSA features.",
    )

    def __init__(
        self,
        *,
        esmfold2_python: str | Path | None = None,
        esm_root: str | Path | None = None,
        model_id_or_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        runner_path: str | Path | None = None,
        num_sampling_steps: int | None = None,
        num_diffusion_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.esmfold2_python = Path(esmfold2_python).expanduser() if esmfold2_python else _env_path(
            "ESMFOLD2_PYTHON",
            required=True,
        )
        self.esm_root = Path(esm_root).expanduser().resolve() if esm_root else _env_path(
            "ESM_ROOT",
            required=True,
        )
        self.model_id_or_path = str(
            model_id_or_path
            or _env_value("ESMFOLD2_MODEL", DEFAULT_ESMFOLD2_MODEL)
        )
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else _env_path(
            "ESMFOLD2_CACHE",
            required=True,
        )
        self.device = device or _env_value("ESMFOLD2_DEVICE", DEFAULT_ESMFOLD2_DEVICE)
        self.num_sampling_steps = num_sampling_steps if num_sampling_steps is not None else _env_int("ESMFOLD2_NUM_SAMPLING_STEPS")
        self.num_diffusion_samples = num_diffusion_samples if num_diffusion_samples is not None else _env_int("ESMFOLD2_NUM_DIFFUSION_SAMPLES", 1)
        self.seed = seed if seed is not None else _env_int("ESMFOLD2_SEED")
        self.runner_path = _resolve_runner_path(runner_path)

        assert self.esmfold2_python is not None
        assert self.esm_root is not None
        assert self.cache_dir is not None
        assert self.device is not None
        assert self.runner_path is not None
        if self.num_diffusion_samples is None or self.num_diffusion_samples < 1:
            raise ValueError("ESMFOLD2_NUM_DIFFUSION_SAMPLES must be >= 1")
        if self.num_sampling_steps is not None and self.num_sampling_steps < 1:
            raise ValueError("ESMFOLD2_NUM_SAMPLING_STEPS must be >= 1")

    def write_input(
        self,
        *,
        design_id: str,
        structure_id: str,
        sequence: str,
        work_dir: Path,
        msa_artifact: MSAArtifact | None = None,
    ) -> tuple[Path, Path]:
        input_dir = work_dir / "inputs" / structure_id / "esmfold2_monomer" / design_id
        output_dir = work_dir / "esmfold2_monomer_outputs" / structure_id / design_id
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = input_dir / f"{design_id}.fa"
        fasta_path.write_text(f">{design_id}\n{sequence}\n")

        if msa_artifact is not None:
            if msa_artifact.format != MSAFormat.A3M:
                raise MSAUnsupportedError("ESMFold2 accepts only A3M MSA artifacts")
            if not msa_artifact.path.exists():
                raise MSAUnsupportedError(f"ESMFold2 MSA artifact does not exist: {msa_artifact.path}")
            sidecar = self.msa_sidecar_path(fasta_path)
            sidecar.write_text(
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
        return fasta_path, output_dir

    @staticmethod
    def msa_sidecar_path(fasta_path: Path) -> Path:
        return fasta_path.with_suffix(".msa.json")

    def command_msa_args(self, input_path: Path) -> list[str]:
        sidecar = self.msa_sidecar_path(input_path)
        if not sidecar.exists():
            return []
        payload = json.loads(sidecar.read_text())
        msa_path = payload.get("path")
        if not msa_path:
            raise MSAUnsupportedError(f"ESMFold2 MSA sidecar is missing path: {sidecar}")
        return ["--msa-a3m", str(msa_path), "--require-msa"]

    def command(self, input_path: Path, output_dir: Path) -> str:
        runner = shlex.quote(str(self.runner_path))
        parts = [
            shlex.quote(str(self.esmfold2_python)),
            runner,
            "--fasta",
            shlex.quote(str(input_path)),
            "--out-dir",
            shlex.quote(str(output_dir)),
            "--model-id-or-path",
            shlex.quote(str(self.model_id_or_path)),
            "--cache-dir",
            shlex.quote(str(self.cache_dir)),
            "--device",
            shlex.quote(str(self.device)),
            "--esm-root",
            shlex.quote(str(self.esm_root)),
        ]
        if self.num_sampling_steps is not None:
            parts.extend(["--num-sampling-steps", shlex.quote(str(self.num_sampling_steps))])
        if self.num_diffusion_samples != 1:
            parts.extend(["--num-diffusion-samples", shlex.quote(str(self.num_diffusion_samples))])
        if self.seed is not None:
            parts.extend(["--seed", shlex.quote(str(self.seed))])
        for arg in self.command_msa_args(input_path):
            parts.append(shlex.quote(str(arg)))
        return " ".join(parts)

    def write_job_bundle(self, jobs: list[FoldJob], bundle_dir: Path) -> None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        path = bundle_dir / "esmfold2_monomer_jobs.tsv"
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
            "esmfold2_num_sampling_steps",
            "esmfold2_num_diffusion_samples",
            "esmfold2_seed",
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
            row["esmfold2_num_sampling_steps"] = self.num_sampling_steps or ""
            row["esmfold2_num_diffusion_samples"] = self.num_diffusion_samples
            row["esmfold2_seed"] = self.seed or ""
            rows.append(row)
        pd.DataFrame(rows, columns=cols).to_csv(path, sep="\t", index=False)

        script = bundle_dir / "run_esmfold2_monomer_jobs.sh"
        script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "",
                    ": ${NBIA_ROOT:?Set NBIA_ROOT to the Atlas repository root}",
                    ": ${ESM_ROOT:?Set ESM_ROOT to the local ESM repository root}",
                    ": ${ESMFOLD2_PYTHON:?Set ESMFOLD2_PYTHON to the local ESMFold2 Python interpreter}",
                    ": ${ESMFOLD2_CACHE:?Set ESMFOLD2_CACHE to the ESMFold2/Hugging Face cache root}",
                    "",
                    f"export ESMFOLD2_MODEL=${{ESMFOLD2_MODEL:-{shlex.quote(str(self.model_id_or_path))}}}",
                    f"export ESMFOLD2_DEVICE=${{ESMFOLD2_DEVICE:-{shlex.quote(str(self.device))}}}",
                    "",
                    "ESMFOLD2_EXTRA_ARGS=(--num-diffusion-samples \"${ESMFOLD2_NUM_DIFFUSION_SAMPLES:-1}\")",
                    "if [ -n \"${ESMFOLD2_NUM_SAMPLING_STEPS:-}\" ]; then ESMFOLD2_EXTRA_ARGS+=(--num-sampling-steps \"$ESMFOLD2_NUM_SAMPLING_STEPS\"); fi",
                    "if [ -n \"${ESMFOLD2_SEED:-}\" ]; then ESMFOLD2_EXTRA_ARGS+=(--seed \"$ESMFOLD2_SEED\"); fi",
                    "",
                    "export PYTHONPATH=\"$ESM_ROOT:${PYTHONPATH:-}\"",
                    "export HF_HOME=\"$ESMFOLD2_CACHE/huggingface\"",
                    "export TRANSFORMERS_CACHE=\"$ESMFOLD2_CACHE/huggingface\"",
                    "export TORCH_HOME=\"$ESMFOLD2_CACHE/torch\"",
                    "",
                    "mkdir -p \"$HF_HOME\" \"$TORCH_HOME\"",
                    "",
                    "JOBS_TSV=${1:-work/decoy/job_bundle/esmfold2_monomer_jobs.tsv}",
                    "STATUS_CSV=${2:-${JOBS_TSV%.tsv}_status.csv}",
                    "",
                    "\"$ESMFOLD2_PYTHON\" \"$NBIA_ROOT/scripts/nbia/run_esmfold2_monomer_jobs.py\" \\",
                    "  --jobs-tsv \"$JOBS_TSV\" \\",
                    "  --status-csv \"$STATUS_CSV\" \\",
                    "  --model-id-or-path \"$ESMFOLD2_MODEL\" \\",
                    "  --cache-dir \"$ESMFOLD2_CACHE\" \\",
                    "  --device \"$ESMFOLD2_DEVICE\" \\",
                    "  --esm-root \"$ESM_ROOT\" \\",
                    "  \"${ESMFOLD2_EXTRA_ARGS[@]}\"",
                    "",
                ]
            )
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def discover_predictions(self, output_dir: Path) -> list[FoldPrediction]:
        if not output_dir.exists():
            return []

        paths = sorted(output_dir.glob("*.pdb"))
        if not paths:
            paths = [
                path
                for path in sorted(output_dir.rglob("*.pdb"))
                if "samples" not in path.relative_to(output_dir).parts
            ]
        if not paths:
            paths = [
                path
                for path in sorted([*output_dir.rglob("*.cif"), *output_dir.rglob("*.mmcif")])
                if "samples" not in path.relative_to(output_dir).parts
            ]

        predictions: list[FoldPrediction] = []
        for path in paths:
            confidence = _parse_confidence(path.parent)
            predictions.append(
                FoldPrediction(
                    prediction_path=path,
                    confidence=confidence,
                    backend=self.name,
                )
            )
        return predictions


def _parse_confidence(output_dir: Path) -> dict[str, object]:
    json_files = sorted(output_dir.rglob("*confidence*.json")) + sorted(
        output_dir.rglob("*metrics*.json")
    )
    if not json_files:
        return {}

    try:
        data = json.loads(json_files[0].read_text())
    except Exception:
        return {}

    out: dict[str, object] = {}
    for key in [
        "ptm",
        "iptm",
        "mean_plddt",
        "mean_pae",
        "model_id_or_path",
        "device",
        "backend",
    ]:
        if key in data:
            out[f"esmfold2_{key}"] = data[key]
    return out
