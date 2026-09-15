#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable

from run_esmfold2_monomer import (
    DEFAULT_ESMFOLD2_DEVICE,
    DEFAULT_ESMFOLD2_MODEL,
    LocalESMFold2Session,
    env_path,
    env_value,
    read_fasta,
    write_prediction_outputs,
)


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def write_status(path: Path, rows: Iterable[dict[str, object]]) -> None:
    columns = [
        "design_id",
        "structure_id",
        "redesign_mode",
        "sequence_role",
        "fold_output_dir",
        "completion_status",
        "exit_code",
        "error",
        "msa_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def sidecar_msa_path(fasta_path: Path) -> Path | None:
    sidecar = fasta_path.with_suffix(".msa.json")
    if not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text())
    raw = payload.get("path")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def row_msa_path(row: dict[str, str], fasta_path: Path) -> Path | None:
    for key in ("msa_path", "msa_derived_a3m_path", "msa_native_a3m_path"):
        raw = str(row.get(key) or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return sidecar_msa_path(fasta_path)


def row_requires_msa(row: dict[str, str]) -> bool:
    command = str(row.get("fold_command") or "")
    return (
        "--require-msa" in command
        or truthy(row.get("msa_consumed"))
        or truthy(row.get("msa_model_consumed"))
    )


def output_complete(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    return any(output_dir.rglob("*.pdb")) or any(output_dir.rglob("*.cif")) or any(output_dir.rglob("*.mmcif"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold ESMFold2 monomer jobs sequentially with one loaded model session."
    )
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path, required=True)
    parser.add_argument("--model-id-or-path", default=env_value("ESMFOLD2_MODEL", DEFAULT_ESMFOLD2_MODEL))
    parser.add_argument("--cache-dir", type=Path, default=env_path("ESMFOLD2_CACHE"))
    parser.add_argument("--esm-root", type=Path, default=env_path("ESM_ROOT"))
    parser.add_argument("--device", default=env_value("ESMFOLD2_DEVICE", DEFAULT_ESMFOLD2_DEVICE))
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--msa-max-sequences", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.jobs_tsv = args.jobs_tsv.expanduser().resolve()
    args.status_csv = args.status_csv.expanduser().resolve()
    if args.cache_dir is None:
        raise SystemExit("Required environment variable is not set: ESMFOLD2_CACHE. Set it or pass --cache-dir.")
    if args.esm_root is None:
        raise SystemExit("Required environment variable is not set: ESM_ROOT. Set it or pass --esm-root.")
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.esm_root = args.esm_root.expanduser().resolve()
    if args.num_diffusion_samples < 1:
        raise SystemExit("--num-diffusion-samples must be >= 1")
    if args.num_sampling_steps is not None and args.num_sampling_steps < 1:
        raise SystemExit("--num-sampling-steps must be >= 1")

    os.environ.setdefault("ESM_ROOT", str(args.esm_root))
    os.environ.setdefault("ESMFOLD2_CACHE", str(args.cache_dir))
    os.environ.setdefault("HF_HOME", str(args.cache_dir / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(args.cache_dir / "torch"))

    if str(args.esm_root) not in sys.path:
        sys.path.insert(0, str(args.esm_root))

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "huggingface").mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "torch").mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.jobs_tsv)
    statuses: list[dict[str, object]] = []
    write_status(args.status_csv, statuses)
    session = LocalESMFold2Session(
        model_id_or_path=args.model_id_or_path,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    for row in rows:
        design_id = str(row.get("design_id") or "")
        structure_id = str(row.get("structure_id") or "")
        redesign_mode = str(row.get("redesign_mode") or "")
        sequence_role = str(row.get("sequence_role") or "")
        fasta_path = Path(str(row.get("fold_input") or "")).expanduser().resolve()
        output_dir = Path(str(row.get("fold_output_dir") or "")).expanduser().resolve()
        stdout_log = Path(str(row.get("stdout_log") or output_dir / f"{design_id}.out")).expanduser()
        stderr_log = Path(str(row.get("stderr_log") or output_dir / f"{design_id}.err")).expanduser()
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        status: dict[str, object] = {
            "design_id": design_id,
            "structure_id": structure_id,
            "redesign_mode": redesign_mode,
            "sequence_role": sequence_role,
            "fold_output_dir": str(output_dir),
            "completion_status": "RUNNING",
            "exit_code": "",
            "error": "",
            "msa_path": "",
        }
        statuses.append(status)
        write_status(args.status_csv, statuses)
        with stdout_log.open("w") as stdout, stderr_log.open("w") as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    print(f"[{design_id}] folding {fasta_path} -> {output_dir}")
                    name, sequence = read_fasta(fasta_path)
                    msa_path = row_msa_path(row, fasta_path)
                    if row_requires_msa(row) and msa_path is None:
                        raise FileNotFoundError(f"required MSA A3M was not found for {design_id}")
                    if msa_path is not None and not msa_path.is_file():
                        raise FileNotFoundError(f"MSA A3M file does not exist: {msa_path}")
                    status["msa_path"] = str(msa_path or "")
                    result, pdb_text, msa_metadata = session.fold(
                        sequence=sequence,
                        num_recycles=args.num_recycles,
                        msa_a3m=msa_path,
                        msa_max_sequences=args.msa_max_sequences,
                        num_sampling_steps=args.num_sampling_steps,
                        num_diffusion_samples=args.num_diffusion_samples,
                        seed=args.seed,
                    )
                    if isinstance(result, list):
                        from run_esmfold2_monomer import select_best_result

                        best_result, best_sample_index = select_best_result(result)
                        for sample_index, sample_result in enumerate(result):
                            sample_name = f"{name}_sample{sample_index:02d}"
                            write_prediction_outputs(
                                name=sample_name,
                                sequence=sequence,
                                result=sample_result,
                                pdb_text=None,
                                msa_metadata=msa_metadata,
                                out_dir=output_dir / "samples",
                                model_id_or_path=args.model_id_or_path,
                                device=args.device,
                                cache_dir=args.cache_dir,
                                sample_index=sample_index,
                                num_diffusion_samples=args.num_diffusion_samples,
                                selected_sample_index=best_sample_index,
                                num_sampling_steps=args.num_sampling_steps,
                                seed=args.seed,
                            )
                        write_prediction_outputs(
                            name=name,
                            sequence=sequence,
                            result=best_result,
                            pdb_text=None,
                            msa_metadata=msa_metadata,
                            out_dir=output_dir,
                            model_id_or_path=args.model_id_or_path,
                            device=args.device,
                            cache_dir=args.cache_dir,
                            sample_index=best_sample_index,
                            num_diffusion_samples=args.num_diffusion_samples,
                            selected_sample_index=best_sample_index,
                            num_sampling_steps=args.num_sampling_steps,
                            seed=args.seed,
                        )
                    else:
                        write_prediction_outputs(
                            name=name,
                            sequence=sequence,
                            result=result,
                            pdb_text=pdb_text,
                            msa_metadata=msa_metadata,
                            out_dir=output_dir,
                            model_id_or_path=args.model_id_or_path,
                            device=args.device,
                            cache_dir=args.cache_dir,
                            num_diffusion_samples=args.num_diffusion_samples,
                            num_sampling_steps=args.num_sampling_steps,
                            seed=args.seed,
                        )
                    if output_complete(output_dir):
                        status["completion_status"] = "COMPLETED"
                        status["exit_code"] = "0"
                    else:
                        status["completion_status"] = "FAILED_OUTPUT_MISSING"
                        status["exit_code"] = "0"
                except Exception as exc:
                    status["completion_status"] = "FAILED"
                    status["exit_code"] = "1"
                    status["error"] = str(exc)
                    traceback.print_exc()
                finally:
                    session.clear_memory()
        write_status(args.status_csv, statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
