#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

from run_opendde_monomer import (
    confidence_for_cif,
    convert_cif_to_pdb,
    load_job,
    parse_bool,
    ranking_score,
    sample_rank,
    validate_msa_paths,
)


STATUS_COLUMNS = [
    "design_id",
    "structure_id",
    "redesign_mode",
    "sequence_role",
    "fold_output_dir",
    "completion_status",
    "exit_code",
    "prediction_count",
    "error",
]


def _env_int(name: str) -> int | None:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cuda_devices() -> list[str]:
    value = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def default_workers() -> int:
    configured = _env_int("OPENDDE_BATCH_WORKERS") or _env_int("SLURM_MAX_CONCURRENT")
    if configured and configured > 0:
        return configured
    devices = _cuda_devices()
    return max(1, len(devices))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a TSV bundle of OpenDDE monomer jobs with one OpenDDE process per worker shard.")
    parser.add_argument("--jobs-tsv", required=True, type=Path)
    parser.add_argument("--status-csv", required=True, type=Path)
    parser.add_argument("--opendde-executable", required=True, type=Path)
    parser.add_argument("--root-dir", required=True, type=Path, help="OpenDDE model root, exported as OPENDDE_ROOT_DIR.")
    parser.add_argument("--model-name", default="opendde_v1")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--cycle", default="10")
    parser.add_argument("--step", default="200")
    parser.add_argument("--sample", default="5")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--use-msa", type=parse_bool, default=True)
    parser.add_argument("--require-msa", action="store_true")
    parser.add_argument("--workers", type=int, default=default_workers())
    return parser


def read_jobs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def prediction_outputs_complete(output_dir: str | Path) -> bool:
    path = Path(output_dir)
    if not path.is_dir():
        return False
    for suffix in ("*.pdb", "*.cif", "*.mmcif"):
        for candidate in path.rglob(suffix):
            if "raw_opendde" in candidate.parts:
                continue
            return True
    return False


def write_status(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in STATUS_COLUMNS})


def _json_safe_name(value: str) -> str:
    import re

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return safe or "opendde_job"


def job_name(row: dict[str, str], job: dict) -> str:
    value = str(job.get("name") or "").strip()
    if value:
        return value
    return _json_safe_name(str(row.get("design_id") or Path(row.get("fold_input", "job")).stem))


def find_job_cifs(raw_dir: Path, name: str) -> list[Path]:
    preferred = raw_dir / name
    cifs: list[Path] = []
    if preferred.exists():
        cifs.extend(sorted(preferred.rglob("*.cif")))
        cifs.extend(sorted(preferred.rglob("*.mmcif")))
    if cifs:
        return cifs
    cifs.extend(sorted(raw_dir.rglob(f"{name}*.cif")))
    cifs.extend(sorted(raw_dir.rglob(f"{name}*.mmcif")))
    return cifs


def normalize_predictions(raw_dir: Path, out_dir: Path, name: str) -> list[dict[str, str]]:
    cif_paths = find_job_cifs(raw_dir, name)
    if not cif_paths:
        raise FileNotFoundError(f"OpenDDE produced no CIF predictions for {name} under {raw_dir}")
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for cif_path in cif_paths:
        pdb_path = prediction_dir / f"{cif_path.stem}.pdb"
        convert_cif_to_pdb(cif_path, pdb_path)
        confidence_json = confidence_for_cif(cif_path)
        copied_confidence = ""
        if confidence_json is not None:
            copied = prediction_dir / f"{pdb_path.stem}_confidence.json"
            shutil.copyfile(confidence_json, copied)
            copied_confidence = str(copied)
        rows.append(
            {
                "prediction_path": str(pdb_path),
                "source_cif": str(cif_path),
                "confidence_json": copied_confidence,
                "sample_rank": sample_rank(cif_path),
                "ranking_score": ranking_score(confidence_json),
            }
        )
    manifest = out_dir / "opendde_prediction_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prediction_path", "source_cif", "confidence_json", "sample_rank", "ranking_score"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def row_log(row: dict[str, str], *, message: str, error: str = "") -> None:
    stdout_value = str(row.get("stdout_log") or "").strip()
    stderr_value = str(row.get("stderr_log") or "").strip()
    if stdout_value:
        stdout_log = Path(stdout_value)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        stdout_log.write_text(message + "\n")
    if stderr_value:
        stderr_log = Path(stderr_value)
        stderr_log.parent.mkdir(parents=True, exist_ok=True)
        stderr_log.write_text((error + "\n") if error else "")


def prepare_rows(rows: list[dict[str, str]], *, require_msa: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prepared: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for row in rows:
        out_dir = Path(str(row.get("fold_output_dir") or ""))
        if prediction_outputs_complete(out_dir):
            statuses.append(status_row(row, "COMPLETED", "0", prediction_count="", error=""))
            row_log(row, message="skipped existing complete OpenDDE output")
            continue
        try:
            job = load_job(Path(str(row.get("fold_input") or "")))
            validate_msa_paths(job, require_msa=require_msa)
        except Exception as exc:
            statuses.append(status_row(row, "FAILED", "1", prediction_count="0", error=str(exc)))
            row_log(row, message="OpenDDE input validation failed", error=str(exc))
            continue
        prepared.append({"row": row, "job": job, "name": job_name(row, job)})
        statuses.append(status_row(row, "QUEUED", "", prediction_count="", error=""))
    return prepared, statuses


def status_row(row: dict[str, str], completion_status: str, exit_code: str, *, prediction_count: str | int = "", error: str = "") -> dict[str, object]:
    return {
        "design_id": row.get("design_id", ""),
        "structure_id": row.get("structure_id", ""),
        "redesign_mode": row.get("redesign_mode", ""),
        "sequence_role": row.get("sequence_role", ""),
        "fold_output_dir": row.get("fold_output_dir", ""),
        "completion_status": completion_status,
        "exit_code": exit_code,
        "prediction_count": prediction_count,
        "error": error,
    }


def shard_items(items: list[dict[str, object]], workers: int) -> list[list[dict[str, object]]]:
    shards = [[] for _ in range(max(1, workers))]
    for idx, item in enumerate(items):
        shards[idx % len(shards)].append(item)
    return [shard for shard in shards if shard]


def run_shard(args: argparse.Namespace, shard_index: int, items: list[dict[str, object]], batch_dir: Path) -> list[dict[str, object]]:
    shard_dir = batch_dir / f"shard_{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_json = shard_dir / "opendde_input.json"
    raw_dir = shard_dir / "raw_opendde"
    shard_json.write_text(json.dumps([item["job"] for item in items], indent=2, sort_keys=True) + "\n")

    env = dict(os.environ)
    env["OPENDDE_ROOT_DIR"] = str(args.root_dir)
    devices = _cuda_devices()
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = devices[shard_index % len(devices)]

    command = [
        str(args.opendde_executable),
        "pred",
        "-i",
        str(shard_json),
        "-o",
        str(raw_dir),
        "--model_name",
        str(args.model_name),
        "--seeds",
        str(args.seeds),
        "--cycle",
        str(args.cycle),
        "--step",
        str(args.step),
        "--sample",
        str(args.sample),
        "--dtype",
        str(args.dtype),
        "--use_msa",
        "true" if args.use_msa else "false",
    ]
    if args.checkpoint is not None:
        command.extend(["--load_checkpoint_path", str(args.checkpoint)])

    stdout_path = shard_dir / "opendde.out"
    stderr_path = shard_dir / "opendde.err"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(command, env=env, text=True, stdout=stdout, stderr=stderr, check=False)

    statuses: list[dict[str, object]] = []
    for item in items:
        row = item["row"]
        out_dir = Path(str(row.get("fold_output_dir") or ""))
        try:
            predictions = normalize_predictions(raw_dir, out_dir, str(item["name"]))
            status = "COMPLETED"
            exit_code = "0"
            error = "" if completed.returncode == 0 else f"OpenDDE exited {completed.returncode} after producing predictions"
            row_log(row, message=f"OpenDDE shard {shard_index} normalized {len(predictions)} prediction(s)", error=error)
            statuses.append(status_row(row, status, exit_code, prediction_count=len(predictions), error=error))
        except Exception as exc:
            if prediction_outputs_complete(out_dir):
                statuses.append(status_row(row, "COMPLETED", "0", prediction_count="", error=""))
                row_log(row, message=f"OpenDDE shard {shard_index} found existing complete output")
            else:
                detail = str(exc)
                if completed.returncode != 0:
                    detail = f"OpenDDE exited {completed.returncode}; {detail}"
                statuses.append(status_row(row, "FAILED", str(completed.returncode or 1), prediction_count="0", error=detail))
                row_log(row, message=f"OpenDDE shard {shard_index} failed", error=detail)
    return statuses


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.require_msa and not args.use_msa:
        raise ValueError("--require-msa conflicts with --use-msa false")
    workers = max(1, int(args.workers or 1))
    rows = read_jobs(args.jobs_tsv)
    prepared, statuses = prepare_rows(rows, require_msa=args.require_msa)
    write_status(args.status_csv, statuses)
    if not prepared:
        return 0 if all(str(row.get("completion_status")) == "COMPLETED" for row in statuses) else 1

    batch_dir = args.status_csv.parent / f"{args.status_csv.stem}_opendde_batch"
    shards = shard_items(prepared, workers)
    final_by_design = {str(row.get("design_id")): row for row in statuses}
    with ThreadPoolExecutor(max_workers=min(workers, len(shards))) as pool:
        futures = {pool.submit(run_shard, args, idx, shard, batch_dir): idx for idx, shard in enumerate(shards)}
        for future in as_completed(futures):
            for row in future.result():
                final_by_design[str(row.get("design_id"))] = row
            ordered = [final_by_design[str(row.get("design_id"))] for row in statuses]
            write_status(args.status_csv, ordered)

    ordered = [final_by_design[str(row.get("design_id"))] for row in statuses]
    write_status(args.status_csv, ordered)
    failed = [row for row in ordered if str(row.get("completion_status")) != "COMPLETED"]
    print(f"OpenDDE batch complete: {len(ordered) - len(failed)} completed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
