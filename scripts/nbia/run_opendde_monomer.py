#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one OpenDDE monomer job and normalize CIF predictions to PDB.")
    parser.add_argument("--json", required=True, type=Path, help="OpenDDE input JSON produced by Jango.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Jango fold output directory.")
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
    return parser


def load_job(input_json: Path) -> dict:
    payload = json.loads(input_json.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"OpenDDE monomer input must contain exactly one job: {input_json}")
    return payload[0]


def validate_msa_paths(job: dict, *, require_msa: bool) -> None:
    protein_chains = [
        entry.get("proteinChain")
        for entry in job.get("sequences", [])
        if isinstance(entry, dict) and entry.get("proteinChain")
    ]
    if not protein_chains:
        raise ValueError("OpenDDE monomer input contains no proteinChain")
    if len(protein_chains) != 1:
        raise ValueError("Jango OpenDDE backend expects exactly one monomer proteinChain")
    for chain in protein_chains:
        msa_paths = [chain.get("unpairedMsaPath"), chain.get("pairedMsaPath")]
        present = [Path(str(path)).expanduser() for path in msa_paths if path]
        if require_msa and not present:
            raise ValueError("OpenDDE MSA is required but no MSA path is present in JSON")
        for path in present:
            if not path.is_file():
                raise FileNotFoundError(f"OpenDDE MSA path does not exist: {path}")


def run_opendde(args: argparse.Namespace) -> None:
    raw_dir = args.out_dir / "raw_opendde"
    raw_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OPENDDE_ROOT_DIR"] = str(args.root_dir)
    command = [
        str(args.opendde_executable),
        "pred",
        "-i",
        str(args.json),
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
    completed = subprocess.run(command, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"OpenDDE prediction failed with exit code {completed.returncode}")


def sample_rank(path: Path) -> str:
    match = re.search(r"_sample_([0-9]+)$", path.stem)
    return match.group(1) if match else ""


def confidence_for_cif(cif_path: Path) -> Path | None:
    rank = sample_rank(cif_path)
    patterns = []
    if rank:
        patterns.extend([f"*summary_confidence*sample_{rank}.json", f"*confidence*sample_{rank}.json"])
    patterns.append("*confidence*.json")
    for pattern in patterns:
        matches = sorted(cif_path.parent.glob(pattern))
        if matches:
            return matches[0]
    return None


def ranking_score(confidence_json: Path | None) -> str:
    if confidence_json is None or not confidence_json.is_file():
        return ""
    try:
        data = json.loads(confidence_json.read_text())
    except Exception:
        return ""
    value = data.get("ranking_score")
    return "" if value is None else str(value)


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(pdb_path))


def normalize_predictions(out_dir: Path) -> list[dict[str, str]]:
    raw_dir = out_dir / "raw_opendde"
    cif_paths = sorted(raw_dir.rglob("*.cif")) + sorted(raw_dir.rglob("*.mmcif"))
    if not cif_paths:
        raise FileNotFoundError(f"OpenDDE produced no CIF predictions under {raw_dir}")
    prediction_dir = out_dir / "predictions"
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    job = load_job(args.json)
    validate_msa_paths(job, require_msa=args.require_msa)
    if args.require_msa and not args.use_msa:
        raise ValueError("--require-msa conflicts with --use-msa false")
    run_opendde(args)
    rows = normalize_predictions(args.out_dir)
    print(f"wrote {len(rows)} OpenDDE PDB prediction(s) to {args.out_dir / 'predictions'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
