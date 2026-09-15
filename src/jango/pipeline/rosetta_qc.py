"""Post-Rosetta QC checkpoint for native and decoy outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from nbia.rosetta_qc import write_rosetta_qc_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosetta-qc",
        description="Validate existing native and decoy Rosetta outputs without rerunning Rosetta.",
    )
    parser.add_argument("--native-relax-manifest", required=True, type=Path)
    parser.add_argument("--native-interface", required=True, type=Path)
    parser.add_argument("--decoy-relax-manifest", required=True, type=Path)
    parser.add_argument("--decoy-interface", required=True, type=Path)
    parser.add_argument("--native-work-dir", type=Path, default=None)
    parser.add_argument("--slurm-submission-manifest", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--native-out", required=True, type=Path)
    parser.add_argument("--decoy-out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    native, decoy = write_rosetta_qc_tables(
        native_relax_manifest=args.native_relax_manifest,
        native_interface_table=args.native_interface,
        decoy_relax_manifest=args.decoy_relax_manifest,
        decoy_interface_table=args.decoy_interface,
        native_work_dir=args.native_work_dir,
        slurm_submission_manifest=args.slurm_submission_manifest,
        source_root=args.source_root,
        native_out=args.native_out,
        decoy_out=args.decoy_out,
    )
    print(f"wrote native Rosetta QC: {len(native)} rows to {args.native_out}")
    print(f"native QC statuses: {native['qc_status'].value_counts(dropna=False).to_dict() if 'qc_status' in native.columns else {}}")
    print(f"wrote decoy Rosetta QC: {len(decoy)} rows to {args.decoy_out}")
    print(f"decoy QC statuses: {decoy['qc_status'].value_counts(dropna=False).to_dict() if 'qc_status' in decoy.columns else {}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
