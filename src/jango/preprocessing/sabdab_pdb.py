#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd

ATLAS_TARBALL_RE = re.compile(
    r"^(?P<pdb_id>[0-9A-Za-z]{4})_(?P<nanobody_chain>.)_(?P<antigen_chain>.)\.pdb\.tar\.gz$"
)


def read_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "pdb": "pdb_id",
            "Hchain": "nanobody_chain",
        }
    ).copy()

    required = {
        "pdb_id",
        "nanobody_chain",
        "antigen_chain",
        "antigen_type",
        "resolution",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    return df


def normalize_chain_id(value: object) -> tuple[str | None, str]:
    if pd.isna(value):
        return None, "missing_chain"

    chain = str(value).strip()

    if not chain:
        return None, "missing_chain"

    if len(chain) == 1:
        return chain, ""

    if "|" in chain or "," in chain or ";" in chain:
        return None, f"multi_chain_rejected:{chain}"

    if len(set(chain)) == 1:
        return chain[0], f"collapsed_repeated_chain:{chain}->{chain[0]}"

    return None, f"multi_char_chain_rejected:{chain}"


def filter_and_normalize_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []

    for index, row in df.iterrows():
        warnings = []

        pdb_id = str(row["pdb_id"]).lower().strip() if pd.notna(row["pdb_id"]) else ""
        antigen_type = str(row["antigen_type"]).lower().strip() if pd.notna(row["antigen_type"]) else ""

        nb_chain, nb_warning = normalize_chain_id(row["nanobody_chain"])
        ag_chain, ag_warning = normalize_chain_id(row["antigen_chain"])

        if nb_warning:
            warnings.append(f"nanobody_{nb_warning}")
        if ag_warning:
            warnings.append(f"antigen_{ag_warning}")

        status = "candidate"

        if not pdb_id or len(pdb_id) != 4:
            status = "rejected"
            warnings.append("invalid_pdb_id")

        if antigen_type != "protein":
            status = "rejected"
            warnings.append("non_protein_antigen")

        if pd.isna(row["resolution"]):
            status = "rejected"
            warnings.append("missing_resolution")

        if nb_chain is None:
            status = "rejected"
        if ag_chain is None:
            status = "rejected"

        record = row.to_dict()
        record.update(
            {
                "metadata_row": index,
                "pdb_id": pdb_id,
                "nanobody_chain": nb_chain,
                "antigen_chain": ag_chain,
                "metadata_status": status,
                "metadata_warnings": ";".join(warnings),
            }
        )

        if status == "candidate":
            record["structure_id"] = f"{pdb_id}_{nb_chain}_{ag_chain}"
        else:
            record["structure_id"] = ""

        records.append(record)

    normalized = pd.DataFrame(records)

    rejected = normalized[normalized["metadata_status"] == "rejected"].copy()
    candidates = normalized[normalized["metadata_status"] == "candidate"].copy()

    candidates["duplicate_rank"] = candidates.groupby("structure_id").cumcount()
    duplicate_skipped = candidates[candidates["duplicate_rank"] > 0].copy()
    duplicate_skipped["metadata_status"] = "duplicate_skipped"
    duplicate_skipped["metadata_warnings"] = duplicate_skipped["metadata_warnings"].fillna("")
    duplicate_skipped["metadata_warnings"] = (
        duplicate_skipped["metadata_warnings"] + ";duplicate_structure_id"
    ).str.strip(";")

    candidates = candidates[candidates["duplicate_rank"] == 0].copy()

    metadata_qc = pd.concat([rejected, duplicate_skipped], ignore_index=True)

    return candidates, metadata_qc


def pdb_chain_from_line(line: str) -> str:
    return line[21].strip() if len(line) > 21 else ""


def scan_pdb_chains(pdb_path: Path) -> tuple[set[str], int]:
    chains = set()
    atom_count = 0

    with pdb_path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom_count += 1
                chain = pdb_chain_from_line(line)
                if chain:
                    chains.add(chain)

    return chains, atom_count


def write_two_chain_pdb(
    source_pdb: Path,
    output_pdb: Path,
    keep_chains: set[str],
) -> tuple[set[str], set[str], int, int]:
    kept_chains = set()
    removed_chains = set()
    atoms_written = 0
    atoms_removed = 0
    wrote_end = False

    with source_pdb.open("r", errors="replace") as fin, output_pdb.open("w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                chain = pdb_chain_from_line(line)

                if chain in keep_chains:
                    fout.write(line)
                    kept_chains.add(chain)
                    atoms_written += 1
                else:
                    if chain:
                        removed_chains.add(chain)
                    atoms_removed += 1

            elif line.startswith("TER"):
                chain = pdb_chain_from_line(line)
                if chain in keep_chains:
                    fout.write(line)

            elif line.startswith("END"):
                fout.write("END\n")
                wrote_end = True

        if not wrote_end:
            fout.write("END\n")

    return kept_chains, removed_chains, atoms_written, atoms_removed


def find_raw_pdb(raw_pdb_dir: Path, pdb_id: str) -> Path | None:
    candidates = [
        raw_pdb_dir / f"{pdb_id}.pdb",
        raw_pdb_dir / f"{pdb_id.upper()}.pdb",
        raw_pdb_dir / f"pdb{pdb_id}.ent",
        raw_pdb_dir / f"pdb{pdb_id.lower()}.ent",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def make_tarball(pdb_file: Path, tarball_path: Path, arcname: str) -> None:
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(pdb_file, arcname=arcname)


def joined(values: Iterable[str]) -> str:
    return "".join(sorted(v for v in values if v))


def prepare_tarballs(
    metadata: pd.DataFrame,
    raw_pdb_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    qc_records = []

    for _, row in metadata.iterrows():
        pdb_id = row["pdb_id"]
        nb_chain = row["nanobody_chain"]
        ag_chain = row["antigen_chain"]
        structure_id = row["structure_id"]

        output_name = f"{structure_id}.pdb.tar.gz"
        output_tar = output_dir / output_name

        warnings = []
        if row.get("metadata_warnings"):
            warnings.extend(str(row["metadata_warnings"]).split(";"))

        if not ATLAS_TARBALL_RE.match(output_name):
            warnings.append("invalid_atlas_filename")

        qc_record = {
            "structure_id": structure_id,
            "pdb_id": pdb_id,
            "nanobody_chain": nb_chain,
            "antigen_chain": ag_chain,
            "output_tar": str(output_tar),
            "status": "",
            "warnings": "",
        }

        source_pdb = find_raw_pdb(raw_pdb_dir, pdb_id)

        if source_pdb is None:
            qc_record.update(
                {
                    "status": "missing_source_pdb",
                    "warnings": ";".join(warnings + ["missing_pdb"]),
                }
            )
            qc_records.append(qc_record)
            continue

        qc_record["source_pdb"] = str(source_pdb)

        try:
            original_chains, atoms_before = scan_pdb_chains(source_pdb)
            keep_chains = {nb_chain, ag_chain}

            if nb_chain not in original_chains:
                warnings.append(f"nanobody_chain_not_in_raw_pdb:{nb_chain}")
            if ag_chain not in original_chains:
                warnings.append(f"antigen_chain_not_in_raw_pdb:{ag_chain}")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_pdb = Path(tmpdir) / f"{structure_id}.pdb"

                kept_chains, removed_chains, atoms_written, atoms_removed = write_two_chain_pdb(
                    source_pdb=source_pdb,
                    output_pdb=tmp_pdb,
                    keep_chains=keep_chains,
                )

                if nb_chain not in kept_chains:
                    warnings.append(f"nanobody_chain_missing_after_write:{nb_chain}")

                if ag_chain not in kept_chains:
                    warnings.append(f"antigen_chain_missing_after_write:{ag_chain}")

                if atoms_written == 0:
                    warnings.append("zero_atoms_written")

                if len(kept_chains) != 2:
                    warnings.append(f"kept_chain_count:{len(kept_chains)}")

                if atoms_written > 0 and nb_chain in kept_chains and ag_chain in kept_chains:
                    make_tarball(
                        pdb_file=tmp_pdb,
                        tarball_path=output_tar,
                        arcname=f"{structure_id}.pdb",
                    )

            valid_created = atoms_written > 0 and nb_chain in kept_chains and ag_chain in kept_chains

            if valid_created:
                status = "created_with_warnings" if warnings else "created"
            else:
                status = "failed_validation"

            qc_record.update(
                {
                    "status": status,
                    "warnings": ";".join(warnings),
                    "original_chains": joined(original_chains),
                    "kept_chains": joined(kept_chains),
                    "removed_chains": joined(removed_chains),
                    "atoms_before": atoms_before,
                    "atoms_written": atoms_written,
                    "atoms_removed": atoms_removed,
                }
            )

            if valid_created:
                manifest_records.append(
                    {
                        "structure_id": structure_id,
                        "pdb_id": pdb_id,
                        "nanobody_chain": nb_chain,
                        "antigen_chain": ag_chain,
                        "source_filename": output_name,
                        "structure_path": str(output_tar),
                        "source": "sabdab",
                        "label": 1,
                        "label_type": "positive_true_complex",
                        "label_confidence": "high",
                        "antigen_type": row.get("antigen_type", ""),
                        "resolution": row.get("resolution", ""),
                        "method": row.get("method", ""),
                        "affinity": row.get("affinity", ""),
                        "affinity_method": row.get("affinity_method", ""),
                    }
                )

        except Exception as exc:
            qc_record.update(
                {
                    "status": "failed",
                    "warnings": ";".join(warnings + [f"exception:{exc}"]),
                }
            )

        qc_records.append(qc_record)

    return pd.DataFrame(manifest_records), pd.DataFrame(qc_records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SAbDab raw PDBs into Atlas-format two-chain tarballs."
    )
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--raw-pdb-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--qc-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)

    raw = read_metadata(args.metadata)
    raw = normalize_columns(raw)

    candidates, metadata_qc = filter_and_normalize_metadata(raw)

    manifest, pdb_qc = prepare_tarballs(
        metadata=candidates,
        raw_pdb_dir=args.raw_pdb_dir,
        output_dir=args.output_dir,
    )

    metadata_qc_out = metadata_qc[
        [
            "metadata_row",
            "pdb_id",
            "nanobody_chain",
            "antigen_chain",
            "structure_id",
            "metadata_status",
            "metadata_warnings",
        ]
    ].rename(
        columns={
            "metadata_status": "status",
            "metadata_warnings": "warnings",
        }
    )

    full_qc = pd.concat([metadata_qc_out, pdb_qc], ignore_index=True, sort=False)

    manifest.to_csv(args.manifest_out, index=False)
    full_qc.to_csv(args.qc_out, index=False)

    print(f"Input rows: {len(raw)}")
    print(f"Candidate rows after filtering/normalization: {len(candidates)}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"QC rows: {len(full_qc)}")
    print(f"Manifest: {args.manifest_out}")
    print(f"QC report: {args.qc_out}")
    print(f"Tarballs: {args.output_dir}")


if __name__ == "__main__":
    main()