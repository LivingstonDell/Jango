#!/usr/bin/env python3
"""
01b_prepare_sabdab2_cif_tarballs.py

Prepare SAbDab2 / abag_split_sd-style mmCIF structures into Atlas-compatible
two-chain .pdb.tar.gz files.

Input:
  --metadata abag_split_sd.csv or filtered SAbDab2 manifest
  --cif-dir directory containing <INSTANCE>.cif files

Output:
  --output-dir Atlas-format tarballs:
      <pdbid>_<nanobody_chain>_<antigen_chain>.pdb.tar.gz

  --manifest-out successful tarball manifest
  --qc-out full QC report

This script is intentionally separate from the raw-PDB converter because mmCIF
files can contain multi-character chain IDs such as DDD or AAA. Atlas requires
single-character chain IDs in both the filename and the PDB content.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.Polypeptide import is_aa


ATLAS_NANOBODY_CHAINS = [
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]

ATLAS_ANTIGEN_CHAINS = [
    "A", "B", "C", "D", "E", "F", "G",
]


class StandardAASelect(Select):
    def accept_residue(self, residue) -> bool:
        return bool(is_aa(residue, standard=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SAbDab2 CIF files into Atlas-format two-chain tarballs."
    )
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cif-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--qc-out", required=True, type=Path)
    parser.add_argument("--min-antigen-len", type=int, default=60)
    parser.add_argument("--max-antigen-len", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--clean-output-dir",
        action="store_true",
        help="Delete existing *.pdb.tar.gz files in output-dir before writing.",
    )
    return parser.parse_args(argv)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}


def split_slash(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return text.split("/")


def safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_pdb_id(value: object) -> str:
    pdb_id = safe_text(value).replace("pdb_", "").replace("PDB_", "")
    while len(pdb_id) > 4 and pdb_id.startswith("0"):
        pdb_id = pdb_id[1:]
    return pdb_id.lower()


def sanitize_name(value: object) -> str:
    text = safe_text(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)


def validate_required_columns(df: pd.DataFrame) -> None:
    required = {
        "INSTANCE",
        "PDB_ID",
        "type",
        "holo",
        "Hchain",
        "agchains",
        "agtypes",
        "agresolvedseqs",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")


def iter_candidate_antigens(
    row: pd.Series,
    min_antigen_len: int,
    max_antigen_len: int,
) -> Iterable[dict[str, object]]:
    agchains = split_slash(row.get("agchains", ""))
    agtypes = split_slash(row.get("agtypes", ""))
    agseqs = split_slash(row.get("agresolvedseqs", ""))

    n = max(len(agchains), len(agtypes), len(agseqs))

    for i in range(n):
        chain_id = safe_text(agchains[i] if i < len(agchains) else "")
        ag_type = safe_text(agtypes[i] if i < len(agtypes) else "").upper()
        seq = safe_text(agseqs[i] if i < len(agseqs) else "")

        if not chain_id:
            continue

        if ag_type != "PROTEIN":
            continue

        if not (min_antigen_len <= len(seq) <= max_antigen_len):
            continue

        yield {
            "antigen_chain_original": chain_id,
            "antigen_type": ag_type,
            "antigen_sequence": seq,
            "antigen_resolved_length": len(seq),
            "antigen_index": i,
        }


def filter_sabdab2_rows(
    df: pd.DataFrame,
    min_antigen_len: int,
    max_antigen_len: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    qc_records = []

    for idx, row in df.iterrows():
        warnings = []

        instance = safe_text(row.get("INSTANCE", ""))
        pdb_id = normalize_pdb_id(row.get("PDB_ID", ""))
        ab_type = safe_text(row.get("type", ""))
        holo = parse_bool(row.get("holo", False))
        heavy_chain_original = safe_text(row.get("Hchain", ""))

        if not instance:
            warnings.append("missing_instance")
        if len(pdb_id) != 4:
            warnings.append("invalid_pdb_id")
        if ab_type != "SD-H":
            warnings.append("not_sd_h")
        if not holo:
            warnings.append("not_holo")
        if not heavy_chain_original:
            warnings.append("missing_hchain")

        antigen_candidates = list(
            iter_candidate_antigens(row, min_antigen_len, max_antigen_len)
        )

        if not antigen_candidates:
            warnings.append("no_protein_antigen_in_length_range")

        if warnings:
            qc_records.append(
                {
                    "metadata_row": idx,
                    "source_instance": instance,
                    "pdb_id": pdb_id,
                    "status": "filtered_out",
                    "warnings": ";".join(warnings),
                }
            )
            continue

        for antigen in antigen_candidates:
            record = row.to_dict()
            record.update(
                {
                    "metadata_row": idx,
                    "source_instance": instance,
                    "pdb_id_normalized": pdb_id,
                    "hchain_original": heavy_chain_original,
                    **antigen,
                }
            )
            records.append(record)

    return pd.DataFrame(records), pd.DataFrame(qc_records)


def get_first_model(structure: Structure) -> Model:
    for model in structure:
        return model
    raise ValueError("structure has no models")


def find_chain(model: Model, chain_id: str):
    for chain in model:
        if chain.id == chain_id:
            return chain
    return None


def copy_chain_with_new_id(chain, new_id: str) -> Chain:
    new_chain = copy.deepcopy(chain)
    new_chain.id = new_id
    return new_chain


def make_two_chain_structure(
    structure_id: str,
    heavy_chain,
    antigen_chain,
    output_nb_chain: str,
    output_ag_chain: str,
) -> Structure:
    structure = Structure(sanitize_name(structure_id))
    model = Model(0)
    model.add(copy_chain_with_new_id(heavy_chain, output_nb_chain))
    model.add(copy_chain_with_new_id(antigen_chain, output_ag_chain))
    structure.add(model)
    return structure


def write_pdb_string(structure: Structure) -> str:
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    handle = io.StringIO()
    pdb_io.save(handle, select=StandardAASelect())
    text = handle.getvalue()

    if not text.endswith("END\n"):
        text += "END\n"

    return text


def write_tarball(tar_path: Path, pdb_name: str, pdb_text: str) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    data = pdb_text.encode("utf-8")

    info = tarfile.TarInfo(name=pdb_name)
    info.size = len(data)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))


def choose_output_chains(pair_index: int) -> tuple[str, str] | None:
    total = len(ATLAS_NANOBODY_CHAINS) * len(ATLAS_ANTIGEN_CHAINS)
    if pair_index >= total:
        return None

    nb_idx = pair_index // len(ATLAS_ANTIGEN_CHAINS)
    ag_idx = pair_index % len(ATLAS_ANTIGEN_CHAINS)

    return ATLAS_NANOBODY_CHAINS[nb_idx], ATLAS_ANTIGEN_CHAINS[ag_idx]


def convert_cifs_to_tarballs(
    candidates: pd.DataFrame,
    metadata_qc: pd.DataFrame,
    cif_dir: Path,
    output_dir: Path,
    manifest_out: Path,
    qc_out: Path,
    limit: int | None,
    clean_output_dir: bool,
) -> None:
    if clean_output_dir and output_dir.exists():
        for path in output_dir.glob("*.pdb.tar.gz"):
            path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    qc_out.parent.mkdir(parents=True, exist_ok=True)

    parser = MMCIFParser(QUIET=True)

    manifest_records = []
    qc_records = metadata_qc.to_dict("records")

    pdb_id_counts: dict[str, int] = defaultdict(int)
    seen_filenames: set[str] = set()

    written = 0

    for _, row in candidates.iterrows():
        source_instance = safe_text(row["source_instance"])
        pdb_id = safe_text(row["pdb_id_normalized"])
        original_hchain = safe_text(row["hchain_original"])
        original_agchain = safe_text(row["antigen_chain_original"])

        cif_path = cif_dir / f"{source_instance}.cif"

        base_qc = {
            "metadata_row": row.get("metadata_row", ""),
            "source_instance": source_instance,
            "pdb_id": pdb_id,
            "hchain_original": original_hchain,
            "agchain_original": original_agchain,
            "source_cif": str(cif_path),
        }

        if not cif_path.exists():
            qc_records.append(
                {
                    **base_qc,
                    "status": "missing_cif",
                    "warnings": "missing_cif_file",
                }
            )
            continue

        pair_index = pdb_id_counts[pdb_id]
        output_pair = choose_output_chains(pair_index)

        if output_pair is None:
            qc_records.append(
                {
                    **base_qc,
                    "status": "too_many_duplicate_pdb_entries",
                    "warnings": f"duplicate_index:{pair_index}",
                }
            )
            continue

        output_nb_chain, output_ag_chain = output_pair
        pdb_id_counts[pdb_id] += 1

        structure_id = f"{pdb_id}_{output_nb_chain}_{output_ag_chain}"
        pdb_name = f"{structure_id}.pdb"
        tar_name = f"{structure_id}.pdb.tar.gz"
        tar_path = output_dir / tar_name

        if tar_name in seen_filenames or tar_path.exists():
            qc_records.append(
                {
                    **base_qc,
                    "structure_id": structure_id,
                    "status": "filename_collision",
                    "warnings": "filename_already_seen_or_exists",
                    "output_tar": str(tar_path),
                }
            )
            continue

        try:
            structure = parser.get_structure(source_instance, str(cif_path))
            model = get_first_model(structure)

            heavy_chain = find_chain(model, original_hchain)
            antigen_chain = find_chain(model, original_agchain)

            if heavy_chain is None:
                qc_records.append(
                    {
                        **base_qc,
                        "structure_id": structure_id,
                        "status": "missing_heavy_chain",
                        "warnings": f"hchain_not_found:{original_hchain}",
                    }
                )
                continue

            if antigen_chain is None:
                qc_records.append(
                    {
                        **base_qc,
                        "structure_id": structure_id,
                        "status": "missing_antigen_chain",
                        "warnings": f"agchain_not_found:{original_agchain}",
                    }
                )
                continue

            two_chain_structure = make_two_chain_structure(
                structure_id=structure_id,
                heavy_chain=heavy_chain,
                antigen_chain=antigen_chain,
                output_nb_chain=output_nb_chain,
                output_ag_chain=output_ag_chain,
            )

            pdb_text = write_pdb_string(two_chain_structure)

            if "ATOM" not in pdb_text:
                qc_records.append(
                    {
                        **base_qc,
                        "structure_id": structure_id,
                        "status": "empty_pdb_after_filter",
                        "warnings": "no_standard_aa_atoms_written",
                    }
                )
                continue

            write_tarball(tar_path, pdb_name, pdb_text)
            seen_filenames.add(tar_name)

            manifest_records.append(
                {
                    "structure_id": structure_id,
                    "pdb_id": pdb_id,
                    "nanobody_chain": output_nb_chain,
                    "antigen_chain": output_ag_chain,
                    "source_filename": tar_name,
                    "structure_path": str(tar_path),
                    "source": "sabdab2",
                    "label": 1,
                    "label_type": "positive_true_complex",
                    "label_confidence": "high",
                    "source_instance": source_instance,
                    "pdb_id_original": row.get("PDB_ID", ""),
                    "sabdab_id": row.get("SABDAB_ID", ""),
                    "ab_ag_split": row.get("ab_ag_split", ""),
                    "ab_ag_cluster": row.get("ab_ag_cluster", ""),
                    "hchain_original": original_hchain,
                    "agchain_original": original_agchain,
                    "output_nanobody_chain": output_nb_chain,
                    "output_antigen_chain": output_ag_chain,
                    "duplicate_index_for_pdb_id": pair_index,
                    "antigen_type": row.get("antigen_type", "PROTEIN"),
                    "antigen_resolved_length": row.get("antigen_resolved_length", ""),
                    "nanobody_sequence": row.get("VH_numerable_seq", row.get("Hseq", "")),
                    "antigen_sequence": row.get("antigen_sequence", ""),
                    "pdb_inside_tar": pdb_name,
                    "source_cif": str(cif_path),
                }
            )

            qc_records.append(
                {
                    **base_qc,
                    "structure_id": structure_id,
                    "status": "created",
                    "warnings": "",
                    "output_tar": str(tar_path),
                    "pdb_inside_tar": pdb_name,
                    "output_nanobody_chain": output_nb_chain,
                    "output_antigen_chain": output_ag_chain,
                }
            )

            written += 1

            if limit is not None and written >= limit:
                break

        except Exception as exc:
            qc_records.append(
                {
                    **base_qc,
                    "structure_id": structure_id,
                    "status": "failed",
                    "warnings": f"exception:{repr(exc)}",
                    "output_tar": str(tar_path),
                }
            )

    pd.DataFrame(manifest_records).to_csv(
        manifest_out,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    pd.DataFrame(qc_records).to_csv(
        qc_out,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    print(f"Candidate rows: {len(candidates)}")
    print(f"Manifest rows: {len(manifest_records)}")
    print(f"QC rows: {len(qc_records)}")
    print(f"Tarballs written: {written}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_out}")
    print(f"QC report: {qc_out}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    raw = read_table(args.metadata)
    validate_required_columns(raw)

    candidates, metadata_qc = filter_sabdab2_rows(
        raw,
        min_antigen_len=args.min_antigen_len,
        max_antigen_len=args.max_antigen_len,
    )

    convert_cifs_to_tarballs(
        candidates=candidates,
        metadata_qc=metadata_qc,
        cif_dir=args.cif_dir,
        output_dir=args.output_dir,
        manifest_out=args.manifest_out,
        qc_out=args.qc_out,
        limit=args.limit,
        clean_output_dir=args.clean_output_dir,
    )


if __name__ == "__main__":
    main()