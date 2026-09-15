from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pdbio import chain_residues, iter_raw_structures, parse_atoms, read_pdb_text, sha256_file


def build_manifest(raw_dir: Path, manifest_csv: Path, checksums_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    checksums: list[str] = []
    for record in iter_raw_structures(raw_dir):
        checksum = sha256_file(record.source_path)
        pdb_text = read_pdb_text(record.source_path)
        atoms = parse_atoms(pdb_text)
        nb_res = chain_residues(atoms, record.nanobody_chain)
        ag_res = chain_residues(atoms, record.antigen_chain)
        chains = sorted({atom.chain_id for atom in atoms})
        warnings: list[str] = []
        if len(chains) != 2:
            warnings.append(f"chain_count={len(chains)}")
        if not nb_res:
            warnings.append("missing_nanobody_chain")
        if not ag_res:
            warnings.append("missing_antigen_chain")
        rows.append(
            {
                "structure_id": record.structure_id,
                "pdb_id": record.pdb_id,
                "nanobody_chain": record.nanobody_chain,
                "antigen_chain": record.antigen_chain,
                "source_filename": record.source_filename,
                "sha256": checksum,
                "chain_count": len(chains),
                "chains": "".join(chains),
                "nanobody_length": len(nb_res),
                "antigen_length": len(ag_res),
                "antigen_class": "peptide" if len(ag_res) <= 30 else "protein",
                "validation_warnings": ";".join(warnings),
            }
        )
        checksums.append(f"{checksum}  {record.source_filename}")
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("structure_id")
    df.to_csv(manifest_csv, index=False)
    checksums_path.write_text("\n".join(sorted(checksums)) + "\n")
    return df

