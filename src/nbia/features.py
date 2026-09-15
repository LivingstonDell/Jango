from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .cdr import annotate_cdrs
from .pdbio import Atom, chain_residues, chain_sequence, iter_raw_structures, parse_atoms, read_pdb_text
from .residues import AA3_TO_1, AROMATIC, BACKBONE_ATOMS, CHARGED_NEG, CHARGED_POS, HBOND_ELEMENTS, HYDROPHOBIC, POLAR


def compute_native_features(raw_dir: Path, out_csv: Path, contact_cutoff: float = 5.0) -> pd.DataFrame:
    rows = []
    for record in iter_raw_structures(raw_dir):
        atoms = parse_atoms(read_pdb_text(record.source_path))
        rows.append(compute_interface_features(record.structure_id, record.pdb_id, record.nanobody_chain, record.antigen_chain, atoms, "native", contact_cutoff))
    df = pd.DataFrame(rows).sort_values("structure_id")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def compute_interface_features(
    structure_id: str,
    pdb_id: str,
    nanobody_chain: str,
    antigen_chain: str,
    atoms: list[Atom],
    state: str,
    contact_cutoff: float = 5.0,
) -> dict[str, object]:
    nb_atoms = [a for a in atoms if a.chain_id == nanobody_chain and a.is_heavy]
    ag_atoms = [a for a in atoms if a.chain_id == antigen_chain and a.is_heavy]
    nb_sequence = chain_sequence(atoms, nanobody_chain)
    cdr = annotate_cdrs(nb_sequence)
    nb_res_order = {(num, icode): i + 1 for i, (num, icode, _) in enumerate(chain_residues(atoms, nanobody_chain))}
    contacts = atom_contacts(nb_atoms, ag_atoms, contact_cutoff)
    nb_interface = {nb_atoms[i].residue_key for i, _, _ in contacts}
    ag_interface = {ag_atoms[j].residue_key for _, j, _ in contacts}
    residue_pairs = {(nb_atoms[i].residue_key, ag_atoms[j].residue_key) for i, j, _ in contacts}
    dists = np.array([dist for _, _, dist in contacts], dtype=float)
    nb_regions = []
    for residue_key in nb_interface:
        _, number, icode, _ = residue_key
        seq_idx = nb_res_order.get((number, icode))
        nb_regions.append(cdr.by_index.get(seq_idx, "unmapped") if seq_idx else "unmapped")
    region_counts = Counter(nb_regions)
    chemistry = classify_contacts(nb_atoms, ag_atoms, contacts)
    nb_len = len(chain_residues(atoms, nanobody_chain))
    ag_len = len(chain_residues(atoms, antigen_chain))
    total_contact_atoms = len(contacts)
    return {
        "structure_id": structure_id,
        "pdb_id": pdb_id,
        "state": state,
        "nanobody_chain": nanobody_chain,
        "antigen_chain": antigen_chain,
        "nanobody_length": nb_len,
        "antigen_length": ag_len,
        "antigen_class": "peptide" if ag_len <= 30 else "protein",
        "atom_contacts_5A": total_contact_atoms,
        "residue_contact_pairs_5A": len(residue_pairs),
        "nanobody_interface_residues": len(nb_interface),
        "antigen_interface_residues": len(ag_interface),
        #added interface position and sequence
        "nanobody_interface_positions": residue_positions(nb_interface),
        "nanobody_interface_sequence": residue_sequence(nb_interface),
        "antigen_interface_positions": residue_positions(ag_interface),
        "antigen_interface_sequence": residue_sequence(ag_interface),
        "contact_residue_pairs": contact_pair_labels(residue_pairs),
        #resume
        "contact_density_per_nb_interface_residue": safe_divide(len(residue_pairs), len(nb_interface)),
        "min_contact_distance": float(dists.min()) if len(dists) else np.nan,
        "median_contact_distance": float(np.median(dists)) if len(dists) else np.nan,
        "p10_contact_distance": float(np.percentile(dists, 10)) if len(dists) else np.nan,
        "backbone_atom_contact_fraction": safe_divide(chemistry["backbone_atom_contacts"], total_contact_atoms),
        "polar_contact_fraction": safe_divide(chemistry["polar_contacts"], total_contact_atoms),
        "hydrophobic_contact_fraction": safe_divide(chemistry["hydrophobic_contacts"], total_contact_atoms),
        "charged_contact_fraction": safe_divide(chemistry["charged_contacts"], total_contact_atoms),
        "aromatic_contacts": chemistry["aromatic_contacts"],
        "salt_bridges": chemistry["salt_bridges"],
        "hbond_like_contacts": chemistry["hbond_like_contacts"],
        "cdr_annotation_method": cdr.method,
        "cdr_annotation_status": cdr.status,
        "cdr1_interface_residues": region_counts["CDR1"],
        "cdr2_interface_residues": region_counts["CDR2"],
        "cdr3_interface_residues": region_counts["CDR3"],
        "framework_interface_residues": region_counts["FR"],
        "unmapped_interface_residues": region_counts["unmapped"],
        "cdr_interface_fraction": safe_divide(region_counts["CDR1"] + region_counts["CDR2"] + region_counts["CDR3"], len(nb_interface)),
        "cdr3_interface_fraction": safe_divide(region_counts["CDR3"], len(nb_interface)),
    }


def atom_contacts(nb_atoms: list[Atom], ag_atoms: list[Atom], cutoff: float) -> list[tuple[int, int, float]]:
    if not nb_atoms or not ag_atoms:
        return []
    nb_xyz = np.array([[a.x, a.y, a.z] for a in nb_atoms], dtype=float)
    ag_xyz = np.array([[a.x, a.y, a.z] for a in ag_atoms], dtype=float)
    tree = cKDTree(ag_xyz)
    contacts: list[tuple[int, int, float]] = []
    for i, point in enumerate(nb_xyz):
        for j in tree.query_ball_point(point, cutoff):
            dist = float(np.linalg.norm(point - ag_xyz[j]))
            contacts.append((i, int(j), dist))
    return contacts


def classify_contacts(nb_atoms: list[Atom], ag_atoms: list[Atom], contacts: list[tuple[int, int, float]]) -> Counter:
    counts: Counter = Counter()
    for i, j, dist in contacts:
        a = nb_atoms[i]
        b = ag_atoms[j]
        if a.atom_name in BACKBONE_ATOMS or b.atom_name in BACKBONE_ATOMS:
            counts["backbone_atom_contacts"] += 1
        if a.residue_name in POLAR or b.residue_name in POLAR:
            counts["polar_contacts"] += 1
        if a.residue_name in HYDROPHOBIC and b.residue_name in HYDROPHOBIC:
            counts["hydrophobic_contacts"] += 1
        if a.residue_name in (CHARGED_POS | CHARGED_NEG) or b.residue_name in (CHARGED_POS | CHARGED_NEG):
            counts["charged_contacts"] += 1
        if dist <= 5.0 and a.residue_name in AROMATIC and b.residue_name in AROMATIC:
            counts["aromatic_contacts"] += 1
        if dist <= 4.0 and ((a.residue_name in CHARGED_POS and b.residue_name in CHARGED_NEG) or (a.residue_name in CHARGED_NEG and b.residue_name in CHARGED_POS)):
            counts["salt_bridges"] += 1
        if dist <= 3.5 and a.element in HBOND_ELEMENTS and b.element in HBOND_ELEMENTS:
            counts["hbond_like_contacts"] += 1
    return counts

# added helper functions
def residue_sort_key(residue_key):
    chain, number, icode, resname = residue_key
    return (chain, number, icode)


def residue_position_label(residue_key) -> str:
    chain, number, icode, resname = residue_key
    ins = str(icode).strip()
    pos = f"{number}{ins}" if ins else str(number)
    return f"{chain}:{pos}:{resname}"


def residue_positions(residue_keys) -> str:
    ordered = sorted(residue_keys, key=residue_sort_key)
    return ";".join(residue_position_label(r) for r in ordered)


def residue_sequence(residue_keys) -> str:
    ordered = sorted(residue_keys, key=residue_sort_key)
    return "".join(AA3_TO_1.get(r[3], "X") for r in ordered)


def contact_pair_labels(residue_pairs) -> str:
    ordered = sorted(
        residue_pairs,
        key=lambda pair: (residue_sort_key(pair[0]), residue_sort_key(pair[1])),
    )
    return ";".join(
        f"{residue_position_label(nb)}--{residue_position_label(ag)}"
        for nb, ag in ordered
    )

#resume
def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")

