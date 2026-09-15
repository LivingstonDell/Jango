from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .boltz2 import BOLTZ_CHAIN_ANTIGEN, BOLTZ_CHAIN_NANOBODY, discover_boltz_predictions, parse_confidence_json, remap_atoms, rmsd, superpose
from .features import atom_contacts
from .pdbio import Atom, chain_residues, chain_sequence, parse_atoms, read_pdb_text
from .plots import set_publication_style
from .residues import AA3_TO_1
from .rosetta import interface_analyzer_docker_command, parse_scorefile


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

@dataclass(frozen=True)
class DecoyDefaults:
    candidates_per_antigen: int = 150
    # With forced contact mutation every candidate passes the sequence gate, so the
    # ProteinMPNN budget (150) is not the refold budget. Refold only the top-K by
    # ProteinMPNN score (most native-likely => most likely to fold) to find the few
    # passing decoys without spending GPU on all 150.
    refold_candidates_per_antigen: int = 24
    passing_designs_per_antigen: int = 5
    # Fraction of (burial-ranked) interface contacts whose native residue is forbidden.
    # 1.0 = force all contacts (best for interface-small antigens); lower it for
    # interface-dominated antigens where forcing every contact destroys the fold.
    forced_contact_fraction: float = 1.0
    # When contacts exceed this fraction of antigen residues, the interface dominates
    # the sequence and forcing all of them breaks the fold (calibration: 8qf5, 39%
    # contacts, 0/8 fold at full force vs 0.819->0.868 self-consistency at 0.5). Such
    # antigens force only the top interface_dominated_fraction of burial-ranked contacts.
    interface_dominated_threshold: float = 0.25
    interface_dominated_fraction: float = 0.5
    # Contact mutation is forced via ProteinMPNN per-position omit, so temperature is
    # tuned for foldability (low) rather than to drive identity down.
    sampling_temp: float = 0.3
    # Reported only; identity no longer gates (see sequence_filter_status).
    max_sequence_identity: float = 0.30
    min_contact_mutation_fraction: float = 0.90
    # Fold validation gates on TM-score (robust to flexible termini); CA-RMSD is
    # still reported but no longer gates pass/fail.
    monomer_tm_score_min: float = 0.90
    monomer_ca_rmsd_max: float = 2.5
    monomer_aligned_fraction_min: float = 0.90


def write_default_decoy_config(path: Path = Path("configs/antigen_redesign_decoys.yml"), defaults: DecoyDefaults = DecoyDefaults()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "case_manifest: results/tables/boltz2_case_manifest.csv",
                "candidates_per_antigen: 150",
                "refold_candidates_per_antigen: 24",
                "passing_designs_per_antigen: 5",
                "sampling_temp: 0.3",
                "force_contact_mutation: true  # ProteinMPNN omit_AA_jsonl forbids native AA at contacts",
                "max_sequence_identity: 0.30  # reported only, not gated",
                "min_contact_mutation_fraction: 0.90",
                "fold_validation: boltz2_monomer_sequence_only",
                "monomer_tm_score_min: 0.90",
                "monomer_ca_rmsd_max: 2.5",
                "monomer_aligned_fraction_min: 0.90",
                "boltz2:",
                "  model: boltz2",
                "  use_msa_server: true",
                "  use_potentials: true",
                "  diffusion_samples: 5",
                "  recycling_steps: 3",
                "  output_format: pdb",
                "  write_full_pae: true",
                "  write_full_pde: true",
                "proteinmpnn:",
                "  candidates_per_antigen: 150",
                "  sampling_temp: 0.3",
                "  omit_native_aa_at_contacts: true",
                "  designed_chain: A",
                "notes: ProteinMPNN is installed externally on <CPU_NODE>; the repo only creates reproducible job bundles.",
                "",
            ]
        )
    )
    return path


def build_decoy_cases(
    case_manifest_csv: Path = Path("results/tables/boltz2_case_manifest.csv"),
    raw_dir: Path = Path("data/raw/pdb"),
    out_csv: Path = Path("results/tables/decoy_redesign_case_manifest.csv"),
    config_path: Path = Path("configs/antigen_redesign_decoys.yml"),
    limit: int | None = None,
) -> pd.DataFrame:
    write_default_decoy_config(config_path)
    cases = pd.read_csv(case_manifest_csv)
    if limit:
        cases = cases.head(limit)
    rows = []
    for case in cases.to_dict("records"):
        structure_id = str(case["structure_id"])
        atoms = parse_atoms(read_pdb_text(raw_dir / str(case["source_filename"])))
        nb_chain = str(case["nanobody_chain"])
        ag_chain = str(case["antigen_chain"])
        ag_seq = chain_sequence(atoms, ag_chain)
        nb_atoms = [a for a in atoms if a.chain_id == nb_chain and a.is_heavy]
        ag_atoms = [a for a in atoms if a.chain_id == ag_chain and a.is_heavy]
        ag_order = {(num, icode, resname): i + 1 for i, (num, icode, resname) in enumerate(chain_residues(atoms, ag_chain))}
        contacts = atom_contacts(nb_atoms, ag_atoms, 5.0)
        contact_keys = {ag_atoms[j].residue_key for _, j, _ in contacts}
        contact_positions = sorted(
            {
                ag_order[(key[1], key[2], key[3])]
                for key in contact_keys
                if (key[1], key[2], key[3]) in ag_order
            }
        )
        rows.append(
            {
                "structure_id": structure_id,
                "pdb_id": str(case["pdb_id"]),
                "nanobody_chain": nb_chain,
                "antigen_chain": ag_chain,
                "source_filename": str(case["source_filename"]),
                "antigen_length": len(ag_seq),
                "native_antigen_sequence": ag_seq,
                "antigen_contact_positions": positions_to_text(contact_positions),
                "antigen_contact_count": len(contact_positions),
                "target_passing_designs": DecoyDefaults().passing_designs_per_antigen,
                "candidate_budget": DecoyDefaults().candidates_per_antigen,
            }
        )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def prepare_decoy_inputs(
    case_manifest_csv: Path = Path("results/tables/decoy_redesign_case_manifest.csv"),
    raw_dir: Path = Path("data/raw/pdb"),
    work_dir: Path = Path("work/decoys"),
    out_mpnn_csv: Path = Path("results/tables/decoy_mpnn_job_manifest.csv"),
    limit: int | None = None,
) -> pd.DataFrame:
    cases = pd.read_csv(case_manifest_csv)
    if limit:
        cases = cases.head(limit)
    rows = []
    for case in cases.to_dict("records"):
        structure_id = str(case["structure_id"])
        atoms = parse_atoms(read_pdb_text(raw_dir / str(case["source_filename"])))
        antigen_chain = str(case["antigen_chain"])
        nb_chain = str(case["nanobody_chain"])
        native_seq = str(case["native_antigen_sequence"])
        input_dir = work_dir / "inputs" / structure_id
        pdb_path = input_dir / f"{structure_id}_antigen_chain_A.pdb"
        write_antigen_only_pdb(atoms, antigen_chain, pdb_path)
        # Burial-rank the contacts and force only the top fraction (all for interface-small
        # antigens, the most-buried hotspots for interface-dominated ones) so the fold survives.
        ranked = ranked_contact_positions(atoms, nb_chain, antigen_chain)
        fraction = adaptive_forced_fraction(len(ranked), len(native_seq))
        forced = select_forced_contacts(ranked, fraction)
        omit_jsonl = write_contact_omit_jsonl(pdb_path, forced, native_seq, input_dir / f"{structure_id}_omit_contacts.jsonl")
        output_dir = work_dir / "mpnn_outputs" / structure_id
        rows.append(
            {
                "structure_id": structure_id,
                "antigen_pdb": str(pdb_path),
                "designed_chain": BOLTZ_CHAIN_ANTIGEN,
                "num_seq_per_target": DecoyDefaults().candidates_per_antigen,
                "contact_omit_jsonl": str(omit_jsonl),
                "n_contacts": len(ranked),
                "forced_contact_fraction": fraction,
                "n_forced_contact_mutations": len(forced),
                "mpnn_output_dir": str(output_dir),
                "mpnn_command": proteinmpnn_command(pdb_path, output_dir, DecoyDefaults().candidates_per_antigen, DecoyDefaults().sampling_temp, omit_jsonl),
            }
        )
    df = pd.DataFrame(rows)
    out_mpnn_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_mpnn_csv, index=False)
    write_decoy_job_bundle(df, work_dir / "job_bundle")
    return df


def write_antigen_only_pdb(atoms: list[Atom], antigen_chain: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    residue_map: dict[tuple[int, str, str], int] = {}
    serial = 1
    for atom in atoms:
        if atom.chain_id != antigen_chain or atom.residue_name not in AA3_TO_1:
            continue
        key = (atom.residue_number, atom.insertion_code, atom.residue_name)
        if key not in residue_map:
            residue_map[key] = len(residue_map) + 1
        lines.append(format_atom(serial, atom, BOLTZ_CHAIN_ANTIGEN, residue_map[key], atom.residue_name))
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def proteinmpnn_command(pdb_path: Path, output_dir: Path, candidates: int, sampling_temp: float = DecoyDefaults().sampling_temp, omit_jsonl: Path | None = None) -> str:
    cmd = (
        "python $PROTEINMPNN_HOME/protein_mpnn_run.py "
        f"--pdb_path {pdb_path} --pdb_path_chains {BOLTZ_CHAIN_ANTIGEN} "
        f"--out_folder {output_dir} --num_seq_per_target {candidates} --sampling_temp {sampling_temp}"
    )
    if omit_jsonl is not None:
        cmd += f" --omit_AA_jsonl {omit_jsonl}"
    return cmd


def ranked_contact_positions(atoms: list[Atom], nb_chain: str, ag_chain: str, cutoff: float = 5.0) -> list[int]:
    """Antigen contact positions (1-indexed) sorted by descending interface burial,
    measured as the number of nanobody heavy atoms within `cutoff` of the residue.
    The most-buried contacts are the binding hotspots; forcing those first disrupts
    the most binding per residue mutated, letting peripheral contacts stay native to
    preserve the fold when only a fraction of contacts can be forced."""
    nb_atoms = [a for a in atoms if a.chain_id == nb_chain and a.is_heavy]
    ag_atoms = [a for a in atoms if a.chain_id == ag_chain and a.is_heavy]
    ag_order = {(num, icode, resname): i + 1 for i, (num, icode, resname) in enumerate(chain_residues(atoms, ag_chain))}
    degree: dict[int, int] = {}
    for _, j, _ in atom_contacts(nb_atoms, ag_atoms, cutoff):
        _, num, icode, resname = ag_atoms[j].residue_key
        pos = ag_order.get((num, icode, resname))
        if pos is not None:
            degree[pos] = degree.get(pos, 0) + 1
    return [pos for pos, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))]


def select_forced_contacts(ranked_positions: list[int], fraction: float) -> list[int]:
    """Top `fraction` of burial-ranked contacts, sorted back into positional order.
    fraction>=1.0 forces all contacts; lower fractions spare peripheral contacts."""
    if fraction >= 1.0:
        return sorted(ranked_positions)
    import math
    k = max(1, math.ceil(fraction * len(ranked_positions)))
    return sorted(ranked_positions[:k])


def adaptive_forced_fraction(n_contacts: int, antigen_len: int, defaults: DecoyDefaults = DecoyDefaults()) -> float:
    """Forced-contact fraction for an antigen: full force normally, but only the top
    interface_dominated_fraction of contacts when the interface dominates the sequence
    (contacts / length > interface_dominated_threshold), so the scaffold can refold."""
    if antigen_len > 0 and n_contacts / antigen_len > defaults.interface_dominated_threshold:
        return defaults.interface_dominated_fraction
    return defaults.forced_contact_fraction


def contact_omit_dict(pdb_stem: str, contact_positions: list[int], native_seq: str, chain: str = BOLTZ_CHAIN_ANTIGEN) -> dict:
    """ProteinMPNN omit_AA_jsonl that forbids the native amino acid at every contact
    residue, forcing the redesign to mutate 100% of interface contacts. Positions are
    1-indexed into the antigen chain (ProteinMPNN subtracts 1 internally), matching the
    contact-position convention. Format: {name: {chain: [[[pos], "AA"], ...]}}."""
    entries = [[[p], native_seq[p - 1]] for p in contact_positions if 1 <= p <= len(native_seq)]
    return {pdb_stem: {chain: entries}}


def write_contact_omit_jsonl(pdb_path: Path, contact_positions: list[int], native_seq: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contact_omit_dict(pdb_path.stem, contact_positions, native_seq)) + "\n")
    return out_path


def write_decoy_job_bundle(mpnn_manifest: pd.DataFrame, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    mpnn_manifest[["structure_id", "antigen_pdb", "mpnn_output_dir", "mpnn_command"]].to_csv(bundle_dir / "proteinmpnn_jobs.tsv", sep="\t", index=False)
    install = bundle_dir / "install_proteinmpnn_<CPU_NODE>.sh"
    install.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "INSTALL_DIR=${1:-$HOME/ProteinMPNN}",
                "if [ ! -d \"$INSTALL_DIR/.git\" ]; then",
                "  git clone https://github.com/dauparas/ProteinMPNN.git \"$INSTALL_DIR\"",
                "else",
                "  git -C \"$INSTALL_DIR\" pull --ff-only",
                "fi",
                "echo \"Set PROTEINMPNN_HOME=$INSTALL_DIR before running jobs.\"",
                "",
            ]
        )
    )
    runner = bundle_dir / "run_proteinmpnn_jobs.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "JOBS_TSV=${1:-work/decoys/job_bundle/proteinmpnn_jobs.tsv}",
                "export PROTEINMPNN_HOME=${PROTEINMPNN_HOME:-$HOME/ProteinMPNN}",
                "tail -n +2 \"$JOBS_TSV\" | while IFS=$'\\t' read -r structure_id antigen_pdb output_dir mpnn_command; do",
                "  mkdir -p \"$output_dir\"",
                "  echo \"[$structure_id] $mpnn_command\"",
                "  eval \"$mpnn_command\"",
                "done",
                "",
            ]
        )
    )
    for script in [install, runner]:
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def validate_decoy_designs(
    case_manifest_csv: Path = Path("results/tables/decoy_redesign_case_manifest.csv"),
    raw_dir: Path = Path("data/raw/pdb"),
    work_dir: Path = Path("work/decoys"),
    tables_dir: Path = Path("results/tables"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(case_manifest_csv)
    # Load forced_contact_fraction per antigen from the prepare manifest so the sequence
    # gate can scale the contact-mutation threshold for partially-forced antigens.
    mpnn_manifest_path = tables_dir / "decoy_mpnn_job_manifest.csv"
    if mpnn_manifest_path.exists():
        mpnn_manifest = pd.read_csv(mpnn_manifest_path).set_index("structure_id")
    else:
        mpnn_manifest = pd.DataFrame()
    candidate_rows = []
    boltz_rows = []
    validation_rows = []
    for case in cases.to_dict("records"):
        structure_id = str(case["structure_id"])
        native_seq = str(case["native_antigen_sequence"])
        contact_positions = parse_positions(str(case.get("antigen_contact_positions", "")))
        forced_fraction = float(mpnn_manifest.at[structure_id, "forced_contact_fraction"]) if structure_id in mpnn_manifest.index else 1.0
        candidates = parse_mpnn_candidates(work_dir / "mpnn_outputs" / structure_id, structure_id)
        seen: set[str] = set()
        filtered = []
        for candidate in candidates:
            seq = candidate["designed_sequence"]
            seq_status = sequence_filter_status(seq, native_seq, contact_positions, forced_contact_fraction=forced_fraction)
            row = {**candidate, **seq_status}
            if seq in seen:
                row["sequence_filter_status"] = "duplicate"
            seen.add(seq)
            candidate_rows.append(row)
            if row["sequence_filter_status"] == "ok":
                filtered.append(row)
        selected = sorted(filtered, key=lambda r: (pd.isna(r.get("mpnn_score")), r.get("mpnn_score", 0.0)))[: DecoyDefaults().refold_candidates_per_antigen]
        # Native-sequence monomer refold: the self-consistency reference for this antigen.
        if selected:
            native_row = {"design_id": f"{structure_id}_native", "structure_id": structure_id, "designed_sequence": native_seq, "is_native_reference": True}
            n_yaml, n_out = write_boltz2_monomer_input(native_row, work_dir)
            boltz_rows.append({**native_row, "boltz_yaml": str(n_yaml), "boltz_output_dir": str(n_out), "boltz_command": boltz_monomer_command(n_yaml, n_out)})
            validation_rows.append(validate_boltz_monomer(native_row, case, raw_dir, n_out))
        for row in selected:
            yaml_path, output_dir = write_boltz2_monomer_input(row, work_dir)
            boltz_rows.append({**row, "boltz_yaml": str(yaml_path), "boltz_output_dir": str(output_dir), "boltz_command": boltz_monomer_command(yaml_path, output_dir)})
            validation_rows.append(validate_boltz_monomer(row, case, raw_dir, output_dir))
        if not candidates:
            validation_rows.append({"structure_id": structure_id, "validation_status": "missing_mpnn_output"})
    candidates_df = pd.DataFrame(candidate_rows)
    boltz_df = pd.DataFrame(boltz_rows)
    validation_df = pd.DataFrame(validation_rows)
    if not validation_df.empty and "validation_status" in validation_df:
        validation_df["selected_decoy"] = False
        if "design_id" in validation_df.columns:
            ok = validation_df[validation_df["validation_status"] == "pass"].copy()
            sort_cols = [col for col in ["structure_id", "monomer_ca_rmsd", "mpnn_score"] if col in ok]
            if sort_cols:
                ok = ok.sort_values(sort_cols)
            selected_ids = set(ok.groupby("structure_id").head(DecoyDefaults().passing_designs_per_antigen)["design_id"]) if "design_id" in ok else set()
            validation_df.loc[validation_df["design_id"].isin(selected_ids), "selected_decoy"] = True
    summary_df = summarize_decoy_validation(cases, candidates_df, validation_df)
    tables_dir.mkdir(parents=True, exist_ok=True)
    if candidates_df.empty:
        candidates_df = pd.DataFrame(columns=["design_id", "structure_id", "mpnn_source", "mpnn_header", "mpnn_score", "designed_sequence", "sequence_filter_status", "sequence_identity", "contact_mutation_fraction", "mutated_contact_positions"])
    if boltz_df.empty:
        boltz_df = pd.DataFrame(columns=["design_id", "structure_id", "designed_sequence", "mpnn_score", "sequence_identity", "contact_mutation_fraction", "boltz_yaml", "boltz_output_dir", "boltz_command"])
    candidates_df.to_csv(tables_dir / "decoy_mpnn_candidates.csv", index=False)
    boltz_df.to_csv(tables_dir / "decoy_boltz2_monomer_manifest.csv", index=False)
    validation_df.to_csv(tables_dir / "decoy_validation.csv", index=False)
    summary_df.to_csv(tables_dir / "decoy_summary.csv", index=False)
    write_boltz_monomer_job_bundle(boltz_df, work_dir / "job_bundle")
    return candidates_df, boltz_df, validation_df


def parse_mpnn_candidates(output_dir: Path, structure_id: str) -> list[dict[str, object]]:
    if not output_dir.exists():
        return []
    paths = sorted([p for p in output_dir.rglob("*") if p.suffix.lower() in {".fa", ".fasta"}])
    rows = []
    idx = 0
    for path in paths:
        header = ""
        seq_parts: list[str] = []
        for line in path.read_text(errors="ignore").splitlines() + [">"]:
            if line.startswith(">"):
                if seq_parts:
                    seq = "".join(seq_parts).replace("/", "").upper()
                    rows.append(
                        {
                            "design_id": f"{structure_id}_mpnn_{idx:04d}",
                            "structure_id": structure_id,
                            "mpnn_source": str(path),
                            "mpnn_header": header,
                            "mpnn_score": parse_header_float(header, "score"),
                            "designed_sequence": seq,
                        }
                    )
                    idx += 1
                header = line[1:].strip()
                seq_parts = []
            elif line.strip():
                seq_parts.append(line.strip())
    return rows


def parse_header_float(header: str, key: str) -> float:
    match = re.search(rf"{re.escape(key)}[=:]\s*(-?\d+(?:\.\d+)?)", header)
    return float(match.group(1)) if match else float("nan")


def sequence_filter_status(
    seq: str,
    native_seq: str,
    contact_positions: list[int],
    forced_contact_fraction: float = 1.0,
) -> dict[str, object]:
    if len(seq) != len(native_seq):
        status = "length_mismatch"
    elif any(aa not in CANONICAL_AA for aa in seq):
        status = "noncanonical"
    else:
        identity = sequence_identity(seq, native_seq)
        contact_mut = contact_mutation_fraction(seq, native_seq, contact_positions)
        # Contact mutation is the sole sequence-validity gate: ProteinMPNN is forced to
        # mutate the native residue at every forced contact position (via omit_AA_jsonl).
        # For interface-dominated antigens (partial forcing), only a subset of contacts
        # are forced, so the overall contact_mut fraction is lower; the threshold scales
        # proportionally so the gate still catches omit-file failures without rejecting
        # valid partial-force designs.
        defaults = DecoyDefaults()
        effective_min = defaults.min_contact_mutation_fraction * forced_contact_fraction
        status = "ok" if contact_mut >= effective_min else "failed_thresholds"
        return {
            "sequence_filter_status": status,
            "sequence_identity": identity,
            "contact_mutation_fraction": contact_mut,
            "mutated_contact_positions": mutated_contact_count(seq, native_seq, contact_positions),
        }
    return {"sequence_filter_status": status, "sequence_identity": np.nan, "contact_mutation_fraction": np.nan, "mutated_contact_positions": np.nan}


def sequence_identity(seq: str, native_seq: str) -> float:
    if len(seq) != len(native_seq) or not native_seq:
        return float("nan")
    return sum(a == b for a, b in zip(seq, native_seq)) / len(native_seq)


def contact_mutation_fraction(seq: str, native_seq: str, positions: list[int]) -> float:
    if not positions:
        return float("nan")
    return mutated_contact_count(seq, native_seq, positions) / len(positions)


def mutated_contact_count(seq: str, native_seq: str, positions: list[int]) -> int:
    return sum(seq[pos - 1] != native_seq[pos - 1] for pos in positions if 1 <= pos <= len(seq) <= len(native_seq))


def write_boltz2_monomer_input(candidate: dict[str, object], work_dir: Path) -> tuple[Path, Path]:
    structure_id = str(candidate["structure_id"])
    design_id = str(candidate["design_id"])
    input_dir = work_dir / "inputs" / structure_id / "boltz2_monomer" / design_id
    output_dir = work_dir / "boltz2_monomer_outputs" / structure_id / design_id
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = input_dir / f"{design_id}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "version: 1",
                "sequences:",
                "  - protein:",
                f"      id: {BOLTZ_CHAIN_ANTIGEN}",
                f"      sequence: {candidate['designed_sequence']}",
                "",
            ]
        )
    )
    return yaml_path, output_dir


def boltz_monomer_command(yaml_path: Path, output_dir: Path) -> str:
    return (
        f"boltz predict {yaml_path} --model boltz2 --use_msa_server --use_potentials "
        f"--diffusion_samples 5 --recycling_steps 3 --output_format pdb --write_full_pae --write_full_pde --out_dir {output_dir}"
    )


def write_boltz_monomer_job_bundle(boltz_df: pd.DataFrame, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / "boltz2_monomer_jobs.tsv"
    cols = ["design_id", "structure_id", "boltz_yaml", "boltz_output_dir", "boltz_command"]
    if boltz_df.empty:
        pd.DataFrame(columns=cols).to_csv(path, sep="\t", index=False)
    else:
        boltz_df[cols].to_csv(path, sep="\t", index=False)
    script = bundle_dir / "run_boltz2_monomer_jobs.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "BOLTZ=${BOLTZ:-boltz}",
                "JOBS_TSV=${1:-work/decoys/job_bundle/boltz2_monomer_jobs.tsv}",
                "tail -n +2 \"$JOBS_TSV\" | while IFS=$'\\t' read -r design_id structure_id yaml_path output_dir boltz_command; do",
                "  mkdir -p \"$output_dir\"",
                "  echo \"[$design_id] $boltz_command\"",
                "  $BOLTZ predict \"$yaml_path\" --model boltz2 --use_msa_server --use_potentials --diffusion_samples 5 --recycling_steps 3 --output_format pdb --write_full_pae --write_full_pde --out_dir \"$output_dir\"",
                "done",
                "",
            ]
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def validate_boltz_monomer(candidate: dict[str, object], case: dict[str, object], raw_dir: Path, output_dir: Path) -> dict[str, object]:
    base = {k: candidate.get(k) for k in ["design_id", "structure_id", "designed_sequence", "mpnn_score", "sequence_identity", "contact_mutation_fraction"]}
    predictions = discover_boltz_predictions(output_dir)
    if not predictions:
        return {**base, "validation_status": "missing_boltz2_output", "boltz_output_dir": str(output_dir)}
    structure_id = str(case["structure_id"])
    crystal_ag = remap_atoms(parse_atoms(read_pdb_text(raw_dir / str(case["source_filename"]))), {str(case["antigen_chain"]): BOLTZ_CHAIN_ANTIGEN})
    # The native-sequence monomer refold is the self-consistency reference (a design
    # "folds" if it adopts the same fold the native sequence adopts as a free monomer);
    # the bound crystal is unreachable for antigens whose free monomer differs.
    native_refold_preds = discover_boltz_predictions(output_dir.parent / f"{structure_id}_native")
    native_ref = normalize_monomer_chain(parse_atoms(read_pdb_text(native_refold_preds[0]))) if native_refold_preds else None
    reference = native_ref if native_ref is not None else crystal_ag
    fold_reference = "native_refold" if native_ref is not None else "crystal_fallback"
    if candidate.get("is_native_reference"):
        # The native job is the reference, not a decoy candidate; do not gate it.
        return {**base, "fold_reference": "self", "prediction_path": str(predictions[0]), "validation_status": "native_reference"}
    # best-of-N: score every diffusion sample against the reference, keep the best fold.
    best = max(
        (
            {**monomer_validation_metrics(reference, normalize_monomer_chain(parse_atoms(read_pdb_text(p)))), "prediction_path": str(p), "_pred": p}
            for p in predictions
        ),
        key=lambda m: (m["monomer_tm_score"] if not np.isnan(m["monomer_tm_score"]) else -1.0),
    )
    pred_path = best.pop("_pred")
    crystal_metrics = monomer_validation_metrics(crystal_ag, normalize_monomer_chain(parse_atoms(read_pdb_text(pred_path))))
    best["monomer_tm_score_vs_crystal"] = crystal_metrics["monomer_tm_score"]
    best["monomer_ca_rmsd_vs_crystal"] = crystal_metrics["monomer_ca_rmsd"]
    confidence = parse_confidence_json(pred_path)
    passed = best["monomer_tm_score"] >= DecoyDefaults().monomer_tm_score_min and best["monomer_aligned_fraction"] >= DecoyDefaults().monomer_aligned_fraction_min
    return {**base, **best, **confidence, "fold_reference": fold_reference, "n_diffusion_samples": len(predictions), "validation_status": "pass" if passed else "failed_fold"}


def normalize_monomer_chain(atoms: list[Atom]) -> list[Atom]:
    chains = sorted({a.chain_id for a in atoms})
    if BOLTZ_CHAIN_ANTIGEN in chains:
        return atoms
    if len(chains) == 1:
        return [Atom(BOLTZ_CHAIN_ANTIGEN, a.residue_number, a.insertion_code, a.residue_name, a.atom_name, a.element, a.x, a.y, a.z) for a in atoms]
    return atoms


def monomer_validation_metrics(native_atoms: list[Atom], pred_atoms: list[Atom]) -> dict[str, float]:
    native = ca_map(native_atoms, BOLTZ_CHAIN_ANTIGEN)
    pred = ca_map(pred_atoms, BOLTZ_CHAIN_ANTIGEN)
    keys = sorted(set(native) & set(pred))
    if len(keys) < 3:
        return {"monomer_tm_score": 0.0, "monomer_ca_rmsd": np.nan, "monomer_aligned_fraction": 0.0, "monomer_matched_ca": len(keys)}
    ref = np.array([native[k] for k in keys])
    mob = np.array([pred[k] for k in keys])
    _, kabsch_aligned = superpose(ref, mob)
    core_aligned = core_locked_alignment(ref, mob, len(native))
    return {
        # TM-score on the TMalign-style core-locked superposition (robust to flexible
        # termini); CA-RMSD on the standard RMSD-minimizing (Kabsch) superposition.
        "monomer_tm_score": tm_score(ref, core_aligned, len(native)),
        "monomer_ca_rmsd": rmsd(ref, kabsch_aligned),
        "monomer_aligned_fraction": len(keys) / len(native),
        "monomer_matched_ca": len(keys),
    }


def core_locked_alignment(ref: np.ndarray, mob: np.ndarray, l_target: int, iters: int = 5) -> np.ndarray:
    """Return mob superposed onto ref with a TMalign-style core-locked rotation:
    superpose on the inlier core, shrink the inlier set to residues within 2*d0,
    repeat. This stops a flexible tail from pulling the core out of register, so the
    TM-score reflects the structured fold rather than terminal flapping. Falls back to
    the plain Kabsch superposition if the core collapses below 3 residues."""
    d0 = _tm_d0(l_target)
    cut = 2.0 * d0
    idx = np.arange(len(ref))
    for _ in range(iters):
        rot, _ = superpose(ref[idx], mob[idx])
        aligned = (mob - mob[idx].mean(axis=0)) @ rot + ref[idx].mean(axis=0)
        new = np.where(np.linalg.norm(ref - aligned, axis=1) < cut)[0]
        if len(new) < 3 or len(new) == len(idx):
            idx = new if len(new) >= 3 else idx
            break
        idx = new
    rot, _ = superpose(ref[idx], mob[idx])
    return (mob - mob[idx].mean(axis=0)) @ rot + ref[idx].mean(axis=0)


def _tm_d0(l_target: int) -> float:
    return max(0.5, 1.24 * (l_target - 15) ** (1.0 / 3.0) - 1.8) if l_target > 15 else 0.5


def tm_score(ref: np.ndarray, aligned: np.ndarray, l_target: int) -> float:
    """TM-score of matched CA pairs (already superposed), normalized by the native
    (target) length. The redesigned antigen shares the native numbering, so residues
    correspond 1:1 and no sequence alignment is needed. Unmatched native residues
    contribute 0, so partial folds are penalized via the L_target denominator.

    `aligned` must be a superposition of mob onto ref. Pair with core_locked_alignment
    for a TMalign-style (flexible-terminus-robust) score; a plain Kabsch superposition
    is a lower bound when termini flap (it gets pulled out of core register)."""
    if l_target <= 0:
        return 0.0
    d0 = _tm_d0(l_target)
    d = np.linalg.norm(ref - aligned, axis=1)
    return float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / l_target)


def ca_map(atoms: list[Atom], chain: str) -> dict[int, np.ndarray]:
    return {a.residue_number: np.array([a.x, a.y, a.z], dtype=float) for a in atoms if a.chain_id == chain and a.atom_name == "CA"}


def summarize_decoy_validation(cases: pd.DataFrame, candidates: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in cases.to_dict("records"):
        sid = str(case["structure_id"])
        c = candidates[candidates.get("structure_id", pd.Series(dtype=str)) == sid] if not candidates.empty else pd.DataFrame()
        v = validation[validation.get("structure_id", pd.Series(dtype=str)) == sid] if not validation.empty else pd.DataFrame()
        rows.append(
            {
                "structure_id": sid,
                "mpnn_candidates": len(c),
                "sequence_filter_ok": int((c.get("sequence_filter_status", pd.Series(dtype=str)) == "ok").sum()) if not c.empty else 0,
                "boltz2_pass": int((v.get("validation_status", pd.Series(dtype=str)) == "pass").sum()) if not v.empty else 0,
                "selected_decoys": int(v.get("selected_decoy", pd.Series(dtype=bool)).fillna(False).sum()) if not v.empty else 0,
                "target_passing_designs": DecoyDefaults().passing_designs_per_antigen,
            }
        )
    return pd.DataFrame(rows)


def graft_decoy_complexes(
    validation_csv: Path = Path("results/tables/decoy_validation.csv"),
    case_manifest_csv: Path = Path("results/tables/decoy_redesign_case_manifest.csv"),
    raw_dir: Path = Path("data/raw/pdb"),
    work_dir: Path = Path("work/decoys"),
    out_csv: Path = Path("results/tables/decoy_grafted_complex_manifest.csv"),
) -> pd.DataFrame:
    """Build refold-graft decoy complexes for every selected design.

    The Boltz2-predicted antigen monomer (full side chains) is rigidly superposed
    onto the native antigen pose, the native antigen is dropped, and the native
    nanobody is kept — producing a complex where the antigen is an independently
    folded sequence-scrambled look-alike. Output is written in the A=antigen /
    N=nanobody convention so the native-crystal metric tools run unchanged.
    """
    validation = pd.read_csv(validation_csv) if validation_csv.exists() else pd.DataFrame()
    cases = pd.read_csv(case_manifest_csv).set_index("structure_id")
    rows = []
    selected = validation[validation.get("selected_decoy", pd.Series(dtype=bool)).fillna(False)] if not validation.empty else pd.DataFrame()
    for row in selected.to_dict("records"):
        sid = str(row["structure_id"])
        case = cases.loc[sid]
        native_atoms = parse_atoms(read_pdb_text(raw_dir / str(case["source_filename"])))
        out_path = work_dir / "grafted_complexes" / sid / f"{row['design_id']}_NA.pdb"
        pred_path = row.get("prediction_path")
        result = graft_complex(
            native_atoms,
            str(case["nanobody_chain"]),
            str(case["antigen_chain"]),
            Path(str(pred_path)) if pred_path is not None and str(pred_path) not in {"", "nan"} else None,
            out_path,
        )
        rows.append(
            {
                "design_id": row["design_id"],
                "structure_id": sid,
                "decoy_type": "refold_grafted",
                "grafted_complex_path": str(out_path) if result["graft_status"] == "ok" else "",
                "nanobody_chain": BOLTZ_CHAIN_NANOBODY,
                "antigen_chain": BOLTZ_CHAIN_ANTIGEN,
                "designed_sequence": row.get("designed_sequence"),
                **result,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["design_id", "structure_id", "decoy_type", "grafted_complex_path", "nanobody_chain", "antigen_chain", "designed_sequence", "monomer_graft_rmsd", "matched_ca", "graft_antigen_completeness", "graft_status"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def graft_complex(native_atoms: list[Atom], nb_chain: str, ag_chain: str, pred_path: Path | None, out_path: Path) -> dict[str, object]:
    """Superpose the predicted antigen monomer onto the native antigen and write
    the decoy complex (predicted antigen on chain A, native nanobody on chain N).
    """
    failure = {"grafted_complex_path": "", "monomer_graft_rmsd": np.nan, "matched_ca": 0, "graft_antigen_completeness": np.nan}
    if pred_path is None:
        return {**failure, "graft_status": "missing_prediction"}
    if pred_path.suffix.lower() != ".pdb":
        return {**failure, "graft_status": "prediction_not_pdb"}
    if not pred_path.exists():
        return {**failure, "graft_status": "missing_prediction"}

    native_ag = remap_atoms(native_atoms, {ag_chain: BOLTZ_CHAIN_ANTIGEN})
    native_nb = remap_atoms(native_atoms, {nb_chain: BOLTZ_CHAIN_NANOBODY})
    pred = normalize_monomer_chain(parse_atoms(read_pdb_text(pred_path)))

    native_ca = ca_map(native_ag, BOLTZ_CHAIN_ANTIGEN)
    pred_ca = ca_map(pred, BOLTZ_CHAIN_ANTIGEN)
    keys = sorted(set(native_ca) & set(pred_ca))
    if len(keys) < 3:
        return {**failure, "matched_ca": len(keys), "graft_status": "too_few_matched_ca"}

    ref = np.array([native_ca[k] for k in keys])
    mob = np.array([pred_ca[k] for k in keys])
    rotation, pred_center, native_center = rigid_transform_from_ca(ref, mob)
    aligned_ca = apply_rigid(mob, rotation, pred_center, native_center)
    graft_rmsd = rmsd(ref, aligned_ca)

    pred_ag_atoms = [a for a in pred if a.chain_id == BOLTZ_CHAIN_ANTIGEN]
    coords = np.array([[a.x, a.y, a.z] for a in pred_ag_atoms])
    moved = apply_rigid(coords, rotation, pred_center, native_center)
    grafted_ag = [
        Atom(BOLTZ_CHAIN_ANTIGEN, a.residue_number, "", a.residue_name, a.atom_name, a.element, float(xyz[0]), float(xyz[1]), float(xyz[2]))
        for a, xyz in zip(pred_ag_atoms, moved)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    serial = 1
    for atom in native_nb:
        if atom.residue_name in AA3_TO_1:
            lines.append(format_atom(serial, atom, BOLTZ_CHAIN_NANOBODY, atom.residue_number, atom.residue_name))
            serial += 1
    for atom in grafted_ag:
        if atom.residue_name in AA3_TO_1:
            lines.append(format_atom(serial, atom, BOLTZ_CHAIN_ANTIGEN, atom.residue_number, atom.residue_name))
            serial += 1
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")
    return {
        "grafted_complex_path": str(out_path),
        "monomer_graft_rmsd": graft_rmsd,
        "matched_ca": len(keys),
        "graft_antigen_completeness": len(keys) / len(native_ca) if native_ca else np.nan,
        "graft_status": "ok",
    }


def rigid_transform_from_ca(native_ca: np.ndarray, pred_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rotation, pred_centroid, native_centroid) for the matched CA arrays.

    Apply with apply_rigid: (X - pred_centroid) @ rotation + native_centroid, the
    same convention as boltz2.superpose / ligand_rmsd_after_receptor_alignment.
    """
    rotation, _ = superpose(native_ca, pred_ca)
    return rotation, pred_ca.mean(axis=0), native_ca.mean(axis=0)


def apply_rigid(coords: np.ndarray, rotation: np.ndarray, pred_centroid: np.ndarray, native_centroid: np.ndarray) -> np.ndarray:
    return (coords - pred_centroid) @ rotation + native_centroid


def compute_decoy_metrics(
    relax_manifest_csv: Path = Path("results/tables/decoy_relax_manifest.csv"),
    case_manifest_csv: Path = Path("results/tables/decoy_redesign_case_manifest.csv"),
    tables_dir: Path = Path("results/tables"),
    work_dir: Path = Path("work/decoys"),
    run_rosetta: bool = True,
    run_protein_interface: bool = True,
    run_sc: bool = True,
    docker_image: str = "rosettacommons/rosetta:latest",
) -> dict[str, pd.DataFrame]:
    """Score relaxed grafted decoys with the same suite the native crystals got
    (33 protein_interface metrics + Lawrence-Colman SC + Rosetta InterfaceAnalyzer)
    and merge against the native baseline for direct comparison.
    """
    manifest = read_csv_or_empty(relax_manifest_csv)
    rows = [r for r in manifest.to_dict("records") if str(r.get("relaxed_pdb", "")) not in {"", "nan"} and Path(str(r["relaxed_pdb"])).exists()]
    design_ids = [str(r["design_id"]) for r in rows]
    structure_ids = [str(r["structure_id"]) for r in rows]
    pdb_paths = [Path(str(r["relaxed_pdb"])) for r in rows]

    pi = _decoy_protein_interface(pdb_paths, design_ids, structure_ids) if run_protein_interface else pd.DataFrame()
    sc = _decoy_shape_complementarity(pdb_paths, design_ids, structure_ids) if run_sc else pd.DataFrame()
    rosetta = pd.DataFrame(
        [run_decoy_rosetta(p, r, work_dir, docker_image) for p, r in zip(pdb_paths, rows)] if run_rosetta
        else [{"design_id": d, "structure_id": s, "rosetta_status": "not_run"} for d, s in zip(design_ids, structure_ids)]
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    for df, name in [(pi, "decoy_protein_interface.csv"), (sc, "decoy_shape_complementarity.csv"), (rosetta, "decoy_rosetta_interface.csv")]:
        (df if not df.empty else pd.DataFrame(columns=["design_id", "structure_id"])).to_csv(tables_dir / name, index=False)

    comparison = build_decoy_native_comparison(pi, sc, rosetta, tables_dir)
    comparison.to_csv(tables_dir / "decoy_native_comparison.csv", index=False)
    return {"protein_interface": pi, "shape_complementarity": sc, "rosetta": rosetta, "comparison": comparison}


# Scalar InterfaceResult fields, matching scripts/backfill_protein_interface.py so the
# decoy protein_interface columns line up 1:1 with native_protein_interface.csv.
PROTEIN_INTERFACE_SCALAR_FIELDS = [
    "dsasa", "dsasa_a", "dsasa_b", "asymmetry",
    "bb_dsasa", "sc_dsasa", "sidechain_fraction",
    "bhsa", "bpsa", "bcsa", "hydrophobic_fraction", "aromatic_dsasa_fraction",
    "hbonds", "hbond_density", "salt_bridges", "pi_pi", "cation_pi", "disulfides",
    "buried_unsat_polar",
    "planarity_rmsd", "planarity_ratio", "elongation", "interface_depth",
    "n_interface_a", "n_interface_b", "atomic_contacts",
    "gly_pro_fraction",
    "charge_a", "charge_b", "charge_complementarity",
    "mean_bfactor_interface", "min_bfactor_interface",
    "prodigy_dg",
    "sc",
]


def _decoy_protein_interface(pdb_paths: list[Path], design_ids: list[str], structure_ids: list[str]) -> pd.DataFrame:
    """Run the sibling-repo protein_interface analyzer (deferred import; antigen=A,
    nanobody=N), extracting the same scalar fields the native backfill stores.
    Returns an `*_unavailable` status frame if the package is absent."""
    try:
        from protein_interface import load_atoms, analyze_batch  # type: ignore
    except Exception:  # noqa: BLE001
        return pd.DataFrame([{"design_id": d, "structure_id": s, "metrics_status": "protein_interface_unavailable"} for d, s in zip(design_ids, structure_ids)])
    rows = []
    for path, design_id, sid in zip(pdb_paths, design_ids, structure_ids):
        try:
            a = load_atoms(str(path), chains=[BOLTZ_CHAIN_ANTIGEN])
            b = load_atoms(str(path), chains=[BOLTZ_CHAIN_NANOBODY])
            if len(a.coords) == 0 or len(b.coords) == 0:
                rows.append({"design_id": design_id, "structure_id": sid, "metrics_status": "empty_chain"})
                continue
            result = analyze_batch([(a, b)])[0]
            row = {f: getattr(result, f, None) for f in PROTEIN_INTERFACE_SCALAR_FIELDS}
            row["n_hotspots_a"] = len(getattr(result, "hotspots_a", []) or [])
            row["n_hotspots_b"] = len(getattr(result, "hotspots_b", []) or [])
            rows.append({"design_id": design_id, "structure_id": sid, "metrics_status": "ok", **row})
        except Exception as exc:  # noqa: BLE001
            rows.append({"design_id": design_id, "structure_id": sid, "metrics_status": f"failed:{type(exc).__name__}"})
    return pd.DataFrame(rows)


def _decoy_shape_complementarity(pdb_paths: list[Path], design_ids: list[str], structure_ids: list[str]) -> pd.DataFrame:
    """Run sibling-repo Lawrence-Colman SC (deferred import; group a = antigen A,
    group b = nanobody N)."""
    if not pdb_paths:
        return pd.DataFrame()
    try:
        from shape_complementarity.batch import score_many  # type: ignore
    except Exception:  # noqa: BLE001
        return pd.DataFrame([{"design_id": d, "structure_id": s, "sc_status": "shape_complementarity_unavailable"} for d, s in zip(design_ids, structure_ids)])
    try:
        scored = pd.DataFrame(score_many(pdb_paths=[str(p) for p in pdb_paths], chains_a=[BOLTZ_CHAIN_ANTIGEN], chains_b=[BOLTZ_CHAIN_NANOBODY])).reset_index(drop=True)
        if len(scored) != len(design_ids):  # score_many dropped/reordered rows — align by path instead
            scored["design_id"] = pd.NA
            scored["structure_id"] = pd.NA
            return scored
        scored.insert(0, "design_id", design_ids)
        scored.insert(1, "structure_id", structure_ids)
        return scored
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame([{"design_id": d, "structure_id": s, "sc_status": f"failed:{type(exc).__name__}"} for d, s in zip(design_ids, structure_ids)])


def build_decoy_native_comparison(pi: pd.DataFrame, sc: pd.DataFrame, rosetta: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Join decoy metrics to the native-crystal baseline (per structure_id) so each
    decoy row carries `native_<metric>` columns for the same antigen."""
    base = pd.DataFrame()
    for df in (pi, sc, rosetta):
        if df.empty or not {"design_id", "structure_id"}.issubset(df.columns):
            continue
        if base.empty:
            base = df.copy()
        else:
            # Only bring columns not already present (besides the join key). Both
            # protein_interface and shape_complementarity emit a column named `sc`
            # with identical values; merging both would suffix them `_x`/`_y` and
            # break the native pairing. Keep the first occurrence.
            new_cols = [c for c in df.columns if c == "design_id" or c not in base.columns]
            base = base.merge(df[new_cols], on="design_id", how="outer")
    if base.empty:
        return base
    # The dedicated shape_complementarity / Rosetta tables are authoritative for
    # `sc` and interface energetics. Join them before protein_interface so its
    # native `sc` (unpopulated, all-NaN for crystals) cannot shadow the real
    # Lawrence-Colman value via the skip-existing guard in _join_native_baseline.
    base = _join_native_baseline(base, tables_dir / "native_shape_complementarity.csv", cols=["sc", "median_distance", "trimmed_area"])
    base = _join_native_baseline(base, tables_dir / "rosetta_interface_native.csv", cols=["dG_separated", "dG_separated/dSASAx100", "dSASA_int", "sc_value", "hbonds_int", "delta_unsatHbonds"])
    base = _join_native_baseline(base, tables_dir / "native_protein_interface.csv")
    return base


def _join_native_baseline(base: pd.DataFrame, native_csv: Path, cols: list[str] | None = None) -> pd.DataFrame:
    """Merge a native-crystal baseline table onto the decoy frame by structure_id,
    prefixing every metric column with `native_`. `cols` restricts which metric
    columns are carried (defaults to all non-id columns); missing columns and any
    `native_<col>` already provided by an earlier, authoritative table are skipped."""
    native = read_csv_or_empty(native_csv)
    if native.empty or "structure_id" not in native.columns:
        return base
    if cols is None:
        metric_cols = [c for c in native.columns if c != "structure_id"]
    else:
        metric_cols = [c for c in cols if c in native.columns]
    metric_cols = [c for c in metric_cols if f"native_{c}" not in base.columns]
    if not metric_cols:
        return base
    keep = native[["structure_id", *metric_cols]].rename(columns={c: f"native_{c}" for c in metric_cols})
    return base.merge(keep, on="structure_id", how="left")


def run_decoy_rosetta(pdb_path: Path, row: dict[str, object], work_dir: Path, docker_image: str) -> dict[str, object]:
    base = {"design_id": row["design_id"], "structure_id": row["structure_id"], "scored_pdb": str(pdb_path)}
    if shutil.which("docker") is None:
        return {**base, "rosetta_status": "docker_unavailable"}
    score_dir = work_dir / "rosetta" / str(row["structure_id"])
    score_dir.mkdir(parents=True, exist_ok=True)
    scorefile = score_dir / f"{row['design_id']}.interface.sc"
    log = score_dir / f"{row['design_id']}.interface.log"
    # Decoys are written in the A=antigen / N=nanobody convention, so the interface
    # is N_A regardless of the original crystal chain letters.
    interface = f"{BOLTZ_CHAIN_NANOBODY}_{BOLTZ_CHAIN_ANTIGEN}"
    cmd = interface_analyzer_docker_command(pdb_path, score_dir, scorefile, docker_image, interface)
    with log.open("w") as handle:
        status = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
    score = parse_scorefile(scorefile)
    keep = {k: score.get(k, np.nan) for k in ["dG_separated", "dG_separated/dSASAx100", "dSASA_int", "sc_value", "hbonds_int", "delta_unsatHbonds"]}
    return {**base, **keep, "rosetta_status": "ok" if status.returncode == 0 else f"failed:{status.returncode}", "interface_log": str(log)}


def build_decoy_report(tables_dir: Path = Path("results/tables"), figures_dir: Path = Path("results/figures")) -> None:
    set_publication_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_csv_or_empty(tables_dir / "decoy_mpnn_candidates.csv")
    validation = read_csv_or_empty(tables_dir / "decoy_validation.csv")
    comparison = read_csv_or_empty(tables_dir / "decoy_native_comparison.csv")
    plot_identity_contacts(candidates, figures_dir / "decoy_identity_contact_mutation.png")
    plot_validation(validation, figures_dir / "decoy_boltz2_monomer_validation.png")
    plot_decoy_vs_native(comparison, figures_dir / "decoy_vs_native_metrics.png")


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def plot_identity_contacts(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    if df.empty or "sequence_identity" not in df:
        ax.text(0.5, 0.5, "No ProteinMPNN candidates found", ha="center", va="center")
    else:
        ax.scatter(df["sequence_identity"], df["contact_mutation_fraction"], s=18, alpha=0.7)
        ax.axvline(0.30, color="black", lw=1, ls="--")
        ax.axhline(0.90, color="black", lw=1, ls="--")
        ax.set_xlabel("Native antigen sequence identity")
        ax.set_ylabel("Contact residue mutation fraction")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_validation(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    if df.empty or "monomer_ca_rmsd" not in df:
        ax.text(0.5, 0.5, "No Boltz2 monomer validations found", ha="center", va="center")
    else:
        ok = df.dropna(subset=["monomer_ca_rmsd"])
        colors = ok.get("validation_status", pd.Series(index=ok.index, data="unknown")).map({"pass": "#2ca02c", "failed_fold": "#d62728"}).fillna("#7f7f7f")
        ax.scatter(ok["monomer_ca_rmsd"], ok.get("confidence_score", pd.Series(index=ok.index, data=np.nan)), c=colors, s=22, alpha=0.8)
        ax.axvline(2.5, color="black", lw=1, ls="--")
        ax.set_xlabel("Boltz2 monomer CA RMSD to native")
        ax.set_ylabel("Boltz2 confidence score")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_decoy_vs_native(df: pd.DataFrame, path: Path) -> None:
    """Decoy metric vs its native-crystal baseline. Points below the y=x line are
    decoys whose interface scores worse than the real interface."""
    pairs = [("sc", "Shape complementarity"), ("dG_separated", "Rosetta interface dG"), ("hbond_density", "H-bond density")]
    available = [(m, label) for m, label in pairs if m in df.columns and f"native_{m}" in df.columns]
    fig, axes = plt.subplots(1, max(1, len(available)), figsize=(4.6 * max(1, len(available)), 4.0), constrained_layout=True, squeeze=False)
    if df.empty or not available:
        axes[0, 0].text(0.5, 0.5, "No decoy-vs-native comparison found", ha="center", va="center")
    else:
        for ax, (metric, label) in zip(axes[0], available):
            sub = df.dropna(subset=[metric, f"native_{metric}"])
            ax.scatter(sub[f"native_{metric}"], sub[metric], s=22, alpha=0.7)
            lo = float(min(sub[f"native_{metric}"].min(), sub[metric].min()))
            hi = float(max(sub[f"native_{metric}"].max(), sub[metric].max()))
            ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
            ax.set_xlabel(f"Native {label}")
            ax.set_ylabel(f"Decoy {label}")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def positions_to_text(positions: list[int]) -> str:
    return ";".join(str(x) for x in positions)


def parse_positions(text: str) -> list[int]:
    if not text or text == "nan":
        return []
    return [int(part) for part in re.split(r"[;,]", text) if part.strip()]


def format_atom(serial: int, atom: Atom, chain: str, resseq: int, residue_name: str) -> str:
    return (
        f"ATOM  {serial:5d} {pdb_atom_name_field(atom.atom_name, atom.element)} {residue_name:>3} {chain:1}{resseq:4d}    "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}  1.00  0.00          {atom.element:>2}"
    )


def pdb_atom_name_field(atom_name: str, element: str) -> str:
    """PDB-standard 4-char atom-name field (columns 13-16).

    Single-character elements with <4-char names are left-justified starting at
    column 14 (a leading space in column 13); 4-char names or 2-char elements
    start at column 13. The old writer placed every name at column 13, which made
    Rosetta misread the element for full side chains.
    """
    name = atom_name[:4]
    if len(name) >= 4 or len(element.strip()) >= 2:
        return f"{name:<4}"
    return f" {name:<3}"
