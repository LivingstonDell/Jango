from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
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
from .config import RosettaRuntimeConfig
from .runtime.rosetta import RosettaRuntimeError
from .rosetta import build_interface_analyzer_command, interface_score_has_required_terms, parse_scorefile, rosetta_runtime_from_options
from .folding import Boltz2Backend, ESMFold2Backend, OpenDDEBackend, FoldJob, MSAConfig, MSAError, MSAMode, MSARequest, MSAResolution, MSAStatus, MSAUnsupportedError
from .folding.msa import resolve_msa_for_backend


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
SEQUENCE_ROLE_REDESIGNED_DECOY = "redesigned_decoy"
SEQUENCE_ROLE_NATIVE_CONTROL = "native_control"
MSA_FAILURE_POLICIES = ("abort", "skip_design", "skip_case")

@dataclass(frozen=True)
class DecoyDefaults:
    candidates_per_antigen: int = 150
    # ProteinMPNN explores sequence space. This explicit limit controls only how
    # many unique valid redesigned sequences are sent to folding. Native controls
    # and backend samples are separate concepts.
    max_decoy_sequences_per_structure: int = 5
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

    @property
    def refold_candidates_per_antigen(self) -> int:
        """Backward-compatible alias; new code uses max_decoy_sequences_per_structure."""

        return self.max_decoy_sequences_per_structure


PRODUCTION_DECOY_CONFIG = Path("configs/production/antigen_redesign_decoys.yml")


def write_default_decoy_config(path: Path = PRODUCTION_DECOY_CONFIG, defaults: DecoyDefaults = DecoyDefaults()) -> Path:
    del defaults
    raise RuntimeError(
        "Legacy decoy config generation is disabled; use the existing production config "
        f"({PRODUCTION_DECOY_CONFIG}) or pass --config explicitly. Refusing to write {path}."
    )

def build_decoy_cases(
    case_manifest_csv: Path = Path("results/tables/boltz2_case_manifest.csv"),
    raw_dir: Path = Path("data/raw/pdb"),
    out_csv: Path = Path("results/tables/decoy_redesign_case_manifest.csv"),
    config_path: Path = PRODUCTION_DECOY_CONFIG,
    limit: int | None = None,
) -> pd.DataFrame:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Decoy config does not exist: {config_path}. Use configs/production/antigen_redesign_decoys.yml "
            "or pass --config explicitly."
        )
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
                "requested_mpnn_sequences": DecoyDefaults().candidates_per_antigen,
                "max_decoy_sequences_per_structure": DecoyDefaults().max_decoy_sequences_per_structure,
            }
        )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

#added redesign mode to function inputs
def prepare_decoy_inputs(
    case_manifest_csv: Path = Path("results/tables/decoy_redesign_case_manifest.csv"),
    raw_dir: Path = Path("data/raw/pdb"),
    work_dir: Path = Path("work/decoys"),
    out_mpnn_csv: Path = Path("results/tables/decoy_mpnn_job_manifest.csv"),
    limit: int | None = None,
    redesign_mode: str = "full_antigen",
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

        if redesign_mode == "hotspot_only":
            forced = select_hotspots(ranked, len(native_seq))
            fraction = len(forced) / len(ranked) if ranked else 0.0
        else:
            fraction = adaptive_forced_fraction(len(ranked), len(native_seq))
            forced = select_forced_contacts(ranked, fraction)
        # added omit and fixed sections
        omit_jsonl = write_contact_omit_jsonl(
            pdb_path,
            forced,
            native_seq,
            input_dir / f"{structure_id}_omit_contacts.jsonl",
        )
        # added option for redesign mode
        fixed_jsonl = None
        if redesign_mode in {
            "interface_only",
            "hotspot_only",
        }:
            fixed_jsonl = write_fixed_positions_jsonl(
            pdb_path,
            len(native_seq),
            forced,
            input_dir / f"{structure_id}_fixed_noncontacts.jsonl",
        )
        output_dir = work_dir / "mpnn_outputs" / structure_id
        # resume
        rows.append(
            {
                "structure_id": structure_id,
                "antigen_pdb": str(pdb_path),
                "designed_chain": BOLTZ_CHAIN_ANTIGEN,
                "num_seq_per_target": DecoyDefaults().candidates_per_antigen,
                "requested_mpnn_sequences": DecoyDefaults().candidates_per_antigen,
                "max_decoy_sequences_per_structure": DecoyDefaults().max_decoy_sequences_per_structure,
                "contact_omit_jsonl": str(omit_jsonl),
                "n_contacts": len(ranked),
                "forced_contact_fraction": fraction,
                "n_forced_contact_mutations": len(forced),
                "forced_positions": positions_to_text(forced),
                "mpnn_output_dir": str(output_dir),
                #added redesign mode
                "redesign_mode": redesign_mode,
                "fixed_positions_jsonl": str(fixed_jsonl) if fixed_jsonl is not None else "",
                # added omit and fixed to mpnn command list
                "mpnn_command": proteinmpnn_command(
                    pdb_path,
                    output_dir,
                    DecoyDefaults().candidates_per_antigen,
                    DecoyDefaults().sampling_temp,
                    omit_jsonl,
                    fixed_jsonl,
                ),
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

#added fixed positions
def proteinmpnn_command(
    pdb_path: Path,
    output_dir: Path,
    candidates: int,
    sampling_temp: float = DecoyDefaults().sampling_temp,
    omit_jsonl: Path | None = None,
    fixed_positions_jsonl: Path | None = None,
) -> str:    
    cmd = (
        "python $PROTEINMPNN_HOME/protein_mpnn_run.py "
        f"--pdb_path {pdb_path} --pdb_path_chains {BOLTZ_CHAIN_ANTIGEN} "
        f"--out_folder {output_dir} --num_seq_per_target {candidates} --sampling_temp {sampling_temp}"
    )
    if omit_jsonl is not None:
        cmd += f" --omit_AA_jsonl {omit_jsonl}"
    if fixed_positions_jsonl is not None: #added fixed positions
        cmd += f" --fixed_positions_jsonl {fixed_positions_jsonl}"
    return cmd

# resume
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
#added hotspot functions
def hotspot_budget(antigen_len: int) -> int:
    if antigen_len < 80:
        return 3
    if antigen_len < 120:
        return 5
    if antigen_len < 200:
        return 8
    return 12

def select_hotspots(
    ranked_positions: list[int],
    antigen_len: int,
) -> list[int]:
    k = min(
        hotspot_budget(antigen_len),
        len(ranked_positions),
    )
    return sorted(ranked_positions[:k])
#resume
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

#add helper functions to omit antigen positions not in contact area
def fixed_positions_dict(pdb_stem: str, antigen_len: int, mutable_positions: list[int], chain: str = BOLTZ_CHAIN_ANTIGEN) -> dict:
    """ProteinMPNN fixed_positions_jsonl.

    Fix every antigen residue except mutable_positions.
    Positions are 1-indexed.
    """
    mutable = set(mutable_positions)
    fixed = [pos for pos in range(1, antigen_len + 1) if pos not in mutable]
    return {pdb_stem: {chain: fixed}}


def write_fixed_positions_jsonl(pdb_path: Path, antigen_len: int, mutable_positions: list[int], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixed_positions_dict(pdb_path.stem, antigen_len, mutable_positions)) + "\n")
    return out_path

#resume
def write_decoy_job_bundle(mpnn_manifest: pd.DataFrame, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    mpnn_manifest[["structure_id", "antigen_pdb", "mpnn_output_dir", "mpnn_command"]].to_csv(bundle_dir / "proteinmpnn_jobs.tsv", sep="\t", index=False)
    install = bundle_dir / "install_proteinmpnn_<CPU_NODE>.sh"
    install.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "INSTALL_DIR=${1:?Usage: install_proteinmpnn.sh /path/to/ProteinMPNN}",
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
                ": ${PROTEINMPNN_HOME:?Set PROTEINMPNN_HOME to the ProteinMPNN repository root}",
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
    fold_backend: str = "boltz2",
    esmfold2_python: Path | None = None,
    esmfold2_root: Path | None = None,
    esmfold2_model_id_or_path: str | None = None,
    esmfold2_cache_dir: Path | None = None,
    esmfold2_device: str | None = None,
    esmfold2_num_sampling_steps: int | None = None,
    esmfold2_num_diffusion_samples: int | None = None,
    esmfold2_seed: int | None = None,
    opendde_python: Path | None = None,
    opendde_executable: Path | None = None,
    opendde_root_dir: Path | None = None,
    opendde_model_name: str | None = None,
    opendde_checkpoint: Path | None = None,
    opendde_seeds: str | None = None,
    opendde_cycle: str | None = None,
    opendde_step: str | None = None,
    opendde_samples: str | None = None,
    opendde_dtype: str | None = None,
    msa_mode: str = "disabled",
    msa_provider: str | None = None,
    msa_cache_dir: Path | None = None,
    msa_input: Path | None = None,
    msa_format: str | None = None,
    msa_database: str | None = None,
    msa_max_sequences: int | None = None,
    msa_pairing: str = "none",
    msa_reuse_policy: str = "exact_sequence",
    msa_script: Path | None = None,
    msa_tool: str | None = None,
    msa_build_if_missing: bool = False,
    msa_failure_policy: str = "abort",
    allow_single_sequence_fallback: bool = False,
    experiment_namespace: str | None = None,
    max_decoy_sequences_per_structure: int = DecoyDefaults().max_decoy_sequences_per_structure,
    mode_policy: str | None = None,
    validation_phase: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(case_manifest_csv)
    if max_decoy_sequences_per_structure <= 0:
        raise ValueError("max_decoy_sequences_per_structure must be positive")
    if msa_failure_policy not in MSA_FAILURE_POLICIES:
        allowed = ", ".join(MSA_FAILURE_POLICIES)
        raise ValueError(f"Invalid MSA failure policy: {msa_failure_policy!r}. Allowed values: {allowed}")
    if validation_phase not in {"full", "sequence_only"}:
        raise ValueError(f"Invalid validation phase: {validation_phase!r}")

    msa_config = MSAConfig.from_values(
        mode=msa_mode,
        provider=msa_provider,
        cache_dir=msa_cache_dir,
        input_path=msa_input,
        format=msa_format,
        database=msa_database,
        max_sequences=msa_max_sequences,
        pairing=msa_pairing,
        reuse_policy=msa_reuse_policy,
        script_path=msa_script,
        tool=msa_tool,
        build_if_missing=msa_build_if_missing,
        allow_single_sequence_fallback=allow_single_sequence_fallback,
    )

    if fold_backend == "boltz2":
        backend = Boltz2Backend(use_msa_server=msa_config.mode == MSAMode.BACKEND_MANAGED)
    elif fold_backend == "esmfold2":
        backend = ESMFold2Backend(
            esmfold2_python=esmfold2_python,
            esm_root=esmfold2_root,
            model_id_or_path=esmfold2_model_id_or_path,
            cache_dir=esmfold2_cache_dir,
            device=esmfold2_device,
            num_sampling_steps=esmfold2_num_sampling_steps,
            num_diffusion_samples=esmfold2_num_diffusion_samples,
            seed=esmfold2_seed,
        )
    elif fold_backend == "opendde":
        backend = OpenDDEBackend(
            opendde_python=opendde_python,
            opendde_executable=opendde_executable,
            root_dir=opendde_root_dir,
            model_name=opendde_model_name,
            checkpoint=opendde_checkpoint,
            seeds=opendde_seeds,
            cycle=opendde_cycle,
            step=opendde_step,
            samples=opendde_samples,
            dtype=opendde_dtype,
        )
    else:
        raise ValueError(f"Unsupported fold backend: {fold_backend}")

    def make_msa_request(row: dict[str, object], case: dict[str, object], sequence: str) -> MSARequest:
        native_sequence = str(case.get("native_antigen_sequence", ""))
        return MSARequest(
            structure_id=str(row.get("structure_id") or case.get("structure_id")),
            design_id=str(row.get("design_id")),
            sequence=sequence,
            backend=backend.name,
            provider=msa_config.provider,
            format=msa_config.format,
            chain_ids=(BOLTZ_CHAIN_ANTIGEN,),
            original_chain_ids=(str(case.get("antigen_chain", "")),),
            remapped_chain_ids=(BOLTZ_CHAIN_ANTIGEN,),
            pairing=msa_config.pairing,
            source_database=msa_config.database,
            max_sequences=msa_config.max_sequences,
            reuse_policy=msa_config.reuse_policy,
            mode=msa_config.mode,
            generation_parameters={
                "sequence_role": str(row.get("sequence_role", "")),
                "native_sequence": native_sequence,
                "native_sequence_hash": sequence_hash(native_sequence) if native_sequence else "",
                "design_sequence_hash": sequence_hash(sequence),
                "derived_msa_dir": str(work_dir / "msa" / "native_reuse"),
            },
        )

    def _value_is_true(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        return str(value).strip().lower() not in {"", "0", "false", "no", "n"}

    def _infer_msa_failure_status(exc: MSAError) -> MSAStatus:
        if isinstance(exc, MSAUnsupportedError):
            return MSAStatus.UNSUPPORTED
        message = str(exc).lower()
        if any(token in message for token in ("not found", "does not exist", "missing")):
            return MSAStatus.MISSING
        return MSAStatus.INVALID

    def _infer_msa_error_path(message: str) -> str:
        match = re.search(r":\s*(/[^\n]+?)\s*$", message)
        return match.group(1) if match else ""

    def msa_failure_fields(
        request: MSARequest,
        exc: MSAError,
        *,
        reason: str = "msa_resolution_failed",
        error_message: str | None = None,
    ) -> dict[str, object]:
        message = error_message or str(exc)
        status = _infer_msa_failure_status(exc)
        fields = MSAResolution(status=status, config=msa_config, request=request, message=message).to_manifest_fields(base_dir=work_dir)
        inferred_path = _infer_msa_error_path(message)
        if inferred_path:
            fields["msa_path"] = inferred_path
            try:
                fields["msa_relative_path"] = str(Path(inferred_path).expanduser().resolve().relative_to(work_dir.expanduser().resolve()))
            except ValueError:
                fields["msa_relative_path"] = ""
        fields.update(
            {
                "msa_error": message,
                "msa_cache_status": status.value,
                "msa_cache_hit": False,
                "msa_model_consumed": False,
                "fold_eligible": False,
                "fold_exclusion_reason": reason,
            }
        )
        return fields

    def resolve_fold_msa(row: dict[str, object], case: dict[str, object], sequence: str):
        request = make_msa_request(row, case, sequence)
        try:
            resolution = resolve_msa_for_backend(
                config=msa_config,
                support=backend.msa_support,
                request=request,
                backend_name=backend.name,
            )
        except MSAError as exc:
            if msa_failure_policy == "abort":
                raise
            return msa_failure_fields(request, exc), None
        fields = resolution.to_manifest_fields(base_dir=work_dir)
        fields.update({"msa_error": "", "fold_eligible": True, "fold_exclusion_reason": ""})
        return fields, resolution.artifact

    def eligible_fold_and_validation_rows(
        row: dict[str, object],
        case: dict[str, object],
        sequence: str,
        msa_fields: dict[str, object],
        msa_artifact,
    ) -> tuple[dict[str, object], dict[str, object]]:
        fold_input, output_dir = backend.write_input(
            design_id=str(row["design_id"]),
            structure_id=str(row["structure_id"]),
            sequence=sequence,
            work_dir=work_dir,
            msa_artifact=msa_artifact,
        )
        fold_cmd = backend.command(fold_input, output_dir)
        row_fields = {**msa_fields, "msa_error": "", "fold_eligible": True, "fold_exclusion_reason": ""}
        fold_row = {
            **row,
            "fold_backend": backend.name,
            "fold_input": str(fold_input),
            "fold_output_dir": str(output_dir),
            "fold_command": fold_cmd,
            **row_fields,
        }
        validation_row = validate_folded_monomer(row, case, raw_dir, output_dir, backend)
        validation_row.update(row_fields)
        return fold_row, validation_row

    def ineligible_fold_row(row: dict[str, object], msa_fields: dict[str, object]) -> dict[str, object]:
        return {
            **row,
            "fold_backend": backend.name,
            "fold_input": "",
            "fold_output_dir": "",
            "fold_command": "",
            **msa_fields,
        }

    def ineligible_validation_row(row: dict[str, object], msa_fields: dict[str, object]) -> dict[str, object]:
        base = {
            k: row.get(k)
            for k in [
                "design_id",
                "structure_id",
                "designed_sequence",
                "sequence",
                "sequence_hash",
                "sequence_role",
                "is_native_reference",
                "mpnn_score",
                "mpnn_rank",
                "sequence_identity",
                "contact_mutation_fraction",
                "experiment_namespace",
                "redesign_mode",
                "mode_policy",
                "requested_mpnn_sequences",
                "max_decoy_sequences_for_folding",
                "actual_decoy_sequences_for_folding",
                "native_controls_selected_for_folding",
                "fold_job_id",
            ]
        }
        return {
            **base,
            "fold_backend": backend.name,
            "validation_status": "msa_ineligible",
            "fold_output_dir": "",
            **msa_fields,
        }

    mpnn_manifest_path = tables_dir / "decoy_mpnn_job_manifest.csv"
    if mpnn_manifest_path.exists():
        mpnn_manifest = pd.read_csv(mpnn_manifest_path).set_index("structure_id")
    else:
        mpnn_manifest = pd.DataFrame()

    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for case in cases.to_dict("records"):
        structure_id = str(case["structure_id"])
        native_seq = str(case["native_antigen_sequence"])
        contact_positions = parse_positions(str(case.get("antigen_contact_positions", "")))
        if structure_id in mpnn_manifest.index:
            mpnn_info = mpnn_manifest.loc[structure_id]
            if isinstance(mpnn_info, pd.DataFrame):
                mpnn_info = mpnn_info.iloc[0]
        else:
            mpnn_info = pd.Series(dtype=object)
        forced_fraction = float(mpnn_info.get("forced_contact_fraction", 1.0))
        requested_mpnn_sequences = int(float(mpnn_info.get("requested_mpnn_sequences", mpnn_info.get("num_seq_per_target", case.get("requested_mpnn_sequences", DecoyDefaults().candidates_per_antigen)))))
        redesign_mode = str(mpnn_info.get("redesign_mode", case.get("redesign_mode", case.get("selected_redesign_mode", ""))))
        row_mode_policy = str(mode_policy or case.get("mode_policy", ""))
        candidates = parse_mpnn_candidates(work_dir / "mpnn_outputs" / structure_id, structure_id)

        rows_for_case, selected = select_mpnn_redesign_candidates(
            candidates,
            native_seq,
            contact_positions,
            forced_contact_fraction=forced_fraction,
            max_decoy_sequences_per_structure=max_decoy_sequences_per_structure,
        )
        actual_decoys_for_folding = len(selected)
        native_controls_for_folding = 1 if selected else 0
        provenance = {
            "experiment_namespace": experiment_namespace or "",
            "redesign_mode": redesign_mode,
            "mode_policy": row_mode_policy,
            "requested_mpnn_sequences": requested_mpnn_sequences,
            "max_decoy_sequences_for_folding": max_decoy_sequences_per_structure,
            "actual_decoy_sequences_for_folding": actual_decoys_for_folding,
            "native_controls_selected_for_folding": native_controls_for_folding,
        }
        for row in rows_for_case:
            row.update(provenance)
        candidate_rows.extend(rows_for_case)

        fold_candidates_for_case: list[tuple[dict[str, object], str]] = []
        if selected:
            native_row = {
                "design_id": f"{structure_id}_native",
                "structure_id": structure_id,
                "designed_sequence": native_seq,
                "sequence": native_seq,
                "sequence_hash": sequence_hash(native_seq),
                "sequence_role": SEQUENCE_ROLE_NATIVE_CONTROL,
                "is_native_reference": True,
                "mpnn_score": np.nan,
                "mpnn_rank": np.nan,
                "selected_for_folding": True,
                "fold_job_id": f"{structure_id}__{redesign_mode or 'mode'}__{backend.name}__native_control",
                **provenance,
            }
            fold_candidates_for_case.append((native_row, native_seq))

        for row in selected:
            sequence = str(row["designed_sequence"])
            mpnn_rank = int(row["mpnn_rank"])
            row.update({**provenance, "sequence_role": SEQUENCE_ROLE_REDESIGNED_DECOY, "is_native_reference": False, "fold_job_id": f"{structure_id}__{redesign_mode or 'mode'}__{backend.name}__mpnn_rank_{mpnn_rank:04d}"})
            fold_candidates_for_case.append((row, sequence))

        if fold_candidates_for_case:
            for row, _sequence in fold_candidates_for_case:
                selected_rows.append(dict(row))
            if validation_phase == "sequence_only":
                continue
            case_fold_rows: list[dict[str, object]] = []
            case_validation_rows: list[dict[str, object]] = []
            if msa_failure_policy == "skip_case":
                prepared: list[tuple[dict[str, object], str, dict[str, object], object]] = []
                failure: tuple[dict[str, object], str, dict[str, object]] | None = None
                for row, sequence in fold_candidates_for_case:
                    msa_fields, msa_artifact = resolve_fold_msa(row, case, sequence)
                    if not _value_is_true(msa_fields.get("fold_eligible", True)):
                        failure = (row, sequence, msa_fields)
                        break
                    prepared.append((row, sequence, msa_fields, msa_artifact))
                if failure is None:
                    for row, sequence, msa_fields, msa_artifact in prepared:
                        fold_row, validation_row = eligible_fold_and_validation_rows(row, case, sequence, msa_fields, msa_artifact)
                        case_fold_rows.append(fold_row)
                        case_validation_rows.append(validation_row)
                else:
                    failed_row, _, failed_fields = failure
                    failed_design_id = str(failed_row.get("design_id", "unknown"))
                    failed_error = str(failed_fields.get("msa_error", "MSA resolution failed"))
                    for row, sequence in fold_candidates_for_case:
                        if row is failed_row:
                            fields = failed_fields
                        else:
                            request = make_msa_request(row, case, sequence)
                            fields = msa_failure_fields(
                                request,
                                MSAError(failed_error),
                                reason="msa_failure_in_case",
                                error_message=f"Excluded by MSA_FAILURE_POLICY=skip_case because {failed_design_id} failed required MSA resolution.",
                            )
                        case_fold_rows.append(ineligible_fold_row(row, fields))
                        case_validation_rows.append(ineligible_validation_row(row, fields))
            else:
                for row, sequence in fold_candidates_for_case:
                    msa_fields, msa_artifact = resolve_fold_msa(row, case, sequence)
                    if _value_is_true(msa_fields.get("fold_eligible", True)):
                        fold_row, validation_row = eligible_fold_and_validation_rows(row, case, sequence, msa_fields, msa_artifact)
                    else:
                        fold_row = ineligible_fold_row(row, msa_fields)
                        validation_row = ineligible_validation_row(row, msa_fields)
                    case_fold_rows.append(fold_row)
                    case_validation_rows.append(validation_row)
            fold_rows.extend(case_fold_rows)
            validation_rows.extend(case_validation_rows)

        if not candidates:
            validation_rows.append({"structure_id": structure_id, "fold_backend": backend.name, "validation_status": "missing_mpnn_output", **provenance})

    candidates_df = pd.DataFrame(candidate_rows)
    selected_df = pd.DataFrame(selected_rows)
    fold_df = pd.DataFrame(fold_rows)
    validation_df = pd.DataFrame(validation_rows)

    if not validation_df.empty and "validation_status" in validation_df:
        validation_df["selected_decoy"] = False
        validation_df["selection_rank"] = np.nan
        if "design_id" in validation_df.columns:
            if "sequence_role" in validation_df.columns:
                designed_mask = validation_df["sequence_role"].astype(str).eq(SEQUENCE_ROLE_REDESIGNED_DECOY)
            else:
                designed_mask = ~validation_df.get("is_native_reference", pd.Series(False, index=validation_df.index)).fillna(False).astype(bool)
            ok = validation_df[(validation_df["validation_status"] == "pass") & designed_mask].copy()
            group_cols = ["structure_id"] + (["redesign_mode"] if "redesign_mode" in ok.columns else [])
            sort_cols = [col for col in ["monomer_ca_rmsd", "mpnn_score", "design_id"] if col in ok]
            for _, group in ok.groupby(group_cols, dropna=False, sort=False):
                ranked = group.sort_values(sort_cols) if sort_cols else group
                for rank, (idx, _) in enumerate(ranked.head(DecoyDefaults().passing_designs_per_antigen).iterrows(), start=1):
                    validation_df.at[idx, "selected_decoy"] = True
                    validation_df.at[idx, "selection_rank"] = rank

    summary_df = summarize_decoy_validation(cases, candidates_df, validation_df)
    exclusion_columns = ["structure_id", "design_id", "sequence_role", "fold_exclusion_reason", "msa_status", "msa_error", "excluded_count"]
    if fold_df.empty or "fold_eligible" not in fold_df.columns:
        exclusion_df = pd.DataFrame(columns=exclusion_columns)
    else:
        ineligible = fold_df[~fold_df["fold_eligible"].map(_value_is_true)].copy()
        if ineligible.empty:
            exclusion_df = pd.DataFrame(columns=exclusion_columns)
        else:
            exclusion_df = (
                ineligible.groupby(exclusion_columns[:-1], dropna=False)
                .size()
                .reset_index(name="excluded_count")
                .reindex(columns=exclusion_columns)
            )
    tables_dir.mkdir(parents=True, exist_ok=True)

    candidate_columns = ["design_id", "structure_id", "mpnn_source", "mpnn_header", "mpnn_score", "mpnn_rank", "designed_sequence", "sequence", "sequence_hash", "sequence_filter_status", "sequence_valid", "selected_for_folding", "sequence_role", "sequence_identity", "contact_mutation_fraction", "mutated_contact_positions", "experiment_namespace", "redesign_mode", "mode_policy", "requested_mpnn_sequences", "max_decoy_sequences_for_folding", "actual_decoy_sequences_for_folding", "native_controls_selected_for_folding"]
    fold_columns = ["experiment_namespace", "structure_id", "design_id", "redesign_mode", "mode_policy", "sequence_role", "sequence", "designed_sequence", "sequence_hash", "mpnn_score", "mpnn_rank", "requested_mpnn_sequences", "max_decoy_sequences_for_folding", "actual_decoy_sequences_for_folding", "native_controls_selected_for_folding", "fold_backend", "fold_job_id", "fold_input", "fold_output_dir", "fold_command", "fold_eligible", "fold_exclusion_reason", "sequence_identity", "contact_mutation_fraction", "msa_status", "msa_error", "msa_mode", "msa_provider", "msa_artifact_id", "msa_path", "msa_relative_path", "msa_format", "msa_pairing", "msa_kind", "msa_sequence_count", "msa_depth", "msa_checksum", "msa_query_sequence_hash", "msa_request_hash", "msa_cache_status", "msa_cache_hit", "msa_model_consumed", "msa_consumed", "msa_source", "msa_native_sequence_hash", "msa_design_sequence_hash", "msa_native_a3m_path", "msa_derived_a3m_path", "msa_mutation_count", "msa_query_replacement_status", "msa_length_match_status", "msa_native_cache_hit", "msa_provenance_json", "msa_config_json"]
    selected_columns = ["experiment_namespace", "structure_id", "design_id", "redesign_mode", "mode_policy", "sequence_role", "sequence", "designed_sequence", "sequence_hash", "mpnn_score", "mpnn_rank", "selected_for_folding", "is_native_reference", "fold_job_id", "requested_mpnn_sequences", "max_decoy_sequences_for_folding", "actual_decoy_sequences_for_folding", "native_controls_selected_for_folding"]
    selected_manifest_path = tables_dir / "decoy_selected_designs.csv"
    upstream_selected_df = pd.read_csv(selected_manifest_path) if validation_phase == "full" and selected_manifest_path.exists() else pd.DataFrame()
    if candidates_df.empty:
        candidates_df = pd.DataFrame(columns=candidate_columns)
    else:
        candidates_df = candidates_df.reindex(columns=[*candidate_columns, *[c for c in candidates_df.columns if c not in candidate_columns]])
    if selected_df.empty:
        selected_df = pd.DataFrame(columns=selected_columns)
    else:
        selected_df = selected_df.reindex(columns=[*selected_columns, *[c for c in selected_df.columns if c not in selected_columns]])
    if fold_df.empty:
        fold_df = pd.DataFrame(columns=fold_columns)
    else:
        fold_df = fold_df.reindex(columns=[*fold_columns, *[c for c in fold_df.columns if c not in fold_columns]])

    if validation_phase == "full" and not upstream_selected_df.empty and "design_id" in upstream_selected_df.columns:
        upstream_ids = set(upstream_selected_df["design_id"].astype(str))
        current_ids = set(selected_df.get("design_id", pd.Series(dtype=str)).astype(str))
        if upstream_ids != current_ids:
            missing = sorted(upstream_ids - current_ids)[:10]
            extra = sorted(current_ids - upstream_ids)[:10]
            raise ValueError(f"Selected-design manifest mismatch before MSA resolution; missing={missing}, extra={extra}")

    candidates_df.to_csv(tables_dir / "decoy_mpnn_candidates.csv", index=False)
    selected_df.to_csv(selected_manifest_path, index=False)
    if validation_phase == "sequence_only":
        summary_df.to_csv(tables_dir / "decoy_sequence_summary.csv", index=False)
        return candidates_df, selected_df, validation_df

    fold_df.to_csv(tables_dir / f"decoy_{backend.name}_monomer_manifest.csv", index=False)
    validation_df.to_csv(tables_dir / "decoy_validation.csv", index=False)
    summary_df.to_csv(tables_dir / "decoy_summary.csv", index=False)
    exclusion_df.to_csv(tables_dir / "decoy_msa_exclusion_summary.csv", index=False)
    msa_status_columns = [col for col in ["experiment_namespace", "structure_id", "design_id", "redesign_mode", "mode_policy", "sequence_role", "msa_source", "msa_status", "msa_path", "msa_cache_hit", "msa_native_cache_hit", "msa_derived_a3m_path", "fold_eligible", "fold_exclusion_reason", "msa_error"] if col in fold_df.columns]
    fold_df.reindex(columns=msa_status_columns).to_csv(tables_dir / "decoy_msa_resolution_status.csv", index=False)

    eligible_fold_df = fold_df
    if "fold_eligible" in eligible_fold_df.columns:
        eligible_fold_df = eligible_fold_df[eligible_fold_df["fold_eligible"].map(_value_is_true)]

    jobs = [
        FoldJob(
            design_id=str(row["design_id"]),
            structure_id=str(row["structure_id"]),
            sequence=str(row.get("sequence", row.get("designed_sequence", ""))),
            input_path=Path(str(row["fold_input"])),
            output_dir=Path(str(row["fold_output_dir"])),
            command=str(row["fold_command"]),
            backend=backend.name,
            metadata=dict(row),
        )
        for row in eligible_fold_df.to_dict("records")
    ]
    backend.write_job_bundle(jobs, work_dir / "job_bundle")

    return candidates_df, fold_df, validation_df
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


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _mpnn_rank_key(row: dict[str, object]) -> tuple[bool, float, str, str, str]:
    score = row.get("mpnn_score")
    try:
        score_value = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        score_value = float("nan")
    score_missing = pd.isna(score_value)
    return (
        bool(score_missing),
        score_value if not score_missing else float("inf"),
        str(row.get("design_id", "")),
        str(row.get("mpnn_source", "")),
        str(row.get("mpnn_header", "")),
    )


def select_mpnn_redesign_candidates(
    candidates: list[dict[str, object]],
    native_seq: str,
    contact_positions: list[int],
    *,
    forced_contact_fraction: float = 1.0,
    max_decoy_sequences_per_structure: int = DecoyDefaults().max_decoy_sequences_per_structure,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Filter, deduplicate, rank, and select redesigned ProteinMPNN sequences."""

    if max_decoy_sequences_per_structure <= 0:
        raise ValueError("max_decoy_sequences_per_structure must be positive")

    candidate_rows: list[dict[str, object]] = []
    unique_valid_rows: list[dict[str, object]] = []
    seen_valid_sequences: set[str] = set()
    native_seq = str(native_seq).upper()

    for candidate in candidates:
        seq = str(candidate.get("designed_sequence", "")).replace("/", "").upper()
        seq_status = sequence_filter_status(seq, native_seq, contact_positions, forced_contact_fraction=forced_contact_fraction)
        original_status = str(seq_status["sequence_filter_status"])
        row = {
            **candidate,
            **seq_status,
            "designed_sequence": seq,
            "sequence": seq,
            "sequence_hash": sequence_hash(seq),
            "sequence_valid": original_status == "ok",
            "selected_for_folding": False,
            "sequence_role": "",
            "mpnn_rank": np.nan,
        }
        if seq == native_seq:
            row["sequence_filter_status"] = "native_sequence"
            row["sequence_valid"] = False
        elif original_status != "ok":
            row["sequence_valid"] = False
        elif seq in seen_valid_sequences:
            row["sequence_filter_status"] = "duplicate"
        else:
            seen_valid_sequences.add(seq)
            unique_valid_rows.append(row)
        candidate_rows.append(row)

    ranked = sorted(unique_valid_rows, key=_mpnn_rank_key)
    for rank, row in enumerate(ranked, start=1):
        row["mpnn_rank"] = rank
    selected = ranked[:max_decoy_sequences_per_structure]
    for row in selected:
        row["selected_for_folding"] = True
        row["sequence_role"] = SEQUENCE_ROLE_REDESIGNED_DECOY
    return candidate_rows, selected

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
        f"${{BOLTZ_EXECUTABLE:-${{BOLTZ:-boltz}}}} predict {yaml_path} --model boltz2 --use_msa_server --use_potentials "
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
                "BOLTZ=${BOLTZ_EXECUTABLE:-${BOLTZ:-boltz}}",
                "CACHE_ARGS=()",
                "if [ -n \"${BOLTZ_CACHE:-}\" ]; then",
                "  mkdir -p \"$BOLTZ_CACHE\"",
                "  CACHE_ARGS=(--cache \"$BOLTZ_CACHE\")",
                "fi",
                "JOBS_TSV=${1:-work/decoys/job_bundle/boltz2_monomer_jobs.tsv}",
                "tail -n +2 \"$JOBS_TSV\" | while IFS=$'\\t' read -r design_id structure_id yaml_path output_dir boltz_command; do",
                "  mkdir -p \"$output_dir\"",
                "  echo \"[$design_id] $boltz_command\"",
                "  $BOLTZ predict \"${CACHE_ARGS[@]}\" \"$yaml_path\" --model boltz2 --use_msa_server --use_potentials --diffusion_samples 5 --recycling_steps 3 --output_format pdb --write_full_pae --write_full_pde --out_dir \"$output_dir\"",
                "done",
                "",
            ]
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)



def validate_folded_monomer(
    candidate: dict[str, object],
    case: dict[str, object],
    raw_dir: Path,
    output_dir: Path,
    backend,
) -> dict[str, object]:
    base = {
        k: candidate.get(k)
        for k in [
            "design_id",
            "structure_id",
            "designed_sequence",
            "sequence",
            "sequence_hash",
            "sequence_role",
            "is_native_reference",
            "mpnn_score",
            "mpnn_rank",
            "sequence_identity",
            "contact_mutation_fraction",
            "experiment_namespace",
            "redesign_mode",
            "mode_policy",
            "requested_mpnn_sequences",
            "max_decoy_sequences_for_folding",
            "actual_decoy_sequences_for_folding",
            "native_controls_selected_for_folding",
            "fold_job_id",
        ]
    }
    base["fold_backend"] = backend.name

    predictions = backend.discover_predictions(output_dir)
    if not predictions:
        return {**base, "validation_status": f"missing_{backend.name}_output", "fold_output_dir": str(output_dir)}

    structure_id = str(case["structure_id"])
    crystal_ag = remap_atoms(
        parse_atoms(read_pdb_text(raw_dir / str(case["source_filename"]))),
        {str(case["antigen_chain"]): BOLTZ_CHAIN_ANTIGEN},
    )

    native_refold_preds = backend.discover_predictions(output_dir.parent / f"{structure_id}_native")
    native_ref = (
        normalize_monomer_chain(parse_atoms(read_pdb_text(native_refold_preds[0].prediction_path)))
        if native_refold_preds
        else None
    )
    reference = native_ref if native_ref is not None else crystal_ag
    fold_reference = "native_refold" if native_ref is not None else "crystal_fallback"

    if candidate.get("is_native_reference"):
        return {
            **base,
            **predictions[0].confidence,
            "fold_reference": "self",
            "prediction_path": str(predictions[0].prediction_path),
            "n_fold_predictions": len(predictions),
            "validation_status": "native_reference",
        }

    best = max(
        (
            {
                **monomer_validation_metrics(
                    reference,
                    normalize_monomer_chain(parse_atoms(read_pdb_text(pred.prediction_path))),
                ),
                **pred.confidence,
                "prediction_path": str(pred.prediction_path),
                "_pred": pred.prediction_path,
            }
            for pred in predictions
        ),
        key=lambda m: (m["monomer_tm_score"] if not np.isnan(m["monomer_tm_score"]) else -1.0),
    )

    pred_path = best.pop("_pred")
    crystal_metrics = monomer_validation_metrics(
        crystal_ag,
        normalize_monomer_chain(parse_atoms(read_pdb_text(pred_path))),
    )
    best["monomer_tm_score_vs_crystal"] = crystal_metrics["monomer_tm_score"]
    best["monomer_ca_rmsd_vs_crystal"] = crystal_metrics["monomer_ca_rmsd"]

    passed = (
        best["monomer_tm_score"] >= DecoyDefaults().monomer_tm_score_min
        and best["monomer_aligned_fraction"] >= DecoyDefaults().monomer_aligned_fraction_min
    )

    return {
        **base,
        **best,
        "fold_reference": fold_reference,
        "n_fold_predictions": len(predictions),
        "validation_status": "pass" if passed else "failed_fold",
    }

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
    def _first(frame: pd.DataFrame, column: str, default: object = np.nan) -> object:
        if frame.empty or column not in frame.columns:
            return default
        values = frame[column].dropna()
        return values.iloc[0] if not values.empty else default

    def _true_count(frame: pd.DataFrame, column: str) -> int:
        if frame.empty or column not in frame.columns:
            return 0
        values = frame[column]
        if values.dtype == bool:
            return int(values.fillna(False).sum())
        return int(values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}).sum())

    rows = []
    for case in cases.to_dict("records"):
        sid = str(case["structure_id"])
        c = candidates[candidates.get("structure_id", pd.Series(dtype=str)) == sid] if not candidates.empty else pd.DataFrame()
        v = validation[validation.get("structure_id", pd.Series(dtype=str)) == sid] if not validation.empty else pd.DataFrame()
        c_status = c.get("sequence_filter_status", pd.Series(dtype=str)).astype(str) if not c.empty else pd.Series(dtype=str)
        v_status = v.get("validation_status", pd.Series(dtype=str)).astype(str) if not v.empty else pd.Series(dtype=str)
        v_role = v.get("sequence_role", pd.Series(dtype=str)).astype(str) if not v.empty else pd.Series(dtype=str)
        redesigned_mask = v_role.eq(SEQUENCE_ROLE_REDESIGNED_DECOY) if not v.empty else pd.Series(dtype=bool)
        native_mask = v_role.eq(SEQUENCE_ROLE_NATIVE_CONTROL) if not v.empty else pd.Series(dtype=bool)
        fold_predictions = pd.to_numeric(v.get("n_fold_predictions", pd.Series(dtype=float)), errors="coerce").fillna(0) if not v.empty else pd.Series(dtype=float)

        selected_decoys = 0
        if not v.empty and "selected_decoy" in v.columns:
            selected_mask = v["selected_decoy"].fillna(False)
            if selected_mask.dtype != bool:
                selected_mask = selected_mask.astype(str).str.lower().isin({"true", "1", "yes"})
            selected_decoys = int((selected_mask & redesigned_mask).sum())

        rows.append(
            {
                "structure_id": sid,
                "requested_mpnn_sequences": _first(c, "requested_mpnn_sequences", _first(pd.DataFrame([case]), "requested_mpnn_sequences", DecoyDefaults().candidates_per_antigen)),
                "max_decoy_sequences_for_folding": _first(c, "max_decoy_sequences_for_folding", DecoyDefaults().max_decoy_sequences_per_structure),
                "mpnn_candidates": len(c),
                "mpnn_candidates_parsed": len(c),
                "sequence_filter_ok": int(c_status.eq("ok").sum()) if not c.empty else 0,
                "valid_unique_redesigns": int(c_status.eq("ok").sum()) if not c.empty else 0,
                "duplicate_sequences": int(c_status.eq("duplicate").sum()) if not c.empty else 0,
                "native_sequence_candidates": int(c_status.eq("native_sequence").sum()) if not c.empty else 0,
                "invalid_sequences": int((~c_status.isin(["ok", "duplicate", "native_sequence"])).sum()) if not c.empty else 0,
                "selected_redesign_sequences_for_folding": _true_count(c, "selected_for_folding"),
                "native_controls_selected_for_folding": int(native_mask.sum()) if not v.empty else int(_first(c, "native_controls_selected_for_folding", 0) or 0),
                "fold_jobs": len(v),
                "redesign_fold_jobs": int(redesigned_mask.sum()) if not v.empty else 0,
                "native_control_fold_jobs": int(native_mask.sum()) if not v.empty else 0,
                "backend_predictions": int(fold_predictions.sum()) if not fold_predictions.empty else 0,
                "fold_pass": int((v_status.eq("pass") & redesigned_mask).sum()) if not v.empty else 0,
                "native_control_status": ";".join(sorted(set(v_status[native_mask]))) if not v.empty and native_mask.any() else "",
                "selected_decoys": selected_decoys,
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
    selected = pd.DataFrame()
    if not validation.empty:
        selected_mask = validation.get("selected_decoy", pd.Series(False, index=validation.index)).fillna(False)
        if selected_mask.dtype != bool:
            selected_mask = selected_mask.astype(str).str.lower().isin({"true", "1", "yes"})
        if "sequence_role" in validation.columns:
            selected_mask &= validation["sequence_role"].astype(str).eq(SEQUENCE_ROLE_REDESIGNED_DECOY)
        if "is_native_reference" in validation.columns:
            native_mask = validation["is_native_reference"].fillna(False)
            if native_mask.dtype != bool:
                native_mask = native_mask.astype(str).str.lower().isin({"true", "1", "yes"})
            selected_mask &= ~native_mask
        if "validation_status" in validation.columns:
            selected_mask &= ~validation["validation_status"].astype(str).eq("native_reference")
        selected = validation[selected_mask]
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
    docker_image: str | None = "rosettacommons/rosetta:latest",
    native_rosetta_csv: Path | None = None,
    native_protein_interface_csv: Path | None = None,
    native_shape_complementarity_csv: Path | None = None,
    runtime_config: RosettaRuntimeConfig | None = None,
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
        [run_decoy_rosetta(p, r, work_dir, docker_image, runtime_config) for p, r in zip(pdb_paths, rows)] if run_rosetta
        else [{"design_id": d, "structure_id": s, "rosetta_status": "not_run"} for d, s in zip(design_ids, structure_ids)]
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    for df, name in [(pi, "decoy_protein_interface.csv"), (sc, "decoy_shape_complementarity.csv"), (rosetta, "decoy_rosetta_interface.csv")]:
        (df if not df.empty else pd.DataFrame(columns=["design_id", "structure_id"])).to_csv(tables_dir / name, index=False)

    comparison = build_decoy_native_comparison(
        pi,
        sc,
        rosetta,
        tables_dir,
        native_rosetta_csv=native_rosetta_csv,
        native_protein_interface_csv=native_protein_interface_csv,
        native_shape_complementarity_csv=native_shape_complementarity_csv,
    )
    if comparison.empty:
        comparison = pd.DataFrame(columns=["design_id", "structure_id"])
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


def build_decoy_native_comparison(
    pi: pd.DataFrame,
    sc: pd.DataFrame,
    rosetta: pd.DataFrame,
    tables_dir: Path,
    native_rosetta_csv: Path | None = None,
    native_protein_interface_csv: Path | None = None,
    native_shape_complementarity_csv: Path | None = None,
) -> pd.DataFrame:
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
    native_shape_path = native_shape_complementarity_csv or tables_dir / "native_shape_complementarity.csv"
    native_rosetta_path = native_rosetta_csv or tables_dir / "rosetta_interface_native.csv"
    native_protein_path = native_protein_interface_csv or tables_dir / "native_protein_interface.csv"
    base = _join_native_baseline(base, native_shape_path, cols=["sc", "median_distance", "trimmed_area"])
    base = _join_native_baseline(base, native_rosetta_path, cols=ROSETTA_NATIVE_BASELINE_COLUMNS)
    base = _join_native_baseline(base, native_protein_path)
    _validate_rosetta_native_columns(base, native_rosetta_path)
    return base


ROSETTA_NATIVE_BASELINE_COLUMNS = ["dG_separated", "dG_separated/dSASAx100", "dSASA_int", "sc_value", "hbonds_int", "delta_unsatHbonds", "packstat"]
REQUIRED_ROSETTA_NATIVE_COMPARISON_COLUMNS = [
    "native_dG_separated",
    "native_dSASA_int",
    "native_sc_value",
    "native_hbonds_int",
    "native_delta_unsatHbonds",
]


def _validate_rosetta_native_columns(comparison: pd.DataFrame, native_csv: Path) -> None:
    if comparison.empty or "structure_id" not in comparison.columns or not native_csv.exists():
        return
    native = read_csv_or_empty(native_csv)
    if native.empty or "structure_id" not in native.columns:
        return
    decoy_structure_ids = set(comparison["structure_id"].dropna().astype(str))
    native_structure_ids = set(native["structure_id"].dropna().astype(str))
    if not decoy_structure_ids.intersection(native_structure_ids):
        return
    missing = [column for column in REQUIRED_ROSETTA_NATIVE_COMPARISON_COLUMNS if column not in comparison.columns]
    if missing:
        raise RuntimeError(
            f"Rosetta native baseline {native_csv} overlaps decoys by structure_id, "
            "but decoy/native comparison is missing required native column(s): "
            + ", ".join(missing)
        )


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


def run_decoy_rosetta(
    pdb_path: Path,
    row: dict[str, object],
    work_dir: Path,
    docker_image: str | None,
    runtime_config: RosettaRuntimeConfig | None = None,
) -> dict[str, object]:
    base = {"design_id": row["design_id"], "structure_id": row["structure_id"], "scored_pdb": str(pdb_path)}
    try:
        runtime = rosetta_runtime_from_options(runtime_config, docker_image)
    except RosettaRuntimeError as exc:
        return {**base, "rosetta_status": f"runtime_unavailable:{exc}"}

    score_dir = work_dir / "rosetta" / str(row["structure_id"])
    score_dir.mkdir(parents=True, exist_ok=True)
    scorefile = score_dir / f"{row['design_id']}.interface.sc"
    log = score_dir / f"{row['design_id']}.interface.log"
    # Decoys are written in the A=antigen / N=nanobody convention, so the interface
    # is N_A regardless of the original crystal chain letters.
    interface = f"{BOLTZ_CHAIN_NANOBODY}_{BOLTZ_CHAIN_ANTIGEN}"
    status = None
    if scorefile.exists() and scorefile.stat().st_size > 0:
        score = parse_scorefile(scorefile)
        if interface_score_has_required_terms(score) and log.exists() and log.stat().st_size > 0:
            rosetta_status = "skipped_existing_valid"
        else:
            command = build_interface_analyzer_command(
                runtime,
                pdb=pdb_path,
                score_dir=score_dir,
                scorefile=scorefile,
                interface=interface,
            )
            with log.open("w") as handle:
                status = runtime.run(command, stdout=handle, stderr=subprocess.STDOUT)
            score = parse_scorefile(scorefile)
            if status.returncode != 0:
                rosetta_status = f"failed:{status.returncode}"
            elif not interface_score_has_required_terms(score):
                rosetta_status = "parse_failed"
            else:
                rosetta_status = "ok"
    else:
        command = build_interface_analyzer_command(
            runtime,
            pdb=pdb_path,
            score_dir=score_dir,
            scorefile=scorefile,
            interface=interface,
        )
        with log.open("w") as handle:
            status = runtime.run(command, stdout=handle, stderr=subprocess.STDOUT)
        score = parse_scorefile(scorefile)
        if status.returncode != 0:
            rosetta_status = f"failed:{status.returncode}"
        elif not interface_score_has_required_terms(score):
            rosetta_status = "parse_failed"
        else:
            rosetta_status = "ok"
    return {
        **base,
        **score,
        "rosetta_runtime": runtime.config.kind,
        "rosetta_status": rosetta_status,
        "interface_scorefile": str(scorefile),
        "interface_log": str(log),
    }


def build_decoy_report(tables_dir: Path = Path("results/tables"), figures_dir: Path = Path("results/figures")) -> None:
    set_publication_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_csv_or_empty(tables_dir / "decoy_mpnn_candidates.csv")
    validation = read_csv_or_empty(tables_dir / "decoy_validation.csv")
    comparison = read_csv_or_empty(tables_dir / "decoy_native_comparison.csv")
    plot_identity_contacts(candidates, figures_dir / "decoy_identity_contact_mutation.png")
    plot_validation(validation, figures_dir / "decoy_monomer_validation.png")
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
        ax.text(0.5, 0.5, "No monomer validations found", ha="center", va="center")
    else:
        ok = df.dropna(subset=["monomer_ca_rmsd"])
        colors = ok.get("validation_status", pd.Series(index=ok.index, data="unknown")).map({"pass": "#2ca02c", "failed_fold": "#d62728"}).fillna("#7f7f7f")
        ax.scatter(ok["monomer_ca_rmsd"], ok.get("confidence_score", pd.Series(index=ok.index, data=np.nan)), c=colors, s=22, alpha=0.8)
        ax.axvline(2.5, color="black", lw=1, ls="--")
        ax.set_xlabel("Monomer CA RMSD to native/reference")
        ax.set_ylabel("Folding confidence score")
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
