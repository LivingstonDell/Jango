from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import stat
import subprocess

import numpy as np
import pandas as pd

from .cdr import annotate_cdrs
from .features import atom_contacts, compute_interface_features
from .pdbio import Atom, chain_residues, chain_sequence, parse_atoms, read_pdb_text
from .plots import set_publication_style
from .config import RosettaRuntimeConfig
from .runtime.rosetta import RosettaRuntimeError
from .rosetta import build_fullscore_command, build_interface_analyzer_command, parse_scorefile, rosetta_runtime_from_options

import matplotlib.pyplot as plt


BOLTZ_CONDITIONS = ["sequence_only", "monomer_templates", "pocket_contacts", "full_complex_template"]
BOLTZ_CHAIN_NANOBODY = "N"
BOLTZ_CHAIN_ANTIGEN = "A"
SELECTION_FEATURES = [
    "antigen_length",
    "dSASA_int",
    "dG_separated",
    "sc_value",
    "cdr3_interface_fraction",
    "residue_contact_pairs_5A",
]
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


@dataclass(frozen=True)
class BoltzPaths:
    config: Path = Path("configs/boltz2.yml")
    case_manifest: Path = Path("results/tables/boltz2_case_manifest.csv")
    run_manifest: Path = Path("results/tables/boltz2_run_manifest.csv")
    work_dir: Path = Path("work/boltz2")
    output_dir: Path = Path("work/boltz2/outputs")
    raw_dir: Path = Path("data/raw/pdb")


def write_default_boltz2_config(path: Path = Path("configs/boltz2.yml")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "case_count: 12",
                "antigen_length_min: 60",
                "antigen_length_max: 250",
                "conditions:",
                "  - sequence_only",
                "  - monomer_templates",
                "  - pocket_contacts",
                "  - full_complex_template",
                "boltz_chain_ids:",
                "  nanobody: N",
                "  antigen: A",
                "pocket_contact_distance_angstrom: 8.0",
                "full_complex_template_threshold_angstrom: 1.0",
                "diffusion_samples: 5",
                "recycling_steps: 3",
                "output_format: pdb",
                "external_execution: true",
                "notes: Full-complex-template runs are positive controls, not independent interface-discovery evidence.",
                "",
            ]
        )
    )
    return path


def select_boltz2_cases(
    manifest_csv: Path = Path("data/manifest/manifest.csv"),
    native_features_csv: Path = Path("results/tables/native_interface_features.csv"),
    rosetta_native_csv: Path = Path("results/tables/rosetta_interface_native.csv"),
    out_csv: Path = Path("results/tables/boltz2_case_manifest.csv"),
    config_path: Path = Path("configs/boltz2.yml"),
    n_cases: int = 12,
    antigen_length_min: int = 60,
    antigen_length_max: int = 250,
) -> pd.DataFrame:
    write_default_boltz2_config(config_path)
    manifest = pd.read_csv(manifest_csv)
    features = pd.read_csv(native_features_csv)
    rosetta = pd.read_csv(rosetta_native_csv)
    rosetta_cols = ["structure_id", "dSASA_int", "dG_separated", "sc_value", "fullpose_total_score"]
    df = (
        manifest.merge(features, on=["structure_id", "pdb_id", "nanobody_chain", "antigen_chain", "nanobody_length", "antigen_length", "antigen_class"])
        .merge(rosetta[rosetta_cols], on="structure_id", how="left")
    )
    df = df[(df["antigen_class"] == "protein") & df["antigen_length"].between(antigen_length_min, antigen_length_max)].copy()
    df = df.dropna(subset=SELECTION_FEATURES)
    df = df.sort_values(["pdb_id", "structure_id"]).drop_duplicates("pdb_id", keep="first")
    selected = greedy_max_min_select(df, SELECTION_FEATURES, n_cases)
    out = selected.copy()
    out["boltz_nanobody_chain"] = BOLTZ_CHAIN_NANOBODY
    out["boltz_antigen_chain"] = BOLTZ_CHAIN_ANTIGEN
    out["selection_rank"] = range(1, len(out) + 1)
    out["selection_features"] = ",".join(SELECTION_FEATURES)
    keep = [
        "selection_rank",
        "structure_id",
        "pdb_id",
        "nanobody_chain",
        "antigen_chain",
        "boltz_nanobody_chain",
        "boltz_antigen_chain",
        "source_filename",
        "nanobody_length",
        "antigen_length",
        "antigen_class",
        "dSASA_int",
        "dG_separated",
        "sc_value",
        "fullpose_total_score",
        "cdr3_interface_fraction",
        "residue_contact_pairs_5A",
        "selection_distance",
        "selection_features",
    ]
    out = out[keep].sort_values("selection_rank")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def greedy_max_min_select(df: pd.DataFrame, features: list[str], n_cases: int) -> pd.DataFrame:
    if df.empty:
        return df.assign(selection_distance=[])
    values = df[features].astype(float)
    std = values.std(ddof=0).replace(0, 1)
    z = ((values - values.mean()) / std).to_numpy()
    selected = [int(np.argmin(np.linalg.norm(z, axis=1)))]
    distances = [0.0]
    while len(selected) < min(n_cases, len(df)):
        remaining = [i for i in range(len(df)) if i not in selected]
        dist_to_selected = np.min(
            np.linalg.norm(z[remaining, None, :] - z[np.array(selected)][None, :, :], axis=2),
            axis=1,
        )
        best_pos = int(np.argmax(dist_to_selected))
        selected.append(remaining[best_pos])
        distances.append(float(dist_to_selected[best_pos]))
    out = df.iloc[selected].copy()
    out["selection_distance"] = distances
    return out


def prepare_boltz2_inputs(
    case_manifest_csv: Path = Path("results/tables/boltz2_case_manifest.csv"),
    raw_dir: Path = Path("data/raw/pdb"),
    work_dir: Path = Path("work/boltz2"),
    out_csv: Path = Path("results/tables/boltz2_run_manifest.csv"),
    limit: int | None = None,
) -> pd.DataFrame:
    cases = pd.read_csv(case_manifest_csv)
    if limit:
        cases = cases.head(limit)
    rows: list[dict[str, object]] = []
    for case in cases.to_dict("records"):
        source = raw_dir / str(case["source_filename"])
        atoms = parse_atoms(read_pdb_text(source))
        nb_chain = str(case["nanobody_chain"])
        ag_chain = str(case["antigen_chain"])
        nb_seq = chain_sequence(atoms, nb_chain)
        ag_seq = chain_sequence(atoms, ag_chain)
        structure_id = str(case["structure_id"])
        template_dir = work_dir / "inputs" / structure_id / "templates"
        template_dir.mkdir(parents=True, exist_ok=True)
        nanobody_template_pdb = template_dir / f"{structure_id}_nanobody_template.pdb"
        antigen_template_pdb = template_dir / f"{structure_id}_antigen_template.pdb"
        complex_template_pdb = template_dir / f"{structure_id}_complex_template.pdb"
        write_remapped_pdb(atoms, nanobody_template_pdb, {nb_chain: BOLTZ_CHAIN_NANOBODY})
        write_remapped_pdb(atoms, antigen_template_pdb, {ag_chain: BOLTZ_CHAIN_ANTIGEN})
        write_remapped_pdb(atoms, complex_template_pdb, {nb_chain: BOLTZ_CHAIN_NANOBODY, ag_chain: BOLTZ_CHAIN_ANTIGEN})
        nanobody_template = write_template_cif(nanobody_template_pdb)
        antigen_template = write_template_cif(antigen_template_pdb)
        complex_template = write_template_cif(complex_template_pdb)
        contacts = native_interface_residue_indices(atoms, nb_chain, ag_chain)
        for condition in BOLTZ_CONDITIONS:
            input_dir = work_dir / "inputs" / structure_id / condition
            output_dir = work_dir / "outputs" / structure_id / condition
            yaml_path = input_dir / f"{structure_id}_{condition}.yaml"
            input_dir.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(
                boltz_yaml(
                    nanobody_sequence=nb_seq,
                    antigen_sequence=ag_seq,
                    condition=condition,
                    nanobody_template=relative_path(nanobody_template, input_dir),
                    antigen_template=relative_path(antigen_template, input_dir),
                    complex_template=relative_path(complex_template, input_dir),
                    pocket_residues=contacts["antigen_residue_indices"],
                    contact_pairs=contacts["contact_pairs"],
                )
            )
            rows.append(
                {
                    "structure_id": structure_id,
                    "condition": condition,
                    "yaml_path": str(yaml_path),
                    "output_dir": str(output_dir),
                    "nanobody_sequence_length": len(nb_seq),
                    "antigen_sequence_length": len(ag_seq),
                    "nanobody_template": str(nanobody_template) if condition in {"monomer_templates", "full_complex_template"} else "",
                    "antigen_template": str(antigen_template) if condition == "monomer_templates" else "",
                    "complex_template": str(complex_template) if condition == "full_complex_template" else "",
                    "positive_control": condition == "full_complex_template",
                    "boltz_command": boltz_command(yaml_path, output_dir),
                }
            )
    run_manifest = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run_manifest.to_csv(out_csv, index=False)
    write_job_bundle(run_manifest, work_dir / "job_bundle")
    return run_manifest


def relative_path(path: Path, start: Path) -> str:
    return str(path.resolve().relative_to(start.resolve())) if path.resolve().is_relative_to(start.resolve()) else str(path)


def boltz_command(yaml_path: Path, output_dir: Path) -> str:
    return (
        f"boltz predict {yaml_path} --model boltz2 --use_msa_server --use_potentials "
        f"--diffusion_samples 5 --recycling_steps 3 --output_format pdb --write_full_pae --write_full_pde --out_dir {output_dir}"
    )


def write_job_bundle(run_manifest: pd.DataFrame, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tsv = bundle_dir / "boltz2_jobs.tsv"
    run_manifest[["structure_id", "condition", "yaml_path", "output_dir", "boltz_command"]].to_csv(tsv, sep="\t", index=False)
    script = bundle_dir / "run_boltz2_jobs.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "JOBS_TSV=${1:-work/boltz2/job_bundle/boltz2_jobs.tsv}",
                "tail -n +2 \"$JOBS_TSV\" | while IFS=$'\\t' read -r structure_id condition yaml_path output_dir boltz_command; do",
                "  mkdir -p \"$output_dir\"",
                "  echo \"[$structure_id/$condition] $boltz_command\"",
                "  $boltz_command",
                "done",
                "",
            ]
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def boltz_yaml(
    nanobody_sequence: str,
    antigen_sequence: str,
    condition: str,
    nanobody_template: str,
    antigen_template: str,
    complex_template: str,
    pocket_residues: list[int],
    contact_pairs: list[tuple[int, int]],
) -> str:
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        f"      id: {BOLTZ_CHAIN_NANOBODY}",
        f"      sequence: {nanobody_sequence}",
        "  - protein:",
        f"      id: {BOLTZ_CHAIN_ANTIGEN}",
        f"      sequence: {antigen_sequence}",
    ]
    if condition == "monomer_templates":
        lines += [
            "templates:",
            f"  - cif: {nanobody_template}",
            f"    chain_id: {BOLTZ_CHAIN_NANOBODY}",
            f"  - cif: {antigen_template}",
            f"    chain_id: {BOLTZ_CHAIN_ANTIGEN}",
        ]
    elif condition == "full_complex_template":
        lines += [
            "templates:",
            f"  - cif: {complex_template}",
            f"    chain_id: [{BOLTZ_CHAIN_NANOBODY}, {BOLTZ_CHAIN_ANTIGEN}]",
            "    template_id: [Nxp, Axp]",
            "    force: true",
            "    threshold: 1.0",
        ]
    elif condition == "pocket_contacts":
        lines += ["constraints:"]
        if pocket_residues:
            contacts = ", ".join(f"[{BOLTZ_CHAIN_ANTIGEN}, {idx}]" for idx in pocket_residues[:12])
            lines += [
                "  - pocket:",
                f"      binder: {BOLTZ_CHAIN_NANOBODY}",
                f"      contacts: [{contacts}]",
                "      max_distance: 8.0",
                "      force: true",
            ]
        for nb_idx, ag_idx in contact_pairs[:12]:
            lines += [
                "  - contact:",
                f"      token1: [{BOLTZ_CHAIN_NANOBODY}, {nb_idx}]",
                f"      token2: [{BOLTZ_CHAIN_ANTIGEN}, {ag_idx}]",
                "      max_distance: 8.0",
                "      force: true",
            ]
    elif condition != "sequence_only":
        raise ValueError(f"Unknown Boltz2 condition: {condition}")
    return "\n".join(lines) + "\n"


def native_interface_residue_indices(atoms: list[Atom], nb_chain: str, ag_chain: str) -> dict[str, list]:
    nb_atoms = [a for a in atoms if a.chain_id == nb_chain and a.is_heavy]
    ag_atoms = [a for a in atoms if a.chain_id == ag_chain and a.is_heavy]
    nb_order = {(num, icode, resname): i + 1 for i, (num, icode, resname) in enumerate(chain_residues(atoms, nb_chain))}
    ag_order = {(num, icode, resname): i + 1 for i, (num, icode, resname) in enumerate(chain_residues(atoms, ag_chain))}
    contacts = atom_contacts(nb_atoms, ag_atoms, 5.0)
    residue_pairs = sorted({(nb_atoms[i].residue_key, ag_atoms[j].residue_key) for i, j, _ in contacts})
    nb_indices = sorted({nb_order[(r[1], r[2], r[3])] for r, _ in residue_pairs if (r[1], r[2], r[3]) in nb_order})
    ag_indices = sorted({ag_order[(r[1], r[2], r[3])] for _, r in residue_pairs if (r[1], r[2], r[3]) in ag_order})
    pair_indices = []
    for nb_key, ag_key in residue_pairs:
        nb_idx = nb_order.get((nb_key[1], nb_key[2], nb_key[3]))
        ag_idx = ag_order.get((ag_key[1], ag_key[2], ag_key[3]))
        if nb_idx and ag_idx:
            pair_indices.append((nb_idx, ag_idx))
    return {"nanobody_residue_indices": nb_indices, "antigen_residue_indices": ag_indices, "contact_pairs": pair_indices}


def write_remapped_pdb(atoms: list[Atom], path: Path, chain_map: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    serial = 1
    residue_counters: dict[str, dict[tuple[int, str, str], int]] = {}
    for atom in atoms:
        if atom.chain_id not in chain_map or atom.residue_name == "HOH":
            continue
        new_chain = chain_map[atom.chain_id]
        key = (atom.residue_number, atom.insertion_code, atom.residue_name)
        residue_counters.setdefault(new_chain, {})
        if key not in residue_counters[new_chain]:
            residue_counters[new_chain][key] = len(residue_counters[new_chain]) + 1
        resseq = residue_counters[new_chain][key]
        lines.append(format_pdb_atom(serial, atom, new_chain, resseq))
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_template_cif(pdb_path: Path) -> Path:
    import gemmi

    structure = gemmi.read_structure(str(pdb_path))
    structure.setup_entities()
    sequence_by_chain = {}
    for model in structure:
        for chain in model:
            sequence_by_chain[chain.name] = [residue.name for residue in chain.get_polymer()]
        break
    for entity in structure.entities:
        if entity.name in sequence_by_chain:
            entity.full_sequence = sequence_by_chain[entity.name]
    cif_path = pdb_path.with_suffix(".cif")
    structure.make_mmcif_document().write_file(str(cif_path))
    return cif_path


def format_pdb_atom(serial: int, atom: Atom, chain: str, resseq: int) -> str:
    atom_name = atom.atom_name[:4]
    return (
        f"ATOM  {serial:5d} {atom_name:<4} {atom.residue_name:>3} {chain:1}{resseq:4d}    "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}  1.00  0.00          {atom.element:>2}"
    )


def analyze_boltz2_outputs(
    run_manifest_csv: Path = Path("results/tables/boltz2_run_manifest.csv"),
    case_manifest_csv: Path = Path("results/tables/boltz2_case_manifest.csv"),
    raw_dir: Path = Path("data/raw/pdb"),
    tables_dir: Path = Path("results/tables"),
    work_dir: Path = Path("work/boltz2"),
    run_rosetta: bool = True,
    docker_image: str | None = "rosettacommons/rosetta:latest",
    runtime_config: RosettaRuntimeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(run_manifest_csv)
    cases = pd.read_csv(case_manifest_csv).set_index("structure_id")
    prediction_rows = []
    quality_rows = []
    recovery_rows = []
    rosetta_rows = []
    for run in runs.to_dict("records"):
        structure_id = str(run["structure_id"])
        condition = str(run["condition"])
        output_dir = Path(str(run["output_dir"]))
        predictions = discover_boltz_predictions(output_dir)
        if not predictions:
            base = {"structure_id": structure_id, "condition": condition, "prediction_status": "missing_output", "output_dir": str(output_dir)}
            prediction_rows.append(base)
            quality_rows.append({**base, **empty_quality()})
            recovery_rows.append({**base, **empty_recovery()})
            rosetta_rows.append({**base, "rosetta_status": "not_run"})
            continue
        case = cases.loc[structure_id]
        native_atoms = parse_atoms(read_pdb_text(raw_dir / str(case["source_filename"])))
        native_remapped = remap_atoms(native_atoms, {str(case["nanobody_chain"]): BOLTZ_CHAIN_NANOBODY, str(case["antigen_chain"]): BOLTZ_CHAIN_ANTIGEN})
        native_contacts = contact_sets(native_remapped, BOLTZ_CHAIN_NANOBODY, BOLTZ_CHAIN_ANTIGEN)
        for sample_idx, pred_path in enumerate(predictions):
            prediction_id = f"{structure_id}_{condition}_model_{sample_idx}"
            confidence = parse_confidence_json(pred_path)
            pred_atoms = parse_atoms(read_pdb_text(pred_path))
            prediction_rows.append(
                {
                    "prediction_id": prediction_id,
                    "structure_id": structure_id,
                    "condition": condition,
                    "sample_index": sample_idx,
                    "prediction_status": "ok",
                    "prediction_path": str(pred_path),
                    "output_dir": str(output_dir),
                    **confidence,
                }
            )
            quality = compute_structure_quality(native_remapped, pred_atoms, native_contacts)
            recovery = compute_interface_recovery(native_remapped, pred_atoms, native_contacts)
            quality_rows.append({"prediction_id": prediction_id, "structure_id": structure_id, "condition": condition, **quality})
            recovery_rows.append({"prediction_id": prediction_id, "structure_id": structure_id, "condition": condition, **recovery})
            if run_rosetta:
                rosetta_rows.append(run_prediction_rosetta(pred_path, prediction_id, structure_id, condition, work_dir, docker_image, runtime_config))
            else:
                rosetta_rows.append({"prediction_id": prediction_id, "structure_id": structure_id, "condition": condition, "rosetta_status": "not_run", **empty_rosetta()})
    prediction_df = pd.DataFrame(prediction_rows)
    quality_df = pd.DataFrame(quality_rows)
    recovery_df = pd.DataFrame(recovery_rows)
    rosetta_df = pd.DataFrame(rosetta_rows)
    summary_df = summarize_boltz2(prediction_df, quality_df, recovery_df)
    tables_dir.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(tables_dir / "boltz2_prediction_manifest.csv", index=False)
    quality_df.to_csv(tables_dir / "boltz2_structure_quality.csv", index=False)
    recovery_df.to_csv(tables_dir / "boltz2_interface_recovery.csv", index=False)
    rosetta_df.to_csv(tables_dir / "boltz2_rosetta_interface.csv", index=False)
    summary_df.to_csv(tables_dir / "boltz2_true_like_summary.csv", index=False)
    return prediction_df, quality_df, recovery_df, rosetta_df, summary_df


def run_prediction_rosetta(
    pred_path: Path,
    prediction_id: str,
    structure_id: str,
    condition: str,
    work_dir: Path,
    docker_image: str | None,
    runtime_config: RosettaRuntimeConfig | None = None,
) -> dict[str, object]:
    base = {"prediction_id": prediction_id, "structure_id": structure_id, "condition": condition}
    try:
        runtime = rosetta_runtime_from_options(runtime_config, docker_image)
    except RosettaRuntimeError as exc:
        return {**base, "rosetta_status": f"runtime_unavailable:{exc}", **empty_rosetta()}
    score_dir = work_dir / "rosetta" / structure_id / condition
    score_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re_safe(prediction_id)
    interface_scorefile = score_dir / f"{safe_id}.interface.sc"
    interface_log = score_dir / f"{safe_id}.interface.log"
    fullscore_scorefile = score_dir / f"{safe_id}.fullscore.sc"
    fullscore_log = score_dir / f"{safe_id}.fullscore.log"
    try:
        interface_command = build_interface_analyzer_command(
            runtime,
            pdb=pred_path,
            score_dir=score_dir,
            scorefile=interface_scorefile,
            interface=f"{BOLTZ_CHAIN_NANOBODY}_{BOLTZ_CHAIN_ANTIGEN}",
        )
        with interface_log.open("w") as handle:
            interface_status = runtime.run(interface_command, stdout=handle, stderr=subprocess.STDOUT)
        fullscore_command = build_fullscore_command(runtime, pdb=pred_path, score_dir=score_dir, scorefile=fullscore_scorefile)
        with fullscore_log.open("w") as handle:
            fullscore_status = runtime.run(fullscore_command, stdout=handle, stderr=subprocess.STDOUT)
        interface = parse_scorefile(interface_scorefile)
        fullscore = parse_scorefile(fullscore_scorefile)
        out = {**base, **empty_rosetta()}
        for col in ["dG_separated", "dG_separated/dSASAx100", "dSASA_int", "sc_value", "hbonds_int", "delta_unsatHbonds"]:
            if col in interface:
                out[col] = interface[col]
        if "total_score" in fullscore:
            out["fullpose_total_score"] = fullscore["total_score"]
        out["rosetta_status"] = "ok" if interface_status.returncode == 0 and fullscore_status.returncode == 0 else f"failed:{interface_status.returncode}:{fullscore_status.returncode}"
        out["interface_log"] = str(interface_log)
        out["fullscore_log"] = str(fullscore_log)
        return out
    except Exception as exc:
        return {**base, "rosetta_status": f"exception:{type(exc).__name__}", **empty_rosetta()}


def re_safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def discover_boltz_predictions(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    paths = [
        p
        for p in output_dir.rglob("*")
        if p.suffix.lower() in {".pdb", ".cif"} and "templates" not in {part.lower() for part in p.parts}
    ]
    return sorted(paths)


def parse_confidence_json(prediction_path: Path) -> dict[str, object]:
    preferred = prediction_path.parent / f"confidence_{prediction_path.stem}.json"
    if preferred.exists():
        candidates = [preferred]
    else:
        candidates = sorted(prediction_path.parent.glob("confidence*.json"))
        if len(prediction_path.parents) > 1:
            candidates.extend(sorted(prediction_path.parents[1].glob("**/confidence*.json")))
    if not candidates:
        return {"confidence_status": "missing"}
    try:
        data = json.loads(candidates[0].read_text())
    except Exception:
        return {"confidence_status": "unparsed"}
    keys = ["confidence_score", "ptm", "iptm", "complex_plddt", "complex_iplddt", "protein_iptm"]
    out = {"confidence_status": "ok"}
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    out["confidence_json"] = str(candidates[0])
    return out


def remap_atoms(atoms: list[Atom], chain_map: dict[str, str]) -> list[Atom]:
    remapped = []
    counters: dict[str, dict[tuple[int, str, str], int]] = {}
    for atom in atoms:
        if atom.chain_id not in chain_map:
            continue
        chain = chain_map[atom.chain_id]
        key = (atom.residue_number, atom.insertion_code, atom.residue_name)
        counters.setdefault(chain, {})
        if key not in counters[chain]:
            counters[chain][key] = len(counters[chain]) + 1
        remapped.append(Atom(chain, counters[chain][key], "", atom.residue_name, atom.atom_name, atom.element, atom.x, atom.y, atom.z))
    return remapped


def contact_sets(atoms: list[Atom], nb_chain: str, ag_chain: str, cutoff: float = 5.0) -> dict[str, set]:
    nb_atoms = [a for a in atoms if a.chain_id == nb_chain and a.is_heavy]
    ag_atoms = [a for a in atoms if a.chain_id == ag_chain and a.is_heavy]
    contacts = atom_contacts(nb_atoms, ag_atoms, cutoff)
    atom_pairs = {((nb_atoms[i].residue_number, nb_atoms[i].atom_name), (ag_atoms[j].residue_number, ag_atoms[j].atom_name)) for i, j, _ in contacts}
    residue_pairs = {(nb_atoms[i].residue_number, ag_atoms[j].residue_number) for i, j, _ in contacts}
    paratope = {pair[0] for pair in residue_pairs}
    epitope = {pair[1] for pair in residue_pairs}
    return {"atom_pairs": atom_pairs, "residue_pairs": residue_pairs, "paratope": paratope, "epitope": epitope}


def compute_structure_quality(native_atoms: list[Atom], pred_atoms: list[Atom], native_contacts: dict[str, set]) -> dict[str, float | str]:
    try:
        nb_rmsd = chain_backbone_rmsd(native_atoms, pred_atoms, BOLTZ_CHAIN_NANOBODY)
        ag_rmsd = chain_backbone_rmsd(native_atoms, pred_atoms, BOLTZ_CHAIN_ANTIGEN)
        lrmsd = ligand_rmsd_after_receptor_alignment(native_atoms, pred_atoms)
        irmsd = interface_rmsd(native_atoms, pred_atoms, native_contacts)
        pred_contacts = contact_sets(pred_atoms, BOLTZ_CHAIN_NANOBODY, BOLTZ_CHAIN_ANTIGEN)
        fnat = safe_fraction(len(native_contacts["residue_pairs"] & pred_contacts["residue_pairs"]), len(native_contacts["residue_pairs"]))
        dockq = dockq_like(fnat, irmsd, lrmsd)
        return {
            "quality_status": "ok",
            "nanobody_backbone_rmsd": nb_rmsd,
            "antigen_backbone_rmsd": ag_rmsd,
            "lrmsd": lrmsd,
            "irmsd": irmsd,
            "fnat": fnat,
            "dockq_like": dockq,
            "correctness_label": correctness_label(dockq, fnat),
        }
    except Exception as exc:
        return {**empty_quality(), "quality_status": f"failed:{type(exc).__name__}"}


def compute_interface_recovery(native_atoms: list[Atom], pred_atoms: list[Atom], native_contacts: dict[str, set]) -> dict[str, float | str]:
    try:
        pred_contacts = contact_sets(pred_atoms, BOLTZ_CHAIN_NANOBODY, BOLTZ_CHAIN_ANTIGEN)
        native_pairs = native_contacts["residue_pairs"]
        pred_pairs = pred_contacts["residue_pairs"]
        nb_sequence = chain_sequence(native_atoms, BOLTZ_CHAIN_NANOBODY)
        cdr = annotate_cdrs(nb_sequence)
        native_regions = Counter(cdr.by_index.get(idx, "unmapped") for idx in native_contacts["paratope"])
        pred_regions = Counter(cdr.by_index.get(idx, "unmapped") for idx in pred_contacts["paratope"])
        return {
            "recovery_status": "ok",
            "native_contact_recovery": safe_fraction(len(native_pairs & pred_pairs), len(native_pairs)),
            "predicted_false_contact_fraction": safe_fraction(len(pred_pairs - native_pairs), len(pred_pairs)),
            "epitope_jaccard": jaccard(native_contacts["epitope"], pred_contacts["epitope"]),
            "paratope_jaccard": jaccard(native_contacts["paratope"], pred_contacts["paratope"]),
            "cdr1_contact_recovery": safe_fraction(min(native_regions["CDR1"], pred_regions["CDR1"]), native_regions["CDR1"]),
            "cdr2_contact_recovery": safe_fraction(min(native_regions["CDR2"], pred_regions["CDR2"]), native_regions["CDR2"]),
            "cdr3_contact_recovery": safe_fraction(min(native_regions["CDR3"], pred_regions["CDR3"]), native_regions["CDR3"]),
            "cdr3_contribution_delta": safe_fraction(pred_regions["CDR3"], sum(pred_regions.values())) - safe_fraction(native_regions["CDR3"], sum(native_regions.values())),
        }
    except Exception as exc:
        return {**empty_recovery(), "recovery_status": f"failed:{type(exc).__name__}"}


def chain_backbone_rmsd(native_atoms: list[Atom], pred_atoms: list[Atom], chain: str) -> float:
    native_xyz, pred_xyz = matched_backbone(native_atoms, pred_atoms, chain)
    _, pred_aligned = superpose(native_xyz, pred_xyz)
    return rmsd(native_xyz, pred_aligned)


def ligand_rmsd_after_receptor_alignment(native_atoms: list[Atom], pred_atoms: list[Atom]) -> float:
    native_nb, pred_nb = matched_backbone(native_atoms, pred_atoms, BOLTZ_CHAIN_NANOBODY)
    rotation, pred_nb_aligned = superpose(native_nb, pred_nb)
    pred_center = pred_nb.mean(axis=0)
    native_center = native_nb.mean(axis=0)
    native_ag, pred_ag = matched_backbone(native_atoms, pred_atoms, BOLTZ_CHAIN_ANTIGEN)
    pred_ag_aligned = (pred_ag - pred_center) @ rotation + native_center
    return rmsd(native_ag, pred_ag_aligned)


def interface_rmsd(native_atoms: list[Atom], pred_atoms: list[Atom], native_contacts: dict[str, set]) -> float:
    residues = {
        BOLTZ_CHAIN_NANOBODY: native_contacts["paratope"],
        BOLTZ_CHAIN_ANTIGEN: native_contacts["epitope"],
    }
    native_xyz, pred_xyz = matched_backbone(native_atoms, pred_atoms, None, residues)
    _, pred_aligned = superpose(native_xyz, pred_xyz)
    return rmsd(native_xyz, pred_aligned)


def matched_backbone(
    native_atoms: list[Atom],
    pred_atoms: list[Atom],
    chain: str | None,
    residues: dict[str, set[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    native = backbone_map(native_atoms, chain, residues)
    pred = backbone_map(pred_atoms, chain, residues)
    keys = sorted(set(native) & set(pred))
    if len(keys) < 3:
        raise ValueError("Fewer than three matched backbone atoms")
    return np.array([native[k] for k in keys]), np.array([pred[k] for k in keys])


def backbone_map(atoms: list[Atom], chain: str | None, residues: dict[str, set[int]] | None) -> dict[tuple[str, int, str], np.ndarray]:
    out = {}
    for atom in atoms:
        if atom.atom_name not in BACKBONE_ATOMS:
            continue
        if chain is not None and atom.chain_id != chain:
            continue
        if residues is not None and atom.residue_number not in residues.get(atom.chain_id, set()):
            continue
        out[(atom.chain_id, atom.residue_number, atom.atom_name)] = np.array([atom.x, atom.y, atom.z], dtype=float)
    return out


def superpose(reference: np.ndarray, mobile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    ref = reference - ref_center
    mob = mobile - mob_center
    cov = mob.T @ ref
    u, _, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    aligned = mob @ rotation + ref_center
    return rotation, aligned


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def dockq_like(fnat: float, irmsd: float, lrmsd: float) -> float:
    return float((fnat + 1 / (1 + (irmsd / 1.5) ** 2) + 1 / (1 + (lrmsd / 8.5) ** 2)) / 3)


def correctness_label(dockq: float, fnat: float) -> str:
    if dockq >= 0.80 and fnat >= 0.70:
        return "true_like"
    if dockq >= 0.49:
        return "acceptable"
    if dockq < 0.23 or fnat < 0.30:
        return "decoy_like"
    return "intermediate"


def summarize_boltz2(prediction_df: pd.DataFrame, quality_df: pd.DataFrame, recovery_df: pd.DataFrame) -> pd.DataFrame:
    merged = prediction_df.merge(quality_df, on=["prediction_id", "structure_id", "condition"], how="left") if "prediction_id" in prediction_df else quality_df
    status_col = next((col for col in ("prediction_status", "prediction_status_x", "prediction_status_y") if col in merged.columns), None)
    rows = []
    for condition, group in merged.groupby("condition", dropna=False):
        status = group[status_col] if status_col else pd.Series(index=group.index, data="missing_output")
        ok = group[status == "ok"]
        rows.append(
            {
                "condition": condition,
                "n_jobs_or_predictions": int(group.shape[0]),
                "n_predictions_ok": int(ok.shape[0]),
                "missing_or_failed": int((status != "ok").sum()),
                "median_dockq_like": float(ok["dockq_like"].median()) if "dockq_like" in ok and not ok.empty else np.nan,
                "true_like_fraction": float((ok.get("correctness_label", pd.Series(dtype=str)) == "true_like").mean()) if not ok.empty else np.nan,
                "acceptable_fraction": float(ok.get("correctness_label", pd.Series(dtype=str)).isin(["true_like", "acceptable"]).mean()) if not ok.empty else np.nan,
                "decoy_like_fraction": float((ok.get("correctness_label", pd.Series(dtype=str)) == "decoy_like").mean()) if not ok.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_boltz2_report(tables_dir: Path = Path("results/tables"), figures_dir: Path = Path("results/figures")) -> None:
    set_publication_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    quality = read_optional_csv(tables_dir / "boltz2_structure_quality.csv")
    recovery = read_optional_csv(tables_dir / "boltz2_interface_recovery.csv")
    rosetta = read_optional_csv(tables_dir / "boltz2_rosetta_interface.csv")
    pred = read_optional_csv(tables_dir / "boltz2_prediction_manifest.csv")
    plot_condition_quality(quality, figures_dir / "boltz2_condition_quality.png")
    plot_interface_recovery(recovery, figures_dir / "boltz2_interface_recovery.png")
    plot_rosetta_vs_correctness(quality, rosetta, figures_dir / "boltz2_rosetta_vs_correctness.png")
    plot_confidence_vs_correctness(quality, pred, figures_dir / "boltz2_confidence_vs_correctness.png")


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def plot_condition_quality(df: pd.DataFrame, path: Path) -> None:
    metrics = ["dockq_like", "fnat", "irmsd", "lrmsd"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    if df.empty or not set(metrics).issubset(df.columns):
        write_empty_panel(fig, axes.flat, "No Boltz2 predictions found")
    else:
        ok = df[df.get("quality_status", "ok") == "ok"]
        for ax, metric in zip(axes.flat, metrics):
            plot_box_or_empty(ok, metric, ax)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_interface_recovery(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    if df.empty or "native_contact_recovery" not in df:
        ax.text(0.5, 0.5, "No Boltz2 predictions found", ha="center", va="center")
    else:
        ok = df[df.get("recovery_status", "ok") == "ok"]
        if ok.empty:
            ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
        else:
            for condition, group in ok.groupby("condition"):
                ax.scatter(group["predicted_false_contact_fraction"], group["native_contact_recovery"], s=22, alpha=0.75, label=condition)
            ax.legend(frameon=False)
    ax.set_xlabel("Predicted false-contact fraction")
    ax.set_ylabel("Native contact recovery")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_rosetta_vs_correctness(quality: pd.DataFrame, rosetta: pd.DataFrame, path: Path) -> None:
    metrics = ["dG_separated", "sc_value", "delta_unsatHbonds"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), constrained_layout=True)
    df = quality.merge(rosetta, on=["prediction_id", "structure_id", "condition"], how="left") if not quality.empty and not rosetta.empty and "prediction_id" in quality else pd.DataFrame()
    if df.empty or "dockq_like" not in df:
        write_empty_panel(fig, axes.flat, "Rosetta not run for Boltz2 outputs")
    else:
        for ax, metric in zip(axes.flat, metrics):
            ax.scatter(df[metric], df["dockq_like"], s=18, alpha=0.7)
            ax.set_xlabel(metric)
            ax.set_ylabel("DockQ-like")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_confidence_vs_correctness(quality: pd.DataFrame, pred: pd.DataFrame, path: Path) -> None:
    metrics = ["confidence_score", "iptm", "complex_plddt", "complex_iplddt"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    df = quality.merge(pred, on=["prediction_id", "structure_id", "condition"], how="left") if not quality.empty and not pred.empty and "prediction_id" in quality else pd.DataFrame()
    if df.empty or "dockq_like" not in df:
        write_empty_panel(fig, axes.flat, "No Boltz2 confidence data found")
    else:
        for ax, metric in zip(axes.flat, metrics):
            if metric in df:
                ax.scatter(df[metric], df["dockq_like"], s=18, alpha=0.7)
            ax.set_xlabel(metric)
            ax.set_ylabel("DockQ-like")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def write_empty_panel(fig: plt.Figure, axes, message: str) -> None:
    for ax in axes:
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])


def plot_box_or_empty(df: pd.DataFrame, metric: str, ax: plt.Axes) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
    else:
        data = [group[metric].dropna().to_numpy() for _, group in df.groupby("condition")]
        labels = [condition for condition, _ in df.groupby("condition")]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.tick_params(axis="x", rotation=35)
    ax.set_title(metric)


def empty_quality() -> dict[str, object]:
    return {
        "quality_status": "missing_output",
        "nanobody_backbone_rmsd": np.nan,
        "antigen_backbone_rmsd": np.nan,
        "lrmsd": np.nan,
        "irmsd": np.nan,
        "fnat": np.nan,
        "dockq_like": np.nan,
        "correctness_label": "missing",
    }


def empty_recovery() -> dict[str, object]:
    return {
        "recovery_status": "missing_output",
        "native_contact_recovery": np.nan,
        "predicted_false_contact_fraction": np.nan,
        "epitope_jaccard": np.nan,
        "paratope_jaccard": np.nan,
        "cdr1_contact_recovery": np.nan,
        "cdr2_contact_recovery": np.nan,
        "cdr3_contact_recovery": np.nan,
        "cdr3_contribution_delta": np.nan,
    }


def empty_rosetta() -> dict[str, object]:
    return {
        "dG_separated": np.nan,
        "dG_separated/dSASAx100": np.nan,
        "dSASA_int": np.nan,
        "sc_value": np.nan,
        "hbonds_int": np.nan,
        "delta_unsatHbonds": np.nan,
        "fullpose_total_score": np.nan,
    }


def safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def jaccard(a: set, b: set) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else float("nan")



