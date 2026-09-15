from pathlib import Path

from nbia.cdr import annotate_cdrs
from nbia.features import atom_contacts
from nbia.pdbio import Atom, parse_structure_filename
import pandas as pd
import pytest

from nbia.compare import build_paired_delta_table
from nbia.boltz2 import (
    BOLTZ_CHAIN_ANTIGEN,
    BOLTZ_CHAIN_NANOBODY,
    boltz_yaml,
    compute_structure_quality,
    compute_interface_recovery,
    contact_sets,
    correctness_label,
    discover_boltz_predictions,
    dockq_like,
    greedy_max_min_select,
    parse_confidence_json,
    write_remapped_pdb,
)
from nbia.rosetta import insert_ter_records, normalize_interface_score_columns, prepare_pdb_for_rosetta
from nbia.decoys import (
    build_decoy_native_comparison,
    SEQUENCE_ROLE_NATIVE_CONTROL,
    SEQUENCE_ROLE_REDESIGNED_DECOY,
    contact_mutation_fraction,
    contact_omit_dict,
    compute_decoy_metrics,
    format_atom,
    graft_complex,
    graft_decoy_complexes,
    monomer_validation_metrics,
    parse_mpnn_candidates,
    pdb_atom_name_field,
    proteinmpnn_command,
    sequence_filter_status,
    sequence_identity,
    select_mpnn_redesign_candidates,
    summarize_decoy_validation,
)
from nbia.vhh_contacts import (
    aggregate_contact_frequency,
    parse_anarci_numbered_residues,
    vhh_contact_sequence_indices,
    write_report,
)


def test_parse_structure_filename_tarball():
    record = parse_structure_filename(Path("1kxq_E_D.pdb.tar.gz"))
    assert record.pdb_id == "1kxq"
    assert record.nanobody_chain == "E"
    assert record.antigen_chain == "D"
    assert record.structure_id == "1kxq_E_D"


def test_atom_contacts_detects_close_heavy_atoms():
    nb = [Atom("E", 1, "", "TYR", "OH", "O", 0.0, 0.0, 0.0)]
    ag = [
        Atom("D", 1, "", "ASP", "OD1", "O", 0.0, 0.0, 3.0),
        Atom("D", 2, "", "ASP", "OD1", "O", 0.0, 0.0, 8.0),
    ]
    contacts = atom_contacts(nb, ag, 5.0)
    assert len(contacts) == 1
    assert contacts[0][1] == 0


def test_cdr_annotation_has_regions_for_vhh_length_sequence():
    sequence = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSDGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAK"
    cdr = annotate_cdrs(sequence)
    assert cdr.by_index[30] == "CDR1"
    assert "CDR3" in set(cdr.by_index.values())


def test_insert_ter_records_uses_next_atom_serial_at_chain_break():
    pdb = "\n".join(
        [
            "ATOM   1894  CA  SER C 128      55.442  44.105  24.759  1.00  0.00           C",
            "ATOM   1895  N   SER C 128      54.623  42.920  24.990  1.00  0.00           N",
            "ATOM   1898  C   SER A  -2       2.763   0.950  12.434  1.00  0.00           C",
        ]
    )
    fixed = insert_ter_records(pdb).splitlines()
    assert fixed[2].startswith("TER    1896")
    assert fixed[2][21] == "C"


def test_prepare_pdb_for_rosetta_renumbers_negative_chain_residues():
    pdb = "\n".join(
        [
            "ATOM   1894  CA  SER C 128      55.442  44.105  24.759  1.00  0.00           C",
            "ATOM   1898  C   SER A  -2       2.763   0.950  12.434  1.00  0.00           C",
            "ATOM   1904  C   GLN A  -1       2.198  -0.804  15.436  1.00  0.00           C",
        ]
    )
    fixed = prepare_pdb_for_rosetta(pdb).splitlines()
    assert fixed[0][22:27] == "   1 "
    assert fixed[2][22:27] == "   1 "
    assert fixed[3][22:27] == "   2 "


def test_normalize_interface_score_columns_drops_ambiguous_total_score():
    df = pd.DataFrame(
        {
            "structure_id": ["a", "b"],
            "total_score": [0.0, -10.0],
            "dslf_fa13": [pd.NA, 0.0],
            "fa_atr": [pd.NA, -1.0],
            "fa_dun": [pd.NA, 2.0],
            "fa_elec": [pd.NA, -1.0],
            "fa_intra_rep": [pd.NA, 0.1],
            "fa_intra_sol_xover4": [pd.NA, 0.1],
            "fa_rep": [pd.NA, 1.0],
            "fa_sol": [pd.NA, 1.0],
            "hbond_bb_sc": [pd.NA, -1.0],
            "hbond_lr_bb": [pd.NA, -1.0],
            "hbond_sc": [pd.NA, -1.0],
            "hbond_sr_bb": [pd.NA, -1.0],
            "lk_ball_wtd": [pd.NA, -1.0],
            "omega": [pd.NA, 0.0],
            "p_aa_pp": [pd.NA, -1.0],
            "pro_close": [pd.NA, 0.0],
            "rama_prepro": [pd.NA, 0.0],
            "ref": [pd.NA, 1.0],
            "yhh_planarity": [pd.NA, 0.0],
        }
    )
    normalized = normalize_interface_score_columns(df)
    assert "total_score" not in normalized.columns
    assert pd.isna(normalized.loc[0, "fullpose_total_score"])
    assert normalized.loc[1, "fullpose_total_score"] == -10.0


def test_normalize_interface_score_columns_does_not_promote_zero_without_regular_terms():
    df = pd.DataFrame({"structure_id": ["a"], "total_score": [0.0]})
    normalized = normalize_interface_score_columns(df)
    assert "total_score" not in normalized.columns
    assert pd.isna(normalized.loc[0, "fullpose_total_score"])


def test_build_paired_delta_table_computes_relaxed_minus_native():
    base = {
        "structure_id": ["s1"],
        "pdb_id": ["1abc"],
        "nanobody_chain": ["B"],
        "antigen_chain": ["A"],
        "interface": ["B_A"],
        "fullpose_total_score": [-10.0],
        "dG_separated": [-5.0],
        "dG_separated/dSASAx100": [-1.0],
        "dSASA_int": [100.0],
        "sc_value": [0.6],
        "hbonds_int": [4.0],
        "delta_unsatHbonds": [3.0],
        "nres_int": [20.0],
    }
    native = pd.DataFrame(base)
    relaxed = native.copy()
    relaxed["dG_separated"] = [-7.5]
    relaxed["sc_value"] = [0.7]
    paired = build_paired_delta_table(native, relaxed)
    assert paired.loc[0, "delta_dG_separated"] == -2.5
    assert paired.loc[0, "delta_sc_value"] == pytest.approx(0.1)
    assert bool(paired.loc[0, "interface_dg_improved"])
    assert bool(paired.loc[0, "shape_complementarity_improved"])


def test_greedy_max_min_select_is_deterministic_on_synthetic_features():
    df = pd.DataFrame(
        {
            "structure_id": ["center", "left", "right"],
            "antigen_length": [100, 60, 200],
            "dSASA_int": [1000, 400, 2200],
            "dG_separated": [-20, -5, -70],
            "sc_value": [0.7, 0.5, 0.85],
            "cdr3_interface_fraction": [0.4, 0.1, 0.8],
            "residue_contact_pairs_5A": [40, 10, 90],
        }
    )
    selected = greedy_max_min_select(
        df,
        ["antigen_length", "dSASA_int", "dG_separated", "sc_value", "cdr3_interface_fraction", "residue_contact_pairs_5A"],
        2,
    )
    assert selected["structure_id"].tolist() == ["center", "right"]


def test_boltz_yaml_contains_four_condition_specific_blocks():
    seq = boltz_yaml("AAAA", "CCCC", "sequence_only", "", "", "", [1, 2], [(1, 1)])
    monomer = boltz_yaml("AAAA", "CCCC", "monomer_templates", "nb.cif", "ag.cif", "", [1, 2], [(1, 1)])
    pocket = boltz_yaml("AAAA", "CCCC", "pocket_contacts", "", "", "", [1, 2], [(1, 1)])
    full = boltz_yaml("AAAA", "CCCC", "full_complex_template", "", "", "complex.cif", [1, 2], [(1, 1)])
    assert "templates:" not in seq
    assert "constraints:" not in seq
    assert "nb.cif" in monomer and "ag.cif" in monomer
    assert "pocket:" in pocket and "contact:" in pocket and "max_distance: 8.0" in pocket
    assert "complex.cif" in full and "template_id: [Nxp, Axp]" in full and "threshold: 1.0" in full


def test_write_remapped_pdb_standardizes_chains(tmp_path):
    atoms = [
        Atom("B", 5, "", "ALA", "CA", "C", 0.0, 0.0, 0.0),
        Atom("A", 9, "", "GLY", "CA", "C", 5.0, 0.0, 0.0),
    ]
    out = write_remapped_pdb(atoms, tmp_path / "template.pdb", {"B": BOLTZ_CHAIN_NANOBODY, "A": BOLTZ_CHAIN_ANTIGEN})
    text = out.read_text()
    assert "ALA N   1" in text
    assert "GLY A   1" in text


def test_parse_confidence_json_and_prediction_discovery(tmp_path):
    pred = tmp_path / "model_0.pdb"
    pred.write_text("END\n")
    (tmp_path / "confidence_model_0.json").write_text('{"confidence_score": 0.91, "iptm": 0.82, "complex_plddt": 88.0}')
    assert discover_boltz_predictions(tmp_path) == [pred]
    parsed = parse_confidence_json(pred)
    assert parsed["confidence_status"] == "ok"
    assert parsed["confidence_score"] == 0.91
    assert parsed["iptm"] == 0.82


def test_contact_recovery_and_dockq_like_on_toy_interface():
    native_atoms = toy_interface_atoms(antigen_offset=3.0)
    pred_atoms = toy_interface_atoms(antigen_offset=3.0)
    decoy_atoms = toy_interface_atoms(antigen_offset=30.0)
    native_contacts = contact_sets(native_atoms, BOLTZ_CHAIN_NANOBODY, BOLTZ_CHAIN_ANTIGEN)
    recovered = compute_interface_recovery(native_atoms, pred_atoms, native_contacts)
    decoy = compute_interface_recovery(native_atoms, decoy_atoms, native_contacts)
    assert recovered["native_contact_recovery"] == 1.0
    assert decoy["native_contact_recovery"] == 0.0
    assert dockq_like(1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert correctness_label(0.85, 0.75) == "true_like"
    assert correctness_label(0.20, 0.20) == "decoy_like"


def test_structure_quality_labels_native_copy_and_translated_decoy():
    native_atoms = toy_interface_atoms(antigen_offset=3.0)
    native_contacts = contact_sets(native_atoms, BOLTZ_CHAIN_NANOBODY, BOLTZ_CHAIN_ANTIGEN)
    true_like = compute_structure_quality(native_atoms, toy_interface_atoms(antigen_offset=3.0), native_contacts)
    decoy = compute_structure_quality(native_atoms, toy_interface_atoms(antigen_offset=30.0), native_contacts)
    assert true_like["correctness_label"] == "true_like"
    assert true_like["dockq_like"] == pytest.approx(1.0)
    assert decoy["fnat"] == 0.0
    assert decoy["correctness_label"] == "decoy_like"


def test_decoy_sequence_identity_and_contact_mutation_filters():
    native = "ACDEFGHIKL"
    designed = "YYYYYYYYYY"
    contacts = [1, 3, 5, 7, 9]
    assert sequence_identity(designed, native) == 0.0
    assert contact_mutation_fraction(designed, native, contacts) == 1.0
    status = sequence_filter_status(designed, native, contacts)
    assert status["sequence_filter_status"] == "ok"
    assert status["mutated_contact_positions"] == 5


def test_decoy_sequence_filter_rejects_identity_and_length_mismatch():
    native = "ACDEFGHIKL"
    assert sequence_filter_status(native, native, [1, 2])["sequence_filter_status"] == "failed_thresholds"
    assert sequence_filter_status("ACD", native, [1, 2])["sequence_filter_status"] == "length_mismatch"


def _non_native_sequence(index: int, length: int = 10) -> str:
    alphabet = "CDEFGHIKLMNPQRSTVWY"
    chars = []
    value = index
    for _ in range(length):
        chars.append(alphabet[value % len(alphabet)])
        value //= len(alphabet)
    return "".join(chars)


def test_select_mpnn_redesign_candidates_limits_to_five_unique_designs():
    native = "AAAAAAAAAA"
    candidates = [
        {"design_id": f"d{i:03d}", "structure_id": "s1", "designed_sequence": _non_native_sequence(i), "mpnn_score": float(i)}
        for i in range(150)
    ]
    rows, selected = select_mpnn_redesign_candidates(candidates, native, list(range(1, 11)), max_decoy_sequences_per_structure=5)
    assert len(rows) == 150
    assert len(selected) == 5
    assert [row["mpnn_rank"] for row in selected] == [1, 2, 3, 4, 5]
    assert all(row["selected_for_folding"] for row in selected)
    assert {row["sequence_role"] for row in selected} == {SEQUENCE_ROLE_REDESIGNED_DECOY}


def test_select_mpnn_redesign_candidates_excludes_native_and_duplicates():
    native = "AAAAAAAAAA"
    candidates = [
        {"design_id": "native", "structure_id": "s1", "designed_sequence": native, "mpnn_score": -10.0},
        {"design_id": "best", "structure_id": "s1", "designed_sequence": "CCCCCCCCCC", "mpnn_score": -5.0},
        {"design_id": "dup", "structure_id": "s1", "designed_sequence": "CCCCCCCCCC", "mpnn_score": -6.0},
        {"design_id": "next", "structure_id": "s1", "designed_sequence": "DDDDDDDDDD", "mpnn_score": -4.0},
    ]
    rows, selected = select_mpnn_redesign_candidates(candidates, native, list(range(1, 11)), max_decoy_sequences_per_structure=5)
    assert [row["design_id"] for row in selected] == ["best", "next"]
    status_by_id = {row["design_id"]: row["sequence_filter_status"] for row in rows}
    assert status_by_id["native"] == "native_sequence"
    assert status_by_id["dup"] == "duplicate"


def test_decoy_summary_counts_redesigns_and_native_controls_separately():
    cases = pd.DataFrame([{"structure_id": "s1", "requested_mpnn_sequences": 150}])
    candidates = pd.DataFrame(
        [
            {"structure_id": "s1", "sequence_filter_status": "ok", "selected_for_folding": True, "requested_mpnn_sequences": 150, "max_decoy_sequences_for_folding": 5},
            {"structure_id": "s1", "sequence_filter_status": "ok", "selected_for_folding": True, "requested_mpnn_sequences": 150, "max_decoy_sequences_for_folding": 5},
            {"structure_id": "s1", "sequence_filter_status": "duplicate", "selected_for_folding": False, "requested_mpnn_sequences": 150, "max_decoy_sequences_for_folding": 5},
        ]
    )
    validation = pd.DataFrame(
        [
            {"structure_id": "s1", "sequence_role": SEQUENCE_ROLE_NATIVE_CONTROL, "validation_status": "native_reference", "n_fold_predictions": 5, "selected_decoy": False},
            {"structure_id": "s1", "sequence_role": SEQUENCE_ROLE_REDESIGNED_DECOY, "validation_status": "pass", "n_fold_predictions": 5, "selected_decoy": True},
            {"structure_id": "s1", "sequence_role": SEQUENCE_ROLE_REDESIGNED_DECOY, "validation_status": "failed_fold", "n_fold_predictions": 5, "selected_decoy": False},
        ]
    )
    summary = summarize_decoy_validation(cases, candidates, validation).iloc[0]
    assert summary["requested_mpnn_sequences"] == 150
    assert summary["max_decoy_sequences_for_folding"] == 5
    assert summary["valid_unique_redesigns"] == 2
    assert summary["selected_redesign_sequences_for_folding"] == 2
    assert summary["native_control_fold_jobs"] == 1
    assert summary["fold_pass"] == 1
    assert summary["selected_decoys"] == 1
    assert summary["backend_predictions"] == 15

def test_parse_mpnn_candidates_reads_fasta_scores(tmp_path):
    out = tmp_path / "case" / "seqs"
    out.mkdir(parents=True)
    (out / "designs.fa").write_text(">native score=0.0\nAAAA\n>sample score=-1.25\nCCCC\n")
    rows = parse_mpnn_candidates(tmp_path / "case", "case")
    assert [r["designed_sequence"] for r in rows] == ["AAAA", "CCCC"]
    assert rows[1]["mpnn_score"] == -1.25


def test_monomer_validation_metrics_pass_native_copy_and_fail_translation():
    native = [
        Atom("A", 1, "", "ALA", "CA", "C", 0.0, 0.0, 0.0),
        Atom("A", 2, "", "ALA", "CA", "C", 1.0, 0.0, 0.0),
        Atom("A", 3, "", "ALA", "CA", "C", 2.0, 0.0, 0.0),
    ]
    copy = list(native)
    sparse = [Atom("A", 1, "", "ALA", "CA", "C", 10.0, 0.0, 0.0)]
    perfect = monomer_validation_metrics(native, copy)
    assert perfect["monomer_ca_rmsd"] == pytest.approx(0.0)
    # an identical fold scores TM-score 1.0 (sum of L identical terms / L)
    assert perfect["monomer_tm_score"] == pytest.approx(1.0)
    assert monomer_validation_metrics(native, sparse)["monomer_aligned_fraction"] == 0.0
    assert monomer_validation_metrics(native, sparse)["monomer_tm_score"] == 0.0


def test_contact_omit_dict_forbids_native_aa_at_contacts():
    native = "ACDEFGHIK"
    # contacts at 1-indexed positions 2, 5, 9 -> native residues C, F, K
    d = contact_omit_dict("8qf5_A_B_antigen_chain_A", [2, 5, 9], native)
    entries = d["8qf5_A_B_antigen_chain_A"]["A"]
    assert entries == [[[2], "C"], [[5], "F"], [[9], "K"]]
    # out-of-range positions are dropped, not crash
    assert contact_omit_dict("x", [99], native)["x"]["A"] == []


def test_ranked_contacts_and_partial_forcing_selection():
    from nbia.decoys import ranked_contact_positions, select_forced_contacts

    # nanobody chain N near antigen residues 1 (2 contacts) and 3 (1 contact); res 2 none
    atoms = [
        Atom("N", 1, "", "GLY", "CA", "C", 0.0, 0.0, 0.0),
        Atom("N", 2, "", "GLY", "CA", "C", 0.0, 0.0, 1.0),
        Atom("A", 1, "", "ALA", "CA", "C", 0.0, 0.0, 2.0),  # near both N atoms -> degree 2
        Atom("A", 2, "", "ALA", "CA", "C", 50.0, 0.0, 0.0),  # far -> not a contact
        Atom("A", 3, "", "ALA", "CA", "C", 0.0, 0.0, 4.0),  # near N res2 only -> degree 1
    ]
    ranked = ranked_contact_positions(atoms, "N", "A", cutoff=5.0)
    assert ranked == [1, 3]  # most-buried (degree 2) first
    assert select_forced_contacts(ranked, 1.0) == [1, 3]
    assert select_forced_contacts(ranked, 0.5) == [1]  # top 50% by burial, positional order


def test_proteinmpnn_command_includes_omit_jsonl():
    from pathlib import Path as _P

    cmd = proteinmpnn_command(_P("ag.pdb"), _P("out"), 150, 0.3, _P("omit.jsonl"))
    assert "--omit_AA_jsonl omit.jsonl" in cmd
    assert "--sampling_temp 0.3" in cmd
    # omitted when not provided
    assert "--omit_AA_jsonl" not in proteinmpnn_command(_P("ag.pdb"), _P("out"), 150)


def test_sequence_filter_gates_on_contact_mutation_not_identity():
    native = "AAAAAAAAAA"
    contacts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # near-native sequence (high identity) but all contacts mutated -> passes
    all_contacts_mutated = "CCCCCCCCCC"
    s = sequence_filter_status(all_contacts_mutated, native, contacts)
    assert s["sequence_filter_status"] == "ok"
    assert s["contact_mutation_fraction"] == 1.0
    # two contacts left native -> 0.8 < 0.90 -> fails regardless of overall identity
    two_native = "AACCCCCCCC"
    assert sequence_filter_status(two_native, native, contacts)["sequence_filter_status"] == "failed_thresholds"
    # partial forcing (0.5): only 5/10 contacts are forced; threshold scales to 0.45
    # a sequence with 5/10 contacts mutated (0.5 >= 0.45) should pass
    half_mutated = "CCCCCAAAAA"  # contacts 1-5 mutated, 6-10 native; contact_mut = 0.5
    s2 = sequence_filter_status(half_mutated, native, contacts, forced_contact_fraction=0.5)
    assert s2["sequence_filter_status"] == "ok"
    assert s2["contact_mutation_fraction"] == pytest.approx(0.5)


def test_tm_score_penalizes_displacement_and_partial_coverage():
    import numpy as np
    from nbia.decoys import tm_score

    ref = np.array([[float(i), 0.0, 0.0] for i in range(30)])
    assert tm_score(ref, ref, 30) == pytest.approx(1.0)
    # a rigid 5 A displacement on every residue drops TM-score well below 1
    shifted = ref + np.array([5.0, 0.0, 0.0])
    assert tm_score(ref, shifted, 30) < 0.5
    # matching only half the residues caps TM-score at ~0.5 via the L_target denominator
    assert tm_score(ref[:15], ref[:15], 30) == pytest.approx(0.5)


def test_core_locked_alignment_ignores_flexible_tail():
    import numpy as np
    from nbia.decoys import core_locked_alignment, tm_score, superpose

    # 60-residue helix-like trace; identical core, last 6 residues flap far away in mob
    ref = np.array([[float(i), 0.0, 0.0] for i in range(60)])
    mob = ref.copy()
    mob[-6:] += np.array([0.0, 25.0, 0.0])  # flailing tail
    # Kabsch superposition is pulled by the tail and depresses the core fit
    _, kabsch = superpose(ref, mob)
    tm_kabsch = tm_score(ref, kabsch, 60)
    # core-locked superposition locks onto the rigid core, tail contributes ~0
    core = core_locked_alignment(ref, mob, 60)
    tm_core = tm_score(ref, core, 60)
    assert tm_core > tm_kabsch
    assert tm_core > 0.85  # 54/60 residues align perfectly -> high score


def test_adaptive_forced_fraction_interface_dominated_vs_small():
    from nbia.decoys import adaptive_forced_fraction

    # interface-small (13% contacts) -> force all
    assert adaptive_forced_fraction(18, 141) == 1.0
    # interface-dominated (39% contacts) -> partial force
    assert adaptive_forced_fraction(28, 72) == 0.5
    # boundary just above 25% -> partial
    assert adaptive_forced_fraction(30, 100) == 0.5


def _native_complex_atoms(n_ag_res: int = 4):
    """Native complex: nanobody chain B, antigen chain C with CA+CB side chains."""
    ca = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 1.5)]
    atoms = [Atom("B", 1, "", "TYR", "CA", "C", 10.0, 10.0, 10.0), Atom("B", 1, "", "TYR", "CB", "C", 11.0, 10.0, 10.0)]
    for i in range(n_ag_res):
        x, y, z = ca[i]
        atoms.append(Atom("C", i + 1, "", "ALA", "CA", "C", x, y, z))
        atoms.append(Atom("C", i + 1, "", "ALA", "CB", "C", x + 0.5, y + 0.5, z + 0.5))
    return atoms


def _write_pred_pdb(antigen_atoms, path):
    """Write an antigen-only 'predicted monomer' PDB (chain A) for graft tests."""
    lines = [format_atom(i + 1, a, "A", a.residue_number, a.residue_name) for i, a in enumerate(antigen_atoms)]
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_pdb_atom_name_field_column_positions():
    # 1-char element, short name → leading space (starts column 14); element field unchanged.
    assert pdb_atom_name_field("CA", "C") == " CA "
    assert pdb_atom_name_field("CD1", "C") == " CD1"
    # 4-char name or 2-char element → starts column 13.
    assert pdb_atom_name_field("HD11", "H") == "HD11"
    assert pdb_atom_name_field("FE", "FE") == "FE  "
    line = format_atom(1, Atom("A", 1, "", "ALA", "CA", "C", 0.0, 0.0, 0.0), "A", 1, "ALA")
    assert line[12:16] == " CA "          # atom-name columns 13-16
    assert line[76:78] == " C"            # element right-justified, ending column 78


def test_graft_decoy_complexes_excludes_native_control_rows(tmp_path):
    native = _native_complex_atoms()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_pdb = raw_dir / "s1.pdb"
    raw_pdb.write_text("\n".join(format_atom(i + 1, a, a.chain_id, a.residue_number, a.residue_name) for i, a in enumerate(native)) + "\nEND\n")
    ag = [a for a in native if a.chain_id == "C"]
    pred_decoy = _write_pred_pdb(ag, tmp_path / "pred_decoy.pdb")
    pred_native = _write_pred_pdb(ag, tmp_path / "pred_native.pdb")
    case_manifest = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {
                "structure_id": "s1",
                "source_filename": "s1.pdb",
                "nanobody_chain": "B",
                "antigen_chain": "C",
            }
        ]
    ).to_csv(case_manifest, index=False)
    validation = tmp_path / "validation.csv"
    pd.DataFrame(
        [
            {"structure_id": "s1", "design_id": "s1_native", "sequence_role": SEQUENCE_ROLE_NATIVE_CONTROL, "is_native_reference": True, "validation_status": "native_reference", "selected_decoy": True, "prediction_path": str(pred_native)},
            {"structure_id": "s1", "design_id": "s1_decoy", "sequence_role": SEQUENCE_ROLE_REDESIGNED_DECOY, "is_native_reference": False, "validation_status": "pass", "selected_decoy": True, "prediction_path": str(pred_decoy)},
        ]
    ).to_csv(validation, index=False)
    out = graft_decoy_complexes(validation, case_manifest, raw_dir, tmp_path / "work", tmp_path / "grafted.csv")
    assert out["design_id"].tolist() == ["s1_decoy"]
    assert out.loc[0, "graft_status"] == "ok"

def test_graft_native_onto_itself_is_identity(tmp_path):
    native = _native_complex_atoms()
    ag = [a for a in native if a.chain_id == "C"]
    pred = _write_pred_pdb(ag, tmp_path / "pred.pdb")
    result = graft_complex(native, "B", "C", pred, tmp_path / "decoy_NA.pdb")
    assert result["graft_status"] == "ok"
    assert result["monomer_graft_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert result["graft_antigen_completeness"] == pytest.approx(1.0)
    text = (tmp_path / "decoy_NA.pdb").read_text()
    assert " CA " in text and " CB " in text          # antigen side chains retained
    chains = {ln[21] for ln in text.splitlines() if ln.startswith("ATOM")}
    assert chains == {"A", "N"}                        # A=antigen, N=nanobody


def test_graft_recovers_known_rigid_transform(tmp_path):
    import numpy as np
    native = _native_complex_atoms()
    ag = [a for a in native if a.chain_id == "C"]
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([5.0, -3.0, 2.0])
    moved = [Atom("A", a.residue_number, "", a.residue_name, a.atom_name, a.element, *(rot @ np.array([a.x, a.y, a.z]) + shift)) for a in ag]
    pred = _write_pred_pdb(moved, tmp_path / "pred.pdb")
    result = graft_complex(native, "B", "C", pred, tmp_path / "decoy_NA.pdb")
    assert result["graft_status"] == "ok"
    # PDB coordinates are written at 3-decimal precision, so a perfect rigid
    # transform recovers to ~1e-3, not machine epsilon.
    assert result["monomer_graft_rmsd"] == pytest.approx(0.0, abs=5e-3)


def test_graft_partial_ca_match_reports_completeness(tmp_path):
    native = _native_complex_atoms(n_ag_res=4)
    ag = [a for a in native if a.chain_id == "C" and a.residue_number <= 3]   # predicted misses residue 4
    pred = _write_pred_pdb(ag, tmp_path / "pred.pdb")
    result = graft_complex(native, "B", "C", pred, tmp_path / "decoy_NA.pdb")
    assert result["graft_status"] == "ok"
    assert result["matched_ca"] == 3
    assert result["graft_antigen_completeness"] == pytest.approx(0.75)


def test_graft_rejects_missing_or_non_pdb_prediction(tmp_path):
    native = _native_complex_atoms()
    assert graft_complex(native, "B", "C", None, tmp_path / "x.pdb")["graft_status"] == "missing_prediction"
    cif = tmp_path / "pred.cif"
    cif.write_text("data_\n")
    assert graft_complex(native, "B", "C", cif, tmp_path / "x.pdb")["graft_status"] == "prediction_not_pdb"


def test_compute_decoy_metrics_writes_header_valid_empty_comparison(tmp_path):
    out = compute_decoy_metrics(
        relax_manifest_csv=tmp_path / "missing_relax_manifest.csv",
        tables_dir=tmp_path,
        work_dir=tmp_path / "work",
        run_rosetta=False,
        run_protein_interface=False,
        run_sc=False,
    )
    comparison_path = tmp_path / "decoy_native_comparison.csv"
    assert comparison_path.exists()
    comparison = pd.read_csv(comparison_path)
    assert list(comparison.columns) == ["design_id", "structure_id"]
    assert comparison.empty
    assert list(out["comparison"].columns) == ["design_id", "structure_id"]


def test_build_decoy_native_comparison_joins_native_baseline(tmp_path):
    # protein_interface AND shape_complementarity both emit `sc` (identical values);
    # the native protein_interface `sc` is unpopulated (NaN) while the dedicated
    # native_shape_complementarity table carries the real Lawrence-Colman value.
    pi = pd.DataFrame([{"design_id": "d1", "structure_id": "s1", "metrics_status": "ok", "hbond_density": 0.4, "sc": 0.68}])
    sc = pd.DataFrame([{"design_id": "d1", "structure_id": "s1", "sc": 0.68, "median_distance": 0.5}])
    ros = pd.DataFrame([{"design_id": "d1", "structure_id": "s1", "dG_separated": -30.0}])
    pd.DataFrame([{"structure_id": "s1", "hbond_density": 0.9, "sc": float("nan")}]).to_csv(tmp_path / "native_protein_interface.csv", index=False)
    pd.DataFrame([{"structure_id": "s1", "path": "x.pdb", "sc": 0.78, "median_distance": 0.44, "status": "ok"}]).to_csv(tmp_path / "native_shape_complementarity.csv", index=False)
    pd.DataFrame([{"structure_id": "s1", "dG_separated": -45.0, "sc_value": 0.8, "dSASA_int": 125.0, "hbonds_int": 3, "delta_unsatHbonds": 1}]).to_csv(tmp_path / "rosetta_interface_native.csv", index=False)
    out = build_decoy_native_comparison(pi, sc, ros, tmp_path)
    assert out.loc[0, "hbond_density"] == 0.4
    assert out.loc[0, "native_hbond_density"] == 0.9
    # no _x/_y suffix collision from the duplicated `sc` column
    assert "sc_x" not in out.columns and "native_sc_x" not in out.columns
    assert out.loc[0, "sc"] == 0.68
    # native_sc must come from the dedicated SC table (0.78), not the NaN pi column
    assert out.loc[0, "native_sc"] == 0.78
    assert "native_path" not in out.columns
    # native Rosetta dG join
    assert out.loc[0, "native_dG_separated"] == -45.0


def test_parse_anarci_numbered_residues_maps_sequence_indices_and_insertions():
    numbered = [((26, " "), "Q"), ((27, " "), "A"), ((111, "A"), "G"), ((112, " "), "-"), ((112, " "), "Y")]
    rows = parse_anarci_numbered_residues(numbered)
    assert [r["sequence_index"] for r in rows] == [1, 2, 3, 4]
    assert rows[1]["imgt_region"] == "CDR1"
    assert rows[2]["anarci_label"] == "111A"
    assert rows[3]["anarci_label"] == "112"


def test_vhh_contact_sequence_indices_uses_4p5_angstrom_cutoff():
    atoms = [
        Atom("N", 1, "", "ALA", "CA", "C", 0.0, 0.0, 0.0),
        Atom("N", 2, "", "GLY", "CA", "C", 10.0, 0.0, 0.0),
        Atom("A", 1, "", "ASP", "OD1", "O", 0.0, 0.0, 4.4),
        Atom("A", 2, "", "ASP", "OD1", "O", 10.0, 0.0, 4.6),
    ]
    assert vhh_contact_sequence_indices(atoms, "N", "A", 4.5) == {1}


def test_aggregate_contact_frequency_denominators_and_insertions():
    df = pd.DataFrame(
        [
            {"structure_id": "s1", "anarci_number": 27, "anarci_insertion": "", "anarci_label": "27", "anarci_sort_key": 27.0, "imgt_region": "CDR1", "is_contact_4p5A": 1},
            {"structure_id": "s2", "anarci_number": 27, "anarci_insertion": "", "anarci_label": "27", "anarci_sort_key": 27.0, "imgt_region": "CDR1", "is_contact_4p5A": 0},
            {"structure_id": "s1", "anarci_number": 111, "anarci_insertion": "A", "anarci_label": "111A", "anarci_sort_key": 111.01, "imgt_region": "CDR3", "is_contact_4p5A": 1},
        ]
    )
    out = aggregate_contact_frequency(df, total_structures=4)
    pos27 = out[out["anarci_label"] == "27"].iloc[0]
    assert pos27["n_numbered"] == 2
    assert pos27["n_contact"] == 1
    assert pos27["contact_frequency_numbered"] == pytest.approx(0.5)
    assert pos27["contact_frequency_all_structures"] == pytest.approx(0.25)
    assert "111A" in set(out["anarci_label"])


def test_vhh_contact_report_includes_methodology_and_figures(tmp_path):
    frequency = pd.DataFrame(
        [
            {
                "anarci_label": "105",
                "imgt_region": "CDR3",
                "n_numbered": 2,
                "n_contact": 2,
                "contact_frequency_numbered": 1.0,
                "contact_frequency_all_structures": 0.5,
            }
        ]
    )
    summary = pd.DataFrame([{"structure_id": "s1", "anarci_status": "ok"}, {"structure_id": "s2", "anarci_status": "failed:ValueError"}])
    write_report(tmp_path, frequency, summary, 4.5)
    report = (tmp_path / "report.md").read_text()
    assert "4.5" in report
    assert "ANARCI with IMGT" in report
    assert "figures/vhh_contact_frequency_by_anarci.png" in report


def toy_interface_atoms(antigen_offset: float) -> list[Atom]:
    atoms = []
    for chain, start_x in [(BOLTZ_CHAIN_NANOBODY, 0.0), (BOLTZ_CHAIN_ANTIGEN, antigen_offset)]:
        for resi in range(1, 4):
            atoms.extend(
                [
                    Atom(chain, resi, "", "ALA", "N", "N", start_x + resi, 0.0, 0.0),
                    Atom(chain, resi, "", "ALA", "CA", "C", start_x + resi, 1.0, 0.0),
                    Atom(chain, resi, "", "ALA", "C", "C", start_x + resi, 2.0, 0.0),
                    Atom(chain, resi, "", "ALA", "O", "O", start_x + resi, 3.0, 0.0),
                ]
            )
    return atoms
