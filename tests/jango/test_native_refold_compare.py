from __future__ import annotations

from pathlib import Path

import pandas as pd

from jango.analysis.native_refold_compare import main


def atom_line(serial: int, atom: str, resname: str, chain: str, resseq: int, x: float, y: float, z: float) -> str:
    return f"ATOM  {serial:5d} {atom:<4} {resname:>3} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom[0]:>2}\n"


def write_chain(path: Path, chain: str, offset: float = 0.0) -> None:
    lines = []
    serial = 1
    for idx, resname in enumerate(["ALA", "CYS", "ASP", "GLU", "PHE"], start=1):
        lines.append(atom_line(serial, "CA", resname, chain, idx, float(idx), offset, 0.0))
        serial += 1
    path.write_text("".join(lines) + "END\n")


def test_native_refold_compare_writes_similarity_and_metric_outputs(tmp_path: Path) -> None:
    native_pdb = tmp_path / "native.pdb"
    refold_pdb = tmp_path / "refold.pdb"
    write_chain(native_pdb, "A", offset=0.0)
    write_chain(refold_pdb, "A", offset=2.0)
    pairs = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {
                "structure_id": "1abc_H_A",
                "antigen_chain": "A",
                "native_relaxed_pdb": str(native_pdb),
                "refold_relaxed_pdb": str(refold_pdb),
            }
        ]
    ).to_csv(pairs, index=False)
    native_metrics = tmp_path / "native_metrics.csv"
    refold_metrics = tmp_path / "refold_metrics.csv"
    pd.DataFrame([{"structure_id": "1abc_H_A", "dG_separated": -12.0, "dSASA_int": 800.0, "packstat": 0.60, "sc_value": 0.70}]).to_csv(native_metrics, index=False)
    pd.DataFrame([{"structure_id": "1abc_H_A", "dG_separated": -9.0, "dSASA_int": 760.0, "packstat": 0.55, "sc_value": 0.64}]).to_csv(refold_metrics, index=False)
    out_dir = tmp_path / "out"

    rc = main([
        "--pairs-csv",
        str(pairs),
        "--native-metrics",
        str(native_metrics),
        "--refold-metrics",
        str(refold_metrics),
        "--out-dir",
        str(out_dir),
        "--refold-label",
        "OpenDDE native refold",
    ])

    assert rc == 0
    similarity = pd.read_csv(out_dir / "native_refold_similarity.csv")
    assert similarity.loc[0, "similarity_status"] == "ok"
    assert similarity.loc[0, "percent_structural_similarity"] > 99.0
    assert similarity.loc[0, "antigen_ca_rmsd"] < 1e-6
    assert similarity.loc[0, "delta_dG_separated"] == 3.0
    assert (out_dir / "native_refold_metric_summary.csv").is_file()
    assert (out_dir / "native_refold_metric_long.csv").is_file()
    assert (out_dir / "figures" / "native_refold_similarity_distribution.png").is_file()
    assert (out_dir / "figures" / "native_refold_metric_pairs.png").is_file()
    assert (out_dir / "figures" / "native_refold_metric_scatter.png").is_file()


def test_native_refold_compare_accepts_distinct_native_and_refold_antigen_chains(tmp_path: Path) -> None:
    native_pdb = tmp_path / "native.pdb"
    refold_pdb = tmp_path / "refold.pdb"
    write_chain(native_pdb, "B", offset=0.0)
    write_chain(refold_pdb, "A", offset=2.0)
    pairs = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {
                "structure_id": "1abc_H_B",
                "native_antigen_chain": "B",
                "refold_antigen_chain": "A",
                "native_relaxed_pdb": str(native_pdb),
                "refold_relaxed_pdb": str(refold_pdb),
            }
        ]
    ).to_csv(pairs, index=False)
    out_dir = tmp_path / "out"

    rc = main(["--pairs-csv", str(pairs), "--out-dir", str(out_dir)])

    assert rc == 0
    similarity = pd.read_csv(out_dir / "native_refold_similarity.csv")
    assert similarity.loc[0, "similarity_status"] == "ok"
    assert similarity.loc[0, "antigen_chain"] == "B->A"
    assert similarity.loc[0, "native_antigen_chain"] == "B"
    assert similarity.loc[0, "refold_antigen_chain"] == "A"


def test_native_refold_compare_preserves_missing_file_failure(tmp_path: Path) -> None:
    native_pdb = tmp_path / "native.pdb"
    write_chain(native_pdb, "A")
    pairs = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {
                "structure_id": "1abc_H_A",
                "antigen_chain": "A",
                "native_relaxed_pdb": str(native_pdb),
                "refold_relaxed_pdb": str(tmp_path / "missing.pdb"),
            }
        ]
    ).to_csv(pairs, index=False)
    out_dir = tmp_path / "out"

    rc = main(["--pairs-csv", str(pairs), "--out-dir", str(out_dir)])

    assert rc == 0
    similarity = pd.read_csv(out_dir / "native_refold_similarity.csv")
    assert similarity.loc[0, "similarity_status"] == "failed"
    assert "not found" in similarity.loc[0, "similarity_error"]
