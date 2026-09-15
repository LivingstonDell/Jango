from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jango.analysis.lineage_audit import AuditConfig, audit_roots, write_reports


def test_lineage_audit_reports_missing_stale_and_mode_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "data" / "outputs" / "folds" / "esmfold2" / "interface_only"
    existing = root / "monomer_outputs" / "case1.pdb"
    existing.parent.mkdir(parents=True)
    existing.write_text("ATOM\n")
    missing = root / "monomer_outputs" / "missing.pdb"
    hotspot = tmp_path / "data" / "outputs" / "folds" / "esmfold2" / "hotspot_only" / "case2.pdb"
    hotspot.parent.mkdir(parents=True)
    hotspot.write_text("ATOM\n")
    table = root / "manifest.csv"
    pd.DataFrame(
        [
            {"structure_id": "case1", "fold_pdb": str(existing)},
            {"structure_id": "case2", "fold_pdb": str(missing)},
            {"structure_id": "case3", "fold_pdb": "<HOME>/old/case3.pdb"},
            {"structure_id": "case4", "fold_pdb": str(hotspot)},
        ]
    ).to_csv(table, index=False)

    references, issues, summary = audit_roots(
        AuditConfig(
            roots=(tmp_path / "data" / "outputs",),
            allowed_roots=(tmp_path / "data" / "outputs",),
            stale_roots=("<HOME>",),
        )
    )

    assert len(references) == 4
    assert {"missing_path", "stale_root", "outside_allowed_root", "mode_mismatch"}.issubset(
        set(issues["issue_type"])
    )
    row = summary.loc[summary["source_file"].eq(str(table))].iloc[0]
    assert row["path_references"] == 4
    assert row["missing_path"] == 2
    assert row["mode_mismatch"] == 1


def test_lineage_audit_reads_json_and_writes_reports(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    existing = root / "folds" / "esmfold2" / "hotspot_only" / "case1.pdb"
    existing.parent.mkdir(parents=True)
    existing.write_text("ATOM\n")
    manifest = root / "fold_manifest.json"
    manifest.write_text(json.dumps({"folds": [{"pdb_path": str(existing)}]}))

    references, issues, summary = audit_roots(AuditConfig(roots=(root,), allowed_roots=(root,)))
    paths = write_reports(references, issues, summary, tmp_path / "audit")

    assert len(references) == 1
    assert issues.empty
    assert summary["path_references"].sum() == 1
    assert paths["references"].is_file()
    assert paths["issues"].is_file()
    assert paths["summary"].is_file()


def test_lineage_audit_ignores_command_strings(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    table = root / "jobs.tsv"
    pd.DataFrame(
        [
            {
                "fold_command": f"{tmp_path}/bin/python script.py --out {tmp_path}/out.pdb",
                "output_pdb": str(root / "missing.pdb"),
            }
        ]
    ).to_csv(table, sep="\t", index=False)

    references, issues, _summary = audit_roots(AuditConfig(roots=(root,)))

    assert len(references) == 1
    assert references.loc[0, "column"] == "output_pdb"
    assert issues["issue_type"].tolist() == ["missing_path"]


def test_lineage_audit_ignores_slurm_log_templates(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    table = root / "jobs.csv"
    pd.DataFrame([{"bundle_stdout_log": str(root / "%x-%j.out")}]).to_csv(table, index=False)

    references, issues, summary = audit_roots(AuditConfig(roots=(root,)))

    assert references.empty
    assert issues.empty
    assert summary["path_references"].sum() == 0
