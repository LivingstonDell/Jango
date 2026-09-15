from __future__ import annotations

import json
from pathlib import Path

from jango.pipeline.decoy_fold import mirror_mpnn_artifacts, seed_mpnn_input_artifacts


def test_mirror_mpnn_artifacts_preserves_fastas_and_metadata(tmp_path: Path) -> None:
    work_dir = tmp_path / "work" / "mode"
    tables_dir = tmp_path / "tables" / "mode"
    artifact_root = tmp_path / "data" / "mpnn"

    fasta = work_dir / "mpnn_outputs" / "case1" / "seqs" / "case1.fa"
    fasta.parent.mkdir(parents=True)
    fasta.write_text(">native\nAAAA\n>designed\nAAAT\n")
    bundle = work_dir / "job_bundle"
    bundle.mkdir(parents=True)
    (bundle / "proteinmpnn_jobs.tsv").write_text("structure_id\tcommand\ncase1\trun\n")
    (bundle / "run_proteinmpnn_jobs.sh").write_text("#!/usr/bin/env bash\n")

    tables_dir.mkdir(parents=True)
    (tables_dir / "decoy_mpnn_job_manifest.csv").write_text("structure_id,redesign_mode\ncase1,hotspot_only\n")
    (tables_dir / "proteinmpnn_execution_status.csv").write_text("structure_id,job_status\ncase1,completed\n")
    (tables_dir / "decoy_prepare_status.csv").write_text("redesign_mode,prepared_jobs\nhotspot_only,1\n")

    mirror_mpnn_artifacts(
        mode="hotspot_only",
        work_dir=work_dir,
        tables_dir=tables_dir,
        artifact_root=artifact_root,
        experiment="unit_exp",
    )

    target = artifact_root / "hotspot_only"
    assert (target / "mpnn_outputs" / "case1" / "seqs" / "case1.fa").read_text().startswith(">native")
    assert (target / "tables" / "proteinmpnn_execution_status.csv").is_file()
    assert (target / "job_bundle" / "proteinmpnn_jobs.tsv").is_file()
    manifest = json.loads((target / "artifact_manifest.json").read_text())
    assert manifest["experiment"] == "unit_exp"
    assert manifest["redesign_mode"] == "hotspot_only"
    assert manifest["copied_fasta_files"] == 1



def test_seed_mpnn_input_artifacts_reuses_mode_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "data" / "mpnn"
    artifact = artifact_root / "hotspot_only"
    fasta = artifact / "mpnn_outputs" / "case1" / "seqs" / "case1.fa"
    fasta.parent.mkdir(parents=True)
    fasta.write_text(">native\nAAAA\n>design_1\nAAAT\n")
    (artifact / "tables").mkdir()
    (artifact / "tables" / "decoy_mpnn_job_manifest.csv").write_text("structure_id,redesign_mode\ncase1,hotspot_only\n")
    (artifact / "tables" / "proteinmpnn_execution_status.csv").write_text("structure_id,job_status\ncase1,completed\n")
    (artifact / "job_bundle").mkdir()
    (artifact / "job_bundle" / "proteinmpnn_jobs.tsv").write_text("structure_id\ncase1\n")
    (artifact / "artifact_manifest.json").write_text(json.dumps({"redesign_mode": "hotspot_only"}) + "\n")

    work_dir = tmp_path / "run" / "work"
    tables_dir = tmp_path / "run" / "tables"
    seed_mpnn_input_artifacts(
        mode="hotspot_only",
        work_dir=work_dir,
        tables_dir=tables_dir,
        mpnn_input=artifact_root,
    )

    assert (work_dir / "mpnn_outputs" / "case1" / "seqs" / "case1.fa").read_text().startswith(">native")
    assert (tables_dir / "decoy_mpnn_job_manifest.csv").is_file()
    assert (tables_dir / "proteinmpnn_execution_status.csv").is_file()
    assert (work_dir / "job_bundle" / "proteinmpnn_jobs.tsv").is_file()


def test_seed_mpnn_input_artifacts_rejects_mode_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "mpnn" / "hotspot_only"
    fasta = artifact / "mpnn_outputs" / "case1" / "seqs" / "case1.fa"
    fasta.parent.mkdir(parents=True)
    fasta.write_text(">native\nAAAA\n")
    (artifact / "artifact_manifest.json").write_text(json.dumps({"redesign_mode": "interface_only"}) + "\n")

    try:
        seed_mpnn_input_artifacts(
            mode="hotspot_only",
            work_dir=tmp_path / "work",
            tables_dir=tmp_path / "tables",
            mpnn_input=artifact,
        )
    except SystemExit as exc:
        assert "mode mismatch" in str(exc)
    else:
        raise AssertionError("expected mode mismatch failure")
