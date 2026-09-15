from pathlib import Path

import pytest

from nbia.decoys import build_decoy_cases, write_default_decoy_config


def test_legacy_decoy_config_writer_is_disabled(tmp_path):
    config = tmp_path / "configs" / "antigen_redesign_decoys.yml"

    with pytest.raises(RuntimeError, match="Legacy decoy config generation is disabled"):
        write_default_decoy_config(config)

    assert not config.exists()


def test_build_decoy_cases_requires_existing_config_without_writing(tmp_path):
    case_manifest = tmp_path / "source_cases.csv"
    raw_dir = tmp_path / "raw"
    out_csv = tmp_path / "decoy_cases.csv"
    config = tmp_path / "missing.yml"
    raw_dir.mkdir()
    case_manifest.write_text("structure_id,pdb_id,source_filename,nanobody_chain,antigen_chain\n")

    with pytest.raises(FileNotFoundError, match="Decoy config does not exist"):
        build_decoy_cases(case_manifest, raw_dir, out_csv, config)

    assert not config.exists()
    assert not out_csv.exists()
