from __future__ import annotations

from pathlib import Path

from jango.pipeline import fold


def test_fold_prepare_runs_sequence_validation_then_msa_resolution(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_decoy_fold_main(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(fold.decoy_fold, "main", fake_decoy_fold_main)
    args = fold.build_parser().parse_args(
        [
            "prepare",
            "--prefold-dir",
            str(tmp_path / "prefold"),
            "--atlas-root",
            str(tmp_path / "repo"),
            "--backend",
            "esmfold2",
            "--mpnn-input",
            str(tmp_path / "data" / "mpnn" / "hotspot_only"),
            "--msa-cache-dir",
            str(tmp_path / "data" / "msa"),
            "--output-root",
            str(tmp_path / "run"),
            "--paths-config",
            str(tmp_path / "paths.env"),
            "--runtime-config",
            str(tmp_path / "runtime.env"),
        ]
    )

    assert fold.prepare_from_args(args) == 0
    assert len(calls) == 2
    assert calls[0][calls[0].index("--fett-stage") + 1] == "sequence_validation"
    assert calls[1][calls[1].index("--fett-stage") + 1] == "msa_resolution"
    for call in calls:
        assert "--skip-mpnn" in call
        assert "--skip-folding" in call
        assert call[call.index("--mpnn-input") + 1].endswith("hotspot_only")
        assert call[call.index("--msa-cache-dir") + 1].endswith("data/msa")


def test_fold_submit_delegates_to_fold_jobs(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_fold_jobs_main(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(fold.fold_jobs, "main", fake_fold_jobs_main)
    args = fold.build_parser().parse_args(
        [
            "submit",
            "--fold-dir",
            str(tmp_path / "fold"),
            "--output-root",
            str(tmp_path / "run"),
            "--execution-mode",
            "slurm",
            "--slurm-gpus",
            "1",
            "--slurm-mem",
            "20G",
        ]
    )

    assert fold.submit_from_args(args) == 0
    assert calls == [
        [
            "submit",
            "--fold-dir",
            str(tmp_path / "fold"),
            "--output-root",
            str(tmp_path / "run"),
            "--execution-mode",
            "slurm",
            "--paths-config",
            "configs/test/paths.env",
            "--runtime-config",
            "configs/test/esmfold2.env",
            "--slurm-bundle-mode",
            "combined",
            "--slurm-gpus",
            "1",
            "--slurm-mem",
            "20G",
        ]
    ]



def test_decoy_command_delegates_to_nbia(monkeypatch) -> None:
    from jango.pipeline import decoy

    calls: list[list[str]] = []

    def fake_nbia_main(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    import nbia.cli

    monkeypatch.setattr(nbia.cli, "main", fake_nbia_main)
    assert decoy.main(["dockq", "--out", "dockq.csv"]) == 0
    assert calls == [["decoy-dockq", "--out", "dockq.csv"]]
