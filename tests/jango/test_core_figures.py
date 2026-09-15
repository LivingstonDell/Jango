from __future__ import annotations

from pathlib import Path

import pandas as pd

from jango.analysis.core_figures import (
    BackendSpec,
    ESM_COLOR,
    ESM_DARK,
    FigureConfig,
    OPENDDE_COLOR,
    OPENDDE_DARK,
    load_backend_data,
    write_core_outputs,
)


def _write_inputs(root: Path, backend: str) -> tuple[Path, Path]:
    landscape = root / f"{backend}_landscape.csv"
    native_refold = root / f"{backend}_native_refold.csv"
    pd.DataFrame(
        [
            {
                "structure_id": "case1",
                "design_id": "case1_low",
                "monomer_tm_score": 0.60,
                "monomer_ca_rmsd": 5.0,
                "dockq": 0.50,
                "delta_vs_native_dG_separated": 1.0,
                "binding_perturbation": 0.1,
            },
            {
                "structure_id": "case1",
                "design_id": "case1_high",
                "monomer_tm_score": 0.95,
                "monomer_ca_rmsd": 1.0,
                "dockq": 0.70,
                "delta_vs_native_dG_separated": 2.0,
                "binding_perturbation": 1.4,
            },
            {
                "structure_id": "case2",
                "design_id": "case2",
                "monomer_tm_score": 0.85,
                "monomer_ca_rmsd": 3.0,
                "dockq": 0.30,
                "delta_vs_native_dG_separated": -4.0,
                "binding_perturbation": 1.2,
            },
        ]
    ).to_csv(landscape, index=False)
    pd.DataFrame(
        [
            {"structure_id": "case1", "similarity_status": "ok", "antigen_ca_tm_score": 0.92},
            {"structure_id": "case2", "similarity_status": "ok", "antigen_ca_tm_score": 0.88},
            {"structure_id": "case3", "similarity_status": "failed", "antigen_ca_tm_score": 0.99},
        ]
    ).to_csv(native_refold, index=False)
    return landscape, native_refold


def test_core_figures_write_canonical_outputs(tmp_path: Path) -> None:
    esm_landscape, esm_native_refold = _write_inputs(tmp_path, "esm")
    opendde_landscape, opendde_native_refold = _write_inputs(tmp_path, "opendde")
    config = FigureConfig(evaluation_root=tmp_path / "evaluation")
    backends = [
        load_backend_data(
            BackendSpec("esm", "ESMFold2", ESM_COLOR, ESM_DARK, "o", esm_landscape, esm_native_refold),
            config,
        ),
        load_backend_data(
            BackendSpec("opendde", "OpenDDE", OPENDDE_COLOR, OPENDDE_DARK, "^", opendde_landscape, opendde_native_refold),
            config,
        ),
    ]
    stale_tables = config.evaluation_root / "esm" / "tables"
    stale_tables.mkdir(parents=True)
    (stale_tables / "landscape_top1.csv").write_text("stale\n")
    (stale_tables / "component_metric_summary.csv").write_text("stale\n")

    summary = write_core_outputs(backends, config)

    assert len(summary) == 6
    for backend in ("esm", "opendde"):
        base = config.evaluation_root / backend
        assert (base / "figures" / "landscape_dockq_delta_dg.png").is_file()
        assert (base / "figures" / "native_refold_tm_histogram.png").is_file()
        assert (base / "figures" / "q1_native_refold_tm_histogram.png").is_file()
        assert not (base / "tables" / "landscape_top1.csv").exists()
        assert not (base / "tables" / "component_metric_summary.csv").exists()
        assert (base / "tables" / "core_figure_summary.csv").is_file()
        ml = pd.read_csv(base / "tables" / "decoy_ranked_ml_dataset.csv")
        assert ml.loc[ml["structure_id"].eq("case1"), "design_id"].item() == "case1_high"
        assert ml.loc[ml["structure_id"].eq("case1"), "quadrant"].item() == "Q1"
        assert ml.loc[ml["structure_id"].eq("case1"), "quadrant_rank"].item() == 1
        assert ml.loc[ml["structure_id"].eq("case2"), "quadrant"].item() == "Q2"
        backend_summary = pd.read_csv(base / "tables" / "core_figure_summary.csv")
        landscape_summary = backend_summary[backend_summary["figure"].eq("landscape_dockq_delta_dg")].iloc[0]
        assert landscape_summary["q1_dockq_median"] == 0.70
        assert landscape_summary["q1_delta_dg_median"] == 2.0
        assert landscape_summary["q1_binding_perturbation_median"] == 1.4
        assert landscape_summary["q1_coordinate_score_median"] == 2.1
        assert landscape_summary["q1_native_refold_tm_percent_median"] == 92.0
        assert landscape_summary["q1_native_refold_quality_pass_count"] == 1
    assert (config.evaluation_root / "comparison" / "figures" / "backend_comparison_landscape_dockq_delta_dg.png").is_file()
