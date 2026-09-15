from __future__ import annotations

import pandas as pd

from jango.analysis.dockq_component_figures import main as component_main
from jango.analysis.high_high_figures import main as high_high_main
from jango.analysis.reference_benchmark_figures import main as reference_benchmark_main
from jango.cli import _COMMAND_MODULES


def _landscape_rows(backend: str) -> list[dict[str, object]]:
    return [
        {
            "structure_id": "case_a",
            "design_id": f"{backend}_a_1",
            "monomer_tm_score": 94.0,
            "monomer_ca_rmsd": 1.4,
            "dockq": 0.72,
            "dockq_fnat": 0.48,
            "dockq_irmsd": 0.9,
            "dockq_lrmsd": 1.2,
            "delta_vs_native_dG_separated": 8.5,
            "binding_perturbation": 4.0,
        },
        {
            "structure_id": "case_a",
            "design_id": f"{backend}_a_2",
            "monomer_tm_score": 91.0,
            "monomer_ca_rmsd": 1.9,
            "dockq": 0.64,
            "dockq_fnat": 0.42,
            "dockq_irmsd": 1.1,
            "dockq_lrmsd": 1.4,
            "delta_vs_native_dG_separated": 6.0,
            "binding_perturbation": 3.0,
        },
        {
            "structure_id": "case_b",
            "design_id": f"{backend}_b_1",
            "monomer_tm_score": 88.0,
            "monomer_ca_rmsd": 2.5,
            "dockq": 0.41,
            "dockq_fnat": 0.22,
            "dockq_irmsd": 2.3,
            "dockq_lrmsd": 3.2,
            "delta_vs_native_dG_separated": -2.0,
            "binding_perturbation": -1.0,
        },
    ]


def _write_inputs(root) -> dict[str, str]:
    paths: dict[str, str] = {}
    for mode in ("hotspot", "interface"):
        for backend in ("esm", "opendde"):
            path = root / f"{mode}_{backend}_landscape.csv"
            pd.DataFrame(_landscape_rows(backend)).to_csv(path, index=False)
            paths[f"{mode}_{backend}"] = str(path)

    refold = root / "native_refold.csv"
    pd.DataFrame(
        [
            {"structure_id": "case_a", "similarity_status": "ok", "antigen_ca_tm_score": 93.0},
            {"structure_id": "case_b", "similarity_status": "ok", "antigen_ca_tm_score": 87.0},
        ]
    ).to_csv(refold, index=False)
    paths["refold"] = str(refold)
    return paths


def test_canonical_figure_commands_are_registered() -> None:
    assert _COMMAND_MODULES["high-high-figures"] == "jango.analysis.high_high_figures"
    assert _COMMAND_MODULES["reference-benchmark-figures"] == "jango.analysis.reference_benchmark_figures"
    assert _COMMAND_MODULES["dockq-component-figures"] == "jango.analysis.dockq_component_figures"


def test_high_high_and_component_commands_write_expected_packages(tmp_path) -> None:
    inputs = _write_inputs(tmp_path)
    high_high_root = tmp_path / "high_high"
    component_root = tmp_path / "components"

    rc = high_high_main(
        [
            "--output-root",
            str(high_high_root),
            "--hotspot-esm-landscape",
            inputs["hotspot_esm"],
            "--hotspot-opendde-landscape",
            inputs["hotspot_opendde"],
            "--interface-esm-landscape",
            inputs["interface_esm"],
            "--interface-opendde-landscape",
            inputs["interface_opendde"],
            "--esm-native-refold",
            inputs["refold"],
            "--opendde-native-refold",
            inputs["refold"],
        ]
    )
    assert rc == 0
    assert (high_high_root / "hotspot" / "esm" / "tables" / "high_high_decoy_ranked_ml_dataset.csv").is_file()
    summary = pd.read_csv(high_high_root / "comparison" / "high_high_core_figure_summary.csv")
    assert len(summary) == 4
    assert set(summary["redesign_mode"]) == {"hotspot", "interface"}
    assert set(summary["backend"]) == {"esm", "opendde"}

    rc = component_main([
        "--input-root",
        str(high_high_root),
        "--output-root",
        str(component_root),
        "--esm-backend-key",
        "esm",
    ])
    assert rc == 0
    component_summary = pd.read_csv(component_root / "comparison" / "dockq_component_figure_summary.csv")
    assert len(component_summary) == 12
    assert set(component_summary["component"]) == {"dockq_fnat", "dockq_irmsd", "dockq_lrmsd"}
    assert len(list(component_root.rglob("*.png"))) == 24


def test_reference_benchmark_command_writes_reference_package(tmp_path) -> None:
    inputs = _write_inputs(tmp_path)
    output_root = tmp_path / "reference_benchmark"

    rc = reference_benchmark_main(
        [
            "--output-root",
            str(output_root),
            "--references",
            "relaxed",
            "--relaxed-hotspot-esm-landscape",
            inputs["hotspot_esm"],
            "--relaxed-hotspot-opendde-landscape",
            inputs["hotspot_opendde"],
            "--relaxed-interface-esm-landscape",
            inputs["interface_esm"],
            "--relaxed-interface-opendde-landscape",
            inputs["interface_opendde"],
            "--esm-native-refold",
            inputs["refold"],
            "--opendde-native-refold",
            inputs["refold"],
        ]
    )

    assert rc == 0
    summary = pd.read_csv(output_root / "summary_stats.csv")
    assert len(summary) == 6
    assert set(summary["reference_mode"]) == {"relaxed"}
    assert len(summary[summary["redesign_mode"] == "native_refold_quality"]) == 2
    assert len(summary[summary["redesign_mode"] != "native_refold_quality"]) == 4
    assert (output_root / "reference_relaxed" / "hotspot" / "esmfold2" / "figures" / "landscape_dockq_delta_dg.png").is_file()
    assert (
        output_root
        / "native_refold_quality"
        / "refold_vs_relaxed"
        / "esmfold2"
        / "figures"
        / "native_refold_tm_histogram.png"
    ).is_file()
    assert (output_root / "reference_relaxed" / "hotspot" / "comparison" / "figures" / "backend_comparison_landscape_dockq_delta_dg.png").is_file()
    inventory = pd.read_csv(output_root / "plot_inventory.csv")
    assert {"decoy_vs_reference", "backend_comparison", "redesign_mode_comparison"} <= set(inventory["figure_type"])
