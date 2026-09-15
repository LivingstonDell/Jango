"""Thin Jango umbrella CLI for orchestration commands."""

from __future__ import annotations

import argparse
import importlib
from typing import Sequence

_COMMAND_MODULES = {
    "native": "jango.pipeline.nanobody_pipeline",
    "nanobody-pipeline": "jango.pipeline.nanobody_pipeline",
    "decoy-prefold": "jango.pipeline.decoy_prefold",
    "decoy-fold": "jango.pipeline.decoy_fold",
    "decoy": "jango.pipeline.decoy",
    "decoy-analyze": "jango.pipeline.decoy_analyze",
    "decoy-fold-relax": "jango.pipeline.decoy_fold_relax",
    "decoy-results": "jango.pipeline.decoy_results",
    "core-figures": "jango.analysis.core_figures",
    "high-high-figures": "jango.analysis.high_high_figures",
    "reference-benchmark-figures": "jango.analysis.reference_benchmark_figures",
    "dockq-component-figures": "jango.analysis.dockq_component_figures",
    "mutation-profile": "jango.analysis.mutation_profile",
    "evaluation-package": "jango.analysis.evaluation_package",
    "lineage-audit": "jango.analysis.lineage_audit",
    "fold": "jango.pipeline.fold",
    "fold-jobs": "jango.pipeline.fold_jobs",
    "fett": "jango.pipeline.fett",
    "fold-qc": "jango.pipeline.fold_qc",
    "native-refold-compare": "jango.analysis.native_refold_compare",
    "rosetta-qc": "jango.pipeline.rosetta_qc",
    "interface-analyzer": "jango.pipeline.interface_analyzer",
    "analyze-interface": "jango.pipeline.interface_analyzer",
    "prep-sabdab": "jango.preprocessing.sabdab_pdb",
    "prep-sabdab2": "jango.preprocessing.sabdab2_cif",
    "delphi-prep-sabdab": "jango.preprocessing.sabdab_pdb",
    "delphi-prep-sabdab2": "jango.preprocessing.sabdab2_cif",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jango",
        description="Jango orchestration CLI. Scientific primitives remain available through nbia.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted([*_COMMAND_MODULES, "doctor"]),
        help="Workflow command to delegate.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the delegated command.")
    return parser


def _doctor(argv: Sequence[str]) -> int:
    from jango.doctor import main as doctor_main

    return doctor_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.command is None:
        parser.print_help()
        return 0
    if ns.command == "doctor":
        return _doctor(ns.args)
    module = importlib.import_module(_COMMAND_MODULES[ns.command])
    return int(module.main(list(ns.args)) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
