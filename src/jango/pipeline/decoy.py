"""Artifact-oriented decoy command group backed by NBIA primitives."""

from __future__ import annotations

import argparse
from typing import Sequence

NBIA_DECOY_COMMANDS = {
    "graft": "decoy-graft",
    "relax": "decoy-relax",
    "dockq": "decoy-dockq",
    "metrics": "decoy-metrics",
    "report": "decoy-report",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jango decoy",
        description="Run modular post-fold decoy stages using the canonical NBIA implementations.",
    )
    parser.add_argument("stage", choices=sorted(NBIA_DECOY_COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected NBIA decoy command.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    from nbia.cli import main as nbia_main

    return int(nbia_main([NBIA_DECOY_COMMANDS[ns.stage], *list(ns.args)]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
