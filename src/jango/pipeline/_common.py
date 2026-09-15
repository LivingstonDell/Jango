"""Shared CLI helpers for DELPHI pipeline entrypoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from jango.paths import resolve_output_root

from .cases import REDESIGN_MODES


MODE_POLICIES = ("manual", "auto_by_length")
MSA_POLICIES = ("required", "optional", "unsupported")
MSA_FAILURE_POLICIES = ("abort", "skip_design", "skip_case")
MSA_MODES = ("required", "optional", "disabled", "backend-managed")
MSA_PROVIDERS = ("precomputed", "mmseqs2", "backend-managed", "abforge_get_or_build", "none")
MSA_FORMATS = ("a3m", "fasta", "csv", "boltz-csv", "stockholm", "unknown")
MSA_PAIRINGS = ("auto", "paired", "unpaired", "none")
ROSETTA_RUNTIMES = ("docker", "apptainer", "local")
FOLD_BACKENDS = ("boltz2", "esmfold2", "opendde")
EXECUTION_MODES = ("local", "slurm")


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


def env_path(name: str) -> Path | None:
    value = env_value(name)
    return Path(value).expanduser() if value else None




def env_int(name: str, default: int | None = None) -> int | None:
    value = env_value(name)
    if value is None:
        return default
    return int(value)
def env_bool(name: str, default: bool = False) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def add_output_root(parser: argparse.ArgumentParser) -> None:
    default = env_path("JANGO_TEST_RUN_ROOT") or env_path("JANGO_RUN_ROOT")
    parser.add_argument(
        "--output-root",
        required=default is None,
        default=default,
        type=Path,
        help="User-specified destination for generated artifacts; never inside the canonical repo.",
    )


def add_atlas_root(parser: argparse.ArgumentParser) -> None:
    default = env_path("JANGO_SOURCE")
    parser.add_argument(
        "--atlas-root",
        required=default is None,
        default=default,
        type=Path,
        help="Path to the installed or checked-out Nanobody Interface Atlas repository.",
    )


def add_plan_only(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-only", action="store_true", help="Print planned work without launching tools.")


def ensure_output_root_outside_repo(output_root: Path) -> Path:
    return resolve_output_root(output_root)


def require_manual_modes(parser: argparse.ArgumentParser, *, mode_policy: str, modes: list[str] | None) -> list[str]:
    if mode_policy == "manual" and not modes:
        parser.error("--modes is required when --mode-policy manual")
    if mode_policy == "auto_by_length" and not modes:
        return list(REDESIGN_MODES)
    return list(modes or [])
