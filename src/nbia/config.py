"""Package-level configuration models for NBIA infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


RosettaRuntimeKind = Literal["docker", "apptainer", "local", "executable"]


@dataclass(frozen=True)
class NbiaPaths:
    """Filesystem paths used by reusable NBIA commands."""

    raw_dir: Path
    manifest_path: Path
    work_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class RosettaRuntimeConfig:
    """Configuration for a Rosetta runtime adapter."""

    kind: RosettaRuntimeKind
    image: str | None = None
    bin_dir: Path | None = None
    apptainer_cache_dir: Path | None = None
    container_bin_dir: str = "/usr/local/bin"

    def validate(self) -> None:
        kind = "local" if self.kind == "executable" else self.kind
        if kind in {"docker", "apptainer"} and not self.image:
            raise ValueError(f"{kind} Rosetta runtime requires --rosetta-image.")
        if kind == "local" and self.bin_dir is None:
            raise ValueError("Local Rosetta runtime requires --rosetta-bin-dir.")


