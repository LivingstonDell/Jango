"""Backend protocol seeds for DELPHI decoy folding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


MsaRequirement = Literal["required", "optional", "unsupported"]


@dataclass(frozen=True)
class MsaSettings:
    """MSA policy requested for a folding backend."""

    requirement: MsaRequirement
    cache_dir: Path | None = None


@dataclass(frozen=True)
class FoldRequest:
    """Backend-agnostic fold request produced by `decoy-fold`."""

    design_id: str
    sequence: str
    output_dir: Path
    msa_path: Path | None = None


class DecoyFoldBackend(Protocol):
    """Protocol future Boltz2, ESMFold2, and OpenDDE adapters should satisfy."""

    name: str
    msa_requirement: MsaRequirement

    def validate_request(self, request: FoldRequest) -> None:
        """Validate backend-specific request requirements."""

    def build_command(self, request: FoldRequest) -> tuple[str, ...]:
        """Return the command tuple without executing external tools."""
