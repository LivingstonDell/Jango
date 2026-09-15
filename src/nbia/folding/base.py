from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .msa import BackendMSASupport, MSAArtifact


@dataclass(frozen=True)
class FoldJob:
    """One monomer-folding job."""

    design_id: str
    structure_id: str
    sequence: str
    input_path: Path
    output_dir: Path
    command: str
    backend: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldPrediction:
    """One predicted monomer structure returned/discovered by a folding backend."""

    prediction_path: Path
    confidence: dict[str, object] = field(default_factory=dict)
    backend: str = ""


class FoldBackend(Protocol):
    """Common interface implemented by all folding backends."""

    name: str
    msa_support: BackendMSASupport

    def write_input(
        self,
        *,
        design_id: str,
        structure_id: str,
        sequence: str,
        work_dir: Path,
        msa_artifact: MSAArtifact | None = None,
    ) -> tuple[Path, Path]:
        """Write backend-specific input files and return (input_path, output_dir)."""

    def command(self, input_path: Path, output_dir: Path) -> str:
        """Return the shell command used by the job bundle."""

    def write_job_bundle(self, jobs: list[FoldJob], bundle_dir: Path) -> None:
        """Write backend-specific TSV + runner shell script."""

    def discover_predictions(self, output_dir: Path) -> list[FoldPrediction]:
        """Discover completed predictions in output_dir."""
