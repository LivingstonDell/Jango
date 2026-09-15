"""Configuration models for DELPHI-VHH orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ModePolicy = Literal["manual", "auto_by_length"]
PipelineStage = Literal["nanobody", "prefold", "fold", "analyze"]


def experiment_name(*, backend: str, policy: str, max_structures: int | None = None, case_count: int | None = None) -> str:
    """Return the canonical experiment namespace for an input-size capped run.

    ``case_count`` is accepted as a compatibility alias for older call sites.
    New code should pass ``max_structures`` because the number caps raw input
    structures before native analysis rather than selecting validated cases.
    """

    if not backend:
        raise ValueError("backend is required; no default backend is allowed")
    value = max_structures if max_structures is not None else case_count
    if value is None or value <= 0:
        raise ValueError("max_structures must be positive")
    if not policy:
        raise ValueError("policy is required")
    return f"{backend}_max{value}_{policy}"


@dataclass(frozen=True)
class DelphiPaths:
    """User-supplied paths for a DELPHI run."""

    input_path: Path
    output_root: Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Minimal experiment identity used across pipeline stages."""

    backend: str
    case_count: int
    policy: str
    output_root: Path

    @property
    def namespace(self) -> str:
        return experiment_name(
            backend=self.backend,
            case_count=self.case_count,
            policy=self.policy,
        )
