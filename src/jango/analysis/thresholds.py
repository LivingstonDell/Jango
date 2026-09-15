"""Shared analysis threshold parsing helpers."""

from __future__ import annotations

import argparse
import math
import os

DEFAULT_DOCKQ_PRESERVATION_THRESHOLD = 0.49
DOCKQ_PRESERVATION_THRESHOLD_ENV = "DOCKQ_PRESERVATION_THRESHOLD"
DEFAULT_DOCKQ_HIGH_QUALITY_THRESHOLD = 0.80
DOCKQ_HIGH_QUALITY_THRESHOLD_ENV = "DOCKQ_HIGH_QUALITY_THRESHOLD"


def parse_unit_interval(value: object, *, name: str = "threshold") -> float:
    """Parse a finite value in the closed unit interval."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a number between 0 and 1") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError(f"{name} must be a finite number between 0 and 1")
    return parsed


def parse_dockq_preservation_threshold(value: object) -> float:
    """Parse a DockQ preservation threshold for CLI use."""

    return parse_unit_interval(value, name="--structure-threshold")


def default_dockq_preservation_threshold() -> float:
    """Return the configured DockQ preservation threshold default."""

    value = os.environ.get(DOCKQ_PRESERVATION_THRESHOLD_ENV)
    if value is None or value.strip() == "":
        return DEFAULT_DOCKQ_PRESERVATION_THRESHOLD
    return parse_unit_interval(value, name=DOCKQ_PRESERVATION_THRESHOLD_ENV)


def parse_dockq_high_quality_threshold(value: object) -> float:
    """Parse a high-quality DockQ threshold for ML export filtering."""

    return parse_unit_interval(value, name="--dockq-high-quality-threshold")


def default_dockq_high_quality_threshold() -> float:
    """Return the configured high-quality DockQ export threshold."""

    value = os.environ.get(DOCKQ_HIGH_QUALITY_THRESHOLD_ENV)
    if value is None or value.strip() == "":
        return DEFAULT_DOCKQ_HIGH_QUALITY_THRESHOLD
    return parse_unit_interval(value, name=DOCKQ_HIGH_QUALITY_THRESHOLD_ENV)
