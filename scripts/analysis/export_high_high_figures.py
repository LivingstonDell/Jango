#!/usr/bin/env python3
"""Compatibility wrapper for ``jango high-high-figures``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from jango.analysis.high_high_figures import main


if __name__ == "__main__":
    raise SystemExit(main())
