from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
from typing import Iterable

CANONICAL_NUMBERING_SCHEME = "ANARCI_IMGT"
CANONICAL_NUMBERING_DESCRIPTION = "ANARCI with IMGT scheme for VHH/nanobody sequences"
IMGT_CDR_SPANS: dict[str, tuple[int, int]] = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}
IMGT_REGION_ORDER = ["FR", "CDR1", "CDR2", "CDR3"]
DISPLAY_REGION = {"FR": "Framework", "CDR1": "CDR1", "CDR2": "CDR2", "CDR3": "CDR3"}


class AnarciUnavailableError(RuntimeError):
    """Raised when an explicit ANARCI runtime is unavailable."""


class AnarciRuntimeError(RuntimeError):
    """Raised when an explicit ANARCI runtime fails to number a sequence."""


def _existing_file(path_value: str | Path | None, label: str) -> Path:
    if path_value is None or not str(path_value).strip():
        raise AnarciUnavailableError(f"{label} is not configured")
    path = Path(path_value).expanduser()
    if not path.exists() or not path.is_file():
        raise AnarciUnavailableError(f"{label} does not exist or is not a file: {path}")
    return path.resolve()


def require_anarci_runtime(anarci_python: str | Path | None = None) -> Path:
    """Validate the explicit ANARCI Python runtime used for IMGT numbering."""

    try:
        return _existing_file(anarci_python, "ANARCI_PYTHON")
    except AnarciUnavailableError as exc:
        if anarci_python is None or not str(anarci_python).strip():
            raise AnarciUnavailableError(
                "ANARCI is required for canonical VHH numbering. Set ANARCI_PYTHON "
                "or ANARCI_BIN, or provide a validated precomputed "
                "vhh_anarci_numbering_per_structure.csv table."
            ) from exc
        raise


def require_anarci_executable(anarci_bin: str | Path | None = None) -> Path:
    """Validate an explicit ANARCI command-line executable."""

    return _existing_file(anarci_bin, "ANARCI_BIN")


def require_anarci(
    anarci_python: str | Path | None = None,
    anarci_bin: str | Path | None = None,
) -> Path:
    """Compatibility validator for callers that preflight ANARCI availability."""

    python = anarci_python or os.environ.get("ANARCI_PYTHON")
    binary = anarci_bin or os.environ.get("ANARCI_BIN")
    if python:
        return require_anarci_runtime(python)
    if binary:
        return require_anarci_executable(binary)
    return require_anarci_runtime(None)


def imgt_region(position: int) -> str:
    """Return the canonical IMGT region for one ANARCI-numbered VHH position."""

    for region, (start, stop) in IMGT_CDR_SPANS.items():
        if start <= int(position) <= stop:
            return region
    return "FR"


def framework_or_cdr(region: str) -> str:
    """Collapse detailed IMGT region labels to Framework/CDR."""

    return "CDR" if str(region).upper().startswith("CDR") else "Framework"


def cdr_name(region: str) -> str:
    """Return CDR1/CDR2/CDR3 for CDR positions and an empty value otherwise."""

    text = str(region).upper()
    return text if text in {"CDR1", "CDR2", "CDR3"} else ""


def display_region(region: str) -> str:
    """Return publication-facing region labels."""

    return DISPLAY_REGION.get(str(region), str(region) or "unmapped")


def canonical_position_label(position: int, insertion: str = "") -> str:
    """Format an ANARCI IMGT position plus optional insertion code."""

    insertion = "" if insertion in {" ", "-", None} else str(insertion).strip()
    return f"{int(position)}{insertion}"


def anarci_sort_key(position: int, insertion: str) -> float:
    """Sort insertion-coded ANARCI residues immediately after their base position."""

    if not insertion:
        return float(position)
    return float(position) + (sum(ord(ch) for ch in insertion.upper()) - 64) / 100.0


def parse_anarci_numbered_residues(
    numbered_residues: Iterable[tuple[tuple[int, str], str]],
    sequence_index_offset: int = 0,
) -> list[dict[str, object]]:
    """Parse ANARCI numbered residues into canonical per-sequence-index rows.

    ANARCI emits gap rows with ``aa == '-'``. Those rows are skipped so
    ``sequence_index`` remains 1-based over residues present in the input VHH.
    ``sequence_index_offset`` preserves any unnumbered N-terminal residues
    reported by ANARCI's domain start coordinate.
    """

    rows: list[dict[str, object]] = []
    sequence_index = int(sequence_index_offset)
    for (position, insertion), aa in numbered_residues:
        if aa == "-":
            continue
        sequence_index += 1
        insertion_text = "" if insertion in {" ", "-"} else str(insertion).strip()
        region = imgt_region(int(position))
        rows.append(
            {
                "sequence_index": sequence_index,
                "anarci_number": int(position),
                "anarci_insertion": insertion_text,
                "anarci_label": canonical_position_label(int(position), insertion_text),
                "anarci_sort_key": anarci_sort_key(int(position), insertion_text),
                "anarci_aa": aa,
                "imgt_region": region,
                "framework_or_cdr": framework_or_cdr(region),
                "cdr_name": cdr_name(region),
                "numbering_scheme": CANONICAL_NUMBERING_SCHEME,
            }
        )
    return rows




def _parse_imgt_label(label: str) -> tuple[int, str] | None:
    match = re.match(r"^(\d+)([A-Za-z]*)$", str(label).strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def number_vhh_sequence_imgt_cli(
    sequence: str,
    name: str = "vhh",
    *,
    anarci_bin: str | Path | None = None,
    scheme: str = "imgt",
    timeout_seconds: int = 60,
) -> list[dict[str, object]]:
    """Number a VHH sequence using the explicit ANARCI command-line executable."""

    executable = require_anarci_executable(anarci_bin)
    env = os.environ.copy()
    env["PATH"] = f"{executable.parent}:{env.get('PATH', '')}"
    with tempfile.TemporaryDirectory(prefix="jango_anarci_") as tmpdir:
        prefix = Path(tmpdir) / "numbered"
        completed = subprocess.run(
            [
                str(executable),
                "-i",
                str(sequence),
                "-s",
                str(scheme),
                "-r",
                "heavy",
                "--csv",
                "-o",
                str(prefix),
            ],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "ANARCI failed").strip()
            raise AnarciRuntimeError(message)
        csv_paths = sorted(Path(tmpdir).glob("numbered_*.csv"))
        if not csv_paths:
            raise AnarciRuntimeError("ANARCI produced no CSV numbering output")
        with csv_paths[0].open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        raise AnarciRuntimeError("ANARCI CSV numbering output was empty")
    row = rows[0]
    sequence_index_offset = int(float(row.get("seqstart_index") or 0))
    metadata = {
        "Id",
        "domain_no",
        "hmm_species",
        "chain_type",
        "e-value",
        "score",
        "seqstart_index",
        "seqend_index",
        "identity_species",
        "v_gene",
        "v_identity",
        "j_gene",
        "j_identity",
    }
    numbered_residues = []
    for label, aa in row.items():
        if label in metadata:
            continue
        parsed = _parse_imgt_label(label)
        if parsed is None:
            continue
        numbered_residues.append((parsed, str(aa or "-")))
    return parse_anarci_numbered_residues(
        numbered_residues,
        sequence_index_offset=sequence_index_offset,
    )


def number_vhh_sequence_imgt(
    sequence: str,
    name: str = "vhh",
    *,
    anarci_python: str | Path | None = None,
    anarci_bin: str | Path | None = None,
    scheme: str = "imgt",
    timeout_seconds: int = 60,
) -> list[dict[str, object]]:
    """Number a VHH sequence with an explicit ANARCI Python runtime.

    Jango intentionally does not import ``anarci`` from the active environment
    implicitly. Callers must either pass a validated precomputed numbering table
    or provide ``ANARCI_PYTHON`` / ``--anarci-python`` pointing at a Python
    executable that can import ANARCI and find its HMMER tools.
    """

    if anarci_python is None:
        env_python = os.environ.get("ANARCI_PYTHON")
        anarci_python = env_python if env_python else None
    if anarci_bin is None:
        env_bin = os.environ.get("ANARCI_BIN")
        anarci_bin = env_bin if env_bin else None

    if anarci_python is None and anarci_bin is not None:
        return number_vhh_sequence_imgt_cli(
            sequence,
            name=name,
            anarci_bin=anarci_bin,
            scheme=scheme,
            timeout_seconds=timeout_seconds,
        )

    runtime = require_anarci_runtime(anarci_python)
    payload = {"name": str(name), "sequence": str(sequence), "scheme": str(scheme)}
    script = textwrap.dedent(
        r'''
        import json
        import sys

        from anarci import anarci

        payload = json.loads(sys.stdin.read())
        numbering, _, _ = anarci(
            [(payload["name"], payload["sequence"])],
            scheme=payload.get("scheme", "imgt"),
            output=False,
        )
        if not numbering or numbering[0] is None:
            raise RuntimeError("ANARCI returned no numbering")
        domain = numbering[0][0]
        numbered_residues = domain[0]
        sequence_index_offset = int(domain[1])
        rows = []
        for (position, insertion), aa in numbered_residues:
            rows.append([int(position), str(insertion), str(aa)])
        print(json.dumps({"numbered_residues": rows, "sequence_index_offset": sequence_index_offset}))
        '''
    )
    env = os.environ.copy()
    env["PATH"] = f"{runtime.parent}:{env.get('PATH', '')}"
    completed = subprocess.run(
        [str(runtime), "-c", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "ANARCI failed").strip()
        raise AnarciRuntimeError(message)
    try:
        decoded = json.loads(completed.stdout)
        sequence_index_offset = int(decoded.get("sequence_index_offset", 0))
        numbered_residues = [
            ((int(position), str(insertion)), str(aa))
            for position, insertion, aa in decoded["numbered_residues"]
        ]
    except Exception as exc:  # noqa: BLE001 - converted to user-facing error.
        raise AnarciRuntimeError(f"Unable to parse ANARCI output: {exc}") from exc
    return parse_anarci_numbered_residues(numbered_residues, sequence_index_offset=sequence_index_offset)
