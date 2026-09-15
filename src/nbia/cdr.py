from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .numbering import imgt_region, number_vhh_sequence_imgt


@dataclass(frozen=True)
class CDRAnnotation:
    by_index: dict[int, str]
    method: str
    status: str


def annotate_cdrs(sequence: str) -> CDRAnnotation:
    """Return 1-based sequence-index CDR labels.

    The pipeline records ANARCI availability explicitly. A conservative VHH
    sequence-index fallback keeps the atlas runnable and auditable when ANARCI
    is not installed or cannot number a chain.
    """
    if importlib.util.find_spec("anarci") is not None:
        annotation = _try_anarci(sequence)
        if annotation is not None:
            return annotation
    return _heuristic_vhh_annotation(sequence, "heuristic_no_anarci")


def _try_anarci(sequence: str) -> CDRAnnotation | None:
    try:
        numbering = number_vhh_sequence_imgt(sequence, name="nanobody")
        by_index = {int(row["sequence_index"]): str(row["imgt_region"]) for row in numbering}
        return CDRAnnotation(by_index=by_index, method="anarci_imgt", status="ok")
    except Exception as exc:
        fallback = _heuristic_vhh_annotation(sequence, "heuristic_after_anarci_failure")
        return CDRAnnotation(fallback.by_index, fallback.method, f"anarci_failed:{type(exc).__name__}")



def _heuristic_vhh_annotation(sequence: str, method: str) -> CDRAnnotation:
    # IMGT-like approximate positions for a single-domain VHH sequence.
    cdr1 = range(27, 39)
    cdr2 = range(56, 66)
    cdr3_start = max(95, len(sequence) - 24)
    cdr3_stop = max(cdr3_start, len(sequence) - 8)
    cdr3 = range(cdr3_start, cdr3_stop + 1)
    by_index: dict[int, str] = {}
    for idx in range(1, len(sequence) + 1):
        if idx in cdr1:
            by_index[idx] = "CDR1"
        elif idx in cdr2:
            by_index[idx] = "CDR2"
        elif idx in cdr3:
            by_index[idx] = "CDR3"
        else:
            by_index[idx] = "FR"
    return CDRAnnotation(by_index=by_index, method=method, status="fallback")

