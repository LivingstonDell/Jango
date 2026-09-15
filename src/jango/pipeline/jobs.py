"""Helpers for generated external-tool job bundles."""

from __future__ import annotations

from pathlib import Path

from .execution import ensure_file


def patch_proteinmpnn_tsv(tsv: Path, proteinmpnn_python: Path) -> None:
    """Force generated ProteinMPNN jobs to use the requested interpreter."""

    ensure_file(tsv, "ProteinMPNN job TSV")
    text = tsv.read_text()
    text = text.replace(
        f"{proteinmpnn_python} $PROTEINMPNN_HOME/protein_mpnn_run.py",
        "python $PROTEINMPNN_HOME/protein_mpnn_run.py",
    )
    text = text.replace(
        "python $PROTEINMPNN_HOME/protein_mpnn_run.py",
        f"{proteinmpnn_python} $PROTEINMPNN_HOME/protein_mpnn_run.py",
    )
    tsv.write_text(text)


def patch_boltz_cache(path: Path, boltz_cache: Path) -> None:
    """Inject the requested Boltz cache into generated shell/TSV jobs."""

    ensure_file(path, "Boltz job file")
    cache = str(boltz_cache.resolve())
    text = path.read_text()
    text = text.replace(f"boltz predict --cache {cache}", "boltz predict")
    text = text.replace(f"$BOLTZ predict --cache {cache}", "$BOLTZ predict")
    text = text.replace("boltz predict", f"boltz predict --cache {cache}")
    text = text.replace("$BOLTZ predict", f"$BOLTZ predict --cache {cache}")
    text = text.replace(f"--cache {cache} --cache {cache}", f"--cache {cache}")
    path.write_text(text)
