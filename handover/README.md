# Jango Handover Package

This package is a compact orientation and reproducibility bundle for Jango. It is meant to help a new user understand what exists, where canonical data live, how to regenerate the report-facing figures, and how to validate that the package is clean.

## Validated scope

- Dataset: SAbDab2 nanobody-antigen benchmark set used for the hotspot/interface analyses.
- Redesign modes: hotspot and interface.
- Folding backends: ESM and OpenDDE.
- Final report-facing evaluation package: `data/outputs/evaluation`.
- Heavy structures are not copied here; this package points to their canonical locations.

## Start here

1. Read `LOCATION_MAP.md`.
2. Read `FULL_SOP.md` for the full step-by-step protocol and design rationale.
3. Use `DATA_LOCATION_TREE.txt` for a quick visual map of the canonical data layout.
4. Use `REPRODUCE_FIGURES.md` to regenerate the compact figure package.
5. Use `VALIDATION_CHECKLIST.md` and `commands/validate_handover.sh` before handing off or publishing.

## Included contents

- `figures/`: flat copy of the 21 canonical report figures.
- `tables/`: compact CSV tables required to regenerate or audit the figures.
- `FULL_SOP.md`: full project SOP, including inputs, outputs, validation checks, debugging notes, and design-choice justifications.
- `DATA_LOCATION_TREE.txt`: tree-style visual map of the shared data, output, and handover layout.
- `commands/`: command templates for regeneration and validation.
- `checksums/`: figure and table inventories with sizes and row counts.
- `maps/`: CSV maps of canonical locations and package contents.

## Not included

- Folded/relaxed PDB directories.
- Slurm logs.
- Temporary work directories.
- Historical non-canonical ESM outputs.

Those artifacts remain in the shared/user Jango data roots listed in `LOCATION_MAP.md`.
