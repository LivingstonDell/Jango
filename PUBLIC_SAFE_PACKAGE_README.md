# Jango Public-Safe Package

This archive contains the public-safe Jango codebase, command-line scripts, tests, documentation, and compact analysis outputs needed to understand and reproduce the published figure/table package.

Excluded from this archive: raw structure files (PDB/CIF), FASTA/A3M/MSA caches, generated work directories, model checkpoints/weights, bulky provenance/source bundles, and private machine-specific paths. Placeholder tokens such as `<JANGO_REPO>`, `<TOOLS_ROOT>`, `<MODEL_ROOT>`, and `<JANGO_USER_ROOT>` mark paths that must be configured in a new environment.

Primary review locations:
- `README.md` for setup and workflow overview.
- `handover/` for figures, compact result tables, inventories, and SOP notes.
- `src/` and `scripts/` for reproducible Jango code.
- `tests/` for smoke/unit validation.
