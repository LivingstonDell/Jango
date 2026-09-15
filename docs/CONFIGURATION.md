# Configuration

Normal users do not edit or source production config files for each run. Fett owns configuration resolution for ordinary runs:

```bash
fett submit --input /path/to/raw/data --run-id validation_50 --max-structures 50
```

Fett derives:

- Jango source root: `<JANGO_REPO>`
- user workspace: `<JANGO_USER_ROOT>`
- run root: `<JANGO_USER_ROOT>/work/<run-id>`
- user cache: `<JANGO_USER_ROOT>/cache`
- default backend: `esmfold2`
- default mode policy: `auto_by_length`
- default max decoy sequences: `5`
- default role/confidence: `validation` / `medium`
- canonical decoy policy: `configs/production/antigen_redesign_decoys.yml`
- backend runtime config from `configs/production/<backend>.env`
- installation-owned Fett stage resources; normal users do not pass CPU, memory, GPU, or wall-time settings

`configs/` remains the administrator-maintained installation surface:

```text
configs/
+-- environment.yml
+-- examples/
|   +-- paths.example.env
|   +-- runtime.<SLURM_NODE>.example.env
+-- test/
|   +-- paths.env
|   +-- esmfold2.env
+-- production/
|   +-- paths.env
|   +-- runtime.<SLURM_NODE>.common.env
|   +-- esmfold2.env
|   +-- boltz2.env
|   +-- antigen_redesign_decoys.yml
+-- test_esm_5_paths.env
+-- test_esm_5_runtime.env
```

For a concrete run, prefer `fett preflight` over manually sourcing production configs. Use `--json` for machine-readable PASS/WARN/FAIL output.

Advanced overrides remain available for administrators: `--paths-config`, `--runtime-config`, `--decoy-config`, `--input-kind`, `--output-root`, `--mode-policy`, `--label`, `--role`, `--confidence`, and `--max-decoy-sequences`. Stage resource overrides are administrator-owned runtime settings, not normal per-run user options; preflight rejects any stage above 24G RAM or any GPU request outside folding.

Reusable cache belongs under `<JANGO_USER_ROOT>/cache`; run-local intermediates belong under `<JANGO_USER_ROOT>/work/<run-id>`; final exported PDBs, tables, and figures belong under `<JANGO_USER_ROOT>/outputs/<run-id>`. Raw input data remains at its original read-only location and is recorded in `inputs/input_manifest.json` rather than copied wholesale.

ML export filtering uses `DOCKQ_HIGH_QUALITY_THRESHOLD` for `q1_high_dockq_ml_decoys.csv`; the default is `0.80`. This is separate from `DOCKQ_PRESERVATION_THRESHOLD`, whose default remains `0.49` for landscape quadrant assignment.
