# Validation Checklist

Run this before handoff.

## Environment

```bash
cd <JANGO_REPO>
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango
jango doctor
```

Expected: `FAIL - none`.

## Public package checks

```bash
find data/outputs/evaluation -type f -name '*.png' | wc -l
find data/outputs/evaluation -type f -iname '*interface*density*'

bad_esm_a='esm[ _-]?'
bad_esm_b='esmfold2_'
bad_esm_c='ESMFold2 '
bad_density='interface[_ -]?'
bad_binding='binding '
bad_pattern="${bad_esm_a}diffusion|${bad_esm_b}diffusion|${bad_esm_c}diffusion|${bad_density}density|${bad_binding}perturbed"

grep -RIn --include='*.md' --include='*.py' --include='*.csv' --include='*.json' \
  -E "$bad_pattern" \
  README.md docs src scripts tests data/outputs/evaluation || true
```

Expected:

- PNG count: `21`
- no retired map files
- no stale public-label grep hits

## Focused tests

```bash
pytest -q \
  tests/jango/test_canonical_figure_commands.py \
  tests/jango/test_interface_analyzer_cli.py \
  tests/nbia/test_esmfold2_msa_runner.py
```

Expected: all pass.
