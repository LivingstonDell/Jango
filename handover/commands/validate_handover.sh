#!/usr/bin/env bash
set -euo pipefail
cd <JANGO_REPO>
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango

jango doctor

png_count=$(find handover/figures -type f -name '*.png' | wc -l)
test "$png_count" -eq 21

test -z "$(find data/outputs/evaluation -type f -iname '*interface*density*' -print)"

bad_esm_a='esm[ _-]?'
bad_esm_b='esmfold2_'
bad_esm_c='ESMFold2 '
bad_density='interface[_ -]?'
bad_binding='binding '
bad_pattern="${bad_esm_a}diffusion|${bad_esm_b}diffusion|${bad_esm_c}diffusion|${bad_density}density|${bad_binding}perturbed"

if grep -RIn --include='*.md' --include='*.py' --include='*.csv' --include='*.json' \
  -E "$bad_pattern" \
  README.md docs src scripts tests data/outputs/evaluation; then
  echo 'stale public label found' >&2
  exit 1
fi

pytest -q \
  tests/jango/test_canonical_figure_commands.py \
  tests/jango/test_interface_analyzer_cli.py \
  tests/nbia/test_esmfold2_msa_runner.py

echo "handover validation passed"
