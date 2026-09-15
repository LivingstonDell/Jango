#!/usr/bin/env bash
set -euo pipefail
cd <JANGO_REPO>
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango

jango mutation-profile \
  --output-root data/outputs/evaluation/mutation_profile \
  --clean-output
