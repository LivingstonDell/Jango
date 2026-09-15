#!/usr/bin/env bash
set -euo pipefail
cd <JANGO_REPO>
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango

jango evaluation-package --clean-output

# Optional: remove regeneratable intermediate source packages after publishing.
rm -rf data/outputs/evaluation/reference_benchmark data/outputs/evaluation/mutation_profile
