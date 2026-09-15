#!/usr/bin/env bash
set -euo pipefail
cd <JANGO_REPO>
source <CONDA_ROOT>/etc/profile.d/conda.sh
conda activate jango

SOURCE_ROOT=<JANGO_REPO>/handover/source/reference_benchmark
ST="$SOURCE_ROOT/source_tables"
PREVIEW="$SOURCE_ROOT/preview"
ESM_SOURCE_DIR=esmfold2

jango reference-benchmark-figures \
  --output-root data/outputs/evaluation/reference_benchmark \
  --references refold \
  --esm-backend-key esmfold2 --esm-label ESM \
  --opendde-backend-key opendde --opendde-label OpenDDE \
  --esm-native-refold "$ST/native_refold_quality/refold_vs_relaxed/$ESM_SOURCE_DIR/native_refold_similarity.csv" \
  --opendde-native-refold "$ST/native_refold_quality/refold_vs_relaxed/opendde/native_refold_similarity.csv" \
  --esm-native-refold-crystal "$ST/native_refold_quality/refold_vs_crystal/$ESM_SOURCE_DIR/native_refold_similarity.csv" \
  --opendde-native-refold-crystal "$ST/native_refold_quality/refold_vs_crystal/opendde/native_refold_similarity.csv" \
  --refold-hotspot-esm-landscape "$PREVIEW/reference_refold/hotspot/$ESM_SOURCE_DIR/tables/decoy_ranked_ml_dataset.csv" \
  --refold-interface-esm-landscape "$PREVIEW/reference_refold/interface/$ESM_SOURCE_DIR/tables/decoy_ranked_ml_dataset.csv" \
  --refold-hotspot-opendde-landscape "$PREVIEW/reference_refold/hotspot/opendde/tables/decoy_ranked_ml_dataset.csv" \
  --refold-interface-opendde-landscape "$PREVIEW/reference_refold/interface/opendde/tables/decoy_ranked_ml_dataset.csv"
