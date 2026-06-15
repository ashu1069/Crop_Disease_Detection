#!/usr/bin/env bash
# Reviewer #1: DenseXnet vs DenseNet variants, all crops, 5-fold CV.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

MODELS=(densenet121 densenet169 densenet201 densenet161 densexnet densexnet_cbam)
CROPS=(potato okra cowpea cluster_bean)

for crop in "${CROPS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== $crop / $model (5-fold) ==="
    uv run python -m cropdl.run_train --config configs/densenet_variants.yaml \
      --all-folds --set data.crop="$crop" model.name="$model"
  done
done
