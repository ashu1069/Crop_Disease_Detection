#!/usr/bin/env bash
# Reviewer #6: transformer / attention models vs the best CNN, all crops, 5-fold.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

MODELS=(vit_small swin_tiny convnext_tiny deit_small densexnet_cbam)
CROPS=(potato okra cowpea cluster_bean)

for crop in "${CROPS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== $crop / $model (5-fold) ==="
    uv run python -m cropdl.run_train --config configs/transformers.yaml \
      --all-folds --set data.crop="$crop" model.name="$model"
  done
done
