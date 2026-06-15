#!/usr/bin/env bash
# Reviewer #8/#9: complexity profiling + Grad-CAM for the headline models.
# (Statistical aggregation/McNemar are produced automatically by --all-folds runs
#  and via `python -m cropdl.stats.analyze`.)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== Computational complexity table (params / FLOPs / latency) ==="
uv run python -m cropdl.profiling.complexity --num-classes 3 \
  --out outputs/complexity.csv --wandb

# Grad-CAM for one representative checkpoint per crop (edit ckpt paths as needed).
# Example for potato / densexnet fold 0:
# uv run python -m cropdl.explain.gradcam --config configs/densenet_variants.yaml \
#   --ckpt outputs/densenet_variants-potato-densexnet-f0/best.pt \
#   --n-per-class 4 --set data.crop=potato model.name=densexnet

# Example McNemar significance test (DenseXnet vs DenseNet121, potato fold 0):
# uv run python -m cropdl.stats.analyze mcnemar \
#   --a outputs/densenet_variants-potato-densexnet-f0 \
#   --b outputs/densenet_variants-potato-densenet121-f0
