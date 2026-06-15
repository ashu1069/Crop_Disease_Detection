#!/usr/bin/env bash
# Reviewer #4/#5: dehazing quality + robustness of classification under haze.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

CROPS=(okra cowpea cluster_bean potato)

for crop in "${CROPS[@]}"; do
  echo "=== [$crop] train AOD-Net dehazer on synthetic haze ==="
  uv run python -m cropdl.dehaze.train_aodnet --crop "$crop" --epochs 20 \
    --out "outputs/dehaze/aodnet_${crop}.pt"

  echo "=== [$crop] dehazing image-quality (PSNR/SSIM): DCP vs AOD-Net ==="
  uv run python -m cropdl.dehaze.evaluate_quality --crop "$crop" \
    --methods dcp aodnet --n-images 200 \
    --aod-weights "outputs/dehaze/aodnet_${crop}.pt"

  echo "=== [$crop] train classifier (clean), fold 0 ==="
  uv run python -m cropdl.run_train --config configs/dehazing.yaml \
    --set data.crop="$crop" model.name=densexnet

  echo "=== [$crop] robustness eval: clean / hazy / hazy+dcp / hazy+aodnet ==="
  uv run python -m cropdl.run_eval --config configs/dehazing.yaml \
    --ckpt "outputs/dehazing-${crop}-densexnet-f0/best.pt" \
    --conditions clean hazy hazy_dcp hazy_aodnet \
    --aod-weights "outputs/dehaze/aodnet_${crop}.pt" \
    --set data.crop="$crop" model.name=densexnet
done
