#!/usr/bin/env bash
# Reviewer #4/#5/#8: dehazing quality, classification robustness under haze, and
# Grad-CAM. Reuses fold-0 checkpoints from the pilot (no retraining of classifiers).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

LOG=outputs/dehaze_status.log
mkdir -p outputs/dehaze
: > "$LOG"

# per-crop transformer winner (for Grad-CAM alongside densexnet)
declare -A WINNER=( [potato]=convnext_tiny [cowpea]=vit_small \
                    [cluster_bean]=swin_tiny [okra]=vit_small )

step() { echo ">>> $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

for crop in okra cowpea cluster_bean potato; do
  dx_ckpt="outputs/densenet_variants-${crop}-densexnet-f0/best.pt"
  aod="outputs/dehaze/aodnet_${crop}.pt"

  step "[$crop] train AOD-Net dehazer (20 ep)"
  uv run python -m cropdl.dehaze.train_aodnet --crop "$crop" --epochs 20 \
    --out "$aod" >> "outputs/dehaze_${crop}_aodtrain.out" 2>&1 \
    && echo "    aodnet OK" | tee -a "$LOG" || echo "    aodnet FAIL" | tee -a "$LOG"

  step "[$crop] dehazing quality PSNR/SSIM (DCP vs AOD-Net, 150 imgs)"
  uv run python -m cropdl.dehaze.evaluate_quality --crop "$crop" \
    --methods dcp aodnet --n-images 150 --aod-weights "$aod" \
    >> "outputs/dehaze_${crop}_quality.out" 2>&1 \
    && echo "    quality OK" | tee -a "$LOG" || echo "    quality FAIL" | tee -a "$LOG"

  step "[$crop] robustness eval on densexnet: clean/hazy/dcp/aodnet"
  uv run python -m cropdl.run_eval --config configs/dehazing.yaml \
    --ckpt "$dx_ckpt" --conditions clean hazy hazy_dcp hazy_aodnet \
    --aod-weights "$aod" \
    --set data.crop="$crop" model.name=densexnet \
    >> "outputs/dehaze_${crop}_robust.out" 2>&1 \
    && echo "    robust OK" | tee -a "$LOG" || echo "    robust FAIL" | tee -a "$LOG"

  step "[$crop] Grad-CAM: densexnet"
  uv run python -m cropdl.explain.gradcam --config configs/densenet_variants.yaml \
    --ckpt "$dx_ckpt" --n-per-class 4 \
    --set data.crop="$crop" model.name=densexnet \
    >> "outputs/dehaze_${crop}_gradcam.out" 2>&1 \
    && echo "    gradcam-densexnet OK" | tee -a "$LOG" || echo "    gradcam-densexnet FAIL" | tee -a "$LOG"

  win="${WINNER[$crop]}"
  win_ckpt="outputs/transformers-${crop}-${win}-f0/best.pt"
  step "[$crop] Grad-CAM: $win (winner)"
  uv run python -m cropdl.explain.gradcam --config configs/transformers.yaml \
    --ckpt "$win_ckpt" --n-per-class 4 \
    --set data.crop="$crop" model.name="$win" \
    >> "outputs/dehaze_${crop}_gradcam_win.out" 2>&1 \
    && echo "    gradcam-$win OK" | tee -a "$LOG" || echo "    gradcam-$win FAIL" | tee -a "$LOG"
done

echo "=== DEHAZE/GRADCAM COMPLETE $(date) ===" | tee -a "$LOG"
