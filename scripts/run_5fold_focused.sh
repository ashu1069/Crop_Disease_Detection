#!/usr/bin/env bash
# Focused 5-fold CV (reviewer #3): folds 1-4 for the discriminative model set on
# the three non-saturated crops. Fold 0 already exists from the pilot, so
# aggregation (stats/analyze.py) combines f0..f4. Okra is excluded (ceiling effect).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

CROPS=(potato cowpea cluster_bean)
CNN_MODELS=(densenet121 densenet201 densexnet)
TFM_MODELS=(convnext_tiny vit_small swin_tiny)
FOLDS=(1 2 3 4)

LOG=outputs/cv_status.log
mkdir -p outputs
: > "$LOG"

run() {  # $1 config, $2 crop, $3 model, $4 fold
  echo ">>> $(date +%H:%M:%S) START $2/$3/f$4" | tee -a "$LOG"
  if uv run python -m cropdl.run_train --config "$1" \
       --set data.crop="$2" model.name="$3" data.fold="$4" \
            wandb.group="cv-$2-$3" \
       >> "outputs/cv_${2}_${3}_f${4}.out" 2>&1; then
    echo "<<< $(date +%H:%M:%S) OK    $2/$3/f$4" | tee -a "$LOG"
  else
    echo "!!! $(date +%H:%M:%S) FAIL  $2/$3/f$4 (see outputs/cv_${2}_${3}_f${4}.out)" | tee -a "$LOG"
  fi
}

for crop in "${CROPS[@]}"; do
  for f in "${FOLDS[@]}"; do
    for m in "${CNN_MODELS[@]}"; do run configs/densenet_variants.yaml "$crop" "$m" "$f"; done
    for m in "${TFM_MODELS[@]}"; do run configs/transformers.yaml "$crop" "$m" "$f"; done
  done
done

echo "=== CV COMPLETE $(date) ===" | tee -a "$LOG"
