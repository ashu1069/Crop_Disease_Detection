#!/usr/bin/env bash
# Pilot: fold-0 only, every model x every crop. Validates the full grid quickly
# before committing to 5-fold CV. Resilient — a single run failing (e.g. OOM)
# does not abort the sweep; failures are recorded in the log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

CROPS=(potato okra cowpea cluster_bean)
CNN_MODELS=(densenet121 densenet169 densenet201 densenet161 densexnet densexnet_cbam)
TFM_MODELS=(vit_small swin_tiny convnext_tiny deit_small)

LOG=outputs/pilot_status.log
mkdir -p outputs
: > "$LOG"

run() {  # $1 config, $2 crop, $3 model
  echo ">>> $(date +%H:%M:%S) START $2/$3" | tee -a "$LOG"
  if uv run python -m cropdl.run_train --config "$1" \
       --set data.crop="$2" model.name="$3" data.fold=0 \
            wandb.group="pilot-$2-$3" \
       >> "outputs/pilot_${2}_${3}.out" 2>&1; then
    echo "<<< $(date +%H:%M:%S) OK    $2/$3" | tee -a "$LOG"
  else
    echo "!!! $(date +%H:%M:%S) FAIL  $2/$3 (see outputs/pilot_${2}_${3}.out)" | tee -a "$LOG"
  fi
}

for crop in "${CROPS[@]}"; do
  for m in "${CNN_MODELS[@]}"; do run configs/densenet_variants.yaml "$crop" "$m"; done
  for m in "${TFM_MODELS[@]}"; do run configs/transformers.yaml "$crop" "$m"; done
done

echo "=== PILOT COMPLETE $(date) ===" | tee -a "$LOG"
