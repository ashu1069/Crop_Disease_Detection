# Crop Disease Classification — Revision Codebase

PyTorch + timm + Weights & Biases pipeline built to answer the reviewer comments
on *"Deep Learning Approaches for Crop Disease Classification: Adapting to
Environmental Challenges and Hazy Conditions."* One config-driven path for
training, evaluation, dehazing, explainability and profiling, with full W&B
experiment tracking.

## Dataset

Built from the four crop archives (field photos + PlantVillage potato). After
cleaning (227 zero-byte files dropped, 324 HEIC/HEIC-as-jpg files converted):

| Crop | Classes | Images (trainval / test) |
|------|---------|--------------------------|
| potato | early_blight, healthy, late_blight | 2634 / 659 |
| okra | healthy, jasid, powdery | 1421 / 356 |
| cowpea | healthy, leaf_minor | 1104 / 276 |
| cluster_bean | healthy, mosaic | 1236 / 310 |

Split: stratified **80/20 held-out test**, then **stratified 5-fold CV** on the
trainval portion (fixed seed). One classifier per crop.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # if needed
uv sync                                                # installs torch cu121 etc.
wandb login                                            # online tracking
uv run python -m cropdl.data.prepare \
  --root /path/to/extracted --out data/manifest.csv --num-folds 5
```

GPU verified on an RTX 3060 (CUDA 12.1, torch 2.3.1).

## How each reviewer comment is addressed

| # | Comment | What addresses it |
|---|---------|-------------------|
| 1 | DenseXnet vs DenseNet variants | `models/factory.py` (densenet121/161/169/201 + `densexnet`); `scripts/run_densenet_variants.sh` |
| 2 | Training details | All hyperparameters live in `configs/*.yaml`; logged to W&B per run |
| 3 | Statistical validation | 5-fold CV with mean±std/95%CI (`stats/analyze.py aggregate`); McNemar + paired t-test |
| 4 | Dehazing comparison + image quality | `dehaze/` (DCP + AOD-Net), PSNR/SSIM via `dehaze/evaluate_quality.py` |
| 5 | Accuracy under haze | `run_eval.py` scores clean / hazy / dehazed test sets |
| 6 | Transformer/attention models | ViT, Swin, ConvNeXt, DeiT, CBAM-CNN; `scripts/run_transformers.sh` |
| 7 | Dataset/augmentation/resolution/split | This table + `data/prepare.py` (manifest) + augmentation in `data/dataset.py` |
| 8 | Confusion matrices, Grad-CAM, error analysis | Auto per run (`engine/metrics.py`); `explain/gradcam.py` |
| 9 | FLOPs, params, inference time | `profiling/complexity.py` |

## Running experiments

```bash
# Single model, single fold
uv run python -m cropdl.run_train --config configs/densenet_variants.yaml \
  --set data.crop=potato model.name=densexnet data.fold=0

# Full 5-fold CV (prints aggregated mean±std, logs an AGG run)
uv run python -m cropdl.run_train --config configs/densenet_variants.yaml \
  --all-folds --set data.crop=potato model.name=densexnet

# Full sweeps
bash scripts/run_densenet_variants.sh     # reviewer #1
bash scripts/run_transformers.sh          # reviewer #6
bash scripts/run_dehazing.sh              # reviewer #4/#5
bash scripts/run_analysis.sh              # reviewer #9 (+ Grad-CAM examples)
```

## Layout

```
configs/                 experiment configs (one file per study)
src/cropdl/
  config.py              YAML -> dataclass, CLI overrides
  data/prepare.py        scan/clean/convert -> manifest + k-fold splits
  data/dataset.py        datasets, transforms, class weights, loaders
  data/haze.py           atmospheric-scattering haze synthesis
  models/factory.py      one registry for every architecture
  models/densexnet.py    paper's custom head (+ optional CBAM)
  engine/train.py        AMP training loop, early stop, best-ckpt, W&B
  engine/evaluate.py     inference with optional haze/dehaze pipeline
  engine/metrics.py      metric suite + confusion-matrix plots
  dehaze/                DCP, AOD-Net, PSNR/SSIM, quality eval
  explain/gradcam.py     Grad-CAM for CNNs and transformers
  profiling/complexity.py  params / FLOPs / latency
  stats/analyze.py       k-fold aggregation, McNemar, paired t-test
  run_train.py / run_eval.py   top-level entry points
scripts/                 reproducible sweep scripts
```
