# Experimental Results — Revision

Generated from the cleaned 4-crop dataset (7,996 images) with one config-driven
PyTorch+timm pipeline, all runs tracked in W&B
(`ashu-1069-rochester-institute-of-technology/crop_disease`). Hardware: RTX 3060.

Two evaluation tiers:
- **Pilot** — fold-0, all 10 models × 4 crops (40 runs).
- **5-fold CV** — folds 0–4, focused 6-model set × 3 discriminative crops
  (okra excluded: ceiling effect, see below). 90 runs total (18 from pilot + 72 new), 0 failures.

Metric of record is **balanced accuracy** (robust to class imbalance, e.g. Okra
powdery 1184 vs healthy 255). Selection/checkpointing also used balanced accuracy.

---

## 1. Training configuration (reviewer #2)

| Item | Value |
|---|---|
| Optimizer | AdamW (CNN lr 3e-4 / transformer lr 5e-5), weight decay 1e-4 (CNN) / 0.05 (TFM) |
| Schedule | Cosine with 3–5 epoch warmup |
| Epochs | 40, early stopping (patience 10) on val balanced-acc |
| Batch size | 32 | Input | 224² (299² for inception/xception) |
| Loss | Class-weighted cross-entropy, label smoothing 0.05–0.1 |
| Precision | Mixed (AMP), grad-clip 1.0 |
| Pretraining | ImageNet (timm) |
| Split | Stratified 80/20 held-out test; stratified 5-fold CV on the 80% |

## 2. Dataset (reviewer #7)

| Crop | Classes | trainval / test | Notes |
|---|---|---|---|
| potato | early_blight, healthy, late_blight | 2634 / 659 | PlantVillage lab, 256² |
| okra | healthy, jasid, powdery | 1421 / 356 | field, imbalanced |
| cowpea | healthy, leaf_minor | 1104 / 276 | field |
| cluster_bean | healthy, mosaic | 1236 / 310 | field |

Cleaning: 227 zero-byte files removed; 324 HEIC (incl. 5 HEIC-as-`.jpg` caught by
magic bytes) converted. Augmentation: RandomResizedCrop, ±30° rotation, H/V flip,
color jitter 0.2.

## 3. Headline result — 5-fold balanced accuracy (mean ± std)

**Potato**
| Model | Bal-acc | Acc | F1-w |
|---|---|---|---|
| convnext_tiny | 0.9873 ± 0.0050 | 0.9833 | 0.9834 |
| vit_small | 0.9855 ± 0.0042 | 0.9818 | 0.9819 |
| swin_tiny | 0.9850 ± 0.0024 | 0.9824 | 0.9825 |
| **densexnet** | 0.9840 ± 0.0042 | 0.9818 | 0.9819 |
| densenet201 | 0.9822 ± 0.0044 | 0.9788 | 0.9788 |
| densenet121 | 0.9817 ± 0.0028 | 0.9800 | 0.9801 |

**Cowpea**
| Model | Bal-acc |
|---|---|
| vit_small | 0.9933 ± 0.0000 |
| densenet201 | 0.9924 ± 0.0017 |
| densenet121 | 0.9912 ± 0.0034 |
| **densexnet** | 0.9900 ± 0.0040 |
| swin_tiny | 0.9896 ± 0.0036 |
| convnext_tiny | 0.9875 ± 0.0042 |

**Cluster bean**
| Model | Bal-acc |
|---|---|
| swin_tiny | 0.9955 ± 0.0050 |
| convnext_tiny | 0.9945 ± 0.0038 |
| vit_small | 0.9914 ± 0.0059 |
| densenet201 | 0.9907 ± 0.0047 |
| densenet121 | 0.9889 ± 0.0037 |
| **densexnet** | 0.9878 ± 0.0052 |

Full numbers in `outputs/cv_5fold_summary.csv`; per-model fold-0 across all 4 crops
in `outputs/pilot_results.csv`.

## 4. Statistical validation (reviewer #3) — DenseXnet vs baselines

McNemar (pooled over 5 folds) + paired t-test (per-fold accuracy):

| Crop | Comparison | McNemar p | t-test p | Verdict |
|---|---|---|---|---|
| potato | densexnet vs densenet121 | 0.512 | 0.516 | n.s. |
| potato | densexnet vs densenet201 | 0.268 | 0.463 | n.s. |
| potato | densexnet vs convnext_tiny | 0.590 | 0.742 | n.s. |
| cowpea | densexnet vs densenet121 | 0.774 | 0.587 | n.s. |
| cowpea | densexnet vs densenet201 | 0.549 | 0.426 | n.s. |
| cowpea | densexnet vs convnext_tiny | 0.454 | 0.099 | n.s. |
| cluster_bean | densexnet vs densenet121 | 1.000 | 0.828 | n.s. |
| cluster_bean | densexnet vs densenet201 | 0.359 | 0.189 | n.s. |
| cluster_bean | densexnet vs convnext_tiny | 0.143 | 0.108 | n.s. |

**No pairwise difference is statistically significant (all p > 0.05).**

## 5. Computational complexity (reviewer #9)

`outputs/complexity.csv` (16 models). DenseXnet = 7.22M params / 2.87 GFLOPs /
554 img/s — only +0.26M params over DenseNet121 at identical FLOPs/latency.
EfficientNet-B0 is the efficiency outlier (4.0M / 0.40 GFLOPs / 1266 img/s);
VGG16 the heaviest (134M / 15.5 GFLOPs).

---

## Interpretation for the manuscript (honest framing)

1. **DenseXnet is competitive, not superior.** It lands mid-pack on every crop and
   **no comparison vs DenseNet121/201 or ConvNeXt is significant**. The defensible
   claim is *"DenseXnet matches deeper DenseNet variants and transformers at a
   fraction of the parameters (+0.26M over DenseNet121),"* not that it is the best.
   This directly answers reviewer #1 with evidence.
2. **Transformer/attention models (ConvNeXt, ViT, Swin) lead by small margins** on
   potato and cluster_bean — supports adding them per reviewer #6, while honestly
   noting the gaps are within noise.
3. **Single-fold rankings were misleading** (e.g. DenseXnet's apparent cowpea "win"
   vanished under CV). The 5-fold + significance protocol is the core fix for #3.
4. **Okra excluded from CV** — multiple models hit 100% on its test set (powdery
   mildew is visually dominant); a ceiling effect makes it non-discriminative.
   Report it as an easy case, not as architecture evidence.

## 6. Dehazing image quality (reviewer #4)

Synthetic haze via the atmospheric-scattering model (β=1.5, A=0.8), 150 test
images/crop. PSNR(dB) / SSIM vs the clean reference:

| Crop | Hazy (no dehaze) | DCP | AOD-Net |
|---|---|---|---|
| potato | 12.60 / 0.693 | **23.61 / 0.941** | 20.01 / 0.855 |
| okra | 12.59 / 0.797 | 20.02 / 0.926 | **21.45 / 0.890** |
| cowpea | 11.92 / 0.758 | 18.42 / 0.912 | **20.13 / 0.861** |
| cluster_bean | 10.59 / 0.648 | **21.08 / 0.904** | 20.21 / 0.833 |

DCP gives the best SSIM everywhere; AOD-Net is competitive on PSNR for the field
crops. Both substantially restore the degraded images.

## 7. Classification robustness under haze (reviewer #5)

DenseXnet (fold-0) balanced accuracy by input condition:

| Crop | Clean | Hazy | Hazy+DCP | Hazy+AOD-Net |
|---|---|---|---|---|
| potato | 0.9838 | 0.8919 | **0.9898** | 0.9810 |
| okra | 0.9850 | 0.9616 | 0.9773 | 0.9808 |
| cowpea | 0.9966 | 0.9732 | 0.9899 | 0.9787 |
| cluster_bean | 0.9844 | 0.8235 | 0.9688 | 0.9784 |

**Key finding:** haze degrades accuracy by 2–16 points (worst on cluster_bean,
−16; potato, −9), and **dehazing recovers nearly all of it** — DCP restores potato
to 0.990 and cluster_bean to 0.969; AOD-Net is the most consistent recoverer
across crops. This directly answers reviewer #5 (the old manuscript's "very low
hazy accuracy" is recovered by an explicit dehazing front-end).

## 8. Grad-CAM + confusion matrices (reviewer #8)

- Confusion matrices saved per run: `outputs/<run>/confusion_test.png`.
- Grad-CAM panels (4 imgs/class, correct + misclassified) for DenseXnet and the
  per-crop winner: `outputs/<run>/gradcam/`. CNN and transformer (ViT/Swin/ConvNeXt)
  attention both supported.

---

## Coverage of reviewer comments

| # | Comment | Status |
|---|---|---|
| 1 | DenseXnet vs DenseNet variants | ✅ 5-fold + significance (n.s. — competitive, not superior) |
| 2 | Training details | ✅ documented |
| 3 | Statistical validation | ✅ 5-fold mean±std, McNemar, paired-t |
| 4 | Dehazing comparison + image quality | ✅ DCP vs AOD-Net, PSNR/SSIM |
| 5 | Accuracy under haze | ✅ clean/hazy/dehazed table |
| 6 | Transformer/attention models | ✅ ViT/Swin/ConvNeXt/CBAM trained + compared |
| 7 | Dataset/augmentation/resolution/split | ✅ documented |
| 8 | Confusion matrices, Grad-CAM, error analysis | ✅ generated |
| 9 | FLOPs, params, inference time | ✅ complexity.csv |

All nine experimentally-addressable comments are now covered with results.
