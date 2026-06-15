# Paper Revision — Details Extractable from This Repository

Material below is pulled directly from the committed notebook source/outputs. It covers the
reviewer comments that can be answered **without new experiments**: #2 (training details),
#7 (dataset/augmentation), #8a (confusion matrices / per-class error analysis),
#9-partial (parameter counts). See the "Inconsistencies to resolve" section at the end before
copying any number into the manuscript.

---

## Reviewer #2 — Model training details (epochs, optimizer, batch size, train/test split)

### Common configuration (DenseNet121 / Xception crop notebooks)
| Setting | Value |
|---|---|
| Optimizer | Adam, lr = 0.002, β₁ = 0.9, β₂ = 0.999, ε = 0.1, decay = 0.0 |
| Loss | Categorical cross-entropy |
| Epochs | 100 |
| Batch size (fit) | 16 |
| LR schedule | ReduceLROnPlateau(monitor = val_accuracy, factor = 0.5, patience = 5, min_lr = 1e-3) |
| Checkpointing | ModelCheckpoint(save_best_only = True) |
| Class imbalance | `class_weight` passed to `fit` |
| Weights init | ImageNet (transfer learning) + scratch variant |
| Input adapter | `Conv2D(3, (3,3), padding='same')` before the backbone |
| Head | GAP → BN → Dropout(0.5) → Dense(256, relu) → BN → Dropout(0.5) → Dense(n, softmax) |

### Train / validation / test split
Two-stage split with a fixed `random_state` (SEED):
1. `train_test_split(test_size=0.2)` → **20% held-out test**
2. The remaining 80% is split again `test_size=0.2` → **64% train / 16% validation**

So effective partition ≈ **64% train / 16% val / 20% test** for the crop (DenseNet/Xception) notebooks.

### Per-architecture differences (Potato lab notebooks)
| Notebook | Epochs | Batch | Input size |
|---|---|---|---|
| Potato_Lab_DenseNet | 30 / 60 | 16 / 128 | 224 |
| Potato_Lab_VGG16 | 60 | 32 / 128 | 224 |
| Potato_Lab_InceptionV3 | 30 / 60 | 16 / 128 | 299 |
| Potato_Lab_Xception | 100 | 32 | 299 |
| Potato_Lab_AlexNet | 30 | 16 / 32 / 64 | 320 |
| Potato_Lab_Custom_CNN | 30 / 60 | 16 / 128 | 320 |
| Crop DenseNet121 (Okra/Cowpea/Cluster bean) | 100 | 16 | 64 |
| Crop Xception (Mixed) | 100 (50 noted) | 16 | 299 |

---

## Reviewer #7 — Dataset description, augmentation, resolution, split

### Dataset size (from README)
| Crop | Lab | Field | Total | Classes |
|---|---|---|---|---|
| Potato | 2152 | 1151 | 3303 | Healthy, Early_blight, Late_blight |
| Okra | 915 | 588 | 1503 | Healthy, Jasid, Powdery |
| Cowpea | 882 | 638 | 1520 | Healthy, Leaf_minor |
| Cluster bean | 758 | 695 | 1453 | Healthy, Mosaic |

Data organization in code: three condition splits — **Lab**, **Field**, **Mixed** (Lab+Field combined).
Classes are folder names; labels assigned by directory (`disease_type` list → integer id → one-hot).
**"Mixed" is the closest proxy in the repo to the "environmental/hazy conditions" claim** — there is no
synthetic-haze or dehazing code present.

### Exact test-set distribution (per-class support, Potato lab, 20% test = 659 images)
| Class | Test support | ≈ Full set (×5) |
|---|---|---|
| Healthy | 120 | ~600 |
| Early_blight | 331 | ~1655 |
| Late_blight | 208 | ~1040 |

(Per-class counts for the other crops are derivable by running the data-loader cells; the test split
is the same 20%.)

### Image resolution (resize target per model)
64×64 (crop DenseNet/ResNet) · 224×224 (Potato DenseNet, VGG16) · 299×299 (Xception, InceptionV3) · 320×320 (AlexNet, Custom CNN).

### Augmentation (Keras `ImageDataGenerator`) — two regimes
**Regime A (most notebooks):** rotation_range = 360, width_shift = 0.2, height_shift = 0.2,
zoom = 0.2, horizontal_flip = True, vertical_flip = True.

**Regime B (Potato AlexNet / Custom_CNN / cnn):** rotation_range = 20, width_shift = 0.2,
height_shift = 0.2, shear_range = 0.2, zoom = 0.2, horizontal_flip = True, fill_mode = "nearest".

### Annotation strategy
Not documented in code — labels are **folder/directory-level (image-level class labels)**, no bounding
boxes or masks. You must describe the labeling protocol from your own records.

---

## Reviewer #8a — Confusion matrices & per-class error analysis

Confusion matrices are computed (`sklearn.confusion_matrix`) and rendered as seaborn heatmaps (figures
present in every notebook). Full per-class classification reports are present in the **Potato lab** notebooks:

| Model (Potato Lab) | Acc | Class 0 (Healthy) P/R/F1 | Class 1 (Early) P/R/F1 | Class 2 (Late) P/R/F1 |
|---|---|---|---|---|
| Custom CNN | 0.99 | 0.98/1.00/0.99 | 0.99/0.98/0.99 | 0.98/0.99/0.98 |
| AlexNet | 0.95 | 0.89/0.99/0.94 | 0.99/0.92/0.95 | 0.93/0.98/0.96 |
| DenseNet121 | 0.86 | 0.73/0.96/0.83 | 0.86/0.90/0.88 | 0.97/0.72/0.83 |
| InceptionV3 | 0.77 | 0.52/0.72/0.60 | 0.92/0.86/0.89 | 0.76/0.66/0.71 |
| VGG16 | 0.50 | 0.50/1.00/0.67 | 0.00/0.00/0.00 | 0.00/0.00/0.00 |

**Error-analysis signal already in the data:** the dominant confusion is **Late_blight ↔ Early_blight**
(DenseNet recall on Late_blight drops to 0.72 while precision stays 0.97 → late-blight leaves misread as
early-blight). InceptionV3 most confuses **Healthy** (precision 0.52). VGG16 collapsed to a single class
(see inconsistencies). These narratives can anchor the error-analysis section.

---

## Reviewer #9 (partial) — Parameter counts

| Architecture | Total params | Trainable |
|---|---|---|
| Custom CNN | 148,867 | 148,867 |
| DenseNet121 ("DenseXnet" head) | 7,305,879 (3-cls) / 7,305,622 (2-cls) | ~7,219,671 |
| VGG16 | 14,849,943 | 14,848,407 |
| Xception | 21,396,095 | 21,336,959 |
| InceptionV3 | 22,337,399 | 22,298,359 |
| ResNet50 | 24,122,327 | 24,064,599 |
| AlexNet | 91,783,747 | 91,762,611 |

Still **missing for #9:** FLOPs and inference time (need new measurement — can be scripted on the saved models).

---

## Full accuracy compendium (final evaluate() per notebook)

| Crop | Condition | Model | Final accuracy |
|---|---|---|---|
| Potato | Lab | Custom CNN | 0.986 |
| Potato | Lab | Xception | 0.993 |
| Potato | Lab | AlexNet | 0.950 |
| Potato | Lab | DenseNet121 | 0.856 |
| Potato | Lab | InceptionV3 | 0.774 |
| Potato | Lab | VGG16 | 0.499 ⚠ |
| Okra | Lab | DenseNet121 | 0.995 |
| Okra | Field | DenseNet121 | 0.856 |
| Okra | Mixed | DenseNet121 | 0.867 |
| Okra | Lab | ResNet50 | 0.905 |
| Cowpea | Lab | DenseNet121 | 0.989 |
| Cowpea | Field | DenseNet121 | 0.953 |
| Cowpea | Mixed | DenseNet121 | 0.984 |
| Cowpea | Mixed | Xception | 1.000 ⚠ |
| Cluster bean | Field | DenseNet121 | 0.978 |

**Relevant to Reviewer #5 (low accuracy under hazy/mixed):** the drop only shows for **Okra**
(Lab 0.995 → Field 0.856 / Mixed 0.867). Cowpea Mixed (0.984) and Cluster bean Field (0.978) stay high,
so the "low accuracy under environmental conditions" claim is currently **crop-specific, not universal** —
state it carefully.

---

## ⚠ Inconsistencies to resolve before using these numbers

1. **README ≠ notebook outputs for Potato.** README table reports DenseNet 0.99, InceptionV3 0.98,
   VGG16 0.86; the committed notebook outputs show 0.856, 0.774, 0.499. The saved runs look
   non-final/buggy. Decide which numbers are authoritative and re-run for a clean set.
2. **VGG16 collapsed** — predicts one class (acc 0.499, two classes at 0.00 F1). Either a training bug
   (LR too high / no convergence) or a bad checkpoint. Must be re-run; do not report as-is.
3. **Cowpea Mixed Xception = 1.000** — perfect accuracy on a 2-class set is a red flag (possible
   leakage between the Lab+Field "Mixed" duplicate images). Verify before publishing.
4. **Batch-size ambiguity** — `BATCH_SIZE = 64` is defined but `model.fit` uses `batch_size = 16`.
   Report the value actually used (16 for crop notebooks).
5. **"DenseXnet" is undocumented as a distinct model** — in code it is DenseNet121 + a Conv2D(3,3)
   input adapter + custom GAP/BN/Dropout/Dense head. The manuscript must define it explicitly and (per
   Reviewer #1) compare against stock DenseNet121/169/201 — those runs do not exist yet.
