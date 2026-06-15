"""Model factory: one entry point for every architecture in the study.

Registry keys are the `model.name` values used in configs. Backbones come from
timm so DenseNet variants, ViT, Swin and ConvNeXt share an identical training and
evaluation path (fair comparison for reviewers #1 and #6).
"""
from __future__ import annotations

import timm
import torch.nn as nn

from .densexnet import DenseXNet

# name -> (kind, timm_id). kind drives how it's built.
REGISTRY = {
    # --- DenseNet variants (reviewer #1) ---
    "densenet121": ("timm", "densenet121"),
    "densenet161": ("timm", "densenet161"),
    "densenet169": ("timm", "densenet169"),
    "densenet201": ("timm", "densenet201"),
    # paper's custom architecture + an attention-augmented version
    "densexnet": ("densexnet", "densenet121"),
    "densexnet_cbam": ("densexnet_cbam", "densenet121"),
    # --- transformer / attention models (reviewer #6) ---
    "vit_small": ("timm", "vit_small_patch16_224"),
    "vit_base": ("timm", "vit_base_patch16_224"),
    "swin_tiny": ("timm", "swin_tiny_patch4_window7_224"),
    "swin_small": ("timm", "swin_small_patch4_window7_224"),
    "convnext_tiny": ("timm", "convnext_tiny"),
    "deit_small": ("timm", "deit_small_patch16_224"),
    # --- classic CNN baselines (paper comparison) ---
    "resnet50": ("timm", "resnet50"),
    "vgg16": ("timm", "vgg16"),
    "inception_v3": ("timm", "inception_v3"),
    "xception": ("timm", "xception"),
    "efficientnet_b0": ("timm", "efficientnet_b0"),
}


def build_model(cfg_model, num_classes: int) -> nn.Module:
    name = cfg_model.name
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(REGISTRY)}")
    kind, timm_id = REGISTRY[name]

    if kind == "timm":
        return timm.create_model(
            timm_id, pretrained=cfg_model.pretrained, num_classes=num_classes,
            drop_rate=cfg_model.drop_rate)
    if kind == "densexnet":
        return DenseXNet(cfg_model.densexnet_backbone, num_classes,
                         cfg_model.pretrained, cfg_model.densexnet_hidden,
                         use_cbam=False)
    if kind == "densexnet_cbam":
        return DenseXNet(cfg_model.densexnet_backbone, num_classes,
                         cfg_model.pretrained, cfg_model.densexnet_hidden,
                         use_cbam=True)
    raise RuntimeError(kind)


def input_size_for(name: str) -> int:
    """Native input size; ViT/Swin/DeiT need 224, CNNs are flexible."""
    if name in {"inception_v3", "xception"}:
        return 299
    return 224


def gradcam_target_layer(model: nn.Module, name: str):
    """Return the conv/norm layer Grad-CAM should hook for each architecture."""
    if name.startswith("densexnet"):
        # last dense block norm of the timm densenet backbone
        return [model.backbone.features[-1]]
    if name.startswith("densenet"):
        return [model.features[-1]]
    if name.startswith("resnet"):
        return [model.layer4[-1]]
    if name.startswith("convnext"):
        return [model.stages[-1]]
    if name.startswith(("vit", "deit")):
        return [model.blocks[-1].norm1]
    if name.startswith("swin"):
        return [model.layers[-1].blocks[-1].norm1]
    if name == "vgg16":
        return [model.features[-1]]
    if name in {"inception_v3", "xception", "efficientnet_b0"}:
        return [list(model.children())[-3]]
    raise ValueError(f"No Grad-CAM layer mapping for {name}")
