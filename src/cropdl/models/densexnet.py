"""DenseXnet: the paper's custom head on a DenseNet backbone, reimplemented in
PyTorch/timm, plus an optional CBAM attention block.

The original Keras model was: DenseNet121(features) -> GAP -> BN -> Dropout ->
Dense(256, relu) -> BN -> Dropout -> Dense(softmax). We keep that head verbatim
and expose the backbone choice so it can be compared against stock DenseNet
variants (reviewer #1) on equal footing.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .cbam import CBAM


class DenseXNet(nn.Module):
    def __init__(self, backbone: str = "densenet121", num_classes: int = 3,
                 pretrained: bool = True, hidden: int = 256, drop_rate: float = 0.5,
                 use_cbam: bool = False):
        super().__init__()
        # features_only-style: get pooled feature vector from timm
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="")
        feat_dim = self.backbone.num_features
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(drop_rate),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        feat = self.backbone.forward_features(x)        # B,C,H,W
        if self.use_cbam:
            feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)
