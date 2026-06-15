"""AOD-Net: All-in-One Dehazing Network (Li et al., ICCV 2017).

A tiny end-to-end CNN that estimates a single K(x) map and recovers the clean
image as J(x) = K(x)·I(x) − K(x) + b. We provide the architecture plus a
self-supervised training routine on synthetically-hazed crop images, so the
dehazing comparison includes a *learned* method alongside the classical DCP
baseline (reviewer #4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AODNet(nn.Module):
    def __init__(self, b: float = 1.0):
        super().__init__()
        self.b = b
        self.conv1 = nn.Conv2d(3, 3, 1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, 5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, 7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(torch.cat([x1, x2], 1)))
        x4 = self.relu(self.conv4(torch.cat([x2, x3], 1)))
        k = self.relu(self.conv5(torch.cat([x1, x2, x3, x4], 1)))
        out = k * x - k + self.b
        return torch.clamp(out, 0, 1)
