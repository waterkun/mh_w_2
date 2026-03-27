"""CNN regression model for health bar percentage prediction.

Input:  (B, 3, 32, 256) — ROI resized to 256x32
Output: (B, 2)           — [health_pct, damage_pct] in [0, 1]
"""

import torch
import torch.nn as nn


class HealthBarNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 4-layer conv: 3 -> 16 -> 32 -> 64 -> 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # -> (16, 16, 128)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> (32, 8, 64)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> (64, 4, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> (128, 2, 16)
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (128, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
