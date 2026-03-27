"""MobileNetV3-Small 特征提取器 — 576 维特征向量."""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetBackbone(nn.Module):
    """MobileNetV3-Small 去掉分类头，输出 576-d 特征."""

    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None
        base = mobilenet_v3_small(weights=weights)

        # features: 卷积层 + avgpool
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 输出维度
        self.feat_dim = 576

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            (batch, 576)
        """
        x = self.features(x)        # (batch, 576, 7, 7)
        x = self.pool(x)            # (batch, 576, 1, 1)
        x = x.flatten(1)            # (batch, 576)
        return x

    def freeze(self):
        """冻结所有参数."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """解冻所有参数."""
        for param in self.parameters():
            param.requires_grad = True
