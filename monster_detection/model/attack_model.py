"""攻击预测模型 — CNN (MobileNetV3) + GRU 组合."""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (NUM_CLASSES, BACKBONE_FEAT_DIM, GRU_HIDDEN,
                    GRU_LAYERS, DROPOUT, SEQ_LENGTH)
from model.backbone import MobileNetBackbone
from model.temporal_head import GRUTemporalHead


class AttackModel(nn.Module):
    """CNN + GRU 攻击预测模型.

    对序列中每帧提取 CNN 特征，然后用 GRU 做时序分类.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained_cnn=True):
        super().__init__()
        self.backbone = MobileNetBackbone(pretrained=pretrained_cnn)
        self.temporal = GRUTemporalHead(
            feat_dim=BACKBONE_FEAT_DIM,
            num_classes=num_classes,
            hidden_size=GRU_HIDDEN,
            num_layers=GRU_LAYERS,
            dropout=DROPOUT,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 3, 224, 224)
        Returns:
            (batch, num_classes) logits
        """
        batch, seq_len, c, h, w = x.shape

        # 合并 batch 和 seq 维度给 CNN
        x_flat = x.view(batch * seq_len, c, h, w)
        feats = self.backbone(x_flat)          # (batch*seq, 576)

        # 还原序列维度
        feats = feats.view(batch, seq_len, -1)  # (batch, seq, 576)

        # 时序分类
        logits = self.temporal(feats)           # (batch, num_classes)
        return logits

    def freeze_backbone(self):
        """冻结 CNN backbone."""
        self.backbone.freeze()

    def unfreeze_backbone(self):
        """解冻 CNN backbone."""
        self.backbone.unfreeze()
