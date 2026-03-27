"""GRU 时序分类头 — 从帧特征序列预测攻击类型."""

import torch
import torch.nn as nn


class GRUTemporalHead(nn.Module):
    """单向 GRU 时序分类头.

    输入: (batch, seq_len, feat_dim) 帧特征序列
    输出: (batch, num_classes) 分类 logits
    """

    def __init__(self, feat_dim, num_classes, hidden_size=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, feat_dim)
        Returns:
            (batch, num_classes)
        """
        # GRU 输出: (batch, seq_len, hidden)
        output, _ = self.gru(x)
        # 取最后一个时间步
        last = output[:, -1, :]      # (batch, hidden)
        logits = self.classifier(last)  # (batch, num_classes)
        return logits
