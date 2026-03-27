"""
CRNN 数字识别模型

架构: CNN 特征提取 → BiLSTM 序列建模 → CTC 解码
专门针对伤害数字（0-9 序列）识别设计。

输入: 灰度图片，统一缩放到 (32, 128)
输出: 数字字符串（如 "127", "3456"）
"""

import torch
import torch.nn as nn


# 字符集: 0-9 + blank (CTC 需要)
CHARS = "0123456789"
BLANK_IDX = len(CHARS)  # 10
NUM_CLASSES = len(CHARS) + 1  # 11 (含 blank)

# 输入图片尺寸
IMG_HEIGHT = 32
IMG_WIDTH = 128


class CRNN(nn.Module):
    """
    轻量 CRNN 模型:
    - CNN: 4 层卷积提取视觉特征
    - RNN: 2 层 BiLSTM 建模序列依赖
    - FC:  映射到字符类别
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # CNN 特征提取
        # 输入: (batch, 1, 32, 128)
        self.cnn = nn.Sequential(
            # Conv1: 1 -> 32, 32x128 -> 16x64
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv2: 32 -> 64, 16x64 -> 8x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv3: 64 -> 128, 8x32 -> 4x32 (只在高度方向池化)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Conv4: 128 -> 128, 4x32 -> 2x32
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        # CNN 输出: (batch, 128, 2, 32) → 沿宽度方向形成 32 个时间步

        # 将 CNN 特征的高度维度合并到通道维度
        # (batch, 128, 2, 32) → (batch, 256, 32) → (batch, 32, 256)
        self.rnn_input_size = 128 * 2  # 256

        # BiLSTM
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        # 全连接层: BiLSTM 输出 256 → 字符类别数
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        # x: (batch, 1, 32, 128)
        conv = self.cnn(x)  # (batch, 128, 2, 32)

        # 重排: (batch, 128, 2, 32) → (batch, 32, 256)
        batch, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # (batch, 32, 128, 2)
        conv = conv.reshape(batch, w, c * h)  # (batch, 32, 256)

        # BiLSTM
        rnn_out, _ = self.rnn(conv)  # (batch, 32, 256)

        # 分类
        output = self.fc(rnn_out)  # (batch, 32, num_classes)

        # CTC 需要 (T, batch, num_classes) 格式
        output = output.permute(1, 0, 2)  # (32, batch, num_classes)
        return output


def decode_predictions(output, blank_idx=BLANK_IDX):
    """
    CTC 贪心解码: 取每个时间步最大概率的字符，去重 + 去 blank。

    Args:
        output: 模型输出 (T, batch, num_classes)

    Returns:
        list[str]: 解码后的字符串列表
    """
    # (T, batch, num_classes) → (batch, T)
    _, preds = output.max(2)
    preds = preds.permute(1, 0)  # (batch, T)

    results = []
    for pred in preds:
        chars = []
        prev = -1
        for idx in pred:
            idx = idx.item()
            if idx != blank_idx and idx != prev:
                chars.append(CHARS[idx])
            prev = idx
        results.append("".join(chars))
    return results
