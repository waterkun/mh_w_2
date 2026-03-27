"""AttackDetector — 帧缓冲 + 模型推理，单帧接口."""

import json
import os
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import (SEQ_LENGTH, INPUT_SIZE, NUM_CLASSES, ATTACK_CLASSES,
                    IDX_TO_CLASS, ROI_CONFIG_PATH, FRAME_INTERVAL_MS)
from model.attack_model import AttackModel


class AttackDetector:
    """从单帧输入检测攻击类型.

    内部维护长度为 SEQ_LENGTH 的帧缓冲 (deque).
    每次 detect() 将新帧压入缓冲，缓冲满后执行模型推理.
    """

    def __init__(self, model_path, roi=None, device=None):
        """
        Args:
            model_path: 训练好的模型 checkpoint 路径
            roi: (x, y, w, h) ROI 区域，None 则从 roi_config.json 加载
            device: torch device, None 则自动选择
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 加载 ROI
        if roi is not None:
            self.roi = roi
        else:
            self.roi = self._load_roi()

        # 加载模型
        self.model = AttackModel(num_classes=NUM_CLASSES, pretrained_cnn=False)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # 帧缓冲
        self.buffer = deque(maxlen=SEQ_LENGTH)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_roi(self):
        """从 roi_config.json 加载 ROI."""
        if os.path.exists(ROI_CONFIG_PATH):
            with open(ROI_CONFIG_PATH) as f:
                cfg = json.load(f)
            return (cfg["x"], cfg["y"], cfg["w"], cfg["h"])
        return None

    def _crop_roi(self, frame):
        """裁剪 ROI 区域."""
        if self.roi is None:
            return frame
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def _preprocess(self, frame):
        """BGR 帧 → 归一化 tensor."""
        roi = self._crop_roi(frame)
        resized = cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)

    def detect(self, frame) -> dict:
        """处理一帧，返回攻击预测结果.

        Args:
            frame: BGR 全屏图像

        Returns:
            dict:
                ready    — 缓冲是否已满
                attack   — 预测攻击类别名 (str)
                class_id — 类别索引 (int)
                confidence — 最大 softmax 概率
                probs    — 所有类别概率 dict
        """
        tensor = self._preprocess(frame)
        self.buffer.append(tensor)

        if len(self.buffer) < SEQ_LENGTH:
            return {
                "ready": False,
                "attack": "idle",
                "class_id": 0,
                "confidence": 0.0,
                "probs": {},
            }

        # 构建序列 tensor: (1, seq_len, 3, H, W)
        seq = torch.stack(list(self.buffer)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(seq)                     # (1, num_classes)
            probs = torch.softmax(logits, dim=1)[0]      # (num_classes,)

        class_id = probs.argmax().item()
        confidence = probs[class_id].item()

        probs_dict = {ATTACK_CLASSES[i]: round(probs[i].item(), 4)
                      for i in range(NUM_CLASSES)}

        return {
            "ready": True,
            "attack": IDX_TO_CLASS[class_id],
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "probs": probs_dict,
        }

    def reset(self):
        """清空帧缓冲."""
        self.buffer.clear()
