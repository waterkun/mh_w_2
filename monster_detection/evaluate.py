"""评估脚本 — 混淆矩阵 + per-class precision/recall."""

import json
import os
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (NUM_CLASSES, ATTACK_CLASSES, INPUT_SIZE, BATCH_SIZE)
from model.attack_model import AttackModel
from train import AttackSequenceDataset, get_transforms


def evaluate(seq_dir, model_path):
    """评估模型在验证集上的表现."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载标签
    labels_path = os.path.join(seq_dir, "labels.json")
    with open(labels_path) as f:
        label_data = json.load(f)

    val_labels = label_data["val"]
    val_ds = AttackSequenceDataset(seq_dir, val_labels,
                                   transform=get_transforms(train=False))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    print(f"验证集: {len(val_ds)} 样本")

    # 加载模型
    model = AttackModel(num_classes=NUM_CLASSES, pretrained_cnn=False)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # 推理
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            logits = model(frames)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 混淆矩阵
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    # 打印混淆矩阵
    print("\n=== 混淆矩阵 (行=真实, 列=预测) ===")
    header = "".join(f"{name[:6]:>8}" for name in ATTACK_CLASSES)
    print(f"{'':>12}{header}")
    for i, row in enumerate(confusion):
        row_str = "".join(f"{v:>8}" for v in row)
        print(f"{ATTACK_CLASSES[i]:>12}{row_str}")

    # Per-class 指标
    print("\n=== Per-class 指标 ===")
    print(f"{'类别':>15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")

    total_correct = 0
    total_samples = 0

    for i in range(NUM_CLASSES):
        tp = confusion[i][i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        support = confusion[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        total_correct += tp
        total_samples += support

        print(f"{ATTACK_CLASSES[i]:>15} {precision:>10.3f} {recall:>10.3f} "
              f"{f1:>10.3f} {support:>10}")

    overall_acc = total_correct / max(total_samples, 1)
    print(f"\n总体准确率: {overall_acc:.3f} ({total_correct}/{total_samples})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="评估攻击预测模型")
    parser.add_argument("seq_dir", type=str, help="序列数据目录")
    parser.add_argument("model", type=str, help="模型 checkpoint 路径")
    args = parser.parse_args()
    evaluate(args.seq_dir, args.model)
