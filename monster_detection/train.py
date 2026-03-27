"""训练脚本 — WeightedRandomSampler + 分阶段训练."""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import cv2
import numpy as np

from config import (NUM_CLASSES, ATTACK_CLASSES, CLASS_TO_IDX, SEQ_LENGTH,
                    INPUT_SIZE, BATCH_SIZE, NUM_EPOCHS, FREEZE_EPOCHS,
                    LR, LR_FINETUNE, WEIGHT_DECAY, RUNS_DIR, DATA_DIR)
from model.attack_model import AttackModel


class AttackSequenceDataset(Dataset):
    """帧序列数据集 — 每个样本是 SEQ_LENGTH 帧 + 1 个标签."""

    def __init__(self, seq_dir, labels_dict, transform=None):
        self.seq_dir = seq_dir
        self.transform = transform
        self.samples = []  # [(seq_path, class_idx), ...]

        for seq_name, class_name in labels_dict.items():
            seq_path = os.path.join(seq_dir, seq_name)
            if os.path.isdir(seq_path):
                cidx = CLASS_TO_IDX.get(class_name, 0)
                self.samples.append((seq_path, cidx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]

        frames = []
        for i in range(SEQ_LENGTH):
            img_path = os.path.join(seq_path, f"{i:03d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # (seq_len, 3, H, W)
        frames_tensor = torch.stack(frames)
        return frames_tensor, label


def get_transforms(train=True):
    """数据增强 / 标准化."""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def make_weighted_sampler(dataset):
    """根据类别频率创建 WeightedRandomSampler."""
    class_counts = [0] * NUM_CLASSES
    for _, label in dataset.samples:
        class_counts[label] += 1

    # 每个类别的权重 = 1 / count
    weights_per_class = [0.0] * NUM_CLASSES
    for i, cnt in enumerate(class_counts):
        if cnt > 0:
            weights_per_class[i] = 1.0 / cnt

    sample_weights = [weights_per_class[label] for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train(seq_dir, epochs=NUM_EPOCHS, resume_path=None):
    """训练攻击预测模型.

    Args:
        seq_dir: 序列数据目录 (包含 labels.json)
        epochs: 训练 epoch 数
        resume_path: 恢复训练的 checkpoint 路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载标签
    labels_path = os.path.join(seq_dir, "labels.json")
    with open(labels_path) as f:
        label_data = json.load(f)

    train_labels = label_data["train"]
    val_labels = label_data["val"]

    # 数据集
    train_ds = AttackSequenceDataset(seq_dir, train_labels,
                                     transform=get_transforms(train=True))
    val_ds = AttackSequenceDataset(seq_dir, val_labels,
                                   transform=get_transforms(train=False))

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2,
                            pin_memory=True)

    # 模型
    model = AttackModel(num_classes=NUM_CLASSES, pretrained_cnn=True)
    model.to(device)

    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"从 epoch {start_epoch} 恢复训练")

    criterion = nn.CrossEntropyLoss()

    # 输出目录
    run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(start_epoch, epochs):
        # 冻结策略
        if epoch < FREEZE_EPOCHS:
            model.freeze_backbone()
            lr = LR
        else:
            if epoch == FREEZE_EPOCHS:
                model.unfreeze_backbone()
                print(f"[Epoch {epoch}] 解冻 CNN backbone, lr → {LR_FINETUNE}")
            lr = LR_FINETUNE

        optimizer = AdamW(filter(lambda p: p.requires_grad,
                                 model.parameters()),
                          lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=epochs - epoch, eta_min=1e-6)

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device)  # (B, seq, 3, H, W)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                logits = model(frames)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        frozen = "frozen" if epoch < FREEZE_EPOCHS else "finetune"
        print(f"[Epoch {epoch+1}/{epochs}] ({frozen}) "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(run_dir, "best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
            }, best_path)
            print(f"  → 最佳模型已保存 (val_acc={val_acc:.3f})")

    # 保存最终模型 + 训练历史
    final_path = os.path.join(run_dir, "final.pt")
    torch.save({
        "epoch": epochs,
        "model_state": model.state_dict(),
        "val_acc": val_acc,
    }, final_path)

    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n训练完成。最佳 val_acc: {best_val_acc:.3f}")
    print(f"模型保存在: {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练攻击预测模型")
    parser.add_argument("seq_dir", type=str, help="序列数据目录")
    parser.add_argument("-e", "--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径")
    args = parser.parse_args()
    train(args.seq_dir, epochs=args.epochs, resume_path=args.resume)
