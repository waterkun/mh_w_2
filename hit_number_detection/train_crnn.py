"""
CRNN 数字识别模型训练脚本

用法：
  python train_crnn.py
  python train_crnn.py --train crnn_data/train --val crnn_data/val --epochs 100
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from crnn_model import (
    CRNN, CHARS, BLANK_IDX, NUM_CLASSES,
    IMG_HEIGHT, IMG_WIDTH, decode_predictions,
)


class DamageNumberDataset(Dataset):
    """从 labels.txt 加载裁剪的数字图片和对应标签。"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.samples = []

        labels_path = os.path.join(data_dir, "labels.txt")
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                img_name, label = parts
                if label == "???":
                    continue  # 跳过未标注的
                # 验证标签只包含数字
                if not label.isdigit():
                    continue
                self.samples.append((img_name, label))

        print(f"加载数据集: {data_dir}, 共 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # 读取灰度图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # 返回空白图
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        # 缩放到固定尺寸，保持宽高比
        img = self._resize_pad(img)

        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # 转 tensor: (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # 标签编码: 字符 → 索引
        label_indices = [CHARS.index(c) for c in label]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return img_tensor, label_tensor, len(label_indices)

    def _resize_pad(self, img):
        """缩放图片到 IMG_HEIGHT 高度，宽度按比例缩放后 pad 到 IMG_WIDTH。"""
        h, w = img.shape
        ratio = IMG_HEIGHT / h
        new_w = min(int(w * ratio), IMG_WIDTH)
        img = cv2.resize(img, (new_w, IMG_HEIGHT))

        # 右侧 pad 到 IMG_WIDTH
        if new_w < IMG_WIDTH:
            pad = np.zeros((IMG_HEIGHT, IMG_WIDTH - new_w), dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)

        return img


def collate_fn(batch):
    """自定义 collate: 处理变长标签。"""
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 数据集
    train_dataset = DamageNumberDataset(args.train)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )

    val_loader = None
    if args.val and os.path.exists(args.val):
        val_dataset = DamageNumberDataset(args.val)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch, shuffle=False,
            num_workers=0, collate_fn=collate_fn,
        )

    # 模型
    model = CRNN(NUM_CLASSES).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数 + 优化器
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    best_val_acc = 0.0
    os.makedirs("runs/crnn", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0

        for images, labels, lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)  # (T, batch, num_classes)
            log_probs = output.log_softmax(2)

            T = output.size(0)
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), T, dtype=torch.long)

            loss = criterion(log_probs, labels, input_lengths, lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- 验证 ----
        val_acc = 0.0
        if val_loader:
            val_acc = evaluate(model, val_loader, device)

        scheduler.step(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.2%} | "
                  f"LR: {lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "runs/crnn/best.pt")

    # 保存最终模型
    torch.save(model.state_dict(), "runs/crnn/last.pt")
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2%}")
    print(f"模型保存位置: runs/crnn/")


def evaluate(model, dataloader, device):
    """评估模型：计算完整字符串准确率。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, lengths in dataloader:
            images = images.to(device)
            output = model(images)
            predictions = decode_predictions(output)

            # 还原标签字符串
            offset = 0
            for i, length in enumerate(lengths):
                label_indices = labels[offset:offset + length]
                label_str = "".join(CHARS[idx] for idx in label_indices)
                offset += length

                if predictions[i] == label_str:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="训练 CRNN 数字识别模型")
    parser.add_argument("--train", default="crnn_data/train", help="训练数据目录")
    parser.add_argument("--val", default="crnn_data/val", help="验证数据目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
