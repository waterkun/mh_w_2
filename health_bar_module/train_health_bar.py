"""Train HealthBarNet on labeled ROI images.

Usage:
    python train_health_bar.py [--data health_bar_data] [--epochs 100] [--lr 1e-3]
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add parent path so we can import the model
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mh_w_2_health_bar"))
from health_bar_model import HealthBarNet


INPUT_H, INPUT_W = 32, 256


class HealthBarDataset(Dataset):
    def __init__(self, samples, rois_dir, augment=False):
        """
        Args:
            samples: list of (filename, health_pct, damage_pct)
            rois_dir: path to ROI images
            augment: apply data augmentation
        """
        self.samples = samples
        self.rois_dir = rois_dir
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, health_pct, damage_pct = self.samples[idx]
        img = cv2.imread(os.path.join(self.rois_dir, fname))
        img = cv2.resize(img, (INPUT_W, INPUT_H))

        if self.augment:
            img = self._augment(img)

        # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        label = np.array([health_pct, damage_pct], dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(label)

    def _augment(self, img):
        """Random brightness, contrast, and hue shifts."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Brightness: +/- 30
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.uniform(-30, 30), 0, 255)
        # Saturation: +/- 20
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.uniform(-20, 20), 0, 255)
        # Hue: +/- 10
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180

        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Random contrast
        alpha = random.uniform(0.8, 1.2)
        img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        return img


def load_labels(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    rois_dir = os.path.join(data_dir, "rois")

    samples = []
    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            if os.path.exists(os.path.join(rois_dir, fname)):
                samples.append((
                    fname,
                    float(row["health_pct"]),
                    float(row["damage_pct"]),
                ))

    print(f"Loaded {len(samples)} labeled samples")
    return samples


def train(args):
    samples = load_labels(args.data)
    if len(samples) < 10:
        print("Error: need at least 10 labeled samples to train")
        return

    # 80/20 split
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    rois_dir = os.path.join(args.data, "rois")
    train_ds = HealthBarDataset(train_samples, rois_dir, augment=True)
    val_ds = HealthBarDataset(val_samples, rois_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HealthBarNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                loss = criterion(preds, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}"
              + ("  *best*" if val_loss < best_val_loss else ""))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output, "best.pt"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output, "last.pt"))
    print(f"\nTraining done. Best val loss: {best_val_loss:.6f}")
    print(f"Models saved to {args.output}/")


def main():
    parser = argparse.ArgumentParser(description="Train health bar CNN model")
    parser.add_argument("--data", default="health_bar_training",
                        help="Data directory with labels.csv and rois/")
    parser.add_argument("--output", default="runs",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
