"""整理 labelImg 标注数据 — 80/20 分割 train/val + 生成 dataset.yaml.

用法:
  python -m monster_detect.split_dataset
  python -m monster_detect.split_dataset --ratio 0.8 --neg-ratio 0.3
"""

import argparse
import os
import random
import shutil

import yaml


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_MODULE_DIR, "monster_yolo_data")
_UNLABELED_DIR = os.path.join(_DATA_DIR, "images", "unlabeled")


def split(train_ratio: float = 0.8, neg_ratio: float = 0.3, seed: int = 42):
    random.seed(seed)

    # 收集已标注 (有 .txt) 和未标注的图片
    labeled = []
    unlabeled = []

    for fname in sorted(os.listdir(_UNLABELED_DIR)):
        if not fname.endswith(".jpg"):
            continue
        txt_name = fname.replace(".jpg", ".txt")
        txt_path = os.path.join(_UNLABELED_DIR, txt_name)
        if os.path.exists(txt_path):
            labeled.append(fname)
        else:
            unlabeled.append(fname)

    print(f"已标注图片: {labeled.__len__()} 张")
    print(f"未标注图片 (无怪负样本): {unlabeled.__len__()} 张")

    if not labeled:
        print("错误: 没有找到已标注的图片 (需要 .txt 标注文件)")
        return

    # 从未标注图片中随机选取一部分作为负样本
    n_neg = int(len(labeled) * neg_ratio)
    n_neg = min(n_neg, len(unlabeled))
    neg_samples = random.sample(unlabeled, n_neg) if n_neg > 0 else []
    print(f"选取负样本: {len(neg_samples)} 张 (比例 {neg_ratio})")

    # 合并所有样本
    all_samples = labeled + neg_samples
    random.shuffle(all_samples)

    # 分割 train / val
    split_idx = int(len(all_samples) * train_ratio)
    train_set = all_samples[:split_idx]
    val_set = all_samples[split_idx:]

    print(f"\n分割结果: train={len(train_set)}, val={len(val_set)}")

    # 复制文件
    for subset_name, subset in [("train", train_set), ("val", val_set)]:
        img_dir = os.path.join(_DATA_DIR, "images", subset_name)
        lbl_dir = os.path.join(_DATA_DIR, "labels", subset_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # 清空旧文件
        for old in os.listdir(img_dir):
            os.remove(os.path.join(img_dir, old))
        for old in os.listdir(lbl_dir):
            os.remove(os.path.join(lbl_dir, old))

        for fname in subset:
            # 复制图片
            src_img = os.path.join(_UNLABELED_DIR, fname)
            shutil.copy2(src_img, os.path.join(img_dir, fname))

            # 复制标注 (如果有)
            txt_name = fname.replace(".jpg", ".txt")
            src_txt = os.path.join(_UNLABELED_DIR, txt_name)
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, os.path.join(lbl_dir, txt_name))
            # 负样本无 .txt → YOLO 自动视为无目标

        print(f"  {subset_name}: {len(subset)} 张已复制")

    # 生成 dataset.yaml
    yaml_path = os.path.join(_DATA_DIR, "dataset.yaml")
    dataset_cfg = {
        "path": _DATA_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["body", "head"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    print(f"\ndataset.yaml 已生成: {yaml_path}")
    print("完成!")


def main():
    parser = argparse.ArgumentParser(description="分割标注数据为 train/val")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="训练集比例 (默认 0.8)")
    parser.add_argument("--neg-ratio", type=float, default=0.3,
                        help="负样本比例 (相对于已标注数, 默认 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()
    split(args.ratio, args.neg_ratio, args.seed)


if __name__ == "__main__":
    main()
