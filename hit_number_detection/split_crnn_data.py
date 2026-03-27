"""
将 OCR 预标注审查后的数据分割成 CRNN 训练集和验证集。

从 crnn_data/ocr_prelabel/ 读取审查后的 labels.txt，
按 80/20 分割到 crnn_data/train/ 和 crnn_data/val/。

用法：
  python split_crnn_data.py
"""

import os
import random
import shutil
from pathlib import Path


def main():
    base = Path(__file__).parent
    src_dir = base / "crnn_data" / "ocr_prelabel"
    train_dir = base / "crnn_data" / "train"
    val_dir = base / "crnn_data" / "val"

    labels_path = src_dir / "labels.txt"
    if not labels_path.exists():
        print("错误: labels.txt 不存在")
        return

    # 读取标注
    entries = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].isdigit():
                entries.append((parts[0], parts[1]))

    print(f"有效标注: {len(entries)} 条")

    # 随机打乱并分割
    random.seed(42)
    random.shuffle(entries)

    split_idx = int(len(entries) * 0.8)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    print(f"训练集: {len(train_entries)}")
    print(f"验证集: {len(val_entries)}")

    # 清理旧数据
    for d in [train_dir, val_dir]:
        if d.exists():
            shutil.rmtree(d)

    # 复制文件并写 labels.txt
    for split_name, split_entries, out_dir in [
        ("train", train_entries, train_dir),
        ("val", val_entries, val_dir),
    ]:
        img_out = out_dir / "images"
        img_out.mkdir(parents=True, exist_ok=True)

        label_lines = []
        for fname, label in split_entries:
            src_img = src_dir / "images" / fname
            if src_img.exists():
                shutil.copy2(str(src_img), str(img_out / fname))
                label_lines.append(f"{fname}\t{label}")

        with open(out_dir / "labels.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        print(f"  {split_name}: {len(label_lines)} 张已复制")

    print(f"\n完成！")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"\n下一步: 训练 CRNN")
    print(f"  E:/anaconda3/envs/mh_ai/python.exe train_crnn.py")


if __name__ == "__main__":
    main()
