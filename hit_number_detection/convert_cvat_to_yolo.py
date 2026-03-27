"""
CVAT XML → YOLO 格式转换脚本

将 yolo-data/ 下的 CVAT XML 标注转换成 YOLOv8/v10 训练格式，
自动分割 train/val/test 并输出到 datasets/damage_numbers/。

用法：
  conda run -n mh_ai python convert_cvat_to_yolo.py
  conda run -n mh_ai python convert_cvat_to_yolo.py --val-ratio 0.15 --test-ratio 0.1
"""

import xml.etree.ElementTree as ET
import os
import shutil
import random
import argparse
from pathlib import Path


def parse_cvat_xml(xml_path: str) -> list:
    """
    解析 CVAT XML，返回 [(image_name, width, height, [boxes])] 列表。
    每个 box 为 (xtl, ytl, xbr, ybr)。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []
    for image in root.findall("image"):
        name = image.get("name")
        w = int(image.get("width"))
        h = int(image.get("height"))

        boxes = []
        for box in image.findall("box"):
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            boxes.append((xtl, ytl, xbr, ybr))

        if boxes:  # 只保留有标注的图片
            data.append((name, w, h, boxes))

    return data


def convert_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h) -> str:
    """将 (xtl, ytl, xbr, ybr) 转成 YOLO 归一化格式。"""
    cx = (xtl + xbr) / 2.0 / img_w
    cy = (ytl + ybr) / 2.0 / img_h
    bw = (xbr - xtl) / img_w
    bh = (ybr - ytl) / img_h
    # class_id = 0 (damage_number)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    parser = argparse.ArgumentParser(description="CVAT XML → YOLO 格式转换")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="验证集比例 (默认 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.10,
                        help="测试集比例 (默认 0.10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    yolo_data_dir = base_dir / "yolo-data"
    output_dir = base_dir / "datasets" / "damage_numbers"

    # 收集所有 CVAT XML 数据
    all_data = []
    for folder in sorted(yolo_data_dir.iterdir()):
        if not folder.is_dir():
            continue
        xml_path = folder / "annotations.xml"
        if not xml_path.exists():
            continue

        print(f"解析: {xml_path}")
        entries = parse_cvat_xml(str(xml_path))
        for name, w, h, boxes in entries:
            img_path = folder / name
            if img_path.exists():
                all_data.append((str(img_path), name, w, h, boxes))
            else:
                print(f"  警告: 图片不存在 {img_path}")

    print(f"\n共找到 {len(all_data)} 张有效标注图片")

    # 随机分割 train/val/test
    random.seed(args.seed)
    random.shuffle(all_data)

    n = len(all_data)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": all_data[:n_train],
        "valid": all_data[n_train:n_train + n_val],
        "test": all_data[n_train + n_val:],
    }

    print(f"分割: train={n_train}, valid={n_val}, test={n_test}")

    # 创建输出目录并写入文件
    total_boxes = 0
    for split_name, entries in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, name, w, h, boxes in entries:
            # 复制图片
            dst_img = img_dir / name
            shutil.copy2(img_path, dst_img)

            # 写标签文件
            stem = Path(name).stem
            dst_lbl = lbl_dir / f"{stem}.txt"
            lines = [convert_box_to_yolo(*box, w, h) for box in boxes]
            dst_lbl.write_text("\n".join(lines) + "\n")
            total_boxes += len(boxes)

    print(f"\n转换完成！")
    print(f"  总图片: {n}")
    print(f"  总标注框: {total_boxes}")
    print(f"  输出目录: {output_dir}")
    print(f"\n目录结构:")
    for split_name in ["train", "valid", "test"]:
        img_count = len(list((output_dir / "images" / split_name).glob("*.png")))
        lbl_count = len(list((output_dir / "labels" / split_name).glob("*.txt")))
        print(f"  {split_name}: {img_count} 图片, {lbl_count} 标签")


if __name__ == "__main__":
    main()
