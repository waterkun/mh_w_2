"""
从 YOLO 标注数据生成 CRNN 训练数据

流程：
1. 读取 YOLO 格式的标注（txt 文件中的 bbox）
2. 根据 bbox 从原图裁剪出数字区域
3. 配合用户在 Roboflow 标注的数字值（metadata/tags），生成 labels.txt

标注备注格式约定：
  Roboflow 导出的标注若无法携带备注，可手动创建 annotations.json：
  {
    "frame_00001.png": [
      {"bbox": [x_center, y_center, w, h], "label": "127"},
      {"bbox": [x_center, y_center, w, h], "label": "54"}
    ]
  }

用法：
  python prepare_crnn_data.py --images datasets/damage_numbers/images/train
                              --labels datasets/damage_numbers/labels/train
                              --annotations annotations.json
                              --output crnn_data/train
"""

import argparse
import json
import os
from pathlib import Path

import cv2


def yolo_to_pixel(bbox_norm, img_w, img_h):
    """YOLO 归一化 bbox → 像素坐标 (x1, y1, x2, y2)"""
    x_center, y_center, w, h = bbox_norm
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    # 边界裁剪
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2


def prepare_from_annotations(images_dir, labels_dir, annotations_path, output_dir):
    """
    使用 annotations.json（含数字值）来生成 CRNN 数据。
    annotations.json 将每张图的每个 bbox 关联一个数字字符串。
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    entries = []
    crop_idx = 0

    for img_name, bboxes in annotations.items():
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取 {img_path}，跳过")
            continue

        img_h, img_w = img.shape[:2]

        for item in bboxes:
            bbox = item["bbox"]  # [x_center, y_center, w, h] 归一化
            label = str(item["label"])

            x1, y1, x2, y2 = yolo_to_pixel(bbox, img_w, img_h)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            crop_name = f"crop_{crop_idx:06d}.png"
            cv2.imwrite(os.path.join(output_dir, "images", crop_name), crop)
            entries.append(f"{crop_name}\t{label}")
            crop_idx += 1

    # 写入 labels.txt
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))

    print(f"CRNN 数据准备完成！")
    print(f"  裁剪图片: {crop_idx} 张")
    print(f"  标签文件: {labels_path}")
    print(f"  输出目录: {output_dir}")


def prepare_from_yolo_only(images_dir, labels_dir, output_dir):
    """
    仅使用 YOLO 标注裁剪数字区域（无数字值标签）。
    裁剪后需用户手动补充 labels.txt 中的数字值。
    同时生成 crop_mapping.json 记录每个 crop 对应的原图和 bbox 行号。
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    images = sorted(Path(images_dir).glob("*.png")) + \
             sorted(Path(images_dir).glob("*.jpg"))

    entries = []
    mapping = {}  # crop_name → {label_file, line_index, bbox_line}
    crop_idx = 0

    for img_path in images:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # class_id, x_center, y_center, w, h
            bbox = [float(x) for x in parts[1:5]]
            x1, y1, x2, y2 = yolo_to_pixel(bbox, img_w, img_h)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            crop_name = f"crop_{crop_idx:06d}.png"
            cv2.imwrite(os.path.join(output_dir, "images", crop_name), crop)
            entries.append(f"{crop_name}\t???")
            mapping[crop_name] = {
                "label_file": str(label_path),
                "line_index": line_idx,
                "bbox_line": line.strip(),
            }
            crop_idx += 1

    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))

    mapping_path = os.path.join(output_dir, "crop_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"CRNN 数据裁剪完成！")
    print(f"  裁剪图片: {crop_idx} 张")
    print(f"  标签文件: {labels_path}")
    print(f"  映射文件: {mapping_path}")
    print(f"  注意: labels.txt 中的数字值为 '???'，请手动填写实际数字！")


def main():
    parser = argparse.ArgumentParser(description="从 YOLO 标注生成 CRNN 训练数据")
    parser.add_argument("--images", required=True, help="图片目录")
    parser.add_argument("--labels", required=True, help="YOLO 标注目录")
    parser.add_argument("--annotations", default=None,
                        help="annotations.json 路径（含数字值）")
    parser.add_argument("--output", default="crnn_data/train", help="输出目录")
    args = parser.parse_args()

    if args.annotations and os.path.exists(args.annotations):
        print("使用 annotations.json 模式（含数字标签）")
        prepare_from_annotations(args.images, args.labels,
                                 args.annotations, args.output)
    else:
        print("使用纯 YOLO 标注模式（仅裁剪，需手动标注数字值）")
        prepare_from_yolo_only(args.images, args.labels, args.output)


if __name__ == "__main__":
    main()
