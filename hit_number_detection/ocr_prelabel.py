"""
OCR 预标注工具 — 用 EasyOCR 自动识别裁剪的伤害数字

流程：
1. 从 valid-yolo-data/ 裁剪所有 bbox 到 crnn_data/ocr_prelabel/images/
2. 用 EasyOCR 识别每个 crop 的数字值
3. 生成 labels.txt（预填 OCR 结果）和 crop_mapping.json
4. 之后用 label_crnn_data.py 审查确认/修正/删除

用法：
  python ocr_prelabel.py
  python ocr_prelabel.py --yolo-dir valid-yolo-data --output crnn_data/ocr_prelabel
"""

import argparse
import json
import os
import re
import cv2
import numpy as np
from pathlib import Path


def yolo_to_pixel(bbox_norm, img_w, img_h):
    x_center, y_center, w, h = bbox_norm
    x1 = max(0, int((x_center - w / 2) * img_w))
    y1 = max(0, int((y_center - h / 2) * img_h))
    x2 = min(img_w, int((x_center + w / 2) * img_w))
    y2 = min(img_h, int((y_center + h / 2) * img_h))
    return x1, y1, x2, y2


def ocr_crop(reader, crop_img):
    """用 EasyOCR 识别 crop 中的数字，返回纯数字字符串。"""
    # EasyOCR 识别
    results = reader.readtext(crop_img, allowlist="0123456789",
                               detail=1, paragraph=False)

    if not results:
        return "???"

    # 合并所有识别到的文本，只保留数字
    texts = []
    for (bbox, text, conf) in results:
        digits = re.sub(r"[^0-9]", "", text)
        if digits:
            texts.append(digits)

    if texts:
        return "".join(texts)
    return "???"


def main():
    parser = argparse.ArgumentParser(description="OCR 预标注伤害数字")
    parser.add_argument("--yolo-dir", default="valid-yolo-data",
                        help="YOLO 数据目录 (默认 valid-yolo-data)")
    parser.add_argument("--output", default="crnn_data/ocr_prelabel",
                        help="输出目录 (默认 crnn_data/ocr_prelabel)")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="使用 GPU (默认开启)")
    args = parser.parse_args()

    base = Path(__file__).parent
    yolo_dir = base / args.yolo_dir
    output_dir = base / args.output
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # 初始化 EasyOCR
    print("正在加载 EasyOCR 模型...")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=args.gpu)
    print("EasyOCR 模型加载完成！")

    entries = []
    mapping = {}
    crop_idx = 0
    ocr_success = 0

    for split in ["train", "valid", "test"]:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split

        if not img_dir.exists():
            print(f"  跳过 {split}（目录不存在）")
            continue

        images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        print(f"\n处理 {split}: {len(images)} 张图片")

        for img_idx, img_path in enumerate(images):
            label_path = lbl_dir / (img_path.stem + ".txt")
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
                bbox = [float(x) for x in parts[1:5]]
                x1, y1, x2, y2 = yolo_to_pixel(bbox, img_w, img_h)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                crop_name = f"crop_{crop_idx:06d}.png"
                cv2.imwrite(str(images_out / crop_name), crop)

                # OCR 识别
                ocr_result = ocr_crop(reader, crop)
                if ocr_result != "???":
                    ocr_success += 1

                entries.append(f"{crop_name}\t{ocr_result}")
                mapping[crop_name] = {
                    "label_file": str(label_path),
                    "line_index": line_idx,
                    "bbox_line": line.strip(),
                    "split": split,
                    "source_image": img_path.name,
                }
                crop_idx += 1

            # 进度
            if (img_idx + 1) % 50 == 0:
                print(f"  {split}: {img_idx + 1}/{len(images)} 张已处理")

    # 保存
    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))

    mapping_path = output_dir / "crop_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print(f"OCR 预标注完成！")
    print(f"  总 crop 数: {crop_idx}")
    print(f"  OCR 成功识别: {ocr_success} ({ocr_success*100//max(crop_idx,1)}%)")
    print(f"  未识别 (???): {crop_idx - ocr_success}")
    print(f"  输出目录: {output_dir}")
    print(f"  标签文件: {labels_path}")
    print(f"  映射文件: {mapping_path}")
    print(f"{'=' * 50}")
    print(f"\n下一步：审查 OCR 结果")
    print(f"  E:/anaconda3/envs/mh_ai/python.exe label_crnn_data.py --data {args.output}")


if __name__ == "__main__":
    main()
