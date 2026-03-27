"""
将所有 YOLO 数据集（train/valid/test）的 bbox 裁剪到一个临时文件夹供审查。
"""

import json
import os
import cv2
from pathlib import Path


def yolo_to_pixel(bbox_norm, img_w, img_h):
    x_center, y_center, w, h = bbox_norm
    x1 = max(0, int((x_center - w / 2) * img_w))
    y1 = max(0, int((y_center - h / 2) * img_h))
    x2 = min(img_w, int((x_center + w / 2) * img_w))
    y2 = min(img_h, int((y_center + h / 2) * img_h))
    return x1, y1, x2, y2


def main():
    base = Path(__file__).parent
    dataset_dir = base / "datasets" / "damage_numbers"
    output_dir = base / "crnn_data" / "review_all"
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    entries = []
    mapping = {}
    crop_idx = 0

    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            continue

        images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        print(f"{split}: {len(images)} images")

        for img_path in images:
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
                entries.append(f"{crop_name}\t???")
                mapping[crop_name] = {
                    "label_file": str(label_path),
                    "line_index": line_idx,
                    "bbox_line": line.strip(),
                    "split": split,
                    "source_image": img_path.name,
                }
                crop_idx += 1

    # Save
    with open(output_dir / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(entries))

    with open(output_dir / "crop_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {crop_idx} crops in {output_dir}")
    print(f"Run: python review_crops.py --data crnn_data/review_all")


if __name__ == "__main__":
    main()
