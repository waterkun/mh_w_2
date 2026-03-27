"""
生成修正前后的 bbox 对比图，供人工审查。

对每个被修正的 1XX bbox:
- 左图: 原始 bbox (红框)
- 右图: 修正后 bbox (绿框)
- 下方: 修正前后的 crop 放大对比

Usage:
    E:/anaconda3/envs/mh_ai/python.exe hit_number_detection/review_bbox_fix.py
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
YOLO_DIR = BASE / "valid-yolo-data"
FIXED_DIR = BASE / "yolo_diagnosis" / "fixed_labels"
LABELS_TXT = BASE / "crnn_data" / "ocr_prelabel" / "labels.txt"
CROP_MAPPING = BASE / "crnn_data" / "ocr_prelabel" / "crop_mapping.json"
OUTPUT_DIR = BASE / "yolo_diagnosis" / "fix_review"


def yolo_to_pixel(bbox_norm, img_w, img_h):
    x_center, y_center, w, h = bbox_norm
    x1 = max(0, int((x_center - w / 2) * img_w))
    y1 = max(0, int((y_center - h / 2) * img_h))
    x2 = min(img_w, int((x_center + w / 2) * img_w))
    y2 = min(img_h, int((y_center + h / 2) * img_h))
    return x1, y1, x2, y2


def parse_label_file(path):
    """解析 label 文件，返回 [(cls, cx, cy, w, h), ...]"""
    bboxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                bboxes.append(tuple(float(x) for x in parts[:5]))
    return bboxes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 CRNN 标签
    crnn_labels = {}
    with open(LABELS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                name, text = line.strip().split("\t", 1)
                crnn_labels[name.strip()] = text.strip()

    # 加载 crop_mapping
    with open(CROP_MAPPING, "r", encoding="utf-8") as f:
        crop_mapping = json.load(f)

    # 找出所有被修改的 label 文件
    fixed_files = []
    for split in ["train", "valid", "test"]:
        fixed_dir = FIXED_DIR / "labels" / split
        if not fixed_dir.exists():
            continue
        for f in sorted(fixed_dir.glob("*.txt")):
            orig = YOLO_DIR / "labels" / split / f.name
            # 注意: 原始文件已被覆盖，所以当前 orig == fixed
            # 但 fixed_dir 里存的就是修正后的版本
            # 原始版本需要从 crop_mapping 的 bbox_line 还原
            fixed_files.append((split, f.name, f))

    print(f"找到 {len(fixed_files)} 个修正过的 label 文件")

    # 建立: (split, label_filename, line_index) -> (crop_name, digit_text, original_bbox_line)
    orig_bbox_map = {}
    for crop_name, info in crop_mapping.items():
        label_path = Path(info["label_file"])
        key = (info["split"], label_path.name, info["line_index"])
        digit_text = crnn_labels.get(crop_name, "?")
        orig_bbox_map[key] = {
            "crop_name": crop_name,
            "digit_text": digit_text,
            "bbox_line": info["bbox_line"],
        }

    # 逐文件生成对比
    total_pairs = 0
    for split, label_name, fixed_path in fixed_files:
        img_stem = Path(label_name).stem
        # 查找对应图片
        img_path = None
        for ext in [".png", ".jpg"]:
            candidate = YOLO_DIR / "images" / split / (img_stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 解析修正后的 bbox
        fixed_bboxes = parse_label_file(fixed_path)

        # 找出哪些行被修改了
        for line_idx, fixed_bb in enumerate(fixed_bboxes):
            key = (split, label_name, line_idx)
            orig_info = orig_bbox_map.get(key)
            if orig_info is None:
                continue

            digit_text = orig_info["digit_text"]
            # 只关注 3 位数以 1 开头的
            if len(digit_text) != 3 or digit_text[0] != "1":
                continue

            # 解析原始 bbox
            orig_parts = orig_info["bbox_line"].split()
            orig_bb = tuple(float(x) for x in orig_parts[:5])

            # 检查是否确实有修改
            if abs(orig_bb[3] - fixed_bb[3]) < 1e-6:  # w 没变
                continue

            # 生成对比图
            # 在原图上画两个框
            canvas = img.copy()
            # 原始框 (红色)
            ox1, oy1, ox2, oy2 = yolo_to_pixel(orig_bb[1:], img_w, img_h)
            cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
            # 修正框 (绿色)
            fx1, fy1, fx2, fy2 = yolo_to_pixel(fixed_bb[1:], img_w, img_h)
            cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

            # 裁剪局部区域 (包含两个框 + padding)
            pad = 40
            region_x1 = max(0, min(ox1, fx1) - pad)
            region_y1 = max(0, min(oy1, fy1) - pad)
            region_x2 = min(img_w, max(ox2, fx2) + pad)
            region_y2 = min(img_h, max(oy2, fy2) + pad)
            region = canvas[region_y1:region_y2, region_x1:region_x2]

            # 放大裁剪区域
            scale = max(1, 300 // max(region.shape[1], 1))
            if scale > 1:
                region = cv2.resize(region, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_NEAREST)

            # 原始 crop vs 修正 crop (放大)
            orig_crop = img[oy1:oy2, ox1:ox2]
            fixed_crop = img[fy1:fy2, fx1:fx2]
            crop_h = 60
            if orig_crop.size > 0:
                orig_crop = cv2.resize(orig_crop, (int(orig_crop.shape[1] * crop_h / max(orig_crop.shape[0], 1)), crop_h),
                                       interpolation=cv2.INTER_NEAREST)
            if fixed_crop.size > 0:
                fixed_crop = cv2.resize(fixed_crop, (int(fixed_crop.shape[1] * crop_h / max(fixed_crop.shape[0], 1)), crop_h),
                                        interpolation=cv2.INTER_NEAREST)

            # 拼接: 上方局部对比，下方两个 crop 并排
            # 添加标签
            label_bar = np.zeros((30, region.shape[1], 3), dtype=np.uint8)
            cv2.putText(label_bar, f"'{digit_text}' red=old green=fixed",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # crop 并排
            gap = np.ones((crop_h, 10, 3), dtype=np.uint8) * 128
            crops_row_parts = []
            if orig_crop.size > 0:
                crops_row_parts.append(orig_crop)
            crops_row_parts.append(gap)
            if fixed_crop.size > 0:
                crops_row_parts.append(fixed_crop)

            if len(crops_row_parts) >= 3:
                crops_row = np.hstack(crops_row_parts)
                # 对齐宽度
                target_w = region.shape[1]
                if crops_row.shape[1] < target_w:
                    pad_right = np.zeros((crops_row.shape[0], target_w - crops_row.shape[1], 3), dtype=np.uint8)
                    crops_row = np.hstack([crops_row, pad_right])
                else:
                    crops_row = crops_row[:, :target_w]

                crop_label = np.zeros((20, target_w, 3), dtype=np.uint8)
                cv2.putText(crop_label, "old crop | fixed crop",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                final = np.vstack([label_bar, region, crop_label, crops_row])
            else:
                final = np.vstack([label_bar, region])

            out_name = f"{digit_text}_{img_stem}_line{line_idx}.png"
            cv2.imwrite(str(OUTPUT_DIR / out_name), final)
            total_pairs += 1

    print(f"生成 {total_pairs} 张对比图 -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
