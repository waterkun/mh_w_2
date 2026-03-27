"""
合并审查数据 + 重训 YOLO & CRNN

流程:
  0. 清理上次错误合并的数据 (如有)
  1. 导出审查数据 (仅已审查的帧，带进度显示)
  2. 合并 YOLO 数据 (80/10/10 拆分追加到 valid-yolo-data)
  3. 合并 CRNN 数据 (80/20 拆分追加到 crnn_data)
  4. 删除 YOLO cache 文件
  5. 重训 YOLO (150 epochs, batch 16)
  6. 重训 CRNN (100 epochs, batch 32)

用法:
  E:/anaconda3/envs/mh_ai/python.exe hit_number_detection/merge_and_retrain.py
"""

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

random.seed(42)

BASE = Path(__file__).resolve().parent  # hit_number_detection/
PROJECT = BASE.parent                   # mh_w_2.1/

# 审查数据路径
REVIEW_DIR = BASE / "detection_review"
DETECTIONS_FILE = REVIEW_DIR / "detections.json"
YOLO_EXPORT = REVIEW_DIR / "yolo_export"
CRNN_EXPORT = REVIEW_DIR / "crnn_export"

# 源帧目录
FRAMES_DIR = BASE / "frames_filtered"

# 训练数据路径
YOLO_DATA = BASE / "valid-yolo-data"
CRNN_DATA = BASE / "crnn_data"

PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Step 0: 清理上次错误合并的数据
# ---------------------------------------------------------------------------
def step0_cleanup():
    print("=" * 60)
    print("Step 0: 清理上次错误合并的数据")
    print("=" * 60)

    cleaned = False

    # --- 清理 YOLO: 删除 yolo_export 中存在的文件 ---
    old_export_img = YOLO_EXPORT / "images"
    if old_export_img.exists():
        old_names = set(p.name for p in old_export_img.iterdir() if p.is_file())
        if old_names:
            yolo_removed = 0
            for split in ["train", "valid", "test"]:
                img_dir = YOLO_DATA / "images" / split
                lbl_dir = YOLO_DATA / "labels" / split
                if not img_dir.exists():
                    continue
                for f in list(img_dir.iterdir()):
                    if f.name in old_names:
                        f.unlink()
                        # 删除对应 label
                        lbl = lbl_dir / (f.stem + ".txt")
                        if lbl.exists():
                            lbl.unlink()
                        yolo_removed += 1
            if yolo_removed > 0:
                print(f"  YOLO: 删除了 {yolo_removed} 个错误合并的文件")
                cleaned = True

                # 验证
                for split in ["train", "valid", "test"]:
                    img_dir = YOLO_DATA / "images" / split
                    count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
                    print(f"    {split}: {count} 张")

    # --- 清理 CRNN: 删除 new_crop_* 文件和对应 label 行 ---
    for split_name in ["train", "val"]:
        split_dir = CRNN_DATA / split_name
        img_dir = split_dir / "images"
        labels_path = split_dir / "labels.txt"

        if not img_dir.exists():
            continue

        # 删除 new_crop_ 图片
        removed = 0
        for f in list(img_dir.iterdir()):
            if f.name.startswith("new_crop_"):
                f.unlink()
                removed += 1

        # 清理 labels.txt 中的 new_crop_ 行
        if labels_path.exists() and removed > 0:
            with open(labels_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            clean_lines = [l for l in lines if not l.startswith("new_crop_")]
            with open(labels_path, "w", encoding="utf-8") as f:
                f.writelines(clean_lines)

        if removed > 0:
            count = _count_crnn_labels(labels_path)
            print(f"  CRNN {split_name}: 删除了 {removed} 个 new_crop_, 剩余 {count} 条")
            cleaned = True

    if not cleaned:
        print("  无需清理")
    else:
        print("  清理完成!")


# ---------------------------------------------------------------------------
# Step 1: 导出审查数据 (仅已审查的帧)
# ---------------------------------------------------------------------------
def step1_export():
    print("\n" + "=" * 60)
    print("Step 1: 导出审查数据 (仅已审查的帧)")
    print("=" * 60)

    if not DETECTIONS_FILE.exists():
        print("错误: 无检测结果文件")
        sys.exit(1)

    with open(DETECTIONS_FILE, "r", encoding="utf-8") as f:
        detections = json.load(f)

    # 读取审查进度 — 只导出已审查的帧 (0 ~ last_idx)
    progress_file = REVIEW_DIR / "review_progress.json"
    if not progress_file.exists():
        print("错误: 无审查进度文件")
        sys.exit(1)

    with open(progress_file, "r") as f:
        progress = json.load(f)

    reviewed_names = set(progress.get("reviewed_frames", []))
    # 向后兼容
    if not reviewed_names and progress.get("last_idx", 0) > 0:
        all_frame_names = sorted(detections.keys())
        reviewed_names = set(all_frame_names[:progress["last_idx"] + 1])

    print(f"检测结果: {len(detections)} 帧, 已审查: {len(reviewed_names)} 帧")

    # 准备导出目录 (清空旧的)
    export_img_dir = YOLO_EXPORT / "images"
    export_lbl_dir = YOLO_EXPORT / "labels"
    crnn_dir = CRNN_EXPORT / "images"
    for d in [YOLO_EXPORT, CRNN_EXPORT]:
        if d.exists():
            shutil.rmtree(d)
    for d in [export_img_dir, export_lbl_dir, crnn_dir]:
        d.mkdir(parents=True, exist_ok=True)

    crnn_labels = []
    crop_idx = 0
    exported = 0
    total = len(reviewed_names)

    for i, frame_name in enumerate(sorted(reviewed_names)):
        data = detections.get(frame_name)
        if not data:
            continue
        dets = data["detections"]
        if not dets:
            continue

        img_path = FRAMES_DIR / frame_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        # 复制图片
        cv2.imwrite(str(export_img_dir / frame_name), img)

        # 写 YOLO label
        label_name = Path(frame_name).stem + ".txt"
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # CRNN crop
            number = d["number"]
            if number and number != "???" and number != "":
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(img_w, x2)
                y2c = min(img_h, y2)
                if x2c > x1c and y2c > y1c:
                    crop = img[y1c:y2c, x1c:x2c]
                    crop_name = f"new_crop_{crop_idx:06d}.png"
                    cv2.imwrite(str(crnn_dir / crop_name), crop)
                    crnn_labels.append(f"{crop_name}\t{number}")
                    crop_idx += 1

        with open(export_lbl_dir / label_name, "w") as f:
            f.write("\n".join(lines) + "\n")
        exported += 1

        # 进度显示
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  [{i + 1}/{total}] 已导出 {exported} 张图, {crop_idx} 个 crop")

    # 写 CRNN labels
    if crnn_labels:
        with open(CRNN_EXPORT / "labels.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(crnn_labels) + "\n")

    print(f"\nYOLO 导出: {exported} 张图 -> {YOLO_EXPORT}")
    print(f"CRNN 导出: {crop_idx} 个 crop -> {CRNN_EXPORT}")
    return exported, crop_idx


# ---------------------------------------------------------------------------
# Step 2: 合并 YOLO 数据
# ---------------------------------------------------------------------------
def step2_merge_yolo():
    print("\n" + "=" * 60)
    print("Step 2: 合并 YOLO 数据")
    print("=" * 60)

    src_img_dir = YOLO_EXPORT / "images"
    src_lbl_dir = YOLO_EXPORT / "labels"

    if not src_img_dir.exists():
        print("跳过: 无 YOLO 导出数据")
        return

    # 收集所有导出的图片 (stem 列表)
    stems = sorted([p.stem for p in src_img_dir.iterdir() if p.is_file()])
    if not stems:
        print("跳过: 无 YOLO 导出图片")
        return

    # 统计合并前
    before = {}
    for split in ["train", "valid", "test"]:
        d = YOLO_DATA / "images" / split
        before[split] = len(list(d.glob("*"))) if d.exists() else 0
    print(f"合并前: train={before['train']}, valid={before['valid']}, test={before['test']}")

    # 80/10/10 随机拆分
    random.shuffle(stems)
    n = len(stems)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    splits = {
        "train": stems[:n_train],
        "valid": stems[n_train:n_train + n_valid],
        "test": stems[n_train + n_valid:],
    }

    for split, split_stems in splits.items():
        print(f"  {split}: +{len(split_stems)} 张")

    # 复制文件
    for split, split_stems in splits.items():
        dst_img = YOLO_DATA / "images" / split
        dst_lbl = YOLO_DATA / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            # 找图片 (可能是 png 或 jpg)
            src_img = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = src_img_dir / (stem + ext)
                if candidate.exists():
                    src_img = candidate
                    break
            if src_img is None:
                continue

            shutil.copy2(src_img, dst_img / src_img.name)

            # 标签
            src_lbl = src_lbl_dir / (stem + ".txt")
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl / src_lbl.name)

    # 统计合并后
    after = {}
    for split in ["train", "valid", "test"]:
        d = YOLO_DATA / "images" / split
        after[split] = len(list(d.glob("*"))) if d.exists() else 0
    print(f"合并后: train={after['train']}, valid={after['valid']}, test={after['test']}")
    total_new = sum(after[s] - before[s] for s in ["train", "valid", "test"])
    print(f"YOLO 总计新增: {total_new} 张")


# ---------------------------------------------------------------------------
# Step 3: 合并 CRNN 数据
# ---------------------------------------------------------------------------
def step3_merge_crnn():
    print("\n" + "=" * 60)
    print("Step 3: 合并 CRNN 数据")
    print("=" * 60)

    src_img_dir = CRNN_EXPORT / "images"
    src_labels = CRNN_EXPORT / "labels.txt"

    if not src_labels.exists():
        print("跳过: 无 CRNN 导出数据")
        return

    # 读取导出的 labels
    samples = []
    with open(src_labels, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                samples.append((parts[0], parts[1]))

    if not samples:
        print("跳过: 无 CRNN 样本")
        return

    # 统计合并前
    before_train = _count_crnn_labels(CRNN_DATA / "train" / "labels.txt")
    before_val = _count_crnn_labels(CRNN_DATA / "val" / "labels.txt")
    print(f"合并前: train={before_train}, val={before_val}")

    # 80/20 拆分
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.8)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    print(f"  train: +{len(train_samples)} 个 crop")
    print(f"  val:   +{len(val_samples)} 个 crop")

    # 追加到 train
    _append_crnn(CRNN_DATA / "train", src_img_dir, train_samples)
    # 追加到 val
    _append_crnn(CRNN_DATA / "val", src_img_dir, val_samples)

    # 统计合并后
    after_train = _count_crnn_labels(CRNN_DATA / "train" / "labels.txt")
    after_val = _count_crnn_labels(CRNN_DATA / "val" / "labels.txt")
    print(f"合并后: train={after_train}, val={after_val}")
    total_new = (after_train - before_train) + (after_val - before_val)
    print(f"CRNN 总计新增: {total_new} 个 crop")


def _count_crnn_labels(labels_path):
    if not labels_path.exists():
        return 0
    with open(labels_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _append_crnn(split_dir, src_img_dir, samples):
    """追加 crop 图片和 labels 到指定 split 目录"""
    dst_img_dir = split_dir / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = split_dir / "labels.txt"

    new_lines = []
    for img_name, label in samples:
        src = src_img_dir / img_name
        if src.exists():
            shutil.copy2(src, dst_img_dir / img_name)
            new_lines.append(f"{img_name}\t{label}")

    if new_lines:
        # 确保已有文件以换行结尾
        needs_newline = False
        if labels_path.exists() and labels_path.stat().st_size > 0:
            with open(labels_path, "rb") as fb:
                fb.seek(-1, 2)
                if fb.read(1) != b"\n":
                    needs_newline = True

        with open(labels_path, "a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write("\n".join(new_lines) + "\n")


# ---------------------------------------------------------------------------
# Step 4: 删除 YOLO cache
# ---------------------------------------------------------------------------
def step4_clear_cache():
    print("\n" + "=" * 60)
    print("Step 4: 删除 YOLO cache 文件")
    print("=" * 60)

    deleted = 0
    for cache in YOLO_DATA.rglob("*.cache"):
        cache.unlink()
        print(f"  已删除: {cache}")
        deleted += 1

    if deleted == 0:
        print("  无 cache 文件")
    else:
        print(f"  共删除 {deleted} 个 cache 文件")


# ---------------------------------------------------------------------------
# Step 5: 重训 YOLO
# ---------------------------------------------------------------------------
def step5_train_yolo():
    print("\n" + "=" * 60)
    print("Step 5: 重训 YOLO (150 epochs, batch 16)")
    print("=" * 60)

    result = subprocess.run(
        [PYTHON, str(BASE / "train_yolo.py"),
         "--epochs", "150", "--batch", "16"],
        cwd=str(BASE),
    )
    if result.returncode != 0:
        print("警告: YOLO 训练异常退出")
        return False
    return True


# ---------------------------------------------------------------------------
# Step 6: 重训 CRNN
# ---------------------------------------------------------------------------
def step6_train_crnn():
    print("\n" + "=" * 60)
    print("Step 6: 重训 CRNN (100 epochs, batch 32)")
    print("=" * 60)

    result = subprocess.run(
        [PYTHON, str(BASE / "train_crnn.py"),
         "--epochs", "100", "--batch", "32"],
        cwd=str(BASE),
    )
    if result.returncode != 0:
        print("警告: CRNN 训练异常退出")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("合并审查数据 + 重训 YOLO & CRNN")
    print("=" * 60)
    print(f"YOLO 数据: {YOLO_DATA}")
    print(f"CRNN 数据: {CRNN_DATA}")
    print(f"审查数据: {REVIEW_DIR}")
    print()

    # Step 0: 清理
    step0_cleanup()

    # Step 1: 导出
    n_yolo, n_crnn = step1_export()

    # Step 2-3: 合并
    if n_yolo > 0:
        step2_merge_yolo()
    if n_crnn > 0:
        step3_merge_crnn()

    # Step 4: 清 cache
    step4_clear_cache()

    # Step 5-6: 训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    step5_train_yolo()
    step6_train_crnn()

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
