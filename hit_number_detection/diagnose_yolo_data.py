"""
YOLO 标注数据问题排查脚本

分析 valid-yolo-data/ 的标注质量，排查三类问题：
1. Bbox 偏窄（尤其以 "1" 开头的三位数）
2. 左侧留白不足（数字笔画紧贴左边缘）
3. 假阳性（异常小的 bbox）

不需要加载模型，纯统计分析。

Usage:
    E:/anaconda3/envs/mh_ai/python.exe hit_number_detection/diagnose_yolo_data.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
YOLO_DIR = BASE / "valid-yolo-data"
LABELS_TXT = BASE / "crnn_data" / "ocr_prelabel" / "labels.txt"
CROP_MAPPING = BASE / "crnn_data" / "ocr_prelabel" / "crop_mapping.json"

OUTPUT_DIR = BASE / "yolo_diagnosis"
NARROW_DIR = OUTPUT_DIR / "narrow_bbox_crops"
EDGE_DIR = OUTPUT_DIR / "edge_touching_crops"
SMALL_DIR = OUTPUT_DIR / "small_bbox_crops"
STATS_DIR = OUTPUT_DIR / "stats"


def yolo_to_pixel(bbox_norm, img_w, img_h):
    """YOLO 归一化坐标 -> 像素坐标 (x1, y1, x2, y2)"""
    x_center, y_center, w, h = bbox_norm
    x1 = max(0, int((x_center - w / 2) * img_w))
    y1 = max(0, int((y_center - h / 2) * img_h))
    x2 = min(img_w, int((x_center + w / 2) * img_w))
    y2 = min(img_h, int((y_center + h / 2) * img_h))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_crnn_labels():
    """加载 labels.txt -> {crop_name: digit_text}"""
    mapping = {}
    with open(LABELS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            name, text = line.split("\t", 1)
            mapping[name.strip()] = text.strip()
    return mapping


def load_crop_mapping():
    """加载 crop_mapping.json -> {crop_name: {label_file, line_index, bbox_line, split, source_image}}"""
    with open(CROP_MAPPING, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_label_line(label_file, line_index):
    """从当前 label 文件读取指定行的 bbox（反映手动标注后的最新值）"""
    try:
        with open(label_file, "r") as f:
            lines = f.readlines()
        if line_index < len(lines):
            parts = lines[line_index].strip().split()
            if len(parts) >= 5:
                return tuple(float(x) for x in parts[1:5])
    except FileNotFoundError:
        pass
    return None


def build_bbox_records(crnn_labels, crop_mapping):
    """
    合并 CRNN 标签和 bbox 信息，返回记录列表。
    优先从当前 label 文件读取 bbox（反映手动修改），
    读取失败则回退到 crop_mapping 中的原始值。
    """
    records = []
    for crop_name, info in crop_mapping.items():
        digit_text = crnn_labels.get(crop_name)
        if digit_text is None:
            continue

        # 优先读当前 label 文件（反映手动标注）
        bbox_norm = _read_label_line(info["label_file"], info["line_index"])
        if bbox_norm is None:
            # 回退到 crop_mapping 原始值
            parts = info["bbox_line"].split()
            if len(parts) < 5:
                continue
            bbox_norm = tuple(float(x) for x in parts[1:5])

        records.append({
            "crop_name": crop_name,
            "digit_text": digit_text,
            "num_digits": len(digit_text),
            "first_digit": digit_text[0] if digit_text else "?",
            "bbox_norm": bbox_norm,  # (cx, cy, w, h) normalized
            "split": info["split"],
            "source_image": info["source_image"],
            "label_file": info["label_file"],
        })
    return records


def resolve_image_path(source_image, split):
    """根据 source_image 名 + split 找到实际图片路径"""
    # crop_mapping 里 source_image 可能是 frame1_00073.png 形式
    # 实际 valid-yolo-data/images/ 下可能是 frame_00073.png 或同名
    img_dir = YOLO_DIR / "images" / split
    candidate = img_dir / source_image
    if candidate.exists():
        return candidate
    # 尝试去掉 frame 后面的数字前缀差异
    # e.g. frame1_00073.png -> frame_00073.png 不一定对，直接用 stem 匹配
    return None


def get_image_size_cached(path, cache={}):
    """读取图片尺寸，带缓存"""
    path_str = str(path)
    if path_str not in cache:
        img = cv2.imread(path_str)
        if img is None:
            cache[path_str] = None
        else:
            cache[path_str] = (img.shape[1], img.shape[0])  # (w, h)
    return cache[path_str]


# ---------------------------------------------------------------------------
# Check 1: Bbox 宽高比分析
# ---------------------------------------------------------------------------

def check_bbox_width(records, report_lines):
    """按位数 & 首位数字分组，对比 bbox 归一化宽度"""
    report_lines.append("=" * 70)
    report_lines.append("检查 1: Bbox 宽度分析 (归一化宽度)")
    report_lines.append("=" * 70)

    # 按位数分组
    by_digits = defaultdict(list)
    for r in records:
        by_digits[r["num_digits"]].append(r["bbox_norm"][2])  # w

    report_lines.append("\n--- 按位数分组的 bbox 归一化宽度 ---")
    report_lines.append(f"{'位数':<6} {'数量':<8} {'平均宽度':<12} {'中位数':<12} {'标准差':<12} {'最小':<12} {'最大':<12}")
    for nd in sorted(by_digits.keys()):
        ws = np.array(by_digits[nd])
        report_lines.append(
            f"{nd:<6} {len(ws):<8} {ws.mean():.6f}     {np.median(ws):.6f}     "
            f"{ws.std():.6f}     {ws.min():.6f}     {ws.max():.6f}"
        )

    # 按首位数字分组 (仅 3 位数)
    report_lines.append("\n--- 3 位数: 按首位数字分组的 bbox 归一化宽度 ---")
    three_digit = [r for r in records if r["num_digits"] == 3]
    by_first = defaultdict(list)
    for r in three_digit:
        by_first[r["first_digit"]].append(r["bbox_norm"][2])

    report_lines.append(f"{'首位':<6} {'数量':<8} {'平均宽度':<12} {'中位数':<12} {'标准差':<12}")
    for fd in sorted(by_first.keys()):
        ws = np.array(by_first[fd])
        report_lines.append(
            f"{fd:<6} {len(ws):<8} {ws.mean():.6f}     {np.median(ws):.6f}     {ws.std():.6f}"
        )

    # 以1开头 vs 其他
    w1 = [r["bbox_norm"][2] for r in three_digit if r["first_digit"] == "1"]
    w_other = [r["bbox_norm"][2] for r in three_digit if r["first_digit"] != "1"]
    if w1 and w_other:
        mean1, mean_o = np.mean(w1), np.mean(w_other)
        report_lines.append(f"\n以 '1' 开头的 3 位数平均宽度: {mean1:.6f} (n={len(w1)})")
        report_lines.append(f"以 '2-9' 开头的 3 位数平均宽度: {mean_o:.6f} (n={len(w_other)})")
        report_lines.append(f"差异: {(mean_o - mean1) / mean_o * 100:.1f}% (正值 = '1' 开头更窄)")

    # 同样分析 2 位数
    report_lines.append("\n--- 2 位数: 按首位数字分组的 bbox 归一化宽度 ---")
    two_digit = [r for r in records if r["num_digits"] == 2]
    by_first_2 = defaultdict(list)
    for r in two_digit:
        by_first_2[r["first_digit"]].append(r["bbox_norm"][2])

    report_lines.append(f"{'首位':<6} {'数量':<8} {'平均宽度':<12} {'中位数':<12}")
    for fd in sorted(by_first_2.keys()):
        ws = np.array(by_first_2[fd])
        report_lines.append(f"{fd:<6} {len(ws):<8} {ws.mean():.6f}     {np.median(ws):.6f}")

    # 找出偏窄的 3 位数 bbox (宽度低于 3 位数中位数 - 1.5*IQR 或低于 2 位数平均值)
    narrow_records = []
    if three_digit:
        ws_3 = np.array([r["bbox_norm"][2] for r in three_digit])
        q1, q3 = np.percentile(ws_3, 25), np.percentile(ws_3, 75)
        iqr = q3 - q1
        threshold = q1 - 1.0 * iqr  # 较宽松的阈值
        two_digit_ws = [r["bbox_norm"][2] for r in records if r["num_digits"] == 2]
        two_digit_mean = np.mean(two_digit_ws) if two_digit_ws else 0

        # 选择更有意义的阈值: 3位数宽度 < 2位数平均宽度
        effective_threshold = max(threshold, two_digit_mean) if two_digit_mean else threshold
        report_lines.append(f"\n偏窄阈值: {effective_threshold:.6f} (3位数Q1-IQR={threshold:.6f}, 2位数均值={two_digit_mean:.6f})")

        for r in three_digit:
            if r["bbox_norm"][2] < effective_threshold:
                narrow_records.append(r)

        report_lines.append(f"偏窄的 3 位数 bbox 数量: {len(narrow_records)}")
        for r in sorted(narrow_records, key=lambda x: x["bbox_norm"][2]):
            report_lines.append(
                f"  {r['crop_name']} -> '{r['digit_text']}' w={r['bbox_norm'][2]:.6f} "
                f"src={r['source_image']}"
            )

    return narrow_records


# ---------------------------------------------------------------------------
# Check 2: 左侧留白分析
# ---------------------------------------------------------------------------

def check_left_margin(records, report_lines):
    """检查 crop 图像左侧留白是否充足"""
    report_lines.append("\n" + "=" * 70)
    report_lines.append("检查 2: 左侧留白分析")
    report_lines.append("=" * 70)

    crop_images_dir = BASE / "crnn_data" / "ocr_prelabel" / "images"
    if not crop_images_dir.exists():
        report_lines.append(f"crop 图像目录不存在: {crop_images_dir}")
        return []

    edge_records = []
    margin_by_first_digit = defaultdict(list)
    margin_cols = 3  # 检查最左 3 列

    for r in records:
        crop_path = crop_images_dir / r["crop_name"]
        if not crop_path.exists():
            continue

        img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[1] < margin_cols + 1:
            continue

        # 左侧 margin_cols 列的平均亮度
        left_strip = img[:, :margin_cols]
        left_brightness = float(np.mean(left_strip))

        # 整体平均亮度 (用于归一化)
        overall_brightness = float(np.mean(img))

        # 计算左侧相对亮度 (如果左侧很亮 = 有笔画紧贴边缘)
        r["left_brightness"] = left_brightness
        r["overall_brightness"] = overall_brightness

        # 按首位数字分组收集
        margin_by_first_digit[r["first_digit"]].append(left_brightness)

        # 检测: 左侧亮度相对整体很高 → 数字笔画紧贴左边缘
        # 使用相对比例: 左侧亮度 / 整体亮度 > 0.9 且 左侧绝对亮度 > 120
        if overall_brightness > 10:
            left_ratio = left_brightness / overall_brightness
            r["left_ratio"] = left_ratio
            if left_ratio > 0.9 and left_brightness > 120:
                edge_records.append(r)

    report_lines.append(f"\n--- 按首位数字分组的左侧 {margin_cols} 列平均亮度 ---")
    report_lines.append(f"{'首位':<6} {'数量':<8} {'平均亮度':<12} {'中位数':<12}")
    for fd in sorted(margin_by_first_digit.keys()):
        vals = np.array(margin_by_first_digit[fd])
        report_lines.append(f"{fd:<6} {len(vals):<8} {vals.mean():.1f}         {np.median(vals):.1f}")

    # 按位数分组
    report_lines.append(f"\n--- 按位数分组的左侧留白亮度 ---")
    margin_by_ndigits = defaultdict(list)
    for r in records:
        if "left_brightness" in r:
            margin_by_ndigits[r["num_digits"]].append(r["left_brightness"])

    report_lines.append(f"{'位数':<6} {'数量':<8} {'平均亮度':<12} {'中位数':<12}")
    for nd in sorted(margin_by_ndigits.keys()):
        vals = np.array(margin_by_ndigits[nd])
        report_lines.append(f"{nd:<6} {len(vals):<8} {vals.mean():.1f}         {np.median(vals):.1f}")

    # 3位数以1开头 vs 其他
    three_digit = [r for r in records if r["num_digits"] == 3 and "left_brightness" in r]
    bright_1 = [r["left_brightness"] for r in three_digit if r["first_digit"] == "1"]
    bright_o = [r["left_brightness"] for r in three_digit if r["first_digit"] != "1"]
    if bright_1 and bright_o:
        report_lines.append(f"\n3位数以 '1' 开头左侧平均亮度: {np.mean(bright_1):.1f} (n={len(bright_1)})")
        report_lines.append(f"3位数以 '2-9' 开头左侧平均亮度: {np.mean(bright_o):.1f} (n={len(bright_o)})")

    report_lines.append(f"\n左边缘留白不足 (左侧亮度/整体 > 0.9 且绝对 > 120) 的 crop 数量: {len(edge_records)}")
    # 按亮度排序，显示前 30
    edge_records.sort(key=lambda x: x["left_brightness"], reverse=True)
    for r in edge_records[:30]:
        report_lines.append(
            f"  {r['crop_name']} -> '{r['digit_text']}' 左侧亮度={r['left_brightness']:.1f} "
            f"整体={r['overall_brightness']:.1f}"
        )
    if len(edge_records) > 30:
        report_lines.append(f"  ... 还有 {len(edge_records) - 30} 条")

    return edge_records


# ---------------------------------------------------------------------------
# Check 3: 假阳性排查 (异常小的 bbox)
# ---------------------------------------------------------------------------

def check_small_bbox(records, report_lines):
    """找出尺寸异常小的 bbox，可能是非数字误标"""
    report_lines.append("\n" + "=" * 70)
    report_lines.append("检查 3: 假阳性排查 (异常小/极端宽高比 bbox)")
    report_lines.append("=" * 70)

    widths = np.array([r["bbox_norm"][2] for r in records])
    heights = np.array([r["bbox_norm"][3] for r in records])
    areas = widths * heights
    aspect_ratios = widths / np.maximum(heights, 1e-6)

    mean_area = areas.mean()
    mean_w = widths.mean()
    mean_h = heights.mean()

    report_lines.append(f"\n整体 bbox 统计 (归一化):")
    report_lines.append(f"  宽度:  均值={mean_w:.6f}  中位={np.median(widths):.6f}  std={widths.std():.6f}")
    report_lines.append(f"  高度:  均值={mean_h:.6f}  中位={np.median(heights):.6f}  std={heights.std():.6f}")
    report_lines.append(f"  面积:  均值={mean_area:.8f}  中位={np.median(areas):.8f}")
    report_lines.append(f"  宽高比: 均值={aspect_ratios.mean():.3f}  中位={np.median(aspect_ratios):.3f}")

    # 异常小: 面积 < 均值 50%
    small_records = []
    area_threshold = mean_area * 0.5
    # 极端宽高比: < 0.2 或 > 2.0
    for i, r in enumerate(records):
        a = areas[i]
        ar = aspect_ratios[i]
        reason = []
        if a < area_threshold:
            reason.append(f"面积偏小({a:.8f} < {area_threshold:.8f})")
        if ar < 0.2:
            reason.append(f"宽高比极小({ar:.3f})")
        if ar > 2.0:
            reason.append(f"宽高比极大({ar:.3f})")
        if reason:
            r["small_reason"] = "; ".join(reason)
            r["area"] = a
            r["aspect_ratio"] = ar
            small_records.append(r)

    report_lines.append(f"\n异常 bbox 数量: {len(small_records)} (面积<50%均值 或 宽高比极端)")
    for r in sorted(small_records, key=lambda x: x.get("area", 0)):
        report_lines.append(
            f"  {r['crop_name']} -> '{r['digit_text']}' "
            f"w={r['bbox_norm'][2]:.6f} h={r['bbox_norm'][3]:.6f} "
            f"AR={r.get('aspect_ratio', 0):.3f} | {r['small_reason']}"
        )

    return small_records


# ---------------------------------------------------------------------------
# Check 4: 综合统计 + 图表
# ---------------------------------------------------------------------------

def generate_stats_plots(records):
    """生成统计图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过图表生成")
        return

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    widths = [r["bbox_norm"][2] for r in records]
    heights = [r["bbox_norm"][3] for r in records]
    aspect_ratios = [w / max(h, 1e-6) for w, h in zip(widths, heights)]

    # --- 图1: 宽度/高度/宽高比 分布直方图 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(widths, bins=50, color="steelblue", edgecolor="black")
    axes[0].set_title("Bbox Width Distribution (normalized)")
    axes[0].set_xlabel("Width")

    axes[1].hist(heights, bins=50, color="coral", edgecolor="black")
    axes[1].set_title("Bbox Height Distribution (normalized)")
    axes[1].set_xlabel("Height")

    axes[2].hist(aspect_ratios, bins=50, color="seagreen", edgecolor="black")
    axes[2].set_title("Aspect Ratio (W/H) Distribution")
    axes[2].set_xlabel("W/H")

    plt.tight_layout()
    plt.savefig(str(STATS_DIR / "distributions.png"), dpi=150)
    plt.close()

    # --- 图2: 按位数分组的 bbox 宽度箱线图 ---
    by_digits = defaultdict(list)
    for r in records:
        by_digits[r["num_digits"]].append(r["bbox_norm"][2])

    fig, ax = plt.subplots(figsize=(8, 5))
    digit_groups = sorted(by_digits.keys())
    data = [by_digits[d] for d in digit_groups]
    bp = ax.boxplot(data, tick_labels=[f"{d}-digit (n={len(by_digits[d])})" for d in digit_groups],
                    patch_artist=True)
    colors = ["#AED6F1", "#F9E79F", "#F5B7B1", "#A9DFBF"]
    for patch, color in zip(bp["boxes"], colors[:len(digit_groups)]):
        patch.set_facecolor(color)
    ax.set_title("Bbox Width by Digit Count")
    ax.set_ylabel("Normalized Width")
    plt.tight_layout()
    plt.savefig(str(STATS_DIR / "width_by_digits.png"), dpi=150)
    plt.close()

    # --- 图3: 3位数 按首位数字分组的宽度箱线图 ---
    three_digit = [r for r in records if r["num_digits"] == 3]
    by_first = defaultdict(list)
    for r in three_digit:
        by_first[r["first_digit"]].append(r["bbox_norm"][2])

    if by_first:
        fig, ax = plt.subplots(figsize=(10, 5))
        first_digits = sorted(by_first.keys())
        data = [by_first[d] for d in first_digits]
        bp = ax.boxplot(data, tick_labels=[f"'{d}xx' (n={len(by_first[d])})" for d in first_digits],
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#AED6F1")
        # 高亮 "1" 开头
        for i, d in enumerate(first_digits):
            if d == "1":
                bp["boxes"][i].set_facecolor("#F5B7B1")
        ax.set_title("3-Digit Bbox Width by First Digit")
        ax.set_ylabel("Normalized Width")
        plt.tight_layout()
        plt.savefig(str(STATS_DIR / "3digit_width_by_first.png"), dpi=150)
        plt.close()

    # --- 图4: 位置热力图 (cx, cy) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cxs = [r["bbox_norm"][0] for r in records]
    cys = [r["bbox_norm"][1] for r in records]
    ax.scatter(cxs, cys, alpha=0.3, s=5, c="red")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y 轴翻转 (图像坐标)
    ax.set_title("Bbox Center Position Heatmap")
    ax.set_xlabel("cx (normalized)")
    ax.set_ylabel("cy (normalized)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(str(STATS_DIR / "position_heatmap.png"), dpi=150)
    plt.close()

    # --- 图5: 每张图 bbox 数量分布 ---
    bbox_per_image = defaultdict(int)
    for r in records:
        bbox_per_image[r["source_image"]] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = list(bbox_per_image.values())
    ax.hist(counts, bins=range(0, max(counts) + 2), color="mediumpurple", edgecolor="black",
            align="left")
    ax.set_title(f"Bboxes per Image (total {len(bbox_per_image)} images)")
    ax.set_xlabel("Number of bboxes")
    ax.set_ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(str(STATS_DIR / "bboxes_per_image.png"), dpi=150)
    plt.close()

    # --- 图6: 数字文本分布 ---
    from collections import Counter
    digit_counts = Counter(r["digit_text"] for r in records)
    top30 = digit_counts.most_common(30)
    fig, ax = plt.subplots(figsize=(12, 5))
    labels_top = [f"{t} ({c})" for t, c in top30]
    ax.barh(range(len(top30)), [c for _, c in top30], color="steelblue")
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(labels_top, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Top 30 Most Common Digit Labels")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(str(STATS_DIR / "digit_distribution.png"), dpi=150)
    plt.close()

    print(f"  图表已保存到 {STATS_DIR}")


# ---------------------------------------------------------------------------
# Fix: 修正 1XX 三位数偏窄的 bbox
# ---------------------------------------------------------------------------

FIXED_DIR = OUTPUT_DIR / "fixed_labels"


def fix_1xx_bbox_width(records, crop_mapping, report_lines):
    """
    基于 1/2 位数的宽度比例关系，修正 3 位数以 "1" 开头的偏窄 bbox。

    原理:
    - 2 位数中，首位 "1" 相对 "2-9" 有一个自然的窄度比 (digit "1" 字体本身窄)
    - 3 位数中，首位 "1" 的窄度应该接近这个比例
    - 如果 3 位数 "1" 的窄度比远大于 2 位数的 → 说明标注系统性偏窄
    - 修正方式: 将 bbox 左边缘向左扩展，使宽度达到预期值
    """
    report_lines.append("\n" + "=" * 70)
    report_lines.append("修正: 1XX 三位数 bbox 宽度校正")
    report_lines.append("=" * 70)

    # --- Step 1: 从 2 位数计算自然 "1" 窄度比 ---
    two_digit = [r for r in records if r["num_digits"] == 2]
    w2_first1 = [r["bbox_norm"][2] for r in two_digit if r["first_digit"] == "1"]
    w2_other = [r["bbox_norm"][2] for r in two_digit if r["first_digit"] != "1"]

    if not w2_first1 or not w2_other:
        report_lines.append("数据不足，无法计算修正因子")
        return []

    mean_w2_1 = np.mean(w2_first1)
    mean_w2_other = np.mean(w2_other)
    natural_ratio = mean_w2_1 / mean_w2_other  # "1" 自然窄度比

    report_lines.append(f"\n2 位数首位 '1' 平均宽度: {mean_w2_1:.6f} (n={len(w2_first1)})")
    report_lines.append(f"2 位数首位 '2-9' 平均宽度: {mean_w2_other:.6f} (n={len(w2_other)})")
    report_lines.append(f"自然窄度比 (1/other): {natural_ratio:.4f} (即 '1' 开头自然窄 {(1 - natural_ratio) * 100:.1f}%)")

    # --- Step 2: 计算 3 位数的预期宽度 ---
    three_digit = [r for r in records if r["num_digits"] == 3]
    w3_first1 = [r for r in three_digit if r["first_digit"] == "1"]
    w3_other = [r for r in three_digit if r["first_digit"] != "1"]

    if not w3_other:
        # 如果没有非 "1" 的 3 位数参考，用 2 位数推算
        mean_w2_all = np.mean([r["bbox_norm"][2] for r in two_digit])
        mean_w1 = np.mean([r["bbox_norm"][2] for r in records if r["num_digits"] == 1]) if \
            any(r["num_digits"] == 1 for r in records) else mean_w2_all / 2
        per_digit_increment = mean_w2_all - mean_w1
        expected_w3_non1 = mean_w2_all + per_digit_increment
    else:
        expected_w3_non1 = np.mean([r["bbox_norm"][2] for r in w3_other])

    expected_w3_1 = expected_w3_non1 * natural_ratio
    actual_w3_1 = np.mean([r["bbox_norm"][2] for r in w3_first1]) if w3_first1 else 0

    report_lines.append(f"\n3 位数首位 '2-9' 平均宽度: {expected_w3_non1:.6f} (n={len(w3_other)})")
    report_lines.append(f"3 位数首位 '1' 预期宽度: {expected_w3_non1:.6f} × {natural_ratio:.4f} = {expected_w3_1:.6f}")
    report_lines.append(f"3 位数首位 '1' 实际宽度: {actual_w3_1:.6f} (n={len(w3_first1)})")
    report_lines.append(f"平均差异: {expected_w3_1 - actual_w3_1:.6f} ({(expected_w3_1 - actual_w3_1) / actual_w3_1 * 100:.1f}%)")

    # --- Step 3: 逐个修正 ---
    # 对每个 1XX bbox，如果宽度 < 预期值，向左扩展
    # 使用个体修正: 目标宽度 = max(当前宽度, 预期宽度)
    # 扩展全部加到左侧 (cx 左移 delta/2, w 增加 delta)
    fixed_records = []
    for r in w3_first1:
        cx, cy, w, h = r["bbox_norm"]
        if w >= expected_w3_1:
            continue  # 已经够宽

        delta = expected_w3_1 - w
        # 向左扩展: cx 左移 delta/2, w 增加 delta
        new_cx = cx - delta / 2
        new_w = w + delta

        # 确保不越界 (归一化坐标 0~1)
        left_edge = new_cx - new_w / 2
        if left_edge < 0:
            # 裁剪到图像边界
            new_cx = new_w / 2
            new_w = min(new_w, 1.0)

        r["fixed_bbox"] = (new_cx, cy, new_w, h)
        r["delta_w"] = delta
        fixed_records.append(r)

    report_lines.append(f"\n需修正的 bbox 数量: {len(fixed_records)} / {len(w3_first1)} (1XX 总数)")
    report_lines.append(f"\n修正详情 (前 30):")
    for r in fixed_records[:30]:
        old_cx, _, old_w, _ = r["bbox_norm"]
        new_cx, _, new_w, _ = r["fixed_bbox"]
        report_lines.append(
            f"  {r['crop_name']} '{r['digit_text']}': "
            f"w {old_w:.6f} -> {new_w:.6f} (+{r['delta_w']:.6f}), "
            f"cx {old_cx:.6f} -> {new_cx:.6f}"
        )
    if len(fixed_records) > 30:
        report_lines.append(f"  ... 还有 {len(fixed_records) - 30} 条")

    # --- Step 4: 写修正后的 label 文件 ---
    if not fixed_records:
        report_lines.append("\n无需修正")
        return fixed_records

    _write_fixed_labels(fixed_records, crop_mapping, report_lines)

    return fixed_records


def _write_fixed_labels(fixed_records, crop_mapping, report_lines):
    """将修正后的 bbox 写入新的 label 文件目录"""
    FIXED_DIR.mkdir(parents=True, exist_ok=True)

    # 按 label_file 分组
    by_label_file = defaultdict(list)
    for r in fixed_records:
        by_label_file[r["label_file"]].append(r)

    files_written = 0
    for label_file, recs in by_label_file.items():
        label_path = Path(label_file)
        if not label_path.exists():
            continue

        # 读原始 label
        with open(label_path, "r") as f:
            lines = f.readlines()

        # 建立 line_index -> fixed_bbox 的映射
        fix_map = {}  # line_index -> fixed_bbox
        for r in recs:
            info = crop_mapping.get(r["crop_name"])
            if info:
                fix_map[info["line_index"]] = r["fixed_bbox"]

        # 替换对应行
        new_lines = []
        for i, line in enumerate(lines):
            if i in fix_map:
                parts = line.strip().split()
                cls_id = parts[0]
                new_cx, new_cy, new_w, new_h = fix_map[i]
                new_line = f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        # 保留原始目录结构 (split/filename)
        # label_file 格式: .../labels/train/frame_xxx.txt
        rel_parts = label_path.parts
        # 找 "labels" 之后的部分
        try:
            labels_idx = rel_parts.index("labels")
            rel_path = Path(*rel_parts[labels_idx:])
        except ValueError:
            rel_path = Path(label_path.name)

        out_path = FIXED_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.writelines(new_lines)
        files_written += 1

    report_lines.append(f"\n修正后的 label 文件已写入: {FIXED_DIR}")
    report_lines.append(f"共修改 {files_written} 个文件")
    print(f"  修正后的 label 文件: {files_written} 个 -> {FIXED_DIR}")


# ---------------------------------------------------------------------------
# 保存 crop 到诊断目录
# ---------------------------------------------------------------------------

def save_diagnosis_crops(narrow_records, edge_records, small_records):
    """将可疑 crop 复制到诊断输出目录"""
    crop_images_dir = BASE / "crnn_data" / "ocr_prelabel" / "images"
    if not crop_images_dir.exists():
        print(f"[WARN] crop 图像目录不存在: {crop_images_dir}")
        return

    for out_dir, record_list, max_save in [
        (NARROW_DIR, narrow_records, 100),
        (EDGE_DIR, edge_records, 100),
        (SMALL_DIR, small_records, 100),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for r in record_list[:max_save]:
            src = crop_images_dir / r["crop_name"]
            if not src.exists():
                continue
            # 用数字文本和原名命名，方便审查
            stem = src.stem
            dst_name = f"{r['digit_text']}_{stem}.png"
            dst = out_dir / dst_name
            img = cv2.imread(str(src))
            if img is not None:
                cv2.imwrite(str(dst), img)
                saved += 1
        print(f"  {out_dir.name}: 保存 {saved} 张 crop")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("YOLO 标注数据问题排查")
    print("=" * 60)

    # 加载数据
    print("\n[1/6] 加载数据...")
    crnn_labels = load_crnn_labels()
    print(f"  CRNN labels: {len(crnn_labels)} 条")

    crop_mapping = load_crop_mapping()
    print(f"  Crop mapping: {len(crop_mapping)} 条")

    records = build_bbox_records(crnn_labels, crop_mapping)
    print(f"  合并后有效记录: {len(records)} 条")

    if not records:
        print("ERROR: 没有有效记录，请检查数据路径")
        sys.exit(1)

    # 基础统计
    from collections import Counter
    digit_dist = Counter(r["num_digits"] for r in records)
    print(f"  位数分布: {dict(sorted(digit_dist.items()))}")

    report_lines = []
    report_lines.append("YOLO 标注数据问题排查报告")
    report_lines.append(f"总记录数: {len(records)}")
    report_lines.append(f"位数分布: {dict(sorted(digit_dist.items()))}")

    # Check 1
    print("\n[2/6] 检查 bbox 宽度...")
    narrow_records = check_bbox_width(records, report_lines)

    # Check 2
    print("\n[3/6] 检查左侧留白...")
    edge_records = check_left_margin(records, report_lines)

    # Check 3
    print("\n[4/6] 检查假阳性...")
    small_records = check_small_bbox(records, report_lines)

    # Check 4: 综合统计
    report_lines.append("\n" + "=" * 70)
    report_lines.append("检查 4: 综合统计概览")
    report_lines.append("=" * 70)

    # 数字文本分布
    text_dist = Counter(r["digit_text"] for r in records)
    report_lines.append(f"\n数字文本分布 (前 30):")
    for text, count in text_dist.most_common(30):
        report_lines.append(f"  '{text}': {count}")

    # 重复数字统计 (99, 88 等)
    repeat_digits = {t: c for t, c in text_dist.items() if len(t) >= 2 and len(set(t)) == 1}
    report_lines.append(f"\n重复数字样本 (如 99, 88, 111):")
    for text, count in sorted(repeat_digits.items()):
        report_lines.append(f"  '{text}': {count}")

    # Fix: 修正 1XX bbox
    print("\n[5/6] 修正 1XX 偏窄 bbox...")
    fixed_records = fix_1xx_bbox_width(records, crop_mapping, report_lines)

    # 图表
    print("\n[6/6] 生成图表...")
    generate_stats_plots(records)

    # 保存 crop
    print("\n保存诊断 crop 图像...")
    save_diagnosis_crops(narrow_records, edge_records, small_records)

    # 写入报告
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n报告已保存: {report_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("排查摘要")
    print("=" * 60)
    print(f"  总 bbox 记录: {len(records)}")
    print(f"  偏窄的 3 位数 bbox: {len(narrow_records)}")
    print(f"  左边缘留白不足: {len(edge_records)}")
    print(f"  异常小/极端宽高比: {len(small_records)}")
    print(f"  已修正 1XX bbox: {len(fixed_records)}")
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"  report.txt          - 完整文字报告")
    print(f"  narrow_bbox_crops/  - 偏窄 bbox crop")
    print(f"  edge_touching_crops/- 左边缘不足 crop")
    print(f"  small_bbox_crops/   - 异常小 crop")
    print(f"  stats/              - 统计图表")


if __name__ == "__main__":
    main()
