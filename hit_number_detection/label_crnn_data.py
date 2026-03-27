"""
CRNN 数字标注辅助工具

弹窗显示裁剪的数字图片，用户输入实际数字值。
支持：
  - 输入数字后 Enter 确认，自动跳到下一张
  - 输入 's' 跳过不确定的图片
  - 输入 'q' 保存并退出（下次从上次位置继续）
  - 输入 'b' 回退上一张修改
  - 输入 'd' 删除（标注错误/非数字），记录原图来源
  - 自动保存进度，中断后可继续
  - 退出时输出删除报告（按原图分组）

用法：
  python label_crnn_data.py --data crnn_data/train
  python label_crnn_data.py --data crnn_data/val
"""

import argparse
import json
import os
import cv2
import numpy as np


def load_labels(labels_path):
    """读取 labels.txt，返回 [(filename, label)] 列表。"""
    entries = []
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    entries.append([parts[0], parts[1]])
    return entries


def save_labels(labels_path, entries):
    """保存 labels.txt，自动过滤已删除的条目。"""
    with open(labels_path, "w", encoding="utf-8") as f:
        for fname, label in entries:
            if label != "DELETE":
                f.write(f"{fname}\t{label}\n")


def load_mapping(data_dir):
    """读取 crop_mapping.json。"""
    mapping_path = os.path.join(data_dir, "crop_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_source_name(mapping, crop_name):
    """从 mapping 中获取原图文件名。"""
    if crop_name in mapping:
        label_file = mapping[crop_name]["label_file"]
        # 从标签路径提取图片名: .../labels/train/xxx.txt → xxx.png
        stem = os.path.splitext(os.path.basename(label_file))[0]
        return f"{stem}.png"
    return "unknown"


def print_delete_report(entries, mapping):
    """输出被删除 crop 的原图来源报告。"""
    deleted = [(f, l) for f, l in entries if l == "DELETE"]
    if not deleted:
        return

    print(f"\n{'=' * 60}")
    print(f"删除报告: 共 {len(deleted)} 个错误标注")
    print(f"{'=' * 60}")

    # 按原图分组
    from collections import defaultdict
    by_source = defaultdict(list)
    for crop_name, _ in deleted:
        source = get_source_name(mapping, crop_name)
        bbox_line = mapping.get(crop_name, {}).get("bbox_line", "?")
        by_source[source].append((crop_name, bbox_line))

    for source, crops in sorted(by_source.items()):
        print(f"\n  原图: {source}  ({len(crops)} 个错误)")
        for crop_name, bbox_line in crops:
            print(f"    {crop_name} → bbox: {bbox_line}")

    print(f"\n请在 CVAT 中找到上述原图，删除或修正对应的 bbox。")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="CRNN 数字标注辅助工具")
    parser.add_argument("--data", required=True, help="CRNN 数据目录 (含 images/ 和 labels.txt)")
    parser.add_argument("--scale", type=int, default=6, help="图片放大倍数 (默认 6)")
    args = parser.parse_args()

    images_dir = os.path.join(args.data, "images")
    labels_path = os.path.join(args.data, "labels.txt")

    entries = load_labels(labels_path)
    if not entries:
        print("错误: labels.txt 为空或不存在")
        return

    mapping = load_mapping(args.data)
    has_mapping = len(mapping) > 0
    if has_mapping:
        print(f"已加载 crop_mapping.json ({len(mapping)} 条映射)")
    else:
        print("警告: 未找到 crop_mapping.json，无法显示原图来源")

    total = len(entries)

    # 找到第一个未标注的位置
    start_idx = 0
    for i, (fname, label) in enumerate(entries):
        if label == "???":
            start_idx = i
            break
    else:
        print(f"所有 {total} 张图片都已标注完成！")
        print_delete_report(entries, mapping)
        return

    labeled_count = sum(1 for _, l in entries if l not in ("???", "SKIP", "DELETE"))
    print(f"共 {total} 张图片，已标注 {labeled_count} 张")
    print(f"从第 {start_idx + 1} 张开始")
    print()
    print("操作说明:")
    print("  输入数字 + Enter  → 确认标注")
    print("  s + Enter         → 跳过（标记为 SKIP）")
    print("  d + Enter         → 删除此图（非数字/标注错误）")
    print("  b + Enter         → 回退上一张")
    print("  q + Enter         → 保存并退出")
    print()

    idx = start_idx

    while idx < total:
        fname, label = entries[idx]
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  警告: 无法读取 {img_path}，跳过")
            idx += 1
            continue

        # 原图来源
        source = get_source_name(mapping, fname) if has_mapping else ""

        # 放大显示
        h, w = img.shape[:2]
        display = cv2.resize(img, (w * args.scale, h * args.scale),
                             interpolation=cv2.INTER_NEAREST)

        # 添加状态栏
        bar_h = 50
        min_width = max(display.shape[1], 500)
        if display.shape[1] < min_width:
            pad = np.zeros((display.shape[0], min_width - display.shape[1], 3), dtype=np.uint8)
            display = np.hstack([display, pad])

        bar = 200 * np.ones((bar_h, display.shape[1], 3), dtype=np.uint8)
        line1 = f"[{idx + 1}/{total}] '{label}' | number/s/d/b/q"
        line2 = f"Source: {source}" if source else ""
        cv2.putText(bar, line1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(bar, line2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 0), 1)
        display = np.vstack([display, bar])

        cv2.imshow("Label Tool", display)
        cv2.waitKey(1)

        # 终端输入
        source_hint = f" ← {source}" if source else ""
        user_input = input(f"  [{idx + 1}/{total}] {fname}{source_hint} (当前: {label}): ").strip()

        if user_input.lower() == "q":
            save_labels(labels_path, entries)
            labeled_count = sum(1 for _, l in entries if l not in ("???", "SKIP", "DELETE"))
            print(f"\n已保存！标注进度: {labeled_count}/{total}")
            print_delete_report(entries, mapping)
            cv2.destroyAllWindows()
            return

        elif user_input.lower() == "s":
            entries[idx][1] = "SKIP"
            idx += 1

        elif user_input.lower() == "d":
            # 标记删除，保留图片文件（方便查看报告）
            entries[idx][1] = "DELETE"
            print(f"  标记删除: {fname} (来自 {source})")
            idx += 1

        elif user_input.lower() == "b":
            if idx > 0:
                idx -= 1
                print(f"  ← 回退到 {entries[idx][0]}")
            else:
                print("  已经是第一张了")

        elif user_input.isdigit() or user_input == "":
            if user_input:
                entries[idx][1] = user_input
                idx += 1
            else:
                idx += 1

        else:
            print(f"  无效输入: '{user_input}'，请输入数字、s、d、b 或 q")

        # 每 50 张自动保存
        if idx % 50 == 0 and idx > 0:
            save_labels(labels_path, entries)
            labeled_count = sum(1 for _, l in entries if l not in ("???", "SKIP", "DELETE"))
            print(f"  [自动保存] 进度: {labeled_count}/{total}")

    # 全部标完
    save_labels(labels_path, entries)
    labeled_count = sum(1 for _, l in entries if l not in ("???", "SKIP", "DELETE"))
    skipped = sum(1 for _, l in entries if l == "SKIP")
    deleted = sum(1 for _, l in entries if l == "DELETE")
    print(f"\n全部完成！")
    print(f"  已标注: {labeled_count}")
    print(f"  跳过: {skipped}")
    print(f"  删除: {deleted}")
    print(f"  标签文件: {labels_path}")
    print_delete_report(entries, mapping)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
