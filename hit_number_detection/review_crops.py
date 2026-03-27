"""
Crop 审查工具 — 快速找出错误标注

只做一件事：过一遍所有 crop，标记哪些是数字、哪些不是。
最后输出报告：哪些 crop 是错的，来自哪张原图。

操作：
  Enter     → 正确（是数字），下一张
  d + Enter → 错误（不是数字）
  b + Enter → 回退上一张
  q + Enter → 保存退出，输出报告

用法：
  python review_crops.py --data crnn_data/train
"""

import argparse
import json
import os
import cv2
import numpy as np
from collections import defaultdict


def get_source_name(mapping, crop_name):
    if crop_name in mapping:
        label_file = mapping[crop_name]["label_file"]
        stem = os.path.splitext(os.path.basename(label_file))[0]
        return f"{stem}.png"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Crop 审查工具")
    parser.add_argument("--data", required=True, help="CRNN 数据目录")
    parser.add_argument("--scale", type=int, default=8, help="图片放大倍数 (默认 8)")
    args = parser.parse_args()

    images_dir = os.path.join(args.data, "images")
    mapping_path = os.path.join(args.data, "crop_mapping.json")
    review_path = os.path.join(args.data, "review_result.json")

    # 加载 mapping
    if not os.path.exists(mapping_path):
        print("错误: crop_mapping.json 不存在，请先运行 prepare_crnn_data.py")
        return
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # 加载已有审查进度
    reviewed = {}
    if os.path.exists(review_path):
        with open(review_path, "r", encoding="utf-8") as f:
            reviewed = json.load(f)

    # 获取所有 crop 文件
    all_crops = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    total = len(all_crops)

    # 找到第一个未审查的位置
    start_idx = 0
    for i, fname in enumerate(all_crops):
        if fname not in reviewed:
            start_idx = i
            break
    else:
        print(f"所有 {total} 张已审查完毕！")
        print_report(all_crops, reviewed, mapping)
        return

    ok_count = sum(1 for v in reviewed.values() if v == "ok")
    bad_count = sum(1 for v in reviewed.values() if v == "bad")
    print(f"共 {total} 张 crop，已审查 {len(reviewed)} 张 (正确: {ok_count}, 错误: {bad_count})")
    print(f"从第 {start_idx + 1} 张开始")
    print()
    print("操作：Enter=正确  d=错误  b=回退  q=退出并输出报告")
    print()

    idx = start_idx

    while idx < total:
        fname = all_crops[idx]
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            idx += 1
            continue

        source = get_source_name(mapping, fname)
        status = reviewed.get(fname, "?")

        # 放大显示
        h, w = img.shape[:2]
        display = cv2.resize(img, (w * args.scale, h * args.scale),
                             interpolation=cv2.INTER_NEAREST)

        # 状态栏
        bar_h = 50
        min_width = max(display.shape[1], 550)
        if display.shape[1] < min_width:
            pad = np.zeros((display.shape[0], min_width - display.shape[1], 3), dtype=np.uint8)
            display = np.hstack([display, pad])

        bar = 200 * np.ones((bar_h, display.shape[1], 3), dtype=np.uint8)
        bad_so_far = sum(1 for v in reviewed.values() if v == "bad")
        line1 = f"[{idx + 1}/{total}] Bad: {bad_so_far} | Enter=OK  d=BAD  b=back  q=quit"
        line2 = f"Source: {source}"
        cv2.putText(bar, line1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(bar, line2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 0), 1)
        display = np.vstack([display, bar])

        cv2.imshow("Review", display)
        cv2.waitKey(1)

        user_input = input(f"  [{idx + 1}/{total}] {fname} ← {source}: ").strip().lower()

        if user_input == "q":
            save_and_report(review_path, all_crops, reviewed, mapping)
            cv2.destroyAllWindows()
            return

        elif user_input == "d":
            reviewed[fname] = "bad"
            print(f"    ✗ 错误 (来自 {source})")
            idx += 1

        elif user_input == "b":
            if idx > 0:
                idx -= 1
                print(f"    ← 回退到 {all_crops[idx]}")
            else:
                print("    已经是第一张了")

        elif user_input == "":
            reviewed[fname] = "ok"
            idx += 1

        else:
            print(f"    无效输入，Enter=正确 d=错误 b=回退 q=退出")

        # 每 100 张自动保存
        if idx % 100 == 0 and idx > 0:
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(reviewed, f, indent=2)
            bad_so_far = sum(1 for v in reviewed.values() if v == "bad")
            print(f"    [自动保存] 已审查 {len(reviewed)}/{total}，错误 {bad_so_far}")

    save_and_report(review_path, all_crops, reviewed, mapping)
    cv2.destroyAllWindows()


def save_and_report(review_path, all_crops, reviewed, mapping):
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(reviewed, f, indent=2)
    print_report(all_crops, reviewed, mapping)


def print_report(all_crops, reviewed, mapping):
    bad_crops = [f for f in all_crops if reviewed.get(f) == "bad"]
    ok_count = sum(1 for v in reviewed.values() if v == "ok")

    print(f"\n{'=' * 60}")
    print(f"审查结果")
    print(f"{'=' * 60}")
    print(f"  总数: {len(all_crops)}")
    print(f"  正确: {ok_count}")
    print(f"  错误: {len(bad_crops)}")
    print(f"  未审查: {len(all_crops) - len(reviewed)}")

    if not bad_crops:
        print("\n没有错误标注！")
        return

    # 按原图分组
    by_source = defaultdict(list)
    for crop_name in bad_crops:
        source = "unknown"
        bbox_line = "?"
        line_idx = "?"
        if crop_name in mapping:
            label_file = mapping[crop_name]["label_file"]
            source = os.path.splitext(os.path.basename(label_file))[0] + ".png"
            bbox_line = mapping[crop_name]["bbox_line"]
            line_idx = mapping[crop_name]["line_index"]
        by_source[source].append((crop_name, line_idx, bbox_line))

    print(f"\n错误标注来源（按原图分组）：")
    print(f"{'-' * 60}")
    for source, crops in sorted(by_source.items()):
        print(f"\n  {source}  ({len(crops)} 个错误 bbox)")
        for crop_name, line_idx, bbox_line in crops:
            print(f"    {crop_name} | 行 {line_idx} | {bbox_line}")

    print(f"\n{'=' * 60}")
    print(f"请在 CVAT 中搜索上述原图名，删除或修正对应的 bbox。")
    print(f"审查结果已保存到 review_result.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
