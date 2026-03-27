"""
根据 review_result.json 中标记为 bad 的 crop，自动清理 YOLO 标签中的错误 bbox。

用法：
  python fix_yolo_labels.py --data crnn_data/train --dry-run  # 预览
  python fix_yolo_labels.py --data crnn_data/train            # 执行
"""

import argparse
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="根据审查结果清理 YOLO 错误标签")
    parser.add_argument("--data", required=True, help="CRNN 数据目录")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不实际修改")
    args = parser.parse_args()

    mapping_path = os.path.join(args.data, "crop_mapping.json")
    review_path = os.path.join(args.data, "review_result.json")

    if not os.path.exists(mapping_path):
        print("错误: crop_mapping.json 不存在")
        return
    if not os.path.exists(review_path):
        print("错误: review_result.json 不存在，请先运行 review_crops.py")
        return

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    with open(review_path, "r", encoding="utf-8") as f:
        reviewed = json.load(f)

    bad_crops = [f for f, v in reviewed.items() if v == "bad"]
    if not bad_crops:
        print("没有错误标注，无需修改。")
        return

    # 按 YOLO 标签文件分组
    fixes = defaultdict(list)
    for crop_name in bad_crops:
        if crop_name not in mapping:
            continue
        info = mapping[crop_name]
        fixes[info["label_file"]].append({
            "line_index": info["line_index"],
            "bbox_line": info["bbox_line"],
            "crop_name": crop_name,
        })

    print(f"找到 {len(bad_crops)} 个错误标注，涉及 {len(fixes)} 个标签文件：")
    print()

    total_removed = 0
    for label_file, items in sorted(fixes.items()):
        items.sort(key=lambda x: x["line_index"], reverse=True)
        source_name = os.path.splitext(os.path.basename(label_file))[0] + ".png"

        print(f"  {source_name} → 删除 {len(items)} 行")
        for item in items:
            print(f"    行 {item['line_index']}: {item['bbox_line']}")

        if not args.dry_run:
            with open(label_file, "r") as f:
                lines = f.readlines()

            indices_to_remove = set(item["line_index"] for item in items)
            new_lines = [line for i, line in enumerate(lines) if i not in indices_to_remove]

            with open(label_file, "w") as f:
                f.writelines(new_lines)

            # 如果文件空了，删除标签文件和对应图片
            if not new_lines or all(l.strip() == "" for l in new_lines):
                os.remove(label_file)
                # 删除对应图片
                img_file = label_file.replace("/labels/", "/images/").replace("\\labels\\", "\\images\\").replace(".txt", ".png")
                if os.path.exists(img_file):
                    os.remove(img_file)
                    print(f"    → 标签和图片都已删除（无剩余 bbox）")

        total_removed += len(items)

    print(f"\n共 {'将' if args.dry_run else '已'}移除 {total_removed} 个错误 bbox")
    if args.dry_run:
        print("（dry-run 模式，未修改。去掉 --dry-run 执行修改）")
    else:
        print("YOLO 标签已清理！可重新训练。")


if __name__ == "__main__":
    main()
