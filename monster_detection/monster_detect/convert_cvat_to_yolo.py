"""将 CVAT XML 标注转换为 YOLO detection txt 格式 (仅 body/head bbox).

YOLO 格式: class_id cx cy w h (归一化坐标)

用法:
  python -m monster_detect.convert_cvat_to_yolo
  python -m monster_detect.convert_cvat_to_yolo --xml path1.xml path2.xml
"""

import argparse
import os
import xml.etree.ElementTree as ET

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_MODULE_DIR, "monster_yolo_data")
_OUTPUT_DIR = os.path.join(_DATA_DIR, "images", "unlabeled")

CLASSES = ["body", "head"]


def convert(xml_paths: list[str]):
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    print(f"类别映射: {dict(enumerate(CLASSES))}")

    total_converted = 0
    total_boxes = 0

    for xml_path in xml_paths:
        if not os.path.exists(xml_path):
            print(f"警告: 跳过不存在的文件 {xml_path}")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        images = root.findall("image")
        print(f"\n处理 {xml_path}")
        print(f"  找到 {len(images)} 张图片")

        converted = 0
        boxes = 0

        for img_elem in images:
            img_name = img_elem.get("name")
            img_w = int(img_elem.get("width"))
            img_h = int(img_elem.get("height"))

            lines = []

            for box in img_elem.findall("box"):
                label = box.get("label")
                if label not in CLASSES:
                    continue

                cls_id = CLASSES.index(label)
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))

                cx = (xtl + xbr) / 2.0 / img_w
                cy = (ytl + ybr) / 2.0 / img_h
                bw = (xbr - xtl) / img_w
                bh = (ybr - ytl) / img_h

                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            txt_name = img_name.replace(".jpg", ".txt")
            txt_path = os.path.join(_OUTPUT_DIR, txt_name)
            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")

            if lines:
                converted += 1
                boxes += len(lines)

        print(f"  转换 {converted} 张图片, {boxes} 个 bbox")
        total_converted += converted
        total_boxes += boxes

    print(f"\n总计: {total_converted} 张图片, {total_boxes} 个 bbox")
    print(f"输出目录: {_OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="CVAT XML → YOLO detection txt")
    parser.add_argument("--xml", nargs="+", default=None,
                        help="CVAT XML 文件路径 (可多个)")
    args = parser.parse_args()

    if args.xml:
        xml_paths = args.xml
    else:
        # 默认: 扫描所有 cvat_batch*_labels/annotations.xml
        xml_paths = []
        for d in sorted(os.listdir(_DATA_DIR)):
            ann = os.path.join(_DATA_DIR, d, "annotations.xml")
            if d.startswith("cvat_batch") and d.endswith("_labels") and os.path.isfile(ann):
                xml_paths.append(ann)
        if not xml_paths:
            # fallback to cvat_export
            fallback = os.path.join(_DATA_DIR, "cvat_export", "annotations.xml")
            if os.path.exists(fallback):
                xml_paths.append(fallback)

    if not xml_paths:
        print("错误: 未找到 CVAT XML 文件")
        return

    print(f"将处理 {len(xml_paths)} 个 XML 文件")
    convert(xml_paths)


if __name__ == "__main__":
    main()
