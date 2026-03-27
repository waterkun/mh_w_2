"""将 PascalVOC XML 标注转换为 YOLO txt 格式.

用法:
  python -m monster_detect.convert_voc_to_yolo
"""

import os
import xml.etree.ElementTree as ET

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_XML_DIR = os.path.join(_MODULE_DIR, "monster_yolo_data", "labels")
_IMG_DIR = os.path.join(_MODULE_DIR, "monster_yolo_data", "images", "unlabeled")

CLASSES = ["body", "chain", "head", "arm"]


def convert():
    xml_files = [f for f in os.listdir(_XML_DIR) if f.endswith(".xml")]
    if not xml_files:
        print("没有找到 XML 标注文件")
        return

    print(f"找到 {len(xml_files)} 个 XML 标注")
    print(f"类别映射: {dict(enumerate(CLASSES))}")
    print()

    converted = 0
    for xml_name in sorted(xml_files):
        xml_path = os.path.join(_XML_DIR, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_w = int(root.find("size/width").text)
        img_h = int(root.find("size/height").text)

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in CLASSES:
                print(f"  警告: 跳过未知类 '{cls_name}' in {xml_name}")
                continue

            cls_id = CLASSES.index(cls_name)
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            # 转为 YOLO 归一化 cx, cy, w, h
            cx = (xmin + xmax) / 2.0 / img_w
            cy = (ymin + ymax) / 2.0 / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # 保存到 unlabeled/ 目录 (和图片在一起)
        txt_name = xml_name.replace(".xml", ".txt")
        txt_path = os.path.join(_IMG_DIR, txt_name)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        converted += 1

    print(f"完成! 转换 {converted} 个文件到 {_IMG_DIR}")


if __name__ == "__main__":
    convert()
