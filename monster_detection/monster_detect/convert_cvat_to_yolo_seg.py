"""将 CVAT XML 标注转换为 YOLO segmentation txt 格式.

CVAT XML 中:
- box (body/head/arm) → 转为 4 点 polygon
- polyline (chain/tail) → 膨胀为有宽度的闭合 polygon

YOLO seg 格式: class_id x1 y1 x2 y2 ... xn yn (归一化坐标)

用法:
  python -m monster_detect.convert_cvat_to_yolo_seg
  python -m monster_detect.convert_cvat_to_yolo_seg --width 60
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_CVAT_XML = os.path.join(_MODULE_DIR, "monster_yolo_data", "cvat_export", "annotations.xml")
_OUTPUT_DIR = os.path.join(_MODULE_DIR, "monster_yolo_data", "images", "unlabeled")

CLASSES = ["body", "chain", "head", "arm", "tail"]


def dilate_polyline(points_px, width):
    """将 polyline 点序列膨胀为闭合 polygon.

    沿每个线段的法线方向向两侧各偏移 width/2，
    正向走一遍 + 反向走一遍形成闭合区域。

    Args:
        points_px: [(x0,y0), (x1,y1), ...] 像素坐标
        width: 膨胀宽度 (像素)

    Returns:
        [(x,y), ...] 闭合 polygon 的点序列
    """
    if len(points_px) < 2:
        return []

    half_w = width / 2.0
    left_side = []
    right_side = []

    for i in range(len(points_px)):
        # 计算该点处的法线方向
        if i == 0:
            dx = points_px[1][0] - points_px[0][0]
            dy = points_px[1][1] - points_px[0][1]
        elif i == len(points_px) - 1:
            dx = points_px[i][0] - points_px[i - 1][0]
            dy = points_px[i][1] - points_px[i - 1][1]
        else:
            # 中间点: 取前后线段方向的平均
            dx1 = points_px[i][0] - points_px[i - 1][0]
            dy1 = points_px[i][1] - points_px[i - 1][1]
            dx2 = points_px[i + 1][0] - points_px[i][0]
            dy2 = points_px[i + 1][1] - points_px[i][1]
            dx = dx1 + dx2
            dy = dy1 + dy2

        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            nx, ny = 0, 0
        else:
            # 法线 = 旋转 90 度
            nx = -dy / length * half_w
            ny = dx / length * half_w

        px, py = points_px[i]
        left_side.append((px + nx, py + ny))
        right_side.append((px - nx, py - ny))

    # 正向左侧 + 反向右侧 = 闭合 polygon
    return left_side + right_side[::-1]


def convert(polyline_width=50):
    if not os.path.exists(_CVAT_XML):
        print(f"错误: CVAT XML 不存在 {_CVAT_XML}")
        return

    tree = ET.parse(_CVAT_XML)
    root = tree.getroot()

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    images = root.findall("image")
    print(f"找到 {len(images)} 张图片")
    print(f"类别映射: {dict(enumerate(CLASSES))}")
    print(f"Polyline 膨胀宽度: {polyline_width}px")
    print()

    converted = 0
    total_annotations = 0

    for img_elem in images:
        img_name = img_elem.get("name")
        img_w = int(img_elem.get("width"))
        img_h = int(img_elem.get("height"))

        lines = []

        # 处理 bbox → 4 点 polygon
        for box in img_elem.findall("box"):
            label = box.get("label")
            if label not in CLASSES:
                print(f"  警告: 跳过未知类 '{label}' in {img_name}")
                continue

            cls_id = CLASSES.index(label)
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            # bbox 转 4 点 polygon (归一化)
            points = [
                xtl / img_w, ytl / img_h,
                xbr / img_w, ytl / img_h,
                xbr / img_w, ybr / img_h,
                xtl / img_w, ybr / img_h,
            ]
            coords = " ".join(f"{p:.6f}" for p in points)
            lines.append(f"{cls_id} {coords}")

        # 处理 polyline → 膨胀为闭合 polygon
        for polyline in img_elem.findall("polyline"):
            label = polyline.get("label")
            if label not in CLASSES:
                print(f"  警告: 跳过未知类 '{label}' in {img_name}")
                continue

            cls_id = CLASSES.index(label)
            points_str = polyline.get("points")

            # 解析像素坐标
            points_px = []
            for pt in points_str.split(";"):
                x, y = pt.split(",")
                points_px.append((float(x), float(y)))

            if len(points_px) < 2:
                continue

            # 膨胀为闭合 polygon
            polygon = dilate_polyline(points_px, polyline_width)
            if len(polygon) < 3:
                continue

            # 归一化并 clamp 到 [0, 1]
            points = []
            for px, py in polygon:
                points.append(max(0.0, min(1.0, px / img_w)))
                points.append(max(0.0, min(1.0, py / img_h)))

            coords = " ".join(f"{p:.6f}" for p in points)
            lines.append(f"{cls_id} {coords}")

        # 写入 txt
        txt_name = img_name.replace(".jpg", ".txt")
        txt_path = os.path.join(_OUTPUT_DIR, txt_name)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        if lines:
            converted += 1
            total_annotations += len(lines)

    print(f"完成! 转换 {converted} 张图片, 共 {total_annotations} 个标注")
    print(f"输出目录: {_OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVAT XML → YOLO seg txt")
    parser.add_argument("--width", type=int, default=50,
                        help="Polyline 膨胀宽度 (像素, 默认 50)")
    args = parser.parse_args()
    convert(polyline_width=args.width)
