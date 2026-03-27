import xml.etree.ElementTree as ET
import statistics
from collections import defaultdict
import sys

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 解析XML文件
tree = ET.parse(r'F:\AI_Learn\mh_w_2.1\hit_number_detection\yolo-data\1-100\annotations.xml')
root = tree.getroot()

# 图像尺寸
IMG_WIDTH = 1920
IMG_HEIGHT = 804

# 数据收集
total_images = 0
total_boxes = 0
images_with_zero_annotations = []
boxes_per_image = {}
all_box_widths = []
all_box_heights = []
all_aspect_ratios = []
boxes_near_edge = []
box_positions = []
all_boxes_info = []

# 遍历所有图像
for image in root.findall('image'):
    total_images += 1
    image_name = image.get('name')
    image_id = image.get('id')

    # 获取该图像的所有边界框
    boxes = image.findall('box')
    num_boxes = len(boxes)

    boxes_per_image[image_name] = num_boxes
    total_boxes += num_boxes

    if num_boxes == 0:
        images_with_zero_annotations.append(image_name)

    # 分析每个边界框
    for box in boxes:
        xtl = float(box.get('xtl'))
        ytl = float(box.get('ytl'))
        xbr = float(box.get('xbr'))
        ybr = float(box.get('ybr'))

        # 计算宽度和高度
        width = xbr - xtl
        height = ybr - ytl

        all_box_widths.append(width)
        all_box_heights.append(height)

        # 计算宽高比
        aspect_ratio = width / height if height > 0 else 0
        all_aspect_ratios.append(aspect_ratio)

        # 检查是否靠近边缘
        near_edge = False
        edge_info = []
        if xtl <= 5:
            near_edge = True
            edge_info.append("左边缘")
        if ytl <= 5:
            near_edge = True
            edge_info.append("上边缘")
        if xbr >= IMG_WIDTH - 5:
            near_edge = True
            edge_info.append("右边缘")
        if ybr >= IMG_HEIGHT - 5:
            near_edge = True
            edge_info.append("下边缘")

        if near_edge:
            boxes_near_edge.append({
                'image': image_name,
                'box': f"({xtl:.2f}, {ytl:.2f}, {xbr:.2f}, {ybr:.2f})",
                'edges': ', '.join(edge_info)
            })

        # 记录中心位置
        center_x = (xtl + xbr) / 2
        center_y = (ytl + ybr) / 2
        box_positions.append((center_x, center_y))

        # 保存完整信息
        all_boxes_info.append({
            'image': image_name,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'box': f"({xtl:.2f}, {ytl:.2f}, {xbr:.2f}, {ybr:.2f})"
        })

# 计算统计数据
avg_boxes_per_image = total_boxes / total_images if total_images > 0 else 0
min_boxes = min(boxes_per_image.values())
max_boxes = max(boxes_per_image.values())

# 找到最小和最大框数的图像
images_with_min_boxes = [name for name, count in boxes_per_image.items() if count == min_boxes]
images_with_max_boxes = [name for name, count in boxes_per_image.items() if count == max_boxes]

# 边界框尺寸统计
avg_width = statistics.mean(all_box_widths) if all_box_widths else 0
avg_height = statistics.mean(all_box_heights) if all_box_heights else 0
min_width = min(all_box_widths) if all_box_widths else 0
max_width = max(all_box_widths) if all_box_widths else 0
min_height = min(all_box_heights) if all_box_heights else 0
max_height = max(all_box_heights) if all_box_heights else 0

# 宽高比统计
avg_aspect_ratio = statistics.mean(all_aspect_ratios) if all_aspect_ratios else 0

# 查找异常大小的框（2倍或0.5倍）
outlier_boxes = []
for box_info in all_boxes_info:
    is_outlier = False
    reasons = []

    if box_info['width'] > avg_width * 2:
        is_outlier = True
        reasons.append(f"宽度过大({box_info['width']:.2f}px, 平均{avg_width:.2f}px的{box_info['width']/avg_width:.2f}倍)")
    elif box_info['width'] < avg_width * 0.5:
        is_outlier = True
        reasons.append(f"宽度过小({box_info['width']:.2f}px, 平均{avg_width:.2f}px的{box_info['width']/avg_width:.2f}倍)")

    if box_info['height'] > avg_height * 2:
        is_outlier = True
        reasons.append(f"高度过大({box_info['height']:.2f}px, 平均{avg_height:.2f}px的{box_info['height']/avg_height:.2f}倍)")
    elif box_info['height'] < avg_height * 0.5:
        is_outlier = True
        reasons.append(f"高度过小({box_info['height']:.2f}px, 平均{avg_height:.2f}px的{box_info['height']/avg_height:.2f}倍)")

    if is_outlier:
        outlier_boxes.append({
            'image': box_info['image'],
            'box': box_info['box'],
            'width': box_info['width'],
            'height': box_info['height'],
            'reasons': reasons
        })

# 查找异常宽高比的框
unusual_aspect_ratio_boxes = []
for box_info in all_boxes_info:
    # 宽高比过大（太宽）或过小（太窄）
    if box_info['aspect_ratio'] > 2.5:  # 宽度是高度的2.5倍以上
        unusual_aspect_ratio_boxes.append({
            'image': box_info['image'],
            'box': box_info['box'],
            'aspect_ratio': box_info['aspect_ratio'],
            'reason': '过宽'
        })
    elif box_info['aspect_ratio'] < 0.4:  # 宽度小于高度的0.4倍
        unusual_aspect_ratio_boxes.append({
            'image': box_info['image'],
            'box': box_info['box'],
            'aspect_ratio': box_info['aspect_ratio'],
            'reason': '过窄'
        })

# 位置分布分析 - 将图像分成9个区域
def get_region(x, y):
    """获取框中心所在的区域（3x3网格）"""
    col = 0 if x < IMG_WIDTH/3 else (1 if x < IMG_WIDTH*2/3 else 2)
    row = 0 if y < IMG_HEIGHT/3 else (1 if y < IMG_HEIGHT*2/3 else 2)
    return (row, col)

region_counts = defaultdict(int)
for x, y in box_positions:
    region = get_region(x, y)
    region_counts[region] += 1

region_names = {
    (0, 0): '左上', (0, 1): '上中', (0, 2): '右上',
    (1, 0): '左中', (1, 1): '中心', (1, 2): '右中',
    (2, 0): '左下', (2, 1): '下中', (2, 2): '右下'
}

# 输出结果
print("=" * 80)
print("CVAT标注质量分析报告")
print("=" * 80)

print("\n【1. 基本统计】")
print(f"总图像数: {total_images}")
print(f"总边界框数: {total_boxes}")

print("\n【2. 无标注图像】")
print(f"无标注图像数量: {len(images_with_zero_annotations)}")
if images_with_zero_annotations:
    print("无标注图像列表:")
    for img in images_with_zero_annotations:
        print(f"  - {img}")
else:
    print("  所有图像都有标注")

print("\n【3. 每张图像的边界框统计】")
print(f"平均每张图像的框数: {avg_boxes_per_image:.2f}")
print(f"最少框数: {min_boxes} 框")
print(f"最少框数的图像 ({len(images_with_min_boxes)}张):")
for img in images_with_min_boxes[:10]:  # 只显示前10个
    print(f"  - {img}")
if len(images_with_min_boxes) > 10:
    print(f"  ... 还有 {len(images_with_min_boxes) - 10} 张图像")

print(f"\n最多框数: {max_boxes} 框")
print(f"最多框数的图像:")
for img in images_with_max_boxes:
    print(f"  - {img}")

print("\n【4. 边界框尺寸分析】")
print(f"平均宽度: {avg_width:.2f} px")
print(f"平均高度: {avg_height:.2f} px")
print(f"最小宽度: {min_width:.2f} px")
print(f"最大宽度: {max_width:.2f} px")
print(f"最小高度: {min_height:.2f} px")
print(f"最大高度: {max_height:.2f} px")

print("\n【5. 异常尺寸边界框】")
print(f"异常尺寸边界框数量: {len(outlier_boxes)}")
if outlier_boxes:
    print("详细列表:")
    for box in outlier_boxes:
        print(f"\n  图像: {box['image']}")
        print(f"  坐标: {box['box']}")
        print(f"  尺寸: {box['width']:.2f} x {box['height']:.2f} px")
        print(f"  异常原因: {'; '.join(box['reasons'])}")
else:
    print("  未发现异常尺寸的边界框")

print("\n【6. 宽高比分析】")
print(f"平均宽高比: {avg_aspect_ratio:.2f}")
print(f"异常宽高比边界框数量: {len(unusual_aspect_ratio_boxes)}")
if unusual_aspect_ratio_boxes:
    print("详细列表:")
    for box in unusual_aspect_ratio_boxes:
        print(f"\n  图像: {box['image']}")
        print(f"  坐标: {box['box']}")
        print(f"  宽高比: {box['aspect_ratio']:.2f}")
        print(f"  原因: {box['reason']}")
else:
    print("  未发现异常宽高比的边界框")

print("\n【7. 位置分布分析】")
print("边界框中心位置分布（3x3网格）:")
print("\n  区域分布:")
for row in range(3):
    for col in range(3):
        region = (row, col)
        count = region_counts.get(region, 0)
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"  {region_names[region]:4s}: {count:3d} 框 ({percentage:5.1f}%)")

# 判断分布是否集中
max_region_count = max(region_counts.values()) if region_counts else 0
max_region_percentage = (max_region_count / total_boxes * 100) if total_boxes > 0 else 0
if max_region_percentage > 50:
    max_regions = [region_names[k] for k, v in region_counts.items() if v == max_region_count]
    print(f"\n  [警告] 边界框分布较为集中，{max_region_percentage:.1f}% 集中在 {', '.join(max_regions)} 区域")
else:
    print(f"\n  [正常] 边界框分布较为均匀")

print("\n【8. 边缘位置边界框】")
print(f"靠近边缘（5px内）的边界框数量: {len(boxes_near_edge)}")
if boxes_near_edge:
    print("详细列表:")
    for box in boxes_near_edge[:20]:  # 只显示前20个
        print(f"\n  图像: {box['image']}")
        print(f"  坐标: {box['box']}")
        print(f"  靠近: {box['edges']}")
    if len(boxes_near_edge) > 20:
        print(f"\n  ... 还有 {len(boxes_near_edge) - 20} 个靠近边缘的边界框")
else:
    print("  未发现靠近边缘的边界框")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
