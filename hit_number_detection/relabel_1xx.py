"""
手动重标 1XX 三位数 bbox 工具

操作:
    A / D       左边缘 左移/右移 1px
    ← / →       右边缘 左移/右移 1px
    W / S       上边缘 上移/下移 1px
    ↑ / ↓       下边缘 上移/下移 1px
    R           重置为原始 bbox
    Enter/Space 确认并下一个
    Q           保存已完成的并退出
    N           跳过 (不修改)

显示:
    红框 = 原始 bbox
    绿框 = 当前编辑中的 bbox
    左上角显示数字内容和进度

Usage:
    E:/anaconda3/envs/mh_ai/python.exe hit_number_detection/relabel_1xx.py
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
YOLO_DIR = BASE / "valid-yolo-data"
LABELS_TXT = BASE / "crnn_data" / "ocr_prelabel" / "labels.txt"
CROP_MAPPING = BASE / "crnn_data" / "ocr_prelabel" / "crop_mapping.json"
PROGRESS_FILE = BASE / "yolo_diagnosis" / "relabel_progress.json"


def yolo_to_pixel(bbox_norm, img_w, img_h):
    cx, cy, w, h = bbox_norm
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w, int((cx + w / 2) * img_w))
    y2 = min(img_h, int((cy + h / 2) * img_h))
    return x1, y1, x2, y2


def pixel_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def _read_current_bbox(label_file, line_index):
    """从当前 label 文件读取实际 bbox（而不是 crop_mapping 的原始值）"""
    p = Path(label_file)
    if not p.exists():
        return None
    with open(p, "r") as f:
        lines = f.readlines()
    if line_index >= len(lines):
        return None
    parts = lines[line_index].strip().split()
    if len(parts) < 5:
        return None
    return tuple(float(x) for x in parts[1:5])


def load_data():
    """加载所有 1XX 记录，bbox 从当前 label 文件读取"""
    with open(LABELS_TXT, "r", encoding="utf-8") as f:
        crnn_labels = {}
        for line in f:
            if "\t" in line:
                name, text = line.strip().split("\t", 1)
                crnn_labels[name.strip()] = text.strip()

    with open(CROP_MAPPING, "r", encoding="utf-8") as f:
        crop_mapping = json.load(f)

    records = []
    for crop_name, info in crop_mapping.items():
        digit_text = crnn_labels.get(crop_name, "")
        if len(digit_text) != 3 or digit_text[0] != "1":
            continue

        # 从当前 label 文件读取最新 bbox
        bbox_norm = _read_current_bbox(info["label_file"], info["line_index"])
        if bbox_norm is None:
            continue

        records.append({
            "crop_name": crop_name,
            "digit_text": digit_text,
            "bbox_norm": bbox_norm,
            "split": info["split"],
            "source_image": info["source_image"],
            "label_file": info["label_file"],
            "line_index": info["line_index"],
        })

    # 按 source_image 排序，减少图片加载次数
    records.sort(key=lambda r: (r["split"], r["source_image"], r["line_index"]))
    return records


def find_image(source_image, split):
    """找到对应的原始图片"""
    img_dir = YOLO_DIR / "images" / split
    for ext in [".png", ".jpg"]:
        # 先试 crop_mapping 里的名字
        p = img_dir / source_image
        if p.exists():
            return p
        # 试换扩展名
        p = img_dir / (Path(source_image).stem + ext)
        if p.exists():
            return p
    return None


def draw_view(img, orig_box, edit_box, digit_text, idx, total, zoom_pad=60):
    """绘制编辑视图"""
    img_h, img_w = img.shape[:2]
    ox1, oy1, ox2, oy2 = orig_box
    ex1, ey1, ex2, ey2 = edit_box

    # 计算局部区域
    region_x1 = max(0, min(ox1, ex1) - zoom_pad)
    region_y1 = max(0, min(oy1, ey1) - zoom_pad)
    region_x2 = min(img_w, max(ox2, ex2) + zoom_pad)
    region_y2 = min(img_h, max(oy2, ey2) + zoom_pad)

    region = img[region_y1:region_y2, region_x1:region_x2].copy()

    # 放大
    scale = max(1, 800 // max(region.shape[1], region.shape[0], 1))
    scale = min(scale, 12)
    region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 画原始框 (红)
    ro = (
        (ox1 - region_x1) * scale, (oy1 - region_y1) * scale,
        (ox2 - region_x1) * scale, (oy2 - region_y1) * scale,
    )
    cv2.rectangle(region, (ro[0], ro[1]), (ro[2], ro[3]), (0, 0, 255), 2)

    # 画编辑框 (绿)
    re = (
        (ex1 - region_x1) * scale, (ey1 - region_y1) * scale,
        (ex2 - region_x1) * scale, (ey2 - region_y1) * scale,
    )
    cv2.rectangle(region, (re[0], re[1]), (re[2], re[3]), (0, 255, 0), 2)

    # 信息栏
    info_h = 80
    info_bar = np.zeros((info_h, region.shape[1], 3), dtype=np.uint8)
    lines = [
        f"[{idx + 1}/{total}] '{digit_text}'   red=original  green=editing",
        f"A/D=left edge  arrows=right edge  W/S=top  up/dn=bottom",
        f"Enter=confirm  N=skip  R=reset  Q=save&quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(info_bar, text, (10, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # 尺寸信息
    size_bar = np.zeros((30, region.shape[1], 3), dtype=np.uint8)
    ow = ox2 - ox1
    ew = ex2 - ex1
    delta = ew - ow
    sign = "+" if delta >= 0 else ""
    cv2.putText(size_bar,
                f"orig: {ow}x{oy2 - oy1}px  edit: {ew}x{ey2 - ey1}px  delta_w: {sign}{delta}px",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return np.vstack([info_bar, region, size_bar])


def save_label_changes(changes):
    """将所有修改写回 label 文件"""
    # 按 label_file 分组
    by_file = defaultdict(dict)
    for c in changes:
        by_file[c["label_file"]][c["line_index"]] = c["new_bbox_norm"]

    files_modified = 0
    for label_file, line_fixes in by_file.items():
        label_path = Path(label_file)
        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        modified = False
        for line_idx, new_bbox in line_fixes.items():
            if line_idx < len(lines):
                parts = lines[line_idx].strip().split()
                cls_id = parts[0]
                cx, cy, w, h = new_bbox
                lines[line_idx] = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                modified = True

        if modified:
            with open(label_path, "w") as f:
                f.writelines(lines)
            files_modified += 1

    return files_modified


def main():
    print("加载 1XX 数据...")
    records = load_data()
    print(f"共 {len(records)} 个 1XX bbox 需要审查")

    # 加载进度
    done_set = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
            done_set = set(progress.get("done", []))
        print(f"已完成 {len(done_set)} 个，从上次继续")

    # 过滤已完成的
    remaining = [r for r in records if r["crop_name"] not in done_set]
    print(f"剩余 {len(remaining)} 个")

    if not remaining:
        print("全部完成！")
        return

    changes = []
    done_list = list(done_set)
    current_img = None
    current_img_path = None

    window_name = "Relabel 1XX - bbox editor"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    i = 0
    while i < len(remaining):
        r = remaining[i]

        # 加载图片 (缓存同一张)
        img_path = find_image(r["source_image"], r["split"])
        if img_path is None:
            print(f"  [跳过] 找不到图片: {r['source_image']}")
            done_list.append(r["crop_name"])
            i += 1
            continue

        if str(img_path) != current_img_path:
            current_img = cv2.imread(str(img_path))
            current_img_path = str(img_path)

        if current_img is None:
            i += 1
            continue

        img_h, img_w = current_img.shape[:2]

        # 原始 bbox (像素)
        orig_pixel = list(yolo_to_pixel(r["bbox_norm"], img_w, img_h))
        # 编辑中的 bbox
        edit_pixel = list(orig_pixel)

        while True:
            view = draw_view(current_img, orig_pixel, edit_pixel,
                             r["digit_text"], i, len(remaining))
            cv2.imshow(window_name, view)
            key = cv2.waitKeyEx(0)

            # A: 左边缘左移
            if key == ord('a') or key == ord('A'):
                edit_pixel[0] = max(0, edit_pixel[0] - 1)
            # D: 左边缘右移
            elif key == ord('d') or key == ord('D'):
                edit_pixel[0] = min(edit_pixel[2] - 1, edit_pixel[0] + 1)
            # ← (2424832): 右边缘左移
            elif key == 2424832 or key == 81:
                edit_pixel[2] = max(edit_pixel[0] + 1, edit_pixel[2] - 1)
            # → (2555904): 右边缘右移
            elif key == 2555904 or key == 83:
                edit_pixel[2] = min(img_w, edit_pixel[2] + 1)
            # W: 上边缘上移
            elif key == ord('w') or key == ord('W'):
                edit_pixel[1] = max(0, edit_pixel[1] - 1)
            # S: 上边缘下移
            elif key == ord('s') or key == ord('S'):
                edit_pixel[1] = min(edit_pixel[3] - 1, edit_pixel[1] + 1)
            # ↑ (2490368): 下边缘上移
            elif key == 2490368 or key == 82:
                edit_pixel[3] = max(edit_pixel[1] + 1, edit_pixel[3] - 1)
            # ↓ (2621440): 下边缘下移
            elif key == 2621440 or key == 84:
                edit_pixel[3] = min(img_h, edit_pixel[3] + 1)
            # R: 重置
            elif key == ord('r') or key == ord('R'):
                edit_pixel = list(orig_pixel)
            # Enter / Space: 确认
            elif key == 13 or key == 32:
                if edit_pixel != list(orig_pixel):
                    new_norm = pixel_to_yolo(*edit_pixel, img_w, img_h)
                    changes.append({
                        "label_file": r["label_file"],
                        "line_index": r["line_index"],
                        "new_bbox_norm": new_norm,
                        "crop_name": r["crop_name"],
                        "digit_text": r["digit_text"],
                    })
                    print(f"  [{i + 1}/{len(remaining)}] '{r['digit_text']}' 已修改")
                else:
                    print(f"  [{i + 1}/{len(remaining)}] '{r['digit_text']}' 无变化")
                done_list.append(r["crop_name"])
                i += 1
                break
            # N: 跳过
            elif key == ord('n') or key == ord('N'):
                print(f"  [{i + 1}/{len(remaining)}] '{r['digit_text']}' 跳过")
                done_list.append(r["crop_name"])
                i += 1
                break
            # Q: 保存退出
            elif key == ord('q') or key == ord('Q'):
                print("\n保存并退出...")
                i = len(remaining)  # 退出外层循环
                break

    cv2.destroyAllWindows()

    # 保存修改
    if changes:
        files_modified = save_label_changes(changes)
        print(f"\n已修改 {len(changes)} 个 bbox，涉及 {files_modified} 个 label 文件")
    else:
        print("\n没有修改")

    # 保存进度
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done": done_list}, f)
    print(f"进度已保存 ({len(done_list)}/{len(records)})")


if __name__ == "__main__":
    main()
