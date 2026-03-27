"""用训练好的模型对未标注图片生成 YOLO 预标注.

用法:
  python -m monster_detect.prelabel
  python -m monster_detect.prelabel --conf 0.3
  python -m monster_detect.prelabel --model monster_detect/runs/monster_detect/weights/best.pt
"""

import argparse
import os

from ultralytics import YOLO

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_UNLABELED_DIR = os.path.join(_MODULE_DIR, "monster_yolo_data", "images", "unlabeled")
_DEFAULT_MODEL = os.path.join(_MODULE_DIR, "runs", "monster_detect", "weights", "best.pt")


def prelabel(model_path: str, conf: float = 0.3, limit: int = 0):
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 {model_path}")
        return

    model = YOLO(model_path)
    class_names = model.names  # {0: 'body', 1: 'chain', ...}
    print(f"模型: {model_path}")
    print(f"类别: {class_names}")
    print(f"置信度阈值: {conf}")
    print()

    # 找没有 .txt 的图片
    images = []
    for fname in sorted(os.listdir(_UNLABELED_DIR)):
        if not fname.endswith(".jpg"):
            continue
        txt_path = os.path.join(_UNLABELED_DIR, fname.replace(".jpg", ".txt"))
        if not os.path.exists(txt_path):
            images.append(fname)

    if limit > 0:
        images = images[:limit]
    print(f"待预标注图片: {len(images)} 张")
    if not images:
        print("没有需要预标注的图片")
        return

    labeled = 0
    total_boxes = 0

    for fname in images:
        img_path = os.path.join(_UNLABELED_DIR, fname)
        results = model(img_path, verbose=False, conf=conf)
        boxes = results[0].boxes

        if len(boxes) == 0:
            continue

        img_h, img_w = results[0].orig_shape
        lines = []
        for box in boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        txt_path = os.path.join(_UNLABELED_DIR, fname.replace(".jpg", ".txt"))
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        labeled += 1
        total_boxes += len(lines)

    print(f"\n完成! 预标注 {labeled} 张图片, 共 {total_boxes} 个 bbox")
    print(f"请用 labelImg 打开确认/修正:")
    print(f"  labelImg {_UNLABELED_DIR}")


def main():
    parser = argparse.ArgumentParser(description="用已训练模型预标注未标注图片")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="模型路径")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="置信度阈值 (默认 0.3, 低一点多标一些)")
    parser.add_argument("--limit", type=int, default=0,
                        help="只预标注前 N 张 (默认 0 = 全部)")
    args = parser.parse_args()
    prelabel(args.model, args.conf, args.limit)


if __name__ == "__main__":
    main()
