"""怪物检测 — YOLOv10n 训练脚本.

用法:
  python -m monster_detect.train_monster_yolo
  python -m monster_detect.train_monster_yolo --epochs 100 --batch 16
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_HN_DIR = os.path.join(_MODULE_DIR, "..", "..", "hit_number_detection")
_DEFAULT_MODEL = "yolov10s.pt"
_DATA_YAML = os.path.join(_MODULE_DIR, "monster_yolo_data", "dataset.yaml")
_PROJECT_DIR = os.path.join(_MODULE_DIR, "runs")


def train(args):
    print("=" * 60)
    print("MH Wilds 怪物检测 — YOLOv10s 训练")
    print("=" * 60)

    model = YOLO(args.model)
    print(f"基础模型: {args.model}")
    print(f"数据集配置: {args.data}")
    print(f"训练参数: epochs={args.epochs}, batch={args.batch}, imgsz={args.img}")
    print()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        project=_PROJECT_DIR,
        name="monster_detect",
        exist_ok=True,
        # 数据增强 (针对游戏画面)
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=0.0,       # 怪物不会旋转
        translate=0.1,
        scale=0.4,          # 怪物远近大小变化大
        flipud=0.0,
        fliplr=0.5,         # 怪物左右翻转是合理的
        mosaic=0.5,
        patience=30,
    )

    print("\n训练完成!")

    # 验证
    print("\n" + "=" * 60)
    print("验证最佳模型...")
    print("=" * 60)

    best_path = Path(_PROJECT_DIR) / "monster_detect" / "weights" / "best.pt"
    if best_path.exists():
        best_model = YOLO(str(best_path))
        metrics = best_model.val()
        print(f"\n验证结果:")
        print(f"  mAP50:     {metrics.box.map50:.4f}")
        print(f"  mAP50-95:  {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")
    else:
        print("警告: 未找到 best.pt")

    print(f"\n模型保存位置: {best_path}")


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv10s 怪物检测模型")
    parser.add_argument("--data", default=_DATA_YAML, help="数据集配置文件")
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="预训练模型 (默认 yolov10n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    parser.add_argument("--img", type=int, default=1280, help="输入图片尺寸")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
