"""
MH Wilds 伤害数字 — YOLOv10 检测模型训练脚本

YOLOv10 优势: NMS-free 检测，推理更快，适合实时场景。

用法：
  python train_yolo.py
  python train_yolo.py --epochs 200 --batch 32 --img 640
  python train_yolo.py --data dataset.yaml --model yolov10n.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(args):
    print("=" * 60)
    print("MH Wilds 伤害数字检测 — YOLO 训练")
    print("=" * 60)

    # 加载预训练模型
    model = YOLO(args.model)
    print(f"基础模型: {args.model}")
    print(f"数据集配置: {args.data}")
    print(f"训练参数: epochs={args.epochs}, batch={args.batch}, imgsz={args.img}")
    print()

    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        project="runs",
        name="damage_number",
        exist_ok=True,
        # 数据增强（针对游戏画面优化）
        hsv_h=0.01,      # 色相微调（游戏数字颜色比较固定）
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5.0,      # 轻微旋转（数字基本水平）
        translate=0.1,
        scale=0.3,
        flipud=0.0,       # 不上下翻转（数字不会倒着出现）
        fliplr=0.0,       # 不左右翻转
        mosaic=0.5,
        patience=30,       # 早停
    )

    print("\n训练完成！")

    # 验证
    print("\n" + "=" * 60)
    print("验证最佳模型...")
    print("=" * 60)
    best_model_path = Path("runs/damage_number/weights/best.pt")
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        metrics = best_model.val()
        print(f"\n验证结果:")
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")
    else:
        print("警告: 未找到 best.pt，请检查训练输出")

    # 导出 ONNX（方便后续部署）
    if args.export:
        print("\n导出 ONNX 模型...")
        best_model = YOLO(str(best_model_path))
        best_model.export(format="onnx", imgsz=args.img)
        print("ONNX 导出完成！")

    print(f"\n模型保存位置: runs/damage_number/weights/")


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv10 伤害数字检测模型")
    parser.add_argument("--data", default="dataset.yaml", help="数据集配置文件")
    parser.add_argument("--model", default="yolov10n.pt", help="预训练模型 (默认 yolov10n)")
    parser.add_argument("--epochs", type=int, default=150, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--img", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--export", action="store_true", help="训练后导出 ONNX")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
