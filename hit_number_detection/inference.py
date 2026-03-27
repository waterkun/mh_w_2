"""
MH Wilds 伤害数字端到端推理脚本

流程: 游戏画面 → YOLO 检测数字位置 → 裁剪 ROI → CRNN 识别数字内容

用法：
  python inference.py --source photos/damage1.png
  python inference.py --source gameplay.mp4
  python inference.py --source 0  # 摄像头/采集卡
  python inference.py --source photos/damage1.png --show
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from crnn_model import (
    CRNN, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, decode_predictions,
)


class DamageNumberDetector:
    """端到端伤害数字检测+识别器。"""

    def __init__(self, yolo_path, crnn_path, device=None, conf_threshold=0.5):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.conf_threshold = conf_threshold

        # 加载 YOLO
        print(f"加载 YOLO 模型: {yolo_path}")
        self.yolo = YOLO(yolo_path)

        # 加载 CRNN
        print(f"加载 CRNN 模型: {crnn_path}")
        self.crnn = CRNN(NUM_CLASSES).to(self.device)
        self.crnn.load_state_dict(
            torch.load(crnn_path, map_location=self.device, weights_only=True)
        )
        self.crnn.eval()

        print(f"设备: {self.device}")
        print("模型加载完成！\n")

    def detect(self, frame):
        """
        检测并识别一帧中的所有伤害数字。

        Returns:
            list[dict]: 每个检测结果包含 bbox, confidence, number
        """
        # YOLO 检测
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()

                # 裁剪 ROI
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # CRNN 识别
                number = self._recognize(roi)

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "number": number,
                })

        return detections

    def _recognize(self, roi):
        """用 CRNN 识别裁剪出的数字区域。"""
        # 转灰度
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # 缩放到固定尺寸
        h, w = gray.shape
        ratio = IMG_HEIGHT / h
        new_w = min(int(w * ratio), IMG_WIDTH)
        resized = cv2.resize(gray, (new_w, IMG_HEIGHT))

        # Pad
        if new_w < IMG_WIDTH:
            pad = np.zeros((IMG_HEIGHT, IMG_WIDTH - new_w), dtype=np.uint8)
            resized = np.concatenate([resized, pad], axis=1)

        # 归一化 + 转 tensor
        img = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            output = self.crnn(tensor)
            predictions = decode_predictions(output)

        return predictions[0] if predictions else ""

    def draw_results(self, frame, detections):
        """在帧上绘制检测结果。"""
        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            number = det["number"]
            conf = det["confidence"]

            # 画框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 标注数字和置信度
            label = f"{number} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(result_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return result_frame


def run_image(detector, source, output_dir, show):
    """处理单张图片。"""
    img = cv2.imread(source)
    if img is None:
        print(f"错误: 无法读取图片 {source}")
        return

    detections = detector.detect(img)
    result_img = detector.draw_results(img, detections)

    print(f"检测到 {len(detections)} 个伤害数字:")
    for det in detections:
        print(f"  数字: {det['number']:>6s}  "
              f"置信度: {det['confidence']:.2f}  "
              f"位置: {det['bbox']}")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(source))
    cv2.imwrite(out_path, result_img)
    print(f"结果保存到: {out_path}")

    if show:
        cv2.imshow("Detection Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(detector, source, output_dir, show):
    """处理影片或实时画面。"""
    # source 为 "0" 时转为摄像头索引
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"错误: 无法打开 {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 保存结果影片
    os.makedirs(output_dir, exist_ok=True)
    out_name = "output.mp4" if isinstance(source, int) else os.path.basename(str(source))
    out_path = os.path.join(output_dir, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    frame_count = 0
    total_detections = 0
    start_time = time.time()

    print(f"处理影片中... (按 Q 退出)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        total_detections += len(detections)
        result_frame = detector.draw_results(frame, detections)

        writer.write(result_frame)
        frame_count += 1

        if show:
            cv2.imshow("Detection", result_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  已处理 {frame_count} 帧, "
                  f"检测 {total_detections} 个数字, "
                  f"速度 {frame_count / elapsed:.1f} FPS")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n处理完成！")
    print(f"  总帧数: {frame_count}")
    print(f"  检测数字数: {total_detections}")
    print(f"  处理速度: {frame_count / elapsed:.1f} FPS")
    print(f"  结果保存到: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="MH Wilds 伤害数字端到端推理")
    parser.add_argument("--source", required=True,
                        help="输入来源：图片路径 / 影片路径 / 0(摄像头)")
    parser.add_argument("--yolo", default="runs/detect/damage_number/weights/best.pt",
                        help="YOLO 模型路径")
    parser.add_argument("--crnn", default="runs/crnn/best.pt",
                        help="CRNN 模型路径")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO 检测置信度阈值")
    parser.add_argument("--output", default="runs/inference",
                        help="输出目录")
    parser.add_argument("--show", action="store_true",
                        help="显示检测结果窗口")
    args = parser.parse_args()

    detector = DamageNumberDetector(
        yolo_path=args.yolo,
        crnn_path=args.crnn,
        conf_threshold=args.conf,
    )

    # 判断输入类型
    source = args.source
    if source.isdigit():
        # 摄像头
        run_video(detector, source, args.output, args.show)
    elif source.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        # 影片
        run_video(detector, source, args.output, args.show)
    else:
        # 图片
        run_image(detector, source, args.output, args.show)


if __name__ == "__main__":
    main()
