"""
MH Wilds 伤害数字帧提取与筛选工具

功能：
1. 从游戏录屏中按指定 FPS 抽帧
2. 可裁剪出纯游戏画面区域
3. 自动筛选可能包含伤害数字的帧（基于颜色特征检测）
4. 将筛选结果保存到指定文件夹供后续标注

使用方式：
  python extract_frames.py --video gameplay.mp4
  python extract_frames.py --video gameplay.mp4 --fps 10 --crop 0,0,960,540
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: int = 10,
                   crop: tuple = None):
    """
    从影片中按指定 FPS 抽帧。

    Args:
        video_path: 影片路径
        output_dir: 输出文件夹
        fps: 每秒抽几帧
        crop: 裁剪区域 (x, y, w, h)，None 则不裁剪
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开影片 {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"影片信息：")
    print(f"  分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {video_fps:.1f}")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.1f} 秒")
    print(f"  抽帧 FPS: {fps} → 预计输出 {int(duration * fps)} 帧")
    print()

    # 每隔多少帧取一帧
    frame_interval = max(1, int(video_fps / fps))

    saved_paths = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # 裁剪
            if crop is not None:
                x, y, w, h = crop
                frame = frame[y:y+h, x:x+w]

            filename = f"frame12_{saved_count:05d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"  已抽取 {saved_count} 帧...")

        frame_idx += 1

    cap.release()
    print(f"抽帧完成！共保存 {saved_count} 帧到 {output_dir}")
    return saved_paths


def detect_damage_numbers(frame: np.ndarray, min_area: int = 30,
                          max_area: int = 5000) -> dict:
    """
    检测一帧中是否可能包含伤害数字。
    基于 MH Wilds 伤害数字的颜色特征：
      - 白色/浅色：普通伤害
      - 橙色/红色：暴击伤害
      - 黄色：属性伤害

    返回检测结果字典，包含候选区域数量和得分。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    results = {}
    total_contours = 0

    # ---- 检测白色/亮色数字（普通伤害）----
    # 高亮度 + 低饱和度 = 白色文字
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
    white_mask = _clean_mask(white_mask)
    white_contours = _find_valid_contours(white_mask, min_area, max_area)
    results["white"] = len(white_contours)
    total_contours += len(white_contours)

    # ---- 检测橙色/红色数字（暴击伤害）----
    # 橙色范围：色相 5-25，高饱和度，高亮度
    orange_mask = cv2.inRange(hsv, (5, 120, 180), (25, 255, 255))
    # 红色范围（色相环两端）
    red_mask1 = cv2.inRange(hsv, (0, 120, 180), (5, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 120, 180), (180, 255, 255))
    crit_mask = orange_mask | red_mask1 | red_mask2
    crit_mask = _clean_mask(crit_mask)
    crit_contours = _find_valid_contours(crit_mask, min_area, max_area)
    results["critical"] = len(crit_contours)
    total_contours += len(crit_contours)

    # ---- 检测黄色数字（属性伤害）----
    yellow_mask = cv2.inRange(hsv, (20, 100, 180), (35, 255, 255))
    yellow_mask = _clean_mask(yellow_mask)
    yellow_contours = _find_valid_contours(yellow_mask, min_area, max_area)
    results["elemental"] = len(yellow_contours)
    total_contours += len(yellow_contours)

    results["total"] = total_contours
    return results


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """形态学清理：去噪 + 连接断裂的笔画"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _find_valid_contours(mask: np.ndarray, min_area: int,
                         max_area: int) -> list:
    """找出符合面积范围的轮廓（过滤太大或太小的噪点）"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            # 额外检查：伤害数字通常宽高比在合理范围内
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            # 数字或数字串的宽高比通常在 0.2 ~ 5.0 之间
            if 0.2 <= aspect <= 5.0:
                valid.append(c)
    return valid


def filter_frames(input_dir: str, output_dir: str,
                  threshold: int = 2, save_debug: bool = False,
                  roi: tuple = None):
    """
    筛选可能包含伤害数字的帧。

    Args:
        input_dir: 抽帧输出文件夹
        output_dir: 筛选结果文件夹
        threshold: 至少检测到几个候选区域才算有伤害数字
        save_debug: 是否保存带标记的调试图
        roi: 检测区域 (x, y, w, h)，None 则检测整帧
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

    frame_files = sorted(Path(input_dir).glob("*.png"))
    if not frame_files:
        print(f"错误：{input_dir} 中没有找到 PNG 文件")
        return

    if roi is not None:
        print(f"ROI 检测区域: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print(f"开始筛选，共 {len(frame_files)} 帧，阈值={threshold}...")

    selected = 0
    stats = {"white": 0, "critical": 0, "elemental": 0}

    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue

        # 如果指定了 ROI，只检测该区域
        detect_region = frame
        if roi is not None:
            rx, ry, rw, rh = roi
            detect_region = frame[ry:ry+rh, rx:rx+rw]

        result = detect_damage_numbers(detect_region)

        if result["total"] >= threshold:
            # 复制到输出文件夹
            out_path = os.path.join(output_dir, fpath.name)
            cv2.imwrite(out_path, frame)
            selected += 1

            for key in stats:
                if result.get(key, 0) > 0:
                    stats[key] += 1

            # 保存调试图（在检测区域画框）
            if save_debug:
                debug_frame = frame.copy()
                if roi is not None:
                    rx, ry, rw, rh = roi
                    # 画 ROI 边界（蓝色虚线效果）
                    cv2.rectangle(debug_frame, (rx, ry),
                                  (rx+rw, ry+rh), (255, 0, 0), 2)
                    # 只在 ROI 区域内画检测结果
                    roi_region = debug_frame[ry:ry+rh, rx:rx+rw]
                    _draw_debug_on(roi_region)
                else:
                    _draw_debug_on(debug_frame)
                debug_path = os.path.join(debug_dir, fpath.name)
                cv2.imwrite(debug_path, debug_frame)

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(frame_files)} 帧，已选出 {selected} 帧")

    print(f"\n筛选完成！")
    print(f"  总帧数: {len(frame_files)}")
    print(f"  选出帧数: {selected} ({selected/len(frame_files)*100:.1f}%)")
    print(f"  包含普通伤害: {stats['white']} 帧")
    print(f"  包含暴击伤害: {stats['critical']} 帧")
    print(f"  包含属性伤害: {stats['elemental']} 帧")
    print(f"  输出文件夹: {output_dir}")


def _draw_debug_on(image: np.ndarray):
    """在图片上原地画出检测到的候选区域，用于调试确认"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 白色区域 → 绿框
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
    white_mask = _clean_mask(white_mask)
    _draw_contour_boxes(image, white_mask, (0, 255, 0), "normal")

    # 橙红色区域 → 红框
    orange_mask = cv2.inRange(hsv, (5, 120, 180), (25, 255, 255))
    red_mask1 = cv2.inRange(hsv, (0, 120, 180), (5, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 120, 180), (180, 255, 255))
    crit_mask = orange_mask | red_mask1 | red_mask2
    crit_mask = _clean_mask(crit_mask)
    _draw_contour_boxes(image, crit_mask, (0, 0, 255), "crit")

    # 黄色区域 → 黄框
    yellow_mask = cv2.inRange(hsv, (20, 100, 180), (35, 255, 255))
    yellow_mask = _clean_mask(yellow_mask)
    _draw_contour_boxes(image, yellow_mask, (0, 255, 255), "elem")


def _draw_contour_boxes(image: np.ndarray, mask: np.ndarray,
                        color: tuple, label: str,
                        min_area: int = 30, max_area: int = 5000):
    """在图片上画出轮廓的 bounding box"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            if 0.2 <= aspect <= 5.0:
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image, label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def preview_crop(video_path: str, crop: tuple = None, roi: tuple = None):
    """
    预览影片第一帧、裁剪区域和 ROI 检测区域，方便调整参数。
    按任意键关闭窗口。
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("错误：无法读取影片")
        return

    print(f"原始分辨率: {frame.shape[1]}x{frame.shape[0]}")

    preview = frame.copy()

    if crop is not None:
        x, y, w, h = crop
        cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(preview, "CROP", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(preview, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 3)
        cv2.putText(preview, "ROI", (rx, ry - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Preview (green=crop, red=ROI)", preview)

    if crop is not None:
        cropped = frame[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2]]
        cv2.imshow("Cropped Result", cropped)

    print("按任意键关闭预览...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_crop(crop_str: str) -> tuple:
    """解析裁剪参数字符串 'x,y,w,h'"""
    parts = [int(x.strip()) for x in crop_str.split(",")]
    if len(parts) != 4:
        raise ValueError("裁剪参数格式错误，需要 x,y,w,h")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="MH Wilds 伤害数字帧提取与筛选工具"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- preview 子命令 ----
    p_preview = subparsers.add_parser("preview", help="预览影片和裁剪区域")
    p_preview.add_argument("--video", required=True, help="影片路径")
    p_preview.add_argument("--crop", default=None,
                           help="裁剪区域 x,y,w,h（像素）")
    p_preview.add_argument("--roi", default=None,
                           help="检测区域 x,y,w,h（像素），预览时显示为红色框")

    # ---- extract 子命令 ----
    p_extract = subparsers.add_parser("extract", help="从影片中抽帧")
    p_extract.add_argument("--video", required=True, help="影片路径")
    p_extract.add_argument("--fps", type=int, default=10,
                           help="每秒抽几帧 (默认 10)")
    p_extract.add_argument("--crop", default=None,
                           help="裁剪区域 x,y,w,h（像素）")
    p_extract.add_argument("--output", default="frames_raw",
                           help="输出文件夹 (默认 frames_raw)")

    # ---- filter 子命令 ----
    p_filter = subparsers.add_parser("filter", help="筛选包含伤害数字的帧")
    p_filter.add_argument("--input", default="frames_raw",
                          help="抽帧文件夹 (默认 frames_raw)")
    p_filter.add_argument("--output", default="frames_filtered",
                          help="输出文件夹 (默认 frames_filtered)")
    p_filter.add_argument("--threshold", type=int, default=2,
                          help="候选区域阈值 (默认 2)")
    p_filter.add_argument("--debug", action="store_true",
                          help="保存调试图（显示检测区域）")
    p_filter.add_argument("--roi", default="924,25,1534,1398",
                          help="检测区域 x,y,w,h（像素），只在此区域内检测伤害数字 (默认 924,25,1534,1398)")

    # ---- 全流程 ----
    p_all = subparsers.add_parser("run", help="一键执行：抽帧 + 筛选")
    p_all.add_argument("--video", required=True, help="影片路径")
    p_all.add_argument("--fps", type=int, default=10, help="每秒抽几帧")
    p_all.add_argument("--crop", default=None, help="裁剪区域 x,y,w,h")
    p_all.add_argument("--threshold", type=int, default=2,
                        help="筛选阈值")
    p_all.add_argument("--debug", action="store_true",
                        help="保存调试图")
    p_all.add_argument("--roi", default="924,25,1534,1398",
                        help="检测区域 x,y,w,h（像素），只在此区域内检测伤害数字 (默认 924,25,1534,1398)")

    args = parser.parse_args()

    if args.command == "preview":
        crop = parse_crop(args.crop) if args.crop else None
        roi = parse_crop(args.roi) if args.roi else None
        preview_crop(args.video, crop, roi)

    elif args.command == "extract":
        crop = parse_crop(args.crop) if args.crop else None
        extract_frames(args.video, args.output, args.fps, crop)

    elif args.command == "filter":
        roi = parse_crop(args.roi) if args.roi else None
        filter_frames(args.input, args.output, args.threshold, args.debug, roi)

    elif args.command == "run":
        crop = parse_crop(args.crop) if args.crop else None
        print("=" * 50)
        print("步骤 1/2：抽帧")
        print("=" * 50)
        extract_frames(args.video, "frames_raw", args.fps, crop)
        print()
        print("=" * 50)
        print("步骤 2/2：筛选有伤害数字的帧")
        print("=" * 50)
        roi = parse_crop(args.roi) if args.roi else None
        filter_frames("frames_raw", "frames_filtered",
                      args.threshold, args.debug, roi)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
