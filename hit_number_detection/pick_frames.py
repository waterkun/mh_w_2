"""
从 frames_filtered 中挑选帧到 CVAT 标注文件夹

纯键盘操作，不需要按 Enter：
  Space/Enter → 选中，跳 4~7 帧
  S           → 跳过，也跳 4~7 帧
  B           → 回退上一帧
  Q/Esc       → 退出

用法：
  python pick_frames.py
  python pick_frames.py --input frames_filtered --output dataset-cvat3 --skip-min 4 --skip-max 7
"""

import argparse
import os
import random
import shutil
import cv2
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="从筛选帧中挑选帧到 CVAT 标注文件夹")
    parser.add_argument("--input", default="frames_filtered", help="输入帧文件夹")
    parser.add_argument("--output", default="dataset-cvat3", help="输出文件夹")
    parser.add_argument("--skip-min", type=int, default=4, help="跳帧最少 (默认 4)")
    parser.add_argument("--skip-max", type=int, default=7, help="跳帧最多 (默认 7)")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    all_frames = sorted(Path(input_dir).glob("*.png"))
    if not all_frames:
        print(f"错误: {input_dir} 中没有 PNG 文件")
        return

    total = len(all_frames)
    picked = len(list(Path(output_dir).glob("*.png")))

    print(f"共 {total} 帧，已选 {picked} 张")
    print(f"输出到: {output_dir}")
    print()
    print("操作（直接按键，不用 Enter）：")
    print(f"  Space/Enter → 选中，跳 {args.skip_min}~{args.skip_max} 帧")
    print(f"  S           → 跳过，也跳 {args.skip_min}~{args.skip_max} 帧")
    print("  B           → 回退")
    print("  Q/Esc       → 退出")

    idx = 0

    while idx < total:
        frame_path = all_frames[idx]
        fname = frame_path.name

        img = cv2.imread(str(frame_path))
        if img is None:
            idx += 1
            continue

        # 缩小显示
        h, w = img.shape[:2]
        scale = min(1280 / w, 720 / h, 1.0)
        display = cv2.resize(img, (int(w * scale), int(h * scale)))

        # 状态栏
        bar_h = 35
        bar = 50 * np.ones((bar_h, display.shape[1], 3), dtype=np.uint8)
        status = f"[{idx + 1}/{total}] {fname} | Picked: {picked} | Space=pick S=skip B=back Q=quit"
        cv2.putText(bar, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        display = np.vstack([display, bar])

        cv2.imshow("Pick Frames", display)
        key = cv2.waitKey(0) & 0xFF

        # Q or Esc
        if key in (ord('q'), ord('Q'), 27):
            print(f"\n退出！共选了 {picked} 张到 {output_dir}")
            cv2.destroyAllWindows()
            return

        # B = back
        elif key in (ord('b'), ord('B')):
            if idx > 0:
                idx -= 1

        # S = skip with jump
        elif key in (ord('s'), ord('S')):
            skip = random.randint(args.skip_min, args.skip_max)
            idx += skip

        # Space or Enter = pick
        elif key in (32, 13):  # 32=space, 13=enter
            dst = os.path.join(output_dir, fname)
            shutil.copy2(str(frame_path), dst)
            picked += 1
            print(f"  ✓ {fname} (已选 {picked})")

            skip = random.randint(args.skip_min, args.skip_max)
            idx += skip

        # 其他键忽略

    print(f"\n浏览完毕！共选了 {picked} 张到 {output_dir}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
