"""交互式视频标注工具 — 标记攻击时间段.

控制:
    Space       播放 / 暂停
    D / →       下一帧
    A / ←       上一帧
    W           快进 30 帧
    S           快退 30 帧
    [           标记起始帧
    ]           标记结束帧
    1-8         选择攻击类型
    C           确认当前标注
    X           删除最近一条标注
    Z           保存并退出
    Q / Esc     退出 (自动保存)
"""

import json
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ATTACK_CLASSES, CLASS_TO_IDX


def label_video(video_path, output_path=None):
    """打开视频进行攻击时间段标注."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_path is None:
        output_path = os.path.join(os.path.dirname(video_path),
                                   f"{video_name}_labels.json")

    # 加载已有标注
    annotations = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        annotations = data.get("annotations", [])
        print(f"加载已有标注: {len(annotations)} 条")

    current_frame = 0
    playing = False
    start_frame = None
    end_frame = None
    selected_class = None

    # 颜色映射
    class_colors = [
        (128, 128, 128),  # idle — 灰
        (0, 165, 255),    # pounce — 橙
        (0, 0, 255),      # beam — 红
        (0, 255, 255),    # tail_sweep — 黄
        (255, 0, 0),      # flying_attack — 蓝
        (0, 255, 0),      # claw_swipe — 绿
        (255, 0, 255),    # charge — 紫
        (0, 255, 255),    # nova — 青
    ]

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        return frame if ret else None

    def draw_timeline(canvas, w, h=40):
        """在画布底部绘制时间轴，标注已标注区间."""
        timeline = np.zeros((h, w, 3), dtype=np.uint8)
        timeline[:] = (40, 40, 40)

        for ann in annotations:
            s = int(ann["start_frame"] / total_frames * w)
            e = int(ann["end_frame"] / total_frames * w)
            cidx = CLASS_TO_IDX.get(ann["attack"], 0)
            color = class_colors[cidx % len(class_colors)]
            cv2.rectangle(timeline, (s, 2), (e, h - 2), color, -1)

        # 当前标注范围 (半透明)
        if start_frame is not None:
            s = int(start_frame / total_frames * w)
            ef = end_frame if end_frame is not None else current_frame
            e = int(ef / total_frames * w)
            cv2.rectangle(timeline, (s, 0), (e, h), (255, 255, 255), 1)

        # 播放头位置
        px = int(current_frame / total_frames * w)
        cv2.line(timeline, (px, 0), (px, h), (0, 0, 255), 2)

        return timeline

    def draw_info_panel(w, h=120):
        """绘制信息面板."""
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        y = 20
        # 帧信息
        time_sec = current_frame / fps
        cv2.putText(panel, f"Frame: {current_frame}/{total_frames}  "
                    f"Time: {time_sec:.1f}s  FPS: {fps:.0f}  "
                    f"{'PLAYING' if playing else 'PAUSED'}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        y += 22

        # 当前标注状态
        mark_text = f"Start: {start_frame}  End: {end_frame}  "
        if selected_class is not None:
            mark_text += f"Class: {ATTACK_CLASSES[selected_class]} ({selected_class})"
        else:
            mark_text += "Class: (按 1-8 选择)"
        cv2.putText(panel, mark_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 22

        # 类别列表
        class_line = "  ".join(f"{i+1}:{name}"
                               for i, name in enumerate(ATTACK_CLASSES))
        cv2.putText(panel, class_line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        y += 22

        # 标注计数
        cv2.putText(panel, f"已标注: {len(annotations)} 段  "
                    f"[=起 ]=止 C=确认 X=删除 Z=保存退出",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (150, 150, 150), 1)
        return panel

    def save():
        data = {
            "video": os.path.basename(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "annotations": annotations,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已保存 {len(annotations)} 条标注到 {output_path}")

    print(f"视频: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps:.0f}")
    print("操作: Space=播放/暂停  A/D=前后帧  W/S=快进退")
    print("       [=起始  ]=结束  1-8=类别  C=确认  X=删除  Z/Q=退出")

    while True:
        frame = read_frame(current_frame)
        if frame is None:
            current_frame = max(0, current_frame - 1)
            frame = read_frame(current_frame)
            if frame is None:
                break

        # 缩放显示
        disp_w = min(frame.shape[1], 960)
        scale = disp_w / frame.shape[1]
        disp_h = int(frame.shape[0] * scale)
        display = cv2.resize(frame, (disp_w, disp_h))

        # 叠加时间轴和信息
        timeline = draw_timeline(disp_w)
        info = draw_info_panel(disp_w)
        canvas = np.vstack([display, timeline, info])

        cv2.imshow("Attack Labeler", canvas)

        wait_ms = int(1000 / fps) if playing else 0
        key = cv2.waitKey(max(wait_ms, 1)) & 0xFF

        if key == ord("q") or key == 27:  # Q / Esc
            save()
            break
        elif key == ord("z"):
            save()
            break
        elif key == ord(" "):  # Space
            playing = not playing
        elif key == ord("d") or key == 83:  # D / →
            playing = False
            current_frame = min(current_frame + 1, total_frames - 1)
        elif key == ord("a") or key == 81:  # A / ←
            playing = False
            current_frame = max(current_frame - 1, 0)
        elif key == ord("w"):  # 快进
            current_frame = min(current_frame + 30, total_frames - 1)
        elif key == ord("s"):  # 快退
            current_frame = max(current_frame - 30, 0)
        elif key == ord("["):
            start_frame = current_frame
            print(f"起始帧: {start_frame}")
        elif key == ord("]"):
            end_frame = current_frame
            print(f"结束帧: {end_frame}")
        elif ord("1") <= key <= ord("8"):
            selected_class = key - ord("1")
            print(f"类别: {ATTACK_CLASSES[selected_class]}")
        elif key == ord("c"):  # 确认标注
            if start_frame is not None and end_frame is not None and selected_class is not None:
                if end_frame > start_frame:
                    ann = {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "attack": ATTACK_CLASSES[selected_class],
                    }
                    annotations.append(ann)
                    print(f"已添加: {ann}")
                    start_frame = None
                    end_frame = None
                    selected_class = None
                else:
                    print("错误: 结束帧必须大于起始帧")
            else:
                print("请先设置 [起始帧] [结束帧] 和 1-8类别")
        elif key == ord("x"):  # 删除最近标注
            if annotations:
                removed = annotations.pop()
                print(f"已删除: {removed}")
            else:
                print("无标注可删除")

        # 播放状态自动前进
        if playing:
            current_frame = min(current_frame + 1, total_frames - 1)
            if current_frame >= total_frames - 1:
                playing = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频攻击标注工具")
    parser.add_argument("video", type=str, help="视频文件路径")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="标注输出路径 (JSON)")
    args = parser.parse_args()
    label_video(args.video, args.output)
