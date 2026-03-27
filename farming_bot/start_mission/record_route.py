"""录制手柄路线 — 记录左摇杆输入 + 截屏.

使用方式:
  python -m farming_bot.start_mission.record_route

流程:
  1. 倒计时 5 秒 → 切到游戏
  2. 蜂鸣提示开始录制
  3. 左摇杆移动角色走向 NPC
  4. 到达 NPC 后按手柄 B 键停止录制
  5. 保存路线数据 + 最后一帧截图 (NPC 交互提示模板)
"""

import json
import os
import time

import cv2
import mss
import numpy as np
import winsound

try:
    import XInput
except ImportError:
    print("需要安装 XInput-Python: pip install XInput-Python")
    raise

_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorded_route")

# 摇杆死区
DEADZONE = 0.15

# 录制帧率
RECORD_FPS = 10


def beep(freq=1000, duration_ms=200):
    """播放提示音."""
    try:
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass


def record():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    sct = mss.mss()
    monitor = sct.monitors[1]

    print("=" * 50)
    print("路线录制工具")
    print("=" * 50)
    print()
    print("操作说明:")
    print("  - 倒计时结束后开始录制")
    print("  - 用左摇杆走到任务 NPC")
    print("  - 到达 NPC 后按手柄 B 键停止录制")
    print()

    # 倒计时
    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始录制... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    # 开始录制
    beep(1200, 500)
    print()
    print(">>> 录制开始! 用左摇杆走到 NPC，按 B 停止 <<<")
    print()

    route = []  # [(timestamp, lx, ly), ...]
    frame_interval = 1.0 / RECORD_FPS
    start_time = time.time()
    last_frame = None
    frame_count = 0

    try:
        while True:
            t0 = time.perf_counter()

            # 读取手柄状态
            state = XInput.get_state(0)
            lx, ly = XInput.get_thumb_values(state)[0]  # left thumb (x, y)
            buttons = XInput.get_button_values(state)

            # 死区处理
            if abs(lx) < DEADZONE:
                lx = 0.0
            if abs(ly) < DEADZONE:
                ly = 0.0

            # 记录摇杆数据
            elapsed = time.time() - start_time
            route.append({
                "t": round(elapsed, 3),
                "lx": round(lx, 4),
                "ly": round(ly, 4),
            })

            # 截屏
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            last_frame = frame.copy()
            frame_count += 1

            # 状态显示 (每秒一次)
            if frame_count % RECORD_FPS == 0:
                print(f"  [{elapsed:.1f}s] LX={lx:+.2f} LY={ly:+.2f}  "
                      f"frames={frame_count}")

            # B 键停止
            if buttons.get("B", False):
                beep(600, 300)
                print()
                print(">>> B 键按下, 停止录制 <<<")
                break

            # 帧率控制
            dt = time.perf_counter() - t0
            sleep_time = frame_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n键盘中断, 停止录制")

    # 保存
    elapsed = time.time() - start_time
    print(f"\n录制完成: {elapsed:.1f}s, {len(route)} 个数据点, {frame_count} 帧")

    # 保存路线数据
    route_path = os.path.join(_OUTPUT_DIR, "route.json")
    with open(route_path, "w") as f:
        json.dump({
            "duration_sec": round(elapsed, 2),
            "fps": RECORD_FPS,
            "data_points": len(route),
            "route": route,
        }, f, indent=2)
    print(f"路线保存: {route_path}")

    # 保存最后一帧 (NPC 交互提示截图)
    if last_frame is not None:
        last_frame_path = os.path.join(_OUTPUT_DIR, "npc_arrival_frame.png")
        cv2.imwrite(last_frame_path, last_frame)
        print(f"NPC 到达截图: {last_frame_path}")
        print()
        print("下一步: 从这张截图中裁剪出交互提示图标,")
        print("        保存为 templates/npc_interact_prompt.png")

    print("\n完成!")


if __name__ == "__main__":
    record()
