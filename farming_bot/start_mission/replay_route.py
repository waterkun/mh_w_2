"""回放录制路线 — 自动走到 NPC 并按 A 交互.

使用方式:
  python -m farming_bot.start_mission.replay_route

流程:
  1. 倒计时 5 秒 → 切到游戏
  2. 回放 route.json 中的摇杆输入
  3. 每帧模板匹配检测交互提示 (绿色 A 图标)
  4. 检测到 → 停止移动 + 按 A 交互
  5. 路线走完仍未检测到 → 提示失败 (后续接传送回营地重试)
"""

import json
import os
import time

import cv2
import mss
import numpy as np
import vgamepad as vg
import winsound

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROUTE_PATH = os.path.join(_DIR, "recorded_route", "route.json")
_TEMPLATE_PATH = os.path.join(_DIR, "templates", "npc_interact_prompt.png")

# 模板匹配阈值
MATCH_THRESHOLD = 0.75

# 检测帧率
CHECK_FPS = 10


def beep(freq=1000, duration_ms=200):
    try:
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass


def replay():
    # 加载路线
    if not os.path.exists(_ROUTE_PATH):
        print(f"错误: 路线文件不存在 {_ROUTE_PATH}")
        print("请先运行 record_route.py 录制路线")
        return False

    with open(_ROUTE_PATH) as f:
        data = json.load(f)
    route = data["route"]
    print(f"路线: {data['duration_sec']}s, {data['data_points']} 个数据点")

    # 加载模板
    if not os.path.exists(_TEMPLATE_PATH):
        print(f"错误: 模板文件不存在 {_TEMPLATE_PATH}")
        return False

    template = cv2.imread(_TEMPLATE_PATH)
    tmpl_h, tmpl_w = template.shape[:2]
    print(f"交互提示模板: {tmpl_w}x{tmpl_h}")

    # 初始化手柄
    pad = vg.VX360Gamepad()
    pad.update()
    time.sleep(0.3)

    # 初始化截屏
    sct = mss.mss()
    monitor = sct.monitors[1]

    print()
    print("=" * 50)
    print("路线回放")
    print("=" * 50)

    # 倒计时
    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    beep(1200, 500)
    print()
    print(">>> 回放开始 <<<")

    start_time = time.time()
    route_idx = 0
    found_npc = False
    frame_count = 0
    frame_interval = 1.0 / CHECK_FPS
    route_duration = data["duration_sec"]

    try:
        while True:
            t0 = time.perf_counter()
            elapsed = time.time() - start_time

            # 路线时间用完 → 退出
            if elapsed > route_duration + 0.5:
                break

            # 找到当前时间对应的摇杆数据点
            while route_idx < len(route) - 1 and route[route_idx + 1]["t"] <= elapsed:
                route_idx += 1

            # 设置摇杆
            point = route[route_idx]
            lx = point["lx"]
            ly = point["ly"]
            pad.left_joystick_float(x_value_float=lx, y_value_float=ly)
            pad.update()

            # 截屏 + 模板匹配
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            frame_count += 1
            if frame_count % CHECK_FPS == 0:
                print(f"  [{elapsed:.1f}s] LX={lx:+.2f} LY={ly:+.2f}  "
                      f"match={max_val:.3f}  idx={route_idx}/{len(route)}")

            # 检测到交互提示
            if max_val >= MATCH_THRESHOLD:
                print(f"\n>>> 检测到交互提示! score={max_val:.3f} at {max_loc} <<<")
                found_npc = True
                break

            # 帧率控制
            dt = time.perf_counter() - t0
            sleep_time = frame_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n键盘中断")

    # 停止移动
    pad.left_joystick_float(x_value_float=0, y_value_float=0)
    pad.update()

    if found_npc:
        beep(1500, 300)
        print("停止移动, 按 A 交互...")
        time.sleep(0.3)

        # 按 A
        btn_a = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
        pad.press_button(button=btn_a)
        pad.update()
        time.sleep(0.05)
        pad.release_button(button=btn_a)
        pad.update()

        print("已按 A, 交互成功!")
        beep(1500, 500)
        return True
    else:
        elapsed = time.time() - start_time
        print(f"\n路线走完 ({elapsed:.1f}s) 但未检测到交互提示")
        beep(400, 500)
        return False


if __name__ == "__main__":
    success = replay()
    if success:
        print("\n完成! NPC 交互成功")
    else:
        print("\n失败! 未到达 NPC (后续将接入传送回营地重试)")
