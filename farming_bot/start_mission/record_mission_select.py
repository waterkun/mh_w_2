"""录制选任务操作 — NPC 对话后到出发的按键序列.

使用方式:
  python -m farming_bot.start_mission.record_mission_select

流程:
  1. 倒计时 10 秒 → 切到游戏 (先手动走到 NPC 面前并按 A 对话)
  2. 录制选任务 + 确认出发的所有按键
  3. 加载开始或操作完成后长按 B 停止
"""

import json
import os
import time

import winsound

try:
    import XInput
except ImportError:
    print("需要安装 XInput-Python: pip install XInput-Python")
    raise

_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorded_route")

RECORD_FPS = 10
DEADZONE = 0.15

_BUTTON_NAMES = [
    "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
    "START", "BACK",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "A", "B", "X", "Y",
]


def beep(freq=1000, duration_ms=200):
    try:
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass


def record():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    print("=" * 50)
    print("选任务录制工具")
    print("=" * 50)
    print()
    print("操作说明:")
    print("  - 先手动走到 NPC 面前并按 A 开始对话")
    print("  - 倒计时结束后开始录制")
    print("  - 录制选任务 → 确认出发的所有按键")
    print("  - 完成后长按 B (0.8s) 停止录制")
    print()

    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始录制... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    beep(1200, 500)
    print()
    print(">>> 录制开始! 选任务并确认出发，完成后长按 B 停止 <<<")
    print()

    events = []
    frame_interval = 1.0 / RECORD_FPS
    start_time = time.time()
    frame_count = 0
    prev_buttons = {}
    prev_triggers = {"LT": 0, "RT": 0}
    b_hold_start = None
    B_HOLD_THRESHOLD = 0.8

    try:
        while True:
            t0 = time.perf_counter()
            elapsed = time.time() - start_time

            state = XInput.get_state(0)

            (lx, ly), (rx, ry) = XInput.get_thumb_values(state)
            if abs(lx) < DEADZONE: lx = 0.0
            if abs(ly) < DEADZONE: ly = 0.0
            if abs(rx) < DEADZONE: rx = 0.0
            if abs(ry) < DEADZONE: ry = 0.0

            events.append({
                "t": round(elapsed, 3),
                "type": "stick",
                "lx": round(lx, 4), "ly": round(ly, 4),
                "rx": round(rx, 4), "ry": round(ry, 4),
            })

            buttons = XInput.get_button_values(state)
            for btn_name in _BUTTON_NAMES:
                curr = buttons.get(btn_name, False)
                prev = prev_buttons.get(btn_name, False)
                if curr and not prev:
                    events.append({"t": round(elapsed, 3), "type": "button_press", "button": btn_name})
                elif not curr and prev:
                    events.append({"t": round(elapsed, 3), "type": "button_release", "button": btn_name})
                prev_buttons[btn_name] = curr

            lt, rt = XInput.get_trigger_values(state)
            lt_pressed = lt > 100
            rt_pressed = rt > 100
            lt_was = prev_triggers["LT"] > 100
            rt_was = prev_triggers["RT"] > 100
            if lt_pressed and not lt_was:
                events.append({"t": round(elapsed, 3), "type": "button_press", "button": "LT"})
            elif not lt_pressed and lt_was:
                events.append({"t": round(elapsed, 3), "type": "button_release", "button": "LT"})
            if rt_pressed and not rt_was:
                events.append({"t": round(elapsed, 3), "type": "button_press", "button": "RT"})
            elif not rt_pressed and rt_was:
                events.append({"t": round(elapsed, 3), "type": "button_release", "button": "RT"})
            prev_triggers["LT"] = lt
            prev_triggers["RT"] = rt

            b_curr = buttons.get("B", False)
            if b_curr:
                if b_hold_start is None:
                    b_hold_start = time.time()
                elif time.time() - b_hold_start >= B_HOLD_THRESHOLD:
                    beep(600, 300)
                    print(f"\n>>> B 长按 {B_HOLD_THRESHOLD}s, 停止录制 <<<")
                    events = [e for e in events
                              if not (e.get("button") == "B"
                                      and e["t"] >= round(b_hold_start - start_time, 3))]
                    break
            else:
                b_hold_start = None

            frame_count += 1
            if frame_count % RECORD_FPS == 0:
                recent = [e for e in events[-20:] if e["type"].startswith("button")]
                btn_info = ", ".join(f"{e['button']}({'P' if 'press' in e['type'] else 'R'})"
                                     for e in recent[-3:])
                print(f"  [{elapsed:.1f}s] events={len(events)}  {btn_info}")

            dt = time.perf_counter() - t0
            sleep_time = frame_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n键盘中断")

    elapsed = time.time() - start_time
    button_events = [e for e in events if e["type"].startswith("button")]
    print(f"\n录制完成: {elapsed:.1f}s, {len(events)} 事件 ({len(button_events)} 按键)")

    save_path = os.path.join(_OUTPUT_DIR, "mission_select_sequence.json")
    with open(save_path, "w") as f:
        json.dump({
            "duration_sec": round(elapsed, 2),
            "fps": RECORD_FPS,
            "total_events": len(events),
            "button_events": len(button_events),
            "events": events,
        }, f, indent=2)
    print(f"保存: {save_path}")
    print("\n完成!")


if __name__ == "__main__":
    record()
