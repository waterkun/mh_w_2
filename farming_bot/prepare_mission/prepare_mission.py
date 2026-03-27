"""回放任务准备操作 — 任务加载完成后执行录制的准备步骤.

使用方式:
  python -m farming_bot.prepare_mission.prepare_mission
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
_PREPARE_PATH = os.path.join(_DIR, "recorded_route", "prepare_sequence.json")
_ITEM_SLOT_BLANK_PATH = os.path.join(_DIR, "templates", "item_slot_blank.png")
_ITEM_BAR_BLANK_PATH = os.path.join(_DIR, "templates", "item_bar_blank.png")

# 道具槽空白检测
ITEM_BLANK_THRESHOLD = 0.80
ITEM_BLANK_TIMEOUT = 30.0  # 最多等 30 秒

_BUTTON_MAP = {
    "A": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    "B": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "X": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    "Y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    "DPAD_UP": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
    "DPAD_DOWN": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
    "DPAD_LEFT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
    "DPAD_RIGHT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
    "START": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
    "BACK": vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
    "LEFT_SHOULDER": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
    "RIGHT_SHOULDER": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
    "LEFT_THUMB": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
    "RIGHT_THUMB": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
}


def beep(freq=1000, duration_ms=200):
    try:
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass


class PrepareMission:
    """回放任务准备操作."""

    def __init__(self, pad=None, sct=None, monitor=None, display_callback=None):
        """初始化.

        Args:
            pad: 可复用的 VX360Gamepad 实例 (None 则新建).
            sct: 可复用的 mss 实例.
            monitor: 截屏区域.
            display_callback: 实时窗口更新回调 (label) -> None.
        """
        self._display_callback = display_callback
        if not os.path.exists(_PREPARE_PATH):
            raise FileNotFoundError(f"准备序列不存在: {_PREPARE_PATH}\n请先运行 record_prepare.py")

        with open(_PREPARE_PATH) as f:
            self._prepare_seq = json.load(f)

        self._item_slot_blank = cv2.imread(_ITEM_SLOT_BLANK_PATH)
        self._item_bar_blank = cv2.imread(_ITEM_BAR_BLANK_PATH)
        if self._item_slot_blank is None:
            raise FileNotFoundError(f"模板不存在: {_ITEM_SLOT_BLANK_PATH}")
        if self._item_bar_blank is None:
            raise FileNotFoundError(f"模板不存在: {_ITEM_BAR_BLANK_PATH}")

        if pad is None:
            self._pad = vg.VX360Gamepad()
            self._pad.update()
            time.sleep(0.3)
        else:
            self._pad = pad

        if sct is None:
            self._sct = mss.mss()
            self._monitor = self._sct.monitors[1]
        else:
            self._sct = sct
            self._monitor = monitor or self._sct.monitors[1]

        print("PrepareMission 初始化完成")
        print(f"  准备序列: {self._prepare_seq['duration_sec']}s, "
              f"{self._prepare_seq['button_events']} 按键")
        print(f"  空白道具槽模板: {self._item_slot_blank.shape[1]}x{self._item_slot_blank.shape[0]}")
        print(f"  空白道具栏模板: {self._item_bar_blank.shape[1]}x{self._item_bar_blank.shape[0]}")

    def _update_display(self, label="PREPARE"):
        if self._display_callback:
            self._display_callback(label)

    def _grab_frame(self):
        screenshot = self._sct.grab(self._monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _is_item_slot_blank(self, frame):
        """检测右下角道具槽是否为空白 (正常视角)."""
        roi = frame[1250:1430, 3150:3420]
        result = cv2.matchTemplate(roi, self._item_slot_blank, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= ITEM_BLANK_THRESHOLD, max_val

    def _is_item_bar_blank(self, frame):
        """检测 LB 道具栏中高亮格是否为空白."""
        roi = frame[1240:1430, 2850:3400]
        result = cv2.matchTemplate(roi, self._item_bar_blank, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= ITEM_BLANK_THRESHOLD, max_val

    def _release_all(self):
        self._pad.reset()
        self._pad.update()

    def _press_button(self, name, duration=0.15):
        btn = _BUTTON_MAP.get(name)
        if btn is None:
            return
        self._pad.press_button(button=btn)
        self._pad.update()
        time.sleep(duration)
        self._pad.release_button(button=btn)
        self._pad.update()

    def ensure_blank_item_slot(self) -> bool:
        """确保道具槽为空白. 不为空则按 LB+X 切换直到空白.

        Returns:
            True 如果道具槽已变为空白.
        """
        self._update_display("CHECK_ITEM_SLOT")
        print("\n[任务准备] 检查道具槽状态...")

        # 先检查正常视角是否已经空白
        frame = self._grab_frame()
        is_blank, score = self._is_item_slot_blank(frame)
        print(f"  道具槽检查 (正常) score={score:.3f}")
        if is_blank:
            print("  道具槽为空白, 可以开始准备!")
            return True

        # 不为空 → 按住 LB, 反复按 X 切换直到道具栏高亮格为空白
        print("  道具槽不为空, 按住 LB + 反复按 X 切换...")
        lb = _BUTTON_MAP["LEFT_SHOULDER"]
        x = _BUTTON_MAP["X"]
        self._pad.press_button(button=lb)
        self._pad.update()
        time.sleep(0.5)

        wait_start = time.time()
        while time.time() - wait_start < ITEM_BLANK_TIMEOUT:
            # 先检查当前是否已经空白
            frame = self._grab_frame()
            is_blank, score = self._is_item_bar_blank(frame)
            print(f"  道具栏检查 (LB) score={score:.3f}")

            if is_blank:
                self._pad.release_button(button=lb)
                self._pad.update()
                print("  道具栏为空白, 可以开始准备!")
                time.sleep(0.5)
                return True

            # 按 X 切换
            self._pad.press_button(button=x)
            self._pad.update()
            time.sleep(0.15)
            self._pad.release_button(button=x)
            self._pad.update()
            time.sleep(1.0)

        # 超时, 松开 LB
        self._pad.release_button(button=lb)
        self._pad.update()
        print("  超时, 道具槽未能切换到空白")
        return False

    def prepare(self) -> bool:
        """检查道具槽空白后回放准备操作.

        Returns:
            True 回放完成.
        """
        # 先确保道具槽为空白 (不为空则 LB+X 切换)
        if not self.ensure_blank_item_slot():
            print("  跳过准备 (道具槽无法切换到空白)")
            return False

        self._update_display("PREPARE_REPLAY")
        print("\n[任务准备] 开始回放...")

        events = self._prepare_seq["events"]
        start_time = time.time()
        evt_idx = 0

        try:
            while evt_idx < len(events):
                elapsed = time.time() - start_time
                evt = events[evt_idx]

                if elapsed >= evt["t"]:
                    if evt["type"] == "stick":
                        self._pad.left_joystick_float(
                            x_value_float=evt["lx"],
                            y_value_float=evt["ly"],
                        )
                        self._pad.right_joystick_float(
                            x_value_float=evt["rx"],
                            y_value_float=evt["ry"],
                        )
                        self._pad.update()

                    elif evt["type"] == "button_press":
                        name = evt["button"]
                        if name == "LT":
                            self._pad.left_trigger(value=255)
                        elif name == "RT":
                            self._pad.right_trigger(value=255)
                        else:
                            btn = _BUTTON_MAP.get(name)
                            if btn:
                                self._pad.press_button(button=btn)
                        self._pad.update()
                        print(f"  {elapsed:.1f}s  PRESS {name}")

                    elif evt["type"] == "button_release":
                        name = evt["button"]
                        if name == "LT":
                            self._pad.left_trigger(value=0)
                        elif name == "RT":
                            self._pad.right_trigger(value=0)
                        else:
                            btn = _BUTTON_MAP.get(name)
                            if btn:
                                self._pad.release_button(button=btn)
                        self._pad.update()

                    evt_idx += 1
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n键盘中断")

        self._release_all()
        elapsed = time.time() - start_time
        print(f"  准备完成! ({elapsed:.1f}s)")
        beep(1000, 300)
        return True


def main():
    print("=" * 50)
    print("MH Wilds 任务准备回放")
    print("=" * 50)
    print()

    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    beep(1200, 500)
    print()

    preparer = PrepareMission()
    preparer.prepare()


if __name__ == "__main__":
    main()
