"""放弃任务 — 从任务中返回并传送回营地.

使用方式:
  python -m farming_bot.abort_mission.abort_mission

流程:
  1. 按 START 打开菜单
  2. 按 LB/RB 切到 "使命/任务" 标签
  3. 按 DPAD_DOWN/UP 找到 "从任务中返回" (高亮)
  4. 按 A → LEFT → A 确认放弃
  5. 等待 10 秒
  6. 按 B 直到黑屏出现
  7. 等加载结束 + 5 秒稳定
  8. 传送回营地
"""

import ctypes
import os
import time

import cv2
import mss
import numpy as np
import vgamepad as vg
import winsound

_DIR = os.path.dirname(os.path.abspath(__file__))
_TAB_TEMPLATE_PATH = os.path.join(_DIR, "templates", "abort_tab.png")
_ITEM_TEMPLATE_PATH = os.path.join(_DIR, "templates", "abort_item_return.png")

# 模板匹配阈值
TAB_MATCH_THRESHOLD = 0.90
ITEM_MATCH_THRESHOLD = 0.75

# 加载画面检测
LOADING_BRIGHTNESS_THRESHOLD = 30
LOADING_VARIANCE_THRESHOLD = 500
LOADING_CONFIRM_FRAMES = 5

FPS = 10

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
}


def beep(freq=1000, duration_ms=200):
    try:
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass


def _press_key_escape():
    """用键盘模拟按 Escape (vgamepad START 对某些游戏无效)."""
    VK_ESCAPE = 0x1B
    KEYEVENTF_SCANCODE = 0x0008
    KEYEVENTF_KEYUP = 0x0002
    SC_ESCAPE = 0x01

    ctypes.windll.user32.keybd_event(VK_ESCAPE, SC_ESCAPE, KEYEVENTF_SCANCODE, 0)
    time.sleep(0.15)
    ctypes.windll.user32.keybd_event(VK_ESCAPE, SC_ESCAPE, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0)


def _is_loading_screen(frame):
    """检测是否为加载黑屏."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < LOADING_BRIGHTNESS_THRESHOLD and gray.var() < LOADING_VARIANCE_THRESHOLD


class AbortMission:
    """放弃当前任务并返回."""

    def __init__(self, pad=None, sct=None, monitor=None, display_callback=None):
        """初始化.

        Args:
            pad: 可复用的 VX360Gamepad 实例 (None 则新建).
            sct: 可复用的 mss 实例.
            monitor: 截屏区域.
            display_callback: 实时窗口更新回调 (label) -> None.
        """
        self._display_callback = display_callback
        self._tab_template = cv2.imread(_TAB_TEMPLATE_PATH)
        self._item_template = cv2.imread(_ITEM_TEMPLATE_PATH)
        if self._tab_template is None:
            raise FileNotFoundError(f"模板不存在: {_TAB_TEMPLATE_PATH}")
        if self._item_template is None:
            raise FileNotFoundError(f"模板不存在: {_ITEM_TEMPLATE_PATH}")

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

        self._frame_interval = 1.0 / FPS

        print("AbortMission 初始化完成")
        print(f"  Tab 模板: {self._tab_template.shape[1]}x{self._tab_template.shape[0]}")
        print(f"  Item 模板: {self._item_template.shape[1]}x{self._item_template.shape[0]}")

    def _update_display(self, label="ABORT"):
        if self._display_callback:
            self._display_callback(label)

    def _grab_frame(self):
        screenshot = self._sct.grab(self._monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _press_button(self, name, duration=0.15):
        btn = _BUTTON_MAP.get(name)
        if btn is None:
            print(f"  警告: 未知按钮 {name}")
            return
        self._pad.press_button(button=btn)
        self._pad.update()
        time.sleep(duration)
        self._pad.release_button(button=btn)
        self._pad.update()

    def _release_all(self):
        self._pad.reset()
        self._pad.update()

    def _is_tab_selected(self, frame):
        """检测是否在 '使命/任务' 标签."""
        tab_roi = frame[250:380, 50:550]
        result = cv2.matchTemplate(tab_roi, self._tab_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= TAB_MATCH_THRESHOLD, max_val

    def _is_return_item_highlighted(self, frame):
        """检测 '从任务中返回' 是否高亮选中."""
        item_roi = frame[400:800, 50:500]
        result = cv2.matchTemplate(item_roi, self._item_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= ITEM_MATCH_THRESHOLD, max_val

    def _wait_loading_end(self, timeout=30.0):
        """等待加载黑屏出现并结束."""
        loading_started = False
        loading_count = 0
        wait_start = time.time()

        while time.time() - wait_start < timeout:
            frame = self._grab_frame()
            is_loading = _is_loading_screen(frame)

            if is_loading:
                loading_count += 1
                if not loading_started and loading_count >= LOADING_CONFIRM_FRAMES:
                    loading_started = True
                    print("  加载中...")
            else:
                if loading_started:
                    print("  加载完成!")
                    return True
                loading_count = 0

            time.sleep(self._frame_interval)

        return loading_started  # 超时但可能已在加载

    def abort(self) -> bool:
        """执行放弃任务流程.

        Returns:
            True 如果成功放弃并回到据点.
        """
        self._update_display("ABORT_MISSION")
        print("\n[放弃任务] 开始...")

        # 1. 按 Escape 打开菜单 (vgamepad START 对游戏无效)
        _press_key_escape()
        print("  按 Escape 打开菜单")
        time.sleep(2.0)

        # 1.5 按 LB 重置高亮到第一个 tab
        self._press_button("LEFT_SHOULDER")
        print("  按 LB 重置 tab 位置")
        time.sleep(2.0)

        # 2. 按 RB 切到 "使命/任务" 标签 (循环检测)
        print("  切到 使命/任务 标签...")
        tab_found = False
        for i in range(8):
            frame = self._grab_frame()
            ready, score = self._is_tab_selected(frame)
            print(f"  Tab 检查 ({i}) score={score:.3f}")
            if ready:
                tab_found = True
                break
            self._press_button("RIGHT_SHOULDER")
            time.sleep(1.5)

        if not tab_found:
            print("  RB 未找到, 尝试 LB...")
            for i in range(8):
                self._press_button("LEFT_SHOULDER")
                time.sleep(1.5)
                frame = self._grab_frame()
                ready, score = self._is_tab_selected(frame)
                print(f"  LB ({i+1}) score={score:.3f}")
                if ready:
                    tab_found = True
                    break

        if not tab_found:
            print("  未找到 使命/任务 标签, 按 B 退出菜单")
            for _ in range(4):
                self._press_button("B")
                time.sleep(1.0)
            return False

        print("  使命/任务 标签已选中!")
        time.sleep(0.5)

        # 3. 按 DPAD_DOWN 3 次到 "从任务中返回" (第 4 项, RB/LB 会重置高亮到第 1 项)
        print("  导航到 从任务中返回 (DOWN x3)...")
        for i in range(3):
            self._press_button("DPAD_DOWN")
            print(f"  DPAD_DOWN ({i+1}/3)")
            time.sleep(1.0)

        # 3. 按 A 确认
        self._press_button("A")
        print("  按 A")
        time.sleep(1.0)

        # 4. 按 LEFT (选择确认选项)
        self._press_button("DPAD_LEFT")
        print("  按 LEFT")
        time.sleep(1.0)

        # 5. 按 A 最终确认
        self._press_button("A")
        print("  按 A 确认放弃")
        time.sleep(1.0)

        # 6. 等待 10 秒 (游戏处理中)
        print("  等待 10 秒...")
        time.sleep(10.0)

        # 7. 按 B 直到黑屏出现
        print("  按 B 等待黑屏...")
        b_start = time.time()
        while time.time() - b_start < 15.0:
            frame = self._grab_frame()
            if _is_loading_screen(frame):
                print("  检测到黑屏!")
                break
            self._press_button("B")
            time.sleep(0.5)

        # 8. 等加载结束 + 5 秒
        self._wait_loading_end(timeout=30.0)
        print("  等待 5 秒稳定...")
        time.sleep(5.0)

        self._release_all()
        print("  放弃任务完成!")
        beep(1000, 300)
        return True


def main():
    print("=" * 50)
    print("MH Wilds 放弃任务")
    print("=" * 50)
    print()

    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    beep(1200, 500)
    print()

    aborter = AbortMission()
    success = aborter.abort()

    if success:
        print("\n放弃任务成功!")
    else:
        print("\n放弃任务失败!")


if __name__ == "__main__":
    main()
