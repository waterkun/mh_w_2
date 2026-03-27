"""自动接任务 — 串联走到NPC + 选任务 + 失败重试.

使用方式:
  python -m farming_bot.start_mission.start_mission

流程:
  1. 回放摇杆路线走到 NPC
  2. 检测到交互提示 → 按 A 对话
  3. 回放选任务按键序列
  4. 检测加载黑屏 → 任务开始成功
  5. 失败 → 传送回营地 → 重试 (无限循环)
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
_ROUTE_DIR = os.path.join(_DIR, "recorded_route")
_ROUTE_PATH = os.path.join(_ROUTE_DIR, "route.json")
_RESET_PATH = os.path.join(_ROUTE_DIR, "reset_sequence.json")
_MISSION_SELECT_PATH = os.path.join(_ROUTE_DIR, "mission_select_sequence.json")
_TEMPLATE_PATH = os.path.join(_DIR, "templates", "npc_interact_prompt.png")
_MENU_TEMPLATE_PATH = os.path.join(_DIR, "templates", "menu_active_mission_selected.png")

# 模板匹配
MATCH_THRESHOLD = 0.75
MENU_MATCH_THRESHOLD = 0.80

# 加载画面检测
LOADING_BRIGHTNESS_THRESHOLD = 30
LOADING_VARIANCE_THRESHOLD = 500
LOADING_CONFIRM_FRAMES = 5  # 连续 N 帧黑屏确认

# 帧率
FPS = 10

# 按钮映射
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


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _is_loading_screen(frame):
    """检测是否为加载黑屏."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < LOADING_BRIGHTNESS_THRESHOLD and gray.var() < LOADING_VARIANCE_THRESHOLD


class StartMission:
    """自动接任务控制器."""

    def __init__(self, display_callback=None):
        self._display_callback = display_callback
        # 加载数据
        self._route = _load_json(_ROUTE_PATH)
        self._reset_seq = _load_json(_RESET_PATH)
        self._mission_seq = _load_json(_MISSION_SELECT_PATH)
        self._template = cv2.imread(_TEMPLATE_PATH)

        self._menu_template = cv2.imread(_MENU_TEMPLATE_PATH)

        if self._template is None:
            raise FileNotFoundError(f"模板不存在: {_TEMPLATE_PATH}")
        if self._menu_template is None:
            raise FileNotFoundError(f"菜单模板不存在: {_MENU_TEMPLATE_PATH}")

        # 初始化手柄
        self._pad = vg.VX360Gamepad()
        self._pad.update()
        time.sleep(0.3)

        # 截屏
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[1]

        self._frame_interval = 1.0 / FPS
        self._display_tick = 0

        print("StartMission 初始化完成")
        print(f"  路线: {self._route['duration_sec']}s, {self._route['data_points']} 点")
        print(f"  重置: {self._reset_seq['duration_sec']}s, {self._reset_seq['button_events']} 按键")
        print(f"  选任务: {self._mission_seq['duration_sec']}s, {self._mission_seq['button_events']} 按键")
        print(f"  模板: {self._template.shape[1]}x{self._template.shape[0]}")

    def _update_display(self, label="START_MISSION"):
        if self._display_callback:
            self._display_callback(label)

    def _grab_frame(self):
        """截屏并转 BGR."""
        screenshot = self._sct.grab(self._monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _press_button(self, name, duration=0.05):
        """按下并释放按钮."""
        if name == "LT":
            self._pad.left_trigger(value=255)
            self._pad.update()
            time.sleep(duration)
            self._pad.left_trigger(value=0)
            self._pad.update()
            return
        if name == "RT":
            self._pad.right_trigger(value=255)
            self._pad.update()
            time.sleep(duration)
            self._pad.right_trigger(value=0)
            self._pad.update()
            return
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
        """归零所有输入."""
        self._pad.reset()
        self._pad.update()

    def _detect_interact_prompt(self, frame):
        """模板匹配检测交互提示."""
        result = cv2.matchTemplate(frame, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= MATCH_THRESHOLD, max_val

    # ── 阶段 1: 走到 NPC ──

    def navigate_to_npc(self) -> bool:
        """回放摇杆路线走到 NPC.

        Returns:
            True 如果检测到交互提示 (到达 NPC).
        """
        self._update_display("NAVIGATE_TO_NPC")
        print("\n[阶段1] 走到 NPC...")
        route = self._route["route"]
        route_duration = self._route["duration_sec"]
        start_time = time.time()
        route_idx = 0

        while True:
            t0 = time.perf_counter()
            elapsed = time.time() - start_time

            # 路线时间用完 → 退出
            if elapsed > route_duration + 0.5:
                break

            # 推进到当前时间对应的数据点
            while route_idx < len(route) - 1 and route[route_idx + 1]["t"] <= elapsed:
                route_idx += 1

            # 设置摇杆
            point = route[route_idx]
            self._pad.left_joystick_float(
                x_value_float=point["lx"],
                y_value_float=point["ly"],
            )
            self._pad.update()

            # 检测交互提示
            frame = self._grab_frame()
            found, score = self._detect_interact_prompt(frame)

            if found:
                self._release_all()
                print(f"  检测到交互提示! score={score:.3f}")
                return True

            # 帧率控制
            dt = time.perf_counter() - t0
            sleep_time = self._frame_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 路线走完, 再额外等几秒检测 (可能角色还在移动)
        self._release_all()
        print("  路线走完, 额外等待检测...")
        wait_start = time.time()
        while time.time() - wait_start < 5.0:
            frame = self._grab_frame()
            found, score = self._detect_interact_prompt(frame)
            if found:
                print(f"  检测到交互提示! score={score:.3f}")
                return True
            time.sleep(self._frame_interval)

        print("  未检测到交互提示, 将重置")
        return False

    # ── 阶段 2: 选任务 ──

    def _is_active_mission_selected(self, frame):
        """检测左侧菜单是否处于 '活动任务' 选中状态.

        只对比左侧菜单条区域 (不含右侧内容面板),
        选中项有向右偏移 + 金色边框, 和模板差异明显.
        """
        # 裁剪左侧菜单条 (与模板相同 ROI)
        menu_roi = frame[200:780, 250:700]
        result = cv2.matchTemplate(menu_roi, self._menu_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= MENU_MATCH_THRESHOLD, max_val

    def select_mission(self) -> bool:
        """按 A 与 NPC 对话, 导航到活动任务并确认.

        策略: 按 A 进入菜单 → 按 DPAD_DOWN 逐个移动,
              每次截图对比菜单模板, 匹配则说明 '活动任务' 被选中 → 按 A 确认

        Returns:
            True 如果检测到加载画面 (任务开始).
        """
        self._update_display("SELECT_MISSION")
        print("\n[阶段2] 与 NPC 对话 + 选任务...")

        # 按 A 与 NPC 交互 (对话)
        self._press_button("A")
        print("  按 A 与 NPC 对话")
        time.sleep(1.5)

        # 按 A 三次进入任务分类列表
        for i in range(3):
            self._press_button("A")
            print(f"  按 A ({i+1}/3) 进入任务列表")
            time.sleep(1.0)

        # 检查当前是否已经选中活动任务
        frame = self._grab_frame()
        found, score = self._is_active_mission_selected(frame)
        if found:
            print(f"  活动任务已选中! score={score:.3f}")
        else:
            # 按 DPAD_DOWN 逐个移动, 每次检查是否选中
            print("  导航到活动任务...")
            max_attempts = 15  # 防止无限循环
            for i in range(max_attempts):
                self._press_button("DPAD_DOWN")
                time.sleep(0.4)

                frame = self._grab_frame()
                found, score = self._is_active_mission_selected(frame)
                print(f"  DPAD_DOWN ({i+1}) score={score:.3f}")

                if found:
                    print(f"  活动任务已选中! score={score:.3f}")
                    break
            else:
                print("  未能找到活动任务, 按 B 退出菜单")
                for i in range(4):
                    self._press_button("B")
                    time.sleep(0.3)
                return False

        time.sleep(0.3)

        # 按 A 进入活动任务列表
        self._press_button("A")
        print("  按 A 进入活动任务列表")
        time.sleep(1.0)

        # 按 A 选择第一个任务
        self._press_button("A")
        print("  按 A 选择任务")
        time.sleep(1.0)

        # 按 A 确认出发
        self._press_button("A")
        print("  按 A 确认出发")
        time.sleep(1.0)

        # 等待加载画面出现 (最多 30 秒)
        print("  等待加载画面...")
        loading_count = 0
        loading_started = False
        wait_start = time.time()

        while time.time() - wait_start < 60.0:
            frame = self._grab_frame()
            is_loading = _is_loading_screen(frame)

            if is_loading:
                loading_count += 1
                if not loading_started and loading_count >= LOADING_CONFIRM_FRAMES:
                    loading_started = True
                    print("  加载中...")
            else:
                if loading_started:
                    # 加载结束, 画面恢复
                    print("  加载完成! 已进入任务!")
                    time.sleep(3.0)  # 等画面稳定
                    return True
                loading_count = 0

            time.sleep(self._frame_interval)

        if loading_started:
            # 加载超时但确实进入了加载
            print("  加载超时, 但已检测到加载画面, 视为成功")
            return True

        print("  超时, 未检测到加载画面")
        return False

    # ── 重置: 传送回营地 ──

    def reset_to_camp(self):
        """回放传送回营地按键序列."""
        self._update_display("RESET_TO_CAMP")
        print("\n[重置] 传送回营地...")
        self._release_all()
        time.sleep(0.5)

        events = self._reset_seq["events"]
        button_events = [e for e in events if e["type"].startswith("button")]

        start_time = time.time()
        btn_idx = 0

        while btn_idx < len(button_events):
            elapsed = time.time() - start_time
            evt = button_events[btn_idx]

            if elapsed >= evt["t"]:
                name = evt["button"]
                if evt["type"] == "button_press":
                    btn = _BUTTON_MAP.get(name)
                    if name == "LT":
                        self._pad.left_trigger(value=255)
                    elif name == "RT":
                        self._pad.right_trigger(value=255)
                    elif btn:
                        self._pad.press_button(button=btn)
                    self._pad.update()
                    print(f"  {elapsed:.1f}s  PRESS {name}")
                elif evt["type"] == "button_release":
                    btn = _BUTTON_MAP.get(name)
                    if name == "LT":
                        self._pad.left_trigger(value=0)
                    elif name == "RT":
                        self._pad.right_trigger(value=0)
                    elif btn:
                        self._pad.release_button(button=btn)
                    self._pad.update()
                btn_idx += 1
            else:
                time.sleep(0.01)

        self._release_all()

        # 等加载完成 (传送有加载画面)
        print("  等待传送加载...")
        loading_started = False
        wait_start = time.time()

        while time.time() - wait_start < 30.0:
            frame = self._grab_frame()
            is_loading = _is_loading_screen(frame)

            if is_loading:
                loading_started = True
            elif loading_started:
                # 加载结束
                print("  传送完成!")
                time.sleep(1.0)  # 等画面稳定
                return

            time.sleep(self._frame_interval)

        print("  传送等待超时, 继续尝试...")
        time.sleep(2.0)

    # ── 主循环 ──

    def run(self) -> bool:
        """执行完整的接任务流程, 失败无限重试.

        Returns:
            True 当任务成功开始 (检测到加载画面).
        """
        attempt = 0
        first_run = True

        while True:
            attempt += 1
            print(f"\n{'='*50}")
            print(f"接任务尝试 #{attempt}")
            print(f"{'='*50}")

            # 首次启动等 3 秒 (给玩家切换窗口的时间)
            if first_run:
                print("  首次启动, 等待 3 秒...")
                time.sleep(3.0)
                first_run = False

            # 阶段 0: 传送回营地 (确保起始位置一致)
            self.reset_to_camp()

            # 阶段 1: 走到 NPC
            npc_found = self.navigate_to_npc()

            if npc_found:
                # 阶段 2: 选任务
                if self.select_mission():
                    beep(1500, 500)
                    print(f"\n任务开始成功! (第 {attempt} 次尝试)")
                    return True
                else:
                    print("  选任务失败, 将重置重试...")
            else:
                print("  未找到 NPC, 将重置重试...")

            # 失败 → 循环回去重新传送回营地
            beep(400, 300)
            print("  等待 2 秒后重试...")
            time.sleep(2.0)


def main():
    print("=" * 50)
    print("MH Wilds 自动接任务")
    print("=" * 50)
    print()

    # 倒计时
    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    beep(1200, 500)
    print()

    starter = StartMission()
    starter.run()


if __name__ == "__main__":
    main()
