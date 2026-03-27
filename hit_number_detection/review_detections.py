"""
frames_filtered 检测结果审查 & 修正工具

两步流程:
  Step 1 (--run-inference): 批量跑 YOLO+CRNN，保存检测结果到 JSON
  Step 2 (默认):           交互式审查，可修正 bbox 和数字

操作:
  A / D         当前选中 bbox 左边缘 左移/右移
  ← / →         当前选中 bbox 右边缘 左移/右移
  W / S         当前选中 bbox 上边缘 上移/下移
  ↑ / ↓         当前选中 bbox 下边缘 上移/下移
  Tab           切换选中的 bbox (多个检测时)
  T             修改当前 bbox 的数字文本 (在终端输入)
  Delete/X      删除当前选中的 bbox
  B             新增一个 bbox (鼠标框选模式)
  R             重置当前帧所有修改
  Enter/Space   确认当前 bbox 并切到下一个，最后一个确认后跳到随机未审查帧
  Backspace     回到上一个审查过的帧
  F             快进: 跳过相似帧 (跳到数字组合不同的下一帧)
  N / P         跳 10 帧 (前进/后退)
  G             跳转到指定帧号
  +/- or =/     缩放 (1x~6x)
  Z             快速切换 1x / 4x 居中到选中 bbox
  Q             保存并退出 (进度自动保存，下次继续)

Usage:
  # Step 1: 批量推理
  E:/anaconda3/envs/mh_ai/python.exe hit_number_detection/review_detections.py --run-inference

  # Step 2: 交互审查
  E:/anaconda3/envs/mh_ai/python.exe hit_number_detection/review_detections.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
FRAMES_DIR = BASE / "frames_filtered"
OUTPUT_DIR = BASE / "detection_review"
DETECTIONS_FILE = OUTPUT_DIR / "detections.json"
PROGRESS_FILE = OUTPUT_DIR / "review_progress.json"

YOLO_PATH = BASE / "runs" / "detect" / "runs" / "damage_number" / "weights" / "best.pt"
CRNN_PATH = BASE / "runs" / "crnn" / "best.pt"


# ---------------------------------------------------------------------------
# Step 1: 批量推理
# ---------------------------------------------------------------------------

def run_batch_inference(conf=0.5):
    """对 frames_filtered 图片跑 YOLO+CRNN，保留已审查帧，只重跑未审查帧"""
    sys.path.insert(0, str(BASE))
    from inference import DamageNumberDetector

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = DamageNumberDetector(
        yolo_path=str(YOLO_PATH),
        crnn_path=str(CRNN_PATH),
        conf_threshold=conf,
    )

    # 加载已有结果和审查进度
    existing = {}
    if DETECTIONS_FILE.exists():
        with open(DETECTIONS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    reviewed_names = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
        reviewed_names = set(progress.get("reviewed_frames", []))
        # 向后兼容旧格式
        if not reviewed_names and progress.get("last_idx", 0) > 0:
            all_sorted = sorted(existing.keys())
            reviewed_names = set(all_sorted[:progress["last_idx"] + 1])

    print(f"已审查帧: {len(reviewed_names)} 帧 (将保留)")

    frames = sorted(FRAMES_DIR.glob("*.png"))
    todo = [f for f in frames if f.name not in reviewed_names]
    print(f"共 {len(frames)} 张图片, 需推理: {len(todo)} 张")

    # 保留已审查帧的数据
    all_detections = {k: v for k, v in existing.items() if k in reviewed_names}
    new_count = 0

    for i, frame_path in enumerate(todo):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        dets = detector.detect(img)
        if dets:
            all_detections[frame_path.name] = {
                "detections": [
                    {
                        "bbox": [int(x) for x in d["bbox"]],
                        "number": d["number"],
                        "confidence": round(float(d["confidence"]), 4),
                    }
                    for d in dets
                ]
            }
            new_count += 1

        if (i + 1) % 200 == 0 or i == len(todo) - 1:
            print(f"  [{i + 1}/{len(todo)}] 新检测 {new_count} 张有结果")

    with open(DETECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)

    # 重置审查进度的 last_idx (已审查的帧排在前面，新帧追加在后面)
    # 不改 last_idx，下次审查从之前位置继续
    total_dets = sum(len(v["detections"]) for v in all_detections.values())
    print(f"\n完成！{len(all_detections)} 张图有检测，共 {total_dets} 个 bbox")
    print(f"  其中已审查: {len(reviewed_names)} 帧 (已保留)")
    print(f"  新推理: {new_count} 帧有检测")
    print(f"结果保存: {DETECTIONS_FILE}")


# ---------------------------------------------------------------------------
# Step 2: 交互审查
# ---------------------------------------------------------------------------

class ReviewTool:
    ZOOM_LEVELS = [1, 2, 3, 4, 6]  # 缩放倍率
    STEP = 2  # bbox 移动步长 (像素)
    INFO_H = 50
    CROP_H = 120  # 底部预览默认高度
    CROP_SCALE_MIN = 3.0  # crop 最小放大倍率

    def __init__(self):
        self.detections = self._load_detections()
        self.frame_names = sorted(self.detections.keys())
        self.progress = self._load_progress()
        self.current_idx = self.progress.get("last_idx", 0)
        self.selected_bbox = 0
        # 向后兼容: 从旧 progress 迁移
        old_reviewed = set(self.progress.get("reviewed", []))
        # 旧的顺序审查帧 (0 ~ last_idx)
        if "reviewed_frames" not in self.progress and self.progress.get("last_idx", 0) > 0:
            old_sequential = set(self.frame_names[:self.progress["last_idx"] + 1])
            old_reviewed |= old_sequential
        self.reviewed_frames = set(self.progress.get("reviewed_frames", [])) | old_reviewed
        self.review_history = []  # 审查历史栈，用于 Backspace 回退
        self.modified = set()
        self.drawing = False
        self.draw_start = None
        self.draw_end = None
        self.zoom_idx = 0  # 当前缩放级别索引
        self.pan_x = 0     # 缩放时的平移偏移 (像素, 原图坐标)
        self.pan_y = 0

    @property
    def zoom(self):
        return self.ZOOM_LEVELS[self.zoom_idx]

    def _load_detections(self):
        if not DETECTIONS_FILE.exists():
            print(f"错误: 先运行 --run-inference 生成 {DETECTIONS_FILE}")
            sys.exit(1)
        with open(DETECTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_progress(self):
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_progress(self):
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump({
                "last_idx": self.current_idx,
                "reviewed_frames": sorted(self.reviewed_frames),
            }, f)

    def _save_detections(self):
        with open(DETECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.detections, f, indent=2, ensure_ascii=False)
        print(f"检测结果已保存 ({len(self.modified)} 帧有修改)")

    def _get_frame_data(self):
        name = self.frame_names[self.current_idx]
        return name, self.detections[name]

    def _reset_view(self):
        """切换帧时重置视图状态"""
        self.selected_bbox = 0
        self.zoom_idx = 0
        self.pan_x = 0
        self.pan_y = 0

    def _get_signature(self, idx):
        """获取帧的数字签名 (用于判断相似性)"""
        if idx < 0 or idx >= len(self.frame_names):
            return None
        name = self.frame_names[idx]
        data = self.detections.get(name)
        if not data:
            return None
        nums = sorted(d["number"] for d in data["detections"])
        return tuple(nums)

    def _skip_similar(self):
        """快进到下一个不同数字组合的帧"""
        current_sig = self._get_signature(self.current_idx)
        start = self.current_idx
        while self.current_idx < len(self.frame_names) - 1:
            self.current_idx += 1
            if self._get_signature(self.current_idx) != current_sig:
                break
        skipped = self.current_idx - start
        self._reset_view()
        print(f"  跳过 {skipped} 帧相似帧 -> 第 {self.current_idx + 1} 帧")

    def _goto_random_unreviewed(self):
        unreviewed = [i for i, name in enumerate(self.frame_names)
                      if name not in self.reviewed_frames]
        if not unreviewed:
            print("所有帧已审查完毕!")
            return
        self.current_idx = random.choice(unreviewed)
        self._reset_view()
        print(f"  -> 随机帧 [{self.current_idx + 1}/{len(self.frame_names)}] "
              f"(剩余 {len(unreviewed) - 1} 未审查)")

    def _center_on_bbox(self, bbox, img_w, img_h):
        """将视图居中到选中的 bbox"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.pan_x = cx - img_w // (2 * self.zoom)
        self.pan_y = cy - img_h // (2 * self.zoom)
        # 限制范围
        self._clamp_pan(img_w, img_h)

    def _clamp_pan(self, img_w, img_h):
        """限制平移范围"""
        z = self.zoom
        max_pan_x = max(0, img_w - img_w // z)
        max_pan_y = max(0, img_h - img_h // z)
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))

    def _screen_to_img(self, sx, sy, img_w, img_h):
        """屏幕坐标 -> 原图坐标 (考虑 info_bar 偏移和缩放)"""
        sy -= self.INFO_H  # 减去 info bar
        z = self.zoom
        ix = int(sx / z + self.pan_x)
        iy = int(sy / z + self.pan_y)
        return ix, iy

    def _input_in_window(self, window_name, current_view, prompt, digits_only=False):
        """在 OpenCV 窗口内输入文本，不切换到终端。
        返回输入的字符串，Esc 取消返回 None。"""
        buf = ""
        h, w = current_view.shape[:2]
        while True:
            overlay = current_view.copy()
            # 半透明黑色遮罩
            mask = np.zeros_like(overlay)
            cv2.addWeighted(overlay, 0.4, mask, 0.6, 0, overlay)
            # 输入框
            box_w, box_h = 400, 80
            bx = (w - box_w) // 2
            by = (h - box_h) // 2
            cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (60, 60, 60), -1)
            cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (200, 200, 200), 2)
            # 提示文字
            cv2.putText(overlay, prompt, (bx + 10, by + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            # 输入内容 + 光标
            display = buf + "|"
            cv2.putText(overlay, display, (bx + 10, by + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # 底部提示
            cv2.putText(overlay, "Enter=OK  Esc=Cancel", (bx + 100, by + box_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
            cv2.imshow(window_name, overlay)
            k = cv2.waitKeyEx(0)
            if k == 13:  # Enter
                return buf
            elif k == 27:  # Esc
                return None
            elif k == 8 or k == 3342336:  # Backspace
                buf = buf[:-1]
            elif 0 <= k < 128:
                ch = chr(k)
                if digits_only:
                    if ch.isdigit():
                        buf += ch
                else:
                    if ch.isprintable():
                        buf += ch

    def _draw_view(self, img, frame_name, data):
        """绘制审查视图 (支持缩放)"""
        img_h, img_w = img.shape[:2]
        dets = data["detections"]
        z = self.zoom

        # 绘制 bbox 到原图副本
        canvas = img.copy()
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["bbox"]
            is_selected = (i == self.selected_bbox)
            color = (0, 255, 0) if is_selected else (0, 165, 255)
            thickness = max(1, 3 // z) if not is_selected else max(2, 4 // z)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            # 标签
            font_scale = max(0.3, 0.6 / z * 2)
            label = f"[{i}] {d['number']} ({d['confidence']:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_y = max(y1 - 3, th + 3)
            cv2.rectangle(canvas, (x1, label_y - th - 3), (x1 + tw + 2, label_y + 1), color, -1)
            cv2.putText(canvas, label, (x1 + 1, label_y - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

        # 画正在框选的新 bbox
        if self.drawing and self.draw_start and self.draw_end:
            cv2.rectangle(canvas, self.draw_start, self.draw_end, (255, 0, 255), max(1, 2 // z))

        # 裁剪可视区域 (缩放)
        view_w = img_w // z
        view_h = img_h // z
        px, py = int(self.pan_x), int(self.pan_y)
        # 确保不越界
        px = max(0, min(px, img_w - view_w))
        py = max(0, min(py, img_h - view_h))
        visible = canvas[py:py + view_h, px:px + view_w]

        # 放大到原始尺寸
        if z > 1:
            visible = cv2.resize(visible, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # 顶部信息栏
        info_bar = np.zeros((self.INFO_H, img_w, 3), dtype=np.uint8)
        text1 = f"[{self.current_idx + 1}/{len(self.frame_names)}] {frame_name}  |  {len(dets)} bbox  |  sel:[{self.selected_bbox}]  |  zoom: {z}x  |  reviewed: {len(self.reviewed_frames)}"
        cv2.putText(info_bar, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        text2 = "ADWS=edges  arrows=edges  Tab=switch  T=text  X=del  B=new  +/-=zoom  Z=toggle  F=skip  N/P=+-10  G=goto  Q=quit"
        cv2.putText(info_bar, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # 底部: 放大的选中 crop (自适应高度)
        crop_bar_h = self.CROP_H
        crop_bar = np.zeros((crop_bar_h, img_w, 3), dtype=np.uint8)
        if dets and 0 <= self.selected_bbox < len(dets):
            d = dets[self.selected_bbox]
            x1, y1, x2, y2 = d["bbox"]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(img_w, x2), min(img_h, y2)
            if x2c > x1c and y2c > y1c:
                crop = img[y1c:y2c, x1c:x2c]
                # 放大: 至少 CROP_SCALE_MIN 倍，但不超过预览区
                max_crop_w = img_w * 2 // 3  # crop 最多占 2/3 宽度
                crop_scale = max(self.CROP_SCALE_MIN,
                                 min((crop_bar_h - 10) / max(crop.shape[0], 1),
                                     max_crop_w / max(crop.shape[1], 1)))
                crop_resized = cv2.resize(crop, None, fx=crop_scale, fy=crop_scale,
                                          interpolation=cv2.INTER_NEAREST)
                ch, cw = crop_resized.shape[:2]
                # 如果 crop 放大后超过预览区高度，扩展预览区
                needed_h = ch + 10
                if needed_h > crop_bar_h:
                    crop_bar_h = needed_h
                    crop_bar = np.zeros((crop_bar_h, img_w, 3), dtype=np.uint8)
                # 放置 crop
                ch = min(ch, crop_bar_h - 5)
                cw = min(cw, img_w - 20)
                crop_bar[5:5 + ch, 10:10 + cw] = crop_resized[:ch, :cw]
                # 数字和坐标信息 (放在 crop 右侧)
                info_x = min(cw + 20, img_w - 350)
                cv2.putText(crop_bar, f"'{d['number']}'",
                            (info_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(crop_bar, f"bbox: {x2 - x1}x{y2 - y1}px  conf: {d['confidence']:.2f}",
                            (info_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(crop_bar, f"({x1},{y1})-({x2},{y2})",
                            (info_x, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return np.vstack([info_bar, visible, crop_bar])

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调: 新增 bbox 框选 (坐标转换到原图)"""
        if not self.drawing:
            return

        img = param
        if img is None:
            return
        img_h, img_w = img.shape[:2]
        ix, iy = self._screen_to_img(x, y, img_w, img_h)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw_start = (ix, iy)
            self.draw_end = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.draw_start:
            self.draw_end = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and self.draw_start:
            self.draw_end = (ix, iy)
            self.drawing = False
            # 添加新 bbox
            x1 = max(0, min(self.draw_start[0], self.draw_end[0]))
            y1 = max(0, min(self.draw_start[1], self.draw_end[1]))
            x2 = min(img_w, max(self.draw_start[0], self.draw_end[0]))
            y2 = min(img_h, max(self.draw_start[1], self.draw_end[1]))
            if x2 - x1 > 3 and y2 - y1 > 3:
                name, data = self._get_frame_data()
                data["detections"].append({
                    "bbox": [x1, y1, x2, y2],
                    "number": "???",
                    "confidence": 0.0,
                })
                self.selected_bbox = len(data["detections"]) - 1
                self.modified.add(name)
                print(f"  新增 bbox [{self.selected_bbox}] @ ({x1},{y1},{x2},{y2}) — 按 T 输入数字")
            self.draw_start = None
            self.draw_end = None

    def run(self):
        if not self.frame_names:
            print("没有检测结果可审查")
            return

        # 如果当前帧已审查，跳到随机未审查帧
        if self.frame_names[self.current_idx] in self.reviewed_frames:
            self._goto_random_unreviewed()

        print(f"共 {len(self.frame_names)} 帧有检测，从第 {self.current_idx + 1} 帧开始")

        window_name = "Detection Review"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 900)

        current_img = None

        while 0 <= self.current_idx < len(self.frame_names):
            frame_name, data = self._get_frame_data()
            dets = data["detections"]

            img_path = FRAMES_DIR / frame_name
            current_img = cv2.imread(str(img_path))
            if current_img is None:
                self.current_idx += 1
                continue

            img_h, img_w = current_img.shape[:2]

            # 确保 selected_bbox 有效
            if dets:
                self.selected_bbox = min(self.selected_bbox, len(dets) - 1)
                self.selected_bbox = max(self.selected_bbox, 0)

            # 设置鼠标回调，传入当前图片引用
            cv2.setMouseCallback(window_name, self._mouse_callback, current_img)

            current_view = self._draw_view(current_img, frame_name, data)
            cv2.imshow(window_name, current_view)
            key = cv2.waitKeyEx(0)

            if not dets and key not in [ord('b'), ord('B'), ord('q'), ord('Q'), 8,
                                         ord('+'), ord('='), ord('-'), ord('_')]:
                # 无 bbox 的帧自动跳
                self.current_idx += 1
                continue

            sel = self.selected_bbox
            bbox = dets[sel]["bbox"] if dets and sel < len(dets) else None
            step = self.STEP

            # A: 左边缘左移
            if key == ord('a') or key == ord('A'):
                if bbox:
                    bbox[0] = max(0, bbox[0] - step)
                    self.modified.add(frame_name)
            # D: 左边缘右移
            elif key == ord('d') or key == ord('D'):
                if bbox:
                    bbox[0] = min(bbox[2] - 1, bbox[0] + step)
                    self.modified.add(frame_name)
            # ← 右边缘左移
            elif key == 2424832 or key == 81:
                if bbox:
                    bbox[2] = max(bbox[0] + 1, bbox[2] - step)
                    self.modified.add(frame_name)
            # → 右边缘右移
            elif key == 2555904 or key == 83:
                if bbox:
                    bbox[2] = min(img_w, bbox[2] + step)
                    self.modified.add(frame_name)
            # W: 上边缘上移
            elif key == ord('w') or key == ord('W'):
                if bbox:
                    bbox[1] = max(0, bbox[1] - step)
                    self.modified.add(frame_name)
            # S: 上边缘下移
            elif key == ord('s') or key == ord('S'):
                if bbox:
                    bbox[1] = min(bbox[3] - 1, bbox[1] + step)
                    self.modified.add(frame_name)
            # ↑ 下边缘上移
            elif key == 2490368 or key == 82:
                if bbox:
                    bbox[3] = max(bbox[1] + 1, bbox[3] - step)
                    self.modified.add(frame_name)
            # ↓ 下边缘下移
            elif key == 2621440 or key == 84:
                if bbox:
                    bbox[3] = min(img_h, bbox[3] + step)
                    self.modified.add(frame_name)
            # + / =: 放大
            elif key == ord('+') or key == ord('='):
                if self.zoom_idx < len(self.ZOOM_LEVELS) - 1:
                    self.zoom_idx += 1
                    # 居中到选中 bbox
                    if bbox:
                        self._center_on_bbox(bbox, img_w, img_h)
                    print(f"  缩放: {self.zoom}x")
            # - / _: 缩小
            elif key == ord('-') or key == ord('_'):
                if self.zoom_idx > 0:
                    self.zoom_idx -= 1
                    self._clamp_pan(img_w, img_h)
                    print(f"  缩放: {self.zoom}x")
            # Z: 在 1x 和居中 bbox 的缩放之间切换
            elif key == ord('z') or key == ord('Z'):
                if self.zoom_idx == 0:
                    self.zoom_idx = min(3, len(self.ZOOM_LEVELS) - 1)  # 跳到 4x
                    if bbox:
                        self._center_on_bbox(bbox, img_w, img_h)
                else:
                    self.zoom_idx = 0
                    self.pan_x = 0
                    self.pan_y = 0
                print(f"  缩放: {self.zoom}x")
            # Tab: 切换 bbox
            elif key == 9:
                if dets:
                    self.selected_bbox = (self.selected_bbox + 1) % len(dets)
                    # 缩放时自动居中到新 bbox
                    if self.zoom > 1 and dets:
                        self._center_on_bbox(dets[self.selected_bbox]["bbox"], img_w, img_h)
            # T: 修改数字文本 (窗口内输入)
            elif key == ord('t') or key == ord('T'):
                if dets and sel < len(dets):
                    old = dets[sel]["number"]
                    new_text = self._input_in_window(
                        window_name, current_view,
                        f"Edit number (was: {old}):")
                    if new_text is not None and new_text.strip() and new_text.strip() != old:
                        dets[sel]["number"] = new_text.strip()
                        self.modified.add(frame_name)
                        print(f"  已更新: '{old}' -> '{new_text.strip()}'")
            # X/Delete: 删除 bbox
            elif key == ord('x') or key == ord('X') or key == 3014656:
                if dets and sel < len(dets):
                    removed = dets.pop(sel)
                    self.modified.add(frame_name)
                    self.selected_bbox = min(sel, max(0, len(dets) - 1))
                    print(f"  删除 bbox [{sel}] '{removed['number']}'")
                    # 如果该帧无 bbox 了，从 detections 中移除
                    if not dets:
                        del self.detections[frame_name]
                        self.frame_names = sorted(self.detections.keys())
                        self.current_idx = min(self.current_idx, len(self.frame_names) - 1)
                        continue
            # B: 新增 bbox (进入鼠标框选模式)
            elif key == ord('b') or key == ord('B'):
                print("  框选模式: 在图上拖动鼠标框选新 bbox...")
                self.drawing = True
                # 如果帧不在 detections 中，先加入
                if frame_name not in self.detections:
                    self.detections[frame_name] = {"detections": []}
                    self.frame_names = sorted(self.detections.keys())
                continue  # 不推进，等鼠标完成
            # R: 重置
            elif key == ord('r') or key == ord('R'):
                print(f"  重置当前帧 (重新推理需要重新运行 --run-inference)")
            # Enter/Space: 切换 bbox 或跳随机未审查帧
            elif key == 13 or key == 32:
                if dets and self.selected_bbox < len(dets) - 1:
                    # 还有下一个 bbox，切换过去
                    self.selected_bbox += 1
                    if self.zoom > 1:
                        self._center_on_bbox(dets[self.selected_bbox]["bbox"], img_w, img_h)
                else:
                    # 最后一个 bbox 或无 bbox，标记已审查，跳随机帧
                    self.reviewed_frames.add(frame_name)
                    self.review_history.append(self.current_idx)
                    self._goto_random_unreviewed()
                    # 每 10 帧自动保存
                    if len(self.reviewed_frames) % 10 == 0:
                        self._save_detections()
                        self._save_progress()
                        print(f"  [自动保存] 已审查 {len(self.reviewed_frames)} 帧")
            # Backspace: 回到上一个审查过的帧，并取消其已审查状态
            elif key == 8:
                if self.review_history:
                    prev_idx = self.review_history.pop()
                    prev_name = self.frame_names[prev_idx]
                    self.reviewed_frames.discard(prev_name)
                    self.current_idx = prev_idx
                    self._reset_view()
                    print(f"  回到帧 [{prev_idx + 1}] (已取消审查标记，需重新确认)")
            # N: 跳 10 帧
            elif key == ord('n') or key == ord('N'):
                self.current_idx = min(self.current_idx + 10, len(self.frame_names) - 1)
                self._reset_view()
                print(f"  跳到第 {self.current_idx + 1} 帧")
            # P: 回退 10 帧
            elif key == ord('p') or key == ord('P'):
                self.current_idx = max(0, self.current_idx - 10)
                self._reset_view()
                print(f"  跳到第 {self.current_idx + 1} 帧")
            # F: 快进到下一个不同数字组合的帧 (跳过相似帧)
            elif key == ord('f') or key == ord('F'):
                self._skip_similar()
            # G: 跳转到指定帧号 (窗口内输入)
            elif key == ord('g') or key == ord('G'):
                result = self._input_in_window(
                    window_name, current_view,
                    f"Go to frame (1-{len(self.frame_names)}):",
                    digits_only=True)
                if result is not None and result.strip():
                    try:
                        n = int(result.strip())
                        self.current_idx = max(0, min(n - 1, len(self.frame_names) - 1))
                        self._reset_view()
                        print(f"  跳到第 {self.current_idx + 1} 帧")
                    except ValueError:
                        pass
            # Q: 退出
            elif key == ord('q') or key == ord('Q'):
                break

        cv2.destroyAllWindows()
        self._save_detections()
        self._save_progress()


# ---------------------------------------------------------------------------
# Step 3: 导出为 YOLO 训练数据
# ---------------------------------------------------------------------------

def export_yolo_data():
    """将审查后的检测结果导出为 YOLO 标注格式，可直接加入训练集"""
    if not DETECTIONS_FILE.exists():
        print("错误: 无检测结果")
        return

    with open(DETECTIONS_FILE, "r", encoding="utf-8") as f:
        detections = json.load(f)

    export_img_dir = OUTPUT_DIR / "yolo_export" / "images"
    export_lbl_dir = OUTPUT_DIR / "yolo_export" / "labels"
    export_img_dir.mkdir(parents=True, exist_ok=True)
    export_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 同时导出 CRNN 数据
    crnn_dir = OUTPUT_DIR / "crnn_export" / "images"
    crnn_dir.mkdir(parents=True, exist_ok=True)
    crnn_labels = []
    crop_idx = 0

    exported = 0
    for frame_name, data in sorted(detections.items()):
        dets = data["detections"]
        if not dets:
            continue

        img_path = FRAMES_DIR / frame_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        # 复制图片
        dst_img = export_img_dir / frame_name
        if not dst_img.exists():
            cv2.imwrite(str(dst_img), img)

        # 写 YOLO label
        label_name = Path(frame_name).stem + ".txt"
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # CRNN crop
            number = d["number"]
            if number and number != "???" and number != "":
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(img_w, x2)
                y2c = min(img_h, y2)
                if x2c > x1c and y2c > y1c:
                    crop = img[y1c:y2c, x1c:x2c]
                    crop_name = f"new_crop_{crop_idx:06d}.png"
                    cv2.imwrite(str(crnn_dir / crop_name), crop)
                    crnn_labels.append(f"{crop_name}\t{number}")
                    crop_idx += 1

        with open(export_lbl_dir / label_name, "w") as f:
            f.write("\n".join(lines) + "\n")
        exported += 1

    # CRNN labels
    if crnn_labels:
        with open(OUTPUT_DIR / "crnn_export" / "labels.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(crnn_labels))

    print(f"YOLO 数据导出: {exported} 张图 -> {export_img_dir.parent}")
    print(f"CRNN 数据导出: {crop_idx} 个 crop -> {crnn_dir.parent}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="检测结果审查 & 修正工具")
    parser.add_argument("--run-inference", action="store_true",
                        help="Step 1: 批量跑 YOLO+CRNN 推理")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO 置信度阈值 (仅 --run-inference 时有效)")
    parser.add_argument("--export", action="store_true",
                        help="Step 3: 导出为 YOLO+CRNN 训练数据")
    args = parser.parse_args()

    if args.run_inference:
        run_batch_inference(conf=args.conf)
    elif args.export:
        export_yolo_data()
    else:
        tool = ReviewTool()
        tool.run()


if __name__ == "__main__":
    main()
