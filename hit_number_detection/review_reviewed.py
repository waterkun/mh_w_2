"""
临时脚本: 只审查已审查过的帧 (抽检/修正)

操作与 review_detections.py 相同，区别:
- 只显示 reviewed_frames 中的帧
- Enter/Space 确认后跳到下一个随机已审查帧
- Backspace 回到上一个
- 发现问题的帧会从 reviewed_frames 中移除 (按 U 标记为未审查)

Usage:
  E:/anaconda3/envs/mh_ai/python.exe hit_number_detection/review_reviewed.py
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from review_detections import ReviewTool, FRAMES_DIR, DETECTIONS_FILE, PROGRESS_FILE


class ReviewReviewedTool(ReviewTool):
    """只遍历已审查帧的审查工具"""

    def __init__(self):
        super().__init__()
        # 只保留已审查的帧
        self.reviewed_list = sorted(self.reviewed_frames & set(self.frame_names))
        self.reviewed_idx = 0  # 当前在 reviewed_list 中的位置
        self.history = []
        self.removed_count = 0
        print(f"已审查帧: {len(self.reviewed_list)} 个")

    def _goto_random_reviewed(self):
        remaining = [i for i in range(len(self.reviewed_list))
                     if self.reviewed_list[i] in self.reviewed_frames]
        if not remaining:
            print("所有已审查帧都已重新检查完毕!")
            return False
        self.reviewed_idx = random.choice(remaining)
        name = self.reviewed_list[self.reviewed_idx]
        self.current_idx = self.frame_names.index(name)
        self._reset_view()
        print(f"  -> 已审查帧 [{self.reviewed_idx + 1}/{len(self.reviewed_list)}] "
              f"帧号 [{self.current_idx + 1}] {name}")
        return True

    def run(self):
        if not self.reviewed_list:
            print("没有已审查帧")
            return

        self._goto_random_reviewed()
        print(f"\n快捷键: Enter/Space=确认下一个  U=标记为未审查(移除)  Backspace=回退  Q=退出")
        print(f"其他操作与 review_detections.py 相同 (ADWS/箭头/Tab/T/X/B/+/-/Z)\n")

        window_name = "Review Reviewed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 900)

        current_img = None

        while True:
            frame_name, data = self._get_frame_data()
            dets = data["detections"]

            img_path = FRAMES_DIR / frame_name
            current_img = cv2.imread(str(img_path))
            if current_img is None:
                if not self._goto_random_reviewed():
                    break
                continue

            img_h, img_w = current_img.shape[:2]

            if dets:
                self.selected_bbox = min(self.selected_bbox, len(dets) - 1)
                self.selected_bbox = max(self.selected_bbox, 0)

            cv2.setMouseCallback(window_name, self._mouse_callback, current_img)

            current_view = self._draw_view(current_img, frame_name, data)
            cv2.imshow(window_name, current_view)
            key = cv2.waitKeyEx(0)

            # 框选模式中取消
            if self.drawing and key in (8, 27, 3342336):
                self.drawing = False
                self.draw_start = None
                self.draw_end = None
                print("  框选已取消")
                continue

            sel = self.selected_bbox
            bbox = dets[sel]["bbox"] if dets and sel < len(dets) else None
            step = self.STEP

            # --- bbox 编辑键 (与 review_detections 一致) ---
            if key == ord('a') or key == ord('A'):
                if bbox:
                    bbox[0] = max(0, bbox[0] - step)
                    self.modified.add(frame_name)
            elif key == ord('d') or key == ord('D'):
                if bbox:
                    bbox[0] = min(bbox[2] - 1, bbox[0] + step)
                    self.modified.add(frame_name)
            elif key == 2424832 or key == 81:  # ←
                if bbox:
                    bbox[2] = max(bbox[0] + 1, bbox[2] - step)
                    self.modified.add(frame_name)
            elif key == 2555904 or key == 83:  # →
                if bbox:
                    bbox[2] = min(img_w, bbox[2] + step)
                    self.modified.add(frame_name)
            elif key == ord('w') or key == ord('W'):
                if bbox:
                    bbox[1] = max(0, bbox[1] - step)
                    self.modified.add(frame_name)
            elif key == ord('s') or key == ord('S'):
                if bbox:
                    bbox[1] = min(bbox[3] - 1, bbox[1] + step)
                    self.modified.add(frame_name)
            elif key == 2490368 or key == 82:  # ↑
                if bbox:
                    bbox[3] = max(bbox[1] + 1, bbox[3] - step)
                    self.modified.add(frame_name)
            elif key == 2621440 or key == 84:  # ↓
                if bbox:
                    bbox[3] = min(img_h, bbox[3] + step)
                    self.modified.add(frame_name)
            # 缩放
            elif key == ord('+') or key == ord('='):
                if self.zoom_idx < len(self.ZOOM_LEVELS) - 1:
                    self.zoom_idx += 1
                    if bbox:
                        self._center_on_bbox(bbox, img_w, img_h)
            elif key == ord('-') or key == ord('_'):
                if self.zoom_idx > 0:
                    self.zoom_idx -= 1
                    self._clamp_pan(img_w, img_h)
            elif key == ord('z') or key == ord('Z'):
                if self.zoom_idx == 0:
                    self.zoom_idx = min(3, len(self.ZOOM_LEVELS) - 1)
                    if bbox:
                        self._center_on_bbox(bbox, img_w, img_h)
                else:
                    self.zoom_idx = 0
                    self.pan_x = 0
                    self.pan_y = 0
            # Tab
            elif key == 9:
                if dets:
                    self.selected_bbox = (self.selected_bbox + 1) % len(dets)
                    if self.zoom > 1 and dets:
                        self._center_on_bbox(dets[self.selected_bbox]["bbox"], img_w, img_h)
            # T: 修改数字
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
            # B: 新增 bbox
            elif key == ord('b') or key == ord('B'):
                print("  框选模式: 在图上拖动鼠标框选新 bbox...")
                self.drawing = True
                if frame_name not in self.detections:
                    self.detections[frame_name] = {"detections": []}
                    self.frame_names = sorted(self.detections.keys())
                continue
            # U: 标记为未审查 (从 reviewed_frames 移除)
            elif key == ord('u') or key == ord('U'):
                if frame_name in self.reviewed_frames:
                    self.reviewed_frames.discard(frame_name)
                    self.removed_count += 1
                    print(f"  !! 已移除审查标记: {frame_name} (累计移除 {self.removed_count} 个)")
                    self.history.append(self.current_idx)
                    if not self._goto_random_reviewed():
                        break
            # Enter/Space: 确认，跳下一个
            elif key == 13 or key == 32:
                if dets and self.selected_bbox < len(dets) - 1:
                    self.selected_bbox += 1
                    if self.zoom > 1:
                        self._center_on_bbox(dets[self.selected_bbox]["bbox"], img_w, img_h)
                else:
                    self.history.append(self.current_idx)
                    if not self._goto_random_reviewed():
                        break
            # Backspace: 回退
            elif key == 8:
                if self.history:
                    self.current_idx = self.history.pop()
                    self._reset_view()
                    name = self.frame_names[self.current_idx]
                    print(f"  回到帧 [{self.current_idx + 1}] {name}")
            # Q: 退出
            elif key == ord('q') or key == ord('Q'):
                break

        cv2.destroyAllWindows()
        self._save_detections()
        self._save_progress()
        print(f"\n完成! 移除了 {self.removed_count} 个审查标记")


if __name__ == "__main__":
    tool = ReviewReviewedTool()
    tool.run()
