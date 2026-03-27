"""Click-based health bar labeling tool.

Displays ROI images scaled to fill screen width (~3400px), with minimum
height of 300px for easy clicking.

Percentage = click_x / roi_width (no margin, model learns boundaries itself).

  - Left click  -> green (health) boundary
  - Right click -> red (damage) boundary
  - Enter       -> confirm and next
  - S           -> skip frame
  - Z           -> undo last label
  - Q           -> save and quit

Usage:
    python label_health_bar.py [--data health_bar_data]
"""

import argparse
import csv
import os

import cv2
import numpy as np


# Target display width (pixels) — fits most monitors
DISPLAY_WIDTH = 3400
MIN_DISPLAY_HEIGHT = 300

WINDOW_NAME = "Label Health Bar"


class HealthBarLabeler:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.rois_dir = os.path.join(data_dir, "rois")
        self.labels_path = os.path.join(data_dir, "labels.csv")
        self.scale = 1  # computed per image

        # Load existing labels
        self.labels = {}
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.labels[row["filename"]] = {
                        "health_pct": float(row["health_pct"]),
                        "damage_pct": float(row["damage_pct"]),
                    }
            print(f"Loaded {len(self.labels)} existing labels")

        # Get all ROI images
        self.files = sorted([
            f for f in os.listdir(self.rois_dir)
            if f.endswith((".png", ".jpg"))
        ])
        print(f"Found {len(self.files)} ROI images, {len(self.labels)} already labeled")

        # Filter to unlabeled
        self.unlabeled = [f for f in self.files if f not in self.labels]
        print(f"{len(self.unlabeled)} remaining to label")

        # Start from the last labeled frame so user can review it
        self.current_idx = 0
        if self.labels:
            last_labeled = sorted(self.labels.keys())[-1]
            if last_labeled in self.unlabeled:
                self.current_idx = self.unlabeled.index(last_labeled)
            else:
                # Find next unlabeled frame after last labeled
                for i, f in enumerate(self.unlabeled):
                    if f > last_labeled:
                        self.current_idx = max(0, i - 1) if i > 0 else 0
                        break

        self.health_x = None  # click x in display coordinates
        self.damage_x = None
        self.history = []  # for undo

    def _compute_scale(self, roi_w, roi_h):
        """Compute scale factor so image fills screen width with enough height."""
        scale_w = DISPLAY_WIDTH / roi_w
        scale_h = MIN_DISPLAY_HEIGHT / roi_h
        self.scale = max(scale_w, scale_h)

    def _click_to_pct(self, click_x_display, roi_w):
        """Convert display click position to percentage of full ROI width."""
        x_orig = click_x_display / self.scale
        pct = x_orig / roi_w
        return max(0.0, min(1.0, pct))

    def _draw(self, roi):
        """Draw the labeling interface."""
        h, w = roi.shape[:2]
        self._compute_scale(w, h)
        s = self.scale
        sw = int(w * s)
        sh = int(h * s)

        scaled = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_NEAREST)

        # Draw health boundary (green)
        if self.health_x is not None:
            cv2.line(scaled, (self.health_x, 0), (self.health_x, sh),
                     (0, 255, 0), 2)
            pct = self._click_to_pct(self.health_x, w)
            cv2.putText(scaled, f"HP: {pct:.1%}", (self.health_x + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw damage boundary (red)
        if self.damage_x is not None:
            cv2.line(scaled, (self.damage_x, 0), (self.damage_x, sh),
                     (0, 0, 255), 2)
            if self.health_x is not None:
                dmg_pct = self._click_to_pct(self.damage_x, w) - \
                          self._click_to_pct(self.health_x, w)
                dmg_pct = max(0.0, dmg_pct)
                cv2.putText(scaled, f"DMG: {dmg_pct:.1%}",
                            (self.damage_x + 5, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Info bar at bottom
        info = np.zeros((50, sw, 3), dtype=np.uint8)
        fname = self.unlabeled[self.current_idx] if self.current_idx < len(self.unlabeled) else "DONE"
        remaining = len(self.unlabeled) - self.current_idx
        text = (f"{fname}  |  Remaining: {remaining}  |  "
                f"Total labeled: {len(self.labels)}  |  "
                f"LClick=HP  RClick=DMG  Enter=OK  S=Skip  Z=Undo  Q=Quit")
        cv2.putText(info, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return np.vstack([scaled, info])

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.health_x = x
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.damage_x = x

    def _save_labels(self):
        with open(self.labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "health_pct", "damage_pct"])
            writer.writeheader()
            for fname in sorted(self.labels.keys()):
                writer.writerow({
                    "filename": fname,
                    "health_pct": f"{self.labels[fname]['health_pct']:.4f}",
                    "damage_pct": f"{self.labels[fname]['damage_pct']:.4f}",
                })
        print(f"Saved {len(self.labels)} labels to {self.labels_path}")

    def run(self):
        if not self.unlabeled:
            print("All frames already labeled!")
            return

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        while self.current_idx < len(self.unlabeled):
            fname = self.unlabeled[self.current_idx]
            roi = cv2.imread(os.path.join(self.rois_dir, fname))
            if roi is None:
                print(f"Warning: cannot read {fname}, skipping")
                self.current_idx += 1
                continue

            self.health_x = None
            self.damage_x = None

            # Restore previous labels as visual guides
            if fname in self.labels:
                w = roi.shape[1]
                self._compute_scale(w, roi.shape[0])
                hp = self.labels[fname]["health_pct"]
                dp = self.labels[fname]["damage_pct"]
                self.health_x = int(hp * w * self.scale)
                if dp > 0:
                    self.damage_x = int((hp + dp) * w * self.scale)

            while True:
                display = self._draw(roi)
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    self._save_labels()
                    cv2.destroyAllWindows()
                    return

                elif key == 13:  # Enter
                    if self.health_x is not None:
                        w = roi.shape[1]
                        health_pct = self._click_to_pct(self.health_x, w)
                        if self.damage_x is not None:
                            damage_pct = self._click_to_pct(self.damage_x, w) - health_pct
                            damage_pct = max(0.0, damage_pct)
                        else:
                            damage_pct = 0.0

                        self.labels[fname] = {
                            "health_pct": round(health_pct, 4),
                            "damage_pct": round(damage_pct, 4),
                        }
                        self.history.append(fname)
                        self.current_idx += 1

                        # Auto-save every 20 labels
                        if len(self.history) % 20 == 0:
                            self._save_labels()
                        break

                elif key == ord("s"):
                    self.current_idx += 1
                    break

                elif key == ord("z"):
                    if self.history:
                        last = self.history.pop()
                        # Go back to the frame (keep label so lines are visible)
                        try:
                            self.current_idx = self.unlabeled.index(last)
                        except ValueError:
                            # Already labeled, find in full file list
                            if last in self.files:
                                idx = self.files.index(last)
                                # Insert temporarily into unlabeled at right position
                                self.unlabeled.insert(self.current_idx, last)
                                self.current_idx = self.current_idx
                        break

        self._save_labels()
        cv2.destroyAllWindows()
        print("All frames labeled!")


def main():
    parser = argparse.ArgumentParser(description="Label health bar ROI images")
    parser.add_argument("--data", default="health_bar_data",
                        help="Data directory (default: health_bar_data)")
    args = parser.parse_args()
    labeler = HealthBarLabeler(args.data)
    labeler.run()


if __name__ == "__main__":
    main()
