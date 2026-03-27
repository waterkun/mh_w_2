"""Debug HSV distribution of gauge photos to tune thresholds."""

import os
import cv2
import numpy as np


def imread_unicode(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "gauge photo")
FILES = ["红刃.png", "红刃2.png", "红刃3.png", "yelloGauge.png", "whiteGauge.png", "NoGauge.png"]

for filename in FILES:
    path = os.path.join(PHOTOS_DIR, filename)
    img = imread_unicode(path)
    if img is None:
        print(f"[SKIP] {filename}")
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Filter out very dark pixels (background)
    bright_mask = v > 50
    if np.any(bright_mask):
        h_bright = h[bright_mask]
        s_bright = s[bright_mask]
        v_bright = v[bright_mask]
    else:
        h_bright = h.flatten()
        s_bright = s.flatten()
        v_bright = v.flatten()

    print(f"\n{'=' * 60}")
    print(f"{filename}  (shape: {img.shape})")
    print(f"  All pixels:")
    print(f"    H: min={h.min()} max={h.max()} mean={h.mean():.1f}")
    print(f"    S: min={s.min()} max={s.max()} mean={s.mean():.1f}")
    print(f"    V: min={v.min()} max={v.max()} mean={v.mean():.1f}")
    print(f"  Bright pixels (V>50): {np.sum(bright_mask)} / {bright_mask.size}")
    print(f"    H: min={h_bright.min()} max={h_bright.max()} mean={h_bright.mean():.1f}")
    print(f"    S: min={s_bright.min()} max={s_bright.max()} mean={s_bright.mean():.1f}")
    print(f"    V: min={v_bright.min()} max={v_bright.max()} mean={v_bright.mean():.1f}")

    # Count pixels in each HSV range
    # Red
    red1 = cv2.inRange(hsv, np.array([0, 70, 100]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 70, 100]), np.array([180, 255, 255]))
    red_cnt = np.count_nonzero(red1) + np.count_nonzero(red2)
    # Yellow
    yel = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([35, 255, 255]))
    yel_cnt = np.count_nonzero(yel)
    # White
    wht = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    wht_cnt = np.count_nonzero(wht)

    total = img.shape[0] * img.shape[1]
    print(f"  Color hits: red={red_cnt} ({red_cnt/total:.1%})  "
          f"yellow={yel_cnt} ({yel_cnt/total:.1%})  "
          f"white={wht_cnt} ({wht_cnt/total:.1%})")

    # H histogram for bright pixels
    hist_h, _ = np.histogram(h_bright, bins=36, range=(0, 180))
    print(f"  H histogram (bright, 5-degree bins):")
    for i, count in enumerate(hist_h):
        if count > 0:
            bar = '#' * min(count * 40 // max(hist_h.max(), 1), 40)
            print(f"    H[{i*5:3d}-{(i+1)*5:3d}]: {count:5d} {bar}")
