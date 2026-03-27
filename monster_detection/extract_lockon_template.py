"""从 target_photo 裁出锁定标记模板, 保存用于模板匹配."""

import cv2
import numpy as np

img = cv2.imread("target_photo/target_photo.png")
h, w = img.shape[:2]
print(f"原图尺寸: {w}x{h}")

# 显示图片让用户用鼠标框选锁定标记
clone = img.copy()
roi = cv2.selectROI("框选锁定标记 (拖动框选, 按Enter确认, 按C取消)",
                     clone, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, rw, rh = roi
if rw == 0 or rh == 0:
    print("未选择区域, 退出")
else:
    template = img[y:y+rh, x:x+rw]
    out_path = "bot/lockon_template.png"
    cv2.imwrite(out_path, template)
    print(f"模板已保存: {out_path}  尺寸: {rw}x{rh}")
    cv2.imshow("Template", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
