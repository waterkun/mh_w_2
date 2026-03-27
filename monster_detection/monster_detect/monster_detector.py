"""怪物检测推理封装 — YOLO 目标检测模型."""

from ultralytics import YOLO

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class MonsterDetector:
    """怪物检测器 — 对 ROI 帧做 YOLO 推理."""

    def __init__(self, model_path: str,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self._model = YOLO(model_path)
        self._threshold = confidence_threshold

    def predict(self, roi_frame) -> tuple:
        """预测 ROI 帧是否有怪物.

        Args:
            roi_frame: BGR numpy array (任意尺寸, YOLO 内部 resize).

        Returns:
            (monster_visible: bool, max_confidence: float)
        """
        results = self._model(roi_frame, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return False, 0.0

        max_conf = boxes.conf.max().item()
        visible = max_conf > self._threshold
        return visible, max_conf
