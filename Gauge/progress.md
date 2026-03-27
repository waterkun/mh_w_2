# Gauge Detection 开发进度

## 2026-02-28

### 已完成
- [x] 创建 README.md — 模块说明文档
- [x] 收集参考截图 (gauge photo/) — 红刃、黄刃、白刃、无刃
- [x] 确定 gauge 在屏幕上的固定 ROI 区域 — `select_roi.py` + `roi_config.json`
- [x] 设计并实现基于颜色分割的 gauge 检测器 — `gauge_detector.py` (HSV 阈值法)
- [x] 实现红色检测 (HSV 双区间阈值)
- [x] 实现红刃剩余量估算 (span-based 像素比例)
- [x] 实现黄刃 / 白刃 / 无刃状态检测
- [x] 测试 & 调参 — 13/13 全部通过
  - `test_gauge.py` (gauge photo/) 6/6 通过
  - `test_photo/` 额外 7 张测试图 7/7 通过
  - 修复: `MIN_YELLOW_AREA_FRAC` 0.25→0.35→0.55，排除金色装饰边框干扰导致 NoGauge 误判为 yellow

### 进行中
- [x] 实时检测集成与验证

