# Sharpness Detection 开发进度

## 2026-02-28

### 已完成
- [x] 收集参考截图 (photo/) — 蓝斩、白斩、白斩将尽
- [x] 创建 `sharpness_detector.py` — 基于 HSV 颜色分割的锐度检测器
  - 支持 7 种颜色等级: Purple > White > Blue > Green > Yellow > Orange > Red
  - 使用 span-based 方法估算当前颜色剩余百分比
  - 颜色选择: 面积最大颜色策略 (非优先级匹配，避免蓝斩误判白斩)
- [x] 创建 `select_roi.py` — ROI 框选工具，截屏后拖拽选择锐度条区域
- [x] 创建 `sharpness_tracker.py` — 多帧状态追踪器
  - 连续 3 帧确认才切换颜色状态 (防抖动)
  - 检测颜色下降事件 (`color_just_dropped`)
  - 闪烁检测 (`is_flashing`): 2 秒内 4 次 unknown → 判定闪烁，保持已知颜色
  - 记录距上次磨刀的时间
  - RL reward 信号: 维持高锐度 +0.01，掉级 -0.5，掉到低级 -1.0
- [x] 创建 `test_sharpness.py` — 静态截图测试脚本
- [x] 创建 `main.py` — 实时屏幕捕获 + 锐度检测 + 两级磨刀提醒 Demo
- [x] 运行 `select_roi.py` — ROI: x=127, y=155, w=295, h=30
- [x] 运行 `test_sharpness.py` — 5/5 通过
- [x] 修复蓝斩误判为白斩 — 改用"面积最大颜色"策略替代"优先级匹配"
- [x] 新增闪烁检测 — 锐度条将尽时闪烁，闪烁时保持颜色 + 触发 alert
- [x] 实现两级磨刀提醒:
  - `should_sharpen` (软提醒): 蓝斩 — 黄字 "Sharpen when possible"
  - `need_sharpen` (硬提醒): 绿/黄/橙/红斩 — 红字 ">>> SHARPEN NOW! <<<"
  - 白斩及以上: 正常，无提醒

### 待办
- [ ] 运行 `main.py` 实测实时检测效果
- [ ] 收集更多颜色等级的截图 (green, yellow, orange, red, purple)
- [ ] 根据实测结果微调 HSV 参数和 LEFT_MARGIN_FRAC / RIGHT_MARGIN_FRAC

## 模块结构

```
Sharpness/
├── photo/                    # 参考截图
│   ├── blue_sharp.png
│   ├── blue_sharp1.png
│   ├── white_sharp.png
│   ├── white_sharp1.png
│   └── white_sharpness_almost_end.png
├── sharpness_detector.py     # 核心检测器 (单帧, 无状态)
├── sharpness_tracker.py      # 多帧追踪器 (有状态, 防抖动)
├── select_roi.py             # ROI 框选工具
├── test_sharpness.py         # 静态截图测试
├── main.py                   # 实时检测 Demo
├── roi_config.json           # ROI 配置 (select_roi.py 生成)
└── Progress.md               # 本文件
```

## 设计思路

### 检测方法
与 Gauge 模块一致，使用 HSV 颜色分割法，无需训练模型:
1. 截取锐度条 ROI 区域
2. 转换到 HSV 色彩空间
3. 对每种颜色等级做 `inRange` 掩码
4. 选择面积最大的颜色作为当前锐度
5. 用 span-based 方法估算该颜色的剩余填充比例

### 两级磨刀提醒逻辑

| 锐度等级 | `should_sharpen` | `need_sharpen` | 显示 |
|---------|-----------------|---------------|------|
| 白斩及以上 | False | False | 正常 |
| 蓝斩 | **True** | False | 黄字 "Sharpen when possible" |
| 绿/黄/橙/红 | True | **True** | 红字 ">>> SHARPEN NOW! <<<" |

### 闪烁处理
- 锐度条将尽时会闪烁 (颜色 ↔ 消失交替)
- Tracker 检测到 2 秒内 4 次 unknown → `is_flashing = True`
- 闪烁时保持上一个已知颜色，不会误切到 unknown
- 闪烁自动触发 `should_sharpen` 和 `need_sharpen`
