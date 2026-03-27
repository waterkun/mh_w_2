# 实时综合测试 (Realtime Tests)

实时截取游戏画面，同时运行 **血条检测** 和 **伤害数字识别** 两个模块，将结果叠加显示在预览窗口中。

## 前置条件

- Python 3.10+
- 已训练好的模型文件：
  - **YOLO** (伤害数字检测): `hit_number_detection/runs/detect/runs/damage_number/weights/best.pt`
  - **CRNN** (数字识别): `hit_number_detection/runs/crnn/best.pt`
- 游戏需要在屏幕上运行（脚本通过截屏获取画面）

## 安装依赖

```bash
cd realtime_tests
pip install -r requirements.txt
```

## 运行

```bash
# 默认运行（同时测试血条 + 伤害数字）
python test_combined_realtime.py

# 指定显示器
python test_combined_realtime.py --monitor 1

# 调整缩放比例（默认 0.45）
python test_combined_realtime.py --scale 0.5

# 仅测试血条模块
python test_combined_realtime.py --no-damage

# 仅测试伤害数字模块
python test_combined_realtime.py --no-health

# 调低置信度阈值（默认 0.5）
python test_combined_realtime.py --conf 0.4

# 使用自定义模型路径（一般不需要，默认会自动找到训练好的模型）
python realtime_tests/test_combined_realtime.py --yolo hit_number_detection/runs/detect/runs/damage_number/weights/best.pt --crnn hit_number_detection/runs/crnn/best.pt
```

## 快捷键

| 按键 | 功能 |
|------|------|
| `Q` | 退出 |
| `R` | 重置追踪器和统计数据 |
| `S` | 截图保存到 `screenshots/` |
| `+` / `=` | 放大画面 |
| `-` | 缩小画面 |

## 画面说明

- **左上角 HUD**: 血条状态（HP、受击、奖励信号）
- **右上角**: 血条 ROI 预览 + HSV 遮罩
- **画面中**: 伤害数字检测框（黄色）+ 识别结果
- **底部统计条**: FPS、帧数、检测次数、累计伤害

## 参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--monitor` | `1` | mss 显示器编号 |
| `--scale` | `0.45` | 画面缩放比例 |
| `--yolo` | `(自动检测)` | YOLO 模型路径 |
| `--crnn` | `(自动检测)` | CRNN 模型路径 |
| `--conf` | `0.5` | YOLO 检测置信度阈值 |
| `--no-damage` | - | 禁用伤害数字检测 |
| `--no-health` | - | 禁用血条检测 |
| `--screenshot-dir` | `screenshots` | 截图保存目录 |
