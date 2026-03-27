# MH Wilds 伤害数字识别系统 — 进度追踪

## 系统架构

```
游戏画面 → YOLOv10 检测伤害数字位置 → 裁剪 ROI → CRNN 识别数字内容 → 输出结果
              ✅ 已完成                            ✅ 已完成
```

---

## ✅ 已完成

### 1. 帧提取工具 (`extract_frames.py`)
- 从游戏录屏中按指定 FPS 抽帧
- 支持裁剪指定区域（去掉 UI 等无关部分）
- 基于颜色特征（白色/橙色/黄色）自动筛选可能含伤害数字的帧
- 支持预览裁剪区域、一键抽帧+筛选
- **默认 ROI 检测区域: 924,25,1534,1398**

### 2. 帧挑选工具 (`pick_frames.py`)
- 从 `frames_filtered/` 中快速挑选帧到 CVAT 标注文件夹
- 纯键盘操作（Space=选中, S=跳过, B=回退, Q=退出）
- 选中/跳过后自动跳 4~7 帧避免重复场景

### 3. CVAT → YOLO 转换工具 (`convert_cvat_to_yolo.py`)
- 解析 CVAT XML 标注文件
- 自动转换 bbox 为 YOLO 归一化格式
- 自动随机分割 train/val/test (75%/15%/10%)

### 4. 数据审查工具 (`review_crops.py`)
- 裁剪所有 bbox 区域供人工审查
- 快速判断每个 crop 是否为数字（Enter=OK, d=BAD）
- 自动追溯错误 crop 到原图来源
- 退出时输出按原图分组的错误报告

### 5. YOLO 标签修复工具 (`fix_yolo_labels.py`)
- 根据审查结果自动清理 YOLO 标签中的错误 bbox
- 支持 dry-run 预览模式

### 6. CRNN 数据准备工具 (`prepare_crnn_data.py`)
- 从 YOLO 标注的 bbox 裁剪数字区域
- 生成 `crop_mapping.json` 追溯每个 crop 的来源
- 支持 annotations.json 模式和纯 YOLO 标注模式

### 7. OCR 预标注工具 (`ocr_prelabel.py`)
- 用 EasyOCR 自动识别所有 crop 的数字值
- 从 valid-yolo-data/ 裁剪 bbox → OCR 识别 → 生成预填 labels.txt
- OCR 识别率: 66% (1222/1833)，其余需人工输入
- 生成 crop_mapping.json 追溯每个 crop 来源

### 8. CRNN 数字标注工具 (`label_crnn_data.py`)
- 弹窗显示裁剪图片，用户确认/修正/删除 OCR 结果
- 支持: 输入数字、s=跳过、d=删除、b=回退、q=保存退出
- 自动保存进度，中断后可继续
- 退出时输出删除报告（按原图分组）

### 9. CRNN 数据分割工具 (`split_crnn_data.py`)
- 将审查后的标注数据分割为 train/val (80%/20%)
- 自动复制图片和生成 labels.txt

### 10. CRNN 数字识别模型 (`crnn_model.py`)
- 4 层 CNN 提取视觉特征 + 2 层 BiLSTM 序列建模
- CTC 解码输出数字字符串
- 输入: 32x128 灰度图 → 输出: 数字串（如 "127"）
- 模型参数量: 1,034,315

### 11. CRNN 训练脚本 (`train_crnn.py`)
- CTC Loss + ReduceLROnPlateau + 自动保存最佳模型

### 12. 端到端推理脚本 (`inference.py`)
- YOLO 检测 + CRNN 识别 一体化
- 支持: 单张图片 / 影片文件 / 摄像头实时画面

---

### 13. 环境配置
- Conda 环境 `mh_ai` (Python 3.10)
- PyTorch 2.10.0+cu126 (CUDA 12.6)
- torchvision 0.25.0+cu126
- ultralytics 8.4.14
- easyocr 1.7.2
- 硬件: **NVIDIA RTX 4090 24GB, 96GB RAM**

### 14. 数据采集与标注
- 使用 **CVAT (cvat.ai)** 标注（Roboflow 免费版限制 10 张）
- 标注类别: `damage_number`（单类别）
- 标注策略: 框住所有可辨识的伤害数字位置
- 通过 `review_crops.py` 审查所有 1965 个 crop
  - 通过: 1833 个
  - 移除: 132 个错误标注

### 15. 数据集统计

**YOLO 数据 (`valid-yolo-data/`) — 审查后的干净数据:**

| Split | Images | Bboxes |
|-------|--------|--------|
| train | 412 | ~1300 |
| valid | 100 | ~330 |
| test | 68 | ~200 |
| **合计** | **580** | **1833** |

**CRNN 数据 (`crnn_data/`) — OCR 预标注 + 人工审查后:**

| Split | Samples |
|-------|---------|
| train | 1461 |
| val | 366 |
| **合计** | **1827** |
| 删除（非数字/模糊） | 6 |

数字长度分布:

| 长度 | 数量 |
|------|------|
| 1 位数 | 25 |
| 2 位数 | 1599 |
| 3 位数 | 203 |

> **重要区分: YOLO 数据 vs CRNN 数据**
> - YOLO 数据: 包含所有可辨识的伤害数字**位置**，即使数字被部分遮挡无法读出具体值，
>   只要能看出是数字就标注。用途: 训练 YOLO 检测位置。
> - CRNN 数据: 只包含能清晰辨认具体数字值的裁剪图。用途: 训练 CRNN 识别数字内容。

### 16. YOLOv10 模型训练 — 第二轮（审查后数据）

训练命令:
```
E:/anaconda3/envs/mh_ai/python.exe train_yolo.py --batch 32 --img 640
```

| 指标 | 第一轮 | 第二轮 (审查后) | 提升 |
|------|--------|----------------|------|
| mAP50 | 0.827 | **0.982** | +15.5% |
| mAP50-95 | 0.553 | **0.776** | +22.3% |
| Precision | 0.892 | **0.966** | +7.4% |
| Recall | 0.759 | **0.962** | +20.3% |

模型文件:
- `runs/detect/runs/damage_number/weights/best.pt` (5.7MB)
- 模型: YOLOv10n, 2,265,363 参数, 6.5 GFLOPs
- 推理速度: 0.6ms/image (RTX 4090)
- 训练: 150 epochs, 0.11 hours

### 17. CRNN 模型训练

训练命令:
```
E:/anaconda3/envs/mh_ai/python.exe train_crnn.py --epochs 100 --batch 32
```

| Epoch | Loss | Val Acc | LR |
|-------|------|---------|----|
| 1 | 4.8385 | 0.00% | 0.001000 |
| 10 | 0.1168 | 91.53% | 0.001000 |
| 20 | 0.0078 | 97.27% | 0.001000 |
| 60 | 0.0003 | 97.81% | 0.001000 |
| 95 | 0.0006 | **98.09%** | 0.000250 |
| 100 | 0.0006 | 97.81% | 0.000250 |

模型文件:
- `runs/crnn/best.pt` — 最佳模型 (Val Acc=98.09%, Epoch 95)
- `runs/crnn/last.pt` — 最终模型
- 模型: CRNN (CNN+BiLSTM+CTC), 1,034,315 参数
- 训练: 100 epochs

---

## 进度总览

### 阶段一：数据准备与标注 ✅

| 步骤 | 说明 | 状态 |
|------|------|------|
| 1.1 | 录制/准备游戏战斗影片素材 | ✅ |
| 1.2 | 抽帧并筛选 | ✅ |
| 1.3 | 在 CVAT 上标注伤害数字 | ✅ (580 张, 1833 框) |
| 1.4 | 审查标注质量 | ✅ (移除 132 个错误) |
| 1.5 | 转换为 YOLO 格式 | ✅ |

### 阶段二：YOLO 检测模型训练 ✅

| 步骤 | 说明 | 状态 |
|------|------|------|
| 2.1 | 训练 YOLOv10 | ✅ |
| 2.2 | 审查后重新训练 | ✅ mAP50=0.982 |

### 阶段三：CRNN 识别模型训练 ✅

| 步骤 | 说明 | 状态 |
|------|------|------|
| 3.1 | 裁剪数字区域 | ✅ |
| 3.2 | OCR 预标注 (EasyOCR, 66% 识别率) | ✅ |
| 3.3 | 人工审查数字值 (1827 标注, 6 删除) | ✅ |
| 3.4 | 分割 train/val (80/20) | ✅ |
| 3.5 | 训练 CRNN | ✅ Val Acc=98.09% |

### 阶段四：端到端测试与迭代审查 🔄

| 步骤 | 说明 | 状态 |
|------|------|------|
| 4.1 | 批量推理 frames_filtered 全部帧 | ✅ `review_detections.py --run-inference` |
| 4.2 | 第一轮顺序审查 208 帧 (修正 bbox + 数字) | ✅ |
| 4.3 | 合并审查数据 + 重训 YOLO & CRNN | ✅ `merge_and_retrain.py` |
| 4.4 | 改造审查工具支持随机审查 | ✅ 2026-02-26 |
| 4.5 | 整理项目结构: 审查/诊断工具移入 hit_number_detection/ | ✅ 2026-02-27 |
| 4.6 | 第二轮随机审查更多帧 + 重训 | 🔲 迭代进行 |
| 4.7 | 在影片上测试实时检测 | 🔲 |
| 4.8 | 持续审查 → 重训循环，直到 YOLO & CRNN 满意 | 🔲 |

---

## 📁 项目文件结构

```
hit_number_detection/
├── extract_frames.py       # 帧提取与筛选
├── pick_frames.py          # 帧挑选工具
├── convert_cvat_to_yolo.py # CVAT → YOLO 转换
├── crop_all_for_review.py  # 裁剪所有 bbox 供审查
├── review_crops.py         # Crop 审查工具
├── fix_yolo_labels.py      # YOLO 标签修复
├── prepare_crnn_data.py    # CRNN 数据准备
├── ocr_prelabel.py         # OCR 预标注工具
├── label_crnn_data.py      # CRNN 数字标注工具
├── split_crnn_data.py      # CRNN 数据分割
├── dataset.yaml            # YOLO 数据集配置 → valid-yolo-data/
├── train_yolo.py           # YOLO 训练脚本
├── crnn_model.py           # CRNN 模型定义
├── train_crnn.py           # CRNN 训练脚本
├── inference.py            # 端到端推理
├── review_detections.py    # 检测结果审查工具
├── review_bbox_fix.py      # bbox 修正对比审查
├── diagnose_yolo_data.py   # YOLO 标注数据排查
├── relabel_1xx.py          # 1XX bbox 重标工具
├── merge_and_retrain.py    # 合并审查数据 + 重训
├── Progress.md             # 本文件
├── frames_filtered/        # 筛选后的帧
├── detection_review/       # 审查数据
│   ├── detections.json     # 全部帧检测结果
│   └── review_progress.json # 审查进度 (reviewed_frames)
├── yolo_diagnosis/         # YOLO 诊断输出
├── yolo-data/              # CVAT 原始标注数据
│   ├── 100-300/
│   └── 300-500/
├── valid-yolo-data/        # 审查后的干净 YOLO 数据
│   ├── images/{train,valid,test}/
│   └── labels/{train,valid,test}/
├── crnn_data/              # CRNN 数据
│   ├── ocr_prelabel/       # OCR 预标注 + 人工审查后的完整数据
│   ├── train/              # 训练集 (1461 张)
│   └── val/                # 验证集 (366 张)
└── runs/                   # 模型输出
    ├── detect/runs/damage_number/weights/
    │   ├── best.pt         # YOLO 最佳模型 (mAP50=0.982)
    │   └── last.pt
    └── crnn/
        ├── best.pt         # CRNN 最佳模型 (Val Acc=98.09%)
        └── last.pt

realtime_tests/
├── test_combined_realtime.py  # 实时游戏测试
└── requirements.txt
```

**最新模型路径 (用于推理/测试):**
- YOLO: `hit_number_detection/runs/detect/runs/damage_number/weights/best.pt`
- CRNN: `hit_number_detection/runs/crnn/best.pt`

```bash
# 实时测试命令
python realtime_tests/test_combined_realtime.py --yolo hit_number_detection/runs/detect/runs/damage_number/weights/best.pt --crnn hit_number_detection/runs/crnn/best.pt
```

---

## 📝 常用命令

```bash
# 抽帧+筛选
E:/anaconda3/envs/mh_ai/python.exe extract_frames.py run --video gameplay.mp4 --fps 10 --threshold 2

# 挑选帧
E:/anaconda3/envs/mh_ai/python.exe pick_frames.py --input frames_filtered --output dataset-cvat4

# CVAT → YOLO 转换
E:/anaconda3/envs/mh_ai/python.exe convert_cvat_to_yolo.py

# 裁剪所有 bbox 供审查
E:/anaconda3/envs/mh_ai/python.exe crop_all_for_review.py

# 审查 crops
E:/anaconda3/envs/mh_ai/python.exe review_crops.py --data crnn_data/review_all

# 修复 YOLO 标签
E:/anaconda3/envs/mh_ai/python.exe fix_yolo_labels.py --data crnn_data/review_all

# 训练 YOLOv10
E:/anaconda3/envs/mh_ai/python.exe train_yolo.py --batch 32 --img 640

# OCR 预标注
E:/anaconda3/envs/mh_ai/python.exe ocr_prelabel.py

# 审查 OCR 结果
E:/anaconda3/envs/mh_ai/python.exe label_crnn_data.py --data crnn_data/ocr_prelabel

# 分割 CRNN 数据
E:/anaconda3/envs/mh_ai/python.exe split_crnn_data.py

# 训练 CRNN
E:/anaconda3/envs/mh_ai/python.exe train_crnn.py --epochs 100 --batch 32

# 端到端推理
E:/anaconda3/envs/mh_ai/python.exe inference.py --image test.png
E:/anaconda3/envs/mh_ai/python.exe inference.py --video gameplay.mp4
```

---

### 18. 检测结果审查工具 (`hit_number_detection/review_detections.py`)

两步流程:
1. `--run-inference`: 对 frames_filtered 全部帧跑 YOLO+CRNN 批量推理，保存到 JSON
2. 默认模式: 交互式审查，可修正 bbox 位置和数字文本

功能:
- ADWS / 方向键调整 bbox 四边
- Tab 切换 bbox, T 修改数字, X 删除, B 新增 bbox (鼠标框选)
- +/- 缩放 (1x~6x), Z 快速切换 1x/4x
- Enter/Space 逐个确认 bbox，最后一个后跳到随机未审查帧
- Backspace 回到上一个审查过的帧 (历史栈)
- F 快进跳过相似帧, N/P 跳 10 帧, G 跳转指定帧号
- 每 10 帧自动保存，Q 退出时保存全部

审查进度文件: `hit_number_detection/detection_review/review_progress.json`
- `reviewed_frames`: 已审查帧名列表 (支持随机审查)
- 向后兼容旧的 `last_idx` 顺序格式

### 19. 合并审查数据 + 重训 (`merge_and_retrain.py`)

自动化流程:
1. 清理上次错误合并的数据
2. 导出已审查帧 (仅 `reviewed_frames` 中的帧)
3. 合并 YOLO 数据 (80/10/10 拆分追加到 valid-yolo-data)
4. 合并 CRNN 数据 (80/20 拆分追加到 crnn_data)
5. 删除 YOLO cache
6. 重训 YOLO (150 epochs, batch 16)
7. 重训 CRNN (100 epochs, batch 32)

---

---

## ⏭️ 下一步行动

**迭代审查 → 重训循环:**
1. 用 `review_detections.py` 随机审查未审查帧，修正 bbox 和数字
2. 用 `merge_and_retrain.py` 合并审查数据并重训 YOLO & CRNN
3. 重复 1-2 直到 YOLO 和 CRNN 精度满意
4. 在影片上测试实时检测效果
