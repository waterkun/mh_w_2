# Changelog

## 2026-02-27: 移动审查/诊断工具到 hit_number_detection

将 `realtime_tests/` 下的审查和诊断工具移入 `hit_number_detection/`，使 `realtime_tests/` 只保留实时游戏测试。

移动文件:
- `review_detections.py` + `detection_review/`
- `review_bbox_fix.py`
- `diagnose_yolo_data.py`
- `relabel_1xx.py`
- `yolo_diagnosis/`

路径更新: 所有移动的脚本中 `BASE` 改为 `Path(__file__).resolve().parent`，`merge_and_retrain.py` 的 `REVIEW_DIR` 改为 `BASE / "detection_review"`。

## 2026-02-26: 改造 review_detections.py 审查工具

### 背景
完成第一轮顺序审查 208 帧 + 重训后，需要随机审查更多帧，改进审查工具的交互方式和帧追踪机制。

### hit_number_detection/review_detections.py
1. **新增 `reviewed_frames` 追踪机制** — 替代旧的 `last_idx` 顺序追踪，支持随机审查
   - 向后兼容: 自动从旧 `progress["reviewed"]` 和 `last_idx` 迁移
2. **Enter/Space 行为改变** — 先逐个高亮 bbox，最后一个确认后标记已审查并跳到随机未审查帧
3. **Backspace 改为历史栈回退** — 用 `review_history` 栈记录审查顺序，Backspace 弹出回到上一个
4. **新增 `_goto_random_unreviewed()`** — 从未审查帧中随机选一帧跳转
5. **`_save_progress` 更新** — 保存 `reviewed_frames` 列表
6. **Info bar 显示审查进度** — 显示 `reviewed: N`
7. **`run()` 起始帧** — 如果当前帧已审查，自动跳到随机未审查帧
8. **自动保存** — 每审查 10 帧自动保存 detections 和 progress
9. **`run_batch_inference` 适配** — 读取 `reviewed_frames` 而非 `last_idx`
10. **文档更新** — docstring 中 Enter/Space 和 Backspace 操作说明

### hit_number_detection/merge_and_retrain.py
- Step 1 导出逻辑: 读取 `reviewed_frames` 而非 `last_idx`，带向后兼容
