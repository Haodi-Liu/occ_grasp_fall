# ACT_BC_KEYPOINT_STRATEGY AUX_EVAL（指标动态 + 采样可视化）

> 目标：在 **预训练** 与 **评估** 阶段记录 keypoint/strategy/phase 的动态指标，并保存采样可视化图片。  
> 说明：本文档描述**当前已实现**的行为与输出位置，并补充多任务预训练 aux_eval 的实现细节。

---

## 1) 当前实现总览

### 1.1 评估（eval.py / eval_ppi.py）
- `eval.py` / `eval_ppi.py` 通过 `aux_eval_cfg` 传入评估环境  
  → `IndependentEnvRunner.start(..., aux_eval_cfg)`  
  → `_IndependentEnvRunner._run_eval_independent(..., aux_eval_cfg)`  
  → `PreprocessAgent.set_aux_eval_cfg(aux_eval_cfg)`（**修复点**：转发到下层 agent，确保 aux_eval 生效）  
  → `ActBCKeypointStrategyAgent.set_aux_eval_cfg(aux_eval_cfg)`
- `ActBCKeypointStrategyAgent.act()` 在 `aux_eval_cfg.enabled=True` 且观测包含 GT 时：
  - 依赖字段（至少）：`has_affordance`、`{cam}_contact_2d/{cam}_grasp_2d/{cam}_affordance_2d`、
    `{cam}_contact_visible/{cam}_grasp_visible/{cam}_affordance_visible`、`strategy_type`、`phase_type`
  - 计算 `kp_2d`、`strategy_ce`、`phase_ce`
  - 写入 `self._aux_eval_last`
- `ActBCKeypointStrategyAgent.act_summaries()` 以 `ScalarSummary` 输出：  
  （**修复点**：内部将 Tensor 转为 Python float，避免多进程 CUDA Tensor 传输报错）
  - `aux_eval/kp_2d`
  - `aux_eval/strategy_ce`
  - `aux_eval/phase_ce`
- **ACT_BC_KEYPOINT（keypoint-only）**：仅输出 `aux_eval/kp_2d`，不记录 strategy/phase 指标
- `_IndependentEnvRunner` 将 `agent_summaries` 合并到 eval summaries（**修复点**：保证 aux_eval 写入 CSV/TB；CSV 现写入 `aux_eval_data.csv`）  
  同时对 `eval_envs/return` 缺失做兜底，避免 `IndexError` 崩溃

### 1.2 预训练（tools/pretrain_predictors.py）
- 每个 epoch 统计：
  - `proj_2d`、`aff_vis`、`strategy_ce`、`phase_ce`、`kp_total`、`total`
- 同时输出：
  - **stdout 日志**（打印）
  - **CSV**（`pretrain_train.csv` / `pretrain_eval.csv`）
  - **可选 TensorBoard**（`predictor_pretrain.yaml: tensorboard_logging=True`）

> **分离式训练注意**（当前实现已强制 train_mode=keypoint|strategy）  
> - `train_mode=keypoint`：`strategy_ce/phase_ce` 会被置 0（不训练策略头）  
> - `train_mode=strategy`：`proj_2d/aff_vis` 会被置 0（不训练 keypoint）  
> - 可视化仍会绘制 keypoint（来自冻结的 keypoint 头），用于观察条件预测稳定性
>
> **分离式推荐配置**：  
> - `conf/predictor_pretrain_keypoint.yaml`（Keypoint 阶段）  
> - `conf/predictor_pretrain_strategy.yaml`（Strategy 阶段）  
> 两份配置已默认分离 `save_dir/log_dir/aux_eval.save_path`，避免 ckpt/CSV/可视化混淆。

---

## 2) 指标动态记录位置（现状）

### 2.1 评估阶段指标
- **TensorBoard**：`${framework.logdir}/${rlbench.task_name}/${method.name}/seed${framework.start_seed}`
- **CSV**：
  - 评估主指标：同目录下的 `eval_data.csv`
  - 条件损失（aux_eval）：同目录下的 `aux_eval_data.csv`
- 条件损失指标名：
  - `aux_eval/kp_2d`
  - `aux_eval/strategy_ce`
  - `aux_eval/phase_ce`
  - **ACT_BC_KEYPOINT**：仅保留 `aux_eval/kp_2d`

示例目录（与问题中路径一致）：
```
/home/hdliu/arm_test/bimanual_edge_phone/multi/ACT_BC_KEYPOINT_STRATEGY/seed0/
  ├── eval_data.csv
  ├── aux_eval_data.csv
  └── events.out.tfevents.*
```

> 注意：`eval_data.csv` / `aux_eval_data.csv` 若已存在，`LogWriter` 不会重写表头。  
> 若字段发生变更，需要对应删除旧 CSV 才会生成新表头。

### 2.2 预训练阶段指标
- **CSV**：`predictor_pretrain.yaml: log_dir` 下
  - `pretrain_train.csv`
  - `pretrain_eval.csv`
- **TensorBoard（可选）**：同目录下 `events.out.tfevents.*`
- stdout 日志仍会打印（可按需重定向）

---

## 3) 采样可视化输出位置（现状）

### 3.1 统一绘制入口
位置：`occ_grasp_models/helpers/aux_eval_visualizer.py`
- `build_pred_info_from_outputs()`：组装 `pred_info`
- `render_aux_eval_like()`：绘制策略/阶段文字 + 预测/GT 关键点
- GT 关键点使用与预测点相同半径的空心圆（不同颜色），避免遮挡预测点

> **ACT_BC_KEYPOINT**：`pred_info` 不包含 strategy/phase 字段，因此不会渲染策略/阶段文字。

### 3.2 评估阶段可视化
位置：`occ_grasp_models/helpers/custom_rlbench_env.py`
- `CustomRLBenchEnv._maybe_save_aux_sample()` / `CustomMultiTaskRLBenchEnv._maybe_save_aux_sample()`  
  调用 `render_aux_eval_like()` 保存图片
- 输出路径：`aux_eval_cfg.save_path`
- 文件命名：`ep{episode}_step{step}.png`
- 采样尽量均匀覆盖四个阶段（按 `phase_type` 做均衡，单 episode 内受 `max_samples_per_episode` 限制）

默认路径示例（需在 eval.yaml 中设置）：
```
cinematic_recorder.save_path: /home/hdliu/arm_test/bimanual_edge_phone/multi/ACT_BC_KEYPOINT_STRATEGY/seed0
aux_eval.save_path: ${cinematic_recorder.save_path}/aux_eval_samples
```

输出：
```
/home/hdliu/arm_test/bimanual_edge_phone/multi/ACT_BC_KEYPOINT_STRATEGY/seed0/aux_eval_samples/
  ├── ep0_step0000.png
  ├── ep0_step0010.png
  └── ...
```

### 3.3 预训练阶段可视化
位置：`occ_grasp_models/tools/pretrain_predictors.py`
- 在 eval loop 中调用 `render_aux_eval_like()`  
- 输出路径：`predictor_pretrain.yaml: aux_eval.save_path`
- 文件命名：`ep{epoch}_step{batch}_s{sample}.png`
- 采样尽量均匀覆盖四个阶段（按 `phase_type` 做均衡）

默认路径（与评估输出分离，来自 `predictor_pretrain.yaml`）：
```
/home/hdliu/arm_test/multi_task/predictor_pretrain_logs/aux_eval_samples
```

输出示例（多任务按 task 分目录）：
```
/home/hdliu/arm_test/multi_task/predictor_pretrain_logs/aux_eval_samples/
  ├── bimanual_edge_phone/ep000_step0000_s00.png
  ├── bimanual_pivot_phone/ep000_step0000_s00.png
  └── ...
```

**多任务行为（当前实现）**  
- `aux_eval.max_samples_per_epoch` 为**每个任务**在每个 epoch 评估阶段的上限。  
- 输出按任务分目录：`aux_eval.save_path/<task>/ep{epoch}_step{batch}_s{sample}.png`。
- 单任务/多任务均按 `phase_type` 做均衡采样（每个 phase 上限为 `ceil(max_samples_per_epoch / num_phases)`）。

---

## 4) 关键配置项（当前生效）

### 4.1 eval.yaml（评估）
```yaml
aux_eval:
  enabled: True
  log_every_n_steps: 1
  sample_every_n_steps: 10
  max_samples_per_episode: 6
  sample_camera: "over_shoulder_right"
  save_path: "${cinematic_recorder.save_path}/aux_eval_samples"
  sample_min_width: 300
```

### 4.2 predictor_pretrain.yaml（预训练）
```yaml
log_dir: /home/hdliu/arm_test/multi_task/predictor_pretrain_logs
tensorboard_logging: False
aux_eval:
  enabled: True
  sample_every_n_steps: 20
  max_samples_per_epoch: 6
  sample_camera: over_shoulder_right
  save_path: /home/hdliu/arm_test/multi_task/predictor_pretrain_logs/aux_eval_samples
  sample_min_width: 300
  sample_output_scale: 0.0
```

---

## 5) 结论（对问题的直接回答）

1) **8.2 预训练**  
   - 指标动态：**有**（stdout + CSV；可选 TB）  
   - 可视化图片：**有**（保存到 `predictor_pretrain.yaml: aux_eval.save_path`）  
   - 默认与评估输出路径分离（避免混用）

2) **8.4 评估**  
   - 指标动态：**有**（`aux_eval/*` 指标写入 TensorBoard + `aux_eval_data.csv`；主评估指标仍在 `eval_data.csv`）  
   - 可视化图片：**有**（保存到 `aux_eval.save_path`，默认 `cinematic_recorder.save_path/aux_eval_samples`）

> **ACT_BC_KEYPOINT**：仅保留 `aux_eval/kp_2d`，策略/阶段文字不再输出。

---

## 6) 多任务预训练 aux_eval（当前实现）

### 6.1 数据返回
- `PredictorDataset.__getitem__()` 返回 `task`（来自 index 的 `item["task"]`），供 eval loop 按任务统计。

### 6.2 采样计数
- `aux_eval.max_samples_per_epoch` 按任务统计；每个 epoch 开始时计数清零。  
- 采样仍受 `sample_every_n_steps` 控制（按 eval batch 频率触发）。
- 每个 phase 上限为 `ceil(max_samples_per_epoch / num_phases)`，用于均衡覆盖阶段。

### 6.3 输出组织
- 输出按任务分目录：`aux_eval.save_path/<task>/ep{epoch}_step{batch}_s{sample}.png`。

---

## 7) ACT_BC_KEYPOINT（keypoint-only）aux_eval 调整说明

> 目的：说明新 agent 不含 strategy/phase 时的指标与可视化行为差异。

### 7.1 指标输出变化
- `ActBCKeypointAgent.act_summaries()` **仅输出**：`aux_eval/kp_2d`。
- `aux_eval/strategy_ce` 与 `aux_eval/phase_ce` **不会出现**（CSV/TB 中也不再写入）。
- 若 `aux_eval_data.csv` 旧表头包含 strategy/phase 字段，请先删除旧 CSV 以生成新表头。

### 7.2 可视化变化
- `pred_info` 不再包含 `strategy_name / phase_name`，`render_aux_eval_like()` **不会绘制策略/阶段文字**。
- 关键点预测与 GT 关键点绘制逻辑保持一致，仅缺少策略/阶段文本块。

### 7.3 预训练脚本影响
- `tools/pretrain_predictors.py` 仍会统计 `strategy_ce/phase_ce` 字段，但在 `train_mode=keypoint` 时为 0（不参与训练）。
- 对 ACT_BC_KEYPOINT 评估流程无影响；评估阶段仅记录 `aux_eval/kp_2d`。
