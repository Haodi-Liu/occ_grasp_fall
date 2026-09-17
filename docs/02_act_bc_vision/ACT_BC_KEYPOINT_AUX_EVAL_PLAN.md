# ACT_BC_KEYPOINT 在线评估辅助可视化方案（与回流采集一体化）

> 目标：将在线评估阶段的辅助可视化与 C1 回流采集合并成一条链路。  
> 约束：回流训练样本保持“干净 RGB + 结构化 GT”，可视化图像单独产出，不污染训练数据。

---

## 1) 适用范围与结论

- 仅适用于 `ACT_BC_KEYPOINT`（keypoint-only）。
- 在线采集不再依赖 `strategy_type/phase_type` 字段。
- 每个 episode 的辅助可视化来自“该 episode 已采集回流帧”的均匀抽样子集。
- 可视化图像包含：GT、实时预测、夹爪十字。

---

## 2) 当前代码基线（用于定位改动）

### 2.1 预测信息来源

- `occ_grasp_models/agents/act_bc_keypoint/act_bc_keypoint_agent.py:317-406`
  - `act()` 会把 `pred_info` 放入 `ActResult.info`。
  - `pred_info` 对 keypoint-only 只包含关键点相关信息（不含 strategy/phase 文本字段）。

### 2.2 现有可视化入口

- `occ_grasp_models/helpers/custom_rlbench_env.py:135-167`（单任务）
- `occ_grasp_models/helpers/custom_rlbench_env.py:665-697`（多任务）
  - 现状是 env 内 `_maybe_save_aux_sample()` 直接采样保存。
  - 现有采样带 phase 均衡逻辑，不符合本次“从回流样本中均匀抽样”的目标。

### 2.3 绘制函数能力

- `occ_grasp_models/helpers/aux_eval_visualizer.py:46-243`
  - 已支持渲染 GT/预测关键点。
  - 已支持基于相机内外参绘制左右夹爪十字。
  - `strategy_name/phase_name` 只有在 `pred_info` 提供时才绘制；keypoint-only 默认不绘制。

### 2.4 与旧文档口径对齐

- `docs/method_guides/ACT_BC_KEYPOINT_STRATEGY_AUX_EVAL_PLAN.md` 第 7 节已明确：`ACT_BC_KEYPOINT` 只保留 `aux_eval/kp_2d`，无 strategy/phase 文本输出。

---

## 3) 目标流程（C1 一体化）

```
在线 rollout step
  └─> collector.maybe_add_step(...)
        ├─ 保存干净 RGB（训练用）
        ├─ 保存结构化 GT 到 samples.jsonl（训练用）
        └─ 缓存 pred_info 引用（可视化候选）

episode 结束
  └─> collector.end_episode(...)
        ├─ 在该 episode 已采集样本内均匀抽样 K 帧
        ├─ 调用 render_aux_eval_like 生成标注图 aux_vis/*.png
        └─ 记录 aux_vis_manifest.jsonl
```

关键点：

- 训练使用 `images/* + samples.jsonl`。
- 可视化使用 `aux_vis/* + aux_vis_manifest.jsonl`。
- 两者目录分离，语义分离。

---

## 4) 数据协议

### 4.1 目录结构

```text
{save_root}/round_000/{task}/episode_000012/
  images/
    over_shoulder_left/rgb_0000.png
    over_shoulder_right/rgb_0000.png
    overhead/rgb_0000.png
    wrist_right/rgb_0000.png
    wrist_left/rgb_0000.png
    front/rgb_0000.png
  samples.jsonl
  aux_vis/
    ep12_u000.png
    ep12_u001.png
  aux_vis_manifest.jsonl
```

### 4.2 `samples.jsonl`（继续预训练消费）

最小字段：

- `task`, `episode_seed`, `step_id`
- `image_paths`（6 相机）
- `has_affordance`
- `{cam}_contact_2d`, `{cam}_grasp_2d`, `{cam}_affordance_2d`
- `{cam}_contact_visible`, `{cam}_grasp_visible`, `{cam}_affordance_visible`

禁止字段：

- `strategy_type`
- `phase_type`

### 4.3 `aux_vis_manifest.jsonl`（可视化追踪）

最小字段：

- `task`, `episode_seed`, `step_id`
- `sample_index_in_episode`
- `vis_path`
- `sample_camera`

---

## 5) 采样策略

### 5.1 回流采样（训练样本）

采用前缀优先规则：

1. 候选步：`step_id % collect_every_n_steps == 0`
2. 窗口：`step_id < collect_window_steps`
3. 上限：按 step 递增保留前 `max_frames_per_episode` 个候选

这保证回流样本聚焦在线评估开头阶段。

### 5.2 辅助可视化采样（标注图）

在 `end_episode()` 阶段执行：

- 从该 episode 已保存回流样本索引集合中均匀抽样 `num_samples_per_episode`。
- 均匀抽样示例：`idx = linspace(0, n-1, K).round().astype(int)`。
- 每个抽中样本调用 `render_aux_eval_like()` 生成标注图。

这保证可视化覆盖该 episode 采样分布，而不额外偏离训练样本来源。

---

## 6) 配置设计（`eval.yaml`）

```yaml
dagger_collect:
  enabled: false
  round_id: 0
  save_root: /home/hdliu/arm_test/multi_task/dagger_data
  collect_every_n_steps: 2
  collect_window_steps: 80
  max_frames_per_episode: 40
  collect_outcome: both
  aux_vis:
    enabled: false
    num_samples_per_episode: 6
    sample_camera: over_shoulder_right
    save_path: "${cinematic_recorder.save_path}/dagger_aux_samples"
    sample_min_width: 300
    sample_output_scale: 0.0
```

配置解释：

- `collect_every_n_steps` + `collect_window_steps` + `max_frames_per_episode`：控制前缀聚焦。
- `collect_outcome`：统一控制采样结果范围（`both` / `success_only` / `fail_only`）。
- `aux_vis.num_samples_per_episode`：每个 episode 可视化张数。
- `aux_vis.sample_camera`：叠加图展示视角（与现有 `render_aux_eval_like` 一致）。

---

## 7) 代码改动块（用于实现本方案）

> 行号基于当前快照（2026-02-19），以函数名+上下文为主。

### 7.1 `custom_rlbench_env.py`：标签门控与 pred_info 透出

**旧代码（单任务节选）** `occ_grasp_models/helpers/custom_rlbench_env.py:93-96, 382-385, 436`

```python
cfg = self._aux_eval_cfg
if cfg is None or not bool(getattr(cfg, "enabled", False)):
    return obs_dict
...
if self._previous_obs_dict is not None:
    self._aux_eval_step += 1
    self._maybe_save_aux_sample(self._previous_obs_dict, self._last_pred_info)
...
return Transition(obs, reward, terminal, summaries=summaries)
```

**新代码（节选）**

```python
cfg = self._active_label_cfg()  # aux_eval or dagger_collect
if cfg is None:
    return obs_dict
...
if self._previous_obs_dict is not None:
    dagger_on = self._dagger_collect_cfg is not None and bool(getattr(self._dagger_collect_cfg, "enabled", False))
    if not dagger_on:
        self._aux_eval_step += 1
        self._maybe_save_aux_sample(self._previous_obs_dict, self._last_pred_info)
...
info = {}
if self._last_pred_info is not None:
    info["pred_info"] = self._last_pred_info
return Transition(obs, reward, terminal, info=info, summaries=summaries)
```

意义：

- C1 模式由 collector 统一负责可视化抽样。
- 保留 legacy aux 路径用于非 dagger 场景。

### 7.2 `_independent_env_runner.py`：接入 collector 生命周期

**旧代码（节选）** `repos/YARR/yarr/runners/_independent_env_runner.py:122-130, 268-309`

```python
def _run_eval_independent(..., cinematic_recorder_cfg=None, aux_eval_cfg=None):
    ...
    for ep in range(self._eval_episodes):
        ...
        for replay_transition in generator:
            ...
```

**新代码（节选）**

```python
def _run_eval_independent(..., cinematic_recorder_cfg=None, aux_eval_cfg=None, dagger_collect_cfg=None):
    collector = DaggerDataCollector(dagger_collect_cfg) if _enabled(dagger_collect_cfg) else None
    ...
    for ep in range(self._eval_episodes):
        if collector is not None:
            collector.start_episode(task=task_name, episode_seed=eval_demo_seed)
        try:
            for step_id, replay_transition in enumerate(generator):
                if collector is not None:
                    collector.maybe_add_step(
                        task=task_name,
                        episode_seed=eval_demo_seed,
                        step_id=step_id,
                        obs=replay_transition.observation,
                        reward=replay_transition.reward,
                        terminal=replay_transition.terminal,
                        pred_info=(replay_transition.info or {}).get("pred_info"),
                    )
            if collector is not None:
                collector.end_episode(success=(reward > 0.99), aborted=False)
        except Exception:
            if collector is not None:
                collector.end_episode(success=False, aborted=True)
            raise
    if collector is not None:
        collector.close()
```

意义：

- 评估循环只插桩，不改原有成功率和视频逻辑。
- 可视化抽样改为 episode 结束后统一执行。

### 7.3 新增 `dagger_data_collector.py`：核心实现点

**新增代码骨架**

```python
class DaggerDataCollector:
    def maybe_add_step(...):
        if step_id >= self.collect_window_steps:
            return
        if step_id % self.collect_every_n_steps != 0:
            return
        if len(self._saved_records) >= self.max_frames_per_episode:
            return
        self._save_clean_rgb(obs)
        self._append_samples_jsonl(...)
        self._saved_records.append({"step_id": step_id, "obs": obs, "pred_info": pred_info})

    def end_episode(...):
        if not self._should_keep_episode(success):
            self._discard_episode_files()
            return
        vis_ids = _uniform_pick_indices(len(self._saved_records), self.aux_vis_num)
        for vis_i, rec_id in enumerate(vis_ids):
            rec = self._saved_records[rec_id]
            render_aux_eval_like(rec["obs"], rec["pred_info"], self.aux_vis_cfg, vis_path)
            self._append_aux_manifest(...)
```

意义：

- “干净样本”和“标注样本”从代码层面彻底分离。
- 满足每个 episode 均匀抽样可视化要求。

---

## 8) 验收标准

1. `samples.jsonl` 无 `strategy_type/phase_type` 字段。  
2. `images/` 中 PNG 无任何绘制标注。  
3. `aux_vis/` 图像包含 GT/预测/夹爪十字。  
4. 每个 episode 的 `aux_vis` 来自本 episode 的已采回流样本均匀抽样。  
5. 关闭 `dagger_collect.enabled` 后，评估流程行为与当前版本一致。

---

## 9) 与主架构文档关系

本方案是 `docs/architecture/ACT_BC_KEYPOINT_MODEL_DIAGRAM.md` 第 5.5、8.4、10.11 的实现细则。
