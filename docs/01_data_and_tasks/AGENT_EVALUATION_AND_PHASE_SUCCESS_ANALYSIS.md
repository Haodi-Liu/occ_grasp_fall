# occ_grasp_fall agents 评估机制与四阶段成功率链条分析

**探查日期**: 2026-05-05  
**参考背景**: `docs/01_data_and_tasks/DATA_COLLECTION_GUIDE.md`  
**建议运行环境**: `conda activate ppi`，尤其是 PPI 评估入口  
**范围**: `occ_grasp_models`、`repos/YARR`、`repos/RLBench` 中当前本地代码。本文只分析机制，不修改代码。

## 1. 关键结论

1. 所有已注册的 bimanual agents 最终都进入同一套 YARR 评估主循环：入口脚本创建 agent 和 `CustomRLBenchEnv`，`RolloutGenerator` 逐步调用 `agent.act()` 与 `env.step()`，`_IndependentEnvRunner` 汇总指标并写入 `eval_data.csv`。
2. `eval.py` 是通用入口，覆盖 `BIMANUAL_PERACT`、ACT 系列、Diffusion Policy、OpenPI 等。`eval_ppi.py` 是 PPI 专用入口，主要差异是直接使用 PPI 配置、PPI 权重目录和 `method.policy.n_obs_steps`。
3. 任务物理成功来自 RLBench `Task.success()`，当前四个遮挡抓取任务的最终成功条件本质上都是 `LiftedCondition`。`CustomRLBenchEnv.step()` 把成功奖励从 `1.0` 放大到 `100.0`，所以二值任务中 `eval_envs/return` 近似等于任务成功百分比。
4. 四阶段成功率不是从离线 demo 的 `obs.misc["phase_type"]` 直接统计出来的，而是在在线评估 rollout 中，通过任务实例内的 `PhasedSuccessEvaluator` 实时推进状态机，episode 结束后读取 `phase_status` 计算。
5. `eval_envs/phase_i_success_rate = 完成第 i 阶段的 episode 数 / total_episodes`。分母是 `success_count + failed_count`，其中当前代码里的 `success_count` 更准确说是“正常跑完并进入统计流程的 episode 数”，不是严格的任务成功数。
6. 当前四阶段设计采用“阶段里程碑锁存”的思想：前一阶段条件一旦满足就记录为完成，后续即使该物理条件失效也不回退。这是为了避免例如手机被抓起后“边缘悬空”条件自然失效，导致后续阶段被误判为失败。
7. `eval_envs/success_rate` 的命名容易误导。代码层它按 `success_count / (success_count + failed_count)` 计算，不能单独当作任务成功率。任务成功更应看 `eval_envs/return / 100` 或 scheme 分层指标里基于 `reward > 0.99` 的成功计数。

## 2. Agent 评估总链条

### 2.1 通用入口 `eval.py`

`occ_grasp_models/eval.py` 的 `eval_seed()` 做四件事：

1. 通过 `agent_factory.create_agent(train_cfg)` 创建 agent。
2. 创建 `SimpleAccumulator` 和 `IndependentEnvRunner`。
3. 根据 `eval_type` 选择 checkpoint：`missing`、`best`、`last`、`all` 或具体整数 checkpoint。
4. 对每个待评估权重启动 `env_runner.start(...)`。

关键代码位置：

- `occ_grasp_models/eval.py:42` 创建 agent。
- `occ_grasp_models/eval.py:54` 创建 `IndependentEnvRunner`。
- `occ_grasp_models/eval.py:81` 到 `147` 选择待评估权重。
- `occ_grasp_models/eval.py:180` 到 `198` 启动评估进程。

`main()` 负责 Hydra 配置、加载训练时 `config.yaml`、检查 method/task 一致性、构造 RLBench action mode、任务类和观测配置。

### 2.2 PPI 专用入口 `eval_ppi.py`

`occ_grasp_models/eval_ppi.py` 的流程和 `eval.py` 基本同构，但有几个 PPI 相关差异：

- `agent_factory.create_agent(eval_cfg)` 直接用评估配置创建 PPI agent。
- `weightsdir` 来自 `eval_cfg.framework.weightsdir`，不是从训练 logdir 下的 `weights` 推导。
- `time_steps` 使用 `eval_cfg.method.policy.n_obs_steps`。
- 当前只接受整数型 `framework.eval_type`。

关键代码位置：

- `occ_grasp_models/eval_ppi.py:55` 创建 PPI agent。
- `occ_grasp_models/eval_ppi.py:60` 创建 `IndependentEnvRunner`。
- `occ_grasp_models/eval_ppi.py:88` 到 `93` 解析整数 checkpoint。
- `occ_grasp_models/eval_ppi.py:108` 到 `117` 启动 runner。

### 2.3 Agent 工厂与支持列表

`occ_grasp_models/agents/agent_factory.py` 中当前支持的 bimanual methods 包括：

- `BIMANUAL_PERACT`
- `ACT_BC_VISION`
- `ACT_BC_ENC`
- `ACT_BC_KEY`
- `ACT_BC_KEYPOINT`
- `ACT_BC_KEYPOINT_STRATEGY`
- `ACT_BC_ENC_STRATEGY`
- `ACT_BC_ENC_KEYPOINT`
- `ACT_BC_ENC_KEYPOINT_STRATEGY`
- `DIFFUSION_POLICY`
- `PPI`
- `OPENPI_POLICY`

这些 method 在评估链条中的共同接口是 YARR `Agent`：

- `build(training=False, device=...)`
- `load_weights(savedir)`
- `reset()`
- `act(step, observation, deterministic=True) -> ActResult`

不同 agent 的差异主要在 `act()` 如何从 observation 生成动作：

- PerAct 输出离散 Q argmax，再由 stack agent 转成双臂动作。
- ACT 系列直接或经关键点/策略辅助头预测双臂末端动作序列。
- Diffusion Policy 缓存一段动作序列，按 `n_action_steps` 逐步消耗。
- PPI 从多视角点云、DINO 特征、语言和 point flow 条件预测动作。
- OpenPI 走自己的 policy wrapper，但最终仍返回 `ActResult(action)`。

只要返回的 action 和配置的 RLBench `action_mode` 匹配，后续环境评估机制一致。

## 3. 环境与 rollout 执行链

### 3.1 Runner 创建环境

`repos/YARR/yarr/runners/independent_env_runner.py:start()` 根据 `env_config` 创建：

- 单任务：`CustomRLBenchEnv`
- 多任务：`CustomMultiTaskRLBenchEnv`

关键位置：

- `repos/YARR/yarr/runners/independent_env_runner.py:92` 判断单任务或多任务。
- `repos/YARR/yarr/runners/independent_env_runner.py:94` 到 `119` 创建 eval env。
- `repos/YARR/yarr/runners/independent_env_runner.py:143` 到 `153` 创建内部 `_IndependentEnvRunner`。
- `repos/YARR/yarr/runners/independent_env_runner.py:156` 到 `165` 进入 `_run_eval_independent()`。

### 3.2 RolloutGenerator 每步执行

`repos/YARR/yarr/utils/rollout_generator.py` 是单个 episode 的核心循环：

1. `eval=True` 时调用 `env.reset_to_demo(eval_demo_seed)`，用评估集 episode 初始状态重置场景。
2. `agent.reset()`。
3. 维护 `obs_history`，长度为 `timesteps`。
4. 每步构造 tensor observation，调用 `agent.act(..., deterministic=True)`。
5. 调用 `env.step(act_result)`。
6. 把 observation、agent 附加 replay elements、reward、terminal、info 打成 `ReplayTransition`。
7. terminal、timeout 或 `needs_reset` 时结束 episode。

关键位置：

- `repos/YARR/yarr/utils/rollout_generator.py:23` 到 `28` 重置环境和 agent。
- `repos/YARR/yarr/utils/rollout_generator.py:35` 到 `40` 调用 agent。
- `repos/YARR/yarr/utils/rollout_generator.py:49` 调用环境 step。
- `repos/YARR/yarr/utils/rollout_generator.py:56` 到 `62` 处理 timeout。
- `repos/YARR/yarr/utils/rollout_generator.py:75` 到 `78` 生成 `ReplayTransition`。

### 3.3 CustomRLBenchEnv.step 与任务成功

`CustomRLBenchEnv.step()` 做动作执行和奖励归一：

1. 从 `ActResult` 取 action 和可视化 target。
2. 调用 RLBench `TaskEnvironment.step(action, visual_targets)`。
3. 如果 RLBench reward 大于等于 1，设置 `success=True` 并乘以 `reward_scale=100.0`。
4. 否则 reward 置 0。
5. 捕获 IK、路径规划和非法动作异常，将 episode 置 terminal 且 reward 为 0。

关键位置：

- `occ_grasp_models/helpers/custom_rlbench_env.py:410` 到 `418` 执行动作并缩放 reward。
- `occ_grasp_models/helpers/custom_rlbench_env.py:419` 到 `430` 捕获动作异常。
- `repos/RLBench/rlbench/task_environment.py:111` 执行 action mode。
- `repos/RLBench/rlbench/task_environment.py:119` 到 `127` 调用任务 `success()` 并返回 reward。
- `repos/RLBench/rlbench/backend/task.py:306` 到 `319` 中，所有注册成功条件都满足才算任务成功。

## 4. 指标写入链

### 4.1 episode return 与 length

`SimpleAccumulator` 按 transition 累加 reward 和长度，在 terminal 时形成一个 episode 记录：

- `repos/YARR/yarr/utils/stat_accumulator.py:81` 到 `89` 累加 reward 和 length。
- `repos/YARR/yarr/utils/stat_accumulator.py:91` 到 `110` 生成 `eval_envs/return`、`eval_envs/length` 和 `eval_envs/total_transitions`。

由于 `CustomRLBenchEnv` 把任务成功 reward 放大到 100，二值任务中：

```text
eval_envs/return = 100 * 任务成功 episode 数 / 有效统计 episode 数
```

例如 `return=50.0` 通常表示约一半 episode 最终满足 RLBench 任务成功条件。

### 4.2 CSV 与 TensorBoard

`LogWriter` 负责把 `ScalarSummary` 写入 TensorBoard 和 CSV：

- `repos/YARR/yarr/utils/log_writer.py:45` 到 `64` 按 summary 名称分流 train/env/aux。
- `repos/YARR/yarr/utils/log_writer.py:66` 到 `90` 写 scalar、image、video、text。
- `repos/YARR/yarr/utils/log_writer.py:118` 到 `141` 写 `eval_data.csv` 或 `test_data.csv`。

在 `_IndependentEnvRunner` 中，`LogWriter` 的 env CSV 名称是：

- validation/eval：`eval_data.csv`
- test set：`test_data.csv`

## 5. 四阶段成功率的设计想法

四阶段的思想来自数据收集指南中的任务分解：平放薄物体无法直接抓，需要先创造可抓空间，再抓住，再让辅助臂撤离，最后抬起。四个阶段分别是：

| 阶段 | 名称 | 主要意义 |
| --- | --- | --- |
| 1 | PreManipulation | 利用边缘、墙面或按压使物体出现可抓空间 |
| 2 | Grasp | grasper 夹爪实际抓住并稳定目标物 |
| 3 | ClearPath | pusher 或辅助臂撤离，避免阻挡后续抬起 |
| 4 | Lift | 目标物到达任务定义的抬起高度 |

这套指标的目的不是替代最终成功率，而是诊断策略在哪一段断掉：

- phase 1 低：前置接触和物体姿态改变失败。
- phase 2 低但 phase 1 高：可抓空间有了，但抓取动作或抓取点选择失败。
- phase 3 低但 phase 2 高：双臂协同撤离或避让不足。
- phase 4 低但 phase 3 高：已抓住且清道，但抬起路径或稳定性失败。

## 6. 四阶段状态机实现

四个任务都各自复制了一份 `PhasedSuccessEvaluator`。以 `bimanual_edge_phone.py` 为例：

- `current_phase` 从 1 开始。
- `_phase_completion_status = {1: False, 2: False, 3: False, 4: False}`。
- `evaluate_current_phase()` 只检查当前阶段。
- 如果当前阶段所有条件满足，就把该阶段锁存为 True，`current_phase += 1`。
- 同一次调用会继续 while 检查后续阶段，避免某一步状态同时满足多个阶段时漏记。
- 一旦某阶段被标记完成，后续不会回退。

关键位置：

- `repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py:521` 到 `525` 初始化状态。
- `repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py:536` 到 `561` 推进状态机。
- `repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py:575` 到 `582` 返回 `phase_status`。

注意：`bimanual_edge_phone.py:515` 到 `518` 的注释仍写着“累积包含之前阶段的条件”，但当前实际 `phase_conditions` 已不是严格累积条件。实际逻辑更像“顺序锁存的非累积阶段条件”，只有 phase 3 明确复用了 `StableGraspCondition`，用于确保清道时仍保持抓取。

## 7. 各任务阶段条件

### 7.1 BimanualEdgePhone，策略 1 EdgeHang

任务文件：`repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py`

| 阶段 | 条件 |
| --- | --- |
| Phase 1 | `EdgeOverhangCondition`: `phone_edge` 相对 `box_edge` 的 overhang 大于 `0.05`，目标速度低于阈值并稳定 3 帧 |
| Phase 2 | `StableGraspCondition`: grasper 抓住 `Phone`，目标速度稳定 3 帧 |
| Phase 3 | `StableGraspCondition + ClearPathCondition`: 仍保持抓取，pusher gripper 没有抓物，pusher tip 离目标和 lift waypoint 至少 `0.34m` |
| Phase 4 | `LiftedCondition`: `Phone.z >= 1.1` |

执行计划由 `execution_phases` 动态生成：

- phase 1 用 pusher 前 3 个 waypoint。
- phase 2 用 grasper 前 3 个 waypoint。
- phase 3 用 pusher 第 4 个 waypoint。
- phase 4 用 grasper 第 4 个 waypoint。

关键位置：`bimanual_edge_phone.py:743` 到 `750` 定义阶段条件，`bimanual_edge_phone.py:841` 到 `864` 定义执行阶段。

### 7.2 BimanualPivotPhone，策略 2 WallLever

任务文件：`repos/RLBench/rlbench/bimanual_tasks/bimanual_pivot_phone.py`

| 阶段 | 条件 |
| --- | --- |
| Phase 1 | `GraspPointHeightCondition`: `grasp_pt.z >= 0.80`，目标速度稳定 3 帧 |
| Phase 2 | `StableGraspCondition` |
| Phase 3 | `StableGraspCondition + ClearPathCondition`，clearance 为 `0.15m` |
| Phase 4 | `LiftedCondition`: 目标 `z >= 0.9` |

执行计划中 pusher 有 5 个 waypoint，phase 1 使用前 4 个，phase 3 使用第 5 个清道 waypoint。

关键位置：`bimanual_pivot_phone.py:592` 到 `600` 定义阶段条件，`bimanual_pivot_phone.py:747` 到 `766` 定义执行阶段。

### 7.3 BimanualPickPlate，策略 3 PressTilt

任务文件：`repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_plate.py`

| 阶段 | 条件 |
| --- | --- |
| Phase 1 | `GraspPointHeightCondition`: `grasp_pt.z >= 0.80`，目标速度稳定 3 帧 |
| Phase 2 | `StableGraspCondition` |
| Phase 3 | `StableGraspCondition + ClearPathCondition`，clearance 为 `0.18m` |
| Phase 4 | `LiftedCondition`: `plate.z >= 0.9` |

关键位置：`bimanual_pick_plate.py:585` 到 `593` 定义阶段条件，`bimanual_pick_plate.py:672` 到 `691` 定义执行阶段。

### 7.4 BimanualPickFork，策略 3 PressTilt

任务文件：`repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_fork.py`

| 阶段 | 条件 |
| --- | --- |
| Phase 1 | `GraspPointHeightCondition`: `grasp_pt.z >= 0.75`，目标速度稳定 3 帧 |
| Phase 2 | `StableGraspCondition` |
| Phase 3 | `StableGraspCondition + ClearPathCondition`，clearance 为 `0.25m` |
| Phase 4 | `LiftedCondition`: `Fork_phy.z >= 0.88` |

关键位置：`bimanual_pick_fork.py:592` 到 `600` 定义阶段条件，`bimanual_pick_fork.py:672` 到 `691` 定义执行阶段。

## 8. 四阶段成功率计算链条

在线评估时，阶段统计发生在 `_IndependentEnvRunner._run_eval_independent()` 内。

### 8.1 每个 episode 内逐步推进阶段状态机

在每个 rollout transition 后，runner 调用：

```python
phase_completed, completed_phase = env.evaluate_current_phase()
```

对应链条：

```text
_IndependentEnvRunner._run_eval_independent()
  -> CustomRLBenchEnv.evaluate_current_phase()
     -> task.phased_evaluator.evaluate_current_phase()
```

关键位置：

- `repos/YARR/yarr/runners/_independent_env_runner.py:423` 到 `433`
- `occ_grasp_models/helpers/custom_rlbench_env.py:484` 到 `499`

这里还会把刚完成的阶段写到 `replay_transition.info["phase_completed"]`，并写 `completion_frame`。但当前 `phase_completion_frames` 虽然初始化了，并没有在最终 summary 里使用。因此 CSV 目前只有阶段成功率和平均最大阶段，没有各阶段完成帧数。

### 8.2 episode 结束后读取 phase_status

episode rollout 结束并进入正常统计流程后：

```python
phase_progress = env.get_phase_progress()
phase_status = phase_progress.get("phase_status", {})
for phase_id in range(1, 5):
    if phase_status.get(phase_id, False):
        phase_success_counts[phase_id] += 1
        current_max_phase = phase_id
max_phases_reached.append(current_max_phase)
```

关键位置：

- `repos/YARR/yarr/runners/_independent_env_runner.py:455` 到 `466`
- `occ_grasp_models/helpers/custom_rlbench_env.py:467` 到 `482`

这意味着阶段成功率统计的是“这个 episode 到结束时是否曾经完成过该阶段”，不是最后一帧是否仍满足该阶段物理条件。

### 8.3 汇总成 phase_i_success_rate

所有 episode 结束后：

```python
total_episodes = success_count + failed_count
phase_rate = phase_success_counts[phase_id] / total_episodes
summaries.append(ScalarSummary(f"eval_envs/phase_{phase_id}_success_rate", phase_rate))
avg_max_phase = sum(max_phases_reached) / len(max_phases_reached)
```

关键位置：`repos/YARR/yarr/runners/_independent_env_runner.py:625` 到 `637`。

因此四阶段成功率可以写成：

```text
phase_i_success_rate =
    count(episode 中 phase_status[i] == True)
    /
    count(正常统计 episode + 异常失败 episode)
```

`avg_max_phase` 则是每个正常进入阶段统计的 episode 的最大已完成阶段的均值。异常失败 episode 通常不会进入 `max_phases_reached`，但会进入 `total_episodes` 分母。

## 9. 最终成功率、return 与四阶段指标的关系

### 9.1 `eval_envs/return`

这是最接近最终任务成功率的指标。因为当前 task reward 是：

```text
Task.success() True  -> reward = 100.0
Task.success() False -> reward = 0.0
```

所以 `eval_envs/return / 100` 可近似理解为任务最终成功比例。

### 9.2 `eval_envs/success_count` 与 `eval_envs/success_rate`

当前代码里：

- `success_count += 1` 发生在 episode 正常进入统计流程之后。
- `failed_count += 1` 发生在 `StopIteration`、通信错误或其他异常分支。
- `success_rate = success_count / (success_count + failed_count)`。

这不是严格的任务成功率，而更接近“评估流程正常完成率”。在已有样例 CSV 中可以看到这种差异，例如 `edge_phone/PPI/seed0/eval_data.csv` 有一行 `eval_envs/return=50.0` 但 `eval_envs/success_rate=1.0`，说明任务成功约一半，但 30 个 episode 都正常跑完并被统计。

### 9.3 scheme 分层成功率

scheme 分层统计使用 `reward > 0.99` 判定任务成功：

```python
if reward > 0.99:
    scheme_stats[current_gt_scheme]["success"] += 1
```

因此：

- `success_rate_left_grasper_scenes`
- `success_rate_right_grasper_scenes`

比 `eval_envs/success_rate` 更接近真正任务成功率，只是它们按 GT scheme 分组。

相关工具在 `occ_grasp_models/helpers/scheme_utils.py`：

- `build_episode_scheme_map()` 通过 `scheme_info_left_grasper.pkl` 或 `scheme_info_right_grasper.pkl` 文件名构造 episode 到 scheme 的映射。
- `get_scheme_stats_summary()` 输出左右 scheme 成功率、总数、成功步数均值和 balance gap。

## 10. 演示数据阶段标签链条与在线评估链条的区别

`DATA_COLLECTION_GUIDE.md` 说明的链条是 demo 收集链条：

```text
get_demo()
  -> execute_waypoints_bimanual_phased(do_record)
     -> task.evaluate_phase_and_get_labels()
     -> Scene._current_strategy_type / _current_phase_type
     -> Scene._get_misc()
     -> obs.misc["strategy_type"], obs.misc["phase_type"]
```

对应实现：

- `repos/RLBench/rlbench/backend/scene.py:682` 到 `690` 每个 waypoint step 前更新标签。
- `repos/RLBench/rlbench/backend/scene.py:716` 到 `724` wait_after 期间也更新标签。
- `repos/RLBench/rlbench/backend/scene.py:887` 到 `896` 修复首帧标签。
- `repos/RLBench/rlbench/backend/scene.py:1126` 到 `1132` 写入 `strategy_type` 和 `phase_type`。

在线评估链条不同：

```text
agent rollout
  -> env.step(action)
  -> runner 调用 env.evaluate_current_phase()
  -> task.phased_evaluator.evaluate_current_phase()
  -> episode 结束后 env.get_phase_progress()
  -> phase_success_counts
```

也就是说，`phase_i_success_rate` 不是直接读 `obs.misc["phase_type"]` 得到的。`obs.misc` 阶段标签主要服务于数据保存、训练加载和辅助标签监督。

## 11. 重要边界情况与解读注意

1. **Phase 4 不必然等于最终任务成功。** 当前最终任务成功条件多为 `LiftedCondition`，但 phase 4 被状态机顺序门控。如果一个策略绕过或未被检测到 phase 1、2、3，却最终把物体抬高，`eval_envs/return` 可能成功，而 `phase_4_success_rate` 不一定计入。
2. **非累积阶段条件是有意设计。** 例如 EdgeHang 中 phase 1 的 overhang 在物体被抓起后可能失效。如果 phase 2、3、4 仍要求 phase 1 同时满足，会出现物理上成功但阶段统计卡死的情况。
3. **状态机依赖实时采样。** 如果某阶段条件只短暂满足，而没有在 runner 调用 `evaluate_current_phase()` 的时刻被看到，该阶段不会锁存。代码用 while 连续推进只能解决“同一采样时刻满足多个阶段”的漏记，不能解决“采样间隔错过瞬态”的问题。
4. **异常 episode 的阶段信息不完整。** 异常会增加 `failed_count`，进入 phase rate 分母，但通常不会贡献 `phase_success_counts`。这会降低阶段成功率。
5. **在线 obs 中的 `strategy_type/phase_type` 可能不如 demo 链条完整。** `Scene._current_strategy_type` 主要在 live demo waypoint 执行中更新；普通 agent rollout 不是走 `execute_waypoints_bimanual_phased()`，所以在线辅助评估如果依赖 obs.misc 阶段标签，需要确认是否另有路径更新它。当前四阶段成功率不依赖这一路。
6. **四个任务复制了相似 evaluator 代码。** 后续若改阶段统计语义，需要同步 `bimanual_edge_phone.py`、`bimanual_pivot_phone.py`、`bimanual_pick_plate.py`、`bimanual_pick_fork.py`，否则不同任务可能漂移。
7. **`phase_completion_frames` 尚未形成输出指标。** runner 中初始化了该字典，也给 transition info 标记了完成帧，但最终 summary 没有把它写入 CSV。

## 12. 读 `eval_data.csv` 的建议

建议按以下优先级解读：

1. `eval_envs/return / 100`：最终任务成功率的主指标。
2. `eval_envs/phase_1_success_rate` 到 `phase_4_success_rate`：阶段瓶颈定位。
3. `eval_envs/avg_max_phase`：整体推进深度，适合比较训练阶段或 checkpoint。
4. `success_rate_left_grasper_scenes` 与 `success_rate_right_grasper_scenes`：左右 grasper scheme 的泛化和偏置。
5. `eval_envs/success_rate`：更适合作为评估流程稳定性或异常率参考，不建议单独当作任务成功率。

一个典型解读方式：

```text
return 高，phase_4 低:
  可能存在绕过阶段条件的成功，或阶段状态机漏记。

phase_1 高，phase_2 低:
  预操作有效，但抓取点、夹爪闭合或稳定抓取失败。

phase_2 高，phase_3 低:
  已抓住，但辅助臂撤离距离或释放条件不足。

phase_3 高，phase_4 低:
  协作阶段已完成，但抬起路径、姿态稳定或高度不足。
```

## 13. 总链条速查

```text
eval.py / eval_ppi.py
  -> agent_factory.create_agent()
  -> IndependentEnvRunner.start()
  -> CustomRLBenchEnv / CustomMultiTaskRLBenchEnv
  -> _IndependentEnvRunner._run_eval_independent()
     -> RolloutGenerator.generator()
        -> env.reset_to_demo(eval_demo_seed)
        -> agent.reset()
        -> agent.act()
        -> env.step(ActResult)
           -> TaskEnvironment.step()
              -> action_mode.action(scene, action)
              -> task.success()
           -> CustomRLBenchEnv.extract_obs()
     -> env.evaluate_current_phase()
        -> task.phased_evaluator.evaluate_current_phase()
     -> episode end:
        -> env.get_phase_progress()
        -> phase_success_counts
     -> summaries:
        -> eval_envs/return
        -> eval_envs/success_count / failed_count / success_rate
        -> eval_envs/phase_1_success_rate ... phase_4_success_rate
        -> eval_envs/avg_max_phase
        -> scheme-stratified metrics
  -> LogWriter
     -> eval_data.csv
     -> TensorBoard
```

