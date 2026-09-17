# 在线阶段成功评估与环境真实状态视频可视化修改方案

**目标环境**: PPI 在线评估  
**计划类型**: 代码修改设计文档，暂不改运行代码  
**核心约束**:

1. 尽量保持现有 RLBench task、`CustomRLBenchEnv`、YARR runner、cinematic recorder 的调用框架。
2. 阶段成功率改为适合 online policy 随机行为的动态状态机。
3. 视频可视化只显示环境真实状态，不显示 keypoint / strategy / phase 预测结果。

---

## 1. 现状问题

当前 `PhasedSuccessEvaluator` 的主要假设是：阶段完成后锁存，不回退。它适合 scripted demo 或比较平顺的 rollout，但不适合 online policy 中常见的随机情况，例如物体重新落平、抓住后又掉落、辅助臂反复靠近目标等。

现有视频 overlay 也有一个方向偏差：`repos/YARR/yarr/utils/video_utils.py` 中 `_apply_overlay()` 读取的是 `env._last_pred_info`，这是模型预测信息链路；本次需要的是环境真实状态，包括真实关键点、夹爪 dummy、真实策略类型和真实当前阶段。

---

## 2. 术语重新定义

### 2.1 单独阶段条件

以下 `con1` 到 `con4` 指的是单独条件，不是旧代码中 `phase_conditions[phase_id]` 的条件集合。

| 条件 | 含义 | 对应当前类 |
| --- | --- | --- |
| `con1` | 预操作成功，已经创造可抓空间 | `EdgeOverhangCondition` 或 `GraspPointHeightCondition` |
| `con2` | grasper 稳定抓住目标 | `StableGraspCondition` |
| `con3` | pusher / 辅助臂已经清道 | `ClearPathCondition`，但移除 waypoint 依赖 |
| `con4` | 目标物被抬到高度阈值 | `LiftedCondition` |

### 2.2 当前阶段语义

当前阶段 `k` 表示：

```text
前一阶段 k-1 已经完成，
现在正在尝试完成阶段 k。
```

因此：

| `current_phase` | 语义 |
| --- | --- |
| `1` | 正在尝试 PreManipulation |
| `2` | Phase 1 已完成，正在尝试 Grasp |
| `3` | Phase 2 已完成，正在尝试 ClearPath |
| `4` | Phase 3 已完成，正在尝试 Lift |
| `5` | Phase 4 已完成，Complete |

`5` 是建议新增的 complete 状态。它不是第五个任务阶段，而是“四阶段全部完成”的当前状态。

### 2.3 episode 成功率统计语义

阶段成功率不看 episode 结束时的当前阶段，而看整个 episode 中最远到达过的阶段。

```text
max_current_phase_reached = 3
=> 说明 Phase 1 和 Phase 2 曾经完成
=> phase_1_success = True
=> phase_2_success = True
=> phase_3_success = False
=> phase_4_success = False
```

因此：

```python
phase_status[i] = (max_current_phase_reached > i)
```

完整成功：

```text
max_current_phase_reached == 5
=> Phase 1, 2, 3, 4 全部完成
```

---

## 3. 新阶段状态机设计

### 3.1 转换规则

| 当前阶段 | 维持条件 | 转换条件 | 成功后进入 |
| --- | --- | --- | --- |
| `1 PreManipulation` | 无 | `con1` | `2 Grasp` |
| `2 Grasp` | `con1` | `con2` | `3 ClearPath` |
| `3 ClearPath` | `con2` | `con3` | `4 Lift` |
| `4 Lift` | `con2` | `con4` | `5 Complete` |
| `5 Complete` | 无需继续判定 | 无 | 终止成功状态 |

特殊处理：

- 一旦进入 `5 Complete`，这个 episode 的任务阶段评估已经成功完成，不再检查 `con1-con4`，也不再回退。
- 如果当前处于 `3/4` 且 `con2` 失效，则立即回到 `1`，再从 `1` 开始用当前条件重新推进。
- 如果当前处于 `2` 且 `con1` 失效，则立即回到 `1`，再从 `1` 开始重新判断。

### 3.2 两个示例如何落地

示例 1，`bimanual_pick_plate`：

```text
当前 phase = 2，说明盘子已经翘起，正在抓取。
如果盘子落平导致 con1=False：
  phase 2 的维持条件失败
  current_phase 先回到 1
  再检查 con1，仍 False
  新 current_phase = 1
```

示例 2，`bimanual_edge_phone`：

```text
当前 phase = 4，说明手机已抓住、辅助臂已移开，正在抬起。
如果手机掉落导致 con2=False：
  phase 4 的维持条件失败
  current_phase 先回到 1
  再检查 con1
  如果掉落后仍有足够 overhang 且稳定，con1=True
  新 current_phase = 2
```

### 3.3 不重复调用 condition

`EdgeOverhangCondition`、`GraspPointHeightCondition`、`StableGraspCondition` 内部都有 `stable_count`。因此每个仿真 step 中每个 condition 只能调用一次，不能在一个 `evaluate_current_phase()` 内多次 `condition_met()`。

新 evaluator 需要先采样一次：

```python
cond = {
    1: con1.condition_met()[0],
    2: con2.condition_met()[0],
    3: con3.condition_met()[0],
    4: con4.condition_met()[0],
}
```

后续所有维持 / 转换判断都复用这个布尔快照。

---

## 4. 需要修改的代码位置

### 4.1 四个任务脚本

需要同步修改四个文件，因为当前四个任务各自复制了一份条件类和 `PhasedSuccessEvaluator`。

| 文件 | 主要修改 |
| --- | --- |
| `repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py` | 修改 `ClearPathCondition`、`PhasedSuccessEvaluator`、`_setup_phased_evaluator()` |
| `repos/RLBench/rlbench/bimanual_tasks/bimanual_pivot_phone.py` | 同上 |
| `repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_plate.py` | 同上 |
| `repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_fork.py` | 同上 |

以 `bimanual_edge_phone.py` 为例，当前关键位置：

| 位置 | 现有函数 / 类 |
| --- | --- |
| 约第 159-205 行 | `ClearPathCondition` |
| 约第 511-582 行 | `PhasedSuccessEvaluator` |
| 约第 704-748 行 | `_setup_phased_evaluator()` 中构造 `phase_conditions` |
| 约第 878-901 行 | `evaluate_phase_and_get_labels()` / `get_phase_progress()` |

其他三个任务对应位置：

| 文件 | `PhasedSuccessEvaluator` | `_setup_phased_evaluator()` 条件构造 | 标签接口 |
| --- | --- | --- | --- |
| `bimanual_pivot_phone.py` | 约第 393-450 行 | 约第 559-597 行 | 约第 780-797 行 |
| `bimanual_pick_plate.py` | 约第 393-450 行 | 约第 552-590 行 | 约第 705-722 行 |
| `bimanual_pick_fork.py` | 约第 397-454 行 | 约第 559-597 行 | 约第 703-718 行 |

### 4.2 YARR runner

| 文件 | 函数 / 位置 | 修改目的 |
| --- | --- | --- |
| `repos/YARR/yarr/runners/_independent_env_runner.py` | `_run_eval_independent()` 约第 295-299 行 | 阶段统计初始化仍保留，但语义改为读取 `phase_status` 的最远到达阶段 |
| 同文件 | 约第 423-434 行 | 不再在 runner 里直接推进 evaluator，避免和视频实时 callback 重复调用 |
| 同文件 | 约第 455-466 行 | episode 结束后读取 `phase_progress["phase_status"]`、`max_current_phase_reached` |
| 同文件 | 约第 625-637 行 | `phase_i_success_rate` 仍按 `phase_success_counts[i] / total_episodes` 输出 |

### 4.3 环境封装

| 文件 | 函数 / 位置 | 修改目的 |
| --- | --- | --- |
| `occ_grasp_models/helpers/custom_rlbench_env.py` | `CustomRLBenchEnv._my_callback()` 约第 316-320 行 | 每个仿真 step 推进真实阶段状态 |
| 同文件 | `CustomMultiTaskRLBenchEnv._my_callback()` 约第 869-873 行 | 多任务 / 其他 bimanual eval 同样每个仿真 step 推进真实阶段状态 |
| 同文件 | `evaluate_current_phase()` 约第 484-499、975-981 行 | 保留接口，但改为调用 / 返回新 evaluator 语义 |
| 同文件 | 新增 `get_env_overlay_state()` | 给视频 overlay 提供真实策略、阶段、3D 点位 |

### 4.4 视频工具与配置

| 文件 | 函数 / 位置 | 修改目的 |
| --- | --- | --- |
| `repos/YARR/yarr/utils/video_utils.py` | `TaskRecorder.take_snap()` 约第 51-55 行 | 捕获帧后叠加环境真实状态 |
| 同文件 | `_apply_overlay()` 约第 135-215 行 | 从预测 overlay 改为环境真实 overlay，或用 `overlay_source=env` 分支 |
| `occ_grasp_models/conf/eval_ppi.yaml`、`eval.yaml`、`eval_openpi.yaml` | `cinematic_recorder` 配置块 | 增加通用环境 overlay 配置项 |

---

## 5. 阶段评估代码修改方案

### 5.1 `ClearPathCondition` 去除 waypoint 依赖

**旧逻辑**，以 `bimanual_edge_phone.py` 为例：

```python
class ClearPathCondition(Condition):
    def __init__(self, aux_gripper, target_object, aux_tip_dummy,
                 lift_waypoints: List[Dummy] = None,
                 min_clearance: float = 0.15):
        self.aux_gripper = aux_gripper
        self.target_object = target_object
        self.aux_tip_dummy = aux_tip_dummy
        self.lift_waypoints = lift_waypoints or []
        self.min_clearance = min_clearance

    def condition_met(self):
        if len(self.aux_gripper.get_grasped_objects()) > 0:
            return False, False

        aux_tip_pos = np.array(self.aux_tip_dummy.get_position())
        target_pos = np.array(self.target_object.get_position())
        distance_to_target = np.linalg.norm(aux_tip_pos - target_pos)

        if distance_to_target < self.min_clearance:
            return False, False

        for wp_dummy in self.lift_waypoints:
            wp_pos = np.array(wp_dummy.get_position())
            distance_to_wp = np.linalg.norm(aux_tip_pos - wp_pos)
            if distance_to_wp < self.min_clearance:
                return False, False

        return True, False
```

**新逻辑**：

```python
class ClearPathCondition(Condition):
    def __init__(self, aux_gripper, target_object, aux_tip_dummy,
                 lift_waypoints: List[Dummy] = None,
                 min_clearance: float = 0.15):
        self.aux_gripper = aux_gripper
        self.target_object = target_object
        self.aux_tip_dummy = aux_tip_dummy
        # 保留参数兼容旧调用，但 condition_met 不再使用 waypoint。
        self.lift_waypoints = lift_waypoints or []
        self.min_clearance = min_clearance

    def condition_met(self):
        if len(self.aux_gripper.get_grasped_objects()) > 0:
            return False, False

        aux_tip_pos = np.array(self.aux_tip_dummy.get_position())
        target_pos = np.array(self.target_object.get_position())
        distance_to_target = np.linalg.norm(aux_tip_pos - target_pos)

        return distance_to_target >= self.min_clearance, False
```

目的：

- con3 只表示 pusher / 辅助臂没有抓物，并且离目标物足够远。
- 在线 policy 不会沿预设 lift waypoint 执行，因此 Phase 3 不应依赖 waypoint dummy。
- 保留 `lift_waypoints` 参数可以减少 `_setup_phased_evaluator()` 的改动量。

### 5.2 `_setup_phased_evaluator()` 改为单独条件

**旧逻辑**，以 `bimanual_edge_phone.py` 约第 713-748 行为例：

```python
phase1_conditions = [EdgeOverhangCondition(...)]
stable_grasp_condition = StableGraspCondition(...)
phase2_conditions = [stable_grasp_condition]
phase3_conditions = [
    stable_grasp_condition,
    ClearPathCondition(...)
]
phase4_conditions = [LiftedCondition(...)]

phase_conditions = {
    1: phase1_conditions,
    2: phase2_conditions,
    3: phase3_conditions,
    4: phase4_conditions
}
self.phased_evaluator = PhasedSuccessEvaluator(phase_conditions)
```

**新逻辑**：

```python
con1 = EdgeOverhangCondition(...)
con2 = StableGraspCondition(...)
con3 = ClearPathCondition(...)
con4 = LiftedCondition(...)

stage_conditions = {
    1: con1,
    2: con2,
    3: con3,
    4: con4,
}
self.phased_evaluator = PhasedSuccessEvaluator(stage_conditions)
```

对 `pivot_phone`、`pick_plate`、`pick_fork`：

```python
con1 = GraspPointHeightCondition(...)
con2 = StableGraspCondition(...)
con3 = ClearPathCondition(...)
con4 = LiftedCondition(...)
```

目的：

- 明确 `con1` 到 `con4` 是单独阶段条件。
- 维持条件不再通过旧的 phase condition list 隐式表达，而由 evaluator 的转换规则显式表达。

### 5.3 `PhasedSuccessEvaluator` 重写为可回退状态机

**旧逻辑**：

```python
def evaluate_current_phase(self) -> Tuple[bool, int]:
    if self.current_phase > self.num_phases:
        return True, self.num_phases

    any_completed = False
    last_completed_phase = 0

    while self.current_phase <= self.num_phases:
        conditions = self.phase_conditions.get(self.current_phase, [])
        all_met = all(cond.condition_met()[0] for cond in conditions)

        if all_met:
            self._phase_completion_status[self.current_phase] = True
            last_completed_phase = self.current_phase
            self.current_phase += 1
            any_completed = True
        else:
            break

    if any_completed:
        return True, last_completed_phase
    return False, self.current_phase
```

**新逻辑草案**：

```python
PHASE_NAMES = {
    1: "PreManipulation",
    2: "Grasp",
    3: "ClearPath",
    4: "Lift",
    5: "Complete",
}

class PhasedSuccessEvaluator:
    def __init__(self, stage_conditions: Dict[int, Condition]):
        self.stage_conditions = stage_conditions
        self.num_phases = 4
        self.current_phase = 1
        self.max_current_phase_reached = 1
        self._phase_completion_status = {i: False for i in range(1, 5)}
        self._last_condition_status = {i: False for i in range(1, 5)}

    def reset(self):
        self.current_phase = 1
        self.max_current_phase_reached = 1
        self._phase_completion_status = {i: False for i in range(1, 5)}
        self._last_condition_status = {i: False for i in range(1, 5)}
        for cond in self.stage_conditions.values():
            if hasattr(cond, "reset"):
                cond.reset()

    def _sample_conditions_once(self):
        status = {}
        for phase_id, cond in self.stage_conditions.items():
            status[phase_id] = bool(cond.condition_met()[0])
        self._last_condition_status = status
        return status

    def _maintenance_met(self, phase, cond):
        if phase == 1:
            return True
        if phase == 2:
            return cond[1]
        if phase in (3, 4):
            return cond[2]
        return True

    def _transition_met(self, phase, cond):
        if phase == 1:
            return cond[1]
        if phase == 2:
            return cond[1] and cond[2]
        if phase == 3:
            return cond[2] and cond[3]
        if phase == 4:
            return cond[2] and cond[4]
        return False

    def evaluate_current_phase(self) -> Tuple[bool, int]:
        old_phase = self.current_phase

        # Complete 是终止成功状态。一旦进入 5，不再继续判定，也不回退。
        if self.current_phase >= 5:
            return False, 4

        cond = self._sample_conditions_once()

        # 当前阶段维持条件失败，先回到 1。
        if self.current_phase < 5 and not self._maintenance_met(self.current_phase, cond):
            self.current_phase = 1

        # 从当前阶段继续推进；如果刚回到 1，也会从 1 重新推进。
        while self.current_phase <= 4 and self._transition_met(self.current_phase, cond):
            self.current_phase += 1

        self.max_current_phase_reached = max(
            self.max_current_phase_reached, self.current_phase
        )

        for phase_id in range(1, 5):
            self._phase_completion_status[phase_id] = (
                self.max_current_phase_reached > phase_id
            )

        changed = self.current_phase != old_phase
        completed_phase = min(max(self.current_phase - 1, 0), 4)
        return changed, completed_phase

    def get_current_phase(self) -> int:
        return self.current_phase

    def is_task_successful(self) -> bool:
        return self.current_phase >= 5

    def get_phase_progress(self) -> Dict:
        return {
            "current_phase": self.current_phase,
            "current_phase_name": PHASE_NAMES.get(self.current_phase, "Unknown"),
            "max_current_phase_reached": self.max_current_phase_reached,
            "max_completed_phase": max(self.max_current_phase_reached - 1, 0),
            "total_phases": 4,
            "completed": self.current_phase >= 5,
            "phase_status": self._phase_completion_status.copy(),
            "condition_status": self._last_condition_status.copy(),
        }
```

目的：

- `current_phase` 在 1-4 内可回退，用于实时诊断和视频显示；进入 5 后视为任务成功并终止判定。
- `max_current_phase_reached` 只前进不回退，用于 episode 最终阶段成功率。
- `phase_status` 表示 episode 中是否曾经完成该阶段，而不是最后一帧是否仍满足。
- `condition_status` 保留为日志 / debug 信息，不进入视频 overlay 默认显示。

注意：

- 回退时不调用各 condition 的 `reset()`，因为各 condition 自己会在 `condition_met()` 中根据真实状态维护 `stable_count`。
- 进入 `current_phase=5` 后不再采样 `con1-con4`，后续物理状态变化不再改变阶段评估结果。
- `reset()` 只在 episode reset 时调用。

### 5.4 `evaluate_phase_and_get_labels()`

**旧逻辑**：

```python
def evaluate_phase_and_get_labels(self) -> Tuple[int, int]:
    strategy_type = self.STRATEGY_TYPE
    if self.phased_evaluator is None:
        phase_type = 1
    else:
        self.phased_evaluator.evaluate_current_phase()
        phase_type = self.phased_evaluator.get_current_phase()
    return strategy_type, phase_type
```

**新逻辑建议**：

```python
def evaluate_phase_and_get_labels(self) -> Tuple[int, int]:
    strategy_type = self.STRATEGY_TYPE
    if self.phased_evaluator is None:
        phase_type = 1
    else:
        self.phased_evaluator.evaluate_current_phase()
        phase_type = self.phased_evaluator.get_current_phase()
    return strategy_type, phase_type
```

代码形态可以基本不变，但语义改变：

- `phase_type` 现在可能为 `5`，表示 complete。
- PPI 不依赖 `phase_type` 训练标签，因此对 PPI online eval 是安全的。
- 如果后续把这套逻辑用于 ACT strategy/phase 监督，需要同步 `num_phases` 和标签处理，或者在 demo 标签链路中把 `5` 映射回 `4`。

---

## 6. online eval 中阶段推进的调用点

### 6.1 为什么不建议继续只在 runner 中推进

当前 runner 在每个 `ReplayTransition` 后调用：

```python
phase_completed, completed_phase = env.evaluate_current_phase()
```

位置：`repos/YARR/yarr/runners/_independent_env_runner.py` 约第 423-434 行。

这个调用发生在 `env.step(action)` 完成后。一个 action 内部可能包含多次 `scene.step()`，视频 recorder 也是在这些内部 step 里抓帧。因此如果只在 runner 里推进 phase，视频帧上的 phase 会滞后。

### 6.2 新调用点

建议把真实阶段推进放到 `CustomRLBenchEnv._my_callback()` 和 `CustomMultiTaskRLBenchEnv._my_callback()` 的开头。

**旧逻辑**：

```python
def _my_callback(self):
    if self._record_current_episode:
        self._record_cam.handle_explicitly()
        cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(cap)
```

**新逻辑**：

```python
def _update_phase_evaluation(self):
    task = self._task._task if self._task is not None else None
    if task is not None and hasattr(task, "phased_evaluator") and task.phased_evaluator is not None:
        return task.phased_evaluator.evaluate_current_phase()
    return False, 0

def _my_callback(self):
    # 每个仿真 step 先推进真实阶段状态，保证视频 overlay 与环境状态对齐。
    self._update_phase_evaluation()

    if self._record_current_episode:
        self._record_cam.handle_explicitly()
        cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(cap)
```

目的：

- 阶段状态与每个物理 step 对齐。
- cinematic recorder 的 `TaskRecorder.take_snap()` 在 action mode 中发生在 `scene.step()` 后，因此能读到刚更新的阶段。
- 该 callback 所有 eval episode 都会运行，不依赖是否保存视频；阶段成功率统计不会只覆盖被录制 episode。

### 6.3 runner 中对应改法

**旧逻辑**：

```python
if hasattr(env, 'evaluate_current_phase'):
    phase_completed, completed_phase = env.evaluate_current_phase()
    if phase_completed:
        already_recorded = completed_phase in [
            r.info.get('phase_completed') for r in episode_rollout
        ]
        if not already_recorded:
            replay_transition.info['phase_completed'] = completed_phase
            replay_transition.info['completion_frame'] = len(episode_rollout)
```

**新逻辑**：

```python
if hasattr(env, "get_phase_progress"):
    phase_progress = env.get_phase_progress()
    if phase_progress is not None:
        replay_transition.info["phase_progress"] = phase_progress
```

目的：

- runner 不再推进 evaluator，只读取环境 callback 已经维护的状态。
- 避免同一个仿真 step 重复调用 `condition_met()`，导致 `stable_count` 被重复累加。
- `phase_completed` / `completion_frame` 原本也没有写入最终 CSV，可以先不继续维护。

episode 结束统计可以保持现有结构：

```python
phase_progress = env.get_phase_progress()
phase_status = phase_progress.get("phase_status", {})
for phase_id in range(1, 5):
    if phase_status.get(phase_id, False):
        phase_success_counts[phase_id] += 1
max_phases_reached.append(phase_progress.get("max_completed_phase", 0))
```

这里的 `phase_status` 是一个 episode 级别的阶段完成字典，内容固定为四个任务阶段：

```python
{
    1: bool,  # episode 内是否曾经完成 Phase 1 / PreManipulation
    2: bool,  # episode 内是否曾经完成 Phase 2 / Grasp
    3: bool,  # episode 内是否曾经完成 Phase 3 / ClearPath
    4: bool,  # episode 内是否曾经完成 Phase 4 / Lift
}
```

它来自：

```text
env.get_phase_progress()
  -> task.get_phase_progress()
     -> task.phased_evaluator.get_phase_progress()
        -> PhasedSuccessEvaluator._phase_completion_status
```

也就是说，`phase_status` 与 `PhasedSuccessEvaluator` 对象里的 `self._phase_completion_status` 直接对应；`get_phase_progress()` 返回的是它的 copy：

```python
"phase_status": self._phase_completion_status.copy()
```

`_phase_completion_status` 在 `evaluate_current_phase()` 中由 `max_current_phase_reached` 推导：

```python
self._phase_completion_status[phase_id] = (
    self.max_current_phase_reached > phase_id
)
```

因此它不是当前帧 `con1/con2/con3/con4` 的瞬时真假，也不是 `condition_status`。它表示“这个 episode 历史上是否曾经到达过下一阶段”。例如：

```text
max_current_phase_reached = 4
=> phase_status = {1: True, 2: True, 3: True, 4: False}
```

只有当 `current_phase` 曾经到达 `5 Complete` 时，`phase_status[4]` 才会变成 `True`。

---

## 7. 视频可视化设计

### 7.1 显示内容

每个录制帧显示环境真实状态：

| 类型 | 来源 | 显示方式 |
| --- | --- | --- |
| 策略类型 | `task.STRATEGY_TYPE` | 左上角文本，例如 `PressTilt` |
| 当前执行阶段 | `task.phased_evaluator.get_phase_progress()` | 左上角文本，例如 `3 ClearPath` |
| 三个空间关键点 | CoppeliaSim dummy | 投影到 cinematic camera 后画点和标签 |
| 左右夹爪 dummy | `Panda_leftArm_tip` / `Panda_rightArm_tip` | 投影到 cinematic camera 后画点和标签 |

左上角推荐合并成一行紧凑显示：

```text
PressTilt | 3 ClearPath
```

不在视频 overlay 中显示 `condition_status`，避免占用画面空间；如果需要排查阶段状态，可通过日志或 `phase_progress` 调试。

关键点来源：

| 名称 | dummy 候选 |
| --- | --- |
| `contact` | `push_pt` 或 `press_pt` |
| `grasp` | `grasp_pt` |
| `affordance` | `box_edge` 或 `wall_pivot`，无 affordance 的任务不画 |
| `left_tip` | `Panda_leftArm_tip` |
| `right_tip` | `Panda_rightArm_tip` |

### 7.2 新增环境真实状态接口

在 `CustomRLBenchEnv` 和 `CustomMultiTaskRLBenchEnv` 中新增同名方法：

```python
def get_env_overlay_state(self) -> Dict:
    task = self._task._task if self._task is not None else None
    if task is None:
        return {}

    phase_progress = None
    if hasattr(task, "get_phase_progress"):
        phase_progress = task.get_phase_progress()

    return {
        "strategy_type": getattr(task, "STRATEGY_TYPE", 1),
        "strategy_name": STRATEGY_NAMES.get(getattr(task, "STRATEGY_TYPE", 1), "Unknown"),
        "phase_progress": phase_progress,
        "points_3d": self._collect_overlay_points_3d(),
    }
```

辅助方法：

```python
def _collect_overlay_points_3d(self) -> Dict[str, np.ndarray]:
    points = {}

    for name in ("push_pt", "press_pt"):
        if Object.exists(name):
            points["contact"] = np.array(Dummy(name).get_position(), dtype=np.float32)
            break

    if Object.exists("grasp_pt"):
        points["grasp"] = np.array(Dummy("grasp_pt").get_position(), dtype=np.float32)

    for name in ("box_edge", "wall_pivot"):
        if Object.exists(name):
            points["affordance"] = np.array(Dummy(name).get_position(), dtype=np.float32)
            break

    if Object.exists("Panda_leftArm_tip"):
        points["left_tip"] = np.array(Dummy("Panda_leftArm_tip").get_position(), dtype=np.float32)
    if Object.exists("Panda_rightArm_tip"):
        points["right_tip"] = np.array(Dummy("Panda_rightArm_tip").get_position(), dtype=np.float32)

    return points
```

注意：

- 这条链路不读 `pred_info`。
- 也不读 `obs.misc["phase_type"]`，因为 online rollout 的 `obs.misc` 标签链路不一定完整。
- phase 直接来自任务里的真实 `PhasedSuccessEvaluator`。

### 7.3 `TaskRecorder._apply_overlay()` 改造

**旧逻辑**：

```python
if self._env is None or not hasattr(self._env, "_last_pred_info"):
    return frame_rgb
pred_info = getattr(self._env, "_last_pred_info")
if not pred_info:
    return frame_rgb

strategy_name = pred_info.get("strategy_name")
phase_name = pred_info.get("phase_name")
...
kp_by_cam = pred_info.get("keypoints_2d", {})
```

**新逻辑**：

```python
state = {}
if self._env is not None and hasattr(self._env, "get_env_overlay_state"):
    state = self._env.get_env_overlay_state()
if not state:
    return frame_rgb

frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

phase_progress = state.get("phase_progress") or {}
phase = phase_progress.get("current_phase", 1)
phase_name = phase_progress.get("current_phase_name", "PreManipulation")
strategy_name = state.get("strategy_name", "Unknown")

cv2.putText(
    frame_bgr,
    f"{strategy_name} | {phase} {phase_name}",
    (20, 40),
    cv2.FONT_HERSHEY_DUPLEX,
    0.8,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
)

points_3d = state.get("points_3d", {})
for name, point in points_3d.items():
    uv, visible = self._project_world_to_camera(point)
    if visible:
        self._draw_marker(frame_bgr, name, uv)

return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
```

新增投影函数放在 `TaskRecorder` 内部即可，复用当前 cinematic camera 的参数：

```python
def _project_world_to_camera(self, point_3d):
    cam = self._cam_motion.cam
    extrinsic = cam.get_matrix()
    intrinsic = cam.get_intrinsic_matrix()
    width, height = cam.get_resolution()

    R = extrinsic[:3, :3]
    C = extrinsic[:3, 3:4]
    R_inv = R.T
    extrinsics_w2c = np.concatenate([R_inv, -(R_inv @ C)], axis=-1)
    proj = intrinsic @ extrinsics_w2c

    p = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    q = proj @ p
    if q[2] <= 0:
        return np.array([-1, -1]), False

    u = q[0] / q[2]
    v = q[1] / q[2]
    visible = 0 <= u < width and 0 <= v < height
    return np.array([u, v]), visible
```

### 7.4 marker 样式

建议颜色：

| 点 | 颜色 |
| --- | --- |
| `contact` | 红色 |
| `grasp` | 绿色 |
| `affordance` | 蓝色 |
| `left_tip` | 黄色 |
| `right_tip` | 紫色 |

绘制规则：

```python
cv2.circle(frame_bgr, (u, v), radius, color, -1)
cv2.circle(frame_bgr, (u, v), radius + 2, (255, 255, 255), 1)
cv2.putText(frame_bgr, label, (u + 8, v - 8), ...)
```

如果需要显示空间坐标，可在右上角增加小表：

```text
contact:   x y z
grasp:     x y z
left_tip:  x y z
right_tip: x y z
```

默认建议只显示 marker + label，避免遮挡视频主体；坐标表通过配置开关控制。

### 7.5 评估配置

这套 overlay 不应绑定到某一个 agent。只要评估流程使用 `cinematic_recorder` 和同一个 RLBench/YARR 环境封装，就应该走同一套环境真实状态 overlay。

当前需要覆盖三类配置入口：

| 使用场景 | 入口 | 配置文件 | 相关 agent |
| --- | --- | --- | --- |
| PPI online eval | `occ_grasp_models/eval_ppi.py` | `occ_grasp_models/conf/eval_ppi.yaml` | `occ_grasp_models/agents/ppi` |
| 常规 eval | `occ_grasp_models/eval.py` | `occ_grasp_models/conf/eval.yaml` | `occ_grasp_models/agents/act_bc_vision`、`occ_grasp_models/agents/bimanual_peract` |
| OpenPI eval | `occ_grasp_models/eval.py --config-name eval_openpi` | `occ_grasp_models/conf/eval_openpi.yaml` | `occ_grasp_models/agents/openpi_policy` |

这些配置中都已有或可加入同名 `cinematic_recorder` 配置块。以 `eval_ppi.yaml` 为例，当前已有：

```yaml
cinematic_recorder:
    enabled: True
    camera_resolution: [1280, 720]
    fps: 30
    rotate_speed: 0.005
    save_path: ${framework.logdir}/${rlbench.task_name}/${framework.weight_name}
    max_success_videos: 10
    max_fail_videos: 10
```

建议在 `eval_ppi.yaml`、`eval.yaml`、`eval_openpi.yaml` 的 `cinematic_recorder` 下统一新增：

```yaml
cinematic_recorder:
    overlay_enabled: True
    overlay_source: "env"
    overlay_draw_keypoints: True
    overlay_draw_grippers: True
    overlay_draw_xyz_table: False
```

目的：

- 明确这次 overlay 来自环境真实状态。
- 不依赖 `aux_eval`。
- 不依赖模型是否输出 `pred_info`。
- 保证 PPI、ACT_BC_VISION、BIMANUAL_PERACT、OPENPI_POLICY 的评估视频表现一致。

---

## 8. 与各 agent 的关系

各 agent 的动作生成仍然只走现有链路：

```text
RolloutGenerator.generator()
  -> agent.act(...)
  -> env.step(act_result)
```

本方案不修改：

- `occ_grasp_models/agents/agent_factory.py`
- `occ_grasp_models/agents/ppi`
- `occ_grasp_models/agents/act_bc_vision`
- `occ_grasp_models/agents/bimanual_peract`
- `occ_grasp_models/agents/openpi_policy`
- 各 agent 的输入 observation 协议
- checkpoint 加载和动作执行逻辑

新增阶段评估和视频 overlay 都属于环境真实状态的后验分析 / 可视化，不给 policy 提供额外信息。

---

## 9. 修改后的数据流

### 9.1 阶段评估数据流

```text
scene.step()
  -> CustomRLBenchEnv._my_callback()
     -> task.phased_evaluator.evaluate_current_phase()
        -> 采样 con1/con2/con3/con4
        -> 根据维持条件和转换条件更新 current_phase
        -> 更新 max_current_phase_reached
        -> 更新 phase_status

episode end
  -> _IndependentEnvRunner._run_eval_independent()
     -> env.get_phase_progress()
     -> phase_success_counts
     -> eval_envs/phase_i_success_rate
     -> eval_envs/avg_max_phase
```

### 9.2 视频 overlay 数据流

```text
action_mode path step
  -> scene.step()
     -> CustomRLBenchEnv._my_callback()
        -> 更新真实 phase
  -> TaskRecorder.take_snap(scene.get_observation())
     -> env.get_env_overlay_state()
        -> task.STRATEGY_TYPE
        -> task.get_phase_progress()
        -> Dummy positions
     -> project 3D points to cinematic camera
     -> draw text and markers
```

---

## 10. 验证计划

### 10.1 静态检查

```bash
python -m py_compile \
  repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py \
  repos/RLBench/rlbench/bimanual_tasks/bimanual_pivot_phone.py \
  repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_plate.py \
  repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_fork.py \
  occ_grasp_models/helpers/custom_rlbench_env.py \
  repos/YARR/yarr/runners/_independent_env_runner.py \
  repos/YARR/yarr/utils/video_utils.py
```

### 10.2 单 episode smoke test

建议先用 1 个 episode、1 个 checkpoint：

```bash
conda activate ppi
python occ_grasp_models/eval_ppi.py \
  framework.eval_episodes=1 \
  cinematic_recorder.enabled=True \
  cinematic_recorder.max_success_videos=1 \
  cinematic_recorder.max_fail_videos=1
```

如果验证 ACT_BC_VISION / BIMANUAL_PERACT，使用 `occ_grasp_models/eval.py` 和 `conf/eval.yaml`；如果验证 OPENPI_POLICY，使用 `occ_grasp_models/eval.py --config-name eval_openpi` 和 `conf/eval_openpi.yaml`。三类入口都应检查同一套 `cinematic_recorder.overlay_*` 行为。

检查：

- `eval_data.csv` 中 `phase_1_success_rate` 到 `phase_4_success_rate` 是否仍写出。
- 日志中 `phase_progress` 是否包含 `current_phase`、`max_current_phase_reached`、`condition_status`。
- 视频中是否显示策略名、当前阶段、关键点和左右夹爪 tip。
- 视频 marker 是否随物体 / 夹爪运动实时更新。

### 10.3 行为验证案例

重点看两类失败：

1. 已经进入 `Grasp`，但 `con1` 失效，应退回 `PreManipulation`。
2. 已经进入 `Lift`，但 `con2` 失效，应退回 `PreManipulation` 并从当前 `con1` 重新推进。

对应日志应能看到：

```text
current_phase: 4 -> 1 -> 2
max_current_phase_reached: 保持 4
phase_status: {1: True, 2: True, 3: True, 4: False}
```

---

## 11. 风险与注意事项

1. **四个任务存在重复代码**  
   `PhasedSuccessEvaluator` 和条件类在四个任务中各有一份，必须同步修改，否则阶段指标在不同任务间会漂移。

2. **`phase_type=5` 对使用 phase 监督的训练可能有影响**  
   如果后续 ACT strategy/phase 等模型继续使用 demo 里的 `phase_type` 作为监督标签，需要确认配置中的 `num_phases` 是否要从 4 改为 5，或者只在 online eval / 可视化里暴露 5。

3. **不能重复调用 evaluator**  
   evaluator 中的 condition 有稳定帧计数。视频 callback 和 runner 不能同时推进 evaluator，否则稳定条件会被人为加速。

4. **ClearPath 不再看 waypoint 后，Phase 3 会更贴近 online policy**  
   这符合当前目标，但与旧指标不可直接数值对比。旧实验的 Phase 3 success rate 和新实验的 Phase 3 success rate 语义不同。

5. **环境 active scheme 仍用于确定 grasper / pusher 角色**  
   即使 con3 不再使用 lift waypoint，也仍需要根据 `current_role_assignment` 选择 grasper gripper、pusher gripper 和 pusher tip。这个 scheme 是环境内部状态，不是 policy 显式选择。

---

## 12. 推荐实施顺序

1. 先改四个任务的 `ClearPathCondition`，去掉 waypoint 距离检查。
2. 再改四个任务的 `PhasedSuccessEvaluator` 和 `_setup_phased_evaluator()`。
3. 修改 `custom_rlbench_env.py`，把 evaluator 推进放到 `_my_callback()`，并新增 `get_env_overlay_state()`。
4. 修改 `_independent_env_runner.py`，runner 只读取 phase progress，不再推进 evaluator。
5. 修改 `video_utils.py`，将 overlay 数据源切换到环境真实状态。
6. 修改 `eval_ppi.yaml`、`eval.yaml`、`eval_openpi.yaml`，加统一的环境 overlay 配置。
7. 先跑 1 episode smoke test，再扩展到完整评估。
