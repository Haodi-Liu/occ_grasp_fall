# 每阶段成功条件、当前阶段评估与 waypoints 关系说明

可以把它分成三层来理解：

```text
阶段成功条件：每一关怎么判断过关
PhasedSuccessEvaluator：按顺序检查这些关卡，并记录过了哪些关
demo / online eval：在不同场景下，什么时候调用这个评估器
```

## 1. 每阶段成功条件到底在判断什么

四个阶段不是靠“走到了第几个 waypoint”来判定，而是靠真实环境状态判定。

以通用逻辑说：

```text
Phase 1：是否已经创造出可抓空间？
Phase 2：是否已经稳定抓住目标？
Phase 3：辅助臂是否已经撤开，不挡后续动作？
Phase 4：目标物是否已经被抬起到指定高度？
```

具体到条件：

- **Phase 1**
  - `bimanual_edge_phone`：用 `EdgeOverhangCondition`，看 `phone_edge` 相对 `box_edge` 是否已经悬出足够距离，并且手机是否稳定。
  - `bimanual_pivot_phone` / `pick_plate` / `pick_fork`：用 `GraspPointHeightCondition`，看 `grasp_pt.z` 是否升到阈值以上，并且物体稳定。

- **Phase 2**
  - 用 `StableGraspCondition`。
  - 判断指定的 grasper 夹爪是否真的抓住目标物，并且目标物速度足够小、稳定若干帧。

- **Phase 3**
  - 用 `StableGraspCondition + ClearPathCondition`。
  - 也就是说，不仅要保持抓住目标，还要确认 pusher/辅助臂已经松开，并且离目标物、离后续 lift waypoint 足够远。

- **Phase 4**
  - 用 `LiftedCondition`。
  - 判断目标物体世界坐标系下的 `z` 高度是否达到阈值。

所以阶段成功条件的本质是：

```text
读当前仿真环境里的物体、夹爪、dummy 的位置和状态，
判断这个阶段是否已经完成。
```

不是简单看 waypoint 执行到了哪里。

## 2. 当前阶段是怎么被评估出来的

`PhasedSuccessEvaluator` 维护一个状态：

```text
current_phase = 1
phase_status = {
  1: False,
  2: False,
  3: False,
  4: False
}
```

每次调用 `evaluate_current_phase()` 时，它只从当前阶段开始检查。

例如现在 `current_phase = 1`：

```text
检查 Phase 1 条件
  如果不满足：仍然停在 Phase 1
  如果满足：phase_status[1] = True，current_phase 变成 2
```

然后它会继续用 `while` 检查下一阶段。这样做是为了避免同一帧里多个阶段同时满足时漏记。

比如某一帧中：

```text
Phase 1 条件满足
Phase 2 条件也满足
```

那它可以一次推进到 Phase 3，而不是只记录 Phase 1。

一个关键点：**阶段完成后会锁存，不回退。**

例如 EdgePhone 的 Phase 1 是“手机边缘悬空”。手机被抓起来后，这个悬空条件可能不再成立。但 Phase 1 一旦完成，就不会因为后面条件失效而取消。否则会出现“明明已经抓起来了，但 Phase 1 又失败”的怪现象。

所以它的逻辑是：

```text
阶段条件满足 -> 记录该阶段完成 -> 进入下一阶段
已经完成的阶段不再重新判失败
```

## 3. demo 收集时怎么用

demo 收集时，机器人按 waypoints 走：

```text
Phase 1 的 waypoint
Phase 2 的 waypoint
Phase 3 的 waypoint
Phase 4 的 waypoint
```

但阶段标签不是直接写死成“现在正在执行第几个 waypoint phase”。

实际逻辑是：

```text
机器人沿 waypoint 走一步
  -> 看真实环境状态
  -> 调用 evaluate_phase_and_get_labels()
  -> 内部调用 PhasedSuccessEvaluator
  -> 得到当前 phase_type
  -> 写入 obs.misc["phase_type"]
```

对应代码定位：

| 环节 | 脚本 | 函数 / 位置 |
| --- | --- | --- |
| 生成 live demo 并定义 `do_record()` | `repos/RLBench/rlbench/backend/scene.py` | `Scene.get_demo()`，约第 719-770 行 |
| 双臂 demo 入口，检测任务是否有 `execution_phases` | `repos/RLBench/rlbench/backend/scene.py` | `Scene.execute_waypoints_bimanual()`，约第 472-475 行 |
| 按任务定义的四阶段 waypoint 顺序执行 | `repos/RLBench/rlbench/backend/scene.py` | `Scene.execute_waypoints_bimanual_phased()`，约第 583-717 行 |
| 每个 waypoint step 后更新 `strategy_type/phase_type` 并记录帧 | `repos/RLBench/rlbench/backend/scene.py` | `Scene.execute_waypoints_bimanual_phased()`，约第 671-680 行 |
| phase 间 `wait_after` 稳定等待时也更新标签并记录帧 | `repos/RLBench/rlbench/backend/scene.py` | `Scene.execute_waypoints_bimanual_phased()`，约第 691-714 行 |
| 首帧记录前初始化阶段标签，避免首帧无标签或标签错位 | `repos/RLBench/rlbench/backend/scene.py` | `Scene.get_demo()`，约第 736-746 行 |
| 把当前标签写进 observation 的 `misc` | `repos/RLBench/rlbench/backend/scene.py` | `Scene._get_misc()`，约第 950-982 行 |
| 任务侧根据真实环境状态推进阶段并返回标签 | `repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py` | `BimanualEdgePhone.evaluate_phase_and_get_labels()`，约第 878-894 行 |
| 同上，其他任务 | `repos/RLBench/rlbench/bimanual_tasks/bimanual_pivot_phone.py` / `bimanual_pick_plate.py` / `bimanual_pick_fork.py` | `evaluate_phase_and_get_labels()`，分别约第 780、705、703 行 |
| 阶段状态机实际推进 | 四个 bimanual task 脚本 | `PhasedSuccessEvaluator.evaluate_current_phase()`，例如 `bimanual_edge_phone.py` 约第 536-561 行 |
| 数据集保存时记录本 episode 的 active scheme | `repos/RLBench/tools/dataset_generator_bimanual.py` | `main()` 的采集循环，约第 168-218 行 |

所以 demo 里的 `phase_type` 是条件判断出来的，不是简单由 waypoint 序号赋值。

当然，由于 demo 是按设计好的 waypoints 执行的，所以正常情况下二者会高度对应：

```text
执行 Phase 1 waypoints -> 很快满足 Phase 1 条件 -> phase_type 进入 2
执行 Phase 2 waypoints -> 满足抓取条件 -> phase_type 进入 3
...
```

但从原理上说，决定阶段转换的是“条件满足”，不是“waypoint 执行完”。

## 4. 在线评估时怎么用

在线评估时没有 waypoint 控制动作。流程是：

```text
policy 看 observation
  -> agent.act() 输出动作
  -> env.step(action)
  -> 环境状态变化
  -> runner 调用 env.evaluate_current_phase()
  -> PhasedSuccessEvaluator 判断是否完成当前阶段
```

对应代码定位：

| 环节 | 脚本 | 函数 / 位置 |
| --- | --- | --- |
| 每个 eval episode 用评估集 demo seed 重置场景 | `repos/YARR/yarr/utils/rollout_generator.py` | `RolloutGenerator.generator()`，约第 18-29 行 |
| 从磁盘读取对应 demo，并调用 RLBench reset_to_demo | `occ_grasp_models/helpers/custom_rlbench_env.py` | `CustomRLBenchEnv.reset_to_demo()`，约第 521-551 行 |
| RLBench 根据 demo 的随机状态重置任务 | `repos/RLBench/rlbench/task_environment.py` | `TaskEnvironment.reset_to_demo()`，约第 213-219 行 |
| reset 时重新初始化 episode，随机放置后调用任务的 setup hook | `repos/RLBench/rlbench/task_environment.py` / `repos/RLBench/rlbench/backend/scene.py` | `TaskEnvironment.reset()` 约第 81-96 行；`Scene.init_episode()` 约第 150-170 行 |
| policy 根据当前 observation 输出动作 | `repos/YARR/yarr/utils/rollout_generator.py` | `RolloutGenerator.generator()`，约第 35-50 行 |
| runner 每帧后验推进环境阶段状态机 | `repos/YARR/yarr/runners/_independent_env_runner.py` | `_IndependentEnvRunner._run_eval_independent()`，约第 423-434 行 |
| 环境封装把调用转发到实际任务的 `phased_evaluator` | `occ_grasp_models/helpers/custom_rlbench_env.py` | `CustomRLBenchEnv.evaluate_current_phase()`，约第 484-499 行 |
| episode 结束后读取 `phase_status` | `repos/YARR/yarr/runners/_independent_env_runner.py` | `_IndependentEnvRunner._run_eval_independent()`，约第 455-466 行 |
| 环境封装读取任务侧阶段进度 | `occ_grasp_models/helpers/custom_rlbench_env.py` | `CustomRLBenchEnv.get_phase_progress()`，约第 467-482 行 |
| 汇总 `phase_i_success_rate` 和 `avg_max_phase` | `repos/YARR/yarr/runners/_independent_env_runner.py` | `_IndependentEnvRunner._run_eval_independent()`，约第 625-637 行 |
| 任务侧返回阶段进度字典 | 四个 bimanual task 脚本 | `get_phase_progress()`，例如 `bimanual_edge_phone.py` 约第 896-901 行 |

episode 结束后再读：

```text
get_phase_progress()
```

得到：

```text
phase_status = {
  1: True,
  2: True,
  3: False,
  4: False
}
```

然后统计很多 episode：

```text
phase_1_success_rate
phase_2_success_rate
phase_3_success_rate
phase_4_success_rate
```

所以在线评估里，阶段评估完全是后验判断：

```text
policy 做了什么动作不重要；
重要的是它造成的环境状态有没有满足各阶段条件。
```

## 5. 阶段评估里是否涉及 waypoints

答案要分清楚：

```text
waypoints 不决定当前阶段。
但某些阶段条件会用 waypoint dummy 作为几何参考。
```

具体说：

- Phase 1：一般不直接用 waypoint。
  - EdgePhone 用 `phone_edge` 和 `box_edge`。
  - 其他任务用 `grasp_pt`。

- Phase 2：不用 waypoint。
  - 只看 gripper 是否抓住目标物、物体是否稳定。

- Phase 3：**会用 waypoint。**
  - `ClearPathCondition` 会检查辅助臂 tip 离目标物够不够远。
  - 同时还会检查辅助臂 tip 离后续 lift waypoint 够不够远。
  - 这个 lift waypoint 通常是当前 grasper 方案里的最后一个 grasper waypoint，例如 `waypoint6` 或 `waypoint6_a`。

- Phase 4：不用 waypoint。
  - 只看目标物体高度。

所以最准确的说法是：

```text
阶段评估主要基于真实物理状态；
waypoints 只在 Phase 3 的清道条件里作为“后续抬起路径参考点”参与判断。
```

## 6. 在线评估时 waypoints 是否依然存在

是的，通常依然存在。

在线评估虽然不让 agent 沿 waypoints 运动，但任务场景本身还是从 `.ttm/.ttt` 加载的。里面的 waypoint dummy 仍然是 CoppeliaSim 场景对象。

它们在在线 eval 中的作用主要有三个：

1. reset / validate 时，任务仍可能读取和验证 waypoints。
2. 环境内部的任务类会在 reset 初始化时确定当前 active waypoint scheme，即 `right_grasper` 或 `left_grasper`。
3. Phase 3 的 `ClearPathCondition` 会读取 lift waypoint 的当前位置，用它判断辅助臂是否挡住后续抬起路径。

这里的 scheme 要分成两层，避免误解：

- **任务环境内部 active scheme**：reset 时 `Scene.init_episode()` 会在随机放置后调用任务的 `post_placement_setup()`，例如 `BimanualEdgePhone.post_placement_setup()` 约第 801-836 行。该函数通过 `ArmRoleSelector.select_scheme()` 设置 `active_waypoint_mode`、`current_role_assignment` 和 `waypoint_mapping`，并据此初始化 `PhasedSuccessEvaluator`。在线 eval 中，这个 scheme 只属于环境/任务内部状态，主要影响 `ClearPathCondition` 使用哪个 lift waypoint 作为几何参考，以及任务 validate / waypoint mapping 这类环境逻辑。
- **数据集 GT scheme**：演示收集时 `repos/RLBench/tools/dataset_generator_bimanual.py` 会把当时的 active scheme 保存成 `scheme_info_left_grasper.pkl` 或 `scheme_info_right_grasper.pkl`，位置约第 168-218 行。在线 eval 的 runner 会通过 `occ_grasp_models/helpers/scheme_utils.py` 中的 `build_episode_scheme_map()` 扫描这些文件，位置约第 36-119 行，再在 `_IndependentEnvRunner._run_eval_independent()` 约第 222-250、362-380、506-514、640-666 行用于 scheme 分层统计。
- **policy 不显式选择 scheme**：`RolloutGenerator.generator()` 只把 observation 喂给 `agent.act()`，位置约第 35-50 行；当前代码没有把 `active_scheme`、`current_gt_scheme`、`left_grasper/right_grasper` 作为显式控制信号传给 policy。也就是说，GT scheme 对 online eval 的主要意义是“按演示收集时的左右分配方案做结果分层统计”，不是指导 agent 下一步用哪只手、走哪套 waypoint。

但它们不会这样用：

```text
agent 下一步必须去 waypoint3
agent 动作由 waypoint 生成
```

在线 eval 的动作只来自 policy。

## 7. waypoints 是否会跟着物体一起运动

这取决于场景里 waypoint dummy 的父子关系。

文档和任务设计里明确有这个假设：

```text
grasper 路径点通常附着在目标物体上。
```

如果 `waypoint0/2/4/6` 这类 grasper waypoint 是目标物体的子节点，那么目标物体被推、被撬、被抬时，这些 waypoint 会跟着物体一起动，保持相对位姿不变。

这对设计很重要。因为：

```text
Phase 1 后物体姿态变了
grasper waypoint 也跟着变
后续抓取/抬起参考点仍然贴着物体
```

但不能说“所有 waypoint 都一定跟着物体动”。

更准确是：

```text
grasper waypoints 通常应附着在目标物体上，会跟随目标物体运动；
pusher waypoints 通常更多是环境/世界中的操作路径点，不一定跟随目标物体；
具体是否跟随，由 .ttm/.ttt 场景中的父子层级决定，不是 Python 代码每帧手动更新。
```

`ClearPathCondition` 读取的是 waypoint dummy 的当前世界位置：

```text
wp_dummy.get_position()
```

所以如果这个 dummy 跟着物体动，评估时读到的就是更新后的世界位置；如果它没有父子绑定，那读到的就是固定在场景里的位置。

## 8. 最核心理解

```text
demo 收集：
waypoints 控制机器人怎么走；
阶段条件判断什么时候切换 phase_type。

在线评估：
policy 控制机器人怎么走；
阶段条件判断 policy 实际完成了哪些阶段。

waypoints 在线评估时不控制 agent；
但 waypoint dummy 仍可能存在，并且 Phase 3 会用 lift waypoint 作为清道参考。
```

所以你可以把 waypoints 理解成：

```text
demo 中：动作脚本
online eval 中：不再是动作脚本，只剩少量几何参考作用
```
