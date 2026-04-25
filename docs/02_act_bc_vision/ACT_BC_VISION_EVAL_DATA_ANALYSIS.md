# ACT_BC_VISION 评估结果整理与代码研判

## 1. 文档目的

本文档整理并分析以下评估结果文件：

- 评估数据: `/home/hdliu/arm_test/bimanual_four_task/multi/ACT_BC_VISION/seed0/eval_data.csv`
- 训练配置: `/home/hdliu/arm_test/bimanual_four_task/multi/ACT_BC_VISION/seed0/config.yaml`
- 评估配置: `/home/hdliu/occ_grasp_fall/occ_grasp_models/conf/eval.yaml`
- 主要 agent 代码: `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision`

本文档的目标有三点：

1. 将 `eval_data.csv` 中的指标完整整理出来。
2. 结合代码，澄清各指标的真实统计口径，避免误读。
3. 基于 `ACT_BC_VISION` 的实现方式，对结果做尽量有依据的解释和研判。

---

## 2. 实验对象确认

这份结果对应的不是 `bimanual_peract`，而是 `ACT_BC_VISION` 双臂 agent。

证据：

- 训练目录配置 `config.yaml` 中 `method.name: ACT_BC_VISION`
- 评估默认配置 `eval.yaml` 中 `defaults: - method: ACT_BC_VISION`
- `agent_factory.py` 中 `ACT_BC_VISION` 会路由到 `agents/act_bc_vision.launch_utils.create_agent`

因此，后续所有分析都以 `ACT_BC_VISION` 的模型结构、训练方式和推理逻辑为前提。

---

## 3. 实验设置概览

### 3.1 任务与评估设置

- Agent: `ACT_BC_VISION`
- Robot: `bimanual`
- Multi-task:
  - `bimanual_pick_plate`
  - `bimanual_pick_fork`
  - `bimanual_edge_phone`
  - `bimanual_pivot_phone`
- 训练迭代数: `2,000,001`
- 本次评估 checkpoint: `2,000,000`
- 每个任务评估 episode 数: `30`
- episode 长度上限: `400`
- 评估视角: `overhead`
- 图像分辨率: `256 x 256`

### 3.2 训练时间与 loss 快照

训练日志显示：

- 开始时间: `2026-03-19 14:05:30`
- 结束时间: `2026-03-21 20:56:51`
- 总耗时: `197481.40 s`，约 `54.86 h`

训练 loss 里程碑：

| Step | Loss | l1 | kl |
|---|---:|---:|---:|
| 500 | 5.00594 | 2.98933 | 0.20166 |
| 10,000 | 1.41620 | 1.34440 | 0.00718 |
| 50,000 | 0.62195 | 0.62072 | 0.00012 |
| 100,000 | 0.33423 | 0.33402 | 0.00002 |
| 500,000 | 0.19158 | 0.19155 | 0.00000 |
| 1,000,000 | 0.20442 | 0.20440 | 0.00000 |
| 1,500,000 | 0.15411 | 0.15411 | 0.00000 |
| 1,900,000 | 0.11995 | 0.11994 | 0.00000 |
| 1,995,000 | 0.11095 | 0.11085 | 0.00001 |
| 2,000,000 | 0.12762 | 0.12761 | 0.00000 |

初步结论：

- 训练 loss 明显下降并长期维持在较低水平。
- 但低 loss 并没有转化为高评估成功率，这已经提示出典型的 behavior cloning 分布偏移问题。

---

## 4. CSV 指标总览

`eval_data.csv` 共 `1` 行、`64` 列，对应单个 checkpoint `2000000` 的四任务评估结果。

字段结构可以概括为：

- `step`
- 对每个 task 记录一组 `eval_envs/*/<task>` 指标

每个 task 下面主要有以下类型指标：

- `return`: 该任务上所有评估 episode 的平均 reward；在当前环境里成功 episode reward 为 `100`、失败为 `0`，因此它数值上等于“真实任务成功率 × 100”。
- `length`: 该任务上所有评估 episode 的平均执行步数；成功和失败 episode 都会计入，达到时间上限的 episode 会把均值拉高。
- `total_transitions`: 该任务评估过程中累计写入统计器的 transition 总数；通常等于所有 episode 步数之和，也近似等于 `length × episode 数`。
- `success_count`: 正常完成评估循环、没有因为异常中断的 episode 数；它统计的是“评估执行成功”，不是“任务完成成功”。
- `failed_count`: 因异常、通信错误、StopIteration 等原因没有正常完成评估的 episode 数；它统计的是“评估过程失败”，不是“任务执行失败”。
- `success_rate`: 由 `success_count / (success_count + failed_count)` 得到；准确说更接近“评估执行完成率”，不应直接当作任务成功率理解。
- `phase_1_success_rate`: 设计上表示完成第 1 阶段条件的 episode 占比，即进入或完成 phase 1 的比例；它是阶段级过程指标，不等于最终任务成功率。
- `phase_2_success_rate`: 设计上表示完成第 2 阶段条件的 episode 占比，用于观察策略是否能稳定进入抓取或中间操作阶段。
- `phase_3_success_rate`: 设计上表示完成第 3 阶段条件的 episode 占比，通常用于衡量清道、让位或中间协同动作是否达成。
- `phase_4_success_rate`: 设计上表示完成第 4 阶段条件的 episode 占比，通常最接近最终完成动作，但仍应以任务最终 reward 定义的成功为准。
- `success_rate_left_grasper_scenes`: 在 GT scheme 被标记为 `left_grasper` 的场景子集上，真实任务成功的 episode 比例；它反映模型对左抓方案的适应能力。
- `total_left_grasper_episodes`: 本次评估中属于 `left_grasper` 场景的 episode 总数；它决定了左抓成功率的样本量和统计稳定性。
- `avg_steps_left_grasper_scenes` 或缺失: `left_grasper` 场景中“成功 episode”的平均步数；如果该子集没有成功样本，代码不会写出这个字段。
- `success_rate_right_grasper_scenes`: 在 GT scheme 被标记为 `right_grasper` 的场景子集上，真实任务成功的 episode 比例；它反映模型对右抓方案的适应能力。
- `total_right_grasper_episodes`: 本次评估中属于 `right_grasper` 场景的 episode 总数；它决定了右抓成功率的样本量和统计稳定性。
- `avg_steps_right_grasper_scenes` 或缺失: `right_grasper` 场景中“成功 episode”的平均步数；如果该子集没有成功样本，代码同样不会写出这个字段。
- `scheme_balance_gap`: `left_grasper` 与 `right_grasper` 两类场景成功率之差的绝对值；越大说明左右方案表现越不对称，但为 `0` 既可能表示两边都强，也可能表示两边都弱。

注意：

- `pick_fork` 和 `pivot_phone` 的 CSV 中没有 `avg_steps_*` 字段，因为两边都没有成功样本，代码不会写出这些字段。
- `pick_plate` 和 `edge_phone` 有成功样本，因此写出了对应的平均成功步数。

---

## 5. 原始 CSV 指标逐任务整理

### 5.1 bimanual_pick_plate

| 指标 | 数值 |
|---|---:|
| `eval_envs/return/bimanual_pick_plate` | 43.333333 |
| `eval_envs/length/bimanual_pick_plate` | 362.733333 |
| `eval_envs/total_transitions/bimanual_pick_plate` | 10882 |
| `eval_envs/success_count/bimanual_pick_plate` | 30 |
| `eval_envs/failed_count/bimanual_pick_plate` | 0 |
| `eval_envs/success_rate/bimanual_pick_plate` | 1.000000 |
| `eval_envs/phase_1_success_rate/bimanual_pick_plate` | 0.000000 |
| `eval_envs/phase_2_success_rate/bimanual_pick_plate` | 0.000000 |
| `eval_envs/phase_3_success_rate/bimanual_pick_plate` | 0.000000 |
| `eval_envs/phase_4_success_rate/bimanual_pick_plate` | 0.000000 |
| `eval_envs/success_rate_left_grasper_scenes/bimanual_pick_plate` | 0.733333 |
| `eval_envs/total_left_grasper_episodes/bimanual_pick_plate` | 15 |
| `eval_envs/avg_steps_left_grasper_scenes/bimanual_pick_plate` | 312.909091 |
| `eval_envs/success_rate_right_grasper_scenes/bimanual_pick_plate` | 0.133333 |
| `eval_envs/total_right_grasper_episodes/bimanual_pick_plate` | 15 |
| `eval_envs/avg_steps_right_grasper_scenes/bimanual_pick_plate` | 320.000000 |
| `eval_envs/scheme_balance_gap/bimanual_pick_plate` | 0.600000 |

### 5.2 bimanual_pick_fork

| 指标 | 数值 |
|---|---:|
| `eval_envs/return/bimanual_pick_fork` | 0.000000 |
| `eval_envs/length/bimanual_pick_fork` | 400.000000 |
| `eval_envs/total_transitions/bimanual_pick_fork` | 22882 |
| `eval_envs/success_count/bimanual_pick_fork` | 30 |
| `eval_envs/failed_count/bimanual_pick_fork` | 0 |
| `eval_envs/success_rate/bimanual_pick_fork` | 1.000000 |
| `eval_envs/phase_1_success_rate/bimanual_pick_fork` | 0.000000 |
| `eval_envs/phase_2_success_rate/bimanual_pick_fork` | 0.000000 |
| `eval_envs/phase_3_success_rate/bimanual_pick_fork` | 0.000000 |
| `eval_envs/phase_4_success_rate/bimanual_pick_fork` | 0.000000 |
| `eval_envs/success_rate_left_grasper_scenes/bimanual_pick_fork` | 0.000000 |
| `eval_envs/total_left_grasper_episodes/bimanual_pick_fork` | 14 |
| `eval_envs/success_rate_right_grasper_scenes/bimanual_pick_fork` | 0.000000 |
| `eval_envs/total_right_grasper_episodes/bimanual_pick_fork` | 16 |
| `eval_envs/scheme_balance_gap/bimanual_pick_fork` | 0.000000 |

### 5.3 bimanual_edge_phone

| 指标 | 数值 |
|---|---:|
| `eval_envs/return/bimanual_edge_phone` | 33.333333 |
| `eval_envs/length/bimanual_edge_phone` | 379.733333 |
| `eval_envs/total_transitions/bimanual_edge_phone` | 34274 |
| `eval_envs/success_count/bimanual_edge_phone` | 30 |
| `eval_envs/failed_count/bimanual_edge_phone` | 0 |
| `eval_envs/success_rate/bimanual_edge_phone` | 1.000000 |
| `eval_envs/phase_1_success_rate/bimanual_edge_phone` | 0.000000 |
| `eval_envs/phase_2_success_rate/bimanual_edge_phone` | 0.000000 |
| `eval_envs/phase_3_success_rate/bimanual_edge_phone` | 0.000000 |
| `eval_envs/phase_4_success_rate/bimanual_edge_phone` | 0.000000 |
| `eval_envs/success_rate_left_grasper_scenes/bimanual_edge_phone` | 0.666667 |
| `eval_envs/total_left_grasper_episodes/bimanual_edge_phone` | 15 |
| `eval_envs/avg_steps_left_grasper_scenes/bimanual_edge_phone` | 339.200000 |
| `eval_envs/success_rate_right_grasper_scenes/bimanual_edge_phone` | 0.000000 |
| `eval_envs/total_right_grasper_episodes/bimanual_edge_phone` | 15 |
| `eval_envs/scheme_balance_gap/bimanual_edge_phone` | 0.666667 |

### 5.4 bimanual_pivot_phone

| 指标 | 数值 |
|---|---:|
| `eval_envs/return/bimanual_pivot_phone` | 0.000000 |
| `eval_envs/length/bimanual_pivot_phone` | 400.000000 |
| `eval_envs/total_transitions/bimanual_pivot_phone` | 46274 |
| `eval_envs/success_count/bimanual_pivot_phone` | 30 |
| `eval_envs/failed_count/bimanual_pivot_phone` | 0 |
| `eval_envs/success_rate/bimanual_pivot_phone` | 1.000000 |
| `eval_envs/phase_1_success_rate/bimanual_pivot_phone` | 0.000000 |
| `eval_envs/phase_2_success_rate/bimanual_pivot_phone` | 0.000000 |
| `eval_envs/phase_3_success_rate/bimanual_pivot_phone` | 0.000000 |
| `eval_envs/phase_4_success_rate/bimanual_pivot_phone` | 0.000000 |
| `eval_envs/success_rate_left_grasper_scenes/bimanual_pivot_phone` | 0.000000 |
| `eval_envs/total_left_grasper_episodes/bimanual_pivot_phone` | 1 |
| `eval_envs/success_rate_right_grasper_scenes/bimanual_pivot_phone` | 0.000000 |
| `eval_envs/total_right_grasper_episodes/bimanual_pivot_phone` | 29 |
| `eval_envs/scheme_balance_gap/bimanual_pivot_phone` | 0.000000 |

---

## 6. 汇总表

### 6.1 原始 CSV 关键汇总

| Task | return | length | total_transitions | success_count | failed_count | success_rate | phase1 | phase2 | phase3 | phase4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bimanual_pick_plate | 43.33 | 362.73 | 10882 | 30 | 0 | 1.00 | 0 | 0 | 0 | 0 |
| bimanual_pick_fork | 0.00 | 400.00 | 22882 | 30 | 0 | 1.00 | 0 | 0 | 0 | 0 |
| bimanual_edge_phone | 33.33 | 379.73 | 34274 | 30 | 0 | 1.00 | 0 | 0 | 0 | 0 |
| bimanual_pivot_phone | 0.00 | 400.00 | 46274 | 30 | 0 | 1.00 | 0 | 0 | 0 | 0 |

### 6.2 Scheme 分层汇总

| Task | left_total | left_rate | left_avg_steps | right_total | right_rate | right_avg_steps | balance_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| bimanual_pick_plate | 15 | 0.7333 | 312.91 | 15 | 0.1333 | 320.00 | 0.6000 |
| bimanual_pick_fork | 14 | 0.0000 | - | 16 | 0.0000 | - | 0.0000 |
| bimanual_edge_phone | 15 | 0.6667 | 339.20 | 15 | 0.0000 | - | 0.6667 |
| bimanual_pivot_phone | 1 | 0.0000 | - | 29 | 0.0000 | - | 0.0000 |

### 6.3 校正后的真实任务成功率汇总

这里的“真实任务成功率”不是直接取 CSV 里的 `success_rate`，而是依据代码口径从 `return` 反推：

- 成功 episode 奖励被放大为 `100`
- 失败 episode 奖励为 `0`
- 所以 `return / 100 = 成功率`

| Task | 真实成功数 | 真实成功率 |
|---|---:|---:|
| bimanual_pick_plate | 13 / 30 | 43.33% |
| bimanual_pick_fork | 0 / 30 | 0.00% |
| bimanual_edge_phone | 10 / 30 | 33.33% |
| bimanual_pivot_phone | 0 / 30 | 0.00% |
| Overall | 23 / 120 | 19.17% |

附加整体统计：

- 四任务宏平均真实成功率: `19.17%`
- 四任务平均 episode 长度: `385.62`
- 四任务总 transitions: `114,312`

---

## 7. 指标口径澄清

这一部分非常关键，因为原始 CSV 中有几个指标名容易误导。

### 7.1 `return` 的真实含义

在 `CustomMultiTaskRLBenchEnv.step()` 中：

- 如果任务成功，环境 reward 被乘以 `reward_scale=100`
- 如果任务失败，reward 被设为 `0`

因此：

- `eval_envs/return/<task>` 实际上就是 “平均 episode reward”
- 由于 reward 只有 `0` 或 `100`，所以 `return / 100` 正好等于真实成功率

这使得 `return` 成为这份 CSV 里最值得信赖的主指标。

### 7.2 `success_count` / `failed_count` / `success_rate` 的真实含义

在评估 runner 中：

- `success_count += 1` 的条件是 episode 正常执行完，没有抛异常
- 它并不要求任务 reward 成功
- `failed_count` 只统计异常、通信错误、StopIteration 等“评估过程失败”

于是：

- `success_rate = success_count / (success_count + failed_count)`
- 它更准确的名字其实应当是“评估执行完成率”或“episode completion rate”
- 不应被解释为“任务成功率”

这也解释了为什么四个任务的 `success_rate` 全是 `1.0`，但真实任务成功率只有 `19.17%`。

### 7.3 phase 指标目前不可信

表面上看，四个任务的 `phase_1 ~ phase_4_success_rate` 全为 0。

但这里存在一个直接的自相矛盾：

- `pick_plate` 最终成功条件是 `LiftedCondition(min_height=0.9)`
- `pick_plate` 的 phase4 条件也是同一个 `LiftedCondition(min_height=0.9)`
- `edge_phone` 最终成功条件和 phase4 也是同一个 `LiftedCondition(min_height=1.1)`

按理说：

- 只要有真正成功 episode，phase4 不可能是 0

但 CSV 里：

- `pick_plate` 有 `13/30` 成功
- `edge_phone` 有 `10/30` 成功
- phase4 仍然是 `0`

因此可以判断：

- phase 指标不是“模型真的一个阶段都没完成”
- 而是 phase 统计链路本身存在实现或口径问题

结论：

- 当前不能用 `phase_*_success_rate` 来判断 agent 卡在哪个阶段
- 这组指标在本次分析中只能视为“统计异常”

### 7.4 `scheme_balance_gap = 0` 不代表“平衡”

`scheme_balance_gap` 在代码中定义为：

- `abs(success_rate_left_grasper_scenes - success_rate_right_grasper_scenes)`

所以：

- 如果左右都很强，gap 可能是 0
- 如果左右都很差，gap 也可能是 0

本次结果中：

- `pick_fork` 的 gap = 0，是因为左右都 0
- `pivot_phone` 的 gap = 0，也是因为左右都 0

因此，gap 只能用于衡量不对称程度，不能单独当作“好坏”指标。

---

## 8. `ACT_BC_VISION` 的实现特征

为了正确解读结果，必须先明确这个 agent 到底是什么。

### 8.1 本质上是纯视觉行为克隆，不是 value-guided policy

`ACT_BC_VISION` 的核心训练目标是：

- `L1(action_pred, action_gt) + KL`

它没有：

- Q-function
- value 引导
- test-time search
- 显式的错误恢复机制

因此，它本质上是一个纯 BC 的 sequence policy。对长时序双臂接触任务来说，这天然更容易受到分布偏移影响。

### 8.2 一次预测 100 步 action chunk

配置中：

- `next_action_horizon = 100`
- `chunk_size = 100`
- `num_queries = 100`

这意味着模型一次输出很长的一段未来动作序列。

优点：

- 动作更平滑
- 更像“整段动作计划”

代价：

- phase 边界可能被抹平
- 在接触时刻发生微小偏差后，后续整段动作都可能失效
- 对需要精确切换“推 -> 抓 -> 清道 -> 抬起”的任务不友好

### 8.3 推理时每步重预测，并对历史 chunk 做时间聚合

在 `act()` 中：

- 每个 timestep 都会重新预测一段新的 action chunk
- 再对当前时刻所有可用 chunk 的对应 action 做指数加权平均

这是一种 temporal aggregation 机制。

它的效果通常是：

- 提高稳定性
- 减少 jitter

但在双臂接触任务中也会带来副作用：

- 动作切换被过度平滑
- 抓取闭合、清道撤离、抬升开始这些临界动作容易被“平均掉”

### 8.4 实际只用了 RGB，没有真正使用 point cloud

虽然 replay buffer 里确实存了：

- RGB
- point cloud
- 相机内外参

但在 agent 前向时：

- `preprocess_images()` 会同时读出 `stacked_rgb` 和 `stacked_point_cloud`
- 后续真正送进 actor 的只有 `stacked_rgb`

因此，这个 agent 的有效输入实际上是：

- 单视角 RGB
- 双臂 proprio state
- task-specific qpos normalization

而不是：

- RGB-D / point cloud policy

这对依赖几何深度、遮挡关系和侧向悬空量的任务会很不利。

### 8.5 当前评估只用单个 `overhead` 视角

本次配置的 camera 是：

- `overhead`

这进一步强化了上面的限制：

- 缺少侧视几何信息
- 物体厚度、悬空量、clearance 很难仅从顶视图稳定估计
- 两臂相对物体的上下空间关系也不容易看清

对 `fork`、`pivot_phone` 这类强依赖接触几何的任务尤其不利。

### 8.6 没有显式 task token，只用 `task_id` 做按任务归一化

代码里确实保存了 `task_id`，但它主要被用于：

- 为不同任务选择不同的 qpos 归一化统计量

它并没有形成一个强显式的 task condition embedding 去告诉 policy：

- 当前到底是 `pick_plate`
- 还是 `edge_phone`
- 还是 `pivot_phone`

这意味着多任务共享更多依赖视觉外观和状态分布去“隐式区分任务”。

对于视觉外观相近但策略机制不同的任务，这会明显变难。

### 8.7 输出动作只学 pose + gripper，执行期 `ignore_collision` 被硬编码

模型预测的是 16 维：

- 右臂 position(3) + quaternion(4) + gripper(1)
- 左臂 position(3) + quaternion(4) + gripper(1)

在真正输出给环境前，又额外拼了两维：

- `right_ignore_collision = 1.0`
- `left_ignore_collision = 1.0`

这不是模型学出来的，而是执行期硬编码的。

对 clearance 很紧、接触几何复杂的任务来说，这种做法可能进一步放大执行误差。

---

## 9. 结果研判

### 9.1 总体判断

如果只看这次评估，我对 `ACT_BC_VISION` 的判断是：

- 它已经学到了一部分顶视图下的双臂动作模板。
- 但它还没有学到可靠的 contact-rich manipulation policy。
- 当前主要瓶颈不是“训练没收敛”，而是“表示方式 + 控制范式”与任务需求不匹配。

这也是为什么：

- 训练 loss 很低
- 但真实任务成功率只有 `19.17%`

### 9.2 `bimanual_pick_plate`

结果：

- 真实成功率 `43.33%`
- 左抓场景 `11/15 = 73.33%`
- 右抓场景 `2/15 = 13.33%`

判断：

- 这是四个任务里当前表现最好的一个。
- 说明 agent 不是完全不会做双臂接触任务。
- 但它对左右 scheme 的泛化极不对称，说明学到的更像是“某一类几何模板”，而不是镜像对称的可迁移操作策略。

这类不对称通常意味着：

- 单视角视觉表征不足
- 镜像场景在特征空间中没有被很好对齐
- 或者数据分布本身就存在偏置

### 9.3 `bimanual_edge_phone`

结果：

- 真实成功率 `33.33%`
- 左抓场景 `10/15 = 66.67%`
- 右抓场景 `0/15 = 0%`

判断：

- 该任务也学到了一部分策略。
- 但成功全部来自 `left_grasper`，`right_grasper` 完全失效。

这说明问题不是“整个任务没学会”，而是：

- scheme 泛化失败
- 左右角色切换失败
- 视觉上学到的是偏单侧的、非对称的模板

这是本次结果里最强烈的信号之一。

### 9.4 `bimanual_pick_fork`

结果：

- 真实成功率 `0%`
- 左右 scheme 都是 `0%`
- 平均长度 `400`，基本都跑满

判断：

- 该任务对当前 `ACT_BC_VISION` 来说是彻底没学会。

原因上最合理的解释是：

- 叉子较小、较薄
- 接触点和姿态误差容忍度很低
- 仅靠顶视图 RGB 很难稳定估计关键几何
- 100 步长 chunk 对这种精细 phase 切换任务也不友好

另外，loss 中 position 和 gripper 权重大于 quaternion，这也可能进一步弱化精细姿态控制能力。

### 9.5 `bimanual_pivot_phone`

结果：

- 真实成功率 `0%`
- 左抓样本只有 `1`
- 右抓样本 `29`
- 两边都 `0%`

判断：

- 这是当前最难、也最不适配该 agent 的任务之一。

这个任务同时要求：

- 靠墙接触
- 撬起
- 稳定抓取
- 清道撤离
- 再抬起

它对以下能力高度敏感：

- phase 切换
- clearance 判断
- 接触点几何感知
- 小误差下的恢复能力

而这几项恰好都是当前 `ACT_BC_VISION` 的弱项。

### 9.6 关于左右 scheme 偏差的更深解释

这里的左右 `left_grasper` / `right_grasper` 不只是“随机左右手”。

代码中 `ArmRoleSelector` 的选择逻辑是：

1. 先做可达性检查
2. 再比较执行成本
3. 选更可行或更便宜的方案

因此：

- 左右 scheme 指标本质上是在衡量 agent 对两类不同几何执行方案的泛化差异
- 不只是纯粹的数据集左右比例问题

所以本次结果中：

- `pick_plate` 和 `edge_phone` 的巨大左右差异

可以被解释为：

- agent 对某类几何方案已经形成了可用模板
- 但对镜像或另一类几何方案几乎没有建立等价能力

---

## 10. 关键结论

### 10.1 可以直接下结论的部分

1. 当前最可信的主指标是 `return`，其换算出的真实总体成功率为 `19.17%`。
2. CSV 中 `success_rate=1.0` 不代表任务成功，只代表评估执行没有异常中断。
3. `phase_*_success_rate` 当前不可信，不能据此判断卡在哪一阶段。
4. `ACT_BC_VISION` 在 `pick_plate` 和 `edge_phone` 上学到了一部分能力，但表现明显偏单侧 scheme。
5. `pick_fork` 和 `pivot_phone` 在当前配置下基本未学会。

### 10.2 对方法本身的判断

从这份结果看，`ACT_BC_VISION` 当前更像是：

- 学到了部分“顶视图视觉模板 + 双臂大致轨迹”

但还没有学到：

- 稳定可泛化的双臂接触操控策略

换句话说，当前的主要问题不是“优化没跑起来”，而是：

- 单视角 RGB 表征不足
- 长 action chunk 不适配 phase 明确的接触任务
- 多任务共享但缺少强 task condition
- 纯 BC 缺少纠错能力

---

## 11. 建议的后续方向

按优先级排序，我认为最值得做的是以下几项。

### 11.1 先修评估口径

优先级最高。

建议：

- 把当前 CSV 的 `success_rate` 改名为 `completion_rate`
- 另行记录真正的 `task_success_rate`
- 修复 phase 统计链路

原因：

- 如果评估口径本身混乱，后续横向比较模型时很容易得出错误结论。

### 11.2 增强视觉输入

建议：

- 至少增加多视角
- 更进一步，真正把 depth / point cloud 用进 actor

原因：

- 当前 agent 虽然存了 point cloud，但并未实际使用。
- 对 `fork`、`pivot_phone` 这类任务，单目顶视图信息量明显不足。

### 11.3 缩短 action chunk

建议优先尝试：

- 将 `next_action_horizon / num_queries` 从 `100` 降到更短

原因：

- 当前任务是强 phase 化的双臂接触任务。
- 过长 chunk 和时间聚合很可能把关键切换动作平均掉。

### 11.4 加显式 task condition

建议：

- 不要只依赖 `task_id` 做归一化
- 应考虑让 policy 显式感知当前任务身份

原因：

- 当前四个任务中，至少两个 phone 任务视觉上相近，但操作机制并不相同。
- 显式 task condition 往往比隐式从视觉里自己分辨更稳。

### 11.5 重新审视执行期 `ignore_collision=1`

建议：

- 检查其是否导致过多“不受约束”的规划行为
- 尤其关注 `pivot_phone` 这类 clearance 紧张任务

---

## 12. 最终结论

如果用一句话概括这份结果：

> 当前 `ACT_BC_VISION` 已经学到了一部分双臂视觉行为模板，但尚未形成稳定、可镜像泛化、可跨 phase 可靠切换的双臂接触操控能力。

更具体地说：

- `pick_plate`、`edge_phone` 有一定可用性，但明显依赖单侧 scheme。
- `pick_fork`、`pivot_phone` 在当前设置下基本失败。
- 当前最大的风险不是训练 loss，而是评估口径误读和方法本身对任务结构的不适配。

因此，下一步最有价值的工作顺序应该是：

1. 修正评估指标定义
2. 增强视觉输入
3. 缩短 action chunk
4. 加显式 task condition
5. 再做新一轮对比评估

---

## 13. 附：本次分析用到的关键代码位置

- 训练配置:
  - `/home/hdliu/arm_test/bimanual_four_task/multi/ACT_BC_VISION/seed0/config.yaml`
- 评估配置:
  - `/home/hdliu/occ_grasp_fall/occ_grasp_models/conf/eval.yaml`
- agent 路由:
  - `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/agent_factory.py`
- ACT_BC_VISION replay / 数据组织:
  - `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/launch_utils.py`
- ACT_BC_VISION agent:
  - `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py`
- ACT policy:
  - `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_policy.py`
- ACT DETR/VAE 前向:
  - `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/detr/models/detr_vae.py`
- 评估 runner:
  - `/home/hdliu/occ_grasp_fall/repos/YARR/yarr/runners/_independent_env_runner.py`
- 环境 reward 逻辑:
  - `/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py`
- scheme 统计:
  - `/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/scheme_utils.py`
- 任务 phase / success 条件:
  - `/home/hdliu/occ_grasp_fall/repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_plate.py`
  - `/home/hdliu/occ_grasp_fall/repos/RLBench/rlbench/bimanual_tasks/bimanual_pick_fork.py`
  - `/home/hdliu/occ_grasp_fall/repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py`
  - `/home/hdliu/occ_grasp_fall/repos/RLBench/rlbench/bimanual_tasks/bimanual_pivot_phone.py`
