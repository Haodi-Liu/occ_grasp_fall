# 双臂关键帧 Diffuser Actor Agent 方案（occ_grasp_fall）

本文档只描述在 `occ_grasp_fall/occ_grasp_models/agents` 内实现一个“类似 3D Diffuser Actor”的双臂关键帧 Agent 的方案，遵循你已确定的选择：**双臂同步、关键帧序列预测、无语言、单模型联合输出**。

---

## 1. 目标与约束（已定）

- **双臂同步建模**，单模型输出左右臂关键帧序列。
- **关键帧序列预测**（Case B），每次预测 K 个关键帧，缓存后逐帧执行。
- **无语言**，仅使用视觉（RGB+点云）与状态（双臂末端位姿/夹爪）。
- 训练/评估 **完全遵循 occ_grasp_fall 的 YARR 流程**，数据源为 `/mnt/rlbench_data/<task>.train` 和 `/mnt/rlbench_data/<task>`。

---

## 2. 新增目录结构（建议）

```
occ_grasp_models/agents/diffuser_actor/
├── __init__.py
├── launch_utils.py
├── agent.py
├── model.py
├── data_utils.py
└── replay_utils.py
```

同时需要新增或修改的全局文件：

```
occ_grasp_models/conf/method/DIFFUSER_ACTOR.yaml
occ_grasp_models/agents/agent_factory.py
occ_grasp_models/run_seed_fn.py
```

---

## 3. 每个脚本的职责说明

### 3.1 `agents/diffuser_actor/__init__.py`
- 导出 `create_agent`（供 `agent_factory` 调用）。
- 只做最薄的 import 封装，不放逻辑。

### 3.2 `agents/diffuser_actor/launch_utils.py`
- **create_agent(cfg)**：
  - 构造 `DualArmDiffuserActor` 模型实例（见 `model.py`）。
  - 构造 `DiffuserActorAgent`（见 `agent.py`）。
  - 用 `PreprocessAgent` 包装（可复用现有实现）。
- **create_replay(...)**：
  - 定义 YARR replay buffer 所需的 observation/replay elements。
  - 必须包含关键帧序列 `keyframe_traj` 与 `keyframe_mask`。
- **fill_replay(...) / fill_multi_task_replay(...)**：
  - 从 RLBench demos 构造训练样本。
  - 调用 `data_utils` 生成关键帧序列与 mask。
  - 写入 replay buffer。

### 3.3 `agents/diffuser_actor/agent.py`
- 继承 `yarr.agents.agent.Agent`。
- **build()**：创建优化器、设置训练/推理模式。
- **update()**：
  - 从 replay_sample 拿到关键帧序列和 mask。
  - 前向调用模型（扩散训练），计算 loss，反传更新。
- **act()**：
  - 每 `query_freq` 步预测一次关键帧序列并缓存。
  - 每一步从缓存取一个关键帧，转为 18D 动作并返回。
  - 18D 组装形式：
    - 右臂 `[pos(3), quat(4), grip(1), ignore(1)]`
    - 左臂 `[pos(3), quat(4), grip(1), ignore(1)]`
  - `ignore_collisions` 建议固定为 `1` 以匹配 RLBench action_mode。

### 3.4 `agents/diffuser_actor/model.py`
- 实现 **双臂联合版 Diffuser Actor**，复用 `~/claude_repo/3d_diffuser_actor` 组件：
  - `diffuser_actor/trajectory_optimization/diffuser_actor.py`
  - `diffuser_actor/utils/encoder.py`
  - `diffuser_actor/utils/layers.py`
- 关键改动点（见第 4 节）：
  - 输入 token 化（双臂、关键帧）。
  - 额外 arm embedding。
  - 双臂 curr_gripper 历史编码。

### 3.5 `agents/diffuser_actor/data_utils.py`
- 关键帧提取与序列构造：
  - 调用 `helpers/demo_loading_utils.keypoint_discovery()`。
  - 按时间排序关键帧，限制最大 K。
  - 生成 `keyframe_traj` 与 `keyframe_mask`。
- Token 化规则：
  - 顺序固定为 `[t0-R, t0-L, t1-R, t1-L, ...]`。
- 轨迹 padding 与 mask 生成。
- 四元数归一化工具。

### 3.6 `agents/diffuser_actor/replay_utils.py`
- 与 `launch_utils.py` 配合，封装 replay 元素定义和写入逻辑。
- 也可直接合并进 `launch_utils.py`，但建议单独抽出便于维护。

---

## 4. 相比 `~/claude_repo/3d_diffuser_actor` 的显著改动（适配点）

### 4.1 模型输入/输出结构
- 原始 Diffuser Actor：
  - 单臂，token=关键帧序列（每帧 1 token）。
  - token 维度为 8（pos+quat+grip）。
- 新实现：
  - **双臂同步**，每帧 2 个 token（右臂+左臂）。
  - token 顺序固定，避免左右臂混淆。
  - 引入 **arm embedding**（2 类）附加到 trajectory token 上。

### 4.2 `curr_gripper` 历史编码
- 原始 Encoder 只支持 `(B, nhist, 7)`。
- 新实现需要 `(B, nhist, 14)`：左右臂拼接。
- 修改方式：
  - 将 `(B, nhist, 14)` reshape 为 `(B, nhist*2, 7)`，并与 arm embedding 对齐。

### 4.3 训练流程（最显著差异）
- 原始 3d_diffuser_actor 使用独立 `Dataset` + `main_trajectory.py` 训练。
- 新实现改为 **YARR Agent 的 `update()`**：
  - 从 replay buffer 直接取 `keyframe_traj`/`mask`。
  - 训练行为与 occ_grasp_fall 其他 agent 保持一致。

### 4.4 数据来源与格式
- 原始代码依赖打包数据（`task+var/ep*.dat`）。
- 新实现使用 **RLBench 原始 demo**（`.train` 目录）作为训练数据源，
  并在 replay 阶段构造关键帧序列。

### 4.5 无语言分支
- 原始 Diffuser Actor 支持指令编码。
- 新实现 **完全关闭语言**：
  - `use_instruction=0`
  - instruction 输入置零张量
  - 删除 CLIP 依赖

### 4.6 动作格式与环境对齐
- 原始 Diffuser Actor 输出 8D（单臂）。
- 新实现输出 16D（双臂），在 `act()` 中拼成 18D，
  以匹配 `BimanualMoveArmThenGripper`。

---

## 5. 关键数据流说明

### 5.1 训练阶段（YARR update）
1. 从 replay buffer 取样：
   - 当前观测（RGB+点云）
   - `curr_gripper_history`
   - `keyframe_traj` + `keyframe_mask`
2. 模型训练前向：
   - 使用扩散头预测关键帧序列。
3. 计算 loss 并反传。

### 5.2 推理阶段（YARR act）
1. 每 `query_freq` 步预测一次关键帧序列。
2. 缓存关键帧，逐帧执行。
3. 每步输出 18D 动作。

---

## 6. 最小可用版本（MVP）建议步骤

1. 建立 replay buffer + keyframe 构造逻辑（`launch_utils.py` + `data_utils.py`）。
2. 改造 Diffuser Actor 为双臂 token 化版本（`model.py`）。
3. 实现 agent `update()`/`act()`，跑通单任务训练与评估。
4. 扩展到多任务、调参（K、nhist、query_freq、diffusion_steps）。

---

这份方案是面向实现落地的结构性说明，后续如需，我可以基于此进一步输出代码骨架或逐文件的实现草案。
