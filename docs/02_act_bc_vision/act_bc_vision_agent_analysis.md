# ActBCVisionAgent 代码详解

## 文件概述

**文件路径**: `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_bc_vision_agent.py`

**主要功能**: 实现了一个基于行为克隆（Behavior Cloning）的视觉代理，用于双臂机器人控制。该代理使用视觉输入（多相机RGB图像）和机器人状态来预测双臂的末端执行器动作（End-Effector Pose Control）。

**重要更新说明**: 当前版本使用**末端执行器位姿控制**（gripper_pose + gripper_open），而非早期版本的关节位置控制（joint_positions）。这是一个重大的架构变化，影响了动作空间、归一化策略和输出格式。

---

## 1. 导入模块分析

```python
import copy
import logging
from functools import lru_cache
import pickle
import os
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary

from helpers import utils
from helpers.utils import stack_on_channel
```

**关键依赖**:
- **PyTorch**: 深度学习框架，用于模型训练和推理
- **yarr**: 机器人学习框架，提供Agent基类和相关工具
- **lru_cache**: 用于缓存计算结果，避免重复计算归一化统计信息
- **pickle**: 用于加载演示数据

**特别注意**: 代码中有注释 `# NO LANGUAGE - Removed CLIP import`，说明这是一个纯视觉版本，移除了语言处理功能。

---

## 2. ActBCVisionAgent 类结构

### 2.1 类定义与初始化 (行23-40)

```python
class ActBCVisionAgent(Agent):

    def __init__(self,
                 actor_network: nn.Module,
                 camera_names: List[str],
                 lr: float = 0.01,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 20.0,
                 episode_length: int = 400, train_demo_path=None, task_name=None):
        self._camera_names = camera_names
        self._actor = actor_network
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip
        self._episode_length = episode_length
        self.train_demo_path = train_demo_path
        self.task_name = task_name
        self.visual_targets = []  # 新增：用于存储可视化目标
```

**参数说明**:
- `actor_network`: Actor网络模型（ACT模型）
- `camera_names`: 相机名称列表（如 ['front', 'wrist']）
- `lr`: 学习率，默认0.01
- `weight_decay`: 权重衰减，默认1e-5
- `grad_clip`: 梯度裁剪阈值，默认20.0
- `episode_length`: 每个episode的最大长度，默认400步
- `train_demo_path`: 训练演示数据路径
- `task_name`: 任务名称

**成员变量初始化**:
- 存储所有超参数
- 保存actor网络引用
- 记录演示数据路径和任务名称用于计算归一化统计
- `visual_targets`: 新增成员变量，用于存储可视化目标（推理时返回）

---

### 2.2 模型构建方法 (行42-48)

```python
def build(self, training: bool, device: torch.device = None):
    if device is None:
        device = torch.device('cpu')
    self._actor = self._actor.to(device).train(training)
    self._actor_optimizer = self._actor.configure_optimizers()
    self._device = device
```

**功能**:
1. 将actor网络移动到指定设备（CPU/GPU）
2. 设置训练/评估模式
3. 配置优化器（通过actor网络自己的`configure_optimizers`方法）
4. 保存设备引用

---

### 2.3 重置方法 (行50-58)

```python
def reset(self):
    super(ActBCVisionAgent, self).reset()

    self._timestep = 0
    # .. input_dim = input_dim * 2 for bimanual
    self._all_time_actions = torch.zeros([self._episode_length,
                                          self._episode_length+self._actor.model.num_queries,
                                          self._actor.model.input_dim]).to(self._device)
    self._all_actions = None
```

**功能**:
- 在每个episode开始时调用
- 重置时间步计数器
- 初始化动作缓存（用于存储预测的动作序列）
- **重要**: 对于双臂机器人，`input_dim` 为 16（每个手臂8个维度：pos(3) + quat(4) + gripper(1)）

---

## 3. 训练相关方法

### 3.1 梯度更新方法 (行60-65)

```python
def _grad_step(self, loss, opt, model_params=None, clip=None):
    opt.zero_grad()
    loss.backward()
    if clip is not None and model_params is not None:
        nn.utils.clip_grad_value_(model_params, clip)
    opt.step()
```

**功能**: 标准的PyTorch训练步骤
1. 清零梯度
2. 反向传播
3. 梯度裁剪（防止梯度爆炸）
4. 更新参数

**注意**: 当前 `update()` 方法中直接调用 optimizer 的方法，未使用此辅助函数。

---

### 3.2 归一化统计计算 (行69-113)

```python
@lru_cache()
def train_stats(self):

    right_gripper_poses = []  # 7D: xyz(3) + quat(4)
    left_gripper_poses = []

    right_gripper_open = []   # 1D: gripper open state
    left_gripper_open = []

    episodes_dir = f"{self.train_demo_path}/{self.task_name}.train/all_variations/episodes/"

    for episode in os.listdir(episodes_dir):
        with open(os.path.join(episodes_dir, episode, "low_dim_obs.pkl"), "br") as f:
            d = pickle.load(f)

        for o in d:
            right_gripper_poses.append(o.right.gripper_pose)
            left_gripper_poses.append(o.left.gripper_pose)

            right_gripper_open.append([o.right.gripper_open])
            left_gripper_open.append([o.left.gripper_open])

    right_gripper_poses = np.asarray(right_gripper_poses, dtype=np.float32)
    left_gripper_poses = np.asarray(left_gripper_poses, dtype=np.float32)

    right_gripper_open = np.asarray(right_gripper_open, dtype=np.float32)
    left_gripper_open = np.asarray(left_gripper_open, dtype=np.float32)

    # Compute statistics for position (xyz) only, not quaternion
    # Quaternions are unit vectors and should not be normalized with z-score
    stats = {
        "right_pos_mean": right_gripper_poses[:, :3].mean(axis=0),
        "right_pos_std": right_gripper_poses[:, :3].std(axis=0),

        "left_pos_mean": left_gripper_poses[:, :3].mean(axis=0),
        "left_pos_std": left_gripper_poses[:, :3].std(axis=0),

        "right_gripper_open_mean": right_gripper_open.mean(axis=0),
        "right_gripper_open_std": right_gripper_open.std(axis=0),

        "left_gripper_open_mean":  left_gripper_open.mean(axis=0),
        "left_gripper_open_std": left_gripper_open.std(axis=0)
    }

    return {k: torch.from_numpy(v).to(self._device) for k,v in stats.items()}
```

**核心功能**:
- 从演示数据中计算归一化统计信息（均值和标准差）
- 使用`@lru_cache()`装饰器自动缓存结果，避免重复计算
- **重要变化**: 收集的是末端执行器位姿 (`gripper_pose`) 和夹爪开合状态 (`gripper_open`)，而非关节位置

**处理流程**:
1. 遍历指定路径下的所有训练episode（`{train_demo_path}/{task_name}.train/all_variations/episodes/`）
2. 从`low_dim_obs.pkl`文件加载低维观测数据
3. 收集左右手臂的末端执行器位姿（`o.right.gripper_pose`, `o.left.gripper_pose`）和夹爪开合状态（`o.right.gripper_open`, `o.left.gripper_open`）
4. **只对位置(xyz)计算统计信息**，四元数不做Z-score归一化（因为四元数是单位向量）

**返回统计信息**:
```python
stats = {
    "right_pos_mean": ...,      # (3,) - 右臂末端位置均值
    "right_pos_std": ...,       # (3,) - 右臂末端位置标准差
    "left_pos_mean": ...,       # (3,) - 左臂末端位置均值
    "left_pos_std": ...,        # (3,) - 左臂末端位置标准差
    "right_gripper_open_mean": ...,  # (1,) - 右夹爪开合均值
    "right_gripper_open_std": ...,   # (1,) - 右夹爪开合标准差
    "left_gripper_open_mean": ...,   # (1,) - 左夹爪开合均值
    "left_gripper_open_std": ...     # (1,) - 左夹爪开合标准差
}
```

**与旧版本的区别**:
| 旧版本 | 新版本 |
|-------|-------|
| `right_joints_mean/std` (7D) | `right_pos_mean/std` (3D) |
| `left_joints_mean/std` (7D) | `left_pos_mean/std` (3D) |
| `right_gripper_mean/std` (1D) | `right_gripper_open_mean/std` (1D) |
| `left_gripper_mean/std` (1D) | `left_gripper_open_mean/std` (1D) |

---

### 3.3 位置预处理方法 (行124-160)

```python
def preprocess_qpos(self, observation: dict):

    stats = self.train_stats()

    # Right gripper pose: normalize position (xyz), keep quaternion as is
    # Handle both 3D (B, timesteps, 7) from training and 2D (B, 7) from evaluation
    right_pose = observation['right_gripper_pose']
    if right_pose.dim() == 3:
        right_pose = right_pose[:, -1]  # (B, T, 7) -> (B, 7), take last timestep (current state)
    # else: already (B, 7), use directly
    right_pos_norm = self.normalize_z(right_pose[:, :3], stats["right_pos_mean"], stats["right_pos_std"])
    right_quat = right_pose[:, 3:7]  # Keep quaternion as is

    # Right gripper open state (already 0/1, keep as is)
    # Handle both 3D (B, timesteps, 1) from training and 2D (B, 1) from evaluation
    right_gripper = observation['right_gripper_open']
    if right_gripper.dim() == 3:
        right_gripper = right_gripper[:, -1]  # (B, T, 1) -> (B, 1), take last timestep

    # Left gripper pose
    left_pose = observation['left_gripper_pose']
    if left_pose.dim() == 3:
        left_pose = left_pose[:, -1]  # (B, T, 7) -> (B, 7), take last timestep
    left_pos_norm = self.normalize_z(left_pose[:, :3], stats["left_pos_mean"], stats["left_pos_std"])
    left_quat = left_pose[:, 3:7]  # Keep quaternion as is

    # Left gripper open state
    left_gripper = observation['left_gripper_open']
    if left_gripper.dim() == 3:
        left_gripper = left_gripper[:, -1]  # (B, T, 1) -> (B, 1), take last timestep

    # Concatenate: [right_pos(3), right_quat(4), right_gripper(1), left_pos(3), left_quat(4), left_gripper(1)]
    qpos = torch.cat([right_pos_norm, right_quat, right_gripper,
                      left_pos_norm, left_quat, left_gripper], dim=-1)

    return qpos
```

**功能**: 预处理当前机器人末端执行器状态（qpos），用于推理时的输入

**关键处理逻辑**:
1. **维度自适应**: 自动处理3D张量（训练时B, T, dim）和2D张量（推理时B, dim）
2. **位置归一化**: 只对xyz位置进行Z-score归一化
3. **四元数保持**: 四元数保持原值不做归一化（因为四元数是单位向量）
4. **夹爪状态保持**: gripper_open 是0/1值，保持原值

**输出格式**:
- `qpos`: (B, 16) - 归一化的当前状态
- 格式: `[right_pos(3), right_quat(4), right_gripper(1), left_pos(3), left_quat(4), left_gripper(1)]`

---

### 3.4 动作预处理方法 (行164-197)

```python
def preprocess_action(self, replay_sample: dict):

    stats = self.train_stats()

    # Process previous (current) state: [right_pose(7), right_gripper(1), left_pose(7), left_gripper(1)]
    # Use [:, -1] to get the last (current) timestep, works for any prev_action_horizon
    right_prev_pose = replay_sample['right_prev_gripper_pose'][:, -1]  # (B, T, 7) -> (B, 7)
    right_prev_pos_norm = self.normalize_z(right_prev_pose[:, :3], stats["right_pos_mean"], stats["right_pos_std"])
    right_prev_quat = right_prev_pose[:, 3:7]
    right_prev_gripper = replay_sample['right_prev_gripper_open'][:, -1]  # (B, T, 1) -> (B, 1)

    left_prev_pose = replay_sample['left_prev_gripper_pose'][:, -1]  # (B, T, 7) -> (B, 7)
    left_prev_pos_norm = self.normalize_z(left_prev_pose[:, :3], stats["left_pos_mean"], stats["left_pos_std"])
    left_prev_quat = left_prev_pose[:, 3:7]
    left_prev_gripper = replay_sample['left_prev_gripper_open'][:, -1]  # (B, T, 1) -> (B, 1)

    qpos = torch.cat([right_prev_pos_norm, right_prev_quat, right_prev_gripper,
                      left_prev_pos_norm, left_prev_quat, left_prev_gripper], dim=-1)

    # Process action sequence: (B, T, 16)
    right_next_pose = replay_sample['right_next_gripper_pose']  # (B, T, 7)
    right_next_pos_norm = self.normalize_z(right_next_pose[:, :, :3], stats["right_pos_mean"], stats["right_pos_std"])
    right_next_quat = right_next_pose[:, :, 3:7]
    right_next_gripper = replay_sample['right_next_gripper_open']  # (B, T, 1)

    left_next_pose = replay_sample['left_next_gripper_pose']  # (B, T, 7)
    left_next_pos_norm = self.normalize_z(left_next_pose[:, :, :3], stats["left_pos_mean"], stats["left_pos_std"])
    left_next_quat = left_next_pose[:, :, 3:7]
    left_next_gripper = replay_sample['left_next_gripper_open']  # (B, T, 1)

    action_seq = torch.cat([right_next_pos_norm, right_next_quat, right_next_gripper,
                            left_next_pos_norm, left_next_quat, left_next_gripper], dim=-1)

    return qpos, action_seq
```

**功能**: 预处理训练样本中的动作序列和当前状态

**处理流程**:

1. **处理当前状态 (qpos)**:
   - 从 `right_prev_gripper_pose` 和 `left_prev_gripper_pose` 获取当前位姿
   - 使用 `[:, -1]` 取最后一个时间步（当前状态）
   - 分别归一化位置xyz，保持四元数原值

2. **处理动作序列 (action_seq)**:
   - 从 `right_next_gripper_pose` 和 `left_next_gripper_pose` 获取未来动作序列
   - 对整个序列的位置进行归一化，保持四元数原值

**输出**:
- `qpos`: (B, 16) - 当前状态（归一化后）
- `action_seq`: (B, action_horizon, 16) - 未来动作序列（归一化后）

**Replay Buffer 字段说明**:
| 字段名 | 维度 | 说明 |
|-------|------|------|
| `right_prev_gripper_pose` | (B, prev_horizon, 7) | 右臂历史位姿 |
| `right_prev_gripper_open` | (B, prev_horizon, 1) | 右臂历史夹爪状态 |
| `left_prev_gripper_pose` | (B, prev_horizon, 7) | 左臂历史位姿 |
| `left_prev_gripper_open` | (B, prev_horizon, 1) | 左臂历史夹爪状态 |
| `right_next_gripper_pose` | (B, next_horizon, 7) | 右臂未来位姿 |
| `right_next_gripper_open` | (B, next_horizon, 1) | 右臂未来夹爪状态 |
| `left_next_gripper_pose` | (B, next_horizon, 7) | 左臂未来位姿 |
| `left_next_gripper_open` | (B, next_horizon, 1) | 左臂未来夹爪状态 |

---

### 3.5 图像预处理 (行199-215)

```python
def preprocess_images(self, replay_sample: dict):
    stacked_rgb = []
    stacked_point_cloud = []

    for camera in self._camera_names:
        rgb = replay_sample['%s_rgb' % camera]
        rgb = rgb if rgb.dim() == 4 else rgb[:,0]
        stacked_rgb.append(rgb)

        point_cloud = replay_sample['%s_point_cloud' % camera]
        point_cloud = point_cloud if point_cloud.dim() == 4 else point_cloud[:,0]
        stacked_point_cloud.append(point_cloud)

    stacked_rgb = torch.stack(stacked_rgb, dim=1)
    stacked_point_cloud = torch.stack(stacked_point_cloud, dim=1)

    return stacked_rgb, stacked_point_cloud
```

**功能**: 从多个相机收集RGB图像和点云数据并堆叠

**输出**:
- `stacked_rgb`: (N, num_cameras, 3, H, W) - 堆叠的RGB图像
- `stacked_point_cloud`: (N, num_cameras, 3, H, W) - 堆叠的点云数据

**维度处理**:
- 自动处理维度不匹配问题（如果是5维则取第2维的第一个元素）
- 确保所有数据都是4维张量

---

### 3.6 训练更新方法 (行217-243)

```python
def update(self, step: int, replay_sample: dict) -> dict:
    # NO LANGUAGE - removed lang_goal_emb handling
    robot_state = replay_sample['low_dim_state']

    # preprocess input
    qpos, action_seq = self.preprocess_action(replay_sample)
    stacked_rgb, stacked_point_cloud = self.preprocess_images(replay_sample)
    is_pad = replay_sample['is_pad'].bool()

    # forward pass
    loss_dict = self._actor(qpos, stacked_rgb, action_seq, is_pad)

    # gradient step
    loss = loss_dict['total_losses']
    loss.backward()
    self._actor_optimizer.step()
    self._actor_optimizer.zero_grad()

    self._summaries = {
        'loss': loss_dict['total_losses'],
        'l1': loss_dict['l1'],
        'right_l1': loss_dict['right_l1'],
        'left_l1': loss_dict['left_l1'],
        'kl': loss_dict['kl'],
    }

    return loss_dict
```

**功能**: 执行一次训练更新

**流程**:
1. **数据准备**:
   - 预处理动作和状态（调用 `preprocess_action`）
   - 预处理图像和点云（调用 `preprocess_images`）
   - 获取填充标记 `is_pad`

2. **前向传播**:
   - 调用 actor 网络计算损失

3. **梯度更新**:
   - 反向传播
   - 优化器更新
   - 清零梯度

4. **记录训练指标**

**损失函数组成**:
- `total_losses`: 总损失 = l1 + kl * kl_weight
- `l1`: L1损失（预测动作与真实动作的差异）
- `right_l1`: 右臂的L1损失
- `left_l1`: 左臂的L1损失
- `kl`: KL散度损失（用于VAE类型的模型）

**注意**: 该方法不再使用 `_grad_step` 辅助方法，而是直接调用optimizer的step和zero_grad

---

## 4. ACT Policy 损失函数详解

ACT Policy (`act_policy.py`) 中的损失函数计算非常重要，以下是详细说明：

```python
def forward(self, qpos, image, actions=None, is_pad=None):
    # ... 前向传播获得 a_hat, mu, logvar ...

    if actions is not None:  # training time
        # Action format: [right_pos(3), right_quat(4), right_gripper(1),
        #                 left_pos(3), left_quat(4), left_gripper(1)] = 16D

        # Right arm: position, quaternion, gripper
        right_pos_gt, right_pos_pred = actions[:, :, 0:3], a_hat[:, :, 0:3]
        right_quat_gt, right_quat_pred = actions[:, :, 3:7], a_hat[:, :, 3:7]
        right_gripper_gt, right_gripper_pred = actions[:, :, 7], a_hat[:, :, 7]

        # Left arm
        left_pos_gt, left_pos_pred = actions[:, :, 8:11], a_hat[:, :, 8:11]
        left_quat_gt, left_quat_pred = actions[:, :, 11:15], a_hat[:, :, 11:15]
        left_gripper_gt, left_gripper_pred = actions[:, :, 15], a_hat[:, :, 15]

        # Weighted L1 loss: higher weight for position
        pos_weight = 3.0      # Position is more critical
        quat_weight = 1.0     # Quaternion
        gripper_weight = 3.0  # Gripper is important for manipulation

        # Compute weighted losses with is_pad masking
        right_pos_l1 = (right_pos_loss * ~is_pad.unsqueeze(-1)).mean() * pos_weight
        right_quat_l1 = (right_quat_loss * ~is_pad.unsqueeze(-1)).mean() * quat_weight
        right_gripper_l1 = (right_gripper_loss * ~is_pad).mean() * gripper_weight
        # ... 左臂类似 ...

        loss_dict['total_losses'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
```

**损失权重设计**:
- `pos_weight = 3.0`: 位置损失权重较高，因为位置精度对操作成功至关重要
- `quat_weight = 1.0`: 四元数损失权重较低
- `gripper_weight = 3.0`: 夹爪损失权重较高，因为夹爪动作直接影响抓取成功率

**细分损失记录**:
```python
loss_dict = {
    'right_l1': right_l1,
    'left_l1': left_l1,
    'right_pos_l1': right_pos_l1,
    'right_quat_l1': right_quat_l1,
    'right_gripper_l1': right_gripper_l1,
    'left_pos_l1': left_pos_l1,
    'left_quat_l1': left_quat_l1,
    'left_gripper_l1': left_gripper_l1,
    'l1': l1,
    'kl': total_kld[0],
    'total_losses': l1 + kl * kl_weight
}
```

---

## 5. 推理相关方法

### 5.1 辅助归一化方法 (行245-248)

```python
def _normalize_quat(self, x):
    """Normalize quaternion to unit length"""
    return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)
```

**功能**: 将四元数归一化为单位四元数（用于推理时的后处理）

---

### 5.2 推理执行方法 (行252-316)

```python
def act(self, step: int, observation: dict,
        deterministic=False) -> ActResult:
    # NO LANGUAGE - removed lang_goal_tokens and CLIP encoding

    action_horizon = self._actor.model.num_queries
    query_freq = 1

    stats = self.train_stats()

    if self._timestep % query_freq == 0:
        with torch.no_grad():
            # preprocess input
            qpos = self.preprocess_qpos(observation)
            stacked_rgb, stacked_point_cloud = self.preprocess_images(observation)

            # forward pass
            self._all_actions = self._actor(qpos, stacked_rgb, actions=None, is_pad=None)

    # temporal aggregation
    t = self._timestep

    self._all_time_actions[[t], t:t + action_horizon] = self._all_actions
    actions_for_curr_step = self._all_time_actions[:, t]
    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]
    k = 0.01
    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
    exp_weights = exp_weights / exp_weights.sum()
    exp_weights = torch.from_numpy(exp_weights).to(self._device).unsqueeze(dim=1)
    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
    raw_action = raw_action[0]

    # raw_action: [right_pos_norm(3), right_quat(4), right_gripper(1),
    #               left_pos_norm(3), left_quat(4), left_gripper(1)] = 16D

    # Right arm: unnormalize position, normalize quaternion, discretize gripper
    right_pos = self.unnormalize_z(raw_action[0:3], stats["right_pos_mean"], stats["right_pos_std"])
    right_quat = raw_action[3:7]
    right_quat_normalized = right_quat / torch.norm(right_quat)  # Ensure unit quaternion
    right_gripper = torch.tensor([1.0 if raw_action[7] > 0.5 else 0.0], device=self._device)  # Discretize
    right_ignore_collision = torch.tensor([1.0], device=self._device)  # Hard-coded

    # Left arm
    left_pos = self.unnormalize_z(raw_action[8:11], stats["left_pos_mean"], stats["left_pos_std"])
    left_quat = raw_action[11:15]
    left_quat_normalized = left_quat / torch.norm(left_quat)
    left_gripper = torch.tensor([1.0 if raw_action[15] > 0.5 else 0.0], device=self._device)
    left_ignore_collision = torch.tensor([1.0], device=self._device)

    # Output format: [right_pos(3), right_quat(4), right_gripper(1), right_ignore(1),
    #                 left_pos(3), left_quat(4), left_gripper(1), left_ignore(1)] = 18D
    raw_action = torch.cat([right_pos, right_quat_normalized, right_gripper, right_ignore_collision,
                            left_pos, left_quat_normalized, left_gripper, left_ignore_collision], dim=-1)

    self._timestep += 1

    return ActResult(raw_action.detach().cpu().numpy(), visual_targets=self.visual_targets)
```

**功能**: 根据当前观测生成动作，使用时序聚合（temporal aggregation）提高动作平滑性

**关键变量**:
- `action_horizon`: 每次查询生成的动作序列长度（`num_queries`）
- `query_freq`: 查询频率（设置为1，表示每步都查询）
- `_timestep`: 当前时间步
- `_all_actions`: 当前查询生成的动作序列
- `_all_time_actions`: 存储所有历史查询结果的缓冲区，用于时序聚合

#### 5.2.1 动作生成（每个时间步）

```python
if self._timestep % query_freq == 0:
    with torch.no_grad():
        # 1. 预处理当前末端执行器状态
        qpos = self.preprocess_qpos(observation)

        # 2. 预处理图像和点云
        stacked_rgb, stacked_point_cloud = self.preprocess_images(observation)

        # 3. 查询策略网络
        self._all_actions = self._actor(qpos, stacked_rgb, actions=None, is_pad=None)
```

#### 5.2.2 时序聚合（Temporal Aggregation）

```python
# 1. 将当前预测存入缓冲区
t = self._timestep
self._all_time_actions[[t], t:t + action_horizon] = self._all_actions

# 2. 提取所有对当前时间步的预测
actions_for_curr_step = self._all_time_actions[:, t]

# 3. 过滤掉未填充的预测（全零的）
actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
actions_for_curr_step = actions_for_curr_step[actions_populated]

# 4. 计算指数权重
k = 0.01
exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
exp_weights = exp_weights / exp_weights.sum()
exp_weights = torch.from_numpy(exp_weights).to(self._device).unsqueeze(dim=1)

# 5. 加权平均
raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
raw_action = raw_action[0]
```

**时序聚合原理**:
- 在时间步 `t`，有多个历史查询都预测了该时间步的动作
- 例如：时间步5的预测、时间步4的预测、时间步3的预测等都包含了对当前时间步的动作预测
- 使用指数权重对这些预测进行加权平均，越近的预测权重越大
- 这种方式可以平滑动作序列，减少抖动

#### 5.2.3 后处理与反归一化

```python
# raw_action: [right_pos_norm(3), right_quat(4), right_gripper(1),
#               left_pos_norm(3), left_quat(4), left_gripper(1)] = 16D

# Right arm: unnormalize position, normalize quaternion, discretize gripper
right_pos = self.unnormalize_z(raw_action[0:3], stats["right_pos_mean"], stats["right_pos_std"])
right_quat = raw_action[3:7]
right_quat_normalized = right_quat / torch.norm(right_quat)  # Ensure unit quaternion
right_gripper = torch.tensor([1.0 if raw_action[7] > 0.5 else 0.0], device=self._device)  # Discretize
right_ignore_collision = torch.tensor([1.0], device=self._device)  # Hard-coded

# Left arm (similar)
left_pos = self.unnormalize_z(raw_action[8:11], stats["left_pos_mean"], stats["left_pos_std"])
left_quat = raw_action[11:15]
left_quat_normalized = left_quat / torch.norm(left_quat)
left_gripper = torch.tensor([1.0 if raw_action[15] > 0.5 else 0.0], device=self._device)
left_ignore_collision = torch.tensor([1.0], device=self._device)
```

**后处理步骤**:
1. **位置反归一化**: 将归一化的xyz位置转换回真实坐标
2. **四元数归一化**: 确保输出的四元数是单位四元数
3. **夹爪离散化**: 将连续的夹爪值离散化为0/1（阈值0.5）
4. **添加ignore_collision标志**: 硬编码为1.0

#### 5.2.4 输出格式

```python
# Output format: [right_pos(3), right_quat(4), right_gripper(1), right_ignore(1),
#                 left_pos(3), left_quat(4), left_gripper(1), left_ignore(1)] = 18D
raw_action = torch.cat([right_pos, right_quat_normalized, right_gripper, right_ignore_collision,
                        left_pos, left_quat_normalized, left_gripper, left_ignore_collision], dim=-1)
```

**重要变化**: 输出从旧版本的16D变为18D，增加了 `ignore_collision` 标志

**动作维度分解**:
```
输出格式: [right(9), left(9)] = 18D
├── right(9)
│   ├── pos(3): 右臂末端xyz位置
│   ├── quat(4): 右臂末端四元数姿态（单位化后）
│   ├── gripper(1): 右夹爪（离散化0/1）
│   └── ignore_collision(1): 忽略碰撞标志（硬编码1.0）
└── left(9)
    ├── pos(3): 左臂末端xyz位置
    ├── quat(4): 左臂末端四元数姿态（单位化后）
    ├── gripper(1): 左夹爪（离散化0/1）
    └── ignore_collision(1): 忽略碰撞标志（硬编码1.0）
```

---

## 6. 工具方法

### 6.1 归一化方法 (行117-121)

```python
def normalize_z(self, data, mean, std):
    return (data - mean) / std

def unnormalize_z(self, data, mean, std):
    return data * std + mean
```

**功能**: Z-score归一化和反归一化
- 归一化: `z = (x - μ) / σ`
- 反归一化: `x = z * σ + μ`

---

### 6.2 摘要方法 (行318-332)

```python
def update_summaries(self) -> List[Summary]:
    summaries = []
    for n, v in self._summaries.items():
        summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))
    return summaries

def act_summaries(self) -> List[Summary]:
    return []
```

**功能**: 为训练监控提供摘要信息
- `update_summaries`: 返回训练损失等标量摘要（包括 loss, l1, right_l1, left_l1, kl）
- `act_summaries`: 推理时的摘要（此处为空）

---

### 6.3 权重保存与加载 (行334-342)

#### 6.3.1 加载权重

```python
def load_weights(self, savedir: str):
    self._actor.load_state_dict(
        torch.load(os.path.join(savedir, 'bc_actor.pt'),
                   map_location=torch.device('cpu')))
    print('Loaded weights from %s' % savedir)
```

**特点**:
- 简单的权重加载
- 自动处理设备映射（默认映射到CPU）
- 文件名固定为 `bc_actor.pt`
- 不需要加载统计信息（会通过 `@lru_cache` 自动计算）

#### 6.3.2 保存权重

```python
def save_weights(self, savedir: str):
    torch.save(self._actor.state_dict(),
               os.path.join(savedir, 'bc_actor.pt'))
```

**特点**:
- 只保存模型权重，不保存统计信息
- 统计信息通过 `@lru_cache` 在需要时动态计算
- 保存文件名: `bc_actor.pt`

---

## 7. 数据流和维度总结

### 7.1 输入数据维度

| 数据类型 | 维度 | 说明 |
|---------|------|------|
| qpos | (N, 16) 或 (1, 16) | 当前状态，right(3+4+1) + left(3+4+1) |
| stacked_rgb | (N, num_cameras, 3, H, W) | 多相机RGB图像 |
| stacked_point_cloud | (N, num_cameras, 3, H, W) | 多相机点云数据 |
| action_seq | (N, action_horizon, 16) | 动作序列标签（训练时） |
| is_pad | (N, action_horizon) | 填充标记（训练时） |

### 7.2 输出数据维度

| 数据类型 | 维度 | 说明 |
|---------|------|------|
| predicted_actions | (1, action_horizon, 16) | 预测的动作序列（模型输出） |
| raw_action | (18,) | 单步动作输出（经过后处理） |

### 7.3 双臂机器人动作维度分解

```
模型内部表示: [right(8), left(8)] = 16D
├── right(8)
│   ├── pos(3): 右臂末端xyz位置
│   ├── quat(4): 右臂末端四元数姿态
│   └── gripper(1): 右夹爪开合状态
└── left(8)
    ├── pos(3): 左臂末端xyz位置
    ├── quat(4): 左臂末端四元数姿态
    └── gripper(1): 左夹爪开合状态

环境接口输出: [right(9), left(9)] = 18D
├── right(9)
│   ├── pos(3): 右臂末端xyz位置（反归一化后）
│   ├── quat(4): 右臂末端四元数（单位化后）
│   ├── gripper(1): 右夹爪（离散化0/1）
│   └── ignore_collision(1): 忽略碰撞标志
└── left(9)
    ├── pos(3): 左臂末端xyz位置（反归一化后）
    ├── quat(4): 左臂末端四元数（单位化后）
    ├── gripper(1): 左夹爪（离散化0/1）
    └── ignore_collision(1): 忽略碰撞标志
```

---

## 8. 关键设计特点

### 8.1 行为克隆（Behavior Cloning）
- 从演示数据中学习策略
- 使用监督学习方式训练
- 损失函数包括加权L1损失和KL散度

### 8.2 纯视觉输入
- 移除了语言条件（注释中提到 NO LANGUAGE）
- 只依赖视觉和机器人状态
- 支持多相机输入

### 8.3 末端执行器控制（End-Effector Control）
- **重要特性**: 使用末端执行器位姿（gripper_pose）而非关节位置
- 动作空间: 位置(xyz) + 姿态(quaternion) + 夹爪(open/close)
- 更直观的任务空间控制

### 8.4 双臂协调控制
- 同时控制左右两个机械臂
- 独立归一化每个手臂的统计信息
- 独立监控每个手臂的损失

### 8.5 动作序列预测
- 一次预测多步动作（action_horizon）
- 支持动作序列的时序建模
- 可以通过query_freq控制查询频率

### 8.6 高效的统计计算
- 使用`@lru_cache`缓存归一化统计
- 避免重复计算
- 节省内存和计算资源

### 8.7 智能归一化策略
- **位置(xyz)**: Z-score归一化
- **四元数**: 保持原值（因为四元数是单位向量）
- **夹爪状态**: 保持原值（0/1布尔值）

### 8.8 后处理步骤
- 四元数单位化: 确保输出有效的单位四元数
- 夹爪离散化: 将连续值转换为离散0/1
- 添加ignore_collision标志

---

## 9. 训练和推理流程图

### 9.1 训练流程

```
replay_sample (batch data)
    │
    ├─[preprocess_action]
    │   ├── right_prev_gripper_pose[:, -1] -> normalize pos, keep quat
    │   ├── left_prev_gripper_pose[:, -1] -> normalize pos, keep quat
    │   ├── right_next_gripper_pose -> normalize pos, keep quat
    │   └── left_next_gripper_pose -> normalize pos, keep quat
    │   ↓
    │   qpos (B, 16), action_seq (B, T, 16)
    │
    ├─[preprocess_images]
    │   ↓
    │   stacked_rgb, stacked_point_cloud
    │
    └─[actor forward]
        ↓
    loss_dict {
        total_losses,
        l1, right_l1, left_l1,
        right_pos_l1, right_quat_l1, right_gripper_l1,
        left_pos_l1, left_quat_l1, left_gripper_l1,
        kl
    }
        ↓
    [loss.backward()]
        ↓
    [optimizer.step()]
        ↓
    [optimizer.zero_grad()]
        ↓
    [update_summaries]
```

### 9.2 推理流程

```
observation (current state)
    │
    ├─[preprocess_qpos]
    │   ├── right_gripper_pose -> normalize pos (xyz), keep quat
    │   ├── left_gripper_pose -> normalize pos (xyz), keep quat
    │   └── gripper_open -> keep as is
    │   ↓
    │   qpos (1, 16)
    │
    ├─[preprocess_images]
    │   ↓
    │   stacked_rgb, stacked_point_cloud
    │
    └─[actor forward (no gradient)]
        ↓
    predicted actions (1, action_horizon, 16)
        ↓
    [temporal aggregation]
        ├─ store in _all_time_actions buffer
        ├─ extract all predictions for current timestep
        ├─ filter populated predictions
        ├─ compute exponential weights (k=0.01)
        └─ weighted average
        ↓
    raw_action (16D, normalized)
        ↓
    [post-processing]
        ├─ unnormalize pos (xyz)
        ├─ normalize quaternion (ensure unit)
        ├─ discretize gripper (>0.5 -> 1, <=0.5 -> 0)
        └─ add ignore_collision (hard-coded 1.0)
        ↓
    raw_action (18D, ready for environment)
        ↓
    [ActResult]
        ↓
    execute in environment
```

---

## 10. 与环境的交互接口

### 10.1 期望的observation格式

```python
observation = {
    # 末端执行器位姿
    'right_gripper_pose': torch.Tensor (N, 7) 或 (N, T, 7),  # xyz(3) + quat(4)
    'left_gripper_pose': torch.Tensor (N, 7) 或 (N, T, 7),   # xyz(3) + quat(4)

    # 夹爪开合状态
    'right_gripper_open': torch.Tensor (N, 1) 或 (N, T, 1),  # 0/1
    'left_gripper_open': torch.Tensor (N, 1) 或 (N, T, 1),   # 0/1

    # 视觉数据
    '<camera_name>_rgb': torch.Tensor (3, H, W) 或 (N, 3, H, W),
    '<camera_name>_point_cloud': torch.Tensor (3, H, W) 或 (N, 3, H, W),

    # 低维状态（训练时）
    'low_dim_state': torch.Tensor (N, 8),  # 可选
}
```

**注意**:
- 训练时 N 是 batch size
- 推理时通常 N=1
- 支持2D和3D输入，3D输入会自动取最后一个时间步

### 10.2 输出的action格式

```python
raw_action = np.array([
    # 右臂 (9D)
    right_pos[0],           # 右臂末端x
    right_pos[1],           # 右臂末端y
    right_pos[2],           # 右臂末端z
    right_quat[0],          # 右臂四元数w
    right_quat[1],          # 右臂四元数x
    right_quat[2],          # 右臂四元数y
    right_quat[3],          # 右臂四元数z
    right_gripper,          # 右夹爪 (0或1)
    right_ignore_collision, # 右臂忽略碰撞 (1.0)

    # 左臂 (9D)
    left_pos[0],            # 左臂末端x
    left_pos[1],            # 左臂末端y
    left_pos[2],            # 左臂末端z
    left_quat[0],           # 左臂四元数w
    left_quat[1],           # 左臂四元数x
    left_quat[2],           # 左臂四元数y
    left_quat[3],           # 左臂四元数z
    left_gripper,           # 左夹爪 (0或1)
    left_ignore_collision   # 左臂忽略碰撞 (1.0)
])  # shape: (18,)
```

**动作格式**: `[right(9), left(9)]` = 18D

---

## 11. 使用示例

### 11.1 初始化Agent

```python
from agents.act_bc_vision.act_bc_vision_agent import ActBCVisionAgent
from agents.act_bc_vision.act_policy import ACTPolicy

# 创建ACT policy网络
actor_net = ACTPolicy(cfg.method)

# 创建agent
agent = ActBCVisionAgent(
    actor_network=actor_net,
    camera_names=['front', 'wrist'],
    lr=1e-5,
    weight_decay=1e-4,
    grad_clip=20.0,
    episode_length=400,
    train_demo_path='/path/to/demos',
    task_name='pick_and_place'
)

# 构建agent（移动到GPU并初始化优化器）
agent.build(training=True, device=torch.device('cuda'))
```

### 11.2 训练

```python
# 准备训练数据（从replay buffer获取）
replay_sample = {
    # Previous states (当前时刻状态)
    'right_prev_gripper_pose': torch.Tensor (batch_size, prev_horizon, 7),
    'right_prev_gripper_open': torch.Tensor (batch_size, prev_horizon, 1),
    'left_prev_gripper_pose': torch.Tensor (batch_size, prev_horizon, 7),
    'left_prev_gripper_open': torch.Tensor (batch_size, prev_horizon, 1),

    # Next actions (未来动作序列)
    'right_next_gripper_pose': torch.Tensor (batch_size, next_horizon, 7),
    'right_next_gripper_open': torch.Tensor (batch_size, next_horizon, 1),
    'left_next_gripper_pose': torch.Tensor (batch_size, next_horizon, 7),
    'left_next_gripper_open': torch.Tensor (batch_size, next_horizon, 1),

    # Vision data
    'front_rgb': torch.Tensor (batch_size, 3, H, W),
    'front_point_cloud': torch.Tensor (batch_size, 3, H, W),
    'wrist_rgb': torch.Tensor (batch_size, 3, H, W),
    'wrist_point_cloud': torch.Tensor (batch_size, 3, H, W),

    # Padding mask
    'is_pad': torch.Tensor (batch_size, next_horizon),

    # Low dimensional state
    'low_dim_state': torch.Tensor (batch_size, 8),
}

# 执行一步训练更新
loss_dict = agent.update(step=0, replay_sample=replay_sample)

# 打印损失
print(f"Total loss: {loss_dict['total_losses']:.4f}")
print(f"L1 loss: {loss_dict['l1']:.4f}")
print(f"Right arm L1: {loss_dict['right_l1']:.4f}")
print(f"Left arm L1: {loss_dict['left_l1']:.4f}")
print(f"KL divergence: {loss_dict['kl']:.4f}")

# 获取训练摘要（用于tensorboard等）
summaries = agent.update_summaries()
```

### 11.3 推理

```python
# 重置episode（清空时序聚合缓冲区）
agent.reset()

# 在环境中执行
for step in range(episode_length):
    # 获取观测（末端执行器位姿格式）
    observation = {
        'right_gripper_pose': env.get_right_gripper_pose(),      # (1, 7)
        'right_gripper_open': env.get_right_gripper_open(),      # (1, 1)
        'left_gripper_pose': env.get_left_gripper_pose(),        # (1, 7)
        'left_gripper_open': env.get_left_gripper_open(),        # (1, 1)
        'front_rgb': env.get_camera_rgb('front'),                # (3, H, W)
        'front_point_cloud': env.get_camera_pc('front'),         # (3, H, W)
        'wrist_rgb': env.get_camera_rgb('wrist'),                # (3, H, W)
        'wrist_point_cloud': env.get_camera_pc('wrist'),         # (3, H, W)
    }

    # 生成动作（使用时序聚合）
    result = agent.act(step=step, observation=observation, deterministic=True)

    # 执行动作 (result.action shape: (18,))
    # 格式: [right_pos(3), right_quat(4), right_gripper(1), right_ignore(1),
    #        left_pos(3), left_quat(4), left_gripper(1), left_ignore(1)]
    env.step(result.action)
```

### 11.4 保存和加载模型

```python
# 保存模型权重（保存为 bc_actor.pt）
agent.save_weights('/path/to/checkpoints')

# 加载模型权重
agent.load_weights('/path/to/checkpoints')

# 注意：统计信息不需要保存，会通过 @lru_cache 自动从训练数据计算
```

---

## 12. 注意事项和最佳实践

### 12.1 数据预处理
- 确保所有相机名称在配置和数据中一致
- 每个相机需要提供RGB和点云数据
- `preprocess_qpos` 和 `preprocess_action` 都能自动处理2D和3D输入
- 3D输入会使用 `[:, -1]` 取最后一个时间步

### 12.2 归一化策略
- **位置(xyz)**: 使用Z-score归一化（从训练数据计算均值和标准差）
- **四元数**: 保持原值不做归一化（四元数是单位向量）
- **夹爪状态**: 保持原值（0/1布尔值）
- 归一化统计信息从 `{train_demo_path}/{task_name}.train/all_variations/episodes/` 路径加载
- 使用 `@lru_cache` 缓存，避免重复计算

### 12.3 训练稳定性
- 使用梯度裁剪防止梯度爆炸（默认阈值20.0，但当前未在update中使用）
- 监控各个手臂的独立损失（`right_l1`, `left_l1`）
- 监控各个组件的损失（`pos_l1`, `quat_l1`, `gripper_l1`）
- 注意KL散度和L1损失的平衡

### 12.4 损失权重设计
- `pos_weight = 3.0`: 位置损失权重较高
- `quat_weight = 1.0`: 四元数损失权重较低
- `gripper_weight = 3.0`: 夹爪损失权重较高
- 可以根据具体任务调整这些权重

### 12.5 时序聚合（Temporal Aggregation）
- **关键特性**: 使用指数加权平均多个历史预测，提高动作平滑性
- 每个时间步的动作由多个历史查询的预测加权得到
- 指数权重系数 `k=0.01`，越近的预测权重越大
- `_all_time_actions` 缓冲区大小为 `(episode_length, episode_length+num_queries, input_dim)`
- 必须在每个episode开始时调用 `reset()` 清空缓冲区

### 12.6 后处理步骤
- **四元数归一化**: 确保输出有效的单位四元数
- **夹爪离散化**: 阈值0.5，大于0.5输出1.0，否则输出0.0
- **ignore_collision**: 硬编码为1.0

### 12.7 推理效率
- `query_freq=1` 时每步都查询策略网络（当前默认设置）
- 时序聚合机制会自动平滑动作，无需额外滤波
- 推理时使用 `torch.no_grad()` 禁用梯度计算
- 所有预处理操作在推理时自动执行

### 12.8 设备管理
- 确保所有张量在正确的设备上（CPU/GPU）
- 加载模型时使用 `map_location=torch.device('cpu')` 默认映射到CPU
- 归一化统计信息会自动移动到与模型相同的设备

---

## 13. 潜在的改进方向

### 13.1 已实现的特性
- ✅ **点云输入**: 已添加点云数据处理
- ✅ **时序聚合**: 已实现指数加权的时序聚合机制
- ✅ **末端执行器控制**: 使用任务空间位姿控制
- ✅ **智能归一化**: 位置归一化，四元数保持原值
- ✅ **双臂独立监控**: 分别跟踪左右臂的损失
- ✅ **加权损失函数**: 位置和夹爪权重更高
- ✅ **夹爪离散化**: 推理时自动离散化
- ✅ **四元数单位化**: 确保输出有效四元数

### 13.2 可以进一步改进的方向

1. **语言条件**
   - 当前是纯视觉版本（NO LANGUAGE）
   - 可以添加语言指令作为任务条件（参考 act_bc_lang 版本）
   - 使用CLIP编码语言描述

2. **深度图处理**
   - 当前replay buffer中移除了depth数据
   - 可以考虑使用深度图作为额外输入模态

3. **自适应查询频率**
   - 当前固定 `query_freq=1`
   - 可以根据任务复杂度动态调整查询频率
   - 简单动作可以使用更低频率查询，节省计算

4. **在线学习**
   - 添加在线更新机制
   - 支持从交互中持续学习

5. **不确定性估计**
   - 添加预测不确定性的估计
   - 用于主动学习或安全检查

6. **异步执行**
   - 支持异步动作生成和执行
   - 在执行当前动作时预测下一步动作

7. **历史状态编码**
   - 当前只使用当前状态
   - 可以添加历史状态的时序编码（LSTM/Transformer）

8. **动态时序聚合权重**
   - 当前固定 `k=0.01`
   - 可以根据预测置信度动态调整权重

9. **可配置的ignore_collision**
   - 当前硬编码为1.0
   - 可以改为可配置参数或从观测中获取

10. **梯度裁剪集成**
    - 当前 `_grad_step` 方法未被使用
    - 可以在 `update()` 中集成梯度裁剪

---

## 14. 代码质量评价

### 优点
- ✅ **结构清晰**: 预处理、训练、推理逻辑分离良好
- ✅ **与lang版本对齐**: 除语言相关功能外，处理逻辑与 act_bc_lang 基本一致
- ✅ **时序聚合**: 实现了先进的时序聚合机制，提高动作平滑性
- ✅ **缓存优化**: 使用 `@lru_cache` 避免重复计算统计信息
- ✅ **双臂独立监控**: 分别跟踪左右臂的损失，便于调试
- ✅ **多模态输入**: 支持RGB和点云数据
- ✅ **智能归一化**: 针对不同数据类型使用不同归一化策略
- ✅ **加权损失**: 根据重要性对不同组件使用不同权重
- ✅ **维度自适应**: 自动处理2D和3D输入

### 可改进之处
- ⚠️ **硬编码维度**: 3、4、7、8、16、18等维度可以作为配置参数
- ⚠️ **未使用的方法**: `_grad_step` 方法未使用但保留在代码中
- ⚠️ **输入验证**: 缺少维度检查和错误提示
- ⚠️ **文档字符串**: 缺少标准的Python docstring
- ⚠️ **单元测试**: 没有配套的测试代码
- ⚠️ **ignore_collision硬编码**: 可以改为可配置

### 代码一致性
- ✅ 文件命名统一为 `bc_actor.pt`
- ✅ 移除了调试代码（ipdb）
- ✅ 图像预处理逻辑与lang版本一致
- ✅ 时序聚合逻辑与lang版本一致

---

## 15. 版本变更总结

### 从关节空间到任务空间的重大变化

| 特性 | 旧版本（关节空间） | 新版本（任务空间） |
|-----|------------------|------------------|
| 状态表示 | joint_positions (7D) + gripper_joint_positions | gripper_pose (7D: xyz+quat) + gripper_open |
| 动作表示 | 关节位置增量 | 末端执行器位姿 |
| 模型输出 | 16D | 16D |
| 环境输出 | 16D | 18D (增加ignore_collision) |
| 位置归一化 | Z-score for all joints | Z-score for xyz only |
| 四元数处理 | 归一化 | 保持原值（推理时单位化） |
| 夹爪处理 | 连续值 | 离散化 (0/1) |

### Replay Buffer 字段变化

| 旧版本 | 新版本 |
|-------|-------|
| `right_prev_joint_positions` | `right_prev_gripper_pose` |
| `right_prev_gripper_joint_positions` | `right_prev_gripper_open` |
| `left_prev_joint_positions` | `left_prev_gripper_pose` |
| `left_prev_gripper_joint_positions` | `left_prev_gripper_open` |
| `right_next_joint_positions` | `right_next_gripper_pose` |
| `right_next_gripper_joint_positions` | `right_next_gripper_open` |
| `left_next_joint_positions` | `left_next_gripper_pose` |
| `left_next_gripper_joint_positions` | `left_next_gripper_open` |

### 统计信息变化

| 旧版本 | 新版本 |
|-------|-------|
| `right_joints_mean/std` (7D) | `right_pos_mean/std` (3D) |
| `left_joints_mean/std` (7D) | `left_pos_mean/std` (3D) |
| `right_gripper_mean/std` (1D) | `right_gripper_open_mean/std` (1D) |
| `left_gripper_mean/std` (1D) | `left_gripper_open_mean/std` (1D) |

---

## 16. 总结

`ActBCVisionAgent` 是一个设计良好的双臂机器人控制代理，基于行为克隆（Behavior Cloning）和纯视觉输入。当前版本使用**末端执行器位姿控制**，这是与早期版本的主要区别。

### 核心特性

1. **末端执行器控制**
   - 使用任务空间位姿（xyz + quaternion）而非关节空间
   - 更直观的动作表示
   - 夹爪状态离散化为0/1

2. **纯视觉控制**
   - 移除语言条件（NO LANGUAGE）
   - 专注于视觉-运动映射
   - 支持多相机RGB和点云输入

3. **双臂协调控制**
   - 同时控制左右两个机械臂（16D内部表示，18D输出）
   - 独立归一化每个手臂的统计信息
   - 分别监控每个手臂和组件的损失

4. **智能归一化策略**
   - 位置(xyz): Z-score归一化
   - 四元数: 保持原值（推理时单位化）
   - 夹爪: 保持原值（推理时离散化）

5. **时序聚合机制**
   - 使用指数加权平均多个历史预测
   - 提高动作序列的平滑性和一致性
   - 减少高频抖动

6. **高效实现**
   - 使用 `@lru_cache` 缓存统计信息
   - 维度自适应处理（支持2D和3D输入）
   - 模块化的预处理和后处理

### 适用场景

该代码特别适合以下场景：
- 双臂机器人的模仿学习任务
- 需要精确末端执行器控制的操作
- 纯视觉条件（不需要语言指令）的任务
- 需要平滑动作轨迹的应用

### 与act_bc_lang的主要区别

| 特性 | act_bc_vision | act_bc_lang |
|-----|--------------|-------------|
| 语言输入 | ❌ 无 | ✅ 使用CLIP编码 |
| 视觉输入 | ✅ RGB + 点云 | ✅ RGB + 点云 |
| 时序聚合 | ✅ 有 | ✅ 有 |
| 归一化方式 | Z-score (pos only) | Z-score (pos only) |
| 文件名 | bc_actor.pt | bc_actor.pt |
| 处理逻辑 | 一致 | 一致 |

该实现为末端执行器控制的纯视觉机器人学习提供了一个高质量的基准实现。

---

## 7. ACT Policy 与整体模型架构

### 7.1 ACTPolicy类的角色 (act_policy.py)

**文件路径**: `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/act_policy.py`

#### 架构层次关系

```
ActBCVisionAgent (Agent接口层)
    │
    ├─ 数据预处理 (preprocess_qpos, preprocess_action, preprocess_images)
    ├─ 训练统计 (train_stats)
    │
    └─ ACTPolicy (策略包装层) ← 你问的这个
        │
        └─ DETRVAE (核心模型层)
            │
            ├─ Backbone (视觉特征提取)
            ├─ CVAE Encoder (动作序列编码)
            └─ Transformer Decoder (CVAE解码器)
```

#### ACTPolicy的职责

**ACTPolicy不是actor模型本身**，而是一个**策略包装器**（Policy Wrapper），它的职责包括：

```python
class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = build_ACT_model_and_optimizer(args)[0]  # 包含DETRVAE模型
        self.optimizer = build_ACT_model_and_optimizer(args)[1]
        self.kl_weight = args.kl_weight

    def forward(self, qpos, image, actions=None, is_pad=None):
        # 训练时：调用模型，计算损失
        # 推理时：调用模型，返回动作预测
```

**1. 模型管理**
- 持有真正的actor模型（DETRVAE）的引用
- 管理优化器（optimizer）
- 配置KL散度权重

**2. 损失函数计算**（act_policy.py:18-80）

训练时，ACTPolicy负责：
```python
# (1) 调用DETRVAE模型
a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)

# (2) 计算KL散度（CVAE的正则化项）
total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

# (3) 计算加权L1损失
# 右臂损失
right_pos_l1 = (F.l1_loss(...) * ~is_pad.unsqueeze(-1)).mean() * 3.0  # 位置权重3x
right_quat_l1 = (F.l1_loss(...) * ~is_pad.unsqueeze(-1)).mean() * 1.0  # 四元数权重1x
right_gripper_l1 = (F.l1_loss(...) * ~is_pad).mean() * 3.0  # 夹爪权重3x

# 左臂损失（同样的权重）
left_l1 = left_pos_l1 + left_quat_l1 + left_gripper_l1

# (4) 总损失
total_loss = (right_l1 + left_l1) + total_kld * self.kl_weight
```

**关键设计**：
- **加权损失**：位置和夹爪比四元数更重要（3:1:3）
- **Padding mask**：忽略填充位置的损失（`~is_pad`）
- **KL权重**：控制CVAE的探索-利用平衡（默认10）

**3. 推理接口**（act_policy.py:81-83）

推理时，ACTPolicy简单转发：
```python
else:  # inference time
    a_hat, _, (_, _) = self.model(qpos, image, env_state)
    return a_hat  # 返回预测的动作序列
```

---

### 7.2 DETRVAE: 核心Actor模型

**文件路径**: `/home/hdliu/occ_grasp_fall/occ_grasp_models/agents/act_bc_vision/detr/models/detr_vae.py`

DETRVAE是**真正的actor模型**，基于**DETR (DEtection TRansformer)** 架构，结合了**CVAE (Conditional Variational Autoencoder)**。

#### 7.2.1 DETR架构简介

DETR最初是Facebook AI为目标检测设计的Transformer架构：
- **原始用途**：将目标检测视为集合预测问题
- **核心思想**：使用Transformer直接预测一组对象，无需NMS
- **关键组件**：CNN backbone + Transformer encoder-decoder + 可学习的object queries

**在ACT中的改编**：
- 将"对象检测"替换为"动作预测"
- Object queries → Action queries（预测未来动作序列）
- Bounding boxes → Robot actions（末端执行器位姿）

#### 7.2.2 DETRVAE的完整架构

```
输入：
  - qpos: 当前机器人状态 (B, 16)
  - images: 多视角RGB图像 (B, 6, 3, 256, 256)
  - actions: 未来动作序列 (B, 20, 16) [训练时]

┌─────────────────────────────────────────────────────────────┐
│                    CVAE Encoder（训练时）                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │ actions (B, 20, 16)                                │     │
│  │    ↓ Linear projection                             │     │
│  │ action_embed (B, 20, 512)                          │     │
│  │    ↓ concat with CLS token                         │     │
│  │ [CLS] + action_embed (B, 21, 512)                  │     │
│  │    ↓ Transformer Encoder (4 layers)                │     │
│  │ encoder_output[0] (B, 512) ← 取CLS token           │     │
│  │    ↓ Linear projection                             │     │
│  │ mu, logvar (B, 32), (B, 32)                        │     │
│  │    ↓ Reparameterization Trick                      │     │
│  │ latent_sample = mu + std * eps (B, 32)             │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  推理时：latent_sample = zeros(B, 32) ← 从先验采样          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Vision Backbone (ResNet18)                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 6个相机，每个单独处理：                             │     │
│  │ image[i] (B, 3, 256, 256)                          │     │
│  │    ↓ ResNet18 (预训练ImageNet)                     │     │
│  │ features[i] (B, 512, H, W)                         │     │
│  │    ↓ Conv1x1 projection                            │     │
│  │ cam_features[i] (B, 512, H, W)                     │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  所有相机特征拼接到宽度维度：                                │
│  src = concat(cam_features, axis=W) → (B, 512, H, 6*W)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Proprioception & Latent Embedding               │
│  ┌────────────────────────────────────────────────────┐     │
│  │ qpos (B, 16)                                       │     │
│  │    ↓ Linear projection                             │     │
│  │ proprio_input (B, 512)                             │     │
│  │                                                     │     │
│  │ latent_sample (B, 32)                              │     │
│  │    ↓ Linear projection                             │     │
│  │ latent_input (B, 512)                              │     │
│  │                                                     │     │
│  │ 拼接到src的开头：                                   │     │
│  │ src = [latent_input, proprio_input, vision_feats]  │     │
│  │       (序列长度 = 2 + H*6*W)                        │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│            Transformer Decoder (CVAE Decoder)                │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 初始化：                                            │     │
│  │ tgt = zeros(20, B, 512) ← 20个action queries       │     │
│  │ query_embed = learned_embed(20, 512) ← 可学习位置   │     │
│  │                                                     │     │
│  │ Transformer Encoder (无修改):                       │     │
│  │ memory = Encoder(src + pos_embed)                  │     │
│  │        ↓                                            │     │
│  │ Transformer Decoder (7 layers):                    │     │
│  │   - Self-attention on queries                      │     │
│  │   - Cross-attention to memory (vision+proprio+latent) │  │
│  │   - FFN                                             │     │
│  │        ↓                                            │     │
│  │ hs = decoder_output (20, B, 512)                   │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Output Heads                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │ hs (20, B, 512)                                    │     │
│  │    ↓ Linear(512 → 16)                              │     │
│  │ a_hat (B, 20, 16) ← 预测的动作序列                 │     │
│  │    ↓ Linear(512 → 1)                               │     │
│  │ is_pad_hat (B, 20, 1) ← 预测的padding mask         │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘

输出：
  - a_hat: (B, 20, 16) - 预测的20步动作序列
  - is_pad_hat: (B, 20, 1) - padding mask预测
  - mu, logvar: (B, 32), (B, 32) - 潜变量分布参数
```

#### 7.2.3 关键组件详解

**1. CVAE Encoder（detr_vae.py:85-105）**

```python
# 编码action sequence到latent space
action_embed = self.encoder_proj(actions)  # (B, 20, 16) → (B, 20, 512)
cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # (B, 1, 512)
encoder_input = torch.cat([cls_embed, action_embed], axis=1)  # (B, 21, 512)

# Transformer Encoder (4 layers)
encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
encoder_output = encoder_output[0]  # 只取CLS token的输出

# 投影到mu和logvar
latent_info = self.latent_proj(encoder_output)  # (B, 512) → (B, 64)
mu = latent_info[:, :32]
logvar = latent_info[:, 32:]

# Reparameterization trick
latent_sample = mu + std * eps  # std = exp(logvar/2), eps ~ N(0,1)
```

**设计亮点**：
- **CLS token技巧**：类似BERT，用一个特殊token聚合整个序列的信息
- **只编码actions**：不将conditions（images, qpos）输入encoder，符合CVAE定义
- **小潜变量维度**：32维（vs 行为序列320维），强制学习压缩表示

**2. Vision Backbone（detr_vae.py:111-120）**

```python
# 每个相机独立处理
for cam_id, cam_name in enumerate(self.camera_names):
    features, pos = self.backbones[cam_id](image[:, cam_id])
    # features: (B, 512, H, W) - ResNet18 C4特征
    # pos: (B, 512, H, W) - 正弦位置编码
    all_cam_features.append(self.input_proj(features))
    all_cam_pos.append(pos)

# 拼接所有相机特征
src = torch.cat(all_cam_features, axis=3)  # 沿宽度拼接
pos = torch.cat(all_cam_pos, axis=3)
```

**设计亮点**：
- **独立backbone**：每个相机独立处理，不共享权重
- **ResNet18 预训练**：利用ImageNet预训练权重
- **空间拼接**：保持空间结构，不flatten到序列

**3. Transformer Decoder（detr_vae.py:126, transformer.py:46-74）**

```python
# 输入准备
proprio_input = self.input_proj_robot_state(qpos)  # (B, 16) → (B, 512)
latent_input = self.latent_out_proj(latent_sample)  # (B, 32) → (B, 512)

# 拼接到vision features前面
addition_input = torch.stack([latent_input, proprio_input], axis=0)  # (2, B, 512)
src = torch.cat([addition_input, src], axis=0)  # (2+H*6W, B, 512)

# Transformer forward
tgt = torch.zeros_like(query_embed)  # (20, B, 512) - 初始化为0
memory = self.encoder(src, pos=pos_embed)  # 编码memory
hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)
```

**设计亮点**：
- **可学习queries**：20个action queries学习预测未来动作
- **条件注入**：latent和proprio作为memory的一部分，通过cross-attention条件化
- **零初始化tgt**：类似DETR，让decoder从0开始解码

**4. 输出Head（detr_vae.py:132-134）**

```python
a_hat = self.action_head(hs)  # (20, B, 512) → (B, 20, 16)
is_pad_hat = self.is_pad_head(hs)  # (20, B, 512) → (B, 20, 1)
```

---

### 7.3 CVAE在ACT中的作用

#### 什么是CVAE？

**Conditional Variational Autoencoder (条件变分自编码器)** 是VAE的条件版本：

```
标准VAE:
  x → Encoder → (μ, σ) → z ~ N(μ, σ) → Decoder → x̂

Conditional VAE:
  (x, c) → Encoder(x) → (μ, σ) → z ~ N(μ, σ) → Decoder(z, c) → x̂
  
  关键：condition c 只在decoder中使用，encoder只看x
```

#### ACT中的CVAE映射

在ACT中：
- **x（数据）**: 未来动作序列 `actions` (B, 20, 16)
- **c（条件）**: 图像 `images` + 当前状态 `qpos`
- **z（潜变量）**: 32维向量，编码动作的多样性

```
训练时:
  actions → CVAE Encoder → (μ, σ) → z ~ N(μ, σ)
          ↓
  (z, images, qpos) → Transformer Decoder → predicted_actions
  
  Loss = L1(predicted_actions, actions) + β * KL(q(z|actions) || p(z))

推理时:
  z ~ N(0, I)  ← 从标准正态分布采样
  (z, images, qpos) → Transformer Decoder → predicted_actions
```

#### 为什么需要CVAE？

**问题**：行为克隆中的多模态问题
- 同一个场景（images, qpos）可能对应多种合理的动作
- 例如：抓杯子可以从左边或右边接近
- 标准MSE/L1损失会平均多个模式 → 预测卡在中间 → 失败

**CVAE的解决方案**：
- 潜变量 z 编码动作的"风格"或"模式"
- 训练时学习从actions推断z（哪种模式）
- 推理时从先验采样z（选择一种模式）
- KL散度防止z坍缩（确保z有意义）

#### ACT的特殊设计

**1. 推理时z=0（而非随机采样）**

```python
# 标准CVAE推理
z = torch.randn(batch_size, latent_dim)  # 随机采样

# ACT推理 (detr_vae.py:108-109)
z = torch.zeros(batch_size, latent_dim)  # 固定为0
```

**为什么？**
- ACT原论文发现：随机采样z会导致动作不稳定
- 使用z=0（mean模式）更稳定、可复现
- CVAE仍然有用：训练时的KL正则化帮助学习更好的表示

**2. 小KL权重 (β=10)**

```python
loss = l1_loss + kl_weight * kl_divergence  # kl_weight = 10
```

标准β-VAE使用β=1，ACT使用β=10意味着：
- 更强的KL正则化
- 鼓励z接近先验N(0,1)
- 减少对潜变量的依赖，更依赖条件输入

**3. Encoder只看actions**

这是CVAE的标准做法，但很多实现会错误地将condition也输入encoder。ACT正确实现：
```python
# 正确 (ACT)
encoder_input = actions  # 只有actions
latent = Encoder(encoder_input)

# 错误 (常见错误)
encoder_input = concat([actions, images, qpos])  # 不应包含condition
latent = Encoder(encoder_input)
```

---

### 7.4 训练与推理流程对比

#### 训练流程

```python
# 1. 数据准备
qpos = preprocess_qpos(observation)        # (B, 16)
action_seq = preprocess_action(demo)       # (B, 20, 16)
images = preprocess_images(demo)           # (B, 6, 3, 256, 256)
is_pad = demo['is_pad']                    # (B, 20)

# 2. Forward pass
a_hat, is_pad_hat, (mu, logvar) = model(qpos, images, action_seq, is_pad)

# 3. Loss计算 (在ACTPolicy中)
kl_loss = kl_divergence(mu, logvar)
l1_loss = weighted_l1(a_hat, action_seq, ~is_pad)
total_loss = l1_loss + 10 * kl_loss

# 4. Backward
total_loss.backward()
optimizer.step()
```

#### 推理流程

```python
# 1. 数据准备
qpos = preprocess_qpos(observation)        # (B, 16)
images = preprocess_images(observation)    # (B, 6, 3, 256, 256)

# 2. Forward pass (无actions输入)
a_hat = model(qpos, images)  # z自动设为0
# 返回: (B, 20, 16) - 预测的20步动作

# 3. Temporal ensemble (在Agent中)
all_time_actions[t, t:t+20] = a_hat
actions_for_curr_step = all_time_actions[:, t]
exp_weights = exp(-0.01 * arange(len(actions_for_curr_step)))
action = (actions_for_curr_step * exp_weights).sum(axis=0)

# 4. 后处理 (在Agent中)
# - 位置反归一化
# - 四元数归一化
# - 夹爪离散化
# - 添加ignore_collisions
final_action = postprocess(action)  # (18,)
```

---

### 7.5 总结：各组件的职责

| 组件 | 类型 | 文件 | 主要职责 |
|-----|------|------|---------|
| **ActBCVisionAgent** | Agent接口 | act_bc_vision_agent.py | 数据预处理、训练统计、推理后处理、时序聚合 |
| **ACTPolicy** | 策略包装器 | act_policy.py | 损失函数计算、优化器管理、训练/推理分支 |
| **DETRVAE** | 核心模型 | detr/models/detr_vae.py | CVAE架构、Transformer forward |
| **CVAE Encoder** | 编码器 | detr/models/detr_vae.py | 将actions编码为潜变量z |
| **Vision Backbone** | 视觉特征提取 | detr/models/backbone.py | 多视角图像→特征图 |
| **Transformer Decoder** | 解码器 | detr/models/transformer.py | 从z+images+qpos解码动作序列 |

#### 信息流向

```
训练时:
  Demo data
    ↓ preprocess
  (qpos, images, actions, is_pad)
    ↓ ActBCVisionAgent.update()
  ACTPolicy.forward(training=True)
    ↓
  DETRVAE:
    actions → Encoder → z
    (z, images, qpos) → Transformer Decoder → a_hat
    ↓
  ACTPolicy:
    loss = L1(a_hat, actions) + β*KL(z)
    ↓
  Optimizer.step()

推理时:
  Observation
    ↓ preprocess
  (qpos, images)
    ↓ ActBCVisionAgent.act()
  ACTPolicy.forward(training=False)
    ↓
  DETRVAE:
    z = 0
    (z, images, qpos) → Transformer Decoder → a_hat
    ↓
  ActBCVisionAgent:
    temporal_ensemble(a_hat) → action
    postprocess(action) → final_action (18D)
    ↓
  RLBench环境执行
```

#### 关键设计原则

1. **关注点分离**
   - Agent: 数据处理和环境交互
   - Policy: 损失函数和训练逻辑
   - Model: 神经网络forward

2. **CVAE的正确实现**
   - Encoder只看actions
   - Decoder看z+conditions
   - 推理时z=0（稳定性）

3. **多模态处理**
   - Vision: 6个独立ResNet18
   - Proprio: 线性投影
   - Latent: 编码动作多样性

4. **时序建模**
   - Transformer: 并行预测20步
   - Temporal ensemble: 平滑历史预测
   - Action queries: 可学习的时序模式

该架构成功将DETR的集合预测思想应用到机器人控制，通过CVAE处理多模态，通过Transformer利用视觉-运动的长程依赖关系。

