# PPI 数据存储结构和 ReplayBuffer 内存格式详细解释

## 目录
1. [概述与数据流程](#1-概述与数据流程)
2. [核心概念：state 与 action](#2-核心概念state-与-action)
3. [原始数据存储结构 (training_raw)](#3-原始数据存储结构-training_raw)
4. [预处理数据结构 (training_processed)](#4-预处理数据结构-training_processed)
5. [ReplayBuffer 内存格式](#5-replaybuffer-内存格式)
6. [归一化机制](#6-归一化机制)
7. [完整数据流转示例](#7-完整数据流转示例)
8. [总结](#8-总结)

---

## 1. 概述与数据流程

PPI 的数据处理分为四个主要阶段：

```
演示采集 → 原始数据存储 → 特征预处理 → 训练时加载
```

**阶段一：演示采集**
- 通过 RLBench2 环境采集双臂机器人演示
- 每个 episode 包含一次完整的任务执行过程

**阶段二：原始数据存储 (training_raw)**
- 保存 RGB/深度/掩码图像到 PNG 文件
- 保存低维观测数据到 `low_dim_obs.pkl`
- 保存语言描述到 `variation_descriptions.pkl`

**阶段三：特征预处理 (training_processed)**
- 从深度图生成点云数据
- 提取 DINO 视觉特征
- 计算点流（同一批物体表面点在各时刻的世界坐标）
- 计算归一化统计参数

**阶段四：训练时加载**
- 构建 ReplayBuffer 内存结构
- 通过 Sampler 采样训练序列
- 动态加载大型特征数据

---

## 2. 核心概念：state 与 action

在理解数据结构之前，需要先明确 state 和 action 这两个核心概念。

### 2.1 概念定义

| 维度 | state | action |
|------|-------|--------|
| **定义** | 当前时刻的机器人状态 | 训练目标状态（由预测模式与采样器共同决定） |
| **时间** | 时间步 t | 时间步 t+k（k 取决于预测模式） |
| **用途** | 模型输入（观察） | 模型输出（预测目标） |
| **训练角色** | 条件 | 监督信号 |

### 2.2 数据维度

**单个臂的状态维度（8 维）**：
```
gripper_pose:  7 维 [x, y, z, qw, qx, qy, qz]
gripper_open:  1 维 [0=闭合, 1=打开]
```

**双臂总计（16 维）**：
```
索引 0-2:   左臂夹爪位置 (x, y, z) [米]
索引 3-6:   左臂夹爪旋转 (qw, qx, qy, qz) [四元数]
索引 7:     左臂夹爪打开度 [0=闭合, 1=打开]
索引 8-10:  右臂夹爪位置 (x, y, z) [米]
索引 11-14: 右臂夹爪旋转 (qw, qx, qy, qz) [四元数]
索引 15:    右臂夹爪打开度 [0=闭合, 1=打开]
```

代码参考 [occ_grasp_models/ppi/common/get_data_keyframe.py:94](occ_grasp_models/ppi/common/get_data_keyframe.py#L94)：
```python
current_state = np.concatenate([gripper_pose, gripper_state])  # (7 + 1) * 2 = 16
```

### 2.3 不同预测模式下的 action 含义

#### continuous 模式
```python
# GetDataContinuous 实际写入（代码核验）
# 时间线: 0---1---2---3---4---5---...---99
# state:  s0  s1  s2  s3  s4  s5  ...  s99
# action: s0  s1  s2  s3  s4  s5  ...  s99
#
# 未来监督来自 sampler 组装的序列索引，而不是 GetData 阶段把 action 写成 next_state
```
- `GetDataContinuous` 中 action 与 state 相同（当前帧16D姿态）
- “预测未来”的语义由 `SequenceSamplerContinuous` 的序列窗口提供
- 适合密集控制任务
- 预测 horizon 通常较短（50步）

#### keyframe 模式
```python
# 时间线: 0---1---2---3---4---5---6---7---8---9---10--11--12---...
# 关键帧:                      ↑                       ↑
#                             k1=5                   k2=12

# state[0-4]    → action[0-4]    = state[5]  (指向关键帧 k1)
# state[5-11]   → action[5-11]   = state[12] (指向关键帧 k2)
```
- action = 下一个关键帧的 state
- 中间步骤由运动规划器插值
- 适合长期规划任务
- 预测 horizon 较长（4-10 个关键帧）

#### keyframe_continuous 混合模式
```python
# GetDataKeyframeContinuous 实际写入
# state[0:54] 与 action[0:54] 均为当前帧姿态
#
# 训练时由 sampler 组织为:
# [continuous 段索引..., keyframe 段索引...]
# 后段 keyframe 索引承担“远期目标”语义
```
- 结合短期精确控制和长期规划
- 适合复杂的双臂操作任务

### 2.4 代码实现对比

**continuous 模式** ([occ_grasp_models/ppi/common/get_data_continuous.py](occ_grasp_models/ppi/common/get_data_continuous.py))：
```python
for i in range(len(low_dim_obs)):
    gripper_pose = np.concatenate([low_dim_obs[i].left.gripper_pose,
                                   low_dim_obs[i].right.gripper_pose])
    gripper_state = np.array([low_dim_obs[i].left.gripper_open,
                              low_dim_obs[i].right.gripper_open])
    current_state = np.concatenate([gripper_pose, gripper_state])
    current_action = np.concatenate([gripper_pose, gripper_state])
    state.append(current_state)
    action.append(current_action)  # 与 state 相同
```

**keyframe 模式** ([occ_grasp_models/ppi/common/get_data_keyframe.py:76](occ_grasp_models/ppi/common/get_data_keyframe.py#L76))：
```python
keyframe_id = episode_keypoints[0]  # 下一个关键帧索引

for i in range(len(low_dim_obs)):
    current_state = get_gripper_state(low_dim_obs[i])
    keyframe_state = get_gripper_state(low_dim_obs[keyframe_id])

    if i >= keyframe_id and len(episode_keypoints) > 1:
        episode_keypoints.pop(0)  # 到达关键帧，切换目标
        keyframe_id = episode_keypoints[0]

    state.append(current_state)
    action.append(keyframe_state)  # 下一个关键帧
```

---

## 3. 原始数据存储结构 (training_raw)

### 3.1 目录结构

```
occ_grasp_models/data/training_raw/
└── bimanual_push_box/                          # 任务名称
    └── all_variations/                         # 所有变体混合采样
        └── episodes/                           # 演示集合目录
            ├── episode0/                       # 第0个演示
            │   ├── over_shoulder_left_rgb/     # 左肩摄像头RGB图像序列
            │   │   ├── rgb_0000.png           # 第0帧RGB (256×256×3, uint8)
            │   │   ├── rgb_0001.png
            │   │   └── ... (假设100帧)
            │   │
            │   ├── over_shoulder_left_depth/   # 左肩摄像头深度图序列
            │   │   ├── depth_0000.png         # 第0帧深度 (256×256, 0-1归一化)
            │   │   └── ...
            │   │
            │   ├── over_shoulder_left_mask/    # 左肩摄像头语义掩码序列
            │   │   ├── mask_0000.png          # 第0帧掩码 (256×256, 类别ID)
            │   │   └── ...
            │   │
            │   ├── over_shoulder_right_rgb/    # 右肩摄像头 (结构同上)
            │   ├── over_shoulder_right_depth/
            │   ├── over_shoulder_right_mask/
            │   │
            │   ├── overhead_rgb/               # 顶部摄像头
            │   ├── overhead_depth/
            │   ├── overhead_mask/
            │   │
            │   ├── wrist_left_rgb/             # 左手腕摄像头
            │   ├── wrist_left_depth/
            │   ├── wrist_left_mask/
            │   │
            │   ├── wrist_right_rgb/            # 右手腕摄像头
            │   ├── wrist_right_depth/
            │   ├── wrist_right_mask/
            │   │
            │   ├── front_rgb/                  # 前置摄像头
            │   ├── front_depth/
            │   ├── front_mask/
            │   │
            │   ├── low_dim_obs.pkl             # 【核心】低维观察序列
            │   ├── variation_descriptions.pkl  # 语言描述
            │   └── variation_number.pkl        # 变体编号
            │
            ├── episode1/                       # 第1个演示 (结构同上)
            └── ...
```

### 3.2 low_dim_obs.pkl 详解

这是每个 episode 最核心的数据文件，包含完整的低维机器人状态序列。

**文件结构**：
```python
low_dim_obs: List[BimanualObservation]
# 长度: T (episode的总时间步数，如100)
```

#### BimanualObservation 类定义

根据 [repos/RLBench/rlbench/backend/observation.py:84](repos/RLBench/rlbench/backend/observation.py#L84)：

```python
@dataclass
class BimanualObservation(Observation):
    # 继承自 Observation 的字段
    perception_data: Dict[str, np.ndarray]     # 视觉数据（已被清空保存到PNG）
    task_low_dim_state: np.ndarray             # 任务相关的低维状态（如对象位置）
    misc: Dict[str, Any]                       # 其他元数据

    # BimanualObservation 特有字段
    right: UnimanualObservationData            # 右臂观察
    left: UnimanualObservationData             # 左臂观察

    # 额外添加的字段（在PPI中）
    object_6d_pose: Dict[str, np.ndarray]      # 对象6D位姿
        # 常见键: 'position', 'quaternion'
        # 点流生成脚本还会使用: 'matrix' (4,4) 齐次变换
```

#### UnimanualObservationData 结构

根据 [repos/RLBench/rlbench/backend/observation.py:47](repos/RLBench/rlbench/backend/observation.py#L47)：

```python
@dataclass
class UnimanualObservationData:
    # 关节信息 (Franka Panda 有 7 个关节)
    joint_velocities: np.ndarray               # (7,) 关节速度 [rad/s]
    joint_positions: np.ndarray                # (7,) 关节位置 [rad]
    joint_forces: np.ndarray                   # (7,) 关节力矩 [Nm]

    # 夹爪姿态
    gripper_pose: np.ndarray                   # (7,) 夹爪位姿 [x,y,z, qw,qx,qy,qz]
                                               # 前3维: 位置 [m]
                                               # 后4维: 四元数旋转 (wxyz格式)
    gripper_matrix: np.ndarray                 # (4, 4) 夹爪变换矩阵 (齐次变换)

    # 夹爪状态
    gripper_open: float                        # 夹爪打开度 [0=闭合, 1=打开]
    gripper_joint_positions: np.ndarray        # (2,) 夹爪手指关节位置 [m]
    gripper_touch_forces: np.ndarray           # (2,) 夹爪触觉力 [N]

    # 碰撞检测
    ignore_collisions: np.ndarray              # 忽略碰撞的对象列表
```

**单个臂的完整维度**：
```
joint_velocities:        7
joint_positions:         7
joint_forces:            7
gripper_pose:            7
gripper_matrix:          16  (4×4 展平)
gripper_open:            1
gripper_joint_positions: 2
gripper_touch_forces:    2
─────────────────────────
总计:                   49 维
```

**注意**：虽然单臂理论上有 49 维数据，但在 PPI 的实际使用中，只提取了关键的 8 维（gripper_pose 7维 + gripper_open 1维）。

#### 使用示例

```python
import pickle

# 加载文件
with open('episode0/low_dim_obs.pkl', 'rb') as f:
    low_dim_obs = pickle.load(f)

# 结构
print(type(low_dim_obs))  # <class 'list'>
print(len(low_dim_obs))   # 100 (假设episode有100帧)

# 查看第一帧
obs_0 = low_dim_obs[0]
print(type(obs_0))        # <class 'rlbench.backend.observation.BimanualObservation'>

# 左臂信息
print(obs_0.left.gripper_pose)        # [x, y, z, qw, qx, qy, qz]
print(obs_0.left.gripper_open)        # 0.8 (示例值)
print(obs_0.left.joint_velocities)    # [v1, v2, ..., v7]

# 右臂信息
print(obs_0.right.gripper_pose)       # [x, y, z, qw, qx, qy, qz]
print(obs_0.right.gripper_open)       # 0.2 (示例值)

# 对象位姿
print(obs_0.object_6d_pose['position'])    # [x, y, z]
print(obs_0.object_6d_pose['quaternion'])  # [qw, qx, qy, qz]

# 视觉数据已被清空（保存到PNG文件中）
print(obs_0.perception_data)          # {} (空字典)
```

### 3.3 其他文件说明

**variation_descriptions.pkl**：
```python
# Python List，包含任务的语言描述
# 示例: ["push the box to the target"]
```

**variation_number.pkl**：
```python
# int 类型，表示变体编号
# 示例: 3
```

---

## 4. 预处理数据结构 (training_processed)

预处理阶段将原始数据转换为适合训练的特征格式。

### 4.1 目录结构

```
occ_grasp_models/data/training_processed/
├── point_cloud/                                # 点云数据目录
│   └── bimanual_push_box/all_variations/episodes/
│       ├── episode0/
│       │   └── rgb_pcd_rps6144/                # 点云类型（示例：rgb彩色+6144点采样）
│       │       ├── step000.npy                 # 第0帧点云
│       │       │   # shape: (6144, 6)
│       │       │   # dtype: float64 (样例数据)
│       │       │   # 含义: [:, 0:3] = XYZ坐标, [:, 3:6] = RGB颜色
│       │       ├── step001.npy
│       │       └── ... (100个文件)
│       └── episode1/
│           └── ...
│
├── dino_feature/                               # DINO视觉特征目录
│   └── bimanual_push_box/all_variations/episodes/
│       ├── episode0/
│       │   └── rgb_pcd_rps6144/
│       │       ├── step000.npy                 # 第0帧DINO特征
│       │       │   # shape: (6144, 384)
│       │       │   # dtype: float32
│       │       │   # 含义: 每个点的 DINOv2 语义特征
│       │       └── ...
│       └── episode1/
│           └── ...
│
├── point_flow/                                 # 点流目录
│   └── bimanual_push_box/all_variations/episodes/
│       ├── episode0/
│       │   └── world_ordered_rps200/           # 常见点流类型（200点采样）
│       │       ├── step000.npy                 # 第0帧点流
│       │       │   # shape: (200, 3)
│       │       │   # dtype: float64 (样例数据)
│       │       │   # 含义: 该时刻同一组点的世界坐标
│       │       └── ...
│       └── episode1/
│           └── ...
│
├── norm_stats/                                 # 归一化统计目录
│   └── norm_stats_bimanual_pick_laptop_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth
│       # PyTorch 状态字典文件
│       # 包含: action, agent_pos, point_cloud, lang 等的归一化参数
│
└── instruction_embeddings.pkl                  # 语言嵌入字典
    # Python Dict
    # 键: 语言描述字符串
    # 值: 预计算语言嵌入向量 (1024,)
```

> 代码核验注记：训练配置中的 `point_flow_type`（常见写法 `rps200`）必须与磁盘目录名一致；若数据实际生成为 `world_ordered_rps200`，配置也应同步修改。

### 4.2 存储开销统计

以 100 个 episode 为例：

**training_raw**:
| 数据类型 | 计算 | 大小 |
|---------|------|------|
| RGB | 6摄像头 × 100 episodes × 100帧 × 256×256×3 | ~11GB |
| 深度+掩码 | 6摄像头 × 100 episodes × 100帧 × 256×256 | ~7GB |
| low_dim_obs | 100 episodes × ~10KB | ~1MB |
| **小计** | | ~18GB |

**training_processed**:
| 数据类型 | 计算 | 大小 |
|---------|------|------|
| 点云（`rgb_pcd_rps6144`，float64） | 100 episodes × 100帧 × 6144×6×8bytes | ~2.95GB |
| DINO（float32） | 100 episodes × 100帧 × 6144×384×4bytes | ~94.4GB |
| 点流（`world_ordered_rps200`，float64） | 100 episodes × 100帧 × 200×3×8bytes | ~48MB |
| **小计** | | ~97.4GB |

**总计**: ~115.4GB / 100 episodes

> 注：以上是按 `rgb_pcd_rps6144 + world_ordered_rps200` 的估算；若改用其他点数类型（如 `rps1024`/`rps50`），大小随点数近似线性缩放。

---

## 5. ReplayBuffer 内存格式

ReplayBuffer 是训练时的核心数据结构，将磁盘数据组织为高效的内存格式。

### 5.1 整体结构

```python
replay_buffer = {
    'meta': {...},   # 元数据
    'data': {...}    # 实际数据
}
```

### 5.2 meta 元数据字段

```python
meta = {
    'episode_ends': np.ndarray,        # (num_episodes,) int64
    'keyframe_indices': np.ndarray,    # (total_keyframes,) int64
    'openess_indices': np.ndarray      # (total_openess_events,) int64 (仅 keyframe_continuous 模式)
}
```

#### episode_ends

**定义**：累积时间步索引，标记每个 episode 的结束位置

**示例**：
```python
# 假设有3个episode
episode_ends = np.array([100, 250, 380])

# 解释:
# - episode 0: 时间步 0 到 99   (共100步)
# - episode 1: 时间步 100 到 249 (共150步)
# - episode 2: 时间步 250 到 379 (共130步)
# 总时间步数 N = 380
```

**用途**：
1. 确定 episode 边界，防止跨 episode 采样
2. 计算每个时间步属于哪个 episode
3. 采样时过滤验证集 episode

**代码示例** ([occ_grasp_models/ppi/common/replay_buffer.py:497](occ_grasp_models/ppi/common/replay_buffer.py#L497))：
```python
def get_episode_idxs(self):
    """获取每个时间步对应的episode编号"""
    result = np.zeros((episode_ends[-1],), dtype=np.int64)
    for i in range(len(episode_ends)):
        start = 0 if i == 0 else episode_ends[i-1]
        end = episode_ends[i]
        for idx in range(start, end):
            result[idx] = i  # 时间步idx属于episode i
    return result
```

#### keyframe_indices

**定义**：关键帧在全局时间步中的索引位置

**示例**：
```python
# 接上例，假设每个episode有10个关键帧
keyframe_indices = np.array([
    # episode 0 的关键帧 (10个)
    5, 12, 18, 25, 35, 45, 60, 75, 85, 99,
    # episode 1 的关键帧 (10个)
    105, 115, 130, 145, 160, 180, 200, 220, 235, 249,
    # episode 2 的关键帧 (10个)
    255, 265, 280, 295, 310, 330, 345, 360, 370, 379
])
# 总计: 30个关键帧
```

**关键帧发现算法** ([occ_grasp_models/ppi/common/get_data_keyframe.py:124](occ_grasp_models/ppi/common/get_data_keyframe.py#L124))：

```python
def keypoint_discovery_bimanual(low_dim_obs, episode, stopping_delta=0.1, total_kp=10):
    """
    发现episode中的关键帧

    检测条件:
    1. 夹爪状态变化 (打开→关闭 或 关闭→打开)
    2. 运动停止 (关节速度 < 0.1 rad/s)
    3. episode结束帧

    如果检测到的关键帧 < total_kp，则均匀插值补足
    """
    episode_keypoints = []

    for i, obs in enumerate(low_dim_obs):
        # 检测双臂运动是否停止
        right_stopped = _is_stopped_right(low_dim_obs, i, obs.right, stopping_delta)
        left_stopped = _is_stopped_left(low_dim_obs, i, obs.left, stopping_delta)
        stopped = right_stopped and left_stopped

        # 检测夹爪状态变化
        state_changed = (
            obs.right.gripper_open != low_dim_obs[i-1].right.gripper_open or
            obs.left.gripper_open != low_dim_obs[i-1].left.gripper_open
        )

        # 是否是最后一帧
        last = (i == len(low_dim_obs) - 1)

        # 满足任一条件即为关键帧
        if i != 0 and (state_changed or last or stopped):
            episode_keypoints.append(i)

    # 如果关键帧不足，均匀采样补充
    if len(episode_keypoints) < total_kp:
        remaining = [i for i in range(len(low_dim_obs)) if i not in episode_keypoints]
        extra = np.linspace(0, len(remaining)-1, total_kp - len(episode_keypoints), dtype=int)
        episode_keypoints.extend(remaining[extra])
        episode_keypoints.sort()

    return episode_keypoints
```

**用途**：
1. **关键帧预测模式**：模型只预测到下一个关键帧的动作
2. **混合预测模式**：前 N 步连续预测 + 后 M 步关键帧预测

#### openess_indices

**定义**：夹爪打开/关闭事件发生的时间步索引

**示例**：
```python
# 记录所有夹爪状态变化的时刻
openess_indices = np.array([
    # episode 0
    12,   # 左臂夹爪闭合
    35,   # 右臂夹爪打开
    75,   # 左臂夹爪打开
    # episode 1
    115,  # 右臂夹爪闭合
    180,  # 左臂夹爪闭合
    # episode 2
    265,  # 右臂夹爪打开
    330   # 左臂夹爪闭合
])
```

**生成代码** ([occ_grasp_models/ppi/common/get_data_keyframe_continuous.py:137](occ_grasp_models/ppi/common/get_data_keyframe_continuous.py#L137))：
```python
# 在关键帧发现过程中额外记录
if i != 0 and state_changed:
    openess_keypoints.append(i)
```

**用途**：
- **openess_sampling**：在夹爪打开/关闭附近过采样，因为这些是操作的关键时刻
- 提升模型对抓取/释放动作的学习效果

### 5.3 data 数据字段

```python
data = {
    'state': np.ndarray,               # (N, 16) float32
    'action': np.ndarray,              # (N, 16) float32
    'point_cloud': np.ndarray,         # (N, 2) int64
    'dino_feature': np.ndarray,        # (N, 2) int64
    'lang': np.ndarray,                # (N, 1024) float32
    'object_pose': np.ndarray,         # (N, 7) float32 (仅部分模式)
    'point_flow': np.ndarray,          # (N, 2) int64 (仅 keyframe_continuous 模式)
    'initial_point_flow': np.ndarray   # (N, 2) int64 (仅 keyframe_continuous 模式)
}
```

#### N 的含义

**N**：所有 episode 的总时间步数

```python
N = episode_ends[-1]

# 示例：假设100个episode，每个episode平均120帧
episode_ends = [120, 250, 365, ..., 12050]
N = 12050  # 总时间步数
```

**重要性质**：
- data 中所有数组的第一维度都是 N
- 每个时间步对应一个训练样本（经过采样器处理后）
- N 的大小决定了内存占用

#### state

**维度**：`(N, 16)`

**含义**：当前时刻的机器人状态

**组成** ([occ_grasp_models/ppi/common/get_data_keyframe.py:94](occ_grasp_models/ppi/common/get_data_keyframe.py#L94))：
```python
# 对于时间步 i
state[i] = np.concatenate([
    low_dim_obs[i].left.gripper_pose,   # (7,) [x,y,z, qw,qx,qy,qz]
    [low_dim_obs[i].left.gripper_open], # (1,) [0-1]
    low_dim_obs[i].right.gripper_pose,  # (7,) [x,y,z, qw,qx,qy,qz]
    [low_dim_obs[i].right.gripper_open] # (1,) [0-1]
])
# shape: (16,)
```

**示例值**：
```python
state[0] = np.array([
    # 左臂
    0.35, 0.20, 0.15,           # 左夹爪位置 (米)
    0.707, 0.0, 0.0, 0.707,     # 左夹爪旋转 (四元数，表示90度绕Z轴)
    0.8,                         # 左夹爪打开80%
    # 右臂
    0.45, -0.20, 0.15,          # 右夹爪位置 (米)
    1.0, 0.0, 0.0, 0.0,         # 右夹爪旋转 (四元数，无旋转)
    0.2                          # 右夹爪打开20%
])
```

#### action

**维度**：`(N, 16)`

**含义**：训练目标状态（具体语义取决于预测模式与采样器）

**组成**：与 state 相同的 16 维格式

**不同模式下的 action 含义**：

| 模式 | action 的含义 | 代码位置 |
|------|---------------|----------|
| **continuous** | 当前时间步状态（未来监督由序列采样器提供） | [get_data_continuous.py](occ_grasp_models/ppi/common/get_data_continuous.py) |
| **keyframe** | 下一个关键帧的状态 | [get_data_keyframe.py:95](occ_grasp_models/ppi/common/get_data_keyframe.py#L95) |
| **keyframe_continuous** | 当前时间步的状态（用于连续预测） | [get_data_keyframe_continuous.py:84](occ_grasp_models/ppi/common/get_data_keyframe_continuous.py#L84) |

**keyframe 模式可视化**：
```
时间线: 0----5----10---12---15---18---20---25---...---99
关键帧:      ↑         ↑        ↑        ↑          ↑

state[0-4]  → action[0-4]  都指向关键帧5
state[5-11] → action[5-11] 都指向关键帧12
state[12-17]→ action[12-17]都指向关键帧18
...
```

#### (episode_id, step_id) 索引字段

**维度**：`(N, 2)` 整数索引对

**含义**：指向磁盘上对应的特征文件

**适用字段**：`point_cloud`、`dino_feature`、`point_flow`、`initial_point_flow`

**示例**：
```python
point_cloud[0] = np.array([0, 0])      # episode 0, step 0
point_cloud[1] = np.array([0, 1])      # episode 0, step 1
...
point_cloud[99] = np.array([0, 99])    # episode 0, step 99
point_cloud[100] = np.array([1, 0])    # episode 1, step 0
...
```

**动态加载机制** ([occ_grasp_models/ppi/dataset/rlbench2_dataset.py](occ_grasp_models/ppi/dataset/rlbench2_dataset.py))：
```python
def __getitem__(self, idx):
    # 从ReplayBuffer获取索引序列
    sample = self.sampler.sample_sequence(idx)
    # sample['point_cloud']: (1, Np, 6)  # 仅加载首帧视觉

    # 连续/关键帧 action 由 indices 组织，视觉默认只取第一个索引位
    # (详见 sampler_keyframe_continuous.py 的 indices[0] 读取逻辑)
    return sample
```

**优势**：
1. **内存高效**：不在内存中存储大量点云数据
2. **灵活性**：可以按需更换不同的点云类型
3. **可扩展**：易于添加新的特征类型

#### 其他字段

**lang（语言嵌入）**：
```python
# 维度: (N, 1024)
# 来源: instruction_embeddings.pkl
# 由离线脚本预计算后写入（模型侧按 1024 维读取）

# 示例
lang[0] = embedding_dict["push the box to the target"]
# 同一 episode 的所有时间步使用相同的语言嵌入
```

**object_pose（对象6D位姿）**：
```python
# 维度: (N, 7)
# 组成:
object_pose[i] = np.concatenate([
    low_dim_obs[i].object_6d_pose['position'],    # (3,) xyz
    low_dim_obs[i].object_6d_pose['quaternion']   # (4,) wxyz
])
# 总计: 7维
```

**initial_point_flow（初始点流）**：
```python
# 维度: (N, 2)
# 含义: 指向 episode 的第0帧点流
# 用途: 提供初始参考状态

# 示例
initial_point_flow[0] = [0, 0]    # episode 0, step 0
initial_point_flow[50] = [0, 0]   # episode 0的所有帧都指向step 0
initial_point_flow[100] = [1, 0]  # episode 1的所有帧都指向step 0
```

---

## 6. 归一化机制

归一化将不同量纲和范围的数据归一化到统一范围，提升训练稳定性。

### 6.1 归一化模式

#### limits 模式（Min-Max 归一化）

**公式** ([occ_grasp_models/ppi/model/common/normalizer.py:215](occ_grasp_models/ppi/model/common/normalizer.py#L215))：
```python
# 前向归一化
normalized = (data - input_min) / (input_max - input_min) * (output_max - output_min) + output_min
# 简化形式
normalized = data * scale + offset

# 其中
scale = (output_max - output_min) / (input_max - input_min)
offset = output_min - scale * input_min

# 默认输出范围: [-1, 1]
```

**示例**：
```python
# 假设 action 的位置维度统计
input_min = np.array([0.1, -0.3, 0.05, ...])  # (16,) 每个维度的最小值
input_max = np.array([0.6, 0.4, 0.25, ...])   # (16,) 每个维度的最大值

# 计算归一化参数
scale = (1 - (-1)) / (input_max - input_min)  # (16,)
offset = -1 - scale * input_min                # (16,)

# 应用归一化
action_normalized = action * scale + offset    # 范围变为 [-1, 1]
```

#### gaussian 模式（标准化）

**公式** ([occ_grasp_models/ppi/model/common/normalizer.py:237](occ_grasp_models/ppi/model/common/normalizer.py#L237))：
```python
# 标准化
normalized = (data - mean) / std
# 简化形式
normalized = data * scale + offset

# 其中
scale = 1 / std
offset = -mean / std
```

### 6.2 norm_stats 文件结构

**文件路径**：`occ_grasp_models/data/training_processed/norm_stats/norm_stats_{task}_{pcd_type}_{prediction_type}_{point_flow_type}.pth`

**内容（PyTorch 状态字典）**：
```python
norm_stats = torch.load(
    'norm_stats_bimanual_pick_laptop_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth'
)

# state_dict 为扁平键（示例）
{
    'params_dict.action.scale': Tensor(16,),
    'params_dict.action.offset': Tensor(16,),
    'params_dict.point_cloud.scale': Tensor(6,),      # last_n_dims=1
    'params_dict.dino_feature.scale': Tensor(384,),   # last_n_dims=1
    'params_dict.lang.scale': Tensor(1024,),
    'params_dict.point_flow.scale': Tensor(3,),
    'params_dict.initial_point_flow.scale': Tensor(3,),
    ...
}
```

### 6.3 计算与使用

**计算过程** ([occ_grasp_models/scripts/ppi/data_generation/save_norm_stats.py:94](occ_grasp_models/scripts/ppi/data_generation/save_norm_stats.py#L94))：

```python
# 步骤1: 收集所有数据
gd = GetDataKeyframeContinuous(...)
root = gd.process_episodes(0, 99, skip_ep=[], kp_num=10)
data = root['data']

# 步骤2: 加载所有点云和特征到内存
point_cloud_tensor = torch.zeros(N, 6144, 6, device='cuda:6')
for i, (episode, step) in enumerate(data['point_cloud']):
    pcd_path = f'episode{episode}/{pcd_type}/step{step:03d}.npy'
    point_cloud_tensor[i] = torch.tensor(np.load(pcd_path))

# 步骤3: 构建归一化数据字典
data4norm = {
    'action': torch.tensor(data["action"]),
    'agent_pos': torch.tensor(data["state"]),
    'point_cloud': point_cloud_tensor,
    # ...
}

# 步骤4: 计算归一化统计
normalizer = LinearNormalizer()
normalizer.fit(data=data4norm, last_n_dims=1, mode='limits')
# last_n_dims=1: 对最后1个维度独立归一化
# mode='limits': 使用Min-Max归一化

# 步骤5: 保存
torch.save(normalizer.state_dict(), stats_filepath)
```

**训练时使用**（`dataset` 负责加载，`policy` 内执行归一化）：
```python
class RLBench2Dataset:
    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        return sample

# 归一化在 occ_grasp_models/ppi/policy/ppi.py 中进行
nobs = self.normalizer.normalize(batch['obs'])
nactions = self.normalizer['action'].normalize(batch['action'])
```

**反归一化（推理时）**：
```python
# 模型输出归一化的动作
action_normalized = model(obs)  # (-1, 1)

# 反归一化到真实动作空间
action_real = normalizer.unnormalize({'action': action_normalized})['action']
```

---

## 7. 完整数据流转示例

### 7.1 阶段一：演示采集

```python
# 文件: repos/RLBench/tools/dataset_generator_bimanual.py

# 1个episode的采集过程
task_env = rlbench_env.get_task(BimanualPushBox)
demo, = task_env.get_demos(amount=1, live_demos=True)
# demo: List[BimanualObservation], 长度100

# 保存
save_demo(demo, 'occ_grasp_models/data/training_raw/bimanual_push_box/all_variations/episodes/episode0', variation=3)

# 结果:
# - episode0/over_shoulder_left_rgb/*.png (100张)
# - episode0/low_dim_obs.pkl (包含demo列表)
# - episode0/variation_descriptions.pkl (["push the box to the target"])
# - episode0/variation_number.pkl (3)
```

### 7.2 阶段二：特征预处理

**点云生成** (`occ_grasp_models/scripts/ppi/data_generation/save_ptc.py`)：
```python
for episode in range(100):
    low_dim_obs = pickle.load(f'episode{episode}/low_dim_obs.pkl')

    for step in range(len(low_dim_obs)):
        # 6个相机逐帧反投影，拼接为多视角点云
        multiview_pcd = []
        for camera in ['over_shoulder_left', 'over_shoulder_right', 'overhead', 'front', 'wrist_left', 'wrist_right']:
            depth = load_depth_png_as_meter(episode, step, camera)
            rgb = np.array(Image.open(f'episode{episode}/{camera}_rgb/rgb_{step:04d}.png')).reshape(-1, 3)
            extr = low_dim_obs[step].misc[f'{camera}_camera_extrinsics']
            intr = low_dim_obs[step].misc[f'{camera}_camera_intrinsics']
            xyz = pointcloud_from_depth_and_camera_params(depth, extr, intr).reshape(-1, 3)
            multiview_pcd.append(np.concatenate([xyz, rgb], axis=-1))

        # 按任务工作空间 bounding box 过滤后随机采样（RPS）
        merged = np.concatenate(multiview_pcd, axis=0)
        pcd = RPS_with_bounding_box(merged, num_samples=6144, bounding_box=task_bbox)
        # pcd.shape = (6144, 6)  [x,y,z,r,g,b]

        # 保存
        np.save(f'episode{episode}/rgb_pcd_rps6144/step{step:03d}.npy', pcd)
```

**DINO特征提取** (`occ_grasp_models/scripts/ppi/data_generation/save_dino.py`)：
```python
fusion = Fusion(num_cam=6, feat_backbone='dinov2', device=device)

for episode in range(100):
    low_dim_obs = pickle.load(f'episode{episode}/low_dim_obs.pkl')
    for step in range(len(low_dim_obs)):
        pcd = np.load(f'point_cloud/episode{episode}/rgb_pcd_rps6144/step{step:03d}.npy')[:, :3]
        obs = build_multi_view_obs_from_rgb_depth(episode, step, low_dim_obs)  # 6视角颜色/深度/位姿/内参

        # 每个点提取语义特征
        dino_feature = fusion.extract_semantic_feature_from_ptc(
            torch.tensor(pcd, dtype=torch.float32, device=device), obs
        )  # (6144, 384)

        np.save(f'dino_feature/episode{episode}/rgb_pcd_rps6144/step{step:03d}.npy',
                dino_feature.cpu().numpy())
```

**归一化统计计算** (`occ_grasp_models/scripts/ppi/data_generation/save_norm_stats.py`)：
```python
gd = GetDataKeyframeContinuous(...)
root = gd.process_episodes(0, 99, skip_ep=[], kp_num=10)

# 加载所有action
actions = root['data']['action']  # (N, 16)

# 计算统计
action_min = actions.min(axis=0)   # (16,)
action_max = actions.max(axis=0)   # (16,)
action_mean = actions.mean(axis=0) # (16,)
action_std = actions.std(axis=0)   # (16,)

# 保存
normalizer = LinearNormalizer()
normalizer.fit({'action': actions}, mode='limits')
torch.save(
    normalizer.state_dict(),
    'occ_grasp_models/data/training_processed/norm_stats/norm_stats_*.pth'
)
```

### 7.3 阶段三：构建 ReplayBuffer

```python
# 文件: occ_grasp_models/ppi/common/replay_buffer.py, occ_grasp_models/ppi/common/get_data_keyframe.py

gd = GetDataKeyframe(
    data_path='occ_grasp_models/data/training_raw/bimanual_push_box/all_variations/episodes',
    lang_emb_path='occ_grasp_models/data/training_processed/instruction_embeddings.pkl'
)

# ReplayBuffer构建过程
state_list = []
action_list = []
pcd_index_list = []
episode_ends = []
keyframe_indices = []

for episode in range(100):
    low_dim_obs = pickle.load(f'episode{episode}/low_dim_obs.pkl')

    # 发现关键帧
    episode_kps = keypoint_discovery_bimanual(low_dim_obs, kp_num=10)
    # episode_kps = [5, 12, 18, 25, 35, 45, 60, 75, 85, 99]

    # 处理每一帧
    for i in range(len(low_dim_obs)):
        # 当前状态
        current_state = np.concatenate([
            low_dim_obs[i].left.gripper_pose,
            [low_dim_obs[i].left.gripper_open],
            low_dim_obs[i].right.gripper_pose,
            [low_dim_obs[i].right.gripper_open]
        ])  # (16,)

        # 下一个关键帧
        next_kp_idx = episode_kps[0]
        if i >= next_kp_idx and len(episode_kps) > 1:
            episode_kps.pop(0)
            next_kp_idx = episode_kps[0]

        target_action = np.concatenate([
            low_dim_obs[next_kp_idx].left.gripper_pose,
            [low_dim_obs[next_kp_idx].left.gripper_open],
            low_dim_obs[next_kp_idx].right.gripper_pose,
            [low_dim_obs[next_kp_idx].right.gripper_open]
        ])  # (16,)

        state_list.append(current_state)
        action_list.append(target_action)
        pcd_index_list.append([episode, i])

    # 记录episode边界
    if episode == 0:
        episode_ends.append(len(low_dim_obs))
    else:
        episode_ends.append(len(low_dim_obs) + episode_ends[-1])

# 构建最终ReplayBuffer
replay_buffer = {
    'meta': {
        'episode_ends': np.array(episode_ends),
        'keyframe_indices': np.array(keyframe_indices)
    },
    'data': {
        'state': np.array(state_list),          # (12000, 16)
        'action': np.array(action_list),        # (12000, 16)
        'point_cloud': np.array(pcd_index_list), # (12000, 2)
        'dino_feature': np.array(pcd_index_list),
        'lang': np.array(lang_list)             # (12000, 1024)
    }
}
```

### 7.4 阶段四：训练时数据加载

```python
# 文件: occ_grasp_models/ppi/dataset/rlbench2_dataset.py

dataset = RLBench2Dataset(
    data_path='occ_grasp_models/data/training_raw/...',
    pcd_path='occ_grasp_models/data/training_processed/point_cloud/...',
    stats_filepath='occ_grasp_models/data/training_processed/norm_stats/norm_stats_*.pth',
    horizon_keyframe=4,
    horizon_continuous=50,
    prediction_type='keyframe_continuous'
)

# 数据集初始化
# 1. 加载ReplayBuffer
replay_buffer = ReplayBuffer.getData_keyframe_continuous(...)

# 2. 创建采样器
sampler = SequenceSamplerKeyframeContinuous(
    replay_buffer=replay_buffer,
    sequence_length_continuous=50,
    sequence_length_keyframe=4,
)

# 3. 加载归一化器
normalizer = LinearNormalizer()
normalizer.load_state_dict(torch.load(stats_filepath))

# 训练循环中的采样
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

for batch in dataloader:
    # batch 由 dataset.sampler.sample_sequence(idx) 组装
    # 其中 point_cloud / dino_feature 默认仅加载首帧视觉:
    # batch['obs']['point_cloud']: (128, 1, 6144, 6)
    # batch['obs']['lang']: (128, 54, 1024)
    # batch['action']: (128, 54, 16)

    # 送入模型
    loss = model.compute_loss(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 7.5 完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                      完整数据流转示意图                           │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │ RLBench2 环境│
                    └──────┬──────┘
                           │ live_demos
                           ▼
              ┌────────────────────────┐
              │ demo: List[BimanualObs]│
              │  ├─ perception_data    │
              │  ├─ left/right obs     │
              │  └─ object_6d_pose     │
              └────────┬───────────────┘
                       │ save_demo()
                       ▼
        ┌──────────────────────────────────┐
        │ 磁盘存储 (training_raw)           │
        │  ├─ RGB/depth/mask (PNG)         │
        │  ├─ low_dim_obs.pkl              │
        │  └─ variation_descriptions.pkl    │
        └───────┬──────────────────────────┘
                │
                ├─────────────────┬────────────────┬──────────────┐
                ▼                 ▼                ▼              ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
        │ save_ptc.py  │  │save_dino.py  │  │save_point_flow.py│  │save_norm_stats.py│
        │              │  │              │  │          │  │          │
        │ SAM + FPS    │  │ DINOv2融合   │  │ 点流坐标 │  │ 归一化   │
        └───────┬──────┘  └──────┬───────┘  └────┬─────┘  └────┬─────┘
                │                │               │             │
                ▼                ▼               ▼             ▼
        ┌────────────────────────────────────────────────────────┐
        │ 磁盘存储 (training_processed)                           │
        │  ├─ point_cloud/*.npy (Np, 6)                          │
        │  ├─ dino_feature/*.npy (Np, 384)                       │
        │  ├─ point_flow/*.npy (200, 3)                          │
        │  └─ norm_stats_*.pth                                   │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ GetData* 类      │
                    │ process_episodes │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │ ReplayBuffer (内存)    │
                  │ ├─ meta               │
                  │ │  ├─ episode_ends    │
                  │ │  └─ keyframe_indices│
                  │ └─ data               │
                  │    ├─ state (N, 16)   │
                  │    ├─ action (N, 16)  │
                  │    └─ point_cloud索引 │
                  └──────────┬────────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │ Sampler 采样器         │
                  │ create_indices()      │
                  └──────────┬────────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │ RLBench2Dataset       │
                  │ __getitem__()         │
                  │  ├─ 采样序列索引       │
                  │  ├─ 动态加载点云       │
                  │  ├─ 归一化             │
                  │  └─ 返回batch          │
                  └──────────┬────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ DataLoader     │
                    │ (batch_size=128)│
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ 模型训练        │
                    │ DiffusionModel │
                    └────────────────┘
```

---

## 8. 总结

### 核心数据结构

1. **low_dim_obs.pkl**：
   - BimanualObservation 对象列表
   - 每个臂提取 8 维（pose 7维 + open 1维）
   - 双臂总计 16 维

2. **ReplayBuffer 元数据**：
   - `episode_ends`：累积边界索引
   - `keyframe_indices`：全局关键帧位置
   - `openess_indices`：夹爪事件位置

3. **N**：所有 episode 的总时间步数 = `episode_ends[-1]`

### 核心概念

1. **state vs action**：
   - state：当前时刻的机器人状态
   - action：目标状态（下一步/下一关键帧）

2. **(episode_id, step_id) 索引**：
   - 磁盘文件索引对
   - 用于动态加载大型特征数据

3. **归一化统计**：
   - Min-Max 或标准化参数
   - 预计算并保存为 .pth 文件
   - 训练时加载并应用

### 数据流程

```
演示采集 → training_raw → 特征预处理 → training_processed → ReplayBuffer → Dataset → 模型训练
```
