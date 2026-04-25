# PPI点流（Point Stream）简明指南

## 一、具体形式

### 数据结构
```
episodes/episode0/world_ordered_rps200/
├── step000.npy  # shape: (200, 3) - 200个3D点的世界坐标
├── step001.npy  # shape: (200, 3) - 同一组点在下一时刻的位置
└── step002.npy  # ...
```

- **格式**: NumPy数组 `(num_points, 3)`，存储xyz坐标
- **类型**: `rps200`(200个点)、`rps50`(50个点)、`rps6144`(6144个点)
- **坐标系**: 世界坐标系

> 代码核验注记：迁移到 `occ_grasp_fall` 后，对应脚本位于 `occ_grasp_models/scripts/ppi/data_generation/save_point_flow.py`；训练配置的 `point_flow_type` 需要与实际目录名保持一致。

## 二、内容意义

### 核心概念：物体表面点的轨迹追踪

点流不是速度场或位移场，而是**物体表面固定点的时序轨迹**：

1. **初始化**：在第一帧检测物体，随机采样200个表面点
2. **追踪原理**：点"粘在"物体表面，通过物体6D位姿追踪
3. **对应关系**：`step000.npy`的第i个点 ↔ `step001.npy`的第i个点（同一个物理点）

### 数学定义
```python
# 点在不同时刻的位置关系
p_world(t) = R(t) @ p_object + t(t)
# 其中：p_object是点在物体坐标系的固定位置
#      R(t), t(t)是物体在时刻t的旋转和平移
```

## 三、关键代码实现

### 1. 点流生成 (`occ_grasp_models/scripts/ppi/data_generation/save_point_flow.py`)

```python
class GetPointFlow():
    def pointflow_from_tracks(self, depth, extrinsics, intrinsics, pixel_tracks):
        """从2D像素追踪生成3D点流"""
        # 2D像素 → 3D相机坐标
        z_values = depth[pixel_tracks[:, 0].astype(int),
                        pixel_tracks[:, 1].astype(int)]
        x = (pixel_tracks[:, 1] - intrinsics[0, 2]) * z_values / intrinsics[0, 0]
        y = (pixel_tracks[:, 0] - intrinsics[1, 2]) * z_values / intrinsics[1, 1]

        # 相机坐标 → 世界坐标
        points_camera = np.stack([x, y, z_values, np.ones_like(x)], axis=1).T
        points_world = extrinsics @ points_camera
        return points_world[:3].T  # shape: (N, 3)

    def world_to_object_coordinates(self, world_points, matrix):
        """世界坐标 → 物体坐标（用于建立对应关系）"""
        R = matrix[:3,:3]
        t = matrix[:3, 3]
        return (world_points - t) @ R.T

    def object_to_world_coordinates(self, object_points, matrix):
        """物体坐标 → 世界坐标（用于追踪）"""
        R = matrix[:3,:3]
        t = matrix[:3, 3]
        return object_points @ R.T + t

    def process_episodes(self, ...):
        # 核心流程：建立点对应关系
        object_pose_initial = low_dim_obs[0].object_6d_pose["matrix"]
        pc_object_frame = self.world_to_object_coordinates(pc_initial, object_pose_initial)

        # 通过物体位姿追踪每个时间步的点位置
        point_cloud = [pc_initial]
        for step in range(1, step_num):
            object_pose_t = low_dim_obs[step].object_6d_pose["matrix"]
            pc_t = self.object_to_world_coordinates(pc_object_frame, object_pose_t)
            point_cloud.append(pc_t)

        # 随机采样200个点（保持索引一致）
        rand_idx = np.random.choice(total_points, 200, replace=False)
        sampled_pc = point_cloud[:, rand_idx]  # 所有时间步用相同索引
```

### 2. 数据加载 (`occ_grasp_models/ppi/common/sampler_keyframe_continuous.py`)

```python
def load_point_flow(self, indices):
    """仅加载关键帧的点流"""
    point_flow = []
    # 只取最后N个关键帧索引
    indice_keyframe = indices[-self.sequence_length_keyframe:]

    for idx in indice_keyframe:
        episode, step = input_arr[idx]
        path = f'episode{episode}/{self.point_flow_type}/step{step:03d}.npy'
        data = np.load(path)  # (200, 3)
        point_flow.append(data)

    return np.array(point_flow)  # (num_keyframes, 200, 3)
```

### 3. 模型使用 (`occ_grasp_models/ppi/model/vision/observation_encoder.py`)

```python
class ObservationEncoder(nn.Module):
    def __init__(self):
        # 点流编码器：3D坐标 → 特征
        self.point_flow_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, observations):
        # 编码初始点流作为空间参考
        initial_point_flow = observations['initial_point_flow']  # (B, 200, 3)
        point_features = self.point_flow_mlp(initial_point_flow)  # (B, 200, D)
        return point_features
```

## 四、使用场景

1. **空间定位**：提供物体准确的3D位置信息
2. **轨迹规划**：关键帧处的点流指导机器人运动路径
3. **接触预测**：通过点云密度判断抓取/接触位置

## 五、关键文件列表

| 功能 | 文件路径 |
|-----|---------|
| 生成点流 | `occ_grasp_models/scripts/ppi/data_generation/save_point_flow.py` |
| 数据加载 | `occ_grasp_models/ppi/common/sampler_keyframe_continuous.py` |
| 模型编码 | `occ_grasp_models/ppi/model/vision/observation_encoder.py` |
| 策略使用 | `occ_grasp_models/ppi/policy/ppi.py` |

## 总结

PPI点流是一种**基于物体坐标系的点追踪方法**：
- 通过将点固定在物体表面，利用6D位姿变换追踪点的世界坐标
- 每个`.npy`文件存储同一组点在该时刻的位置，而非位移
- 为机器人提供精确的空间目标引导，特别适合操作任务
