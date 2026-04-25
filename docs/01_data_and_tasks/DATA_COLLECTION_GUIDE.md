# 双臂遮挡抓取演示数据收集指南

**创建日期**: 2026-01-01
**更新日期**: 2026-01-14
**项目目标**: 为策略条件注入模型和关键点位姿注入模型提供带标签的演示数据

**关联文档**:
- [STRATEGY_MODEL_DESIGN.md](../06_misc/act_family/STRATEGY_MODEL_DESIGN.md) - 策略/阶段条件注入模型扩展
- [KEYPOINT_POSE_INJECTION_PLAN.md](../06_misc/act_family/KEYPOINT_POSE_INJECTION_PLAN.md) - 关键点位姿预测模型扩展

---

## 目录

1. [概述与目标](#1-概述与目标)
2. [标注元素总览](#2-标注元素总览)
3. [RLBench数据流分析](#3-rlbench数据流分析)
4. [scene.py 修改方案](#4-scenepy-修改方案)
5. [任务定义改进 (BimanualEdgePhone)](#5-任务定义改进-bimanualedgephone)
5B. [其他任务扩展方案](#5b-其他任务扩展方案)
6. [训练数据加载 (launch_utils.py)](#6-训练数据加载-launch_utilspy)
7. [验证与调试](#7-验证与调试)
8. [评估时的阶段判断](#8-评估时的阶段判断)

---

## 1. 概述与目标

### 1.1 问题背景

对于平放在桌面上的扁平物体（如手机），直接抓取会导致夹爪与桌面碰撞。需要利用：
- **外在灵巧性**：利用环境条件（桌边、墙面等）
- **内在灵巧性**：双臂协同配合

### 1.2 三种策略场景

| 策略ID | 策略名称 | 环境条件 | 操作方式 |
|--------|----------|----------|----------|
| 1 | 边缘悬空 (EdgeHang) | 桌子边缘 | 推到边缘→悬空稳定→抓取 |
| 2 | 靠墙撬起 (WallLever) | 垂直墙面 | 推靠墙壁→撬起物体→抓取 |
| 3 | 按压翘起 (PressTilt) | 特殊形状物体 | 按压一端→翘起另一端→抓取 |

### 1.3 四个执行阶段

| 阶段ID | 阶段名称 | 描述 | 成功标准 |
|--------|----------|------|----------|
| 1 | 预操作 (PreManipulation) | 改变物体姿态创造空间 | 足够抓取空间+物体稳定 |
| 2 | 抓取 (Grasp) | 利用创造的空间抓牢物体 | 夹爪牢固抓住+保持稳定 |
| 3 | 清道 (ClearPath) | 移开辅助臂避免阻碍 | 辅助臂不阻碍后续路径 |
| 4 | 拿起 (Lift) | 沿预定路径拿起物体 | 物体到达目标位置 |

### 1.4 当前任务状态

| 任务 | 策略类型 | 阶段成功条件 | 状态 |
|------|---------|-------------|------|
| bimanual_edge_phone | 边缘悬空(1) | 已定义 | **可用** |
| bimanual_pivot_phone | 靠墙撬起(2) | 已定义 | **可用** |
| bimanual_pick_plate | 按压翘起(3) | 已定义 | **可用** |
| bimanual_pick_fork | 按压翘起(3) | 已定义 | **可用** |

> **注意**: 第5节详细描述了各任务的扩展方案，其中 BimanualEdgePhone 是最完整的参考示例。

---

## 2. 标注元素总览

演示数据中需要收集以下标注元素，存储在 `obs.misc` 字典中。

### 2.1 收集方式汇总

| 信息类型 | 收集位置 | 动态更新 | 任务类需提供 |
|---------|---------|---------|-------------|
| 策略类型 | scene.py `_get_misc()` | 每帧 | `STRATEGY_TYPE` 常量 |
| 阶段类型 | scene.py `_get_misc()` | 每帧 | `evaluate_phase_and_get_labels()` |
| 关键点位姿 | scene.py `_get_misc()` | 每帧自动读取 | 无（在TTT场景添加Dummy） |
| 2D投影 | scene.py `_get_misc()` | 每帧自动计算 | 无 |

### 2.2 策略与阶段标签

| 字段名 | 类型 | 说明 | 值域 |
|--------|------|------|------|
| `strategy_type` | int | 当前策略类型 | 1=EdgeHang, 2=WallLever, 3=PressTilt |
| `phase_type` | int | 当前执行阶段 | 1=PreManip, 2=Grasp, 3=ClearPath, 4=Lift |

### 2.3 关键点位姿（语义统一命名）

#### 接触点（所有任务都有）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `contact_position` | np.ndarray(3,) | 预操作接触点的3D位置 |
| `contact_quaternion` | np.ndarray(4,) | 预操作接触点的姿态 (w,x,y,z) |
| `contact_source` | str | 原始Dummy名称 (push_pt / press_pt) |

#### 抓取点（所有任务都有）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `grasp_position` | np.ndarray(3,) | 抓取点的3D位置 |
| `grasp_quaternion` | np.ndarray(4,) | 抓取点的姿态 (w,x,y,z) |

#### 环境约束点（部分任务有）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `has_affordance` | bool | 是否存在环境约束点 |
| `affordance_position` | np.ndarray(3,) | 环境约束点的3D位置 |
| `affordance_quaternion` | np.ndarray(4,) | 环境约束点的姿态 |
| `affordance_source` | str/None | 原始Dummy名称 (box_edge / wall_pivot / None) |

#### 2D投影（每个相机一组）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| `{cam}_contact_2d` | np.ndarray(2,) | 接触点在该相机的2D坐标 |
| `{cam}_grasp_2d` | np.ndarray(2,) | 抓取点在该相机的2D坐标 |
| `{cam}_affordance_2d` | np.ndarray(2,) | 环境约束点在该相机的2D坐标 |
| `{cam}_*_visible` | bool | 该关键点是否在该相机视野内 |

### 2.4 语义统一命名映射

```python
# 场景层 → 数据层 映射
KEYPOINT_MAPPING = {
    'contact': ['push_pt', 'press_pt'],      # → contact_*
    'grasp': ['grasp_pt'],                    # → grasp_*
    'affordance': ['box_edge', 'wall_pivot'], # → affordance_*
}
```

---

## 3. RLBench数据流分析

### 3.1 演示数据收集流程

```
get_demo()
    │
    ├── 创建 demo = [] 列表
    │
    ├── 定义 do_record() 回调:
    │       └── _demo_record_step(demo, record, callback)
    │               └── demo.append(get_observation())  # 逐帧记录
    │
    ├── 首帧记录（修复: 2026-01-14）:
    │       ├── pyrep.step()                           # 物理仿真一步
    │       ├── evaluate_phase_and_get_labels()        # ← 初始化阶段标签
    │       └── demo.append(get_observation())         # 记录首帧
    │
    └── execute_waypoints_bimanual_phased(do_record)
            │
            ├── for each phase:
            │       for each waypoint:
            │           ├── path.step()      # 执行路径一步
            │           ├── self.step()      # 仿真一帧
            │           ├── 更新标签状态     # ← 新增
            │           └── do_record()      # 记录当前帧观测
            │
            └── wait_after: 等待期间也更新标签并调用 do_record()
```

### 3.2 数据流闭环

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. 任务定义 (bimanual_edge_phone.py)                                      │
│    ├── STRATEGY_TYPE = 1                                                  │
│    ├── PhasedSuccessEvaluator(phase_conditions={1:..., 2:..., 3:..., 4:})│
│    └── evaluate_phase_and_get_labels() → (strategy_type, phase_type)     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. 演示执行 (scene.py)                                                    │
│    execute_waypoints_bimanual_phased():                                  │
│    ├── 每帧: task.evaluate_phase_and_get_labels()                        │
│    ├── 设置: _current_strategy_type, _current_phase_type                 │
│    └── do_record() → _get_misc() → obs.misc (包含所有标签)               │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. 数据保存 (dataset_generator_bimanual.py)                               │
│    save_demo() → pickle.dump(demo) → 标签自动包含在 obs.misc 中          │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. 训练加载 (launch_utils.py)                                             │
│    _add_keypoints_to_replay():                                            │
│    └── obs_dict['strategy_type'] = obs.misc.get('strategy_type', 1)      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. scene.py 修改方案

### 4.1 添加状态变量 (__init__)

```python
# 在 Scene.__init__ 中添加
self._current_strategy_type = None
self._current_phase_type = None
```

### 4.2 3D到2D投影函数

**重要**：此函数必须与 PyRep 的相机参数格式兼容。

#### 4.2.1 PyRep 相机参数说明

| 参数 | 来源 | 格式 | 说明 |
|------|------|------|------|
| `extrinsic` | `camera.get_matrix()` | [4,4] | **camera-to-world** 变换（相机位姿在世界坐标系中） |
| `intrinsic` | `camera.get_intrinsic_matrix()` | [3,3] | 内参矩阵，**焦距为负值**（CoppeliaSim约定） |

**注意**：`extrinsic` 不是 world-to-camera，需要先求逆！

#### 4.2.2 正确的投影实现

参考 `PyRep/pyrep/objects/vision_sensor.py` 中的 `pointcloud_from_depth_and_camera_params()` 方法：

```python
from typing import Tuple
import numpy as np

def project_3d_to_2d(
    point_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, bool]:
    """
    将3D世界坐标点投影到2D图像像素坐标

    与 PyRep VisionSensor.pointcloud_from_depth_and_camera_params() 兼容。

    Args:
        point_3d: [3] 世界坐标系中的3D点
        extrinsic: [4, 4] 相机外参矩阵 (camera-to-world, 由 camera.get_matrix() 返回)
        intrinsic: [3, 3] 相机内参矩阵 (由 camera.get_intrinsic_matrix() 返回, 焦距为负)
        image_size: (width, height) 图像分辨率

    Returns:
        point_2d: [2] 像素坐标 (u, v)
        is_visible: 点是否在相机前方且在图像范围内

    投影原理:
        1. extrinsic 是 camera-to-world 变换，我们需要构建 world-to-camera 变换
        2. 设 R = extrinsic[:3,:3] (旋转), C = extrinsic[:3,3] (相机位置)
        3. world-to-camera: p_cam = R^T @ (p_world - C)
        4. 投影矩阵: cam_proj_mat = intrinsic @ [R^T | -R^T @ C]
        5. 投影: p_img_homo = cam_proj_mat @ [p_world, 1]^T
        6. 透视除法: u = p_img_homo[0]/p_img_homo[2], v = p_img_homo[1]/p_img_homo[2]
    """
    # Step 1: 从外参矩阵提取相机位姿
    # extrinsic 是 camera-to-world 变换
    R = extrinsic[:3, :3]   # [3,3] camera-to-world 旋转矩阵
    C = extrinsic[:3, 3:4]  # [3,1] 相机在世界坐标系中的位置

    # Step 2: 构建 world-to-camera 变换
    R_inv = R.T  # world-to-camera 旋转 (正交矩阵的逆等于转置)
    R_inv_C = R_inv @ C  # [3,1]

    # world-to-camera 外参矩阵 [3, 4]
    extrinsics_w2c = np.concatenate([R_inv, -R_inv_C], axis=-1)

    # Step 3: 构建完整投影矩阵 [3, 4]
    # 与 pointcloud_from_depth_and_camera_params 使用相同的矩阵构建方式
    cam_proj_mat = intrinsic @ extrinsics_w2c

    # Step 4: 投影 3D 点
    p_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    p_img_homo = cam_proj_mat @ p_homo  # [3]

    # Step 5: 检查是否在相机前方
    # p_img_homo[2] 是齐次坐标的 z 分量
    # 对于相机前方的点，此值应为正（与深度符号一致）
    z = p_img_homo[2]
    if z <= 0:
        return np.array([-1.0, -1.0]), False

    # Step 6: 透视除法得到像素坐标
    u = p_img_homo[0] / z
    v = p_img_homo[1] / z

    # Step 7: 检查是否在图像范围内
    width, height = image_size
    is_visible = (0 <= u < width) and (0 <= v < height)

    return np.array([u, v]), is_visible


def project_3d_to_2d_batch(
    points_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量投影多个3D点到2D图像坐标（向量化版本，更高效）

    Args:
        points_3d: [N, 3] N个世界坐标3D点
        extrinsic: [4, 4] 相机外参矩阵 (camera-to-world)
        intrinsic: [3, 3] 相机内参矩阵
        image_size: (width, height) 图像分辨率

    Returns:
        points_2d: [N, 2] 像素坐标
        is_visible: [N] 布尔数组，表示每个点是否可见
    """
    N = points_3d.shape[0]

    # 构建投影矩阵
    R = extrinsic[:3, :3]
    C = extrinsic[:3, 3:4]
    R_inv = R.T
    extrinsics_w2c = np.concatenate([R_inv, -R_inv @ C], axis=-1)
    cam_proj_mat = intrinsic @ extrinsics_w2c  # [3, 4]

    # 转换为齐次坐标 [N, 4]
    points_homo = np.concatenate([points_3d, np.ones((N, 1))], axis=1)

    # 投影 [N, 3]
    p_img_homo = (cam_proj_mat @ points_homo.T).T

    # 检查 z > 0
    z = p_img_homo[:, 2]
    valid_z = z > 0

    # 透视除法（避免除以零）
    z_safe = np.where(valid_z, z, 1.0)
    u = p_img_homo[:, 0] / z_safe
    v = p_img_homo[:, 1] / z_safe

    # 检查图像范围
    width, height = image_size
    valid_range = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    # 组合可见性
    is_visible = valid_z & valid_range

    # 不可见的点设为 [-1, -1]
    points_2d = np.stack([u, v], axis=1)
    points_2d[~is_visible] = [-1.0, -1.0]

    return points_2d, is_visible
```

#### 4.2.3 验证投影函数正确性

可以通过点云重建的逆过程验证投影函数：

```python
def verify_projection():
    """验证投影函数与 PyRep 点云重建一致"""
    from pyrep.objects.vision_sensor import VisionSensor

    camera = VisionSensor('cam_front')
    camera.handle_explicitly()

    # 获取相机参数
    extrinsic = camera.get_matrix()
    intrinsic = camera.get_intrinsic_matrix()
    resolution = camera.get_resolution()

    # 获取点云
    depth = camera.capture_depth(in_meters=True)
    pointcloud = camera.pointcloud_from_depth(depth)

    # 取中心点验证
    h, w = depth.shape
    cx, cy = w // 2, h // 2
    world_point = pointcloud[cy, cx]  # 注意: pointcloud 索引是 [y, x]

    # 投影回像素坐标
    pixel_2d, visible = project_3d_to_2d(world_point, extrinsic, intrinsic, (w, h))

    print(f"Ground truth pixel: ({cx}, {cy})")
    print(f"Projected pixel: ({pixel_2d[0]:.1f}, {pixel_2d[1]:.1f})")
    print(f"Error: ({abs(pixel_2d[0]-cx):.2f}, {abs(pixel_2d[1]-cy):.2f})")
    # 误差应该接近 0
```

### 4.3 修改 _get_misc() 方法（完整版）

**说明**：此方法收集所有标注信息，每帧都会被调用。

```python
def _get_misc(self):
    misc = {}

    # ===== 原有：相机参数收集（保持不变）=====
    for camera_name, camera in self.camera_sensors.items():
        if camera.still_exists():
            misc.update({
                f'{camera_name}_camera_extrinsics': camera.get_matrix(),
                f'{camera_name}_camera_intrinsics': camera.get_intrinsic_matrix(),
                f'{camera_name}_camera_near': camera.get_near_clipping_plane(),
                f'{camera_name}_camera_far': camera.get_far_clipping_plane(),
            })

    misc.update({"variation_index": self._variation_index})

    # ===== 原有：executed_demo_joint_position_action =====
    if self.robot.is_bimanual and self._right_execute_demo_joint_position_action is not None:
        misc.update({
            "right_executed_demo_joint_position_action": self._right_execute_demo_joint_position_action,
            "left_executed_demo_joint_position_action": self._left_execute_demo_joint_position_action
        })
        self._right_execute_demo_joint_position_action = None
        self._left_execute_demo_joint_position_action = None

    # ===== 新增1：策略类型和阶段类型 =====
    # 这两个值在 execute_waypoints_bimanual_phased() 中动态更新
    if self._current_strategy_type is not None:
        misc.update({
            "strategy_type": self._current_strategy_type,
            "phase_type": self._current_phase_type
        })

    # ===== 新增2：关键点位姿收集（每帧从Dummy对象读取）=====
    KEYPOINT_MAPPING = {
        'contact': ['push_pt', 'press_pt'],
        'grasp': ['grasp_pt'],
        'affordance': ['box_edge', 'wall_pivot']
    }

    # 接触点（所有任务都有）
    for dummy_name in KEYPOINT_MAPPING['contact']:
        if Object.exists(dummy_name):
            dummy = Dummy(dummy_name)
            misc['contact_position'] = dummy.get_position()
            misc['contact_quaternion'] = dummy.get_quaternion()
            misc['contact_source'] = dummy_name
            break

    # 抓取点（所有任务都有）
    if Object.exists('grasp_pt'):
        dummy = Dummy('grasp_pt')
        misc['grasp_position'] = dummy.get_position()
        misc['grasp_quaternion'] = dummy.get_quaternion()

    # 环境约束点（部分任务有）
    misc['has_affordance'] = False
    misc['affordance_position'] = np.zeros(3, dtype=np.float32)
    misc['affordance_quaternion'] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    misc['affordance_source'] = None

    for dummy_name in KEYPOINT_MAPPING['affordance']:
        if Object.exists(dummy_name):
            dummy = Dummy(dummy_name)
            misc['affordance_position'] = dummy.get_position()
            misc['affordance_quaternion'] = dummy.get_quaternion()
            misc['affordance_source'] = dummy_name
            misc['has_affordance'] = True
            break

    # ===== 新增3：关键点的2D投影（复用已收集的相机参数）=====
    for cam_name, camera in self.camera_sensors.items():
        extrinsic_key = f'{cam_name}_camera_extrinsics'
        intrinsic_key = f'{cam_name}_camera_intrinsics'

        if extrinsic_key not in misc or intrinsic_key not in misc:
            continue

        extrinsic = misc[extrinsic_key]
        intrinsic = misc[intrinsic_key]

        # 获取相机分辨率 (重要: 必须与实际图像尺寸一致)
        # camera.get_resolution() 返回 [width, height]
        resolution = camera.get_resolution() if camera.still_exists() else [128, 128]
        image_size = tuple(resolution)  # (width, height)

        # 投影接触点
        if 'contact_position' in misc:
            pos_2d, visible = project_3d_to_2d(
                misc['contact_position'], extrinsic, intrinsic, image_size
            )
            misc[f'{cam_name}_contact_2d'] = pos_2d if visible else np.array([-1.0, -1.0])
            misc[f'{cam_name}_contact_visible'] = visible

        # 投影抓取点
        if 'grasp_position' in misc:
            pos_2d, visible = project_3d_to_2d(
                misc['grasp_position'], extrinsic, intrinsic, image_size
            )
            misc[f'{cam_name}_grasp_2d'] = pos_2d if visible else np.array([-1.0, -1.0])
            misc[f'{cam_name}_grasp_visible'] = visible

        # 投影环境约束点
        if misc['has_affordance']:
            pos_2d, visible = project_3d_to_2d(
                misc['affordance_position'], extrinsic, intrinsic, image_size
            )
            misc[f'{cam_name}_affordance_2d'] = pos_2d if visible else np.array([-1.0, -1.0])
            misc[f'{cam_name}_affordance_visible'] = visible
        else:
            misc[f'{cam_name}_affordance_2d'] = np.array([-1.0, -1.0])
            misc[f'{cam_name}_affordance_visible'] = False

    return misc
```

### 4.4 修改 execute_waypoints_bimanual_phased()

**关键**：在每次调用 `do_record()` 之前更新策略和阶段状态变量。

```python
def execute_waypoints_bimanual_phased(self, do_record) -> bool:
    """按阶段顺序执行waypoints"""

    execution_phases = self.task.execution_phases
    # ... 现有初始化代码 ...

    for phase_idx, phase in enumerate(execution_phases):
        arm_name = phase['arm']
        waypoint_names = phase['waypoints']
        wait_after = phase.get('wait_after', 0)

        for wp_name in waypoint_names:
            # ... 路径获取和碰撞处理 ...

            done = False
            while not done:
                done = path.step()
                self.step()

                # 记录关节动作 (原有代码)
                executed_action = path.get_executed_joint_position_action()
                if arm_name == 'right':
                    self._right_execute_demo_joint_position_action = executed_action
                    self._left_execute_demo_joint_position_action = self.robot.left_arm.get_joint_positions()
                else:
                    self._left_execute_demo_joint_position_action = executed_action
                    self._right_execute_demo_joint_position_action = self.robot.right_arm.get_joint_positions()

                # ====== 新增：更新策略和阶段标签 ======
                if hasattr(self.task, 'evaluate_phase_and_get_labels'):
                    self._current_strategy_type, self._current_phase_type = \
                        self.task.evaluate_phase_and_get_labels()
                else:
                    # 使用 getattr 从任务获取默认策略类型，更灵活
                    self._current_strategy_type = getattr(self.task, 'STRATEGY_TYPE', 1)
                    self._current_phase_type = 1

                do_record()  # 记录当前帧（_get_misc会自动收集所有标签）
                success, term = self.task.success()

        # 阶段完成后等待
        if wait_after > 0:
            wait_steps = int(wait_after * 50)
            for _ in range(wait_steps):
                self.step()
                self._right_execute_demo_joint_position_action = self.robot.right_arm.get_joint_positions()
                self._left_execute_demo_joint_position_action = self.robot.left_arm.get_joint_positions()

                # ====== 新增：等待期间也更新标签 ======
                if hasattr(self.task, 'evaluate_phase_and_get_labels'):
                    self._current_strategy_type, self._current_phase_type = \
                        self.task.evaluate_phase_and_get_labels()

                do_record()
                success, term = self.task.success()

    # ... 返回 ...
```

### 4.5 修复 get_demo() 首帧标签 (2026-01-14)

**问题**：首帧记录时 `_current_strategy_type` 和 `_current_phase_type` 还未初始化，导致：
- 部分任务首帧显示 `phase=2`（错误）
- 部分任务首帧无 `phase_type` 字段

**修复位置**：`scene.py` 的 `get_demo()` 方法，在首帧记录前初始化标签。

```python
def get_demo(self, record: bool = True, ...):
    """Returns a demo (list of observations)"""
    # ... 初始化代码 ...

    if record:
        self.pyrep.step()  # Need this here or get_force doesn't work...
        # ====== 修复：初始化首帧的策略和阶段标签 ======
        # 确保首帧记录时有正确的阶段标签（之前首帧无标签或标签错误）
        if hasattr(self.task, 'evaluate_phase_and_get_labels'):
            self._current_strategy_type, self._current_phase_type = \
                self.task.evaluate_phase_and_get_labels()
        else:
            self._current_strategy_type = getattr(self.task, 'STRATEGY_TYPE', 1)
            self._current_phase_type = 1
        demo.append(self.get_observation())

    # ... 执行 waypoints ...
```

**效果**：
- 修复前：首帧 `phase=2` 或 `phase=None`
- 修复后：首帧正确显示 `phase=1`（预操作阶段）

---

## 5. 任务定义改进 (BimanualEdgePhone)

**文件位置**: `/home/hdliu/occ_grasp_fall/repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py`

> ⚠️ **重要更新 (2026-01-05)**
>
> **阶段条件设计从"累积条件"改为"非累积条件"**
>
> **问题背景**：
> - 之前使用累积条件（`phase2 = phase1 + new_cond`）
> - 导致 `EdgeOverhangCondition` 在手机被抓起后失效
> - Phase 2-4 因前置条件失效而无法被标记为完成
> - 评估时出现 `return=40` 但 `phase_4_success_rate=0%` 的矛盾
>
> **改动内容**：
> - Phase 1: `[EdgeOverhangCondition]` — 一次性，完成后不再检查
> - Phase 2: `[StableGraspCondition]` — 开始持续检查
> - Phase 3: `[StableGraspCondition, ClearPathCondition]` — 持续检查抓取
> - Phase 4: `[LiftedCondition]` — 只检查高度（与任务成功条件一致）
>
> **对已收集数据的影响**：
> - ❌ **使用旧代码收集的数据中 `phase_type` 标签可能错误**
> - 如果模型使用 `phase_type` 标签（如 `ACT_BC_ENC_STRATEGY`）→ **需要重新收集数据**
> - 如果模型不使用 `phase_type` 标签（如基础 `ACT_BC_ENC`）→ 可继续使用

### 5.1 常量定义

```python
# 在任务文件开头定义
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import logging

from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.conditions import Condition
from rlbench.backend.task import BimanualTask

# 策略类型名称映射（整数 → 字符串，供阅读参考和日志）
STRATEGY_NAMES = {
    1: "EdgeHang",      # 边缘悬空抓取
    2: "WallLever",     # 靠墙撬起
    3: "PressTilt",     # 按压翘起
}

# 阶段类型名称映射
PHASE_NAMES = {
    1: "PreManipulation",  # 预操作：推手机到边缘
    2: "Grasp",            # 抓取：抓住悬空部分
    3: "ClearPath",        # 清道：辅助臂移开
    4: "Lift",             # 拿起：抬起手机
}
```

### 5.2 阶段成功条件（完整实现）

#### 5.2.1 EdgeOverhangCondition - 边缘悬空条件

**关键设计**：使用相对坐标系，不受场景随机化影响。

```python
class EdgeOverhangCondition(Condition):
    """
    边缘悬空条件：检查手机是否悬空在盒子边缘，且处于稳定状态。

    判断逻辑（使用相对坐标系）：
    1. 获取 phone_edge 相对于 box_edge 坐标系的位置
    2. 悬空量 = -relative_y（相对y为负表示悬空）
    3. 条件满足：悬空量 > min_overhang 且手机稳定

    坐标系说明：
    - phone_edge 和 box_edge 姿态相同
    - 在 box_edge 坐标系下，phone_edge 初始时 y>0（手机在盒子上）
    - 推出后 phone_edge 相对 y 变为负值

    Args:
        target_object: 目标物体（Phone）
        box_edge_dummy: 盒子边缘标记点 (box_edge) - 作为参考坐标系
        phone_edge_dummy: 手机边缘标记点 (phone_edge)
        min_overhang: 最小悬空量（米），默认0.08
        velocity_threshold: 稳定性速度阈值，默认0.2
        required_stable_frames: 需要稳定的帧数，默认3
    """

    def __init__(self,
                 target_object: Shape,
                 box_edge_dummy: Dummy,
                 phone_edge_dummy: Dummy,
                 min_overhang: float = 0.08,
                 velocity_threshold: float = 0.2,
                 required_stable_frames: int = 3):
        self.target_object = target_object
        self.box_edge_dummy = box_edge_dummy
        self.phone_edge_dummy = phone_edge_dummy
        self.min_overhang = min_overhang
        self.velocity_threshold = velocity_threshold
        self.required_stable_frames = required_stable_frames
        self.stable_count = 0

    def condition_met(self):
        # 使用相对坐标系计算悬空量
        relative_pos = self.phone_edge_dummy.get_position(relative_to=self.box_edge_dummy)
        relative_y = relative_pos[1]

        # 悬空量 = -relative_y（相对y为负表示手机边缘已超出盒子边缘）
        overhang = -relative_y

        overhang_met = overhang > self.min_overhang

        if not overhang_met:
            self.stable_count = 0
            return False, False

        # 检查手机稳定性
        linear_vel, angular_vel = self.target_object.get_velocity()
        total_vel = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel) * 0.1

        if total_vel < self.velocity_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        is_stable = self.stable_count >= self.required_stable_frames

        return overhang_met and is_stable, False

    def reset(self):
        self.stable_count = 0
```

#### 5.2.2 StableGraspCondition - 稳定抓取条件

```python
class StableGraspCondition(Condition):
    """
    稳定抓取条件：检查目标手机是否被夹爪稳定抓取。

    成功条件：
    1. 目标手机被指定夹爪抓取
    2. 手机速度足够小（稳定）

    Args:
        gripper: 夹爪对象
        target_object: 目标物体（Phone）
        velocity_threshold: 稳定性速度阈值，默认0.1
        required_stable_frames: 需要稳定的帧数，默认3
    """

    def __init__(self,
                 gripper,
                 target_object: Shape,
                 velocity_threshold: float = 0.1,
                 required_stable_frames: int = 3):
        self.gripper = gripper
        self.target_object = target_object
        self.velocity_threshold = velocity_threshold
        self.required_stable_frames = required_stable_frames
        self.stable_count = 0
        self._target_handle = target_object.get_handle()

    def condition_met(self):
        # 检查目标手机是否被抓取
        grasped_objects = self.gripper.get_grasped_objects()
        is_grasped = any(
            obj.get_handle() == self._target_handle
            for obj in grasped_objects
        )

        if not is_grasped:
            self.stable_count = 0
            return False, False

        # 检查手机稳定性
        linear_vel, angular_vel = self.target_object.get_velocity()
        total_vel = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel) * 0.1

        if total_vel < self.velocity_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        is_stable = self.stable_count >= self.required_stable_frames

        return is_grasped and is_stable, False

    def reset(self):
        self.stable_count = 0
```

#### 5.2.3 ClearPathCondition - 清道条件

```python
class ClearPathCondition(Condition):
    """
    清道条件：检查辅助臂是否已移开，不会阻碍抓取臂后续的抬起路径。

    成功条件：
    1. 辅助臂夹爪已松开手机
    2. 辅助臂夹爪tip与抓取臂tip保持足够距离
    3. 辅助臂tip与抓取臂后续所有路径点保持足够距离

    坐标系：使用世界坐标系下的距离计算（两个动态物体间的真实距离）

    Args:
        aux_gripper: 辅助臂夹爪
        grasper_tip_dummy: 抓取臂tip的dummy
        aux_tip_dummy: 辅助臂tip的dummy
        lift_waypoints: 抓取臂后续要经过的路径点dummy列表
        min_clearance: 最小安全距离，默认0.15米
    """

    def __init__(self,
                 aux_gripper,
                 grasper_tip_dummy: Dummy,
                 aux_tip_dummy: Dummy,
                 lift_waypoints: List[Dummy] = None,
                 min_clearance: float = 0.15):
        self.aux_gripper = aux_gripper
        self.grasper_tip_dummy = grasper_tip_dummy
        self.aux_tip_dummy = aux_tip_dummy
        self.lift_waypoints = lift_waypoints or []
        self.min_clearance = min_clearance

    def condition_met(self):
        # 检查辅助臂是否已松开物体
        if len(self.aux_gripper.get_grasped_objects()) > 0:
            return False, False

        # 获取辅助臂tip位置
        aux_tip_pos = np.array(self.aux_tip_dummy.get_position())

        # 检查与抓取臂当前位置的距离
        grasper_tip_pos = np.array(self.grasper_tip_dummy.get_position())
        distance_to_grasper = np.linalg.norm(aux_tip_pos - grasper_tip_pos)

        if distance_to_grasper < self.min_clearance:
            return False, False

        # 检查与所有后续路径点的距离
        for wp_dummy in self.lift_waypoints:
            wp_pos = np.array(wp_dummy.get_position())
            distance_to_wp = np.linalg.norm(aux_tip_pos - wp_pos)
            if distance_to_wp < self.min_clearance:
                return False, False

        return True, False

    def reset(self):
        pass
```

#### 5.2.4 LiftedCondition - 抬起条件

```python
class LiftedCondition(Condition):
    """
    抬起条件：检查手机是否被抬起到目标高度。

    坐标系：使用世界坐标系的绝对 z 高度

    Args:
        target_object: 目标物体（Phone）
        min_height: 最小高度（米），默认1.1
    """

    def __init__(self, target_object: Shape, min_height: float = 1.1):
        self.target_object = target_object
        self.min_height = min_height

    def condition_met(self):
        pos = self.target_object.get_position()
        return pos[2] >= self.min_height, False

    def reset(self):
        pass
```

### 5.3 双臂角色选择器 (ArmRoleSelector)

#### 设计理念

**核心职责**：从两套路径点方案（`right_grasper` / `left_grasper`）中选择一个。方案确定后，臂角色自动确定。

**问题根源**：TTM 文件中的 waypoints 位姿是绝对坐标，针对特定臂（默认右臂）优化设计。当需要左臂作为 grasper 时，需要使用专门为左臂设计的镜像路径点。

**解决方案**：
1. 在 TTM 文件中创建两套路径点：原始路径点（右臂抓取）和镜像路径点（左臂抓取，`_a` 后缀）
2. ArmRoleSelector 从两套方案中选择可行且成本最优的方案
3. 方案确定后，臂角色自动确定

#### 两套路径点方案

| 方案名称 | Grasper臂 | Pusher臂 | Grasper路径点 | Pusher路径点 |
|----------|-----------|----------|---------------|--------------|
| `right_grasper` | right | left | waypoint0, 2, 4, 6 | waypoint1, 3, 5, 7 |
| `left_grasper` | left | right | waypoint0_a, 2_a, 4_a, 6_a | waypoint1_a, 3_a, 5_a, 7_a |

#### 选择策略

采用**两阶段筛选**（Feasibility-First）策略：

1. **可行性检查**（优先）：验证每套方案的 **pusher 臂到第一个 pusher 路径点** 是否可达
   - `right_grasper`：检查 **左臂（pusher）** 到 `waypoint1` 的路径规划
   - `left_grasper`：检查 **右臂（pusher）** 到 `waypoint1_a` 的路径规划

2. **筛选决策**：
   - 只有一套方案可行 → 选择该方案
   - 两套都可行 → 进入成本评估
   - 两套都不可行 → 记录警告，使用默认方案（`right_grasper`），让重试机制处理

3. **成本评估**（两套都可行时）：
   - 计算每套方案的总执行成本 = grasper臂成本 + **1.5 × pusher臂成本**
   - 选择总成本更低的方案

> **设计说明 - 为何检查 pusher 而非 grasper**：
> - **Pusher 先执行**：Phase 1 由 pusher 臂操作，此时物体处于初始位置
> - **Grasper 路径点会移动**：grasper 路径点通常附着在目标物体上，Phase 1 完成后物体翘起，路径点位置已改变
> - **Pusher 路径点稳定**：pusher 的 Phase 1 路径点在物体移动前执行，检测结果可靠
> - **成本权重 1.5x**：pusher 臂先操作且大部分路径点位置不变，其成本对方案选择更具参考价值
>
> **技术实现**：
> - 使用 `arm.get_path()` 进行可行性验证，参数与实际执行一致（`trials=100`）
> - `critical_pusher_indices` 参数允许配置需要检查的关键路径点，默认只检查第一个

#### 完整实现

```python
class ArmRoleSelector:
    """
    双臂角色选择器：通过选择路径点方案来确定臂角色分配。

    核心职责：从两套路径点方案（right_grasper / left_grasper）中选择一个。
    方案确定后，臂角色自动确定：
    - right_grasper 方案：右臂=grasper，左臂=pusher
    - left_grasper 方案：左臂=grasper，右臂=pusher

    选择策略：
    1. 可行性优先：检查每套方案的 pusher 臂到关键 pusher 路径点是否可达
       （pusher 先执行，其路径点位置在物体移动前是稳定的）
    2. 成本次优：两套都可行时，选择总执行成本更低的方案
       （pusher 成本权重 1.5x，因为先执行且路径点位置稳定）

    Args:
        robot: BimanualRobot实例
        right_tip_name: 右臂tip dummy名称
        left_tip_name: 左臂tip dummy名称
        position_weight: 位置成本权重（默认1.0）
        orientation_weight: 姿态成本权重（默认0.5）
        pusher_cost_weight: pusher成本额外权重（默认1.5）
    """

    def __init__(self,
                 robot: BimanualRobot,
                 right_tip_name: str = "Panda_rightArm_tip",
                 left_tip_name: str = "Panda_leftArm_tip",
                 position_weight: float = 1.0,
                 orientation_weight: float = 0.5,
                 pusher_cost_weight: float = 1.5):
        self.robot = robot
        self.right_tip_name = right_tip_name
        self.left_tip_name = left_tip_name
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.pusher_cost_weight = pusher_cost_weight

    def select_scheme(self, waypoint_sets: Dict[str, Dict[str, List[str]]],
                      critical_pusher_indices: List[int] = [0]
                      ) -> Tuple[str, Dict[str, str]]:
        """
        选择最优的路径点方案。

        Args:
            waypoint_sets: 两套路径点方案配置，格式如：
                {
                    'right_grasper': {
                        'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
                        'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
                    },
                    'left_grasper': {
                        'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
                        'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
                    }
                }
            critical_pusher_indices: 需要进行可行性验证的 pusher 路径点索引列表
                                     默认[0]表示只检查第一个 pusher 路径点
                                     （pusher 先执行，其初始路径点位置稳定可靠）

        Returns:
            Tuple[str, Dict[str, str]]: (选中的方案名称, 角色分配字典)
            例如: ('left_grasper', {'grasper': 'left', 'pusher': 'right'})
        """
        try:
            # ========== Step 1: 检查两套方案的可行性（检查 pusher 臂）==========
            right_feasible, right_reason = self._check_scheme_feasibility(
                'right_grasper', waypoint_sets, critical_pusher_indices
            )
            left_feasible, left_reason = self._check_scheme_feasibility(
                'left_grasper', waypoint_sets, critical_pusher_indices
            )

            logging.info(f"ArmRoleSelector: right_grasper feasible={right_feasible} ({right_reason}), "
                        f"left_grasper feasible={left_feasible} ({left_reason})")

            # ========== Step 2: 根据可行性筛选 ==========
            if right_feasible and not left_feasible:
                logging.info("ArmRoleSelector: Only right_grasper feasible, selecting it")
                return 'right_grasper', {'grasper': 'right', 'pusher': 'left'}

            if left_feasible and not right_feasible:
                logging.info("ArmRoleSelector: Only left_grasper feasible, selecting it")
                return 'left_grasper', {'grasper': 'left', 'pusher': 'right'}

            if not right_feasible and not left_feasible:
                logging.warning("ArmRoleSelector: Both schemes infeasible, using default (right_grasper)")
                return 'right_grasper', {'grasper': 'right', 'pusher': 'left'}

            # ========== Step 3: 两套都可行，基于成本选择 ==========
            right_cost = self._compute_scheme_cost('right_grasper', waypoint_sets)
            left_cost = self._compute_scheme_cost('left_grasper', waypoint_sets)

            logging.info(f"ArmRoleSelector: Cost analysis - "
                        f"right_grasper={right_cost:.4f}, left_grasper={left_cost:.4f}")

            if right_cost <= left_cost:
                logging.info("ArmRoleSelector: Selecting right_grasper (lower cost)")
                return 'right_grasper', {'grasper': 'right', 'pusher': 'left'}
            else:
                logging.info("ArmRoleSelector: Selecting left_grasper (lower cost)")
                return 'left_grasper', {'grasper': 'left', 'pusher': 'right'}

        except Exception as e:
            logging.warning(f"ArmRoleSelector failed: {e}, using default (right_grasper)")
            return 'right_grasper', {'grasper': 'right', 'pusher': 'left'}

    def _check_scheme_feasibility(self, scheme_name: str,
                                   waypoint_sets: Dict,
                                   critical_pusher_indices: List[int]
                                   ) -> Tuple[bool, str]:
        """
        检查指定方案的可行性。

        检查内容：该方案的 pusher 臂能否到达关键 pusher 路径点。

        为何检查 pusher 而非 grasper：
        - Pusher 在 Phase 1 先执行，此时物体处于初始位置
        - Grasper 路径点附着在物体上，Phase 1 后位置会改变
        - 检查 pusher 的初始可达性更能反映实际执行情况

        Args:
            scheme_name: 方案名称 ('right_grasper' 或 'left_grasper')
            waypoint_sets: 路径点配置
            critical_pusher_indices: 需要验证的 pusher 路径点索引

        Returns:
            Tuple[bool, str]: (是否可行, 原因说明)
        """
        if scheme_name not in waypoint_sets:
            return False, f"Scheme '{scheme_name}' not defined"

        scheme = waypoint_sets[scheme_name]
        pusher_wps = scheme.get('pusher', [])

        # 确定该方案对应的 pusher 臂
        # right_grasper 方案：pusher = left
        # left_grasper 方案：pusher = right
        if scheme_name == 'right_grasper':
            pusher_arm = self.robot.left_arm
            pusher_arm_name = 'left'
        else:  # left_grasper
            pusher_arm = self.robot.right_arm
            pusher_arm_name = 'right'

        # 对关键 pusher 路径点进行可达性验证
        for idx in critical_pusher_indices:
            if idx >= len(pusher_wps):
                continue
            wp_name = pusher_wps[idx]
            path_ok, reason = self._check_path_feasibility(pusher_arm, wp_name)
            if not path_ok:
                return False, f"{pusher_arm_name} arm (pusher) cannot reach {wp_name}: {reason}"

        return True, "All checks passed"

    def _check_path_feasibility(self, arm, waypoint_name: str) -> Tuple[bool, str]:
        """
        使用路径规划验证臂到路径点的可达性。

        参数与实际执行一致（trials=100）以确保检查结果与实际执行一致。

        Args:
            arm: PyRep Arm对象
            waypoint_name: 路径点名称

        Returns:
            Tuple[bool, str]: (是否可达, 原因说明)
        """
        try:
            wp_dummy = Dummy(waypoint_name)
            position = wp_dummy.get_position()
            euler = wp_dummy.get_orientation()

            arm.get_path(
                position,
                euler=euler,
                ignore_collisions=False,
                trials=100,
                max_configs=10,
                trials_per_goal=10,
                algorithm=Algos.RRTConnect
            )
            return True, "Path found"

        except ConfigurationPathError as e:
            return False, "No collision-free path"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def _compute_scheme_cost(self, scheme_name: str,
                              waypoint_sets: Dict) -> float:
        """
        计算指定方案的总执行成本。

        成本 = grasper臂成本 + pusher_cost_weight × pusher臂成本

        Pusher 成本权重更高的原因：
        - Pusher 在 Phase 1 先执行，其路径点位置稳定
        - Pusher 成本对方案选择更具参考价值

        Args:
            scheme_name: 方案名称
            waypoint_sets: 路径点配置

        Returns:
            float: 方案总成本
        """
        scheme = waypoint_sets[scheme_name]
        grasper_wps = scheme.get('grasper', [])
        pusher_wps = scheme.get('pusher', [])

        # 确定该方案对应的臂tip
        if scheme_name == 'right_grasper':
            grasper_tip = Dummy(self.right_tip_name)
            pusher_tip = Dummy(self.left_tip_name)
        else:  # left_grasper
            grasper_tip = Dummy(self.left_tip_name)
            pusher_tip = Dummy(self.right_tip_name)

        # 计算grasper臂到grasper路径点的成本
        grasper_cost = self._compute_waypoint_cost(grasper_tip, grasper_wps)

        # 计算pusher臂到pusher路径点的成本（乘以权重）
        pusher_cost = self._compute_waypoint_cost(pusher_tip, pusher_wps)

        # Pusher 成本乘以额外权重（默认1.5）
        total_cost = grasper_cost + self.pusher_cost_weight * pusher_cost

        logging.debug(f"Scheme {scheme_name}: grasper_cost={grasper_cost:.4f}, "
                     f"pusher_cost={pusher_cost:.4f} (×{self.pusher_cost_weight}), "
                     f"total={total_cost:.4f}")

        return total_cost

    def _compute_waypoint_cost(self, tip_dummy: Dummy,
                                waypoint_names: List[str]) -> float:
        """
        计算tip到一组路径点的距离成本。

        成本 = Σ (position_weight * 位置距离 + orientation_weight * 姿态差异) * 路径点权重
        其中第一个路径点权重为2.0（更重要），其余为1.0

        Args:
            tip_dummy: 臂tip的Dummy对象
            waypoint_names: 路径点名称列表

        Returns:
            float: 距离成本
        """
        total_cost = 0.0
        tip_pos = np.array(tip_dummy.get_position())
        tip_quat = np.array(tip_dummy.get_quaternion())

        for i, wp_name in enumerate(waypoint_names):
            try:
                wp_dummy = Dummy(wp_name)
                wp_pos = np.array(wp_dummy.get_position())
                wp_quat = np.array(wp_dummy.get_quaternion())

                # 位置距离
                pos_dist = np.linalg.norm(tip_pos - wp_pos)

                # 姿态差异（使用四元数点积）
                quat_dot = np.abs(np.dot(tip_quat, wp_quat))
                quat_dist = 1.0 - min(quat_dot, 1.0)

                # 第一个路径点权重更高
                weight = 2.0 if i == 0 else 1.0
                total_cost += (self.position_weight * pos_dist +
                              self.orientation_weight * quat_dist) * weight

            except Exception as e:
                logging.debug(f"Failed to compute cost for {wp_name}: {e}")
                continue

        return total_cost
```

#### 使用示例

```python
# 在任务 init_task() 中初始化选择器
self.role_selector = ArmRoleSelector(
    robot=self.robot,
    right_tip_name="Panda_rightArm_tip",
    left_tip_name="Panda_leftArm_tip",
    pusher_cost_weight=1.5,  # pusher 成本权重（默认1.5）
)

# 定义两套路径点方案
self.waypoint_sets = {
    'right_grasper': {
        'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
        'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
    },
    'left_grasper': {
        'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
        'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
    }
}

# ⚠️ 重要：必须在 post_placement_setup() 中选择方案！
# 因为 scene.py 的调用顺序是：
#   init_episode() → _place_task() [随机化] → post_placement_setup() → validate()
# 只有在场景随机放置后，waypoints 的位置才是最终位置。

def post_placement_setup(self) -> None:
    """在场景随机放置后选择方案"""
    # 选择方案（检查 pusher 臂可达性，返回方案名称和角色分配）
    self.active_waypoint_mode, role_assignment = self.role_selector.select_scheme(
        self.waypoint_sets,
        critical_pusher_indices=[0]  # 检查第一个 pusher 路径点（Phase 1 起始点）
    )

    # 更新角色分配
    self.current_role_assignment = role_assignment
    self._setup_waypoint_mapping()

    logging.info(f"Selected scheme: {self.active_waypoint_mode}, "
                f"roles: {self.current_role_assignment}")
```

**输出示例（可行性筛选 - 检查 pusher 臂）：**
```
ArmRoleSelector: right_grasper feasible=True (All checks passed),
                 left_grasper feasible=False (right arm (pusher) cannot reach waypoint1_a: No collision-free path)
ArmRoleSelector: Only right_grasper feasible, selecting it
```

**输出示例（成本选择 - pusher 成本权重 1.5x）：**
```
ArmRoleSelector: right_grasper feasible=True (All checks passed),
                 left_grasper feasible=True (All checks passed)
ArmRoleSelector: Cost analysis - right_grasper=1.5678, left_grasper=1.2345
ArmRoleSelector: Selecting left_grasper (lower cost)
```

#### 与原设计的主要差异

| 方面 | 原设计 | 新设计 |
|------|--------|--------|
| 核心职责 | 选择哪个臂做grasper | 从两套方案中选择一个 |
| 主要方法 | `select_roles()` | `select_scheme()` |
| 可行性检查 | 两臂分别检查同一组路径点 | 每套方案检查对应的路径点 |
| 路径点假设 | 固定路径点对两臂都可用 | 每套方案有独立的路径点 |
| 成本计算 | 单臂成本对比（含pusher对比权重） | 方案总成本对比（grasper+pusher） |
| 返回值 | `{'grasper': 'right', 'pusher': 'left'}` | `('right_grasper', {'grasper': 'right', ...})` |

### 5.4 分阶段成功评估器 (PhasedSuccessEvaluator)

**关键设计**：每个阶段的条件列表**累积**包含之前阶段的条件。

```python
class PhasedSuccessEvaluator:
    """
    分阶段成功条件评估器。

    特点：
    - 每个阶段的成功条件累积包含之前阶段的条件
    - 一个阶段的圆满完成预示着下一阶段的开始
    - 使用整数标签表示阶段 (1, 2, 3, 4)

    用途：
    - 数据收集时：标注每帧的phase_type
    - 评估时：判断模型是否成功完成每个阶段
    """

    def __init__(self, phase_conditions: Dict[int, List[Condition]]):
        self.phase_conditions = phase_conditions
        self.num_phases = len(phase_conditions)
        self.current_phase = 1
        self._phase_completion_status = {i: False for i in range(1, self.num_phases + 1)}

    def reset(self):
        """重置评估器状态"""
        self.current_phase = 1
        self._phase_completion_status = {i: False for i in range(1, self.num_phases + 1)}
        for conditions in self.phase_conditions.values():
            for cond in conditions:
                if hasattr(cond, 'reset'):
                    cond.reset()

    def evaluate_current_phase(self) -> Tuple[bool, int]:
        """评估当前阶段是否完成

        修复：一次调用中评估所有可完成的阶段，避免因跳帧导致阶段漏记录
        """
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

    def get_current_phase(self) -> int:
        """获取当前阶段ID"""
        return min(self.current_phase, self.num_phases)

    def is_phase_completed(self, phase_id: int) -> bool:
        """检查指定阶段是否已完成"""
        return self._phase_completion_status.get(phase_id, False)

    def is_task_successful(self) -> bool:
        """检查任务是否整体成功"""
        return self.current_phase > self.num_phases

    def get_phase_progress(self) -> Dict:
        """获取阶段进度信息（用于评估时记录）"""
        return {
            'current_phase': self.get_current_phase(),
            'total_phases': self.num_phases,
            'completed': self.is_task_successful(),
            'phase_status': self._phase_completion_status.copy()
        }
```

### 5.5 任务类完整实现 (BimanualEdgePhone)

基于新的 ArmRoleSelector 设计，任务类需要整合两套路径点方案。

```python
class BimanualEdgePhone(BimanualTask):
    """
    边缘悬空抓取任务：双臂协作将手机从盒子边缘抓取。

    策略类型: 1 (EdgeHang)

    路径点方案:
    - right_grasper: 右臂抓取（默认）
      - grasper路径点: waypoint0, 2, 4, 6
      - pusher路径点: waypoint1, 3, 5, 7
    - left_grasper: 左臂抓取（镜像）
      - grasper路径点: waypoint0_a, 2_a, 4_a, 6_a
      - pusher路径点: waypoint1_a, 3_a, 5_a, 7_a
    """

    STRATEGY_TYPE = 1  # EdgeHang策略

    def init_task(self) -> None:
        """初始化任务"""
        # ===== 获取场景对象 =====
        self.target_object = Shape('Phone')

        # 尝试获取边缘标记dummies（用于条件检测）
        self.box_edge = None
        self.phone_edge = None
        if Object.exists('box_edge'):
            self.box_edge = Dummy('box_edge')
        if Object.exists('phone_edge'):
            self.phone_edge = Dummy('phone_edge')

        # 注册可抓取对象
        self.register_graspable_objects([self.target_object])

        # ===== 初始化角色选择器 =====
        self.role_selector = ArmRoleSelector(
            robot=self.robot,
            right_tip_name="Panda_rightArm_tip",
            left_tip_name="Panda_leftArm_tip",
        )

        # ===== 定义两套路径点方案 =====
        self.waypoint_sets = {
            'right_grasper': {
                'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
                'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
            },
            'left_grasper': {
                'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
                'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
            }
        }

        # 当前激活的方案（默认右臂抓取）
        self.active_waypoint_mode = 'right_grasper'

        # 当前角色分配（从方案推导）
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}

        # 设置waypoint映射
        self._setup_waypoint_mapping()

        # 分阶段评估器（在post_placement_setup中设置）
        self.phased_evaluator = None

        # 注册最终成功条件
        self.register_success_conditions([
            LiftedCondition(self.target_object, min_height=1.1)
        ])

    def _get_active_waypoints(self) -> Dict[str, List[str]]:
        """获取当前激活方案的路径点配置"""
        return self.waypoint_sets[self.active_waypoint_mode]

    def _setup_waypoint_mapping(self):
        """
        根据当前激活方案和角色分配设置 waypoint_mapping。

        waypoint_mapping 告诉 scene.py 的 _get_waypoints() 方法
        每个路径点应该由哪个臂执行。
        """
        active_wps = self._get_active_waypoints()
        self.waypoint_mapping = defaultdict(lambda: 'right')

        for role, arm in self.current_role_assignment.items():
            for wp in active_wps.get(role, []):
                self.waypoint_mapping[wp] = arm

    def _setup_phased_evaluator(self):
        """设置分阶段成功条件评估器"""
        # 检查必要的对象是否存在
        if self.box_edge is None or self.phone_edge is None:
            logging.warning("box_edge or phone_edge not found, phased evaluator disabled")
            self.phased_evaluator = None
            return

        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']

        pusher_gripper = (self.robot.left_gripper if pusher_arm == 'left'
                         else self.robot.right_gripper)
        grasper_gripper = (self.robot.right_gripper if grasper_arm == 'right'
                          else self.robot.left_gripper)

        # 获取tip dummies
        grasper_tip_name = "Panda_rightArm_tip" if grasper_arm == 'right' else "Panda_leftArm_tip"
        pusher_tip_name = "Panda_leftArm_tip" if pusher_arm == 'left' else "Panda_rightArm_tip"

        try:
            grasper_tip = Dummy(grasper_tip_name)
            pusher_tip = Dummy(pusher_tip_name)
        except Exception as e:
            logging.warning(f"Failed to get tip dummies: {e}, phased evaluator disabled")
            self.phased_evaluator = None
            return

        # 获取抬起阶段的路径点（根据激活方案选择）
        active_wps = self._get_active_waypoints()
        lift_waypoints = []
        lift_wp_name = active_wps['grasper'][3]  # waypoint6 或 waypoint6_a
        if Object.exists(lift_wp_name):
            lift_waypoints.append(Dummy(lift_wp_name))

        # ====== 阶段条件定义 ======
        # Phase 1: 悬空条件
        phase1_conditions = [
            EdgeOverhangCondition(
                self.target_object, self.box_edge, self.phone_edge,
                min_overhang=0.05, velocity_threshold=0.2, required_stable_frames=3
            )
        ]

        # 共享的稳定抓取条件
        stable_grasp_condition = StableGraspCondition(
            grasper_gripper, self.target_object,
            velocity_threshold=0.1, required_stable_frames=3
        )

        # Phase 2: 稳定抓取
        phase2_conditions = [stable_grasp_condition]

        # Phase 3: 持续抓取 + 清道
        phase3_conditions = [
            stable_grasp_condition,
            ClearPathCondition(
                pusher_gripper, grasper_tip, pusher_tip,
                lift_waypoints=lift_waypoints, min_clearance=0.15
            )
        ]

        # Phase 4: 抬起
        phase4_conditions = [
            LiftedCondition(self.target_object, min_height=1.1)
        ]

        phase_conditions = {
            1: phase1_conditions,
            2: phase2_conditions,
            3: phase3_conditions,
            4: phase4_conditions
        }

        self.phased_evaluator = PhasedSuccessEvaluator(phase_conditions)
        logging.info("PhasedSuccessEvaluator initialized successfully")

    def init_episode(self, index: int) -> List[str]:
        """
        初始化episode。

        注意：方案选择在 post_placement_setup() 中执行，
        因为需要在场景随机放置之后才能正确评估可行性和成本。
        """
        self._variation_index = index
        self._step_count = 0

        # 重置为默认方案（将在 post_placement_setup 中更新）
        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()

        return ['push the phone over the box edge and grasp it from below']

    def post_placement_setup(self) -> None:
        """
        在场景随机放置后选择方案并设置评估器。

        此方法由 scene.py 在 _place_task() 之后、validate() 之前调用。
        此时场景已经被随机放置，可以正确评估可行性和成本。
        """
        # ===== 选择最优方案 =====
        # 检查 pusher 臂到第一个 pusher 路径点的可达性
        # （pusher 先执行 Phase 1，此时物体处于初始位置，路径点位置稳定可靠）
        self.active_waypoint_mode, role_assignment = self.role_selector.select_scheme(
            self.waypoint_sets,
            critical_pusher_indices=[0]  # 检查第一个 pusher 路径点
        )

        # 更新角色分配
        if role_assignment != self.current_role_assignment:
            self.current_role_assignment = role_assignment
            self._setup_waypoint_mapping()
            logging.info(f"Scheme selected: {self.active_waypoint_mode}, "
                        f"roles: {self.current_role_assignment}")

        # ===== 设置分阶段评估器 =====
        self._setup_phased_evaluator()
        if self.phased_evaluator:
            self.phased_evaluator.reset()

    def variation_count(self) -> int:
        return 1

    @property
    def execution_phases(self):
        """
        动态生成四阶段执行计划，使用当前激活方案的路径点。

        返回的结构供 scene.py 的 execute_waypoints_bimanual_phased() 使用。
        """
        active_wps = self._get_active_waypoints()
        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']

        pusher_wps = active_wps['pusher']   # 4个路径点
        grasper_wps = active_wps['grasper'] # 4个路径点

        return [
            # Phase 1: Pusher推动物体悬空 (3个路径点)
            {'arm': pusher_arm, 'waypoints': pusher_wps[:3], 'wait_after': 0.5},
            # Phase 2: Grasper接近并抓取 (3个路径点)
            {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 0.5},
            # Phase 3: Pusher撤离 (1个路径点)
            {'arm': pusher_arm, 'waypoints': [pusher_wps[3]], 'wait_after': 0.5},
            # Phase 4: Grasper抬起物体 (1个路径点)
            {'arm': grasper_arm, 'waypoints': [grasper_wps[3]], 'wait_after': 0},
        ]

    # ========== 策略和阶段标签接口（scene.py调用）==========

    def get_strategy_type(self) -> int:
        """返回策略类型"""
        return self.STRATEGY_TYPE

    def get_current_phase(self) -> int:
        """返回当前执行阶段"""
        if self.phased_evaluator is None:
            return 1
        return self.phased_evaluator.get_current_phase()

    def evaluate_phase_and_get_labels(self) -> Tuple[int, int]:
        """
        评估当前状态并返回策略类型和阶段类型标签。
        用于演示数据收集时标注每一帧。

        Returns:
            (strategy_type, phase_type) 整数元组
        """
        strategy_type = self.STRATEGY_TYPE

        if self.phased_evaluator is None:
            phase_type = 1
        else:
            self.phased_evaluator.evaluate_current_phase()
            phase_type = self.phased_evaluator.get_current_phase()

        return strategy_type, phase_type

    def get_phase_progress(self) -> Dict:
        """获取阶段进度信息"""
        if self.phased_evaluator is None:
            return {'current_phase': 1, 'total_phases': 4, 'completed': False,
                    'phase_status': {1: False, 2: False, 3: False, 4: False}}
        return self.phased_evaluator.get_phase_progress()

    def get_role_assignment(self) -> Dict[str, str]:
        """返回当前的角色分配"""
        return self.current_role_assignment.copy()

    def get_active_scheme(self) -> str:
        """返回当前激活的路径点方案"""
        return self.active_waypoint_mode
```

#### 调用流程图

```
init_task()
│
├── 初始化 role_selector
├── 定义 waypoint_sets（两套方案）
├── 设置默认 active_waypoint_mode = 'right_grasper'
└── 调用 _setup_waypoint_mapping()

init_episode(index)
│
├── 重置为默认方案
└── 返回任务描述

_place_task() [由scene.py调用]
│
└── 随机化场景放置

post_placement_setup() [关键！场景放置后调用]
│
├── role_selector.select_scheme(waypoint_sets)
│   │
│   ├── 检查 right_grasper 可行性
│   │   └── 右臂 → waypoint0 可达?
│   │
│   ├── 检查 left_grasper 可行性
│   │   └── 左臂 → waypoint0_a 可达?
│   │
│   ├── 可行性筛选
│   │   ├── 只有一个可行 → 选择该方案
│   │   ├── 两个都可行 → 计算成本，选成本低的
│   │   └── 都不可行 → 用默认方案，依赖重试机制
│   │
│   └── 返回 (active_waypoint_mode, role_assignment)
│
├── 更新 active_waypoint_mode
├── 更新 current_role_assignment
├── 调用 _setup_waypoint_mapping()
└── 调用 _setup_phased_evaluator()

validate() [由scene.py调用]
│
└── 验证所有路径点可达（使用更新后的waypoint_mapping）

execute_waypoints_bimanual_phased() [由scene.py调用]
│
├── 获取 task.execution_phases
│   └── 动态生成，使用 active_waypoint_mode 对应的路径点
│
└── 按阶段执行
    └── 每帧调用 evaluate_phase_and_get_labels() 记录标签
```

### 5.6 场景TTT文件要求

在 `bimanual_edge_phone.ttt` 中需要确认以下对象存在：

| 对象 | 类型 | 用途 |
|------|------|------|
| `Phone` | Shape | 目标抓取物体 |
| `box_edge` | Dummy | 盒子边缘标记（参考坐标系） |
| `phone_edge` | Dummy | 手机边缘标记（用于悬空检测） |
| `push_pt` | Dummy | 预操作接触点（关键点位姿） |
| `grasp_pt` | Dummy | 抓取点（关键点位姿） |
| `Panda_rightArm_tip` | Dummy | 右臂夹爪tip |
| `Panda_leftArm_tip` | Dummy | 左臂夹爪tip |
| `waypoint0-7` | Dummy | 8个路径点 |

**坐标系要求**：
- `phone_edge` 和 `box_edge` 的姿态需相同
- 在 `box_edge` 坐标系下，`phone_edge` 初始时 y>0（手机在盒子上）
- 推出方向应是 `box_edge` 的 y 正方向

---

## 5B. 其他任务扩展方案

本节描述 `bimanual_pivot_phone`、`bimanual_pick_plate`、`bimanual_pick_fork` 三个任务的扩展方案。这三个任务采用与 `bimanual_edge_phone` **相同的 scheme-based 设计模式**，主要区别在于**预操作阶段的成功条件**不同。

### 5B.1 共同设计原则

#### 5B.1.1 预操作目标一致性

三个任务的预操作阶段目标相同：**使物体一端翘起，创造足够的下方空间以便夹爪插入抓取**。

| 任务 | 策略类型 | 预操作方式 | 翘起机制 |
|------|---------|-----------|---------|
| bimanual_pivot_phone | WallLever (2) | 推向墙壁并撬动 | 利用墙面作为支点 |
| bimanual_pick_plate | PressTilt (3) | 按压盘子边缘 | 杠杆原理使另一端翘起 |
| bimanual_pick_fork | PressTilt (3) | 按压叉子头部 | 杠杆原理使叉柄翘起 |

#### 5B.1.2 预操作成功条件设计

**核心思路**：检测 `grasp_pt` Dummy 在世界坐标系下的 z 高度是否超过阈值。

**设计理由**：
- `grasp_pt` 是附着在物体上的关键点，其高度直接反映抓取空间是否足够
- 与 `scene.py` 中 `_get_misc()` 收集的关键点体系一致
- 比检测物体倾斜角度更简单直观

**稳定性要求**：与 `EdgeOverhangCondition` 一致，需要物体在满足高度条件后保持稳定。

#### 5B.1.3 Scheme-Based 设计（与 BimanualEdgePhone 一致）

所有三个任务都采用与 BimanualEdgePhone 相同的 scheme-based 设计：

| 组件 | 说明 |
|------|------|
| `waypoint_sets` | 两套路径点方案（`right_grasper` / `left_grasper`） |
| `active_waypoint_mode` | 当前激活的方案名称 |
| `_get_active_waypoints()` | 获取当前激活方案的路径点配置 |
| `ArmRoleSelector.select_scheme()` | 选择可行且成本最优的方案 |
| `execution_phases` | 动态生成，基于激活方案的路径点 |
| `get_active_scheme()` | 返回当前激活的方案名称 |

#### 5B.1.4 可复用组件

| 组件 | 复用情况 | 说明 |
|------|---------|------|
| `GraspPointHeightCondition` | ✅ 三任务共用 | Phase 1 翘起检测（调整阈值） |
| `StableGraspCondition` | ✅ 完全复用 | Phase 2 抓取条件 |
| `ClearPathCondition` | ✅ 完全复用 | Phase 3 清道条件 |
| `LiftedCondition` | ✅ 复用（调整阈值） | Phase 4 拿起条件 |
| `ArmRoleSelector` | ✅ 完全复用 | scheme-based 方案选择器 |
| `PhasedSuccessEvaluator` | ✅ 完全复用 | 分阶段评估框架 |

#### 5B.1.5 step() 调试函数

每个任务都应实现 `step()` 方法，用于实时打印目标物体和 `grasp_pt` 的高度：

```python
def step(self) -> None:
    """每个仿真步骤都会被调用，用于追踪目标物体和 grasp_pt 高度"""
    self._step_count += 1
    # 每2步打印一次
    if self._step_count % 2 == 0:
        obj_z = self.target_object.get_position()[2]
        if self.grasp_pt is not None:
            grasp_pt_z = self.grasp_pt.get_position()[2]
            print(f"[Step {self._step_count:4d}] Object z={obj_z:.4f} | grasp_pt z={grasp_pt_z:.4f}")
        else:
            print(f"[Step {self._step_count:4d}] Object z={obj_z:.4f} | grasp_pt=N/A")
```

---

### 5B.2 GraspPointHeightCondition - 抓取点高度条件

新增的预操作成功条件类，用于检测物体翘起程度。

```python
class GraspPointHeightCondition(Condition):
    """
    抓取点高度条件：检查 grasp_pt 是否达到足够高度，表明物体已翘起。

    适用任务：
    - bimanual_pivot_phone (WallLever策略)
    - bimanual_pick_plate (PressTilt策略)
    - bimanual_pick_fork (PressTilt策略)

    判断逻辑：
    1. 获取 grasp_pt 在世界坐标系下的 z 坐标
    2. 检查是否超过 min_height 阈值
    3. 检查物体是否稳定（可选）

    Args:
        grasp_pt_dummy: 抓取点 Dummy 对象
        target_object: 目标物体（用于稳定性检测）
        min_height: 最小高度阈值（米）
        velocity_threshold: 稳定性速度阈值，默认0.2
        required_stable_frames: 需要稳定的帧数，默认3
    """

    def __init__(self,
                 grasp_pt_dummy: Dummy,
                 target_object: Shape,
                 min_height: float,
                 velocity_threshold: float = 0.2,
                 required_stable_frames: int = 3):
        self.grasp_pt_dummy = grasp_pt_dummy
        self.target_object = target_object
        self.min_height = min_height
        self.velocity_threshold = velocity_threshold
        self.required_stable_frames = required_stable_frames
        self.stable_count = 0

    def condition_met(self):
        # Step 1: 检查 grasp_pt 高度
        grasp_pt_z = self.grasp_pt_dummy.get_position()[2]
        height_met = grasp_pt_z >= self.min_height

        if not height_met:
            self.stable_count = 0
            return False, False

        # Step 2: 检查物体稳定性
        linear_vel, angular_vel = self.target_object.get_velocity()
        total_vel = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel) * 0.1

        if total_vel < self.velocity_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        is_stable = self.stable_count >= self.required_stable_frames

        return height_met and is_stable, False

    def reset(self):
        self.stable_count = 0
```

---

### 5B.3 BimanualPivotPhone 完整实现

#### 5B.3.1 任务参数

| 参数 | 值 | 说明 |
|------|-----|------|
| STRATEGY_TYPE | 2 | WallLever策略 |
| grasp_pt 高度阈值 | 0.80 m | Phase 1 成功条件 |
| LiftedCondition min_height | 0.90 m | Phase 4 成功条件 |
| 目标物体 | `Phone` | Shape 名称 |
| contact 关键点 | `push_pt` | 推动接触点 |
| affordance 关键点 | `wall_pivot` | 墙面支点 |
| Waypoints 数量 | 9 (0-8) | grasper: 0,2,4,6; pusher: 1,3,5,7,8 |

#### 5B.3.2 waypoint_sets 定义

```python
# BimanualPivotPhone 特殊：9个waypoints
# - grasper: 4个 (0,2,4,6)
# - pusher: 5个 (1,3,5,7,8) - waypoint8 用于清道撤退

self.waypoint_sets = {
    'right_grasper': {
        'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
        'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7', 'waypoint8']
    },
    'left_grasper': {
        'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
        'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a', 'waypoint8_a']
    }
}
```

#### 5B.3.3 execution_phases 动态生成

```python
@property
def execution_phases(self):
    """动态生成四阶段执行计划，使用当前激活方案的路径点"""
    active_wps = self._get_active_waypoints()
    pusher_arm = self.current_role_assignment['pusher']
    grasper_arm = self.current_role_assignment['grasper']

    pusher_wps = active_wps['pusher']   # 5个路径点
    grasper_wps = active_wps['grasper'] # 4个路径点

    return [
        # Phase 1: 推向墙壁并撬动 (4个pusher路径点)
        {'arm': pusher_arm, 'waypoints': pusher_wps[:4], 'wait_after': 0.5},
        # Phase 2: 抓取翘起部分 (3个grasper路径点)
        {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 0.5},
        # Phase 3: 辅助臂清道撤退 (1个pusher路径点)
        {'arm': pusher_arm, 'waypoints': [pusher_wps[4]], 'wait_after': 0.5},
        # Phase 4: 抬起物体 (1个grasper路径点)
        {'arm': grasper_arm, 'waypoints': [grasper_wps[3]], 'wait_after': 0.5},
    ]
```

#### 5B.3.4 step() 调试函数

```python
def step(self) -> None:
    """每个仿真步骤都会被调用，用于追踪 Phone 高度和 grasp_pt 高度"""
    self._step_count += 1
    if self._step_count % 2 == 0:
        phone_z = self.target_object.get_position()[2]
        if self.grasp_pt is not None:
            grasp_pt_z = self.grasp_pt.get_position()[2]
            print(f"[Step {self._step_count:4d}] Phone z={phone_z:.4f} | grasp_pt z={grasp_pt_z:.4f}")
        else:
            print(f"[Step {self._step_count:4d}] Phone z={phone_z:.4f} | grasp_pt=N/A")
```

#### 5B.3.5 完整任务类实现

```python
class BimanualPivotPhone(BimanualTask):
    """
    靠墙撬起抓取任务：双臂协作将手机推向墙壁并撬起抓取。

    策略类型: 2 (WallLever)

    路径点方案:
    - right_grasper: 右臂抓取（默认）
      - grasper路径点: waypoint0, 2, 4, 6
      - pusher路径点: waypoint1, 3, 5, 7, 8
    - left_grasper: 左臂抓取（镜像）
      - grasper路径点: waypoint0_a, 2_a, 4_a, 6_a
      - pusher路径点: waypoint1_a, 3_a, 5_a, 7_a, 8_a
    """

    STRATEGY_TYPE = 2  # WallLever策略

    def init_task(self) -> None:
        """初始化任务"""
        # ===== 获取场景对象 =====
        self.target_object = Shape('Phone')

        self.grasp_pt = None
        if Object.exists('grasp_pt'):
            self.grasp_pt = Dummy('grasp_pt')

        self.register_graspable_objects([self.target_object])

        # ===== 初始化角色选择器 =====
        self.role_selector = ArmRoleSelector(
            robot=self.robot,
            right_tip_name="Panda_rightArm_tip",
            left_tip_name="Panda_leftArm_tip",
        )

        # ===== 定义两套路径点方案 =====
        self.waypoint_sets = {
            'right_grasper': {
                'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
                'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7', 'waypoint8']
            },
            'left_grasper': {
                'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
                'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a', 'waypoint8_a']
            }
        }

        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()

        self.phased_evaluator = None

        self.register_success_conditions([
            LiftedCondition(self.target_object, min_height=0.9)
        ])

    def _get_active_waypoints(self) -> Dict[str, List[str]]:
        """获取当前激活方案的路径点配置"""
        return self.waypoint_sets[self.active_waypoint_mode]

    def _setup_waypoint_mapping(self):
        """设置waypoint到臂的映射"""
        active_wps = self._get_active_waypoints()
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for role, arm in self.current_role_assignment.items():
            for wp in active_wps.get(role, []):
                self.waypoint_mapping[wp] = arm

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        self._step_count = 0
        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()
        return ['push the phone against the wall and pivot it to grasp']

    def step(self) -> None:
        """调试用：追踪 Phone 和 grasp_pt 高度"""
        self._step_count += 1
        if self._step_count % 2 == 0:
            phone_z = self.target_object.get_position()[2]
            if self.grasp_pt is not None:
                grasp_pt_z = self.grasp_pt.get_position()[2]
                print(f"[Step {self._step_count:4d}] Phone z={phone_z:.4f} | grasp_pt z={grasp_pt_z:.4f}")
            else:
                print(f"[Step {self._step_count:4d}] Phone z={phone_z:.4f} | grasp_pt=N/A")

    def post_placement_setup(self) -> None:
        """在场景随机放置后选择方案"""
        self.active_waypoint_mode, role_assignment = self.role_selector.select_scheme(
            self.waypoint_sets,
            critical_pusher_indices=[0]  # 检查第一个 pusher 路径点
        )
        if role_assignment != self.current_role_assignment:
            self.current_role_assignment = role_assignment
            self._setup_waypoint_mapping()
            logging.info(f"Scheme selected: {self.active_waypoint_mode}, "
                        f"roles: {self.current_role_assignment}")

        self._setup_phased_evaluator()
        if self.phased_evaluator:
            self.phased_evaluator.reset()

    @property
    def execution_phases(self):
        active_wps = self._get_active_waypoints()
        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']
        pusher_wps = active_wps['pusher']
        grasper_wps = active_wps['grasper']

        return [
            {'arm': pusher_arm, 'waypoints': pusher_wps[:4], 'wait_after': 0.5},
            {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 0.5},
            {'arm': pusher_arm, 'waypoints': [pusher_wps[4]], 'wait_after': 0.5},
            {'arm': grasper_arm, 'waypoints': [grasper_wps[3]], 'wait_after': 0.5},
        ]

    def get_active_scheme(self) -> str:
        """返回当前激活的路径点方案"""
        return self.active_waypoint_mode

    # ... 其他接口方法（get_strategy_type, get_current_phase, 等）与 BimanualEdgePhone 相同
```

#### 5B.3.6 场景TTT文件要求

| 对象 | 类型 | 用途 |
|------|------|------|
| `Phone` | Shape | 目标抓取物体 |
| `push_pt` | Dummy | 推动接触点（contact关键点） |
| `grasp_pt` | Dummy | 抓取点（需附着在Phone上） |
| `wall_pivot` | Dummy | 墙面支点（affordance关键点） |
| `waypoint0-8` | Dummy | 原始9个路径点（right_grasper方案） |
| `waypoint0_a-8_a` | Dummy | 镜像9个路径点（left_grasper方案） |

---

### 5B.4 BimanualPickPlate 完整实现

#### 5B.4.1 任务参数

| 参数 | 值 | 说明 |
|------|-----|------|
| STRATEGY_TYPE | 3 | PressTilt策略 |
| grasp_pt 高度阈值 | 0.80 m | Phase 1 成功条件 |
| LiftedCondition min_height | 0.90 m | Phase 4 成功条件 |
| 目标物体 | `plate` | Shape 名称 |
| contact 关键点 | `press_pt` | 按压接触点 |
| affordance 关键点 | 无 | 此任务无环境约束点 |
| Waypoints 数量 | 8 (0-7) | grasper: 0,2,4,6; pusher: 1,3,5,7 |

#### 5B.4.2 waypoint_sets 定义

```python
# BimanualPickPlate: 8个waypoints，与 BimanualEdgePhone 布局相同
self.waypoint_sets = {
    'right_grasper': {
        'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
        'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
    },
    'left_grasper': {
        'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
        'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
    }
}
```

#### 5B.4.3 execution_phases 动态生成

```python
@property
def execution_phases(self):
    """动态生成四阶段执行计划"""
    active_wps = self._get_active_waypoints()
    pusher_arm = self.current_role_assignment['pusher']
    grasper_arm = self.current_role_assignment['grasper']

    pusher_wps = active_wps['pusher']   # 4个路径点
    grasper_wps = active_wps['grasper'] # 4个路径点

    return [
        # Phase 1: 按压盘子边缘使其翘起 (3个pusher路径点)
        {'arm': pusher_arm, 'waypoints': pusher_wps[:3], 'wait_after': 0.5},
        # Phase 2: 抓取翘起部分 (3个grasper路径点)
        {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 0.5},
        # Phase 3: 辅助臂清道 (1个pusher路径点)
        {'arm': pusher_arm, 'waypoints': [pusher_wps[3]], 'wait_after': 0.5},
        # Phase 4: 抬起物体 (1个grasper路径点)
        {'arm': grasper_arm, 'waypoints': [grasper_wps[3]], 'wait_after': 0.5},
    ]
```

#### 5B.4.4 step() 调试函数

```python
def step(self) -> None:
    """每个仿真步骤都会被调用，用于追踪 plate 高度和 grasp_pt 高度"""
    self._step_count += 1
    if self._step_count % 2 == 0:
        plate_z = self.target_object.get_position()[2]
        if self.grasp_pt is not None:
            grasp_pt_z = self.grasp_pt.get_position()[2]
            print(f"[Step {self._step_count:4d}] Plate z={plate_z:.4f} | grasp_pt z={grasp_pt_z:.4f}")
        else:
            print(f"[Step {self._step_count:4d}] Plate z={plate_z:.4f} | grasp_pt=N/A")
```

#### 5B.4.5 完整任务类实现

```python
class BimanualPickPlate(BimanualTask):
    """
    按压翘起抓取盘子任务：双臂协作，按压盘子边缘使其翘起后抓取。

    策略类型: 3 (PressTilt)

    路径点方案:
    - right_grasper: 右臂抓取（默认）
      - grasper路径点: waypoint0, 2, 4, 6
      - pusher路径点: waypoint1, 3, 5, 7
    - left_grasper: 左臂抓取（镜像）
      - grasper路径点: waypoint0_a, 2_a, 4_a, 6_a
      - pusher路径点: waypoint1_a, 3_a, 5_a, 7_a
    """

    STRATEGY_TYPE = 3  # PressTilt策略

    def init_task(self) -> None:
        """初始化任务"""
        self.target_object = Shape('plate')

        self.grasp_pt = None
        if Object.exists('grasp_pt'):
            self.grasp_pt = Dummy('grasp_pt')

        self.register_graspable_objects([self.target_object])

        self.role_selector = ArmRoleSelector(
            robot=self.robot,
            right_tip_name="Panda_rightArm_tip",
            left_tip_name="Panda_leftArm_tip",
        )

        self.waypoint_sets = {
            'right_grasper': {
                'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
                'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
            },
            'left_grasper': {
                'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
                'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
            }
        }

        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()

        self.phased_evaluator = None

        self.register_success_conditions([
            LiftedCondition(self.target_object, min_height=0.9)
        ])

    def _get_active_waypoints(self) -> Dict[str, List[str]]:
        return self.waypoint_sets[self.active_waypoint_mode]

    def _setup_waypoint_mapping(self):
        active_wps = self._get_active_waypoints()
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for role, arm in self.current_role_assignment.items():
            for wp in active_wps.get(role, []):
                self.waypoint_mapping[wp] = arm

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        self._step_count = 0
        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()
        return ['press the plate edge to tilt it and pick it up']

    def step(self) -> None:
        """调试用：追踪 plate 和 grasp_pt 高度"""
        self._step_count += 1
        if self._step_count % 2 == 0:
            plate_z = self.target_object.get_position()[2]
            if self.grasp_pt is not None:
                grasp_pt_z = self.grasp_pt.get_position()[2]
                print(f"[Step {self._step_count:4d}] Plate z={plate_z:.4f} | grasp_pt z={grasp_pt_z:.4f}")
            else:
                print(f"[Step {self._step_count:4d}] Plate z={plate_z:.4f} | grasp_pt=N/A")

    def post_placement_setup(self) -> None:
        self.active_waypoint_mode, role_assignment = self.role_selector.select_scheme(
            self.waypoint_sets,
            critical_pusher_indices=[0]  # 检查第一个 pusher 路径点
        )
        if role_assignment != self.current_role_assignment:
            self.current_role_assignment = role_assignment
            self._setup_waypoint_mapping()
            logging.info(f"Scheme selected: {self.active_waypoint_mode}, "
                        f"roles: {self.current_role_assignment}")

        self._setup_phased_evaluator()
        if self.phased_evaluator:
            self.phased_evaluator.reset()

    @property
    def execution_phases(self):
        active_wps = self._get_active_waypoints()
        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']
        pusher_wps = active_wps['pusher']
        grasper_wps = active_wps['grasper']

        return [
            {'arm': pusher_arm, 'waypoints': pusher_wps[:3], 'wait_after': 0.5},
            {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 0.5},
            {'arm': pusher_arm, 'waypoints': [pusher_wps[3]], 'wait_after': 0.5},
            {'arm': grasper_arm, 'waypoints': [grasper_wps[3]], 'wait_after': 0.5},
        ]

    def get_active_scheme(self) -> str:
        return self.active_waypoint_mode

    # ... 其他接口方法与 BimanualEdgePhone 相同
```

#### 5B.4.6 场景TTT文件要求

| 对象 | 类型 | 用途 |
|------|------|------|
| `plate` | Shape | 目标抓取物体 |
| `press_pt` | Dummy | 按压接触点（contact关键点） |
| `grasp_pt` | Dummy | 抓取点（需附着在plate上） |
| `waypoint0-7` | Dummy | 原始8个路径点（right_grasper方案） |
| `waypoint0_a-7_a` | Dummy | 镜像8个路径点（left_grasper方案） |

---

### 5B.5 BimanualPickFork 完整实现

#### 5B.5.1 任务参数

| 参数 | 值 | 说明 |
|------|-----|------|
| STRATEGY_TYPE | 3 | PressTilt策略 |
| grasp_pt 高度阈值 | 0.77 m | Phase 1 成功条件（叉子较小，阈值略低） |
| LiftedCondition min_height | 0.88 m | Phase 4 成功条件 |
| 目标物体 | `Fork_phy` | Shape 名称 |
| contact 关键点 | `press_pt` | 按压接触点（叉子头部） |
| affordance 关键点 | 无 | 此任务无环境约束点 |
| Waypoints 数量 | 8 (0-7) | grasper: 0,2,4,6; pusher: 1,3,5,7 |

#### 5B.5.2 waypoint_sets 定义

```python
# BimanualPickFork: 8个waypoints，与 BimanualEdgePhone/BimanualPickPlate 布局相同
self.waypoint_sets = {
    'right_grasper': {
        'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
        'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
    },
    'left_grasper': {
        'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
        'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
    }
}
```

#### 5B.5.3 execution_phases 动态生成

```python
@property
def execution_phases(self):
    """动态生成四阶段执行计划"""
    active_wps = self._get_active_waypoints()
    pusher_arm = self.current_role_assignment['pusher']
    grasper_arm = self.current_role_assignment['grasper']

    pusher_wps = active_wps['pusher']   # 4个路径点
    grasper_wps = active_wps['grasper'] # 4个路径点

    return [
        # Phase 1: 按压叉子头部使叉柄翘起 (3个pusher路径点)
        {'arm': pusher_arm, 'waypoints': pusher_wps[:3], 'wait_after': 0.5},
        # Phase 2: 抓取翘起的叉柄 (3个grasper路径点)
        {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 0.5},
        # Phase 3: 辅助臂清道 (1个pusher路径点)
        {'arm': pusher_arm, 'waypoints': [pusher_wps[3]], 'wait_after': 0.5},
        # Phase 4: 抬起叉子 (1个grasper路径点)
        {'arm': grasper_arm, 'waypoints': [grasper_wps[3]], 'wait_after': 0.5},
    ]
```

#### 5B.5.4 step() 调试函数

```python
def step(self) -> None:
    """每个仿真步骤都会被调用，用于追踪 Fork 高度和 grasp_pt 高度"""
    self._step_count += 1
    if self._step_count % 2 == 0:
        fork_z = self.target_object.get_position()[2]
        if self.grasp_pt is not None:
            grasp_pt_z = self.grasp_pt.get_position()[2]
            print(f"[Step {self._step_count:4d}] Fork z={fork_z:.4f} | grasp_pt z={grasp_pt_z:.4f}")
        else:
            print(f"[Step {self._step_count:4d}] Fork z={fork_z:.4f} | grasp_pt=N/A")
```

#### 5B.5.5 完整任务类实现

```python
class BimanualPickFork(BimanualTask):
    """
    按压翘起抓取叉子任务：双臂协作，按压叉子头部使叉柄翘起后抓取。

    策略类型: 3 (PressTilt)

    路径点方案:
    - right_grasper: 右臂抓取（默认）
      - grasper路径点: waypoint0, 2, 4, 6
      - pusher路径点: waypoint1, 3, 5, 7
    - left_grasper: 左臂抓取（镜像）
      - grasper路径点: waypoint0_a, 2_a, 4_a, 6_a
      - pusher路径点: waypoint1_a, 3_a, 5_a, 7_a
    """

    STRATEGY_TYPE = 3  # PressTilt策略

    def init_task(self) -> None:
        """初始化任务"""
        self.target_object = Shape('Fork_phy')

        self.grasp_pt = None
        if Object.exists('grasp_pt'):
            self.grasp_pt = Dummy('grasp_pt')

        self.register_graspable_objects([self.target_object])

        self.role_selector = ArmRoleSelector(
            robot=self.robot,
            right_tip_name="Panda_rightArm_tip",
            left_tip_name="Panda_leftArm_tip",
        )

        self.waypoint_sets = {
            'right_grasper': {
                'grasper': ['waypoint0', 'waypoint2', 'waypoint4', 'waypoint6'],
                'pusher': ['waypoint1', 'waypoint3', 'waypoint5', 'waypoint7']
            },
            'left_grasper': {
                'grasper': ['waypoint0_a', 'waypoint2_a', 'waypoint4_a', 'waypoint6_a'],
                'pusher': ['waypoint1_a', 'waypoint3_a', 'waypoint5_a', 'waypoint7_a']
            }
        }

        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()

        self.phased_evaluator = None

        self.register_success_conditions([
            LiftedCondition(self.target_object, min_height=0.88)
        ])

    def _get_active_waypoints(self) -> Dict[str, List[str]]:
        return self.waypoint_sets[self.active_waypoint_mode]

    def _setup_waypoint_mapping(self):
        active_wps = self._get_active_waypoints()
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for role, arm in self.current_role_assignment.items():
            for wp in active_wps.get(role, []):
                self.waypoint_mapping[wp] = arm

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        self._step_count = 0
        self.active_waypoint_mode = 'right_grasper'
        self.current_role_assignment = {'grasper': 'right', 'pusher': 'left'}
        self._setup_waypoint_mapping()
        return ['press the fork head to tilt the handle and pick it up']

    def step(self) -> None:
        """调试用：追踪 Fork 和 grasp_pt 高度"""
        self._step_count += 1
        if self._step_count % 2 == 0:
            fork_z = self.target_object.get_position()[2]
            if self.grasp_pt is not None:
                grasp_pt_z = self.grasp_pt.get_position()[2]
                print(f"[Step {self._step_count:4d}] Fork z={fork_z:.4f} | grasp_pt z={grasp_pt_z:.4f}")
            else:
                print(f"[Step {self._step_count:4d}] Fork z={fork_z:.4f} | grasp_pt=N/A")

    def post_placement_setup(self) -> None:
        self.active_waypoint_mode, role_assignment = self.role_selector.select_scheme(
            self.waypoint_sets,
            critical_pusher_indices=[0]  # 检查第一个 pusher 路径点
        )
        if role_assignment != self.current_role_assignment:
            self.current_role_assignment = role_assignment
            self._setup_waypoint_mapping()
            logging.info(f"Scheme selected: {self.active_waypoint_mode}, "
                        f"roles: {self.current_role_assignment}")

        self._setup_phased_evaluator()
        if self.phased_evaluator:
            self.phased_evaluator.reset()

    @property
    def execution_phases(self):
        active_wps = self._get_active_waypoints()
        pusher_arm = self.current_role_assignment['pusher']
        grasper_arm = self.current_role_assignment['grasper']
        pusher_wps = active_wps['pusher']
        grasper_wps = active_wps['grasper']

        return [
            {'arm': pusher_arm, 'waypoints': pusher_wps[:3], 'wait_after': 0.5},
            {'arm': grasper_arm, 'waypoints': grasper_wps[:3], 'wait_after': 0.5},
            {'arm': pusher_arm, 'waypoints': [pusher_wps[3]], 'wait_after': 0.5},
            {'arm': grasper_arm, 'waypoints': [grasper_wps[3]], 'wait_after': 0.5},
        ]

    def get_active_scheme(self) -> str:
        return self.active_waypoint_mode

    # ... 其他接口方法与 BimanualEdgePhone 相同
```

#### 5B.5.6 场景TTT文件要求

| 对象 | 类型 | 用途 |
|------|------|------|
| `Fork_phy` | Shape | 目标抓取物体 |
| `press_pt` | Dummy | 按压接触点（contact关键点） |
| `grasp_pt` | Dummy | 抓取点（需附着在Fork_phy上） |
| `waypoint0-7` | Dummy | 原始8个路径点（right_grasper方案） |
| `waypoint0_a-7_a` | Dummy | 镜像8个路径点（left_grasper方案） |

---

### 5B.6 任务参数汇总

| 任务 | STRATEGY_TYPE | Phase 1 阈值 | Phase 4 阈值 | 目标物体 | 原始WPs | 镜像WPs | has_affordance |
|------|--------------|-------------|-------------|---------|--------|---------|----------------|
| bimanual_edge_phone | 1 (EdgeHang) | overhang≥0.05m | z≥1.1m | Phone | 0-7 | 0_a-7_a | ✅ box_edge |
| bimanual_pivot_phone | 2 (WallLever) | grasp_pt.z≥0.80m | z≥0.9m | Phone | 0-8 | 0_a-8_a | ✅ wall_pivot |
| bimanual_pick_plate | 3 (PressTilt) | grasp_pt.z≥0.80m | z≥0.9m | plate | 0-7 | 0_a-7_a | ❌ |
| bimanual_pick_fork | 3 (PressTilt) | grasp_pt.z≥0.75m | z≥0.88m | Fork_phy | 0-7 | 0_a-7_a | ❌ |

### 5B.7 Scheme-Based 设计统一接口

所有四个双臂任务都实现以下统一接口：

| 方法/属性 | 返回类型 | 说明 |
|----------|---------|------|
| `waypoint_sets` | `Dict[str, Dict[str, List[str]]]` | 两套路径点方案配置 |
| `active_waypoint_mode` | `str` | 当前激活的方案名称 |
| `_get_active_waypoints()` | `Dict[str, List[str]]` | 获取当前激活方案的路径点 |
| `get_active_scheme()` | `str` | 返回当前激活方案名称 |
| `execution_phases` | `List[Dict]` | 动态生成的执行阶段 |

---

### 5B.7.1 父类 `_get_waypoints()` 的 Scheme 感知支持

> ✅ **更新 (2026-01-13)**
>
> 已修改父类 `Task._get_waypoints()` 方法，使其自动检测并支持 scheme-based 任务。
> **任务类无需覆写此方法**。

#### 修改位置

`repos/RLBench/rlbench/backend/task.py` 第 412-489 行

#### 核心改动

```python
def _get_waypoints(self, validating=False) -> List[Waypoint]:
    waypoints = []
    additional_waypoint_inits = []

    # === Step 1: Build waypoint name list ===
    if hasattr(self, '_get_active_waypoints') and callable(self._get_active_waypoints):
        # Scheme-based task: get waypoint names from active scheme
        active_wps = self._get_active_waypoints()
        all_wp_names = active_wps.get('grasper', []) + active_wps.get('pusher', [])
        # Sort by numeric index (waypoint0 < waypoint1 < ...)
        def extract_idx(n):
            m = re.search(r'waypoint(\d+)', n)
            return int(m.group(1)) if m else 0
        all_wp_names.sort(key=extract_idx)
        waypoint_names = all_wp_names
    else:
        # Traditional task: use fixed pattern waypoint0, waypoint1, ...
        waypoint_names = []
        idx = 0
        while True:
            name = 'waypoint%d' % idx
            if idx == self._stop_at_waypoint_index or not Object.exists(name):
                break
            waypoint_names.append(name)
            idx += 1

    # === Step 2: Process each waypoint (original logic preserved) ===
    for name in waypoint_names:
        # Extract numeric index for ability functions lookup
        idx_match = re.search(r'waypoint(\d+)', name)
        i = int(idx_match.group(1)) if idx_match else 0
        # ... 其余处理逻辑保持不变 ...
```

#### 工作原理

| 任务类型 | 检测方式 | 路径点来源 |
|---------|---------|-----------|
| **Scheme-based** | 存在 `_get_active_waypoints()` 方法 | 从 `active_wps['grasper'] + active_wps['pusher']` 获取 |
| **传统任务** | 无此方法 | 固定模式 `waypoint0, waypoint1, ...` |

#### 向后兼容

- 传统任务：行为与原来完全一致
- Scheme-based 任务：自动使用激活方案的路径点（含 `_a` 后缀）

#### 任务类要求

Scheme-based 任务只需实现以下接口，父类自动处理路径点获取：

```python
def _get_active_waypoints(self) -> Dict[str, List[str]]:
    """返回当前激活方案的路径点配置"""
    return self.waypoint_sets[self.active_waypoint_mode]
```

---

### 5B.8 实现验证清单

#### 5B.8.1 新增条件类

| 检查项 | 文件位置 | 状态 |
|--------|---------|------|
| `GraspPointHeightCondition` 类定义 | 各任务文件 | ✅ 已实现 |
| 稳定性检测逻辑 | `condition_met()` | ✅ 已实现 |
| `reset()` 方法 | 重置 stable_count | ✅ 已实现 |

#### 5B.8.2 各任务扩展

| 任务 | 需要添加的内容 | 状态 |
|------|--------------|------|
| bimanual_pivot_phone | STRATEGY_TYPE=2, GraspPointHeightCondition, PhasedSuccessEvaluator | ✅ 已实现 |
| bimanual_pick_plate | STRATEGY_TYPE=3, GraspPointHeightCondition, PhasedSuccessEvaluator | ✅ 已实现 |
| bimanual_pick_fork | STRATEGY_TYPE=3, GraspPointHeightCondition, PhasedSuccessEvaluator | ✅ 已实现 |

#### 5B.8.2.1 父类 `_get_waypoints()` Scheme 感知支持

> ✅ 已通过修改父类 `task.py` 实现，无需在各任务中单独覆写。详见 5B.7.1 节。

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `repos/RLBench/rlbench/backend/task.py` | `_get_waypoints()` 自动检测 scheme-based 任务 | ✅ 已实现 |

#### 5B.8.3 TTT场景确认

| 任务 | 原始Waypoints | 镜像Waypoints | 其他Dummies | 状态 |
|------|--------------|---------------|------------|------|
| bimanual_pivot_phone | waypoint0-8 | waypoint0_a-8_a | push_pt, grasp_pt, wall_pivot | 待确认 |
| bimanual_pick_plate | waypoint0-7 | waypoint0_a-7_a | press_pt, grasp_pt | 待确认 |
| bimanual_pick_fork | waypoint0-7 | waypoint0_a-7_a | press_pt, grasp_pt | 待确认 |

**重要**：所有任务都需要在TTT文件中创建镜像路径点（`_a`后缀），用于 `left_grasper` 方案。

---

## 6. 训练数据加载 (launch_utils.py)

**注意**：训练数据加载的修改是**模型特定**的，已迁移到对应的模型设计文档中。

### 6.1 数据加载修改位置

| 模型 | 加载的数据 | 详细文档 |
|------|-----------|---------|
| **ACT_BC_ENC_STRATEGY** | `strategy_type`, `phase_type` | [STRATEGY_MODEL_DESIGN.md 第4节](../06_misc/act_family/STRATEGY_MODEL_DESIGN.md#4-数据加载扩展-launch_utilspy) |
| **ACT_BC_ENC_KEYPOINT** | 关键点位姿 (`contact_*`, `grasp_*`, `affordance_*`), 2D投影 | [KEYPOINT_POSE_INJECTION_PLAN.md 第4节](../06_misc/act_family/KEYPOINT_POSE_INJECTION_PLAN.md#4-数据加载扩展-launch_utilspy) |

### 6.2 数据来源统一说明

无论哪个模型，数据都来自演示数据的 `obs.misc` 字典：

```python
# 所有标签都在 scene.py 的 _get_misc() 方法中写入
misc = demo[k].misc

# 策略/阶段标签（STRATEGY模型使用）
strategy_type = misc.get('strategy_type', 1)
phase_type = misc.get('phase_type', 1)

# 关键点位姿（KEYPOINT模型使用）
contact_position = misc.get('contact_position', np.zeros(3))
grasp_position = misc.get('grasp_position', np.zeros(3))
# ... 更多字段见各模型文档
```

### 6.3 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│ 数据收集（本文档第4节）                                           │
│ scene.py: _get_misc() → obs.misc                               │
│   ├── strategy_type, phase_type                                │
│   ├── contact_*, grasp_*, affordance_*                         │
│   └── {cam}_*_2d, {cam}_*_visible                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    pickle.dump(demo)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 训练加载（各模型 launch_utils.py）                                │
├───────────────────────┬─────────────────────────────────────────┤
│ ACT_BC_ENC_STRATEGY   │ ACT_BC_ENC_KEYPOINT                     │
│ 加载: strategy_type   │ 加载: contact_position                  │
│       phase_type      │       contact_quaternion                │
│                       │       grasp_position                    │
│                       │       grasp_quaternion                  │
│                       │       affordance_*                      │
│                       │       {cam}_*_2d                        │
└───────────────────────┴─────────────────────────────────────────┘
```

---

## 7. 验证与调试

### 7.1 数据收集验证脚本

```python
"""验证演示数据收集是否正确"""
import pickle
import os
from rlbench.backend.const import LOW_DIM_PICKLE

def verify_demo_labels(episode_path: str):
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
        demo = pickle.load(f)

    print(f"Demo length: {len(demo)} frames")

    phase_transitions = []
    prev_phase = None

    for i, obs in enumerate(demo):
        misc = obs.misc
        strategy = misc.get('strategy_type', 'N/A')
        phase = misc.get('phase_type', 'N/A')
        has_aff = misc.get('has_affordance', False)

        # 记录阶段转换
        if phase != prev_phase:
            phase_transitions.append((i, prev_phase, phase))
            prev_phase = phase

        # 每50帧打印一次
        if i % 50 == 0:
            print(f"Frame {i:3d}: strategy={strategy}, phase={phase}, has_affordance={has_aff}")

    print("\n=== Phase Transitions ===")
    for frame, from_phase, to_phase in phase_transitions:
        print(f"Frame {frame}: {from_phase} → {to_phase}")

    # 验证关键点
    if len(demo) > 0:
        misc = demo[0].misc
        print("\n=== Keypoint Poses (Frame 0) ===")
        for key in ['contact_position', 'grasp_position', 'affordance_position']:
            if key in misc:
                print(f"{key}: {misc[key]}")
```

### 7.2 预期阶段转换

```
预期的阶段转换（BimanualEdgePhone）:
════════════════════════════════════════════════════════
Phase 1 (PreManip)  → Phase 2 (Grasp)    : EdgeOverhangCondition 满足
Phase 2 (Grasp)     → Phase 3 (ClearPath): StableGraspCondition 满足
Phase 3 (ClearPath) → Phase 4 (Lift)     : ClearPathCondition 满足
任务完成                                  : LiftedCondition 满足
════════════════════════════════════════════════════════

验证要点:
- strategy_type 应始终为 1
- phase_type 应随任务执行从 1→2→3→4 递增
- 阶段转换应发生在正确的物理状态变化时
```

### 7.3 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| strategy_type 始终为 None | 任务类缺少 STRATEGY_TYPE | 添加 `STRATEGY_TYPE = 1` |
| phase_type 不变 | 阶段条件参数不合适 | 调整阈值（min_overhang, velocity_threshold等） |
| 关键点位置为零 | Dummy对象不存在 | 在TTT场景文件中添加 |
| 2D投影始终 [-1,-1] | 关键点不在相机视野 | 检查Dummy位置和相机视角 |

### 7.4 实现验证清单

确保 `dataset_generator_bimanual.py` 运行时能够准确可靠地记录所有标注信息。

#### 7.4.1 任务类实现清单 (bimanual_edge_phone.py)

| 检查项 | 说明 | 当前状态 |
|--------|------|----------|
| `STRATEGY_NAMES` 字典 | 脚本开头定义，整数→名称映射 | ✅ 已实现（第29-33行） |
| `PHASE_NAMES` 字典 | 脚本开头定义，整数→名称映射 | ✅ 已实现（第36-41行） |
| `STRATEGY_TYPE = 1` | 类属性，整数值 | ✅ 已实现（第523行） |
| `EdgeOverhangCondition` | **使用相对坐标系** | ✅ 已实现（第48行） |
| `StableGraspCondition` | 稳定抓取检测 | ✅ 已实现（第109行） |
| `ClearPathCondition` | 清道检测 | ✅ 已实现（第159行） |
| `LiftedCondition` | 抬起检测 | ✅ 已有（第209行） |
| `ArmRoleSelector` | 双臂角色选择 | ✅ 已实现（第230行） |
| `PhasedSuccessEvaluator` | 分阶段评估器 | ✅ 已实现（第445行） |
| `evaluate_phase_and_get_labels()` | 返回 `(strategy_type, phase_type)` | ✅ 已实现（第710行） |
| `box_edge`, `phone_edge` Dummy | 在 `init_task()` 中获取 | ✅ 已实现（第531-536行） |

#### 7.4.2 Scene.py 修改清单

| 检查项 | 说明 | 参考节 | 当前状态 |
|--------|------|--------|----------|
| `_current_strategy_type` | `__init__` 中初始化为 None | 4.1节 | ✅ 已实现（第110行） |
| `_current_phase_type` | `__init__` 中初始化为 None | 4.1节 | ✅ 已实现（第111行） |
| `_project_3d_to_2d()` | 3D到2D投影函数 | 4.2节 | ✅ 已实现（第887行） |
| `_get_misc()` 扩展 | 添加策略/阶段/关键点/2D投影 | 4.3节 | ✅ 已实现（第970行附近） |
| `execute_waypoints_bimanual_phased()` | 每帧更新标签 | 4.4节 | ✅ 已实现（第583行） |

#### 7.4.3 TTT场景文件清单

| 对象名 | 类型 | 用途 | 状态 |
|--------|------|------|------|
| `Phone` | Shape | 目标抓取物体 | 待验证 |
| `box_edge` | Dummy | 盒子边缘（参考坐标系） | 待验证 |
| `phone_edge` | Dummy | 手机边缘（悬空检测） | 待验证 |
| `push_pt` | Dummy | 预操作接触点 | 待验证 |
| `grasp_pt` | Dummy | 抓取点 | 待验证 |
| `waypoint0-7` | Dummy | 8个路径点 | 待验证 |

**坐标系要求**：
- `phone_edge` 和 `box_edge` 姿态相同
- 在 `box_edge` 坐标系下，`phone_edge` 初始时 y>0
- 推出方向是 `box_edge` 的 y 正方向

#### 7.4.4 训练数据加载清单 (launch_utils.py)

| 检查项 | 说明 | 参考节 | 当前状态 |
|--------|------|--------|----------|
| `create_replay()` 添加 ReplayElement | 策略/阶段/关键点位姿 | 6.1节 | ✅ 已实现（act_bc_enc_strategy agent，第104-106行） |
| `_add_keypoints_to_replay()` 读取 obs.misc | 从 misc 提取标签 | 6.2节 | ✅ 已实现（act_bc_enc_strategy agent，第275-293行） |

#### 7.4.5 完整数据流闭环

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. 任务定义 (bimanual_edge_phone.py)                                      │
│    ├── STRATEGY_TYPE = 1                                                  │
│    ├── PhasedSuccessEvaluator(phase_conditions={1:..., 2:..., 3:..., 4:})│
│    └── evaluate_phase_and_get_labels() → (strategy_type, phase_type)     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. 演示执行 (scene.py)                                                    │
│    execute_waypoints_bimanual_phased():                                  │
│    ├── 每帧: task.evaluate_phase_and_get_labels()                        │
│    ├── 设置: _current_strategy_type, _current_phase_type                 │
│    └── do_record() → _get_misc() → obs.misc (包含所有标签)               │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. 数据保存 (dataset_generator_bimanual.py)                               │
│    save_demo() → pickle.dump(demo) → 标签自动包含在 obs.misc 中          │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. 训练加载 (launch_utils.py)                                             │
│    _add_keypoints_to_replay():                                            │
│    └── obs_dict['strategy_type'] = obs.misc.get('strategy_type', 1)      │
└──────────────────────────────────────────────────────────────────────────┘
```

**关键调用链**：
```
dataset_generator_bimanual.py
    └── task_env.get_demos()
        └── scene.get_demo()
            └── execute_waypoints_bimanual_phased(do_record)
                ├── task.evaluate_phase_and_get_labels()  ← 获取标签
                ├── scene._current_strategy_type = ...     ← 存储标签
                └── do_record() → _get_misc()              ← 标签写入 obs.misc
```

---

## 8. 评估时的阶段判断

### 8.1 概述与修改目标

评估训练好的模型时，除了判断任务是否最终成功，还需要评估：
1. 模型是否成功完成了每个阶段
2. 在哪个阶段失败了
3. 各阶段的成功率统计

**核心思想**：复用任务类（如`BimanualEdgePhone`）中定义的 `PhasedSuccessEvaluator`，在评估时每帧调用阶段判断逻辑。

### 8.2 修改脚本概览与交织关系

#### 8.2.1 与前七部分的交织关系

| 修改位置 | 前1-7部分（数据收集） | 第8部分（评估） | 交织程度 |
|---------|---------------------|----------------|---------|
| `bimanual_edge_phone.py` | 定义阶段条件类和`PhasedSuccessEvaluator` | 评估时复用这些类 | **完全交织** |
| `scene.py` | 在`execute_waypoints_bimanual_phased()`中调用标签接口 | 不修改 | 无 |
| `_independent_env_runner.py` | 不修改 | 添加阶段评估统计 | **第8部分独立** |
| `custom_rlbench_env.py` | 不修改 | 添加阶段评估接口 | **第8部分独立** |
| `eval.py` | 不修改 | 配置和结果收集 | **第8部分独立** |

**结论**：第8部分的修改需要**依赖前5部分在任务类中定义的阶段条件和评估器**，但评估框架的修改是独立的新增代码。

#### 8.2.2 需要修改的脚本清单

| 脚本 | 相对路径 | 修改类型 | 修改内容 |
|------|---------|---------|---------|
| 任务类 | `repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py` | 与前5部分交织 | 确保`PhasedSuccessEvaluator`和阶段条件已实现（前5部分） |
| 评估执行器 | `repos/YARR/yarr/runners/_independent_env_runner.py` | **第8部分新增** | 添加阶段级别评估统计 |
| 环境封装 | `occ_grasp_models/helpers/custom_rlbench_env.py` | **第8部分新增** | 添加获取阶段状态的接口 |
| 评估入口 | `occ_grasp_models/eval.py` | **第8部分新增** | 传递阶段评估配置 |

---

### 8.3 具体修改方案

#### 8.3.1 任务类依赖（前5部分实现，评估时复用）

**文件**：`/home/hdliu/occ_grasp_fall/repos/RLBench/rlbench/bimanual_tasks/bimanual_edge_phone.py`

确保以下内容已在前5部分实现（评估时复用）：

```python
# 第63行之后添加（如尚未添加）
# ===========================================================
# 以下代码在前5部分已定义，评估时直接复用
# ===========================================================

class BimanualEdgePhone(BimanualTask):
    STRATEGY_TYPE = 1  # EdgeHang策略

    # ... init_task 中需添加 ...
    def init_task(self) -> None:
        # ... 现有代码 ...
        self.target_object = Shape('Phone')
        self.box_edge = Dummy('box_edge')      # 需添加
        self.phone_edge = Dummy('phone_edge')  # 需添加
        # ... 其他代码 ...

    # 关键接口：评估时调用此方法获取阶段状态
    def evaluate_phase_and_get_labels(self) -> Tuple[int, int]:
        """返回 (strategy_type, phase_type)"""
        strategy_type = self.STRATEGY_TYPE
        if self.phased_evaluator is None:
            phase_type = 1
        else:
            self.phased_evaluator.evaluate_current_phase()
            phase_type = self.phased_evaluator.get_current_phase()
        return strategy_type, phase_type

    def get_phase_progress(self) -> Dict:
        """获取阶段进度信息（评估时调用）"""
        if self.phased_evaluator is None:
            return {'current_phase': 1, 'total_phases': 4, 'completed': False,
                    'phase_status': {1: False, 2: False, 3: False, 4: False}}
        return self.phased_evaluator.get_phase_progress()
```

#### 8.3.2 custom_rlbench_env.py 修改

**文件**：`/home/hdliu/occ_grasp_fall/occ_grasp_models/helpers/custom_rlbench_env.py`

**前置修改**：在文件开头添加必要的导入

```python
# 在现有的 from typing import Type, List 行修改为：
from typing import Type, List, Dict, Tuple
```

**修改位置**：在 `CustomRLBenchEnv` 类中添加获取阶段状态的方法

```python
# 在 CustomRLBenchEnv 类中添加（约在 step 方法之后）

def get_phase_progress(self) -> Dict:
    """
    获取当前任务的阶段进度信息

    Returns:
        Dict: 包含阶段完成状态的字典，若任务不支持阶段评估则返回None

    注意：self._task 是 TaskEnvironment 实例，self._task._task 是实际的任务类实例
    """
    if self._task is None or self._task._task is None:
        return None

    task = self._task._task
    if hasattr(task, 'get_phase_progress'):
        return task.get_phase_progress()
    return None

def evaluate_current_phase(self) -> Tuple[bool, int]:
    """
    评估当前阶段是否完成

    Returns:
        (phase_completed, completed_phase): 阶段是否刚完成及刚完成的阶段ID

    注意：当 phase_completed=True 时，返回的第二个值是刚完成的阶段编号，不需要再 -1
    """
    if self._task is None or self._task._task is None:
        return False, 1

    task = self._task._task
    if hasattr(task, 'phased_evaluator') and task.phased_evaluator is not None:
        return task.phased_evaluator.evaluate_current_phase()
    return False, 1

def get_strategy_and_phase(self) -> Tuple[int, int]:
    """
    获取当前策略类型和阶段类型

    Returns:
        (strategy_type, phase_type)
    """
    if self._task is None or self._task._task is None:
        return 1, 1

    task = self._task._task
    if hasattr(task, 'evaluate_phase_and_get_labels'):
        return task.evaluate_phase_and_get_labels()

    # 默认值
    strategy = getattr(task, 'STRATEGY_TYPE', 1)
    return strategy, 1
```

#### 8.3.3 _independent_env_runner.py 修改

**文件**：`/home/hdliu/occ_grasp_fall/repos/YARR/yarr/runners/_independent_env_runner.py`

**修改位置1**：`_run_eval_independent` 方法开头（约第176行附近），添加阶段统计初始化

```python
# 在 success_count = 0 之后添加（约第177行后）
# ===== 新增：阶段级别评估统计 =====
phase_success_counts = {1: 0, 2: 0, 3: 0, 4: 0}  # 各阶段成功次数
max_phases_reached = []  # 每个episode达到的最大阶段
phase_completion_frames = {1: [], 2: [], 3: [], 4: []}  # 各阶段完成帧数
# =================================
```

**修改位置2**：episode 循环内（约第217-240行），在每帧评估阶段状态

```python
# 在 for replay_transition in generator: 循环内
# 在 episode_rollout.append(replay_transition) 之后添加（约第238行后）

# ===== 新增：每帧评估阶段状态 =====
if hasattr(env, 'evaluate_current_phase'):
    phase_completed, completed_phase = env.evaluate_current_phase()
    # 记录阶段完成事件（注意：返回值已经是 completed_phase，不需要 -1）
    if phase_completed:
        # 检查该阶段是否已记录过（避免重复记录）
        already_recorded = completed_phase in [r.info.get('phase_completed') for r in episode_rollout]
        if not already_recorded:
            # 标记该阶段刚完成
            replay_transition.info['phase_completed'] = completed_phase
            replay_transition.info['completion_frame'] = len(episode_rollout)
# ============================================
```

**修改位置3**：episode 结束后（约第254-291行），统计阶段结果

```python
# 在 reward = episode_rollout[-1].reward 之后（约第255行后）
# 添加阶段评估统计

# ===== 新增：统计该episode的阶段完成情况 =====
if hasattr(env, 'get_phase_progress'):
    phase_progress = env.get_phase_progress()
    if phase_progress is not None:
        phase_status = phase_progress.get('phase_status', {})
        current_max_phase = 0
        for phase_id in range(1, 5):
            if phase_status.get(phase_id, False):
                phase_success_counts[phase_id] += 1
                current_max_phase = phase_id
        max_phases_reached.append(current_max_phase)

        logging.info(f"Phase progress: {phase_status}, Max phase: {current_max_phase}")
# ===========================================
```

**修改位置4**：在 summaries 添加部分（约第349行附近），添加阶段评估指标

```python
# 在 summaries.append(ScalarSummary('eval_envs/failed_count', failed_count)) 之后添加

# ===== 新增：阶段级别评估指标 =====
total_episodes = success_count + failed_count
if total_episodes > 0:
    for phase_id in range(1, 5):
        phase_rate = phase_success_counts[phase_id] / total_episodes
        summaries.append(ScalarSummary(f'eval_envs/phase_{phase_id}_success_rate', phase_rate))

    if len(max_phases_reached) > 0:
        avg_max_phase = sum(max_phases_reached) / len(max_phases_reached)
        summaries.append(ScalarSummary('eval_envs/avg_max_phase', avg_max_phase))
# ==================================
```

#### 8.3.4 eval.py 修改（可选）

**文件**：`/home/hdliu/occ_grasp_fall/occ_grasp_models/eval.py`

**修改位置**：main 函数中，添加阶段评估配置（约第196行后）

```python
# 在 @hydra.main 装饰的 main 函数中，eval_cfg 处理之后

# ===== 新增：阶段评估配置（可选）=====
# 可在 conf/eval.yaml 中添加：
# phase_evaluation:
#   enabled: true
#   log_per_episode: true
# 这里只是传递配置，实际逻辑在 _independent_env_runner.py 中
# =====================================
```

---

### 8.4 完整数据流（含评估）

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. 任务定义 (bimanual_edge_phone.py) - 前5部分实现                         │
│    ├── STRATEGY_TYPE = 1                                                  │
│    ├── PhasedSuccessEvaluator(phase_conditions={1:..., 2:..., 3:..., 4:})│
│    ├── evaluate_phase_and_get_labels() → (strategy_type, phase_type)     │
│    └── get_phase_progress() → Dict                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────┴───────────────────────────┐
        ↓                                                       ↓
┌───────────────────────────────┐       ┌───────────────────────────────────┐
│ 2a. 数据收集 (scene.py)        │       │ 2b. 模型评估 (eval.py)              │
│     前4部分实现                 │       │     第8部分实现                     │
├───────────────────────────────┤       ├───────────────────────────────────┤
│ execute_waypoints_phased():   │       │ IndependentEnvRunner.start():     │
│ ├── 每帧: evaluate_phase_...  │       │ ├── 每帧: env.step(action)        │
│ ├── 设置: _current_phase_type │       │ ├── 每帧: env.evaluate_current_phase()│
│ └── do_record() → obs.misc    │       │ └── 统计: phase_success_counts    │
└───────────────────────────────┘       └───────────────────────────────────┘
        ↓                                                       ↓
┌───────────────────────────────┐       ┌───────────────────────────────────┐
│ 3a. 数据保存                   │       │ 3b. 结果保存                        │
│     dataset_generator.py       │       │     eval_data.csv                  │
│     → pickle.dump(demo)       │       │     ├── phase_1_success_rate      │
│     → obs.misc 含阶段标签      │       │     ├── phase_2_success_rate      │
└───────────────────────────────┘       │     ├── phase_3_success_rate      │
        ↓                               │     ├── phase_4_success_rate      │
┌───────────────────────────────┐       │     └── avg_max_phase             │
│ 4a. 训练加载                   │       └───────────────────────────────────┘
│     launch_utils.py            │
│     → replay.add(obs_dict)    │
└───────────────────────────────┘
```

---

### 8.5 评估指标说明

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| `phase_1_success_rate` | phase_1_count / total_episodes | 阶段1（预操作）成功率 |
| `phase_2_success_rate` | phase_2_count / total_episodes | 阶段2（抓取）成功率 |
| `phase_3_success_rate` | phase_3_count / total_episodes | 阶段3（清道）成功率 |
| `phase_4_success_rate` | phase_4_count / total_episodes | 阶段4（拿起）成功率，等于任务成功率 |
| `avg_max_phase` | mean(max_phases_reached) | 平均最大达到阶段（1-4） |
| `success_count` | 已有 | 任务完成总成功数 |
| `failed_count` | 已有 | 任务执行失败数 |

**解读示例**：
- `phase_1_success_rate=0.9, phase_2_success_rate=0.7` → 模型在抓取阶段有问题
- `avg_max_phase=2.5` → 平均能完成到抓取阶段，但经常在清道阶段失败

---

### 8.6 实现验证清单

#### 8.6.1 前置依赖检查（前5部分）

| 检查项 | 脚本位置 | 状态 |
|--------|---------|------|
| `STRATEGY_TYPE` 类属性 | `bimanual_edge_phone.py:523` | ✅ 已实现 |
| `EdgeOverhangCondition` 类 | `bimanual_edge_phone.py:48` | ✅ 已实现 |
| `StableGraspCondition` 类 | `bimanual_edge_phone.py:109` | ✅ 已实现 |
| `ClearPathCondition` 类 | `bimanual_edge_phone.py:159` | ✅ 已实现 |
| `PhasedSuccessEvaluator` 类 | `bimanual_edge_phone.py:445` | ✅ 已实现 |
| `_setup_phased_evaluator()` 方法 | `bimanual_edge_phone.py:573` | ✅ 已实现 |
| `evaluate_phase_and_get_labels()` 方法 | `bimanual_edge_phone.py:710` | ✅ 已实现 |
| `get_phase_progress()` 方法 | `bimanual_edge_phone.py:728` | ✅ 已实现 |

#### 8.6.2 第8部分修改检查

| 检查项 | 脚本位置 | 修改行号 | 状态 |
|--------|---------|---------|------|
| 导入 `Dict, Tuple` 类型 | `custom_rlbench_env.py:1` | 修改导入 | ✅ 已实现 |
| `get_phase_progress()` 接口 | `custom_rlbench_env.py:337` | 新增方法 | ✅ 已实现 |
| `evaluate_current_phase()` 接口 | `custom_rlbench_env.py:354` | 新增方法 | ✅ 已实现 |
| `get_strategy_and_phase()` 接口 | `custom_rlbench_env.py:371` | 新增方法 | ✅ 已实现 |
| 阶段统计初始化 | `_independent_env_runner.py:182` | 新增代码 | ✅ 已实现 |
| 每帧阶段评估 | `_independent_env_runner.py:246` | 新增代码 | ✅ 已实现 |
| Episode阶段统计 | `_independent_env_runner.py:278` | 新增代码 | ✅ 已实现 |
| 阶段指标添加 | `_independent_env_runner.py:386` | 新增代码 | ✅ 已实现 |

---

### 8.7 设计优势

由于数据收集和评估使用相同的 `PhasedSuccessEvaluator`：
1. **一致性**：训练数据标签与评估标准完全一致
2. **可复用**：任务类中定义一次，数据收集和评估都使用
3. **可解释性**：可以精确定位模型在哪个阶段失败
4. **最小改动**：评估框架只需调用任务类已有的接口

---

### 8.8 两层数据流分离（重要说明）

> 本节说明性能评估与模型运行的数据流分离，避免概念混淆。

#### 8.8.1 核心概念

`ACT_BC_ENC_STRATEGY` 模型涉及两个独立的数据流：

| 层面 | 数据来源 | 用途 | 代码位置 |
|------|---------|------|---------|
| **模型运行层面** | `StrategyPhasePredictor` 预测 | 条件注入，指导动作生成 | `detr_vae.py` |
| **性能评估层面** | 环境 `PhasedSuccessEvaluator` | 统计各阶段成功率 | `_independent_env_runner.py` |

#### 8.8.2 两层数据流完全独立

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 性能评估层面（本第8部分）                                                │
│                                                                         │
│   环境真实阶段 ──→ evaluate_current_phase() ──→ 统计 phase_success_rate │
│                                                                         │
│   用途：分析模型在哪个阶段失败（事后统计，不影响模型行为）               │
└─────────────────────────────────────────────────────────────────────────┘

                        ↕ 完全独立，互不影响

┌─────────────────────────────────────────────────────────────────────────┐
│ 模型运行层面                                                             │
│                                                                         │
│   视觉特征 ──→ StrategyPhasePredictor ──→ 预测 strategy/phase           │
│                        │                                                │
│                        ▼                                                │
│               条件注入到动作生成器 ──→ 生成动作                          │
│                                                                         │
│   用途：模型自主预测条件，端到端生成动作（不依赖环境提供条件）           │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 8.8.3 设计决策说明

**为什么不用环境真实阶段传给模型？**

1. **端到端能力测试**：模型需要自己判断"当前处于什么阶段"
2. **部署时无环境支持**：实际部署时没有环境提供阶段信息
3. **训练-推理一致性**：训练和推理使用相同的条件来源（预测值）

**为什么评估时用环境真实阶段？**

1. **公平评估**：用客观标准评估模型表现
2. **精确定位问题**：分析模型在哪个阶段失败
3. **不影响模型行为**：统计是事后进行的

#### 8.8.4 相关文档

- 模型运行层面的详细设计：[STRATEGY_MODEL_DESIGN.md 第6节](../06_misc/act_family/STRATEGY_MODEL_DESIGN.md#6-推理流程端到端设计)
- 条件预测器设计经验：[KEYPOINT_POSE_INJECTION_PLAN.md 6.6节](../06_misc/act_family/KEYPOINT_POSE_INJECTION_PLAN.md#66-条件预测与注入的经验教训来自-act_bc_enc_strategy)

---

**文档版本**: 1.8
**最后更新**: 2026-01-07

**1.8版本更新说明**:
- ✅ **修复 `evaluate_current_phase()` 阶段漏记录 bug**
  - 问题：原方法每次只评估一个阶段，如果多个阶段条件在同一帧满足，后续阶段不会被记录
  - 表现：`return` 与 `phase_4_success_rate` 不一致（如 return=20% 但 phase_4=10%）
  - 修复：改用 `while` 循环，一次调用中评估所有可完成的阶段
  - 修改文件：`bimanual_edge_phone.py` 第470-495行

**1.7版本更新说明**:
- ✅ 新增8.8节"两层数据流分离"
  - 说明性能评估层面 vs 模型运行层面的独立性
  - 解释为什么评估用环境真实阶段，模型用 Predictor 预测
  - 添加相关文档链接（STRATEGY_MODEL_DESIGN.md, KEYPOINT_POSE_INJECTION_PLAN.md）

**1.6版本更新说明（重要）**:
- ⚠️ **阶段条件设计从"累积条件"改为"非累积条件"**
  - 修复：EdgeOverhangCondition 在手机被抓起后失效导致 Phase 2-4 无法完成的问题
  - 修复：评估时 `return=40` 但 `phase_4_success_rate=0%` 的矛盾
  - Phase 1: `[EdgeOverhangCondition]` — 一次性
  - Phase 2-3: `[StableGraspCondition, ...]` — 持续检查抓取稳定性
  - Phase 4: `[LiftedCondition]` — 只检查高度（与任务成功条件一致）
- ⚠️ **影响已收集数据**：使用旧代码收集的数据中 `phase_type` 标签可能错误
  - 使用 `phase_type` 的模型（如 ACT_BC_ENC_STRATEGY）需要重新收集数据
  - 不使用 `phase_type` 的模型（如基础 ACT_BC_ENC）可继续使用

**1.5版本更新说明**:
- ✅ 更新第7.4部分实现验证清单状态（对比代码库逐项验证）
  - 7.4.1 任务类实现清单：全部11项已实现，添加具体行号
  - 7.4.2 Scene.py 修改清单：全部5项已实现，添加具体行号
  - 7.4.4 训练数据加载清单：全部2项已实现（act_bc_enc_strategy agent）
  - 7.4.3 TTT场景文件清单：保持待验证状态（需实际运行验证）

**1.4版本更新说明**:
- ✅ 实施第8部分所有修改，更新验证清单状态

**1.3版本更新说明**:
- 修复 8.3.2 节 `evaluate_current_phase()` 返回值处理 bug（之前错误使用 `current_phase - 1`）
- 修复 `custom_rlbench_env.py` 属性访问路径（`self._task_env` → `self._task`）
- 更新修改位置行号参考（第217行 → 第238行）
- 添加必要的导入语句说明（`Dict`, `Tuple`）
- 修正数据流图中的方法名
