# RLBench2 场景初始化机制详解

本文档以 `bimanual_edge_laptop` 任务为切入点，系统阐述 RLBench2 的场景初始化机制。

---

## 1. 核心概念：Episode 与 Variation

### 1.1 Variation（变体）
- **定义**: 同一任务的不同配置版本，通常用于改变**语义目标**（如目标颜色、目标物体类型）
- **由 `variation_count()` 定义**: 返回该任务支持的变体数量
- **触发方式**: `init_episode(index)` 的 `index` 参数指定使用哪个变体
- **典型用途**:
  - 改变目标物体的颜色（如 "pick up the **red** block"）
  - 改变目标物体类型（如 9 种不同的杂货）
  - 改变干扰物的配置

```python
# coordinated_close_jar.py 示例
def variation_count(self) -> int:
    return len(colors)  # 20种颜色 = 20个variation

def init_episode(self, index: int) -> List[str]:
    target_color_name, target_color_rgb = colors[index]
    self.jars[index % 2].set_color(target_color_rgb)
    return ['close the %s jar' % target_color_name]
```

### 1.2 Episode（回合）
- **定义**: 单次任务执行的完整流程，从场景重置到任务完成/失败
- **同一 variation 可生成多个 episode**: 每个 episode 的物体位置、姿态可能不同
- **初始化入口**: `Scene.init_episode(index, randomly_place=True)`

---

## 2. 初始化调用链路

```
Scene.init_episode(index)
    │
    ├─→ Task.init_episode(index)          # 任务级初始化（物体采样、条件设置）
    │
    ├─→ Scene._place_task()               # 场景级随机化（整个任务在workspace中的位置）
    │       └─→ SpawnBoundary.sample()
    │
    └─→ Task.validate()                   # 验证 waypoint 可达性
            └─→ Task._get_waypoints()
```

---

## 3. 两层随机化机制

### 3.1 第一层：任务整体位置随机化（Scene级）

**关键函数**: `Scene._place_task()` (scene.py:818-826)

```python
def _place_task(self) -> None:
    self._workspace_boundary.clear()
    self.task.boundary_root().set_orientation(self._initial_task_pose)
    min_rot, max_rot = self.task.base_rotation_bounds()
    self._workspace_boundary.sample(
        self.task.boundary_root(),
        min_rotation=min_rot, max_rotation=max_rot)
```

**作用**: 将整个任务（TTM模型）随机放置在机器人工作空间内

---

#### 3.1.1 `_place_task()` 详解

**执行流程**:

```
_place_task()
    │
    ├─① self._workspace_boundary.clear()
    │      清空已放置物体记录，准备新的采样
    │
    ├─② self.task.boundary_root().set_orientation(self._initial_task_pose)
    │      恢复任务根对象的初始朝向（避免累积旋转）
    │
    └─③ self._workspace_boundary.sample(boundary_root, min_rot, max_rot)
           对任务根对象进行位置和旋转采样
```

**核心对象说明**:

| 对象 | 创建位置 | 含义 |
|------|---------|------|
| `_workspace_boundary` | Scene.__init__ | `SpawnBoundary([Shape('workspace')])` - 基于场景中的 `workspace` Shape 创建的采样边界 |
| `boundary_root()` | Task 方法 | 默认返回 `self.get_base()`，即任务的根 Dummy 对象 |
| `_initial_task_pose` | Scene.load() | 任务加载时记录的初始朝向 |

---

#### 3.1.2 SpawnBoundary.sample() 采样过程

当对任务根对象调用 `sample()` 时，执行以下步骤：

```python
# spawn_boundary.py - BoundaryObject.add() 核心逻辑
def add(self, obj, min_rotation, max_rotation, ...):
    # 1. 获取物体的包围盒（相对于物体原点的偏移）
    obj_bbox = BoundingBox(*obj.get_model_bounding_box())

    # 2. 随机生成旋转角度 (rx, ry, rz)
    rotation = np.random.uniform(min_rotation, max_rotation)
    obj_bbox = obj_bbox.rotate(rotation)  # 旋转包围盒

    # 3. 检查旋转后的包围盒尺寸是否仍能放入边界内
    if not obj_bbox.within_boundary(self._boundary_bbox):
        return -1  # 失败，重试

    # 4. 在收缩后的范围内随机采样位置
    #    收缩量 = 物体包围盒的半径，确保物体整体不超出边界
    x = np.random.uniform(
        boundary.min_x + |obj_bbox.min_x|,   # 左边界 + 物体左半宽
        boundary.max_x - |obj_bbox.max_x|)   # 右边界 - 物体右半宽
    y = np.random.uniform(
        boundary.min_y + |obj_bbox.min_y|,   # 同理
        boundary.max_y - |obj_bbox.max_y|)
    z = obj.get_position()[2]  # 平面采样时保持z不变

    # 5. 应用新位置和旋转
    obj.set_position([x, y, z], relative_to=boundary)
    obj.rotate(rotation)

    return 1  # 成功
```

**采样维度**:
- **位置**: X 和 Y 轴在**收缩后的**边界内均匀随机采样（保证物体整体不超出）
- **旋转**: 绕 Z 轴在 `[min_rot[2], max_rot[2]]` 范围内均匀随机采样
- **高度**: 通常保持不变（workspace 是平面时）

---

#### 3.1.3 为什么电脑和盒子会一起移动？

**关键机制：CoppeliaSim 的父子层级结构**

TTM 文件中的场景层级结构（以 bimanual_edge_laptop 为例）:

```
bimanual_edge_laptop (Dummy - 任务根对象)
    ├── base (Shape - 笔记本电脑)
    ├── table_surface (Shape - 桌子/盒子)
    ├── waypoint0 (Dummy)
    ├── waypoint1 (Dummy)
    ├── ...
    └── 其他子对象
```

**当 `_place_task()` 移动任务根对象时**:
1. `boundary_root()` 返回的是任务根 Dummy（名为 `bimanual_edge_laptop`）
2. 调用 `obj.set_position()` 和 `obj.rotate()` 会移动这个根对象
3. **CoppeliaSim 中，子对象相对于父对象的位姿是固定的**
4. 因此，所有子对象（电脑、盒子、waypoint等）会**作为刚体整体**跟随根对象移动

**这就是为什么 bimanual_edge_laptop 任务虽然没有 Task 级物体采样代码，但电脑和盒子的位置仍然会在不同 episode 间变化的原因！**

---

#### 3.1.4 可配置项

| 方法 | 作用 | 默认值 |
|------|------|--------|
| `base_rotation_bounds()` | 控制任务绕Z轴的旋转范围 | `(-π, π)` 即360°自由旋转 |
| `boundary_root()` | 指定作为放置参考的根对象 | 任务基座 |
| `is_static_workspace()` | 是否禁用位置随机化 | `False` |

```python
# bimanual_handover_item.py 示例
def is_static_workspace(self):
    return True  # 禁用整体位置随机化

def base_rotation_bounds(self):
    return [0, 0, -np.pi/8], [0, 0, np.pi/8]  # 仅允许小幅旋转
```

---

#### 3.1.5 workspace 边界的定义

`workspace` 是场景主文件（如 `task_design_bimanual.ttt`）中预定义的 Shape 对象：

```python
# Scene.__init__() 中
self._workspace = Shape('workspace')  # 获取场景中名为 'workspace' 的物体
self._workspace_boundary = SpawnBoundary([self._workspace])  # 用其边界创建采样器
```

workspace 通常是一个扁平的长方体，定义了机器人可操作的桌面区域。任务模型会被随机放置在这个区域内。

---

#### 3.1.6 关于 workspace 和采样边界的常见问题

**Q1: workspace 本身是固定的吗？会被随机化吗？**

**是的，workspace 是完全固定的，不存在位姿上的随机性。**

- `workspace` 是 CoppeliaSim 场景文件（`.ttt`）中预先放置好的一个 Shape 对象
- 它只是**定义采样边界的参考物体**，本身不会被移动或旋转
- `SpawnBoundary` 只是读取它的包围盒来确定采样范围

```python
# SpawnBoundary 初始化时只读取边界信息，不修改 workspace
class BoundaryObject:
    def __init__(self, boundary: Object):
        # 获取 workspace 的包围盒作为采样边界
        minx, maxx, miny, maxy, minz, maxz = boundary.get_bounding_box()
        self._boundary_bbox = BoundingBox(minx, maxx, miny, maxy, minz, maxz)
        # workspace 本身不会被修改
```

---

**Q2: 采样时物体会超出 workspace 边界吗？**

**不会！代码设计上保证了物体的整个包围盒都在边界内。**

采样位置时，会根据物体的包围盒尺寸**收缩采样范围**：

```python
# spawn_boundary.py - _get_position_within_boundary()
def _get_position_within_boundary(self, obj, obj_bbox):
    # 采样范围 = 边界范围 - 物体包围盒的半径
    x = np.random.uniform(
        self._boundary_bbox.min_x + np.abs(obj_bbox.min_x),  # 左边界 + 物体左半宽
        self._boundary_bbox.max_x - np.abs(obj_bbox.max_x))  # 右边界 - 物体右半宽
    y = np.random.uniform(
        self._boundary_bbox.min_y + np.abs(obj_bbox.min_y),  # 下边界 + 物体下半高
        self._boundary_bbox.max_y - np.abs(obj_bbox.max_y))  # 上边界 - 物体上半高
    ...
```

**图示说明**：

```
workspace 边界
┌─────────────────────────────────┐
│                                 │
│   有效采样范围（收缩后）          │
│   ┌───────────────────────┐     │
│   │                       │     │
│   │    物体原点可以出现    │     │
│   │    在这个范围内        │     │
│   │                       │     │
│   └───────────────────────┘     │
│         ↑                       │
│    收缩距离 = 物体包围盒半径      │
└─────────────────────────────────┘
```

**关键逻辑**：
- `obj_bbox.min_x` 和 `obj_bbox.max_x` 是物体包围盒相对于物体原点的偏移
- 采样时从边界两侧各减去物体包围盒的对应尺寸
- 这样物体原点采样后，其包围盒边缘正好不会超出边界

**注意**：这里假设物体的包围盒信息是准确的。如果物体形状不规则或包围盒计算有误，理论上仍可能有微小超出。

---

**Q3: 如果物体太大，大于 workspace 会怎样？**

采样会失败并抛出 `BoundaryError` 异常：

```python
# 如果采样范围变成负数（物体比边界还大），无法采样
if obj_bbox.max_x - obj_bbox.min_x > boundary_bbox.max_x - boundary_bbox.min_x:
    # 采样范围为空，会不断失败直到超过 MAX_SAMPLES=100 次
    raise BoundaryError('Could not place within boundary. '
                        'Perhaps the object is too big for it?')
```

### 3.2 第二层：任务内部物体位置随机化（Task级）

**关键类**: `SpawnBoundary` (spawn_boundary.py)

**核心方法**: `SpawnBoundary.sample(obj, ...)`
- 在指定边界区域内随机采样物体位置和旋转
- 自动处理碰撞检测，避免物体重叠
- 最多尝试 `MAX_SAMPLES=100` 次

```python
# coordinated_close_jar.py 示例
def init_episode(self, index: int) -> List[str]:
    b = SpawnBoundary([self.boundary])  # 创建采样边界
    for obj in self.jars:
        b.sample(obj, min_distance=0.01)  # 在边界内随机放置
```

**SpawnBoundary.sample() 参数**:
| 参数 | 说明 |
|------|------|
| `obj` | 要放置的物体 |
| `ignore_collisions` | 是否忽略碰撞检测 |
| `min_rotation` / `max_rotation` | 旋转范围 (rx, ry, rz) |
| `min_distance` | 与其他物体的最小距离 |

---

## 4. bimanual_edge_laptop 的初始化分析

### 4.1 任务代码结构

```python
class BimanualEdgeLaptop(BimanualTask):

    def init_task(self) -> None:
        laptop = Shape('base')
        self.register_success_conditions([LiftedCondition(laptop, 1.0)])
        self.register_graspable_objects([laptop])
        # waypoint映射设置...

    def init_episode(self, index: int) -> List[str]:
        return ['pick up the laptop']  # 无任务内部随机化

    def variation_count(self) -> int:
        return 1  # 只有一个变体
```

### 4.2 随机化来源

**该任务没有任务内部的物体随机化**！

其位置随机化**完全依赖 Scene 级的 `_place_task()`**:
- 整个笔记本电脑模型（包含在 `bimanual_edge_laptop.ttm` 中）会被随机放置在 workspace 中
- 使用默认的 `base_rotation_bounds()`: Z轴 ±π 旋转

### 4.3 根对象位姿随机化的具体细节

**bimanual_edge_laptop 任务的根对象（Dummy）位姿随机性如下：**

| 随机化维度 | 范围 | 说明 |
|-----------|------|------|
| **X 位置** | workspace 边界内（收缩后） | 均匀随机采样 |
| **Y 位置** | workspace 边界内（收缩后） | 均匀随机采样 |
| **Z 位置** | 保持不变 | workspace 是平面，Z 固定 |
| **X 旋转** | 0 | 不旋转 |
| **Y 旋转** | 0 | 不旋转 |
| **Z 旋转** | **[-π, +π]** (±180°) | 均匀随机采样，**使用默认值** |

**为什么是这样的配置？**

```python
# bimanual_edge_laptop.py 中没有重写以下方法，因此使用 Task 基类的默认值：

# 默认旋转范围（task.py:134-143）
def base_rotation_bounds(self):
    return (0.0, 0.0, -3.14), (0.0, 0.0, 3.14)  # Z轴 ±π

# 默认不禁用位置随机化（task.py:167-172）
def is_static_workspace(self) -> bool:
    return False  # 允许位置随机化
```

**图示说明**：

```
俯视图 (XY平面)

        Y
        ↑
        │    workspace 边界
        │  ┌─────────────────────┐
        │  │                     │
        │  │   ┌───┐  ←任务模型  │
        │  │   │ ↻ │  可在此范围内│
        │  │   └───┘  随机平移+旋转│
        │  │                     │
        │  └─────────────────────┘
        └──────────────────────→ X

旋转范围: θz ∈ [-π, +π]  (360°任意方向)
```

**实际效果**：
- 每个 episode，笔记本电脑+盒子的整体会出现在 workspace 内的**随机位置**
- 同时绕 Z 轴有**任意角度的随机旋转**
- 笔记本电脑与盒子的**相对位置始终固定**（由 TTM 文件定义）

---

### 4.4 类似任务的随机化配置对比

以下任务都属于"无 Task 级物体采样"类型，随机化完全由 Scene 级控制：

| 任务 | `base_rotation_bounds()` | `is_static_workspace()` | 随机化效果 |
|------|-------------------------|------------------------|-----------|
| **bimanual_edge_laptop** | 默认 `[-π, +π]` | 默认 `False` | XY位置随机 + Z轴360°旋转 |
| **bimanual_edge_phone** | 默认 `[-π, +π]` | 默认 `False` | XY位置随机 + Z轴360°旋转 |
| **bimanual_pivot_laptop** | 默认 `[-π, +π]` | 默认 `False` | XY位置随机 + Z轴360°旋转 |
| **bimanual_pivot_phone** | 默认 `[-π, +π]` | 默认 `False` | XY位置随机 + Z轴360°旋转 |
| **bimanual_pick_fork** | **`[0, 0, 0]`** | 默认 `False` | **仅XY位置随机，无旋转** |

---

#### 4.4.1 bimanual_edge_laptop / bimanual_edge_phone / bimanual_pivot_laptop / bimanual_pivot_phone

这四个任务的随机化配置**完全相同**：

```python
# 这些任务都没有重写以下方法，全部使用默认值

class BimanualEdgeLaptop(BimanualTask):  # 以及其他三个
    def init_episode(self, index: int) -> List[str]:
        return ['pick up the laptop']  # 无物体采样代码

    # 未重写 base_rotation_bounds() → 使用默认 [-π, +π]
    # 未重写 is_static_workspace() → 使用默认 False
```

**随机化效果**：
- ✅ X 位置：随机
- ✅ Y 位置：随机
- ❌ Z 位置：固定（workspace 平面高度）
- ❌ X 旋转：固定为 0
- ❌ Y 旋转：固定为 0
- ✅ Z 旋转：**[-π, +π] 范围内随机**

---

#### 4.4.2 bimanual_pick_fork（特殊配置）

该任务**显式禁用了旋转随机化**：

```python
class BimanualPickFork(BimanualTask):

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]  # 完全禁用旋转！

    def init_episode(self, index: int) -> List[str]:
        # 状态重置，但无物体位置采样
        self.lifted_condition.reset()
        return ['pick up the fork']
```

**随机化效果**：
- ✅ X 位置：随机
- ✅ Y 位置：随机
- ❌ Z 位置：固定
- ❌ X 旋转：固定为 0
- ❌ Y 旋转：固定为 0
- ❌ Z 旋转：**固定为 0（禁用旋转）**

**为什么禁用旋转？**
可能是因为叉子任务对机械臂的协调要求较高，固定朝向可以简化 waypoint 设计。

---

### 4.5 TTM 文件的作用

- `.ttm` 是 CoppeliaSim 的模型文件，包含:
  - 物体几何形状和初始相对位置
  - Waypoint dummy 对象（waypoint0, waypoint1, ...）
  - 边界区域定义（如 spawn_boundary）
  - 成功检测传感器等

---

## 5. 随机化内容总结

| 随机化维度 | 实现位置 | 典型方法 |
|-----------|---------|---------|
| 任务整体位置 | `Scene._place_task()` | `SpawnBoundary.sample()` |
| 任务整体旋转 | `Scene._place_task()` | `base_rotation_bounds()` |
| 物体内部位置 | `Task.init_episode()` | `SpawnBoundary.sample()` |
| 物体颜色 | `Task.init_episode()` | `Shape.set_color()` |
| Waypoint位置 | `Task.init_episode()` | `Dummy.set_position(relative_to=obj)` |
| 目标选择 | `Task.init_episode()` | `index` 参数 |

---

## 6. 不同任务的初始化模式对比

### 6.1 无内部随机化（如 bimanual_edge_laptop）
```python
def init_episode(self, index):
    return ['pick up the laptop']  # 仅依赖Scene级随机化
```

### 6.2 物体位置随机化（如 coordinated_close_jar）
```python
def init_episode(self, index):
    b = SpawnBoundary([self.boundary])
    for obj in self.jars:
        b.sample(obj, min_distance=0.01)
```

### 6.3 颜色+位置随机化（如 bimanual_handover_item）
```python
def init_episode(self, index):
    color_name, color = colors[index]
    self.items[0].set_color(color)

    b = SpawnBoundary([self.boundaries])
    for item in self.items:
        b.sample(item, min_distance=0.05)
```

### 6.4 Waypoint动态调整（如 put_groceries_in_cupboard）
```python
def init_episode(self, index):
    self.boundary.clear()
    [self.boundary.sample(g) for g in self.groceries]
    # 根据目标物体位置调整waypoint
    self.waypoint1.set_pose(self.grasp_points[index].get_pose())
```

---

## 7. 关键类/函数速查

| 类/函数 | 文件 | 职责 |
|--------|------|------|
| `Scene.init_episode()` | backend/scene.py | 初始化入口 |
| `Scene._place_task()` | backend/scene.py | 任务整体位置随机化 |
| `Task.init_episode()` | backend/task.py | 任务级初始化（由子类实现） |
| `Task.base_rotation_bounds()` | backend/task.py | 定义旋转范围 |
| `Task.is_static_workspace()` | backend/task.py | 是否禁用位置随机化 |
| `Task.boundary_root()` | backend/task.py | 放置参考根对象 |
| `SpawnBoundary` | backend/spawn_boundary.py | 物体位置采样 |
| `SpawnBoundary.sample()` | backend/spawn_boundary.py | 执行采样 |

---

## 8. 设计要点

1. **两层解耦**: Scene负责全局放置，Task负责内部配置
2. **重试机制**: `init_episode` 失败时会恢复初始状态重试（最多 `max_attempts=5`）
3. **Waypoint验证**: 初始化后会验证所有waypoint的可达性
4. **稳定化步骤**: 初始化完成后执行若干物理仿真步让物体稳定

```python
# Scene.init_episode 中的重试逻辑
while attempts < max_attempts:
    descriptions = self.task.init_episode(index)
    try:
        if randomly_place:
            self._place_task()
        self.task.validate()  # 验证waypoint可达
        break
    except (BoundaryError, WaypointError):
        self.task.restore_state(self._initial_task_state)
        attempts += 1
```
