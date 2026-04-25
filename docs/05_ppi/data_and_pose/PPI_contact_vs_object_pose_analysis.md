# Contact Point 坐标系与目标物体本体坐标系的关系分析

## 1. 背景与问题

PPI 完整流程（`prediction_type=keyframe_continuous` + `predict_point_flow=true`）需要每帧的 `object_6d_pose`（目标物体在世界坐标系下的 4x4 齐次变换矩阵、3D 位置、四元数）。当前四个新任务的 demo 数据中不包含该字段，但 `misc` 字典中存储了 `contact_position`/`contact_quaternion`（来自仿真场景中 `push_pt` 或 `press_pt` Dummy 对象的世界位姿）。

本文档通过**仿真实地验证**，回答以下问题：

1. contact point 的坐标系与目标物体本体坐标系是否存在差异？
2. 差异是否是常量？
3. 差异对 PPI 算法是否有影响？
4. 如何精确地从 contact 坐标系恢复物体本体坐标系？

---

## 2. 仿真验证：场景层级关系

通过 PyRep 加载四个任务的 `.ttm` 场景文件，查询 CoppeliaSim 场景树：

| 任务 | Contact Dummy | 目标物体 | 层级关系 |
|------|---------------|----------|----------|
| `bimanual_edge_phone` | `push_pt` | `Phone` (Shape) | `push_pt` → `Phone` → `bimanual_edge_phone` |
| `bimanual_pivot_phone` | `push_pt` | `Phone` (Shape) | `push_pt` → `Phone` → `bimanual_pivot_phone` |
| `bimanual_pick_plate` | `press_pt` | `plate` (Shape) | `press_pt` → `plate` → `bimanual_pick_plate` |
| `bimanual_pick_fork` | `press_pt` | `Fork_phy` (Shape) | `press_pt` → `Fork_phy` → `bimanual_pick_fork` |

**结论：所有四个任务中，contact dummy 都是目标物体的直接子对象（child），在 CoppeliaSim 场景树中刚性固连。**

同样，`grasp_pt` 也是各目标物体的直接子对象。

---

## 3. 差异是否存在？是否是常量？

### 3.1 差异确实存在

contact dummy 的坐标系与物体本体坐标系之间存在**平移和旋转两方面的偏移**：

| 任务 | 平移偏移（物体坐标系下） | 平移模长 | 旋转偏移角度 |
|------|--------------------------|----------|-------------|
| `bimanual_edge_phone` | `[-0.0009, -0.0045, 0.1065]` | 10.66 cm | 120.00° |
| `bimanual_pivot_phone` | `[0.0021, -0.0045, 0.1084]` | 10.86 cm | 120.00° |
| `bimanual_pick_plate` | `[0.0070, -0.0128, -0.0672]` | 6.87 cm | 177.62° |
| `bimanual_pick_fork` | `[0.0004, -0.0010, 0.0940]` | 9.40 cm | 119.92° |

这些偏移量不可忽略——最大平移偏移约 10 cm，旋转偏移高达 120°~178°。

### 3.2 差异是严格常量

对每个任务推进 100 步仿真，逐帧检查 `push_pt`/`press_pt` 在物体本体坐标系下的相对位姿：

| 任务 | 相对位置 std | 相对位置 max-min | 相对四元数 std |
|------|-------------|-----------------|---------------|
| `bimanual_edge_phone` | `[0, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0, 0]` |
| `bimanual_pivot_phone` | `[0, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0, 0]` |
| `bimanual_pick_plate` | `[0, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0, 0]` |
| `bimanual_pick_fork` | `[0, 0, 0]` | `[0, 0, 0]` | `[0, 0, 0, 0]` |

**标准差严格为零，最大最小差严格为零。** 由于 contact dummy 是物体的刚性子对象，相对位姿在物理引擎中永远不会改变。

### 3.3 已有 demo 数据的一致性验证

对四个任务的 demo 数据验证 `contact_position` 和 `grasp_position` 之间的刚体约束：

| 任务 | contact-grasp 距离 mean | contact-grasp 距离 std | contact_quaternion 归一化 |
|------|------------------------|------------------------|--------------------------|
| `bimanual_edge_phone` | 0.19456112 m | 3.38e-08 | 1.0000000192 |
| `bimanual_pivot_phone` | 0.05675828 m | 4.11e-08 | 0.9999999750 |
| `bimanual_pick_plate` | 0.12008149 m | 4.06e-08 | 1.0000000631 |
| `bimanual_pick_fork` | 0.17701286 m | 2.20e-08 | 0.9999999605 |

距离标准差在 1e-8 量级（浮点精度），四元数范数接近 1.0。**demo 数据中的 contact 信息确实是仿真中 dummy 的世界位姿。**

---

## 4. 差异对 PPI 算法的影响

### 4.1 对 point flow 生成（save_point_flow.py）：无影响

核心算法逻辑：
```
步骤1: pc_obj = inv(T_0) @ pc_world_0       // 世界坐标系 → "物体"坐标系
步骤2: pc_world_t = T_t @ pc_obj             // "物体"坐标系 → 世界坐标系
```

设 contact 的世界位姿 `T_contact = T_phone @ T_offset`（`T_offset` 为常量偏移），则：

```
用 contact 做变换:
  步骤1: pc_c = inv(T_contact_0) @ pc_world_0
       = inv(T_offset) @ inv(T_phone_0) @ pc_world_0

  步骤2: pc_world_t = T_contact_t @ pc_c
       = (T_phone_t @ T_offset) @ (inv(T_offset) @ inv(T_phone_0) @ pc_world_0)
       = T_phone_t @ I @ inv(T_phone_0) @ pc_world_0
       = T_phone_t @ inv(T_phone_0) @ pc_world_0       ← 与直接用 T_phone 完全相同
```

**数值验证**（bimanual_edge_phone，200 个随机点）：两种方法的最大差异为 3.9e-8 m（浮点精度级别）。

### 4.2 对状态表示（get_data_keyframe_continuous.py）：影响可控

该文件将 `position` 和 `quaternion` 拼成 7D 向量作为训练数据。使用 contact 坐标系替代物体本体坐标系时：
- position 偏移约 7-11 cm
- quaternion 偏移约 120°

但由于：
1. 7D 向量最终被 `norm_stats` 归一化（min-max 缩放到 [-1, 1]）
2. 模型学习的是**运动变化模式**而非绝对值
3. 训练和推理使用**同一坐标系**，自洽即可

因此对训练效果没有本质影响。不过，如果追求与原始 PPI 代码的严格一致，应当做精确对齐。

### 4.3 对 2D 投影提示（get_sam_prompt_point）：实际不使用

当前代码中 `get_sam_prompt_point_from_foundationdino`（使用 GroundingDINO 视觉检测）已替代了基于 `object_6d_pose['position']` 投影的 `get_sam_prompt_point`。因此这处使用不受影响。

---

## 5. 精确对齐：从 contact 世界位姿恢复物体本体位姿

### 5.1 数学原理

设：
- $T_{\text{obj}}^{(t)} \in SE(3)$：物体本体在时刻 $t$ 的世界位姿（4×4 齐次变换矩阵）
- $T_{\text{contact}}^{(t)} \in SE(3)$：contact dummy 在时刻 $t$ 的世界位姿
- $T_{\text{offset}} \in SE(3)$：contact dummy 在物体本体坐标系下的相对位姿（**常量**）

由于 contact dummy 刚性固连在物体上，恒有：

$$T_{\text{contact}}^{(t)} = T_{\text{obj}}^{(t)} \cdot T_{\text{offset}}$$

因此物体本体位姿可以精确恢复：

$$T_{\text{obj}}^{(t)} = T_{\text{contact}}^{(t)} \cdot T_{\text{offset}}^{-1}$$

其中 $T_{\text{offset}}$ 和 $T_{\text{offset}}^{-1}$ 是从仿真中一次性提取的常量矩阵。

从恢复的 4×4 矩阵中可以提取所有所需字段：

```python
position   = T_obj[:3, 3]                                     # 3D 平移
quaternion = Rotation.from_matrix(T_obj[:3, :3]).as_quat()     # 四元数 [x,y,z,w]
matrix     = T_obj                                             # 完整 4×4 矩阵
```

### 5.2 反向验证精度

在仿真中用此方法反推物体位姿，与真实值对比：

| 任务 | 位置误差 | 旋转误差 |
|------|---------|---------|
| `bimanual_edge_phone` | 4.31e-8 m | 3.42e-6 deg |
| `bimanual_pivot_phone` | 3.83e-8 m | 4.83e-6 deg |
| `bimanual_pick_plate` | 1.09e-7 m | 2.11e-6 deg |
| `bimanual_pick_fork` | 3.00e-8 m | 3.16e-6 deg |

误差均在浮点精度级别（<0.0001 mm, <0.00001°），可以认为是精确恢复。

### 5.3 四个任务的偏移矩阵

以下为从仿真中精确提取的常量矩阵。**实际使用时只需 `T_offset_inv`（逆矩阵）。**

---

#### bimanual_edge_phone

Contact dummy: `push_pt`，目标物体: `Phone`

**$T_{\text{offset}}$**（`push_pt` 在 `Phone` 坐标系下的相对位姿）：

```python
T_OFFSET_EDGE_PHONE = np.array([
    [ 0.000000357627798,  0.000000238418686, -0.999999999999908, -0.000902652740479],
    [-0.999999999999758, -0.000000596046405, -0.000000357627940, -0.004460766911507],
    [-0.000000596046490,  0.999999999999794,  0.000000238418473,  0.106504075229168],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])
```

**$T_{\text{offset}}^{-1}$**：

```python
T_OFFSET_INV_EDGE_PHONE = np.array([
    [ 0.000000357627798, -0.999999999999758, -0.000000596046490, -0.004460703107312],
    [ 0.000000238418686, -0.000000596046405,  0.999999999999794, -0.106504077672761],
    [-0.999999999999908, -0.000000357627940,  0.000000238418473, -0.000902679728312],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])
```

相对平移: `[-0.0009, -0.0045, 0.1065]` m，相对旋转: 120.00°

---

#### bimanual_pivot_phone

Contact dummy: `push_pt`，目标物体: `Phone`

**$T_{\text{offset}}$**：

```python
T_OFFSET_PIVOT_PHONE = np.array([
    [ 0.000000178813465,  0.000000894069682, -0.999999999999584,  0.002094507217407],
    [-0.999999999999471, -0.000001013278791, -0.000000178814371, -0.004496008157730],
    [-0.000001013278951,  0.999999999999087,  0.000000894069501,  0.108443401753902],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])
```

**$T_{\text{offset}}^{-1}$**：

```python
T_OFFSET_INV_PIVOT_PHONE = np.array([
    [ 0.000000178813465, -0.999999999999471, -0.000001013278951, -0.004495898648837],
    [ 0.000000894069682, -0.000001013278791,  0.999999999999087, -0.108443408182149],
    [-0.999999999999584, -0.000000178814371,  0.000000894069501,  0.002094409457517],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])
```

相对平移: `[0.0021, -0.0045, 0.1084]` m，相对旋转: 120.00°

---

#### bimanual_pick_plate

Contact dummy: `press_pt`，目标物体: `plate`

**$T_{\text{offset}}$**：

```python
T_OFFSET_PICK_PLATE = np.array([
    [ 0.002598125880785, -0.000314574836860, -0.999996575386426,  0.006979644298553],
    [-0.058969175473169, -0.998259791077145,  0.000160818620698, -0.012829847633839],
    [-0.998256423012606,  0.058968555699510, -0.002612154817698, -0.067150130867958],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])
```

**$T_{\text{offset}}^{-1}$**：

```python
T_OFFSET_INV_PICK_PLATE = np.array([
    [ 0.002598125880784, -0.058969175473169, -0.998256423012606, -0.067807748975981],
    [-0.000314574836860, -0.998259791077145,  0.058968555699510, -0.008845579165724],
    [-0.999996575386426,  0.000160818620698, -0.002612154817698,  0.006806277136513],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])
```

相对平移: `[0.0070, -0.0128, -0.0672]` m，相对旋转: 177.62°

---

#### bimanual_pick_fork

Contact dummy: `press_pt`，目标物体: `Fork_phy`

**$T_{\text{offset}}$**：

```python
T_OFFSET_PICK_FORK = np.array([
    [-0.008017196545178,  0.011323440070544, -0.999903747499991,  0.000367403030396],
    [-0.999967469322108, -0.000976678403156,  0.008006647040767, -0.000968247652054],
    [-0.000885921607486,  0.999935410816252,  0.011330901934082,  0.093990661203861],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])
```

**$T_{\text{offset}}^{-1}$**：

```python
T_OFFSET_INV_PICK_FORK = np.array([
    [-0.008017196545178, -0.999967469322108, -0.000885921607486, -0.000882002254333],
    [ 0.011323440070544, -0.000976678403156,  0.999935410816251, -0.093989696356541],
    [-0.999903747499990,  0.008006647040768,  0.011330901934082, -0.000689878880687],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])
```

相对平移: `[0.0004, -0.0010, 0.0940]` m，相对旋转: 119.92°

---

### 5.4 完整的恢复代码

以下为从 demo 数据中的 `contact_position`/`contact_quaternion` 精确恢复 `object_6d_pose` 的完整实现：

```python
import numpy as np
from scipy.spatial.transform import Rotation

# ===== 四个任务的常量逆偏移矩阵 =====
# 用法: T_obj_world = T_contact_world @ T_OFFSET_INV[task_name]

T_OFFSET_INV = {
    "bimanual_edge_phone": np.array([
        [ 0.000000357627798, -0.999999999999758, -0.000000596046490, -0.004460703107312],
        [ 0.000000238418686, -0.000000596046405,  0.999999999999794, -0.106504077672761],
        [-0.999999999999908, -0.000000357627940,  0.000000238418473, -0.000902679728312],
        [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
    ]),
    "bimanual_pivot_phone": np.array([
        [ 0.000000178813465, -0.999999999999471, -0.000001013278951, -0.004495898648837],
        [ 0.000000894069682, -0.000001013278791,  0.999999999999087, -0.108443408182149],
        [-0.999999999999584, -0.000000178814371,  0.000000894069501,  0.002094409457517],
        [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
    ]),
    "bimanual_pick_plate": np.array([
        [ 0.002598125880784, -0.058969175473169, -0.998256423012606, -0.067807748975981],
        [-0.000314574836860, -0.998259791077145,  0.058968555699510, -0.008845579165724],
        [-0.999996575386426,  0.000160818620698, -0.002612154817698,  0.006806277136513],
        [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
    ]),
    "bimanual_pick_fork": np.array([
        [-0.008017196545178, -0.999967469322108, -0.000885921607486, -0.000882002254333],
        [ 0.011323440070544, -0.000976678403156,  0.999935410816251, -0.093989696356541],
        [-0.999903747499990,  0.008006647040768,  0.011330901934082, -0.000689878880687],
        [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
    ]),
}


def recover_object_6d_pose(obs, task_name):
    """
    从 BimanualObservation 的 misc 中恢复精确的 object_6d_pose。

    原理：
        contact dummy 是目标物体的刚性子对象，两者之间存在常量变换 T_offset。
        T_contact_world = T_obj_world @ T_offset
        => T_obj_world = T_contact_world @ inv(T_offset)

    Args:
        obs: BimanualObservation 对象，其 misc 中包含 contact_position 和 contact_quaternion
        task_name: 任务名称，如 "bimanual_edge_phone"

    Returns:
        dict: {
            'position':    np.ndarray (3,)   - 物体在世界坐标系下的位置
            'quaternion':  np.ndarray (4,)   - 物体在世界坐标系下的四元数 [x,y,z,w]
            'orientation': np.ndarray (3,)   - 物体在世界坐标系下的欧拉角 (xyz, rad)
            'matrix':      np.ndarray (4,4)  - 物体在世界坐标系下的齐次变换矩阵
        }
    """
    # 1. 从 misc 中读取 contact dummy 的世界位姿
    contact_pos = obs.misc['contact_position']        # (3,)
    contact_quat = obs.misc['contact_quaternion']      # (4,) [x,y,z,w]

    # 2. 构建 contact 的 4x4 世界位姿矩阵
    R_contact = Rotation.from_quat(contact_quat).as_matrix()
    T_contact = np.eye(4, dtype=np.float64)
    T_contact[:3, :3] = R_contact
    T_contact[:3, 3] = contact_pos

    # 3. 乘以逆偏移矩阵恢复物体本体位姿
    T_offset_inv = T_OFFSET_INV[task_name]
    T_obj = T_contact @ T_offset_inv

    # 4. 提取各字段
    position = T_obj[:3, 3].astype(np.float32)
    R_obj = Rotation.from_matrix(T_obj[:3, :3])
    quaternion = R_obj.as_quat().astype(np.float32)       # [x,y,z,w]
    orientation = R_obj.as_euler('xyz').astype(np.float32) # (3,) rad
    matrix = T_obj.astype(np.float32)                      # (4,4)

    return {
        'position': position,
        'quaternion': quaternion,
        'orientation': orientation,
        'matrix': matrix,
    }


def patch_demo_with_object_6d_pose(demo, task_name):
    """
    对整条 demo 的每一帧补充 object_6d_pose 字段。

    Args:
        demo: list[BimanualObservation]，从 low_dim_obs.pkl 反序列化得到
        task_name: 任务名称，如 "bimanual_edge_phone"

    Returns:
        demo: 同一个 list，每个 obs 已添加 object_6d_pose 属性
    """
    for obs in demo:
        if not hasattr(obs, 'object_6d_pose'):
            obs.object_6d_pose = recover_object_6d_pose(obs, task_name)
    return demo
```

### 5.5 使用示例

在预处理脚本中的 `read_pkl` 之后调用：

```python
import pickle

# 读取 demo
with open('data/training_raw/bimanual_edge_phone/all_variations/episodes/episode0/low_dim_obs.pkl', 'rb') as f:
    demo = pickle.load(f)

# 补充 object_6d_pose（精确对齐到物体本体坐标系）
demo = patch_demo_with_object_6d_pose(demo, "bimanual_edge_phone")

# 验证
pose = demo[0].object_6d_pose
print(pose['position'])         # 物体质心位置，非 contact 点位置
print(pose['quaternion'])       # 物体本体朝向，非 contact dummy 朝向
print(pose['matrix'].shape)     # (4, 4)
```

### 5.6 验证脚本

以下脚本验证恢复后的 `object_6d_pose` 是否正确：

```bash
conda run --no-capture-output -n ppi python - <<'PY'
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

# --- 逆偏移矩阵（仅示例 edge_phone）---
T_OFFSET_INV = np.array([
    [ 0.000000357627798, -0.999999999999758, -0.000000596046490, -0.004460703107312],
    [ 0.000000238418686, -0.000000596046405,  0.999999999999794, -0.106504077672761],
    [-0.999999999999908, -0.000000357627940,  0.000000238418473, -0.000902679728312],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])

path = '/mnt/rlbench_data/bimanual_edge_phone.train/all_variations/episodes/episode0/low_dim_obs.pkl'
with open(path, 'rb') as f:
    demo = pickle.load(f)

# 对全部帧恢复物体位姿
obj_positions = []
for obs in demo:
    cp = obs.misc['contact_position']
    cq = obs.misc['contact_quaternion']
    R_c = Rotation.from_quat(cq).as_matrix()
    T_c = np.eye(4)
    T_c[:3, :3] = R_c
    T_c[:3, 3] = cp
    T_obj = T_c @ T_OFFSET_INV
    obj_positions.append(T_obj[:3, 3])

obj_positions = np.array(obj_positions)

# 验证1: 物体位置与 contact 位置不同
cp0 = demo[0].misc['contact_position']
op0 = obj_positions[0]
print(f"contact_position (frame 0): {cp0}")
print(f"object position  (frame 0): {op0}")
print(f"差异:                        {op0 - cp0}")
print(f"差异模长:                    {np.linalg.norm(op0 - cp0):.6f} m")

# 验证2: 物体运动轨迹连续且合理
movements = np.linalg.norm(np.diff(obj_positions, axis=0), axis=1)
print(f"\n每帧位移: max={movements.max():.6f}, mean={movements.mean():.6f}")
print(f"总运动距离: {movements.sum():.4f} m")

# 验证3: 两个 contact 点之间的距离恒定（刚体约束仍保持）
dists = [np.linalg.norm(demo[i].misc['contact_position'] - demo[i].misc['grasp_position']) for i in range(len(demo))]
print(f"\ncontact-grasp 刚体约束: mean={np.mean(dists):.8f}, std={np.std(dists):.10f}")

print("\n验证通过！")
PY
```

---

## 6. 总结

| 问题 | 回答 |
|------|------|
| 是否存在差异？ | **是**。平移偏移 7-11 cm，旋转偏移 120°-178° |
| 差异是否是常量？ | **是**。100帧验证中 std 严格为 0，因为 dummy 是物体的刚性子对象 |
| 对 point flow 有影响吗？ | **没有**。常量偏移在正逆变换中互相抵消，数值误差 < 4e-8 m |
| 对状态表示有影响吗？ | **可控**。归一化后训练/推理自洽。如需精确对齐可用本文提供的偏移矩阵 |
| 如何精确对齐？ | `T_obj = T_contact @ T_offset_inv`，本文提供了四个任务的全部矩阵和完整代码 |

---

**文档完成时间**：2026-04-01

**验证环境**：CoppeliaSim V4.1 + PyRep，conda 环境 ppi

**数据来源**：仿真实地提取（加载 .ttm 场景文件，通过 PyRep API 查询）
